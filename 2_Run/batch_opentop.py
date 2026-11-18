"""
batch_opentop.py — standalone OpenTOP batch over all persistent-contrail flights.

Runs ONLY the OpenTOP optimal-control solver.
Each algorithm is now batched on its own; this is the OpenTOP one. For every
flight it mirrors ``open_top_run/drive_opentop.py``:

    FlightData → interpolate_meteo(wind) → agg_conditions → prepare_grid(σ)
    → interpolant → top.Cruise (no wind: enable_wind is not called) → OpenTOPOptimizer
    → weighted-sum + epsilon-constraint Pareto sweeps → save.

Output — ONE gzip'd pickle per (flight, sigma), same payload as
``drive_opentop.save_flight`` (context + per-method records + flat summary +
error), so every solve carries its trajectory, objective value and IPOPT
iteration count (``n_iter``). ``df_cost`` is NOT stored per flight (it is the
bulky meteo grid and can be rebuilt); everything needed for post-processing
travels with each record.

On-disk layout::

    <out-dir>/
        progress.csv                         # appended after every flight
        _flight_cache/<flight_id>.parquet    # per-flight slice (perf cache)
        <flight_id>_sigma<σ>.pkl.gz          # one file per (flight, sigma)

Resumable: a flight whose output pkl.gz exists (with no recorded error) is
skipped, so an interrupted run continues where it stopped.

Run modes (--mode)
------------------
    fuel       solve ONLY the fuel-only warm start per flight → <id>_sigma<σ>_fuel.pkl.gz
    contrails  run the weighted-sum + epsilon-constraint Pareto sweeps, REUSING the
               saved fuel-only warm start (no re-solve) → <id>_sigma<σ>.pkl.gz
    full       fuel-only warm start + both sweeps in one go (default)

Split workflow: `--mode fuel` first, then `--mode contrails` — the second reads
each flight's `*_fuel.pkl.gz` and seeds the sweeps with it.

Track modes (--track-mode)
--------------------------
    exact      TrackCruise: the lateral position is the arc length along the flown
               track, so the solution lies ON it; --track-tolerance is ignored
               (src/optimization/opentop/track_cruise.py) (default)
    tolerance  top.Cruise + follow_track: x/y/heading stay free, held within
               --track-tolerance of the flown track

Usage
-----
    # full batch (σ=2 by default)
    python 2_Run/batch_opentop.py

    # two-phase: fuel-only first, then the contrail Pareto reusing it
    python 2_Run/batch_opentop.py --mode fuel
    python 2_Run/batch_opentop.py --mode contrails

    # smoke test: first 3 flights
    python 2_Run/batch_opentop.py --limit 3

    # specific flights / sigmas
    python 2_Run/batch_opentop.py --flights 010108_00015_1 --sigmas 1 2
"""

from __future__ import annotations
import os
# HPC default: keep each solve single-threaded so N parallel workers don't
# oversubscribe the cores — BLAS/OpenMP would otherwise each try to grab every
# core, thrashing when many flights run at once. Set BEFORE numpy/openap import.
# `setdefault` respects anything you export yourself (e.g. via the SLURM script).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from openap.aero import ft
import argparse
import csv
import gzip
import pickle
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

# Windows consoles default to cp1252 and choke on the → / σ / ✓ characters used
# below — force UTF-8 so prints never crash the batch.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 — best effort; older/foreign stream types
        pass

# ── make the project importable regardless of how this file is launched ──────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.route import real_flight_route, route_point_count  # noqa: E402
from src.postprocess.evaluation import reprice  # noqa: E402

import opentop as top
from src.data.traffic import FlightData
from src.data.meteo import GridWeather
from src.optimization.opentop.opentop_optimizer import OpenTOPOptimizer
from src.instrument import four_times, log_progress, phase

# ── configuration ────────────────────────────────────────────────────────────
from src.config import (  # noqa: E402
    ALT_BUFFER_FT_MAX, ALT_BUFFER_FT_MIN, ALT_STEP_FT, DEFAULT_DATA_PATH,
    DEFAULT_GRID_PATH, EPS_GRID, INTERP_SHAPE, M0, MAX_ITER, MULTISTART_ALT_OFFSETS_FT,
    MAX_ROC_FPM, N_STARTS, TRACK_TOLERANCE_M, WARM_START, WIDEBODY_TYPES, WS_GRID,
    level_window,
)
from src.paths import RUN_DIR  # noqa: E402

DEFAULT_OUT_DIR = RUN_DIR / "opentop_batch"
TMP = Path(tempfile.gettempdir())


# ── flight discovery ─────────────────────────────────────────────────────────
from src.data.flights import ensure_flight_slice, load_flight_index  # noqa: E402


# ── output helpers ───────────────────────────────────────────────────────────
def out_path_for(out_dir: Path, flight_id: str, sigma: float, mode: str = "full",
                 multistart: bool = False) -> Path:
    """Result file for a (flight, sigma, mode, start-strategy).

    ``mode="fuel"`` writes a separate ``*_fuel.pkl.gz`` holding just the fuel-only
    warm-start trajectory, so a later ``mode="contrails"`` run can reuse it
    without re-solving. ``contrails`` and ``full`` share the canonical
    ``*_sigma<σ>.pkl.gz`` (the Pareto).

    ``multistart`` adds an ``_ms`` suffix. Without it a multistart run would write
    the singlestart filenames, and is_done() would skip every flight that already
    has a singlestart result — the job would burn its whole allocation doing
    nothing. The suffix is belt-and-braces next to a separate --out-dir: it keeps
    the two comparable even if the directories are ever merged."""
    suffix = "_fuel" if mode == "fuel" else ""
    if multistart:
        suffix += "_ms"
    return out_dir / f"{flight_id}_sigma{sigma:g}{suffix}.pkl.gz"


def load_warm_start(fuel_path: Path):
    """Return the fuel-only warm-start trajectory DataFrame saved by a
    ``mode="fuel"`` run (its single ``fuel_only`` record's ``meta['trajectory']``)."""
    with gzip.open(fuel_path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("error"):
        raise RuntimeError(f"fuel-only run for this flight recorded an error: {payload['error']}")
    recs = payload.get("results", {}).get("fuel_only", [])
    if not recs or recs[0].get("meta", {}).get("trajectory") is None:
        raise RuntimeError(f"no fuel-only trajectory in {fuel_path.name}")
    return recs[0]["meta"]["trajectory"]


def is_done(path: Path, retry_failed: bool = False) -> bool:
    """True if this (flight, sigma) is already finished. When ``retry_failed`` is
    set, a file that recorded an error counts as NOT done (it gets re-attempted)."""
    if not path.exists():
        return False
    if not retry_failed:
        return True
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f).get("error") is None
    except Exception:  # noqa: BLE001 — unreadable → re-run it
        return False


def save_flight(opt, out_path: Path, context: dict, error: str | None = None) -> None:
    """One flight's whole run → a single gzip'd pickle (see module docstring).

    Payload::
        {"context": {...}, "results": {method: [rec, ...]}, "summary": [...], "error": ...}

    Each record is ``OptResult.as_dict()`` — profile arrays (incl. lat/lon) plus
    the trajectory DataFrame, objective value and n_iter carried in ``meta``.
    ``summary`` holds the scalar rows only (fuel/contrail/objective/n_iter/...).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list[dict]] = {}
    for res in opt.results:
        by_method.setdefault(res.meta.get("method", "unknown"), []).append(res.as_dict())
    payload = {
        "context": context,
        "results": by_method,
        "summary": [res.to_row() for res in opt.results],
        "error":   error,
    }
    with gzip.open(out_path, "wb", compresslevel=5) as f:
        pickle.dump(payload, f)


def save_error(out_path: Path, flight_id: str, sigma: float, msg: str,
               actype: str | None = None, m0: float | None = None) -> None:
    """Write an error stub so the flight counts as attempted (skipped on resume
    unless --retry-failed). Mirrors the save_flight payload shape.

    ``actype``/``m0`` are carried even though the solve never ran: without them
    every per-type or per-m0 tally over a result directory has to special-case
    the stubs (or dies on KeyError)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "context": {"flight_id": flight_id, "sigma": sigma,
                    "actype": actype, "m0": m0},
        "results": {}, "summary": [], "error": msg,
    }
    with gzip.open(out_path, "wb", compresslevel=5) as f:
        pickle.dump(payload, f)


# ── non-convergence debug ────────────────────────────────────────────────────
def attach_solve_recorder(cruise) -> list[dict]:
    """Snapshot the IPOPT stats of *every* solve on ``cruise``.

    ``cruise.stats`` exposes only the most recent solve, but a Pareto sweep runs
    one solve per grid point — so every failure but the last would be lost. Wraps
    the instance's ``_solve`` to append one record per call. Returns the list it
    will fill (empty until the sweeps run).
    """
    log: list[dict] = []
    inner = cruise._solve

    def recording_solve(X, U, **kwargs):
        df = inner(X, U, **kwargs)
        try:
            st = cruise.stats
            log.append({
                "return_status": st.get("return_status"),
                "success":       bool(st.get("success")),
                "iter_count":    st.get("iter_count"),
                "t_wall_total":  st.get("t_wall_total"),
            })
        except Exception as exc:  # noqa: BLE001 — the debug path must never kill a run
            log.append({"return_status": f"<unavailable: {exc}>", "success": None})
        return df

    cruise._solve = recording_solve
    return log


def save_debug(opt, out_dir: Path, context: dict, solve_log: list[dict]) -> Path:
    """Dump the evidence for a flight where at least one solve did not converge.

    opentop swallows IPOPT failures — ``base.Base._solve`` catches the RuntimeError
    and returns ``opti.debug``, i.e. the last *iterate* — so a ``success=False``
    result still carries a plausible-looking trajectory that is not an optimum.
    This keeps the per-solve IPOPT status alongside those trajectories.
    """
    dbg_dir = out_dir / "debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    # _ms like the result pickle: without it a multistart run into the same OUT
    # overwrote the singlestart debug file of the same flight and sigma.
    ms = "_ms" if context.get("multistart") else ""
    path = dbg_dir / f"{context['flight_id']}_sigma{context['sigma']:g}{ms}_debug.pkl.gz"

    solves = []
    for res in opt.results:
        m = res.meta
        solves.append({
            "method":          m.get("method"),
            "coef":            m.get("coef"),
            "fuel_slack_frac": m.get("fuel_slack_frac"),
            "success":         res.success,
            "n_iter":          m.get("n_iter"),
            "time_s":          m.get("time_s"),
            "fuel_kg":         res.fuel_kg,
            "contrail":        res.contrail,
            "cost":            res.cost,
            "trajectory":      m.get("trajectory"),
        })

    payload = {
        "context":  context,
        "solves":   solves,
        "ipopt":    solve_log,
        "n_solves": len(opt.results),
        "n_failed": sum(1 for r in opt.results if not r.success),
    }
    with gzip.open(path, "wb", compresslevel=5) as f:
        pickle.dump(payload, f)
    return path


def failed_statuses(solve_log: list[dict]) -> str:
    """Distinct IPOPT return_status values across the non-converged solves, as one
    field for progress.csv — so a whole batch can be triaged with value_counts()."""
    return ";".join(sorted({str(s.get("return_status"))
                            for s in solve_log if not s.get("success")}))


# ── one flight ───────────────────────────────────────────────────────────────
def run_one_flight(meta: dict, sigma: float, flight_slice: Path, out_dir: Path,
                   domain_meteo, mode: str = "full", m0: float = M0,
                   multistart: bool = False, n_starts: int = N_STARTS,
                   alt_buffer_ft: tuple[float, float] = (ALT_BUFFER_FT_MIN,
                                                         ALT_BUFFER_FT_MAX),
                   interp: str = INTERP_SHAPE, global_interp=None,
                   track_tolerance_m: float = TRACK_TOLERANCE_M,
                   track_mode: str = "tolerance") -> dict:
    """Build the interpolant + cruise (no wind) for one flight, then depending on
    ``mode``:

      * ``"fuel"``      — solve only the fuel-only warm start and save it to
                          ``*_fuel.pkl.gz`` (nothing else).
      * ``"contrails"`` — load that saved warm start and run the two Pareto sweeps
                          seeded with it (no fuel-only re-solve), saving the Pareto.
      * ``"full"``      — the fuel-only warm start + both sweeps in one go
                          (default; identical to the previous behaviour).

    Mirrors drive_opentop.run_drive but reads meteo from the precomputed grid
    (``domain_meteo``) — no live ERA5."""
    flight_id = meta["flight_id"]
    actype    = meta["actype"]
    out_path  = out_path_for(out_dir, flight_id, sigma, mode, multistart)
    t0 = time.time()

    # 1) Flight geometry only — no live meteo fetch (base_id/path loads the trace).
    fd = FlightData(start=str(meta["begin"]), stop=str(meta["end"]),
                    path=str(flight_slice), base_id=flight_id, aircraft_type=actype)

    # 2) Attach the precomputed-grid meteo (already σ-smoothed) to the flight, then
    #    build the per-flight cost interpolant + wind field from it. The cost grid's
    #    ts is departure-relative, matching opentop's elapsed-time ts state.
    #    The WIND is needed in every mode, for the wind-corrected cruise Mach only:
    #    the solve itself has no wind (enable_wind is never called).
    #    The contrail interpolant is NOT: mode="fuel" solves objective="fuel", which
    #    never reads it — and passing it anyway flips IPOPT to an exact Hessian
    #    (opentop/base.py: `interpolant is not None` → hessian_approximation="exact"),
    #    silently solving the fuel-only baseline under a different solver config.
    #    fd.interpolant() computes the cost grid itself, so don't call cost_grid too.
    #
    # Timed separately from the solve: this cost is identical for every solver in
    # the benchmark, so folding it into "OpenTOP took N seconds" would compare
    # shared infrastructure rather than optimisers.
    with phase("meteo") as m_meteo:
        fd.attach_meteo("grid", grid=domain_meteo, sigma=sigma, wind=True)
    interpolant = None
    interpolant_mode = None
    interp_cache_hit = None     # None: no per-flight fit (--mode fuel or --global-interp)
    with phase("interpolant") as m_interp:
        if mode != "fuel":
            if global_interp is not None:
                # One fit for the whole domain in ABSOLUTE time, composed here with the
                # constant shift Δ = departure − t0 instead of refitted (164 s median
                # per flight). Imported lazily, exactly as in batch_graph2d: a missing
                # copy of build_global_interpolant must not break a per-flight run.
                #
                # The field is NOT bit-identical to a per-flight fit — it carries a
                # uniform ~2e-4 residual — and the OUTPUT FILENAME DOES NOT SAY SO. Pair
                # every global run with its own --out-dir; interpolant_mode below is the
                # only record inside the pickle.
                from src.contrails.interpolant import shift_interpolant
                _g, _gmeta = global_interp
                interpolant = shift_interpolant(_g, meta["begin"], _gmeta)
                interpolant_mode = f"global_{_gmeta.get('shape', 'unknown')}"
            else:
                # The shape is part of the cache KEY, not just of the build:
                # cached_interpolant_from_dataframe does no invalidation (opentop/tools.py:304),
                # so a shared name would silently serve a bspline built by an earlier run to a
                # --interp linear one, and the campaign would be a bspline campaign wearing
                # the wrong label.
                interp_cache = (TMP /
                                f"opentop_batch_interp_{flight_id}_sigma{sigma:g}_{interp}.casadi")
                # Read BEFORE the call: a hit makes interpolant_s a file load, not a build.
                interp_cache_hit = interp_cache.exists()
                interpolant = fd.interpolant(sigma, cache_path=interp_cache, shape=interp)
                interpolant_mode = "per_flight"

    # Timed apart from the interpolant and the solve: the window and the flown route
    # are problem setup, as ContrailEnvironmentFromGrid is for the graph solvers
    # (env_s there). build_s below is the solver's own construction.
    with phase("env") as m_env:
        # 3) Cruise Mach + altitude window from the real flight. The window is centered
        #    on the real flight's mean cruise altitude ± ALT_BUFFER_FT, and BOTH the
        #    fuel-only warm start and the contrail sweeps are confined to it — so the
        #    optimization stays close to the flown route. Mach is wind-corrected (the
        #    grid slice carries wind → TAS, not ground-referenced).
        alt_buffer_min_ft, alt_buffer_max_ft = alt_buffer_ft
        cruise_alt_ft = float(fd.flight.data.altitude.mean())
        min_alt_ft = cruise_alt_ft - alt_buffer_min_ft
        max_alt_ft = cruise_alt_ft + alt_buffer_max_ft
        mach = fd.compute_cruise_mach()
        route, total_km = real_flight_route(fd.flight, mach=mach, avg_alt_m=cruise_alt_ft*ft,
                                            roc=500, alt_step_ft=500)
        rlat = np.array([p[0] for p in route])
        rlon = np.array([p[1] for p in route])
    # Cruise + NLP configuration + optimizer, the counterpart of the graph solvers'
    # build_s. opentop still assembles the NLP inside every trajectory() call, so
    # part of its setup stays in each solve's time_s; this is only what runs once.
    with phase("build") as m_build:
        # 4) Cruise WITHOUT wind (lateral path held to the flown track by follow_track,
        #    mach fixed, cruise descent allowed). enable_wind is not called, so time is
        #    distance / TAS, as in the graph solvers.
        #    mach_value pins the Mach to the real flight's (wind-corrected) cruise Mach,
        #    so fix_mach_number() constrains it to that value rather than an arbitrary
        #    constant the optimizer would otherwise pick — matching drive_opentop.
        #    track_mode="exact" swaps in TrackCruise: same constructor and the same
        #    follow_track call, but the position is ONE state along the track, so the
        #    tolerance does not exist there (follow_track ignores it). Imported only
        #    here: it is pinned to one opentop version and the tolerance mode must not
        #    depend on it.
        if track_mode == "exact":
            from src.optimization.opentop.track_cruise import TrackCruise
            cruise_cls = TrackCruise
        else:
            cruise_cls = top.Cruise
        cruise = cruise_cls(
            actype,
            (fd.initial_latitude, fd.initial_longitude),
            (fd.final_latitude, fd.final_longitude),
            m0=m0,
            mach_value=mach,
            use_synonym=True,   # resolve types OpenAP lacks directly (e.g. B77L → B77W)
        )
        cruise.fix_mach_number()
        cruise.allow_cruise_descent()
        cruise.follow_track(rlat, rlon, tolerance_m=track_tolerance_m)
        # Discretization: the SAME number of points along the route as the graph
        # solvers, so neither method is compared at a coarser resolution than the
        # other. batch_graph2d's waypoints are real_flight_route(roc=MAX_ROC_FPM,
        # alt_step_ft=ALT_STEP_FT) — one per 2 min of flight, ~26 km — so N waypoints
        # (N-1 segments) here become N-1 collocation intervals, i.e. N boundary nodes.
        # Not len(route): the track reference above is resampled at a 500 ft step,
        # twice as fine. opentop 2.7.0 uses a count passed in as-is (no [20, 120]
        # clamp). Before 2026-10-08 this was one interval per 50 km of
        # origin-destination distance (NODE_KM), as few as 1 on short flights.
        n_dp_points = route_point_count(total_km, mach, cruise_alt_ft * ft,
                                        MAX_ROC_FPM, ALT_STEP_FT)
        cruise.setup(nodes=n_dp_points - 1, max_iter=MAX_ITER)
        solve_log = attach_solve_recorder(cruise)

        # opentop clamps h_max to the aircraft ceiling without telling the caller what
        # it settled on (cruise.py:102-108 — and its message prints the already-clamped
        # value on both sides, so the log reads "41010 exceeds 41010"). Recompute the
        # effective bound here so the saved context states the window that was solved.
        ceiling_ft = float(cruise.aircraft["limits"]["ceiling"]) / ft
        eff_max_alt_ft = min(max_alt_ft, ceiling_ft)
        # The contrail phase ("contrails" / "full") snaps the window outward to whole
        # flight levels, exactly as batch_graph2d does, so OpenTOP, DP and DP_MuO search
        # the same altitude range. --mode fuel is the stage-1 selection and keeps the
        # exact window it was run with. Passing eff_max_alt_ft instead of max_alt_ft
        # changes nothing in fuel mode: opentop clamps to the ceiling anyway.
        alt_snap_ft = None
        if mode != "fuel":
            min_alt_ft, max_alt_ft, eff_max_alt_ft = level_window(
                cruise_alt_ft, ceiling_ft, alt_buffer_ft)
            alt_snap_ft = ALT_STEP_FT

        # 5) Optimizer. Fixed altitude window (real flight ± ALT_BUFFER_FT) for both the
        #    fuel-only warm start and the contrail sweeps. In "contrails" mode, seed it
        #    with the saved fuel-only warm start so it is REUSED (not re-solved).
        initial_guess = None
        if mode == "contrails":
            fuel_path = out_path_for(out_dir, flight_id, sigma, "fuel", multistart)
            if not fuel_path.exists():
                raise RuntimeError(
                    f"mode='contrails' needs the fuel-only run first: {fuel_path.name} "
                    f"not found — run with --mode fuel."
                )
            initial_guess = load_warm_start(fuel_path)

        opt = OpenTOPOptimizer(
            cruise=cruise, interpolant=interpolant,
            h_min_ft=min_alt_ft, h_max_ft=eff_max_alt_ft,  # centered on the real flight altitude
            initial_guess=initial_guess,                # reused warm start in "contrails" mode
            warm_start=WARM_START,                      # False: sweeps don't seed from the fuel-only
            mach=mach, mass_init_kg=cruise.mass_init, actype=actype,
            # Every sweep solve becomes n_starts solves: the fuel-only trajectory, then
            # its shape recentred on the real flight's mean altitude + each of the first
            # n_starts-1 offsets. The sweeps cost ~n_starts x the singlestart run.
            multistart=multistart,
            altitude_offsets_ft=MULTISTART_ALT_OFFSETS_FT[:n_starts - 1],
            centre_alt_ft=cruise_alt_ft,
            # The fuel-only anchor is multistarted too (since 2026-11-03), in every
            # mode: opentop's default guess, then its solution recentred on the real
            # mean altitude + each offset. Before, it was ONE solve in "full", so
            # earlier _ms campaigns have a single-start anchor (context
            # fuel_only_multistart absent).
            multistart_fuel_only=multistart,
        )

    solve_m = None
    # Bound here, not inside the else, so --mode fuel (which has no sweeps to split)
    # reaches the context with an empty split instead of a NameError.
    sub: dict = {}
    if mode == "fuel":
        # Solve ONLY the fuel-only warm start; persist it (with its trajectory) so
        # a later --mode contrails run reuses it. warm_start() also sets the
        # recentered opt.h_min_ft/h_max_ft.
        with phase("solve") as solve_m:
            traj = opt.warm_start()
        if traj is None:
            raise RuntimeError("fuel-only warm start returned no trajectory")
        res = opt._traj_to_result(
            traj, opt.warm_start_s, opt.warm_start_success,
            method="fuel_only", objective=opt.warm_start_objective,
            n_iter=opt.warm_start_n_iter,
        )
        opt.results = [res] if res is not None else []
    else:
        # "contrails" (seeded with the reused warm start) or "full" (solves the
        # fuel-only baseline inline once, inside the first sweep's warm_start()).
        #
        # Solve the fuel-only baseline HERE rather than letting pareto_front()
        # trigger it: it is the F0 the epsilon-constraint budgets are relative to
        # and the min-fuel anchor of the Pareto front, but pareto_front() does not
        # record it in opt.results, so it would be lost. warm_start() is idempotent
        # — the sweeps below reuse this trajectory instead of re-solving it, so the
        # solve count is unchanged. Its success/objective/iterations must be read
        # now, while cruise still holds THIS solve's stats.
        baseline = None
        # Sub-phases of the solve, added 2026-09-08, mirroring batch_graph2d so the two
        # batches can be compared one sweep at a time instead of as one lump. phase()
        # nests, so solver_s keeps exactly the meaning it had and anchor_s + ws_s +
        # eps_s account for it.
        #
        # Nothing post-hoc runs inside these phases: the contrail re-pricing comes
        # after the solve closes, in its own phase (posthoc_s).
        with phase("solve") as solve_m:
            if mode == "full":
                with phase("anchor") as anchor_m:
                    traj = opt.warm_start()
                sub.update(anchor_m.as_dict("anchor"))
                if traj is not None:
                    baseline = opt._traj_to_result(
                        traj, opt.warm_start_s, opt.warm_start_success,
                        method="fuel_only", objective=opt.warm_start_objective,
                        n_iter=opt.warm_start_n_iter,
                    )

            with phase("ws") as ws_m:
                opt.pareto_front(method="weighted_sum", points=WS_GRID)
            sub.update(ws_m.as_dict("ws"))
            with phase("eps") as eps_m:
                opt.pareto_front(method="epsilon_constraint", points=EPS_GRID)
            sub.update(eps_m.as_dict("eps"))

        # First, so summary[0] is the min-fuel anchor of the front.
        if baseline is not None:
            opt.results.insert(0, baseline)

    # Post-hoc, outside solver_s: re-price every trajectory's fuel AND contrail with
    # the evaluator the graph solvers go through (evaluate_path's fuel_burn and
    # contrail_integral), so the families are compared on one measure. The solver's own
    # values stay in meta["fuel_solver"] / meta["contrail_solver"]. --mode fuel has no
    # interpolant and is the stage-1 selection: left on the solver's own fuel.
    posthoc_m = None
    if interpolant is not None:
        with phase("posthoc") as posthoc_m:
            for res in opt.results:
                reprice(res, interpolant, mach, cruise.mass_init, actype)

    # 6) Save (df_cost intentionally NOT stored — bulky and rebuildable).
    context = dict(
        flight_id=flight_id, begin=str(meta["begin"]), end=str(meta["end"]),
        sigma=sigma, m0=m0, actype=actype, mach=mach, mode=mode,
        mass_init_kg=cruise.mass_init, cruise_alt_ft=cruise_alt_ft,
        # Optimization window ACTUALLY SOLVED. opentop silently clamps h_max to
        # the aircraft ceiling (cruise.py:102-108), so for high-flying types the
        # requested +ALT_BUFFER_FT_MAX is not what the solver saw. max_alt_ft is
        # the effective bound; the requested one and the ceiling are kept beside
        # it so the clamp is visible instead of having to be inferred.
        min_alt_ft=min_alt_ft, max_alt_ft=eff_max_alt_ft,
        max_alt_ft_requested=max_alt_ft, ceiling_ft=ceiling_ft,
        alt_clamped_by_ceiling=bool(eff_max_alt_ft < max_alt_ft),
        # The REQUESTED buffers, stated outright. min/max_alt_ft already imply them
        # given cruise_alt_ft, but only if you know to do the subtraction — and not
        # knowing to do it is exactly how the campaigns drifted apart.
        alt_buffer_ft_min=alt_buffer_min_ft, alt_buffer_ft_max=alt_buffer_max_ft,
        # ALT_STEP_FT when the window was snapped outward to whole flight levels
        # (the contrail phase, same window as the graph solvers); None when it is
        # the exact cruise_alt -MIN/+MAX (--mode fuel, and every result before this).
        alt_window_snap_ft=alt_snap_ft,
        origin=(fd.initial_latitude, fd.initial_longitude),
        destination=(fd.final_latitude, fd.final_longitude),
        # False: the wind only corrects the cruise Mach (compute_cruise_mach); the
        # solve itself has no wind, as enable_wind is never called. Pickles written
        # before 2026-10-08 say True for the same solve.
        wind_enabled=False, track_fixed=True,
        # Stated outright for the same reason as the altitude buffers: it changes the
        # problem, and nothing else on disk records it.
        # None under track_mode="exact": the solution lies on the track, there is no
        # tolerance to report.
        track_tolerance_m=(track_tolerance_m if track_mode == "tolerance" else None),
        track_mode=track_mode,
        # Collocation intervals, and how they were chosen: "dp_waypoints" = the graph
        # solvers' segment count. Absent before 2026-10-08 (then NODE_KM = 50 km).
        nodes=int(cruise.nodes), nodes_rule="dp_waypoints",
        # Solver provenance: the interpolant's presence is what selects IPOPT's
        # Hessian mode, so record both — a result directory can otherwise mix two
        # solver configurations with nothing on disk saying which ran.
        # "per_flight", or "global_<shape>" under --global-interp. Set where the
        # interpolant is built, not re-derived here, so the two can never disagree.
        interpolant_mode=interpolant_mode,
        ipopt_hessian="exact" if interpolant is not None else "limited-memory",
        # Which interpolation the cost grid was represented with. Recorded because the
        # filename does not carry it and the two shapes give the solver different
        # derivatives of the SAME field — a directory would otherwise be unreadable.
        # Under --global-interp it is the shape the GLOBAL fit used, read back from its
        # metadata: --interp describes a per-flight fit that did not happen.
        interpolant_shape=((global_interp[1].get("shape", "unknown"))
                           if global_interp is not None
                           else (interp if interpolant is not None else None)),
        # t0 the global fit is referenced to, so the Δ shift is auditable. None for a
        # per-flight fit, which is departure-relative by construction.
        global_interp_t0=(global_interp[1]["t0"] if global_interp is not None else None),
        # What each result's `contrail` and `fuel_kg` are. Since 2026-10-19 the
        # trajectory's integral on the interpolant (contrail_integral), and since
        # 2026-10-19 its fuel by evaluate_path's fuel_burn: the graph solvers'
        # evaluator for both. The solver's own values are in meta["contrail_solver"] /
        # meta["fuel_solver"]. Earlier pickles lack the keys (fuel_measure: before
        # 2026-10-19 fuel_kg IS the solver's mass burnt). None in --mode fuel.
        contrail_measure="interpolant_gl3" if interpolant is not None else None,
        fuel_measure="openap_enroute_segments" if interpolant is not None else None,
        **(posthoc_m.as_dict("posthoc") if posthoc_m is not None else {}),
        # Measured after the solve closes, as for DP and DP_MuO. Absent before
        # 2026-10-19, when OpenTOP had no post-hoc term.
        posthoc_in_solver=False,
        n_points=meta.get("n_points"),
        # Start strategy, so a multistart result is fully described by what is on
        # disk. Since 2026-09-29 the starts are the fuel-only trajectory and fixed
        # altitude offsets of it (multistart_basis, altitude_offsets_ft). Earlier
        # multistart pickles lack those keys and carry seed / lateral_jitter_km /
        # altitude_jitter_ft instead: start 0 was opentop's default guess and the
        # others random perturbations of start 0's own solution. Since 2026-09-30 the
        # offsets are taken from the real flight's MEAN ALTITUDE (the fuel-only shape
        # recentred on it, multistart_centre_ft), not from the fuel-only trajectory, and
        # the fuel-only anchor is a single solve from opentop's default guess: basis
        # "fuel_only_at_real_mean_alt". Each candidate records its initial_guess.
        multistart=multistart,
        n_starts=n_starts if multistart else 1,
        # --mode fuel: start 0 from opentop's default guess, the others its solution
        # recentred on the real mean altitude + each offset.
        multistart_basis=(None if not multistart
                          else "default_solution_at_real_mean_alt" if mode == "fuel"
                          else "fuel_only_at_real_mean_alt"),
        multistart_centre_ft=float(cruise_alt_ft) if multistart else None,
        # Whether the fuel-only anchor itself was multistarted (see the optimizer
        # call). Absent in earlier pickles, whose anchor was always one solve.
        fuel_only_multistart=bool(multistart),
        altitude_offsets_ft=list(opt.altitude_offsets_ft) if multistart else None,
        # Benchmark instrumentation. solver_* is the ONLY part attributable to
        # the optimiser; meteo_* and interpolant_* are shared with every other
        # solver in the comparison and must be reported separately, not summed
        # into it. RSS figures are per process — valid with --workers 1 only.
        # ``work_metric`` names what n_iter counts for this solver, because the
        # graph methods do not iterate at all and the number is not commensurable.
        work_metric="ipopt_iterations",
        **m_meteo.as_dict("meteo"),
        **m_interp.as_dict("interpolant"),
        # True when interpolant_s is a load from the node-local cache in TMP rather
        # than a build — the cache is kept on purpose, so read interpolant_s with it.
        interpolant_cache_hit=interp_cache_hit,
        # Setup, in two parts: the window + flown route, then Cruise + optimizer.
        # Absent from pickles written before 2026-09-28.
        **m_env.as_dict("env"),
        **m_build.as_dict("build"),
        **(solve_m.as_dict("solver") if solve_m is not None else {}),
        # The solve, split: anchor_s + ws_s + eps_s == solver_s in --mode full.
        # Empty in --mode fuel (there is only the anchor, and solver_s already is it).
        # Absent from every pickle written before 2026-09-08 — read with .get(), and do
        # not read a missing field as zero.
        **sub,
    )
    # The four reported times (src.instrument.TIME_FIELDS), the same as batch_graph2d's.
    context.update(four_times(context))
    save_flight(opt, out_path, context)

    n_ok = sum(1 for r in opt.results if r.success)
    n_bad = len(opt.results) - n_ok
    fail_status = ""
    if n_bad:
        # A non-converged solve still returns a trajectory (opentop hands back the
        # last IPOPT iterate), so it would otherwise sail through unnoticed.
        fail_status = failed_statuses(solve_log)
        dbg = save_debug(opt, out_dir, context, solve_log)
        print(f"  [debug] {n_bad}/{len(opt.results)} solves not converged "
              f"[{fail_status}] → {dbg.name}")

    return dict(status="done", elapsed_s=round(time.time() - t0, 1),
                n_solves=len(opt.results), n_success=n_ok,
                fail_status=fail_status)




# ── parallel workers ─────────────────────────────────────────────────────────
# Each worker process holds its OWN GridWeather (built once by the pool
# initializer). We do NOT pickle the big loaded grid to workers — the initializer
# rebuilds it from the path, which works under both fork (Linux/HPC) and spawn
# (Windows). Flights are independent, so parallelism is embarrassingly simple.
_WORKER_GRID: "GridWeather | None" = None
_WORKER_GINTERP: dict = {}      # sigma -> (casadi.Function, meta), when --global-interp


def _worker_init(grid_path: str, sigmas: list[float],
                 global_interp_dir: str | None = None,
                 interp: str = INTERP_SHAPE) -> None:
    global _WORKER_GRID, _WORKER_GINTERP
    _WORKER_GRID = GridWeather(grid_path, sigmas=sigmas)
    _WORKER_GINTERP = {}
    if global_interp_dir is not None:
        # Lazy, like the shift in run_one_flight: --global-interp is optional and a
        # missing build_global_interpolant must not break a per-flight run.
        from src.contrails.interpolant import load_global_interpolant
        # Keyed by SIGMA. One entry per sigma the job runs, never one shared entry:
        # a single interpolant reused across sigmas would solve one sigma's flights
        # against another sigma's field and write them under the right filename, with
        # nothing in the pickle to show it. ~176 MB each, read-only, loaded once per
        # worker process. --interp picks WHICH global fit: global_interp_sigma<s> for
        # bspline, global_linear_interp_sigma<s> for linear (src.contrails.interpolant._paths).
        for sg in sigmas:
            _WORKER_GINTERP[float(sg)] = load_global_interpolant(
                sigma=sg, out_dir=Path(global_interp_dir), shape=interp)


def _run_task(task: tuple) -> dict:
    """Run one (flight, sigma) using this process's GridWeather. Returns a progress
    row; on failure writes an error stub and returns status='error'. Never raises,
    so one bad flight cannot kill the whole pool."""
    (meta, sigma, slice_path, out_dir, mode, m0, multistart, n_starts,
     alt_buffer_ft, interp, track_tolerance_m, track_mode) = task
    out_dir = Path(out_dir)
    flight_id = meta["flight_id"]
    out_path = out_path_for(out_dir, flight_id, sigma, mode, multistart)
    row = dict(utc_time=datetime.now(timezone.utc).isoformat(),
               flight_id=flight_id, sigma=sigma)
    try:
        result = run_one_flight(meta, sigma, Path(slice_path), out_dir, _WORKER_GRID,
                                mode=mode, m0=m0, multistart=multistart,
                                n_starts=n_starts,
                                alt_buffer_ft=alt_buffer_ft, interp=interp,
                                global_interp=_WORKER_GINTERP.get(float(sigma)),
                                track_tolerance_m=track_tolerance_m,
                                track_mode=track_mode)
        row.update(result)
    except Exception as exc:  # noqa: BLE001 — isolate one flight's failure
        err = f"{type(exc).__name__}: {exc}"
        row.update(status="error", error=err)
        save_error(out_path, flight_id, sigma, err,
                   actype=meta.get("actype"), m0=m0)
        traceback.print_exc()
    return row


# ── batch driver ─────────────────────────────────────────────────────────────
def run_batch(data_path: Path, grid_path: Path, out_dir: Path, sigmas: list[float],
              flights: list[str] | None, limit: int | None,
              retry_failed: bool = False, mode: str = "full",
              workers: int = 1, shard: tuple[int, int] | None = None,
              m0: float = M0, actypes: list[str] | None = None,
              fleet: str = "all",
              multistart: bool = False, n_starts: int = N_STARTS,
              alt_buffer_ft: tuple[float, float] = (ALT_BUFFER_FT_MIN,
                                                    ALT_BUFFER_FT_MAX),
              interp: str = INTERP_SHAPE,
              global_interp_dir: Path | None = None,
              track_tolerance_m: float = TRACK_TOLERANCE_M,
              track_mode: str = "tolerance") -> None:
    """Run the batch. ``workers=1`` is sequential (a single PC); ``workers=N``
    runs N flights in parallel processes (a multi-core node). ``shard=(i, N)``
    handles only every N-th flight — for SLURM job arrays across nodes.

    ``m0`` is the initial mass as a fraction of MTOW. ``fleet`` picks the
    population it applies to: ``"narrow"`` or ``"wide"`` split the flights on
    ``WIDEBODY_TYPES`` (narrow bodies run at M0, wide bodies at M0_WIDEBODY), and
    ``actypes`` restricts further to named ICAO types. Output filenames do NOT
    encode m0, so give each m0 its own ``out_dir``.

    ``alt_buffer_ft`` is ``(MIN, MAX)`` in feet around the real flight's mean cruise
    altitude. Filenames do not encode it either, so it needs its own ``out_dir``
    just as much as ``m0`` does.

    ``global_interp_dir`` switches the cost interpolant from a per-flight fit to the
    prebuilt whole-domain one in that directory (one per sigma), composed per flight
    with a constant time shift instead of refitted. Same rule again: the field is not
    identical to a per-flight fit, the filename does not say so, and it needs its own
    ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_flight_cache"

    idx = load_flight_index(data_path)
    if flights:
        idx = idx[idx["flight_id"].isin(flights)].reset_index(drop=True)
        missing = set(flights) - set(idx["flight_id"])
        if missing:
            print(f"  WARNING: flight ids not found and skipped: {sorted(missing)}")
    if fleet != "all":
        wide = idx["typecode"].astype(str).str.upper().isin(WIDEBODY_TYPES)
        idx = idx[wide if fleet == "wide" else ~wide].reset_index(drop=True)
        print(f"  --fleet {fleet}: {len(idx)} flights")
        if idx.empty:
            print("  nothing to do."); return
    if actypes:
        want = {a.upper() for a in actypes}
        idx = idx[idx["typecode"].astype(str).str.upper().isin(want)].reset_index(drop=True)
        print(f"  --actypes {sorted(want)}: {len(idx)} flights match")
        if idx.empty:
            print("  nothing to do."); return
    if limit is not None:
        idx = idx.head(limit).reset_index(drop=True)
    if shard is not None:
        i, n = shard
        idx = idx.iloc[i::n].reset_index(drop=True)   # round-robin partition (load-balanced)
        print(f"  shard {i}/{n}: this task handles {len(idx)} of the selected flights")

    starts = (f"multistart n_starts={n_starts} altitude offsets "
              f"{list(MULTISTART_ALT_OFFSETS_FT[:n_starts - 1])} ft"
              if multistart else "singlestart")
    print(f"Running OpenTOP (mode={mode}, workers={workers}, m0={m0:g}, {starts}) on "
          f"{len(idx)} flights × {len(sigmas)} sigma(s)")
    # Echoed because the filename does not carry it and a wrong window is otherwise
    # only visible by unpickling a result.
    print(f"  window: cruise_alt -{alt_buffer_ft[0]:g} / +{alt_buffer_ft[1]:g} ft"
          + ("" if mode == "fuel" else f", snapped outward to {ALT_STEP_FT:g} ft levels"))
    print(f"  track mode: {track_mode}"
          + (f", tolerance {track_tolerance_m:g} m" if track_mode == "tolerance"
             else " (on the flown track, --track-tolerance ignored)"))
    print(f"  interp: {interp}" + ("  (not used in --mode fuel)" if mode == "fuel" else ""))
    if mode == "fuel":
        print("  fuel-only warm start only → *_fuel.pkl.gz")
    else:
        print(f"  WS grid={WS_GRID}\n  EPS grid={EPS_GRID}")
        if mode == "contrails":
            print("  reusing the saved fuel-only warm start (run --mode fuel first)")
    print(f"  out={out_dir}\n")

    # Build the task list in the PARENT: is_done check + flight-slice creation, so
    # workers only READ slices (no shared-state races; progress.csv single-writer).
    full_df: pd.DataFrame | None = None
    tasks: list[tuple] = []
    skipped = 0
    for _, meta_row in idx.iterrows():
        meta = dict(meta_row)
        typecode = meta.get("typecode")
        meta["actype"] = str(typecode).upper() if pd.notna(typecode) else "B738"
        flight_id = meta["flight_id"]
        for sigma in sigmas:
            out_path = out_path_for(out_dir, flight_id, sigma, mode, multistart)
            if is_done(out_path, retry_failed=retry_failed):
                skipped += 1
                continue
            if full_df is None:
                print("  (loading full parquet once to build flight slices …)")
                full_df = pd.read_parquet(data_path)
            slice_path = ensure_flight_slice(full_df, cache_dir, flight_id)
            tasks.append((meta, sigma, str(slice_path), str(out_dir), mode, m0,
                          multistart, n_starts, tuple(alt_buffer_ft), interp,
                          track_tolerance_m, track_mode))
    del full_df   # free the big frame before spawning workers

    print(f"  {len(tasks)} to run, {skipped} already done\n")

    done = failed = 0

    def _handle(row: dict, i: int) -> None:
        nonlocal done, failed
        tag = f"[{i}/{len(tasks)}] {row['flight_id']} σ={row['sigma']:g}"
        if row.get("status") == "error":
            failed += 1
            print(f"{tag}  ✗ FAILED: {row.get('error')}")
        else:
            done += 1
            print(f"{tag}  ✓ done in {row.get('elapsed_s')}s "
                  f"({row.get('n_success')}/{row.get('n_solves')} solves converged)")
        log_progress(out_dir, row)

    gdir = str(global_interp_dir) if global_interp_dir is not None else None
    if workers <= 1:
        # Sequential (single PC): build the grid once here and run in a loop.
        _worker_init(str(grid_path), sigmas, gdir, interp)
        for i, task in enumerate(tasks, 1):
            _handle(_run_task(task), i)
    else:
        # Parallel: a pool of worker processes, each with its own GridWeather.
        from multiprocessing import Pool
        print(f"  parallel: {workers} worker processes\n")
        with Pool(workers, initializer=_worker_init,
                  initargs=(str(grid_path), sigmas, gdir, interp)) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_task, tasks), 1):
                _handle(row, i)

    print(f"\n{'='*70}\nOpenTOP batch finished: {done} done, {skipped} skipped, "
          f"{failed} failed.\nResults in {out_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--grid-path", default=str(DEFAULT_GRID_PATH),
                   help="Precomputed contrail-cost + wind grid, sliced per flight.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--sigmas", type=float, nargs="+", default=[2.0])
    p.add_argument("--flights", nargs="+", default=None,
                   help="Only run these flight_ids (default: all).")
    p.add_argument("--flights-file", default=None, metavar="PATH",
                   help="Read flight_ids from a file, one per line (blank lines and "
                        "'#' comments ignored). Use this instead of --flights for a "
                        "large set: thousands of ids do not survive the quoting in an "
                        "sbatch's srun bash -c. Merged with --flights if both given.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N flights (smoke test).")
    p.add_argument("--retry-failed", action="store_true",
                   help="Re-attempt flights previously saved with an error.")
    p.add_argument("--mode", choices=["fuel", "contrails", "full"], default="full",
                   help="'fuel': solve only the fuel-only warm start (→ *_fuel.pkl.gz). "
                        "'contrails': run the Pareto sweeps reusing that saved warm "
                        "start (run --mode fuel first). 'full': both in one go (default).")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel worker processes. 1 = sequential (your PC, default); "
                        "N = run N flights at once on a multi-core node. Keep each solve "
                        "single-threaded (OMP_NUM_THREADS=1, set by default) so N workers "
                        "don't oversubscribe cores.")
    p.add_argument("--m0", type=float, default=M0,
                   help=f"Initial mass as a fraction of MTOW (default: {M0}). Widebodies "
                        "do not converge at 0.80; lower values raise the thrust margin. "
                        "Output filenames do NOT encode it — pair every --m0 with its own "
                        "--out-dir or you will overwrite a previous sweep's results.")
    p.add_argument("--alt-buffer", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"),
                   help="Altitude window around the real flight's mean cruise altitude, "
                        f"in feet: -MIN / +MAX (default: {ALT_BUFFER_FT_MIN:g} "
                        f"{ALT_BUFFER_FT_MAX:g}). A wider window is an easier NLP, so "
                        "convergence rates are only comparable between runs that used "
                        "the SAME window. Filenames do not encode it — pair every "
                        "--alt-buffer with its own --out-dir, exactly as for --m0.")
    p.add_argument("--track-tolerance", type=float, default=TRACK_TOLERANCE_M,
                   metavar="M",
                   help="How far, in metres, the optimised ground track may deviate from "
                        f"the real one (default: {TRACK_TOLERANCE_M:g}, opentop's own). It "
                        "changes convergence a lot and filenames do not encode it — "
                        "pair every tolerance with its own --out-dir.")
    p.add_argument("--track-mode", choices=["tolerance", "exact"], default="exact",
                   help="'exact' (default): TrackCruise, the lateral position is reduced "
                        "to the distance along the flown track and --track-tolerance is "
                        "ignored. 'tolerance': top.Cruise.follow_track, the ground track "
                        "may deviate by --track-tolerance. Filenames do not encode "
                        "it — pair each mode with its own --out-dir.")
    p.add_argument("--global-interp", action="store_true",
                   help="Use the prebuilt whole-domain interpolant, shifted per flight: "
                        "global_interp_sigma<s>.casadi, or global_linear_interp_sigma<s>"
                        ".casadi with --interp linear, in --global-interp-dir. Shifted, "
                        "instead of fitting one per flight (164 s median). One file per "
                        "sigma is required. The field is NOT identical to a per-flight "
                        "fit — uniform ~2e-4 residual — so do not mix both in one "
                        "--out-dir; context.interpolant_mode records which ran. Ignored "
                        "by --mode fuel, which builds no cost interpolant at all.")
    p.add_argument("--global-interp-dir", default=None,
                   help="Where the global .casadi/.meta.json live (default: next to "
                        "--grid-path). The shape is in the file name, so the bspline and "
                        "the linear fit of a sigma share one directory; --interp picks.")
    p.add_argument("--fleet", choices=["all", "narrow", "wide"], default="all",
                   help="Run only narrow bodies or only wide bodies, as split by "
                        "src.config.WIDEBODY_TYPES. Narrow bodies run at M0, wide "
                        "bodies at M0_WIDEBODY: pair --fleet with the matching --m0.")
    p.add_argument("--actypes", nargs="+", default=None, metavar="ICAO",
                   help="Only run these aircraft types, e.g. --actypes B77W B77L. "
                        "Combine with --m0 to sweep the mass on one subpopulation.")
    p.add_argument("--multistart", action="store_true",
                   help="Solve each sweep point from N_STARTS initial guesses instead of "
                        "one: the fuel-only trajectory and its MULTISTART_ALT_OFFSETS_FT "
                        "altitude shifts. With --mode fuel, the fuel-only solve itself: "
                        "opentop's default guess, then its solution shifted by the "
                        "offsets. Costs ~n-starts x the wall clock. Results get an "
                        "_ms filename suffix so they cannot overwrite (or be skipped as "
                        "already-done because of) a singlestart run.")
    p.add_argument("--n-starts", type=int, default=N_STARTS, metavar="N",
                   help=f"Starts per solve with --multistart: the fuel-only trajectory and "
                        f"the first N-1 offsets of MULTISTART_ALT_OFFSETS_FT (default "
                        f"{N_STARTS}, at most {N_STARTS}).")
    p.add_argument("--shard", default=None, metavar="i/N",
                   help="Handle only shard i of N flights (round-robin), for SLURM job "
                        "arrays across nodes: --shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT.")
    p.add_argument("--interp", choices=["bspline", "linear"], default=INTERP_SHAPE,
                   help=f"How the contrail cost grid is interpolated (default "
                        f"{INTERP_SHAPE}). 'linear' builds in a fraction of a second "
                        f"instead of minutes but has discontinuous derivatives at every "
                        f"cell face. The filename does not encode it, so give each shape "
                        f"its own --out-dir.")
    args = p.parse_args()
    if not 1 <= args.n_starts <= N_STARTS:
        p.error(f"--n-starts must be 1..{N_STARTS}: the fuel-only trajectory plus at most "
                f"the {N_STARTS - 1} offsets in MULTISTART_ALT_OFFSETS_FT")

    flights = list(args.flights) if args.flights else []
    if args.flights_file:
        path = Path(args.flights_file)
        if not path.exists():
            p.error(f"--flights-file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                flights.append(line)
        print(f"  --flights-file {path.name}: {len(flights)} flight ids")
    flights = sorted(set(flights)) or None

    shard = None
    if args.shard:
        try:
            i_s, n_s = args.shard.split("/")
            shard = (int(i_s), int(n_s))
        except ValueError:
            p.error("--shard must look like i/N (e.g. 3/200)")
        if not (0 <= shard[0] < shard[1]):
            p.error("--shard i/N needs 0 <= i < N")

    run_batch(
        data_path=Path(args.data_path),
        grid_path=Path(args.grid_path),
        out_dir=Path(args.out_dir),
        sigmas=args.sigmas,
        flights=flights,
        limit=args.limit,
        retry_failed=args.retry_failed,
        mode=args.mode,
        workers=args.workers,
        shard=shard,
        m0=args.m0,
        actypes=args.actypes,
        fleet=args.fleet,
        alt_buffer_ft=(tuple(args.alt_buffer) if args.alt_buffer
                       else (ALT_BUFFER_FT_MIN, ALT_BUFFER_FT_MAX)),
        multistart=args.multistart,
        n_starts=args.n_starts,
        interp=args.interp,
        track_tolerance_m=args.track_tolerance,
        track_mode=args.track_mode,
        global_interp_dir=((Path(args.global_interp_dir) if args.global_interp_dir
                            else Path(args.grid_path).parent)
                           if args.global_interp else None),
    )


if __name__ == "__main__":
    main()
