"""
batch_graph2d.py — DP and DP_MuO (2D) over the same flights,
the same grid and the same assumptions as the OpenTOP batch.

The point of this batch is COMPARABILITY, so everything the two solvers share is
imported from ``batch_opentop`` rather than restated here: the flight index, the
per-flight slice cache, the resume/shard/progress machinery, and — critically —
``M0``, the altitude buffers and the Pareto grids. Change one of those constants
and both batches move together; restate them and the benchmark silently drifts.

Per flight it reproduces the OpenTOP setup and then swaps the solver:

    FlightData -> attach_meteo(grid, wind) -> cost_grid(sigma) -> interpolant
    -> real_flight_route (the flown path, same as OpenTOP's follow_track)
    -> ContrailEnvironmentFromGrid (1000 ft levels inside the same window)
    -> DpOptimizer / DpMuoOptimizer

Two solvers, two different shapes of answer:

    dp        scalarises and re-solves: a fuel-only anchor plus a weighted-sum sweep
              over WS_GRID. No epsilon-constraint since 2026-10-19 (it bisected
              coef, ~12 DP solves per point, to reach weighted-sum points);
              older pickles still carry those records.
    dp_muo    one label-setting sweep returns the WHOLE non-dominated set, so the
              front is not sampled on a grid at all — expect a different number
              of points, and do not compare "points per flight" between them.

Output — one gzip'd pickle per (flight, sigma, method), same payload schema as
the OpenTOP batch (``{context, results, summary, error}``) so the same counting
and post-processing scripts work unchanged::

    <out-dir>/<flight_id>_sigma<sigma>_dp.pkl.gz
    <out-dir>/<flight_id>_sigma<sigma>_dp_muo.pkl.gz

Usage
-----
    # the same flights the OpenTOP run used, same masses
    python -m batch_graph2d --methods both \\
        --flights-file 3_Results/flight_selection/selection/flights_m008.txt \\
        --m0 0.80 --out-dir 3_Results/contrail/DP/m_080

    # smoke test
    python -m batch_graph2d --limit 2 --methods dp
"""

from __future__ import annotations

# Thread pinning MUST run before numpy/openap load. It used to arrive as a side
# effect of importing batch_opentop; that import is gone, so it is explicit here.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
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
from openap import prop
from openap.aero import ft

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ALT_BUFFER_FT_MAX, ALT_BUFFER_FT_MIN, ALT_STEP_FT, DEFAULT_DATA_PATH,
    DEFAULT_GRID_PATH, M0, MAX_ROC_FPM, WS_GRID, level_window,
)
from src.flights import ensure_flight_slice, load_flight_index  # noqa: E402
from src.instrument import log_progress  # noqa: E402
from src.paths import RUN_DIR  # noqa: E402
from src.data import FlightData, GridWeather  # noqa: E402
from src.models.grid.contrail_grid import (  # noqa: E402
    ContrailEnvironmentFromGrid,
    real_flight_route,
)
from src.optimization.graph_based.model_2D import (  # noqa: E402
    DpOptimizer,
    DpMuoOptimizer,
)
from src.postprocess.evaluation import evaluate_path  # noqa: E402
from src.instrument import four_times, phase  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 — best effort
        pass

DEFAULT_OUT_DIR = RUN_DIR / "graph2d_run"
TMP = Path(tempfile.gettempdir())


# Interpolation of the contrail cost grid. Unlike OpenTOP, neither graph solver ever
# differentiates it: it is evaluated once to fill env.cost_map and after that both read
# that table by index. So "linear" costs nothing in smoothness here — only the sampled
# values shift. Not encoded in the filename: give each shape its own --out-dir.
INTERP_SHAPE = "bspline"

# Which DP sweeps run, ordered so the front comes out min-fuel first. Campaigns before
# 2026-10-19 also ran "epsilon_constraint", since removed from the DP.
DP_SWEEPS = ("fuel_only", "weighted_sum")


# ── output helpers ───────────────────────────────────────────────────────────
def out_path_for(out_dir: Path, flight_id: str, sigma: float, method: str) -> Path:
    return out_dir / f"{flight_id}_sigma{sigma:g}_{method}.pkl.gz"


def save_results(results: list, out_path: Path, context: dict,
                 error: str | None = None) -> None:
    """Persist one (flight, method) in the OpenTOP batch's payload schema.

    Takes a plain list of ``OptResult`` rather than an optimizer object, because
    the DP_MuO front is assembled by this module, not by a solver method.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list[dict]] = {}
    for res in results:
        by_method.setdefault(res.meta.get("method", "unknown"), []).append(res.as_dict())
    payload = {
        "context": context,
        "results": by_method,
        "summary": [res.to_row() for res in results],
        "error": error,
    }
    with gzip.open(out_path, "wb", compresslevel=5) as f:
        pickle.dump(payload, f)


def save_error(out_path: Path, flight_id: str, sigma: float, msg: str,
               actype: str | None = None, m0: float | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "context": {"flight_id": flight_id, "sigma": sigma,
                    "actype": actype, "m0": m0},
        "results": {}, "summary": [], "error": msg,
    }
    with gzip.open(out_path, "wb", compresslevel=5) as f:
        pickle.dump(payload, f)


def is_done(out_path: Path, retry_failed: bool) -> bool:
    if not out_path.exists():
        return False
    if not retry_failed:
        return True
    try:
        with gzip.open(out_path, "rb") as f:
            return pickle.load(f).get("error") is None
    except Exception:  # noqa: BLE001 — unreadable → redo it
        return False


# ── the DP_MuO front → OptResult ───────────────────────────────────────────
def dp_muo_front_to_results(opt: DpMuoOptimizer, front: dict,
                              env, mach: float, mass_init_kg: float,
                              actype: str, elapsed_s: float) -> list:
    """Re-evaluate every non-dominated path with the SAME post-hoc physics the DP
    points go through (``evaluate_path``).

    The label already carries a (fuel, contrail) pair accumulated from the edge
    arrays, but comparing that against DP/OpenTOP numbers would compare an
    internal graph cost with an evaluated trajectory. Both are kept: the
    evaluated values land in the OptResult, the label's own pair in ``meta``.
    """
    out = []
    n = len(front["paths"])
    for k, (path, label) in enumerate(zip(front["paths"], front["labels"])):
        res = opt._to_result(path)
        _tp = time.perf_counter()
        metrics = evaluate_path(res.alt_ft, res.dist_km, res.ts_s,
                                mach, mass_init_kg, env, actype)
        # Same accumulator the DP fills inside _solve_dp_point, so posthoc_s means the
        # same thing for both methods: evaluate_path and nothing else.
        opt.posthoc_s += time.perf_counter() - _tp
        opt.posthoc_calls += 1
        res.fuel_kg = metrics["total_fuel_kg"]
        res.contrail = metrics["total_contrail"]
        res.ts_s = metrics["ts_s"]  # flown at its own altitudes (flown_ts)
        res.success = True          # a returned label IS a feasible path
        res.meta.update({
            "optimizer": "DP_MuO",
            "method": "pareto_dp_muo",
            "point": k,
            "n_front": n,
            # One solve produces the whole front, so per-point time is undefined;
            # the sweep's total is repeated here and must not be summed over points.
            "time_s": elapsed_s,
            "front_total_s": elapsed_s,
            "label_fuel": label.fuel,
            "label_contrail": label.contrail,
            "contrail_table": metrics["total_contrail_table"],
            "total_alt_changes_ft": metrics["total_alt_changes_ft"],
        })
        out.append(res)
    return out


# ── one flight ───────────────────────────────────────────────────────────────
_WORKER_GRID: "GridWeather | None" = None
_WORKER_GINTERP: dict = {}      # sigma -> (casadi.Function, meta), when --global-interp


def _worker_init(grid_path: str, sigmas: list[float],
                 global_interp_dir: str | None = None) -> None:
    global _WORKER_GRID, _WORKER_GINTERP
    _WORKER_GRID = GridWeather(grid_path, sigmas=sigmas)
    _WORKER_GINTERP = {}
    if global_interp_dir is not None:
        # Imported here, not at module scope: build_global_interpolant is only needed
        # when the flag is on, and a missing copy of it must not break every run.
        from src.interpolant import load_global_interpolant
        # One fit for the whole domain, in ABSOLUTE time; per flight it is composed
        # with a constant shift instead of being refitted (164 s median per flight).
        # Loaded once per worker: each file is ~176 MB and the object is read-only.
        #
        # Keyed by sigma. It used to load load_global_interpolant(sigma=sigmas[0])
        # alone, which is correct for a one-sigma job and SILENTLY WRONG for any
        # other: --sigmas 0 1 --global-interp solved the sigma=1 flights against the
        # sigma=0 field, wrote them into the sigma=1 filenames, and recorded nothing
        # that would show it. Every sigma the job runs now gets its own.
        for s in sigmas:
            _WORKER_GINTERP[float(s)] = load_global_interpolant(
                sigma=s, out_dir=Path(global_interp_dir))


def build_environment(meta: dict, sigma: float, flight_slice: Path, domain_meteo,
                      m0: float, interp: str = INTERP_SHAPE, global_interp=None) -> dict:
    """Everything up to and including the sampled cost map, for one (flight, sigma).

    Extracted from :func:`run_one_flight` so that anything else needing this flight's
    cost field — ``--plot-cost-field`` — gets the SAME field the solvers saw, from one
    definition. The phase metrics are returned rather than recorded here because the
    caller owns the context dict; the order and the boundaries of the two phases are
    unchanged, so the timings stay comparable with everything already measured.
    """
    flight_id = meta["flight_id"]
    actype = meta["actype"]

    fd = FlightData(start=str(meta["begin"]), stop=str(meta["end"]),
                    path=str(flight_slice), base_id=flight_id, aircraft_type=actype)
    # Timed separately: identical cost for every solver in the benchmark.
    with phase("meteo") as m_meteo:
        fd.attach_meteo("grid", grid=domain_meteo, sigma=sigma, wind=True)

    # The 2D environment needs BOTH the cost grid (it samples the cost map along
    # the route) and the interpolant, unlike OpenTOP which only needs the latter.
    with phase("interpolant") as m_interp:
        df_cost = fd.cost_grid(sigma)
        # The shape is part of the cache KEY: cached_interpolant_from_dataframe never
        # invalidates (opentop/tools.py:304), so a shared name would hand a bspline built
        # by an earlier run to a linear campaign without a word.
        interp_cache = TMP / f"graph2d_interp_{flight_id}_sigma{sigma:g}_{interp}.casadi"
        # Read BEFORE the build: a hit makes interpolant_s a file load, not a build.
        # The cache is node-local (TMP), so whether DP_MuO finds DP's file depends on
        # the node its task lands on. None under --global-interp (no per-flight fit).
        interp_cache_hit = None if global_interp is not None else interp_cache.exists()
        # --global-interp: skip the per-flight fit entirely. The cost field is NOT
        # bit-identical to a per-flight fit — it carries a uniform ~2e-4 residual —
        # so a directory mixing the two is not comparable. interpolant_shape below
        # records which was used, because the filename does not.
        if global_interp is not None:
            from src.interpolant import shift_interpolant
            _g, _gmeta = global_interp
            interpolant = shift_interpolant(_g, meta["begin"], _gmeta)
        # No sigma here on purpose: fd.interpolant only rebuilds the cost grid when it is
        # given one (data/data.py:500), and cost_grid() above already cached it in
        # fd._df_cost. Passing sigma would assemble the same 131k-row frame twice and put
        # the second assembly inside interpolant_s, which is then not comparable to the
        # OpenTOP batch's.
        else:
            interpolant = fd.interpolant(cache_path=interp_cache, shape=interp)

    # Timed: the window, the flown route and the cost map. The cost map is filled by
    # calling the interpolant at every (route point, flight level), so this is real
    # work, and it used to fall between the interpolant and build phases untimed.
    with phase("env") as m_env:
        # Same flight condition as the OpenTOP run: real mean cruise altitude, real
        # wind-corrected Mach, mass = m0 x MTOW, and the flown route rather than a
        # great circle (OpenTOP pins the same path with follow_track).
        cruise_alt_ft = float(fd.flight.data.altitude.mean())
        mach = fd.compute_cruise_mach()
        ceiling_ft = float(prop.aircraft(actype, use_synonym=True)["limits"]["ceiling"]) / ft
        mtow_kg = float(prop.aircraft(actype, use_synonym=True)["mtow"])
        mass_init_kg = m0 * mtow_kg

        # Snap the window to whole flight levels: the graph's altitudes are discrete
        # (np.arange(min, max+step, step)), so a fractional bound would shift every
        # level off the 1000 ft structure the cost grid was built on. OpenTOP's contrail
        # phase calls the same function, so the three methods share one window.
        min_alt_ft, _, max_alt_ft = level_window(cruise_alt_ft, ceiling_ft)

        route, total_km = real_flight_route(fd.flight, mach=mach,
                                            avg_alt_m=cruise_alt_ft * ft,
                                            roc=MAX_ROC_FPM, alt_step_ft=ALT_STEP_FT)
        env = ContrailEnvironmentFromGrid(
            interpolant=interpolant, grid_data=df_cost, route=(route, total_km),
            alt_step_ft=ALT_STEP_FT, min_alt_ft=min_alt_ft, max_alt_ft=max_alt_ft,
        )
    return dict(fd=fd, env=env, df_cost=df_cost, interpolant=interpolant,
                cruise_alt_ft=cruise_alt_ft, mach=mach, ceiling_ft=ceiling_ft,
                mtow_kg=mtow_kg, mass_init_kg=mass_init_kg,
                min_alt_ft=min_alt_ft, max_alt_ft=max_alt_ft, total_km=total_km,
                m_meteo=m_meteo, m_interp=m_interp, m_env=m_env,
                interp_cache_hit=interp_cache_hit)


def run_one_flight(meta: dict, sigma: float, flight_slice: Path, out_dir: Path,
                   domain_meteo, method: str, m0: float,
                   interp: str = INTERP_SHAPE, global_interp=None,
                   sweeps: tuple = DP_SWEEPS) -> dict:
    """Solve one (flight, sigma) with one 2D solver and save it.

    ``sweeps`` selects which DP sweeps run (the fuel-only anchor, the weighted sum).
    Ignored by DP_MuO, which has no sweeps."""
    flight_id = meta["flight_id"]
    actype = meta["actype"]
    out_path = out_path_for(out_dir, flight_id, sigma, method)
    t0 = time.time()

    built = build_environment(meta, sigma, flight_slice, domain_meteo, m0,
                              interp=interp, global_interp=global_interp)
    fd, env = built["fd"], built["env"]
    m_meteo, m_interp, m_env = built["m_meteo"], built["m_interp"], built["m_env"]
    cruise_alt_ft, mach = built["cruise_alt_ft"], built["mach"]
    ceiling_ft, mtow_kg = built["ceiling_ft"], built["mtow_kg"]
    mass_init_kg = built["mass_init_kg"]
    min_alt_ft, max_alt_ft = built["min_alt_ft"], built["max_alt_ft"]
    total_km = built["total_km"]

    common = dict(environment=env, cruise_alt_ft=cruise_alt_ft, mach=mach,
                  mass_init_kg=mass_init_kg, actype=actype,
                  max_climb_rate_fpm=MAX_ROC_FPM, max_descent_rate_fpm=MAX_ROC_FPM)

    # The graph solvers' construction is itself real work (mass profile + edge
    # cost tables), unlike OpenTOP where the equivalent is inside each solve, so
    # it is measured as its own phase rather than hidden in the solve.
    # Sub-phases of the solve, added 2026-09-08. Until then phase("solve") wrapped
    # the anchor and both sweeps in one number, and restricting a figure to one sweep
    # meant apportioning solver_s by work count — the per-record meta["time_s"] cannot
    # do it, because for the DP it is the recursion ALONE: it excludes the
    # evaluate_path the DP runs on every point. These phases measure the split
    # instead of estimating it.
    #
    # phase() nests: each opens its own perf_counter and its own RSS sampler thread,
    # so anchor_s + ws_s account for solver_s (both net of post-hoc). Read the inner peak_rss_mb with care — every sampler watches the same
    # process, so the four peaks are near-identical and only the *_s fields separate.
    sub: dict = {}

    def _sweep(name: str, fn):
        """Run one sweep in its own phase and record how much of it was post-hoc.

        ``opt.posthoc_s`` is an accumulator on the optimizer (model_2D bumps it inside
        _solve_dp_point), not a block, so the post-hoc share is taken as a difference
        across the sweep rather than as a phase of its own. Recorded per sweep because
        that is the question the whole-flight posthoc_s cannot answer: it is ~95 % of
        the DP's solve phase, and where it lands decides which sweep looks expensive.
        """
        p0, c0 = float(getattr(opt, "posthoc_s", 0.0)), int(getattr(opt, "posthoc_calls", 0))
        n0 = len(getattr(opt, "dp_solve_log", []))
        with phase(name) as m:
            out = fn()
        for entry in getattr(opt, "dp_solve_log", [])[n0:]:
            entry["sweep"] = name
        posthoc = float(getattr(opt, "posthoc_s", 0.0)) - p0
        # <name>_s is the solve WITHOUT post-hoc (since 2026-09-29): the evaluate_path
        # time, measured around each call, is taken out of the phase and reported
        # beside it. <name>_wall_s keeps the phase as measured, post-hoc included.
        m.extra["wall_s"] = round(m.seconds, 4)
        m.seconds -= posthoc
        m.extra["posthoc_s"] = round(posthoc, 4)
        m.extra["posthoc_calls"] = int(getattr(opt, "posthoc_calls", 0)) - c0
        sub.update(m.as_dict(name))
        return out

    if method == "dp":
        with phase("build") as m_build:
            opt = DpOptimizer(**common)
        with phase("solve") as m_solve:
            # Fuel-only anchor first (coef=0), so the front has its minimum-fuel
            # end explicitly — the role OpenTOP's fuel_only baseline plays.
            results = []
            if "fuel_only" in sweeps:
                anchor = _sweep("anchor", lambda: opt._solve_dp_point(0.0))
                if anchor is not None:
                    anchor.meta["method"] = "fuel_only"
                    results.append(anchor)
            if "weighted_sum" in sweeps:
                results += _sweep("ws", lambda: opt.pareto_dp(grid=WS_GRID,
                                                              method="weighted_sum"))
            if not results:
                raise RuntimeError(f"no DP sweep produced a result (sweeps={sweeps})")
        # solver_s without post-hoc, as for the sweeps above. build does not call
        # evaluate_path, so opt.posthoc_s is exactly what ran inside this phase.
        m_solve.extra["wall_s"] = round(m_solve.seconds, 4)
        m_solve.seconds -= float(opt.posthoc_s)
        # The DP does not iterate — it is exact in one backward pass. Its work
        # metric is how many DP solves the sweeps cost: one per point. (Pickles before
        # 2026-10-19 also count the epsilon-constraint's bisection steps, in
        # n_bisect_solves.)
        extra = dict(work_metric="dp_solves", n_work=len(results),
                     n_front=None,
                     # The filename does not carry it, and a directory holding a
                     # weighted-sum-only run beside a full one is unreadable without it.
                     dp_sweeps=",".join(sweeps),
                     # Every DP solve of the flight in order, one dict each: sweep,
                     # coef, time_s (recursion), posthoc_s (evaluate_path), feasible.
                     # Includes infeasible points, which no record carries.
                     dp_solves=list(opt.dp_solve_log))
    elif method == "dp_muo":
        with phase("build") as m_build:
            opt = DpMuoOptimizer(**common)
        with phase("solve") as m_solve:
            front = opt.build_pareto_front()
        if front is None:
            raise RuntimeError("DP_MuO returned no front")
        # Its own phase, unlike the DP's: DP_MuO evaluates the whole front in
        # one contiguous block AFTER the solve closes, so this is a real phase with a
        # real RSS peak — the front is thousands of trajectories and this is where
        # they are materialised. The DP has no equivalent block; its post-hoc is
        # scattered through the sweeps and is reported as the *_posthoc_s above.
        with phase("posthoc") as m_posthoc:
            results = dp_muo_front_to_results(opt, front, env, mach, mass_init_kg,
                                                actype, m_solve.seconds)
        sub.update(m_posthoc.as_dict("posthoc_phase"))
        # Label-setting does not iterate either. Its work is labels created and
        # kept after dominance pruning, recorded by the solver when available.
        extra = dict(work_metric="labels_expanded",
                     n_work=getattr(opt, "n_labels_expanded", None),
                     n_labels_kept=getattr(opt, "n_labels_kept", None),
                     n_front=len(results))
    else:
        raise ValueError(f"unknown method: {method}")

    context = dict(
        flight_id=flight_id, begin=str(meta["begin"]), end=str(meta["end"]),
        sigma=sigma, m0=m0, actype=actype, mach=mach, method=method,
        solver="graph_2d",
        mass_init_kg=mass_init_kg, mtow_kg=mtow_kg, cruise_alt_ft=cruise_alt_ft,
        min_alt_ft=float(min_alt_ft), max_alt_ft=float(max_alt_ft),
        # The requested buffers, as batch_opentop records them: min/max_alt_ft are
        # those snapped outward to whole levels (src.config.level_window).
        alt_buffer_ft_min=ALT_BUFFER_FT_MIN, alt_buffer_ft_max=ALT_BUFFER_FT_MAX,
        alt_step_ft=ALT_STEP_FT, ceiling_ft=ceiling_ft,
        max_roc_fpm=MAX_ROC_FPM,
        n_alt_levels=len(env.altitudes), n_waypoints=len(env.distances),
        origin=(fd.initial_latitude, fd.initial_longitude),
        destination=(fd.final_latitude, fd.final_longitude),
        route_km=float(total_km),
        setup_s=float(opt.setup_s),
        # Post-hoc evaluation, measured identically for both methods and, since
        # 2026-09-29, OUTSIDE solver_s for both. The DP still calls evaluate_path
        # inside its solve phase (_solve_dp_point evaluates every point), so its time is subtracted from solver_s and the sweep phases;
        # solver_wall_s keeps the phase as measured. DP_MuO evaluates its front after
        # the solve closes. posthoc_in_solver is True only in older DP pickles.
        posthoc_s=float(getattr(opt, "posthoc_s", 0.0)),
        posthoc_calls=int(getattr(opt, "posthoc_calls", 0)),
        posthoc_in_solver=False,
        # What each result's `contrail` is. Since 2026-10-19 the path's integral on the
        # interpolant (evaluate_path -> contrail_integral, 3-point Gauss-Legendre per
        # segment), the same evaluator OpenTOP results are re-priced with; the cost
        # map's own value is in meta["contrail_table"]. Earlier pickles lack the key:
        # their `contrail` IS the table value.
        contrail_measure="interpolant_gl3",
        # evaluate_path's fuel_burn, the measure OpenTOP results are re-priced with
        # since 2026-10-19.
        fuel_measure="openap_enroute_segments",
        n_points=meta.get("n_points"),
        # Which interpolation filled env.cost_map. The filename does not carry it.
        # The shape that was actually FITTED, read back from the interpolant's own
        # metadata. It used to be the literal "global_bspline", so a global LINEAR
        # run recorded itself as bspline — the filename does not carry the shape
        # either, which would have left no way at all to tell the two apart.
        interpolant_shape=(f"global_{global_interp[1].get('shape', 'unknown')}"
                           if global_interp is not None else interp),
        global_interp_t0=(global_interp[1]["t0"] if global_interp is not None else None),
        # Benchmark instrumentation, same fields and same meaning as the OpenTOP
        # batch. Only solver_* and build_* belong to the optimiser; meteo_* and
        # interpolant_* are shared with every other solver. RSS is per process —
        # valid with --workers 1 only.
        **extra,
        **m_meteo.as_dict("meteo"),
        **m_interp.as_dict("interpolant"),
        # True when interpolant_s is a load from the node-local cache, not a build.
        interpolant_cache_hit=built["interp_cache_hit"],
        # Window + flown route + cost-map sampling. Absent before 2026-09-28.
        **m_env.as_dict("env"),
        **m_build.as_dict("build"),
        **m_solve.as_dict("solver"),
        # The solve, split. DP: anchor_s + ws_s == solver_s (+ eps_s before 2026-10-19), all without
        # post-hoc, each with its own <name>_posthoc_s and <name>_wall_s. DP_MuO: no split
        # (one label-setting solve is the whole thing) but posthoc_phase_s, which sits
        # ON TOP of solver_s. Absent from every pickle written before 2026-09-08 — use
        # .get() when reading, and do not compare a run that has them with one that
        # does not by assuming zero.
        **sub,
    )
    # The four reported times (src.instrument.TIME_FIELDS): data_setup_s, solver_setup_s,
    # solve_s, posthoc_s. Sums of the phases above, which stay as the breakdown.
    context.update(four_times(context))
    save_results(results, out_path, context)

    n_ok = sum(1 for r in results if r.success)
    return dict(status="done", elapsed_s=round(time.time() - t0, 1),
                n_solves=len(results), n_success=n_ok, fail_status="")


# ── cost-field inspection (--plot-cost-field) ────────────────────────────────
# Light -> dark in one hue. A sequential field gets one hue by rule: a rainbow (or
# viridis, which walks blue -> green -> yellow) invents boundaries the data does not
# have, and the reader then reads a hue change as a threshold. Near-white at zero so
# "no persistent contrail here" reads as empty rather than as a colour.
COST_RAMP = ["#fdf4ee", "#fadcc4", "#f4bb8d", "#ea8f52", "#d8642a", "#ab400b",
             "#6f2402"]


def plot_cost_field(meta: dict, sigmas: list[float], flight_slice: Path, domain_meteo,
                    m0: float, interps: list[str], fig_dir: Path,
                    scale: str = "per-sigma", cells: bool = False,
                    global_interp=None) -> Path:
    """One figure of ``env.cost_map`` per (interpolation shape x sigma) for one flight.

    This is the field the graph solvers actually optimise over — not the raw grid.
    ``ContrailEnvironmentFromGrid._sample_along_route`` (contrail_grid.py:243) fills
    ``cost_map`` by CALLING THE INTERPOLANT at every (route point, flight level), so
    the shape is not a detail of how OpenTOP differentiates it: linear and bspline
    hand the DP and DP_MuO different numbers on the same grid, and the two
    campaigns' results are not comparable point for point. That is what this figure
    is for.

    Rows are the shapes, columns the sigmas. ``scale`` decides what the colour means:

      per-sigma  one scale per COLUMN, its own colorbar. The comparison the figure
                 exists for — linear against bspline on the same field — is a read of
                 one encoding, and every column stays legible. Columns are then NOT
                 comparable to each other; the per-panel maximum is printed for that.
      shared     one scale for all six. Every comparison is one encoding, but sigma = 0
                 is 3.5x the range of sigma = 2 on this flight, so the smoothed columns
                 wash out to near-white and their structure is unreadable.

    The dashed line is the flight's real mean cruise altitude, the centre of the
    +/-4000 ft window the panel spans.

    The sampled fields are written next to the figure as .npz and reused on the next
    call: rebuilding them costs a bspline fit per sigma (164 s median) and a ~3 GB grid
    load, which is not a price to pay twice for a colour-scale change.
    """
    import matplotlib
    matplotlib.use("Agg")               # no display on a compute node
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    flight_id = meta["flight_id"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache = fig_dir / f"cost_field_{flight_id}.npz"
    want = [(i, s) for i in interps for s in sigmas]

    fields: dict[tuple, dict] = {}
    stored = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    for interp, sigma in want:
        tag = f"{interp}_sigma{sigma:g}"
        if f"{tag}__cost" in stored:
            print(f"  cost field: {flight_id} sigma={sigma:g} {interp}  (from {cache.name})")
            fields[(interp, sigma)] = {
                k.split("__", 1)[1]: (v.item() if v.ndim == 0 else v)
                for k, v in stored.items() if k.startswith(f"{tag}__")}
            continue
        print(f"  cost field: {flight_id} sigma={sigma:g} {interp}")
        # global_interp is the whole {sigma: (fn, meta)} map, so the right one is
        # picked per panel — the sigmas are the columns of this very figure.
        gi = (global_interp or {}).get(float(sigma))
        built = build_environment(meta, sigma, flight_slice, domain_meteo, m0,
                                  interp=interp, global_interp=gi)
        env = built["env"]
        fields[(interp, sigma)] = dict(
            cost=env.cost_map, dist=env.distances, alt=env.altitudes,
            cruise_alt_ft=built["cruise_alt_ft"], km=built["total_km"],
            actype=meta["actype"])

    flat = {f"{i}_sigma{s:g}__{k}": v
            for (i, s), f in fields.items() for k, v in f.items()}
    np.savez_compressed(cache, **{**stored, **flat})
    print(f"  fields cached in {cache.name}")

    cmap = LinearSegmentedColormap.from_list("contrail_cost", COST_RAMP)
    # One vmax per column, or one for the whole figure. Never per PANEL: that would
    # repaint linear and bspline on different scales and make the one comparison the
    # figure is for impossible to read off the colour.
    if scale == "shared":
        vmax = {s: max(float(f["cost"].max()) for f in fields.values()) for s in sigmas}
    else:
        vmax = {s: max(float(fields[(i, s)]["cost"].max()) for i in interps)
                for s in sigmas}

    fig, axes = plt.subplots(len(interps), len(sigmas), squeeze=False,
                             figsize=(3.9 * len(sigmas), 2.7 * len(interps)),
                             sharex=True, sharey=True, layout="constrained")
    meshes = {}
    for r, interp in enumerate(interps):
        for c, sigma in enumerate(sigmas):
            ax = axes[r][c]
            f = fields[(interp, sigma)]
            # pcolormesh with cell EDGES, not centres: the cost map is piecewise
            # constant per (route point, flight level) and imshow would interpolate it
            # a second time, on screen, on top of the interpolation being compared.
            xe = np.append(f["dist"], f["dist"][-1] + (f["dist"][1] - f["dist"][0]))
            step = f["alt"][1] - f["alt"][0]
            ye = np.append(f["alt"] - step / 2, f["alt"][-1] + step / 2)
            meshes[sigma] = ax.pcolormesh(xe, ye, f["cost"].T, cmap=cmap, vmin=0.0,
                                          vmax=vmax[sigma], shading="flat",
                                          rasterized=True)
            ax.axhline(float(f["cruise_alt_ft"]), color="#0b0b0b", lw=0.9, ls=(0, (3, 2)))
            ax.tick_params(labelsize=7, colors="#52514e")
            if r == 0:
                ax.set_title(f"$\\sigma$ = {sigma:g}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{interp}\naltitude (ft)", fontsize=8)
            nz = 100.0 * (f["cost"] > 0).mean()
            ax.annotate(f"max {f['cost'].max():.3g}\n{nz:.0f}% cells > 0",
                        (0.97, 0.96), xycoords="axes fraction", ha="right", va="top",
                        fontsize=6.5, color="#0b0b0b",
                        bbox=dict(facecolor="#ffffff", edgecolor="none", alpha=0.75,
                                  pad=1.5))

    if scale == "shared":
        cb = fig.colorbar(meshes[sigmas[-1]], ax=axes, shrink=0.85, pad=0.015)
        cb.set_label("contrail cost", fontsize=8, color="#52514e")
        cb.ax.tick_params(labelsize=7, colors="#52514e")
        note = "one colour scale for all panels"
    else:
        for c, sigma in enumerate(sigmas):
            cb = fig.colorbar(meshes[sigma], ax=[axes[r][c] for r in range(len(interps))],
                              shrink=0.85, pad=0.015)
            cb.ax.tick_params(labelsize=7, colors="#52514e")
            if c == len(sigmas) - 1:
                cb.set_label("contrail cost", fontsize=8, color="#52514e")
        note = ("one colour scale per column, so columns are NOT comparable — "
                "read the per-panel maximum for that")

    any_f = next(iter(fields.values()))
    fig.supxlabel("distance along the flown route (km)", fontsize=8, color="#52514e")
    fig.suptitle(f"Contrail cost field sampled along the route — {flight_id} "
                 f"({any_f['actype']}, {float(any_f['km']):.0f} km).  Dashed: real mean "
                 f"cruise altitude.  {note}.", fontsize=9)

    out = fig_dir / f"cost_field_{flight_id}_{scale}.png"
    for path in (out, out.with_suffix(".pdf")):
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
        print(f"  wrote {path}")
    plt.close(fig)

    if cells:
        write_cost_field_cells(fields, flight_id, fig_dir, interps, sigmas)
    return out


def write_cost_field_cells(fields: dict, flight_id: str, fig_dir: Path,
                           interps: list[str], sigmas: list[float]) -> None:
    """Every cell of the sampled field, as numbers: one csv plus one labelled panel each.

    The csv is the authoritative copy — full float precision, one row per
    (interpolation, sigma, route point, flight level). The panels round to three
    decimals, which is where the field stops being distinguishable from zero anyway
    (the smallest non-zero value on this flight is 8e-5), and exist because a 41 x 10
    table is faster to read as a grid than as 410 rows.

    One panel PER (interpolation, sigma) rather than the 2 x 3 layout of the figure
    above: 41 columns of "0.000" need about nine inches on their own, and six of those
    in one canvas is a poster, not a figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    rows = []
    for (interp, sigma), f in fields.items():
        for i, d in enumerate(np.asarray(f["dist"])):
            for j, a in enumerate(np.asarray(f["alt"])):
                rows.append((interp, sigma, i, float(d), j, float(a),
                             float(f["cost"][i, j])))
    df = pd.DataFrame(rows, columns=["interp", "sigma", "i_dist", "dist_km",
                                     "j_alt", "alt_ft", "cost"])
    csv = fig_dir / f"cost_field_{flight_id}_cells.csv"
    df.to_csv(csv, index=False)
    print(f"  wrote {csv}  ({len(df):,} cells)")

    cmap = LinearSegmentedColormap.from_list("contrail_cost", COST_RAMP)
    for interp in interps:
        for sigma in sigmas:
            f = fields[(interp, sigma)]
            cost = np.asarray(f["cost"])
            dist = np.asarray(f["dist"])
            alt = np.asarray(f["alt"])
            vmax = float(cost.max()) or 1.0
            fig, ax = plt.subplots(figsize=(0.30 * len(dist) + 1.4,
                                            0.30 * len(alt) + 1.2),
                                   layout="constrained")
            ax.pcolormesh(np.arange(len(dist) + 1), np.arange(len(alt) + 1), cost.T,
                          cmap=cmap, vmin=0.0, vmax=vmax, shading="flat")
            for i in range(len(dist)):
                for j in range(len(alt)):
                    v = cost[i, j]
                    # Ink flips on the fill, which is the one place a value may not
                    # wear a fixed text token: at the dark end of the ramp a dark
                    # label is unreadable, and the number is the point of the figure.
                    ax.text(i + 0.5, j + 0.5, "." if v == 0 else f"{v:.3f}",
                            ha="center", va="center", fontsize=4.6,
                            color="#ffffff" if v / vmax > 0.55 else "#0b0b0b")
            ax.set_xticks(np.arange(len(dist)) + 0.5)
            ax.set_xticklabels([f"{d:.0f}" for d in dist], fontsize=4.6, rotation=90)
            ax.set_yticks(np.arange(len(alt)) + 0.5)
            ax.set_yticklabels([f"{a:.0f}" for a in alt], fontsize=5.5)
            ax.set_xlabel("distance along the flown route (km)", fontsize=7)
            ax.set_ylabel("altitude (ft)", fontsize=7)
            ax.tick_params(length=0)
            ax.set_title(f"{flight_id} — {interp}, $\\sigma$ = {sigma:g}   "
                         f"env.cost_map, {cost.shape[0]} route points x "
                         f"{cost.shape[1]} levels   (max {vmax:.4g}; "
                         f'"." is exactly zero; full precision in the csv)',
                         fontsize=7.5)
            out = fig_dir / f"cost_cells_{flight_id}_{interp}_sigma{sigma:g}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="#ffffff")
            print(f"  wrote {out}")
            plt.close(fig)


def _run_task(task: tuple) -> dict:
    """One (flight, sigma, method). Never raises — one bad flight must not kill
    the pool; the failure is written as a stub and reported in the progress row."""
    meta, sigma, slice_path, out_dir, method, m0, interp, sweeps = task
    out_dir = Path(out_dir)
    flight_id = meta["flight_id"]
    out_path = out_path_for(out_dir, flight_id, sigma, method)
    row = dict(utc_time=datetime.now(timezone.utc).isoformat(),
               flight_id=flight_id, sigma=sigma)
    try:
        row.update(run_one_flight(meta, sigma, Path(slice_path), out_dir,
                                  _WORKER_GRID, method=method, m0=m0,
                                  interp=interp, sweeps=sweeps,
                                  global_interp=_WORKER_GINTERP.get(float(sigma))))
    except Exception as exc:  # noqa: BLE001 — isolate one flight's failure
        err = f"{type(exc).__name__}: {exc}"
        row.update(status="error", error=err)
        save_error(out_path, flight_id, sigma, err,
                   actype=meta.get("actype"), m0=m0)
        traceback.print_exc()
    return row


# ── batch driver ─────────────────────────────────────────────────────────────
def run_batch(data_path: Path, grid_path: Path, out_dir: Path, sigmas: list[float],
              flights: list[str] | None, limit: int | None, methods: list[str],
              retry_failed: bool = False, workers: int = 1,
              shard: tuple[int, int] | None = None,
              m0: float = M0, actypes: list[str] | None = None,
              interp: str = INTERP_SHAPE, sweeps: tuple = DP_SWEEPS,
              global_interp_dir: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_flight_cache"

    idx = load_flight_index(data_path)
    if flights:
        idx = idx[idx["flight_id"].isin(flights)].reset_index(drop=True)
        missing = set(flights) - set(idx["flight_id"])
        if missing:
            print(f"  WARNING: {len(missing)} flight ids not found and skipped")
    if actypes:
        want = {a.upper() for a in actypes}
        idx = idx[idx["typecode"].astype(str).str.upper().isin(want)].reset_index(drop=True)
    if limit is not None:
        idx = idx.head(limit).reset_index(drop=True)
    if shard is not None:
        i, n = shard
        idx = idx.iloc[i::n].reset_index(drop=True)
        print(f"  shard {i}/{n}: this task handles {len(idx)} of the selected flights")

    print(f"Running graph-2D ({', '.join(methods)}, workers={workers}, m0={m0:g}) "
          f"on {len(idx)} flights x {len(sigmas)} sigma(s)")
    print(f"  WS grid={WS_GRID}")
    print(f"  window: cruise_alt -{ALT_BUFFER_FT_MIN:g} / +{ALT_BUFFER_FT_MAX:g} ft "
          f"at {ALT_STEP_FT:g} ft levels\n  out={out_dir}\n")

    full_df: pd.DataFrame | None = None
    tasks: list[tuple] = []
    skipped = 0
    for _, meta_row in idx.iterrows():
        meta = dict(meta_row)
        typecode = meta.get("typecode")
        meta["actype"] = str(typecode).upper() if pd.notna(typecode) else "B738"
        flight_id = meta["flight_id"]
        for sigma in sigmas:
            for method in methods:
                out_path = out_path_for(out_dir, flight_id, sigma, method)
                if is_done(out_path, retry_failed=retry_failed):
                    skipped += 1
                    continue
                if full_df is None:
                    print("  (loading full parquet once to build flight slices ...)")
                    full_df = pd.read_parquet(data_path)
                slice_path = ensure_flight_slice(full_df, cache_dir, flight_id)
                tasks.append((meta, sigma, str(slice_path), str(out_dir), method, m0,
                              interp, sweeps))
    del full_df

    print(f"  {len(tasks)} to run, {skipped} already done\n")
    done = failed = 0

    def _handle(row: dict, i: int) -> None:
        nonlocal done, failed
        tag = f"[{i}/{len(tasks)}] {row['flight_id']} sigma={row['sigma']:g}"
        if row.get("status") == "error":
            failed += 1
            print(f"{tag}  FAILED: {row.get('error')}")
        else:
            done += 1
            print(f"{tag}  done in {row.get('elapsed_s')}s "
                  f"({row.get('n_success')}/{row.get('n_solves')} points)")
        log_progress(out_dir, row)

    gdir = str(global_interp_dir) if global_interp_dir is not None else None
    if workers <= 1:
        _worker_init(str(grid_path), sigmas, gdir)
        for i, task in enumerate(tasks, 1):
            _handle(_run_task(task), i)
    else:
        from multiprocessing import Pool
        print(f"  parallel: {workers} worker processes\n")
        with Pool(workers, initializer=_worker_init,
                  initargs=(str(grid_path), sigmas, gdir)) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_task, tasks), 1):
                _handle(row, i)

    print(f"\n{'='*70}\ngraph-2D batch finished: {done} done, {skipped} skipped, "
          f"{failed} failed.\nResults in {out_dir}")


def run_cost_field(data_path: Path, grid_path: Path, out_dir: Path, fig_dir: Path,
                   sigmas: list[float], flights: list[str] | None, limit: int | None,
                   m0: float, actypes: list[str] | None, interps: list[str],
                   scale: str = "per-sigma", cells: bool = False,
                   global_interp_dir: Path | None = None) -> None:
    """--plot-cost-field: draw the sampled cost field, solve nothing.

    Shares the flight index, the flight-slice cache and the GridWeather load with the
    solve path, so the field drawn is the field a solve would have seen. ``out_dir`` is
    still used — for its ``_flight_cache`` — but nothing is written into it.
    """
    idx = load_flight_index(data_path)
    if flights:
        idx = idx[idx["flight_id"].isin(flights)].reset_index(drop=True)
        missing = set(flights) - set(idx["flight_id"])
        if missing:
            print(f"  WARNING: {len(missing)} flight ids not found and skipped")
    if actypes:
        want = {a.upper() for a in actypes}
        idx = idx[idx["typecode"].astype(str).str.upper().isin(want)].reset_index(drop=True)
    if limit is not None:
        idx = idx.head(limit).reset_index(drop=True)
    if idx.empty:
        print("no flights selected")
        return

    cache_dir = out_dir / "_flight_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cost field for {len(idx)} flight(s) x {len(sigmas)} sigma(s) "
          f"x {len(interps)} interpolation(s)\n  out={fig_dir}\n")

    gdir = str(global_interp_dir) if global_interp_dir is not None else None
    _worker_init(str(grid_path), sigmas, gdir)
    full_df = pd.read_parquet(data_path)
    for _, meta_row in idx.iterrows():
        meta = dict(meta_row)
        typecode = meta.get("typecode")
        meta["actype"] = str(typecode).upper() if pd.notna(typecode) else "B738"
        slice_path = ensure_flight_slice(full_df, cache_dir, meta["flight_id"])
        plot_cost_field(meta, sigmas, slice_path, _WORKER_GRID, m0, interps, fig_dir,
                        scale=scale, cells=cells, global_interp=_WORKER_GINTERP)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--grid-path", default=str(DEFAULT_GRID_PATH))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--sigmas", type=float, nargs="+", default=[2.0])
    p.add_argument("--methods", nargs="+", default=["dp", "dp_muo"],
                   choices=["dp", "dp_muo"],
                   help="Solvers to run; each writes its own pickle per flight.")
    p.add_argument("--flights", nargs="+", default=None)
    p.add_argument("--flights-file", default=None, metavar="PATH",
                   help="Flight ids, one per line — use the list the OpenTOP run "
                        "used so both batches cover the same population.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--m0", type=float, default=M0,
                   help=f"Initial mass as a fraction of MTOW (default: {M0}). Must "
                        "match the OpenTOP run this is compared against.")
    p.add_argument("--actypes", nargs="+", default=None, metavar="ICAO")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard", default=None, metavar="i/N")
    p.add_argument("--global-interp", action="store_true",
                   help="Use the prebuilt global interpolant (global_interp_sigma<s>."
                        "casadi in --global-interp-dir) shifted per flight, instead of "
                        "fitting a bspline per flight (164 s median). The field is NOT "
                        "identical to a per-flight fit — uniform ~2e-4 residual — so do "
                        "not mix both in one OUT; context.interpolant_shape records it.")
    p.add_argument("--global-interp-dir", default=str(Path(DEFAULT_DATA_PATH).parent),
                   help="Where global_interp_sigma<s>.casadi/.meta.json live.")
    p.add_argument("--interp", choices=["bspline", "linear"], nargs="+",
                   default=[INTERP_SHAPE],
                   help=f"How the contrail cost grid is interpolated (default "
                        f"{INTERP_SHAPE}). The graph solvers never differentiate it, so "
                        f"'linear' only shifts the sampled cost_map values — and builds "
                        f"in a fraction of a second. Give each shape its own --out-dir. "
                        f"More than one is accepted ONLY with --plot-cost-field, which "
                        f"draws a row per shape; a solve run writes one --out-dir and "
                        f"cannot hold two.")
    p.add_argument("--sweeps", nargs="+", default=list(DP_SWEEPS),
                   choices=list(DP_SWEEPS),
                   help="Which DP sweeps to run (default: both). The DP has no "
                        "epsilon-constraint since 2026-10-19. Recorded as "
                        "context.dp_sweeps; give each choice its own --out-dir, the "
                        "filename does not carry it. Ignored by --methods dp_muo.")
    p.add_argument("--plot-cost-field", action="store_true",
                   help="Draw env.cost_map for the selected flights (one row per "
                        "--interp, one column per --sigmas) and stop. Solves nothing "
                        "and writes no pickle. Use --flights to name the flight.")
    p.add_argument("--fig-dir", default=str(Path(__file__).resolve().parent
                                            / "figures" / "cost_field"),
                   help="Where --plot-cost-field writes.")
    p.add_argument("--cost-field-scale", choices=["per-sigma", "shared"],
                   default="per-sigma",
                   help="Colour scale of --plot-cost-field. per-sigma (default): one "
                        "scale per column, so linear vs bspline is a fair read and every "
                        "column stays legible; columns are then not comparable. shared: "
                        "one scale for all six panels, which makes the smoothed columns "
                        "nearly white. Both read the cached .npz, so switching is free.")
    p.add_argument("--cost-field-cells", action="store_true",
                   help="Also write every cell of the sampled field as numbers: one csv "
                        "at full precision, plus one labelled panel per interpolation "
                        "and sigma. Reads the cached .npz, so it costs nothing after "
                        "the field has been built once.")
    args = p.parse_args()

    if len(args.interp) > 1 and not args.plot_cost_field:
        p.error("--interp takes one shape for a solve run: each shape needs its own "
                "--out-dir, and a directory mixing the two is not comparable. "
                "Several shapes are only allowed with --plot-cost-field.")

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
            p.error("--shard must look like i/N (e.g. 3/48)")
        if not (0 <= shard[0] < shard[1]):
            p.error("--shard i/N needs 0 <= i < N")

    if args.plot_cost_field:
        run_cost_field(
            data_path=Path(args.data_path), grid_path=Path(args.grid_path),
            out_dir=Path(args.out_dir), fig_dir=Path(args.fig_dir),
            sigmas=args.sigmas, flights=flights, limit=args.limit, m0=args.m0,
            actypes=args.actypes, interps=args.interp, scale=args.cost_field_scale,
            cells=args.cost_field_cells,
            global_interp_dir=(Path(args.global_interp_dir)
                               if args.global_interp else None),
        )
        return

    run_batch(
        data_path=Path(args.data_path), grid_path=Path(args.grid_path),
        out_dir=Path(args.out_dir), sigmas=args.sigmas, flights=flights,
        limit=args.limit, methods=args.methods, retry_failed=args.retry_failed,
        workers=args.workers, shard=shard, m0=args.m0, actypes=args.actypes,
        interp=args.interp[0], sweeps=tuple(args.sweeps),
        global_interp_dir=(Path(args.global_interp_dir) if args.global_interp else None),
    )


if __name__ == "__main__":
    main()
