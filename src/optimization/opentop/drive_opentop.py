from __future__ import annotations
import tempfile
import opentop as top
import time
import gzip
import pickle
import sys
from pathlib import Path
import os

# Windows consoles default to cp1252 and choke on the → / σ / 📈 characters some
# prints (here and in src.postprocess.plots) emit — force UTF-8 so they don't crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 — best effort; older/foreign stream types
        pass
from src.contrails.cost_grid import prepare_grid
from src.data.route import real_flight_route
import numpy as np
from openap.aero import ft
from src.data.traffic import FlightData
from src.data.meteo import GridWeather
from src.contrails.conditions import agg_conditions
from src.optimization.opentop.opentop_optimizer import OpenTOPOptimizer
from src.optimization.base import BaseOptimizer, OptResult

# Project root (…/OPTIMAL), so data paths resolve no matter the working directory
# the driver is launched from (relative paths break when cwd isn't the repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

from src.paths import DATA_DIR, RUN_DIR  # noqa: E402
from src.config import TRACK_TOLERANCE_M  # noqa: E402


def save_flight(opt, out_path: Path, context: dict, error: str | None = None) -> None:
    """Persist one flight's entire run as a *single* gzip'd pickle at ``out_path``.

    One file per flight (not four) keeps a 4500-flight batch to ~4500 files
    instead of ~18000, which is far easier to glob/iterate in post-processing.

    On-disk payload (one dict)::

        {
            "context": {...},            # flight identity, scalars, df_cost grid
            "results": {                 # full OptResult dicts, grouped by method
                "weighted_sum":        [rec, rec, ...],
                "epsilon_constraint":  [rec, rec, ...],
            },
            "summary": [ {fuel_kg, contrail, cost, success, method, coef, ...}, ... ],
            "error":   None,             # str if the solve failed (for batch resume)
        }

    Each record is ``OptResult.as_dict()`` — profile arrays (incl. lat/lon) plus
    the trajectory DataFrame carried in ``meta['trajectory']``. ``summary`` holds
    the scalar rows only, so a batch can concat them into one table without
    unpickling the heavy trajectories.
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

    print(f"\n[save_flight] wrote {len(opt.results)} solve(s) across "
          f"{len(by_method)} method(s) → {out_path}")


def _results_to_records(opt, sigma) -> list[dict]:
    """Adapt ``opt.results`` (OptResult objects) into the plain-dict schema the
    ``src.postprocess.plots`` helpers expect (trajectory + method labels + the
    fuel/contrail/objective/iteration scalars they read)."""
    records = []
    for res in opt.results:
        m = res.meta
        traj = m.get("trajectory")
        method = m.get("method")
        # The Pareto helper does np.sum(grid_cost); the trajectory's grid_cost
        # column can carry NaNs (contrail uses np.nansum), which would poison that
        # sum → NaN y-values → empty plot. Hand it the already-correct scalar.
        grid_cost = [res.contrail]
        rec = {
            "optimizer": "OpenTOP",
            "trajectory": traj,
            "method": method,
            "objective_type": method,
            "success": res.success,
            "solver_success": res.success,
            "fuel_kg": res.fuel_kg,
            "contrail": res.contrail,
            "fuel_cost": float(traj["mass"].iloc[0] - traj["mass"].iloc[-1]) if traj is not None else res.fuel_kg,
            "grid_cost": grid_cost,
            "total_objective": [res.cost],
            "solver_iterations": m.get("n_iter"),
            "smooth_factor": sigma,
        }
        if method == "epsilon_constraint":
            # ε label for the trajectory/altitude plots; fuel budget for the Pareto.
            rec["coef"] = None
            rec["epsilon_factor"] = m.get("fuel_slack_frac")
            rec["fuel_slack_frac"] = m.get("fuel_slack_frac")
            rec["fuel_max"] = m.get("fuel_budget_kg")
        else:
            rec["coef"] = m.get("coef")
        records.append(rec)
    return records


def _fuel_only_baseline(opt):
    """Solve the fuel-only anchor now and return it as an OptResult.

    pareto_front() solves it too but does not keep it in opt.results; as in
    batch_opentop, it is read right after warm_start(), while cruise still holds
    THIS solve's contrail value. warm_start() is idempotent, so the sweeps reuse it.
    """
    traj = opt.warm_start()
    if traj is None:
        return None
    return opt._traj_to_result(traj, opt.warm_start_s, opt.warm_start_success,
                               method="fuel_only", objective=opt.warm_start_objective,
                               n_iter=opt.warm_start_n_iter)


def plot_flight(opt, context: dict | None = None, *, df_contrails=None,
                real_trajectory=None, save_dir=None, with_map: bool = True,
                alt_min: float | None = None, alt_max: float | None = None,
                show: bool = True, pareto_extra=None, with_altitude: bool = True,
                pareto_opentop_style=None, pareto_title=None, contrail_field=None,
                pareto_annotate: bool = True, pareto_eps_anchor=None):
    """Quick visualization of a driver run — reuses the shared plot helpers in
    ``src.postprocess.plots`` instead of saving. Drop-in alternative to
    ``save_flight`` when you just want to *see* the sweep.

    Renders, from ``opt.results``:
      * the Pareto front (weighted-sum vs ε-constraint) — ``plot_pareto_comparison``
      * altitude-vs-distance with the contrail background, per method —
        ``plot_altitude_vs_distance_all_trajectories``
      * (``with_map=True``) the lateral tracks + contrails on a map —
        ``plot_multiple_trajectories_with_contrails`` (cartopy, slower)

    Args:
        opt: the solved ``OpenTOPOptimizer`` (uses ``opt.results``).
        context: optional context dict (as built for ``save_flight``); supplies
            ``sigma``, ``df_cost`` and the altitude window as defaults.
        df_contrails: cost grid for the background (defaults to ``context['df_cost']``).
        real_trajectory: optional real flown flight to overlay on the map.
        pareto_extra: optional extra series for the Pareto panel — see
            ``plot_pareto_comparison``. Lets other solvers (DP, DP_MuO) share the
            SAME axes instead of getting a second figure.
        save_dir: if given, figures are also written there.
        with_map: also draw the cartopy map (off by default — heavier/slower).
        with_altitude: draw the altitude-vs-distance figures (one per method).
        pareto_opentop_style: optional ``{"color", "label"}`` for the OpenTOP
            sweeps on the Pareto panel — see ``plot_pareto_comparison``.
        pareto_title: optional title for the Pareto panel (may be multi-line).
        pareto_annotate: write each Pareto point's sweep value (coefficient, ε) beside it.
        pareto_eps_anchor: (fuel, contrail) of the fuel-only point, to start the
            ε-constraint front line there -- see ``plot_pareto_comparison``.
        contrail_field: optional background for the altitude figures, drawn instead
            of the one-hour grid slice -- see ``plot_altitude_vs_distance_all_trajectories``.
        alt_min, alt_max: VISUALIZATION-ONLY altitude window (ft) for the contrail
            background + y-axis. Default None → use the context's optimization
            window (``min_alt_ft``/``max_alt_ft``). Widen these to see contrails
            outside the searched band WITHOUT changing the optimization.
        show: call ``plt.show()`` at the end.
    """
    # Imported here (not at module top) so the driver stays light — plots.py
    # pulls in cartopy, which is slow to import.
    import matplotlib.pyplot as plt
    from src.postprocess.plots import (
        plot_pareto_comparison,
        plot_altitude_vs_distance_all_trajectories,
        plot_multiple_trajectories_with_contrails,
    )

    ctx = context or {}
    sigma = ctx.get("sigma", "?")
    if df_contrails is None:
        df_contrails = ctx.get("df_cost")
    # Plot window: explicit overrides win; else the optimization window from context.
    if alt_min is None:
        alt_min = ctx.get("min_alt_ft", 29000)
    if alt_max is None:
        alt_max = ctx.get("max_alt_ft", 45000)

    records = _results_to_records(opt, sigma)
    fo = [r for r in records if r["objective_type"] == "fuel_only"]
    ws = [r for r in records if r["objective_type"] not in ("epsilon_constraint", "fuel_only")]
    ec = [r for r in records if r["objective_type"] == "epsilon_constraint"]
    # The fuel-only point on the Pareto (a star), and the epsilon front line starting
    # from it, unless the caller passed its own (post_treat does).
    if fo and pareto_extra is None:
        fo[0]["label"] = "fuel-only"
        pareto_extra = [{"label": "Fuel-only", "records": fo[:1], "color": "black",
                         "marker": "*", "size": 140, "label_keys": ("label",)}]
    if fo and pareto_eps_anchor is None and fo[0]["success"]:
        pareto_eps_anchor = (fo[0]["fuel_cost"], fo[0]["contrail"])

    # 1) Pareto front (matplotlib only — fast).
    if ws or ec:
        try:
            plot_pareto_comparison(ws, ec, save_dir=save_dir,
                                   extra_series=pareto_extra,
                                   opentop_style=pareto_opentop_style,
                                   title=pareto_title, show=False,
                                   annotate=pareto_annotate,
                                   eps_anchor=pareto_eps_anchor)
        except Exception as exc:  # noqa: BLE001 — one panel failing shouldn't kill the rest
            print(f"[plot_flight] Pareto plot failed: {exc}")

    # 2) Altitude vs distance with contrail background, per method.
    for recs, name in [(ws, "weighted_sum"), (ec, "epsilon_constraint")]:
        if not recs or not with_altitude:
            continue
        try:
            plot_altitude_vs_distance_all_trajectories(
                results=recs, df_contrails=df_contrails, ts_select=0,
                alt_min=alt_min, alt_max=alt_max, save_dir=save_dir,
                filename=f"altitude_vs_distance_{name}.png",
                real_trajectory=real_trajectory, field=contrail_field,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[plot_flight] altitude plot ({name}) failed: {exc}")

    # 3) Optional lateral tracks + contrails on a map (cartopy — slower).
    if with_map:
        try:
            plot_multiple_trajectories_with_contrails(
                results=records, df_contrails=df_contrails,
                real_trajectory=real_trajectory, ts_select=0,
                alt_min=alt_min, alt_max=alt_max,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[plot_flight] map plot failed: {exc}")

    if show:
        plt.show()


def run_drive(flight_name, begin, end, base_id):
    sigma = 2
    m0 = 0.8

    # 1) Flight setup
    TMP = Path(tempfile.gettempdir())
    interpolant_cache = os.environ.get(
        "INTERPOLANT_CACHE",
        str(TMP / f"opentop_interp_{flight_name}_sigma{sigma}.casadi"),
    )
    # fd = FlightData(start=begin, stop=end, callsign=callsign)
    fd = FlightData(start=begin, stop=end, base_id=base_id,
                    path=str(DATA_DIR / "eu_flights_2022feb20_persistent_all.parquet"))
    # 2) Attach ERA5 meteo (+ wind for the TAS-corrected Mach), then build the
    #    cost grid and interpolant from it.
    fd.attach_meteo("era5", buffer_hours=2.0, wind=True)
    df_cost = fd.cost_grid(sigma)
    interpolant = fd.interpolant(sigma, cache_path=interpolant_cache)
    # Real-flight mean altitude — only the reference for resampling the lateral
    # track. The optimization window is NOT taken from it: the fuel-only warm
    # start runs unbounded in altitude and the contrail sweeps are then confined
    # to that fuel-optimal trajectory ± RECENTER_FT (see recenter_window_ft).
    avg_alt_m     = float(fd.flight.data.altitude.mean()) * ft
    cruise_alt_ft = avg_alt_m / ft
    RECENTER_FT   = 4000
    mach          = fd.compute_cruise_mach()
    actype        = fd.aircraft_type                    # from traffic's aircraft DB (callsign path)

    print(f"cruise_alt_ft={cruise_alt_ft:.0f}  mach={mach:.3f}  actype={actype}  m0={m0:.2f}  "
          f"(fuel-only unbounded; contrail sweeps ±{RECENTER_FT} ft)")

    # Lateral path = the REAL flown track, resampled exactly as model_2D consumes
    # it, so both optimizers search the same corridor and only the vertical
    # profile differs between them.
    route, total_km = real_flight_route(fd.flight, mach=mach, avg_alt_m=avg_alt_m,
                                        roc=500, alt_step_ft=500)
    rlat = np.array([p[0] for p in route])
    rlon = np.array([p[1] for p in route])

    optimizer = top.Cruise(
        actype,
        (fd.initial_latitude, fd.initial_longitude),
        (fd.final_latitude, fd.final_longitude),
        m0=m0,
        mach_value=mach,
        use_synonym=True,   # resolve types OpenAP lacks directly (e.g. B77L → B77W)
    )
    # optimizer.enable_wind((fd.wind))
    # follow_track pins the ground track; fix_track_angle would pin the *heading*
    # instead, which with wind drifts ~34 km off the real path. The two conflict —
    # heading must stay free to crab.
    optimizer.follow_track(rlat, rlon, tolerance_m=TRACK_TOLERANCE_M)
    optimizer.fix_mach_number()
    optimizer.allow_cruise_descent()
    # 2) The optimizer under test. environment=None keeps opentop's native
    #    fuel/contrail; pass setup.env to score it on the shared graph physics.
    opt = OpenTOPOptimizer(
        cruise=optimizer,
        interpolant=interpolant,
        recenter_window_ft=RECENTER_FT,   # fuel-only unbounded; contrail sweeps ± this
        mach=mach,
        mass_init_kg=optimizer.mass_init,
        actype=actype
    )

    # 4) Weighted-sum solves at a few contrail weights (BaseOptimizer.optimize).
    baseline = _fuel_only_baseline(opt)
    print("\n-- weighted-sum (optimize) --")
    opt.pareto_front(method="weighted_sum", points=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    # 5) Native epsilon-constraint: minimise contrail s.t. fuel ≤ 1.05·F0.
    print("\n-- epsilon-constraint (solve_epsilon) --")
    opt.pareto_front(method="epsilon_constraint", points=[1.01, 1.02, 1.03, 1.04, 1.05, 1.10])
    if baseline is not None:
        opt.results.insert(0, baseline)   # the fuel-only point, first

    print("\n-- results --")
    for res in opt.results:
        print(f"  method={res.meta['method']}  "
              f"fuel={res.fuel_kg:.1f} kg  contrail={res.contrail:.4f}  ")

    # 6) Persist this run as ONE file per flight so its outputs can be inspected
    #    before batching (in a batch, key the filename by flight_id instead).
    context = dict(
        flight_name=flight_name,
        begin=begin, end=end, sigma=sigma, m0=m0,
        actype=actype, mach=mach, mass_init_kg=optimizer.mass_init,
        cruise_alt_ft=cruise_alt_ft,
        # Actual contrail-sweep window, recentered on the fuel-only warm start.
        min_alt_ft=opt.h_min_ft, max_alt_ft=opt.h_max_ft,
        origin=(fd.initial_latitude, fd.initial_longitude),
        destination=(fd.final_latitude, fd.final_longitude),
        wind_enabled=False, track_fixed=True,   # enable_wind is commented out above
        track_source="real_flight_route",
        track_length_km=total_km,
        df_cost=df_cost,
    )
    out_path = (Path(__file__).resolve().parent / "results_test"
                / f"{flight_name}_sigma{sigma}.pkl.gz")
    # save_flight(opt, out_path, context)
    plot_flight(opt, context, df_contrails=df_cost, real_trajectory=fd,
                with_map=True, show=True)


def run_drive_track_cruise(flight_name: str = "Firs_flight",
                           base_id: str = "0100a3_00002_0",
                           begin: str = "2022-02-20 00:00",
                           end: str = "2022-02-20 01:34",
                           sigma: float = 2, m0: float = 0.8,
                           with_map: bool = True):
    """run_drive with TrackCruise (``--track-mode exact``) in place of top.Cruise.

    Same flight setup, same OpenTOPOptimizer sweeps; only the lateral model
    changes: the position is the arc length ``s`` along the flown track, so the
    solution lies ON it and there is no tolerance. Prints a check that ``s`` runs
    from 0 to the track length, then plots as run_drive does.
    """
    # Imported here, as in batch_opentop: TrackCruise is pinned to one opentop
    # version and run_drive must not depend on it.
    from src.optimization.opentop.track_cruise import TrackCruise

    TMP = Path(tempfile.gettempdir())
    interpolant_cache = os.environ.get(
        "INTERPOLANT_CACHE",
        str(TMP / f"opentop_interp_{base_id}_sigma{sigma:g}.casadi"),
    )
    fd = FlightData(start=begin, stop=end, base_id=base_id,
                    path=str(DATA_DIR / "eu_flights_2022feb20_persistent_all.parquet"))
    fd.attach_meteo("era5", buffer_hours=2.0, wind=True)
    df_cost = fd.cost_grid(sigma)
    interpolant = fd.interpolant(sigma, cache_path=interpolant_cache)

    avg_alt_m     = float(fd.flight.data.altitude.mean()) * ft
    cruise_alt_ft = avg_alt_m / ft
    RECENTER_FT   = 4000
    mach          = fd.compute_cruise_mach()
    actype        = fd.aircraft_type

    print(f"[TrackCruise] {base_id}  cruise_alt_ft={cruise_alt_ft:.0f}  mach={mach:.3f}  "
          f"actype={actype}  m0={m0:.2f}")

    route, total_km = real_flight_route(fd.flight, mach=mach, avg_alt_m=avg_alt_m,
                                        roc=500, alt_step_ft=500)
    rlat = np.array([p[0] for p in route])
    rlon = np.array([p[1] for p in route])

    optimizer = TrackCruise(
        actype,
        (fd.initial_latitude, fd.initial_longitude),
        (fd.final_latitude, fd.final_longitude),
        m0=m0,
        mach_value=mach,
        use_synonym=True,
    )
    # No tolerance: TrackCruise.follow_track imposes the track exactly.
    optimizer.follow_track(rlat, rlon)
    optimizer.fix_mach_number()
    optimizer.allow_cruise_descent()

    opt = OpenTOPOptimizer(
        cruise=optimizer,
        interpolant=interpolant,
        recenter_window_ft=RECENTER_FT,
        mach=mach,
        mass_init_kg=optimizer.mass_init,
        actype=actype
    )

    t0 = time.time()
    baseline = _fuel_only_baseline(opt)
    print("\n-- weighted-sum (optimize) --")
    opt.pareto_front(method="weighted_sum", points=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    print("\n-- epsilon-constraint (solve_epsilon) --")
    opt.pareto_front(method="epsilon_constraint", points=[1.01, 1.02, 1.03, 1.04, 1.05, 1.10])
    if baseline is not None:
        opt.results.insert(0, baseline)   # the fuel-only point, first
    print(f"\n[TrackCruise] sweeps took {time.time() - t0:.0f} s")

    # On-track check: s must start at 0 and end at the track length (both pinned).
    length_km = optimizer.track_ref[2] / 1000
    print(f"\n-- results (track length {length_km:.1f} km, real_flight_route {total_km:.1f} km) --")
    for res in opt.results:
        traj = res.meta.get("trajectory")
        s_info = ""
        if traj is not None and "s" in traj:
            s_info = (f"  s=[{traj['s'].iloc[0] / 1000:.1f}, "
                      f"{traj['s'].iloc[-1] / 1000:.1f}] km")
        print(f"  method={res.meta['method']:<20} fuel={res.fuel_kg:.1f} kg  "
              f"contrail={res.contrail:.4f}  success={res.success}{s_info}")

    context = dict(
        flight_name=base_id, flight_id=base_id,
        begin=begin, end=end, sigma=sigma, m0=m0,
        actype=actype, mach=mach, mass_init_kg=optimizer.mass_init,
        cruise_alt_ft=cruise_alt_ft,
        min_alt_ft=opt.h_min_ft, max_alt_ft=opt.h_max_ft,
        origin=(fd.initial_latitude, fd.initial_longitude),
        destination=(fd.final_latitude, fd.final_longitude),
        wind_enabled=False, track_fixed=True,
        track_mode="exact", track_tolerance_m=None,
        track_source="real_flight_route",
        track_length_km=total_km,
        df_cost=df_cost,
    )
    plot_flight(opt, context, df_contrails=df_cost, real_trajectory=fd,
                with_map=with_map, show=True)
    return opt, context


# The batch (batch_opentop.py) writes one gzip'd pickle per flight here, named
# <flight_id>_sigma<σ>.pkl.gz, using the same grid it slices meteo from.
BATCH_RUN_DIR = RUN_DIR / "open_top_run" / "opentop_batch"
GRID_FILE = DATA_DIR / "grid_era5_smoothed.parquet.gz"
FLIGHTS_FILE = DATA_DIR / "eu_flights_2022feb20_persistent_all.parquet"


def load_saved_opt(result_file: Path):
    """Rebuild a ``(opt, context)`` pair from one flight's saved batch pickle so
    :func:`plot_flight` can render it.

    ``save_flight`` stores ``{context, results, summary, error}`` where
    ``results`` is ``{method: [OptResult.as_dict(), ...]}``. Each record maps
    straight back with ``OptResult(**rec)`` (its ``meta`` already carries the
    trajectory + labels the plot helpers read). ``context`` does NOT include the
    bulky ``df_cost`` grid — the caller rebuilds it from the shared grid."""
    obj = pickle.load(gzip.open(result_file, "rb"))
    if obj.get("error"):
        print(f"  note: this flight was saved with an error: {obj['error']}")
    results = [OptResult(**rec)
               for recs in obj["results"].values() for rec in recs]
    opt = type("Opt", (), {"results": results})()
    return opt, obj["context"]


def plot_saved_flight(flight_id: str, sigma: int = 2, with_map: bool = False):
    """Load the saved batch result for ``flight_id`` (at ``sigma``), print its
    solves, and render it via :func:`plot_flight`."""
    result_file = BATCH_RUN_DIR / f"{flight_id}_sigma{sigma:g}.pkl.gz"
    print(f"Plotting {result_file.name}")

    opt, context = load_saved_opt(result_file)

    # Print the solves (method / fuel / contrail / converged).
    print(f"{'method':<20}{'fuel [kg]':>12}{'contrail':>12}{'converged':>12}")
    for res in opt.results:
        print(f"{res.meta.get('method',''):<20}{res.fuel_kg:>12.1f}"
              f"{res.contrail:>12.2f}{str(res.success):>12}")

    # The batch omits df_cost from the saved file (it is large) — slice it back
    # from the same shared grid, using the flight's origin/dest/time from context.
    o_lat, o_lon = context["origin"]
    d_lat, d_lon = context["destination"]
    gw = GridWeather(GRID_FILE, sigmas=[context["sigma"]])
    df_cost = gw.slice_for_flight(o_lat, o_lon, d_lat, d_lon,
                                  start=context["begin"], stop=context["end"],
                                  sigma=context["sigma"])

    # Optional flown-flight overlay (black dashed on the map / altitude plots).
    real = None
    try:
        real = FlightData(start=str(context["begin"]), stop=str(context["end"]),
                          base_id=context["flight_id"], path=str(FLIGHTS_FILE))
    except Exception as exc:  # noqa: BLE001 — overlay is optional
        print(f"  (no real-flight overlay: {exc})")

    plot_flight(opt, context, df_contrails=df_cost,
                real_trajectory=real, with_map=with_map, show=True)


if __name__ == "__main__":
    flight_name = "Berlin-Madrid"
    # flight_name = "France-Sardinia"
    # begin = "2022-02-20 00:00"
    # end = "2022-02-20 01:34"
    begin = "2023-01-05 09:48"
    end = "2023-01-05 11:00"
    callsign = "RYR880W"
    # base_id="0100a3_00002_0"
    # base_id="02a1b3_00400_0"
    run_drive(flight_name, begin, end, base_id)
    run_drive_track_cruise(flight_name, base_id, begin, end)
    # plot_saved_flight("0100a3_00002_0")
