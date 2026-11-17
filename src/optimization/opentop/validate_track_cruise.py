"""
Validate TrackCruise against opentop's own top.Cruise.

The two only solve the same problem in the limit tolerance_m -> 0, where top.Cruise
stops converging, so they are compared in three steps instead:

  1. MODEL      no solver. TrackCruise's dynamics and objectives, evaluated at
                random points, against top.Cruise's at the same virtual point
                [x_ref(s), y_ref(s), h, m, ts] / [mach, vs, psi_ref(s)]. Machine
                precision, except the ground-velocity check, which uses a finite
                difference of the track spline.
  2. REAL TRACK fuel only. TrackCruise's output is re-checked independently: the
                distance flown vs the spline's true arc length, and the solver's
                mass drop vs the numeric openap fuel flow. top.Cruise (tol 10 km
                and 1 km, started from TrackCruise's optimum) is then compared.
                It is NOT a superset of TrackCruise once discretised:
                constrain_track pins every interior node ON the spline (equality,
                opentop/_track.py), tolerance_m only applies between nodes, and the
                heading is linear in time per interval with one dt for all, so it
                snakes from node to node and flies slightly further than the spline
                (02a1b3_00400_0: +0.64 km, heading total variation 190 vs 107 deg,
                same optimum cold or warm). So the check is that its extra fuel is
                the extra distance: dfuel/fuel = ddist/dist, not dfuel <= 0.
  3. SMOOTH     a straight track in the projection, where top.Cruise converges at
                tolerance 100 m (no turns), so both must find the same optimum.

Usage:
    python -m src.optimization.opentop.validate_track_cruise [--contrail]
    python -m src.optimization.opentop.validate_track_cruise \
        --flights 3_Results/flight_selection/selection/flights.txt --limit 10

Without --flights it runs the drive_opentop flight. With it, each flight's begin/end
come from the dataset index and its actype/m0 from the manifest.csv next to the list.

--contrail adds the grid-cost (contrail) objective to steps 1 and 3; it builds the
cost grid and interpolant, as run_drive does.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import casadi as ca
import numpy as np
import opentop as top
from openap.aero import fpm, ft, kts

import pandas as pd

from src.config import DEFAULT_DATA_PATH
from src.data import FlightData
from src.flights import load_flight_index
from src.models.grid.contrail_grid import real_flight_route
from src.optimization.opentop.track_cruise import TrackCruise

# The drive_opentop flight, run when no --flights list is given.
DEFAULT_FLIGHT = dict(flight_id="02a1b3_00400_0", begin="2022-02-20 00:00",
                      end="2022-02-20 01:34", actype=None, m0=0.8)
SIGMA = 2
N_POINTS = 200            # random points for step 1
CONTRAIL_WEIGHT = 5.0     # fuel (kg) + w * grid cost, step 3 with --contrail

failures: list[str] = []


def check(name: str, value: float, limit: float, unit: str = "") -> None:
    ok = bool(np.isfinite(value) and value <= limit)
    if not ok:
        failures.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {value:11.3e} {unit:<4} (limit {limit:g})")


def build(cls, setup: dict, lat, lon, tolerance_m: float | None = None):
    opt = cls(setup["actype"], setup["origin"], setup["destination"], m0=setup["m0"],
              mach_value=setup["mach"], use_synonym=True)
    if tolerance_m is None:
        opt.follow_track(lat, lon)
    else:
        opt.follow_track(lat, lon, tolerance_m=tolerance_m)
    opt.fix_mach_number()
    opt.allow_cruise_descent()
    return opt


def load(meta: dict, with_contrail: bool) -> dict:
    fd = FlightData(start=str(meta["begin"]), stop=str(meta["end"]),
                    base_id=meta["flight_id"], path=str(DEFAULT_DATA_PATH),
                    aircraft_type=meta["actype"])
    # The wind is needed for the TAS-corrected Mach, as in run_drive.
    fd.attach_meteo("era5", buffer_hours=2.0, wind=True)
    avg_alt_m = float(fd.flight.data.altitude.mean()) * ft
    mach = fd.compute_cruise_mach()
    route, _ = real_flight_route(fd.flight, mach=mach, avg_alt_m=avg_alt_m,
                                 roc=500, alt_step_ft=500)
    setup = dict(
        actype=fd.aircraft_type, mach=mach, m0=meta["m0"],
        origin=(fd.initial_latitude, fd.initial_longitude),
        destination=(fd.final_latitude, fd.final_longitude),
        lat=np.array([p[0] for p in route]), lon=np.array([p[1] for p in route]),
        interpolant=None,
    )
    if with_contrail:
        fd.cost_grid(SIGMA)
        cache = Path(tempfile.gettempdir()) / f"opentop_interp_{meta['flight_id']}_sigma{SIGMA:g}.casadi"
        setup["interpolant"] = fd.interpolant(SIGMA, cache_path=str(cache))
    print(f"flight {meta['flight_id']}  {setup['actype']}  m0={meta['m0']:g}  "
          f"mach={mach:.3f}  track {len(route)} points")
    return setup


def distance_flown(df) -> float:
    """Ground distance flown, m: the integral of v·cos(gamma) over time."""
    t = df.ts.to_numpy()
    v = df.tas.to_numpy() * kts
    vs = df.vertical_rate.to_numpy() * fpm
    return float(np.trapezoid(v * np.cos(np.arctan2(vs, v)), t))


def spline_arc_length(opt, n: int = 200_000) -> float:
    """True arc length of the track spline, by dense sampling."""
    x_ref, y_ref, length = opt.track_ref
    s = np.linspace(0.0, length, n)
    x = np.asarray(x_ref(s)).ravel()
    y = np.asarray(y_ref(s)).ravel()
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


# ── step 1 ───────────────────────────────────────────────────────────────────

def step_model(setup: dict) -> None:
    print("\n== 1. MODEL: TrackCruise vs top.Cruise at the same virtual point ==")
    tc = build(TrackCruise, setup, setup["lat"], setup["lon"])
    cr = build(top.Cruise, setup, setup["lat"], setup["lon"])
    tc._build_track()
    kw = {}
    if setup["interpolant"] is not None:
        kw = dict(interpolant=setup["interpolant"], n_dim=4, time_dependent=True)
    tc.init_model("fuel", **kw)
    cr.init_model("fuel", **kw)

    x_ref, y_ref, length = tc.track_ref
    rng = np.random.default_rng(0)
    h_lo, h_hi = 30_000 * ft, 39_000 * ft
    err = dict(hmt=0.0, gv=0.0, fuel=0.0, grid=0.0)
    dl_ds = []
    eps = 1.0   # m, central difference step on s
    for _ in range(N_POINTS):
        s = rng.uniform(eps, length - eps)
        x = ca.DM([s, rng.uniform(h_lo, h_hi),
                   rng.uniform(cr.mass_min, cr.mass_init), rng.uniform(0, 3 * 3600)])
        vs = rng.uniform(-500, 500) * fpm
        # mach_value is set, so TrackCruise's u is [vs]: Mach is a model constant.
        u = ca.DM([vs]) if tc._mach_fixed else ca.DM([setup["mach"], vs])
        dt = ca.DM(rng.uniform(10, 300))
        xv, uv = (ca.DM(v) for v in tc._virtual(x, u))

        f_tc, L_tc = tc.func_dynamics(x, u, dt)
        f_cr, L_cr = cr.func_dynamics(xv, uv, dt)
        f_tc, f_cr = f_tc.full().ravel(), f_cr.full().ravel()

        # h, m, ts rates: identical model, so identical numbers.
        err["hmt"] = max(err["hmt"], np.max(np.abs(f_tc[1:] - f_cr[2:]) /
                                            np.maximum(np.abs(f_cr[2:]), 1e-12)))
        # Ground velocity: top.Cruise's (dx, dy) must equal r'(s) * ds/dt, with
        # r'(s) taken by finite difference, independent of TrackCruise's jacobian.
        r_prime = np.array([float(x_ref(s + eps) - x_ref(s - eps)),
                            float(y_ref(s + eps) - y_ref(s - eps))]) / (2 * eps)
        dl_ds.append(np.hypot(*r_prime))
        gv = r_prime * f_tc[0]
        err["gv"] = max(err["gv"], np.hypot(*(gv - f_cr[:2])) / np.hypot(*f_cr[:2]))
        err["fuel"] = max(err["fuel"], abs(float(L_tc - L_cr)) / abs(float(L_cr)))

        if kw:
            g_tc = float(tc.func_grid_cost(x, u, dt))
            g_cr = float(cr.func_grid_cost(xv, uv, dt))
            err["grid"] = max(err["grid"], abs(g_tc - g_cr) / max(abs(g_cr), 1e-12))

    dl_ds = np.array(dl_ds)
    print(f"  |r'(s)| over {N_POINTS} points: [{dl_ds.min():.5f}, {dl_ds.max():.5f}]  "
          f"(1 = arc-length parameterisation; the size of the fixed ds/dt bug)")
    check("h, m, ts rates  (max rel. error)", err["hmt"], 1e-10)
    check("ground velocity vs r'(s)*ds/dt  (max rel. error)", err["gv"], 1e-5)
    check("fuel objective L  (max rel. error)", err["fuel"], 1e-10)
    if kw:
        check("grid-cost objective L  (max rel. error)", err["grid"], 1e-10)


# ── step 2 ───────────────────────────────────────────────────────────────────

def step_nesting(setup: dict) -> None:
    print("\n== 2. REAL TRACK: fuel only ==")
    tc = build(TrackCruise, setup, setup["lat"], setup["lon"])
    df = tc.trajectory(objective="fuel")
    obj_t = float(tc.objective_value)
    print(f"  TrackCruise: fuel objective {obj_t:.2f} kg  success={tc.success}  "
          f"iters={tc.stats.get('iter_count')}  time {df.ts.iloc[-1] / 60:.1f} min")
    check("TrackCruise converged  (0 = yes)", float(not tc.success), 0)

    # Independent re-checks of the output DataFrame.
    t = df.ts.to_numpy()
    flown = distance_flown(df)
    arc = spline_arc_length(tc)
    print(f"  distance flown {flown / 1000:.2f} km   spline arc length {arc / 1000:.2f} km   "
          f"chord length {tc.track_ref[2] / 1000:.2f} km")
    check("distance flown vs spline arc length  (rel.)", abs(flown - arc) / arc, 2e-3)
    burnt = df.mass.iloc[0] - df.mass.iloc[-1]
    ff = df.fuelflow.to_numpy()    # numeric openap.FuelFlow, not the CasADi model
    burnt_ff = np.trapezoid(ff, t)
    print(f"  fuel: mass drop {burnt:.2f} kg   numeric fuel-flow integral {burnt_ff:.2f} kg")
    check("mass drop vs numeric fuel-flow integral  (rel.)", abs(burnt - burnt_ff) / burnt, 5e-3)

    # top.Cruise from TrackCruise's optimum. Not a superset (module docstring): it
    # snakes between nodes pinned on the spline, so it may burn more, but only
    # in proportion to the extra distance flown. Anything else is a model gap.
    for tol in (10_000.0, 1_000.0):
        cr = build(top.Cruise, setup, setup["lat"], setup["lon"], tolerance_m=tol)
        dfc = cr.trajectory(objective="fuel", initial_guess=df, return_failed=True)
        obj_c = float(cr.objective_value) if cr.objective_value is not None else np.nan
        print(f"  top.Cruise tol {tol:>6.0f} m from TrackCruise's optimum: "
              f"{obj_c:.2f} kg  success={cr.success}  iters={cr.stats.get('iter_count')}")
        if cr.success:
            d_fuel = (obj_c - obj_t) / obj_t
            d_dist = (distance_flown(dfc) - flown) / flown
            print(f"    fuel {d_fuel:+.3e}  distance {d_dist:+.3e}  (rel. to TrackCruise)")
            check(f"tol {tol:.0f} m: |dfuel/fuel - ddist/dist|",
                  abs(d_fuel - d_dist), 1e-4)
        else:
            print(f"  [SKIP] tol {tol:.0f} m: top.Cruise did not converge, nothing to compare")


# ── step 3 ───────────────────────────────────────────────────────────────────

def straight_track(setup: dict, n: int = 20):
    """n points on the straight line, in the optimizer's projection, from origin to
    destination: no turns, so top.Cruise's turn-rate limit and piecewise-linear
    heading are not in the way of a tight tolerance."""
    probe = top.Cruise(setup["actype"], setup["origin"], setup["destination"],
                       m0=setup["m0"], use_synonym=True)
    x0, y0 = probe.proj(setup["origin"][1], setup["origin"][0])
    x1, y1 = probe.proj(setup["destination"][1], setup["destination"][0])
    lon, lat = probe.proj(np.linspace(x0, x1, n), np.linspace(y0, y1, n), inverse=True)
    return np.asarray(lat), np.asarray(lon)


def compare_solves(label: str, setup: dict, lat, lon, objective, fix_alt: bool) -> None:
    out = {}
    for name, cls, tol in (("TrackCruise", TrackCruise, None), ("top.Cruise", top.Cruise, 100.0)):
        opt = build(cls, setup, lat, lon, tolerance_m=tol)
        if fix_alt:
            opt.fix_cruise_altitude()
        obj = objective(opt) if callable(objective) else objective
        kw = {}
        if setup["interpolant"] is not None:
            kw = dict(interpolant=setup["interpolant"], n_dim=4, time_dependent=True)
        df = opt.trajectory(objective=obj, return_failed=True, **kw)
        out[name] = (df, float(opt.objective_value), opt.success)
        print(f"  {name:<12} objective {out[name][1]:12.3f}  success={opt.success}  "
              f"iters={opt.stats.get('iter_count')}  fuel {df.mass.iloc[0] - df.mass.iloc[-1]:.2f} kg  "
              f"time {df.ts.iloc[-1] / 60:.2f} min")
    (dt_, ot, st), (dc, oc_, sc) = out["TrackCruise"], out["top.Cruise"]
    if not (st and sc):
        failures.append(f"{label}: a solve did not converge")
        print(f"  [FAIL] {label}: a solve did not converge")
        return
    check(f"{label}: objective  (rel. diff.)", abs(ot - oc_) / abs(oc_), 1e-3)
    check(f"{label}: flight time  (rel. diff.)",
          abs(dt_.ts.iloc[-1] - dc.ts.iloc[-1]) / dc.ts.iloc[-1], 1e-3)
    # Altitude profiles on the same normalised time axis.
    grid = np.linspace(0, 1, 200)
    a_t = np.interp(grid, dt_.ts / dt_.ts.iloc[-1], dt_.altitude)
    a_c = np.interp(grid, dc.ts / dc.ts.iloc[-1], dc.altitude)
    check(f"{label}: altitude profile  (max abs. diff.)", float(np.max(np.abs(a_t - a_c))), 100, "ft")


def step_smooth(setup: dict) -> None:
    print("\n== 3. SMOOTH: straight track, top.Cruise at tolerance 100 m ==")
    lat, lon = straight_track(setup)
    print("  -- fuel, fixed altitude --")
    compare_solves("fuel, fixed alt", setup, lat, lon, "fuel", fix_alt=True)
    print("  -- fuel, free altitude --")
    compare_solves("fuel, free alt", setup, lat, lon, "fuel", fix_alt=False)
    if setup["interpolant"] is not None:
        print(f"  -- fuel + {CONTRAIL_WEIGHT:g} x contrail, free altitude --")

        def weighted(opt):
            def obj(x, u, dt, **kw):
                return (opt.obj_fuel(x, u, dt) + CONTRAIL_WEIGHT * opt.obj_grid_cost(
                    x, u, dt, interpolant=setup["interpolant"], n_dim=4, time_dependent=True))
            return obj
        compare_solves("fuel + contrail", setup, lat, lon, weighted, fix_alt=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--contrail", action="store_true",
                   help="also check the grid-cost objective (loads ERA5 meteo)")
    p.add_argument("--steps", default="123", help="steps to run, e.g. 13 (default 123)")
    p.add_argument("--flights", type=Path, default=None,
                   help="flight-id list (one per line), e.g. "
                        "3_Results/flight_selection/selection/flights.txt; actype and m0 "
                        "are read from the manifest.csv beside it")
    p.add_argument("--limit", type=int, default=None, help="only the first N flights of --flights")
    args = p.parse_args()

    if args.flights is None:
        metas = [DEFAULT_FLIGHT]
    else:
        ids = [line.strip() for line in args.flights.read_text().splitlines() if line.strip()]
        ids = ids[:args.limit] if args.limit else ids
        manifest = pd.read_csv(args.flights.parent / "manifest.csv").set_index("flight_id")
        idx = load_flight_index(DEFAULT_DATA_PATH).set_index("flight_id")
        metas = [dict(flight_id=f, begin=idx.at[f, "begin"], end=idx.at[f, "end"],
                      actype=manifest.at[f, "actype"], m0=float(manifest.at[f, "m0"]))
                 for f in ids]

    summary = []
    for i, meta in enumerate(metas, 1):
        print(f"\n######## [{i}/{len(metas)}] {meta['flight_id']} ########")
        failures.clear()
        try:
            setup = load(meta, args.contrail)
            if "1" in args.steps:
                step_model(setup)
            if "2" in args.steps:
                step_nesting(setup)
            if "3" in args.steps:
                step_smooth(setup)
        except Exception as e:  # noqa: BLE001 — one flight must not stop the list
            failures.append(f"error: {type(e).__name__}: {e}")
            print(f"  [FAIL] {failures[-1]}")
        summary.append((meta["flight_id"], meta["actype"], list(failures)))
        print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")

    if len(metas) > 1:
        print("\n======== summary ========")
        for flight_id, actype, failed in summary:
            print(f"  {flight_id}  {str(actype):<5} "
                  f"{'PASS' if not failed else 'FAIL: ' + '; '.join(failed)}")
        print(f"  {sum(not f for _, _, f in summary)}/{len(summary)} flights passed")
    raise SystemExit(1 if any(f for _, _, f in summary) else 0)


if __name__ == "__main__":
    main()
