"""
mass_profile_error.py — what does the fixed mass profile cost the graph-2D solver?

The claim this supports
-----------------------
``model_2D`` does not carry mass as a search state. It precomputes ONE mass profile
before the search (``_compute_mass_profile``) and evaluates every edge at that mass,
whatever altitude the path actually takes. Carrying mass properly would mean another
index on a graph that already OOMs at 26 GB, so the paper has to say why dropping it
is admissible — with a number, not an assertion.

This script measures it, three ways, weakest evidence to strongest:

  1. MASS ERROR     profile mass vs the mass the optimal path really has at each
                    waypoint. The profile takes the *minimum* fuel flow over the
                    feasible levels at every step, so it is expected to over-estimate
                    the mass — the assumed aircraft is never lighter than the real
                    one. The measured sign is reported, not assumed.

  2. COST ERROR     per-edge relative error of the scalarised cost the DP actually
                    minimises. This is the quantity the old ``study_fixed_mass.py``
                    reported (mean / max relative error) for the constant-mass A*,
                    recomputed for today's distance-indexed profile.

  3. DECISION TEST  re-solve the DP with the true mass profile of the optimal path —
                    one step of a fixed-point iteration — and compare trajectories.
                    A cost error only matters if it changes the answer; this reports
                    whether it does, and the optimality gap of the original path
                    measured against the exact-mass cost.

Everything is rebuilt exactly as ``batch_graph2d`` builds it (same grid, same route,
same altitude window, read back from the stored DP context), so the numbers describe
the campaign and not a fresh setup.

Which mass the solver uses
--------------------------
Two code paths disagree by one waypoint: the DP edge tables use the mass at the edge's
START node (``_precompute_edge_costs``, ``mass = self._mass_at_dist[i]``) while
``_edge_cost`` — the graph/DP_MuO path — looks it up with ``searchsorted`` on the
edge midpoint and therefore lands on the END node. This script follows the DP tables,
since the trajectory under test is the DP's.

One flight, or a sample
-----------------------
A single flight gives an anecdote. The mass error accumulates with flight time, so a
number from one medium-haul B738 does not transfer to a 4 500 km widebody, and the
paper needs a spread rather than a case. ``--sample N`` draws N flights from EACH haul
category (short / medium / long) among those that have a DP pickle, and reports the
median / p90 / worst over the sample. The sweep appends to its CSV as it goes and
skips flights already in it, so an interrupted run resumes where it stopped.

Usage
-----
    # one flight — figure and detailed print
    python 4_Post_Treatment/mass_profile_error.py --flight 3c6599_04056_0 --sigma 2

    # a stratified sample — the table the paper quotes (quote --seed with it)
    python 4_Post_Treatment/mass_profile_error.py --sample 6 --sigma 2 --seed 0

    # an explicit set, or a list from a file
    python 4_Post_Treatment/mass_profile_error.py --flights id1,id2,id3
    python 4_Post_Treatment/mass_profile_error.py --flights-file flights.txt

    # errors only, no decision test — roughly twice as fast
    python 4_Post_Treatment/mass_profile_error.py --sample 6 --no-refeed

Every flight must already have a DP pickle: its context supplies mass, Mach and the
altitude window, so the rebuild matches the campaign.

Budget it: about one minute per (flight, coef) with the decision test on, so the
default three coefs are ~3 min per flight. ``--sample 6`` is 18 flights, roughly an
hour. Run it in the background and let the resume handle interruptions.

Outputs (--out-dir, default 4_Post_Treatment/DP_mass_error/):
    mass_profile_error_<flight>_sigma<s>.csv    single flight: one row per coef
    mass_profile_edges_<flight>_sigma<s>.csv    single flight: per-edge (--edge-csv)
    mass_profile_error_<flight>_sigma<s>.png    the figure (--figures in a sweep)
    mass_profile_sweep_sigma<s>.csv             sweep: one row per (flight, coef)
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import results_root  # noqa: E402
from src.data.flights import ensure_flight_slice, load_flight_index  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_GRID_PATH,
)
from src.config import ALT_STEP_FT, MAX_ROC_FPM  # noqa: E402
from src.contrails.interpolant import (  # noqa: E402
    load_global_interpolant, shift_interpolant)
from src.data.traffic import FlightData  # noqa: E402
from src.data.meteo import GridWeather  # noqa: E402
from src.optimization.environment import ContrailEnvironmentFromGrid  # noqa: E402
from src.data.route import real_flight_route  # noqa: E402
from src.optimization.graph_based.model_2D import DpOptimizer  # noqa: E402
from src.postprocess.evaluation import evaluate_path  # noqa: E402
from openap import aero  # noqa: E402
from openap.aero import ft, kts  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 — best effort
        pass

BENCH = results_root()
DEFAULT_OUT_DIR = PROJECT_ROOT / "4_Post_Treatment" / "DP_mass_error"
TMP = Path(tempfile.gettempdir())

# Haul categories for the stratified sample, on the great-circle distance between a
# flight's first and last point (km).
HAUL_EDGES_KM = (1500, 3500)

# Two entities, one colour each, used with the same meaning in every panel:
# the approximation the solver ran with, and the exact quantity it approximates.
C_PROF, C_TRUE, C_INK = "#2a78d6", "#c0392b", "#3a3a3a"

# One fixed schema for every row written. The sweep APPENDS to its CSV, and a row with
# a different set of keys would be written under the existing header in its own order —
# silently shifting values into the wrong columns. Rows are reindexed onto this list, so
# a flight that failed, or one whose decision test came back infeasible, still lines up.
SWEEP_COLUMNS = [
    "flight_id", "actype", "route_km", "cruise_alt_ft", "m0", "mass_init_kg",
    "n_waypoints", "coef", "n_edges",
    "fuel_solver_kg", "fuel_true_kg", "fuel_err_kg", "fuel_err_pct", "fuel_check_kg",
    "contrail",
    "mass_err_max_kg", "mass_err_max_pct", "mass_err_mean_kg", "mass_err_mean_pct",
    "mass_err_final_kg", "mass_err_final_pct", "mass_profile_heavier",
    "cost_err_mean_pct", "cost_err_max_pct", "alt_changes_ft",
    "refeed_status", "path_identical", "alt_diff_nodes", "alt_diff_max_ft",
    "refeed_fuel_kg", "refeed_contrail", "refeed_dfuel_kg", "refeed_dcontrail_pct",
    "optimality_gap_pct",
    "error",
]


def as_table(rows: list) -> pd.DataFrame:
    """Rows onto the fixed schema, refusing to drop anything it does not know about."""
    df = pd.DataFrame(rows)
    unknown = [c for c in df.columns if c not in SWEEP_COLUMNS]
    if unknown:
        raise RuntimeError(f"add {unknown} to SWEEP_COLUMNS — appending them as-is "
                           f"would misalign the CSV")
    return df.reindex(columns=SWEEP_COLUMNS)


# ── the rebuild ──────────────────────────────────────────────────────────────
def build_environment(flight_id: str, sigma: float, context: dict,
                      data_path: Path, grid_path: Path, interp: str,
                      grid: "GridWeather | None" = None, interpolant=None):
    """Rebuild one flight's 2D environment exactly as ``batch_graph2d`` does.

    Same grid, same route, same altitude window read back from the saved context —
    otherwise the numbers measured here would not describe the campaign.

    ``grid`` lets a caller that rebuilds MANY flights load the 293 MB meteo grid once
    and hand the same object to every call; left at None each call loads its own, which
    is what a single-flight run wants.

    ``interpolant`` short-circuits the per-flight bspline fit (164 s median) with one
    the caller already has — e.g. the global interpolant shifted to this flight's
    departure. It must take [lon, lat, h, ts_elapsed] like the per-flight one. Left at
    None the fit runs as the campaign ran it.
    """
    idx = load_flight_index(data_path)
    match = idx[idx.flight_id == flight_id]
    if match.empty:
        raise SystemExit(f"flight {flight_id} not in {data_path.name}")
    meta = dict(match.iloc[0])
    actype = context["actype"]

    # Cache names kept from the former dp_saturation.py so existing slices and
    # interpolant fits in the temp dir are still hits.
    cache = TMP / "_dp_saturation_slices"
    cache.mkdir(parents=True, exist_ok=True)
    flight_slice = ensure_flight_slice(
        pd.read_parquet(data_path, filters=[("flight_id", "==", flight_id)]),
        cache, flight_id)

    grid = grid if grid is not None else GridWeather(str(grid_path), sigmas=[sigma])
    fd = FlightData(start=str(meta["begin"]), stop=str(meta["end"]),
                    path=str(flight_slice), base_id=flight_id, aircraft_type=actype)
    fd.attach_meteo("grid", grid=grid, sigma=sigma, wind=True)
    df_cost = fd.cost_grid(sigma)
    # The shape is part of the cache key — see batch_graph2d. Use the SAME shape the
    # stored run was computed with, or the two cost fields differ by ~1%.
    if interpolant is None:
        interpolant = fd.interpolant(
            cache_path=TMP / f"dp_sat_{flight_id}_sigma{sigma:g}_{interp}.casadi",
            shape=interp)

    route, total_km = real_flight_route(fd.flight, mach=context["mach"],
                                        avg_alt_m=context["cruise_alt_ft"] * ft,
                                        roc=MAX_ROC_FPM, alt_step_ft=ALT_STEP_FT)
    env = ContrailEnvironmentFromGrid(
        interpolant=interpolant, grid_data=df_cost, route=(route, total_km),
        alt_step_ft=ALT_STEP_FT, min_alt_ft=context["min_alt_ft"],
        max_alt_ft=context["max_alt_ft"])
    return DpOptimizer(environment=env, cruise_alt_ft=context["cruise_alt_ft"],
                       mach=context["mach"], mass_init_kg=context["mass_init_kg"],
                       actype=actype, max_climb_rate_fpm=MAX_ROC_FPM,
                       max_descent_rate_fpm=MAX_ROC_FPM)


# ── the two mass profiles ────────────────────────────────────────────────────
def mass_along_path(opt: DpOptimizer, alt_ft: np.ndarray, ts_s: np.ndarray):
    """True mass at every node, and the fuel really burned on every edge.

    Deliberately the same integration as :func:`evaluate_path` — same averaged
    altitude, same vertical speed, same ``FuelFlow.enroute`` call — so the totals
    here and the campaign's stored ``fuel_kg`` are the same number, and any gap
    against the profile is the approximation and nothing else.
    """
    n = len(alt_ft)
    mass = np.empty(n)
    mass[0] = opt.mass_init_kg
    fuel = np.zeros(n - 1)
    for i in range(n - 1):
        dt = float(ts_s[i + 1] - ts_s[i])
        avg_alt = (alt_ft[i] + alt_ft[i + 1]) / 2.0
        vs = (alt_ft[i + 1] - alt_ft[i]) / (dt / 60.0) if dt > 0 else 0.0
        tas = aero.mach2tas(opt.mach, avg_alt * ft) / kts
        ff = opt.fuel_model.enroute(mass=mass[i], tas=tas, alt=avg_alt, vs=vs)
        fuel[i] = max(0.0, float(ff)) * dt
        mass[i + 1] = mass[i] - fuel[i]
    return mass, fuel


def edge_indices(opt: DpOptimizer, alt_ft: np.ndarray):
    """``(i, j, s)`` index triples that address this path in the edge tables."""
    env = opt.env
    j = np.clip(
        np.round((alt_ft - env.min_alt_ft) / env.alt_step_ft).astype(int),
        0, len(env.altitudes) - 1,
    )
    i = np.arange(len(alt_ft) - 1)
    s = (j[1:] - j[:-1]) + opt._max_steps
    return i, j[:-1], s


def table_costs(opt: DpOptimizer, alt_ft: np.ndarray):
    """The unweighted (fuel, contrail) the SOLVER used on this path.

    Read straight out of the precomputed tables rather than recomputed, so this is
    literally the number the DP minimised — no second implementation to disagree with.
    """
    i, j, s = edge_indices(opt, alt_ft)
    fuel = opt._fuel_edge_array[i, j, s]
    contrail = opt._contrail_edge_array[i, j, s]
    if not (np.isfinite(fuel).all() and np.isfinite(contrail).all()):
        raise RuntimeError("the solved path uses an edge the tables call infeasible — "
                           "the index mapping is wrong, not the physics")
    return fuel, contrail


def path_cost(opt: DpOptimizer, alt_ft: np.ndarray) -> float:
    """Scalarised cost of a path under the tables as they stand right now."""
    i, j, s = edge_indices(opt, alt_ft)
    return float(opt._edge_cost_array[i, j, s].sum())


# ── one scalarisation ────────────────────────────────────────────────────────
def analyse(opt: DpOptimizer, coef: float, mass0: np.ndarray, refeed: bool):
    """Solve at ``coef``, measure the approximation, optionally re-solve exactly."""
    opt._mass_at_dist = mass0.copy()
    opt._precompute_edge_costs()
    res = opt._solve_dp_point(coef)
    if res is None:
        raise SystemExit(f"no feasible DP solution at coef={coef:g}")

    alt_ft, dist_km, ts_s = res.alt_ft, res.dist_km, res.ts_s
    mass_true, fuel_true = mass_along_path(opt, alt_ft, ts_s)
    fuel_prof, contrail = table_costs(opt, alt_ft)

    # Mass the tables charged this path against: the start node of each edge.
    i, _, _ = edge_indices(opt, alt_ft)
    mass_prof_edge = mass0[i]

    w_f, w_c = 1.0 - coef, coef
    cost_prof = w_f * fuel_prof + w_c * contrail
    cost_true = w_f * fuel_true + w_c * contrail
    rel = np.abs(cost_prof - cost_true) / (cost_true + 1e-12)

    dmass = mass0 - mass_true                      # >0 when the profile is too heavy
    row = dict(
        coef=coef,
        n_edges=len(fuel_prof),
        fuel_solver_kg=float(fuel_prof.sum()),
        fuel_true_kg=float(fuel_true.sum()),
        fuel_err_kg=float(fuel_prof.sum() - fuel_true.sum()),
        fuel_err_pct=float((fuel_prof.sum() - fuel_true.sum()) / fuel_true.sum() * 100),
        contrail=float(contrail.sum()),
        mass_err_max_kg=float(np.max(np.abs(dmass))),
        mass_err_max_pct=float(np.max(np.abs(dmass) / mass_true) * 100),
        mass_err_mean_kg=float(np.mean(np.abs(dmass))),
        mass_err_mean_pct=float(np.mean(np.abs(dmass) / mass_true) * 100),
        mass_err_final_kg=float(dmass[-1]),
        mass_err_final_pct=float(dmass[-1] / mass_true[-1] * 100),
        mass_profile_heavier=bool((dmass >= 0).all()),
        cost_err_mean_pct=float(rel.mean() * 100),
        cost_err_max_pct=float(rel.max() * 100),
        alt_changes_ft=float(np.abs(np.diff(alt_ft)).sum()),
    )

    # Self-check: the per-edge integration must reproduce the campaign's own metric.
    metrics = evaluate_path(alt_ft, dist_km, ts_s, opt.mach, opt.mass_init_kg,
                            opt.env, opt.actype, retime=False)
    row["fuel_check_kg"] = float(metrics["total_fuel_kg"])

    detail = dict(coef=coef, dist_km=dist_km, alt_ft=alt_ft, ts_s=ts_s,
                  mass_prof=mass0, mass_true=mass_true, rel_err=rel,
                  mass_prof_edge=mass_prof_edge, fuel_prof=fuel_prof,
                  fuel_true=fuel_true, alt_refeed=None)

    if not refeed:
        return row, detail

    # ── decision test: does the exact mass move the trajectory? ───────────────
    opt._mass_at_dist = mass_true.copy()
    opt._precompute_edge_costs()
    res2 = opt._solve_dp_point(coef)
    if res2 is None:
        row["refeed_status"] = "infeasible"
        return row, detail
    # _solve_dp_point has just reweighted the tables at this coef, so the original
    # path can be priced against the exact-mass cost with no further setup.
    gap = (path_cost(opt, alt_ft) - float(res2.cost)) / abs(float(res2.cost) + 1e-12)
    dalt = res2.alt_ft - alt_ft
    row.update(
        refeed_status="ok",
        path_identical=bool(np.array_equal(res2.alt_ft, alt_ft)),
        alt_diff_max_ft=float(np.max(np.abs(dalt))),
        alt_diff_nodes=int((dalt != 0).sum()),
        refeed_fuel_kg=float(res2.fuel_kg),
        refeed_contrail=float(res2.contrail),
        refeed_dfuel_kg=float(res2.fuel_kg - metrics["total_fuel_kg"]),
        refeed_dcontrail_pct=float((res2.contrail - metrics["total_contrail"])
                                   / (metrics["total_contrail"] + 1e-12) * 100),
        optimality_gap_pct=float(gap * 100),
    )
    detail["alt_refeed"] = res2.alt_ft
    return row, detail


# ── figure ───────────────────────────────────────────────────────────────────
def draw(flight_id: str, sigma: float, actype: str, details: list, out_dir: Path) -> Path:
    """Three panels on one distance axis: the mass, the cost error it causes, and the
    trajectory it does (or does not) change."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Panels (a) and (c) show the WORST scalarisation, not the first: a figure that
    # justifies an approximation has to show where it hurts most, and panel (b) has
    # all of them anyway.
    ref = max(details, key=lambda d: d["rel_err"].mean())
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.05, 1.0, 1.0]))

    # (a) the two mass profiles
    ax = axes[0]
    ax.plot(ref["dist_km"], ref["mass_prof"] / 1000, "-", lw=2.0, color=C_PROF,
            label="Profile the solver used (mass fixed per waypoint)")
    ax.plot(ref["dist_km"], ref["mass_true"] / 1000, "--", lw=2.0, color=C_TRUE,
            label="True mass along the optimal trajectory")
    ax.set_ylabel("Mass [t]")
    gap_kg = ref["mass_prof"][-1] - ref["mass_true"][-1]
    ax.annotate(f"{gap_kg:+,.0f} kg at the last waypoint "
                f"({gap_kg / ref['mass_true'][-1] * 100:+.2f} %)",
                xy=(ref["dist_km"][-1], ref["mass_true"][-1] / 1000),
                xytext=(-8, 14), textcoords="offset points", ha="right",
                fontsize=8.5, color=C_INK)
    ax.set_title(f"The fixed mass profile and what it costs\n"
                 f"{flight_id} · {actype} · $\\sigma$={sigma:g} · "
                 f"m$_0$={ref['mass_prof'][0] / 1000:.1f} t · "
                 f"worst scalarisation, coef={ref['coef']:g}", fontsize=11)

    # (b) per-edge relative cost error, one line per scalarisation.
    #     A single hue, light to dark with coef — coef is a magnitude, not an identity.
    ax = axes[1]
    n = len(details)
    for k, d in enumerate(details):
        shade = 0.72 - 0.52 * (k / max(n - 1, 1))
        ax.plot(d["dist_km"][:-1], d["rel_err"] * 100, "-", lw=1.5, color=str(shade))
        ax.annotate(f"coef={d['coef']:g}",
                    xy=(d["dist_km"][-2], d["rel_err"][-1] * 100),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=str(max(shade - 0.1, 0.0)))
    ax.set_ylabel("Edge cost error [%]")

    # (c) does the approximation change the decision?
    ax = axes[2]
    ax.step(ref["dist_km"], ref["alt_ft"] / 1000, where="post", lw=2.0, color=C_PROF,
            label="DP with the fixed profile")
    if ref["alt_refeed"] is not None:
        ax.step(ref["dist_km"], ref["alt_refeed"] / 1000, where="post", lw=2.0,
                ls="--", color=C_TRUE, label="DP re-solved with the true mass")
        same = np.array_equal(ref["alt_refeed"], ref["alt_ft"])
        ax.text(0.01, 0.06,
                "identical trajectory" if same else
                f"differs at {int((ref['alt_refeed'] != ref['alt_ft']).sum())} waypoints, "
                f"max {np.max(np.abs(ref['alt_refeed'] - ref['alt_ft'])):.0f} ft",
                transform=ax.transAxes, fontsize=9, color=C_INK)
    ax.set_ylabel("Altitude [kft]")
    ax.set_xlabel("Along-track distance [km]")

    for ax in axes:
        ax.grid(alpha=.22, lw=.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8.5, loc="lower left", frameon=False)
    axes[2].legend(fontsize=8.5, loc="best", frameon=False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"mass_profile_error_{flight_id}_sigma{sigma:g}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path



def edge_table(fid: str, details: list) -> pd.DataFrame:
    """One row per (flight, coef, edge): the mass the solver charged, the mass the
    aircraft really had, and what that did to the edge.

    Indexed on the edge's START node, which is the mass ``_precompute_edge_costs``
    used — not the midpoint ``_edge_cost`` would have used.
    """
    out = []
    for d in details:
        n = len(d["fuel_prof"])
        mp, mt = d["mass_prof"][:n], d["mass_true"][:n]
        out.append(pd.DataFrame(dict(
            flight_id=fid, coef=d["coef"], edge=np.arange(n),
            dist_km=d["dist_km"][:n], alt_ft=d["alt_ft"][:n],
            mass_solver_kg=mp, mass_true_kg=mt,
            mass_err_kg=mp - mt, mass_err_pct=(mp - mt) / mt * 100,
            fuel_solver_kg=d["fuel_prof"], fuel_true_kg=d["fuel_true"],
            fuel_err_kg=d["fuel_prof"] - d["fuel_true"],
            rel_cost_err_pct=d["rel_err"] * 100,
        )))
    return pd.concat(out, ignore_index=True)


# ── one flight, end to end ───────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _ids_in(path: str) -> frozenset:
    """flight_ids held by one parquet, read once per path."""
    return frozenset(pd.read_parquet(path, columns=["flight_id"]).flight_id.unique())


def data_path_for(fid: str, args) -> Path:
    """The flight parquet, after checking it actually holds this flight."""
    path = Path(args.data_path)
    if fid not in _ids_in(str(path)):
        raise FileNotFoundError(f"flight {fid} not in {path.name}")
    return path


def run_flight(fid: str, sigma: float, coefs: list, args, grid=None,
               global_interp=None) -> tuple:
    """Rebuild one flight and measure every scalarisation. Returns (rows, details, ctx).

    ``grid`` is the shared meteo grid a sweep loads once; None means load one here.
    ``global_interp`` is ``(interpolant, meta)`` from the global fit: the per-flight
    bspline (164 s median) is then replaced by a constant time shift of it. The field
    is NOT identical to the campaign's per-flight fit — it carries a uniform residual
    of ~2e-4 in cost — so a run using it measures the mass effect on an equivalent
    field, not a reproduction of a given campaign trajectory.
    """
    name = f"{fid}_sigma{sigma:g}_dp.pkl.gz"
    dp_path = next(Path(args.dp_dir).glob(f"m_0*/{name}"), None)
    if dp_path is None:
        raise FileNotFoundError(f"no {name} in any m0 folder of {args.dp_dir}")
    with gzip.open(dp_path, "rb") as fh:
        payload = pickle.load(fh)
    if payload.get("error"):
        raise RuntimeError(f"the stored DP run failed — {payload['error']}")
    ctx = payload["context"]

    interpolant = None
    if global_interp is not None:
        g, meta = global_interp
        interpolant = shift_interpolant(g, ctx["begin"], meta)
    opt = build_environment(fid, sigma, ctx, data_path_for(fid, args),
                            Path(args.grid_path), args.interp, grid=grid,
                            interpolant=interpolant)
    mass0 = opt._mass_at_dist.copy()

    rows, details = [], []
    for coef in coefs:
        print(f"\n-- {fid} coef={coef:g} " + "-" * 40)
        row, detail = analyse(opt, coef, mass0, refeed=not args.no_refeed)
        # Flight identity first, so the sweep CSV reads as one table per flight.
        row = dict(flight_id=fid, actype=ctx["actype"], route_km=ctx.get("route_km"),
                   cruise_alt_ft=ctx.get("cruise_alt_ft"), m0=ctx.get("m0"),
                   mass_init_kg=ctx.get("mass_init_kg"),
                   n_waypoints=ctx.get("n_waypoints"), **row)
        rows.append(row)
        details.append(detail)
    return rows, details, ctx


# ── which flights ────────────────────────────────────────────────────────────
def flight_metadata(data_path: Path) -> pd.DataFrame:
    """typecode / distance_km / haul_category per flight, from the flight parquet.

    The DP context carries route_km too, but reading thousands of pickles to draw a
    sample of 30 is not a trade worth making; four columns of the parquet are."""
    df = pd.read_parquet(data_path, columns=["flight_id", "timestamp", "typecode",
                                             "latitude", "longitude"])
    ends = df.sort_values("timestamp").groupby("flight_id").agg(
        typecode=("typecode", "first"),
        lat0=("latitude", "first"), lon0=("longitude", "first"),
        lat1=("latitude", "last"), lon1=("longitude", "last"),
    ).reset_index()
    lat0, lon0, lat1, lon1 = (np.radians(ends[c]) for c in ("lat0", "lon0", "lat1", "lon1"))
    a = np.sin((lat1 - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin((lon1 - lon0) / 2) ** 2
    ends["distance_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    ends["haul_category"] = pd.cut(ends["distance_km"], [0, *HAUL_EDGES_KM, np.inf],
                                   labels=["short", "medium", "long"], right=False).astype(str)
    return ends[["flight_id", "typecode", "distance_km", "haul_category"]]


def stratified_sample(sigma: float, dp_dir: Path, per_stratum: int, seed: int,
                      data_path: Path) -> list:
    """``per_stratum`` flights from each haul category, among those with a DP pickle.

    Stratified rather than "the N biggest" or "the N first" on purpose: the mass error
    accumulates with flight time, so a sample drawn by convenience would either
    understate it (all short haul) or overstate it (all long haul). Existence of the
    pickle is checked on disk, not assumed from the parquet — the parquet holds the
    whole flight pool, not the flights that were actually run.
    """
    have = {p.name.split("_sigma")[0]
            for p in dp_dir.glob(f"m_0*/*_sigma{sigma:g}_dp.pkl.gz")}
    meta = flight_metadata(data_path)
    meta = meta[meta.flight_id.isin(have)]
    if meta.empty:
        raise SystemExit(f"no flight in {dp_dir} is in {Path(data_path).name} at sigma={sigma:g}")
    picked = (meta.groupby("haul_category", group_keys=False)
                  .apply(lambda g: g.sample(min(per_stratum, len(g)), random_state=seed))
                  .sort_values("distance_km"))
    print(f"stratified sample, {per_stratum} per haul category, seed={seed}:")
    print(picked[["flight_id", "typecode", "haul_category", "distance_km"]]
          .to_string(index=False, formatters={"distance_km": "{:,.0f}".format}))
    return picked.flight_id.tolist()


def resolve_flights(args, sigma: float) -> list:
    """The flight list, from whichever of the four selectors was given."""
    if args.flight:
        return [args.flight]
    if args.flights:
        return [f.strip() for f in args.flights.split(",") if f.strip()]
    if args.flights_file:
        return [ln.strip() for ln in Path(args.flights_file).read_text().split()
                if ln.strip()]
    if args.sample:
        return stratified_sample(sigma, Path(args.dp_dir), args.sample, args.seed,
                                 Path(args.data_path))
    raise SystemExit("give one of --flight / --flights / --flights-file / --sample")


# ── sweep ────────────────────────────────────────────────────────────────────
def sweep(ids: list, sigma: float, coefs: list, args, out_dir: Path) -> Path:
    """Every flight in ``ids``, one row per (flight, coef), appended as it goes.

    Appending rather than collecting is what makes it resumable: a sweep of 30 flights
    is over an hour, and a crash at flight 25 must not cost the first 24. A flight
    already in the CSV is skipped unless --overwrite.
    """
    csv_path = Path(args.sweep_csv) if args.sweep_csv else \
        out_dir / f"mass_profile_sweep_sigma{sigma:g}.csv"
    edge_path = csv_path.with_name(csv_path.stem + "_edges.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    done: set = set()
    if csv_path.exists() and not args.overwrite:
        done = set(pd.read_csv(csv_path).flight_id.astype(str))
        print(f"resuming {csv_path.name}: {len(done)} flights already there")
    elif csv_path.exists():
        csv_path.unlink()
        edge_path.unlink(missing_ok=True)

    todo = [f for f in ids if f not in done]
    print(f"\n{len(todo)} flights x {len(coefs)} coef(s) to run "
          f"({len(ids) - len(todo)} skipped)\n")

    # Loaded once and handed to every flight: it is 293 MB on disk and reloading it per
    # flight would dominate the sweep.
    grid = GridWeather(str(args.grid_path), sigmas=[sigma]) if todo else None
    gi = None
    if todo and args.global_interp:
        g, meta = load_global_interpolant(sigma=sigma, out_dir=Path(args.data_path).parent)
        gi = (g, meta)
        print(f"global interpolant: {meta['n_nodes']:,} nodes, t0={meta['t0']} "
              f"— per-flight bspline fit skipped")

    for n, fid in enumerate(todo, 1):
        print(f"\n{'='*78}\n[{n}/{len(todo)}] {fid}\n{'='*78}")
        try:
            rows, details, ctx = run_flight(fid, sigma, coefs, args, grid=grid,
                                            global_interp=gi)
        # SystemExit too, deliberately: build_environment raises it for a flight the
        # parquet does not hold, and it is a BaseException — an `except Exception` here
        # let it kill the whole sweep with no row written for the flight that caused it.
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            print(f"  SKIPPED {fid}: {type(exc).__name__}: {exc}")
            rows = [dict(flight_id=fid, error=f"{type(exc).__name__}: {exc}")]
            details = None
        df = as_table(rows)
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        if details and args.edge_csv:
            e = edge_table(fid, details)
            e.to_csv(edge_path, mode="a", header=not edge_path.exists(), index=False)
        if details and args.figures:
            draw(fid, sigma, ctx["actype"], details, out_dir)

    report_sweep(csv_path, coefs)
    if args.edge_csv and edge_path.exists():
        print(f"per-edge table: {edge_path}")
    return csv_path


def report_sweep(csv_path: Path, coefs: list) -> None:
    """What the sample says, across flights — the numbers a paper quotes."""
    df = pd.read_csv(csv_path)
    ok = df[df["error"].isna()] if "error" in df else df
    failed = df["flight_id"][df["error"].notna()].nunique() if "error" in df else 0
    if ok.empty:
        print("\nno flight produced a result")
        return

    # One line per flight: its worst scalarisation, since that is what has to be
    # defensible. Averaging over coefs would hide it.
    per_flight = ok.groupby("flight_id").agg(
        actype=("actype", "first"), route_km=("route_km", "first"),
        mass_err_max_pct=("mass_err_max_pct", "max"),
        fuel_err_pct=("fuel_err_pct", lambda s: s.abs().max()),
        cost_err_mean_pct=("cost_err_mean_pct", "max"),
        cost_err_max_pct=("cost_err_max_pct", "max"),
    )
    line = "=" * 78
    print(f"\n{line}\nMASS PROFILE APPROXIMATION — {len(per_flight)} flights"
          f"{f', {failed} failed' if failed else ''}\n{line}")
    print("worst scalarisation per flight, then the spread over flights:\n")
    stats = per_flight[["mass_err_max_pct", "fuel_err_pct",
                        "cost_err_mean_pct", "cost_err_max_pct"]].describe(
        percentiles=[.5, .9]).loc[["50%", "90%", "max"]]
    print(stats.to_string(float_format=lambda v: f"{v:,.4f}"))

    # Gated on real values, not on the column existing: the schema always carries these,
    # and --no-refeed leaves them NaN. Reporting "unchanged for 0 flights" when the test
    # never ran would read as a failure of the approximation.
    if ok["path_identical"].notna().any():
        tested = ok[ok["path_identical"].notna()]
        by_flight = tested.groupby("flight_id")["path_identical"]
        all_same = int(by_flight.all().sum())
        n_moved = int((~tested["path_identical"].astype(bool)).sum())
        print(f"\ndecision test: the trajectory is unchanged at every scalarisation for "
              f"{all_same} of {by_flight.ngroups} flights "
              f"({n_moved} of {len(tested)} solves moved).")
        # A non-finite gap is not a bug: the original path uses an edge the exact-mass
        # tables call infeasible, so its cost is inf. The trajectory sits ON the
        # feasibility boundary and a few kg of assumed mass decide which side. Counted
        # separately — averaging it in would poison every statistic with inf.
        gap = pd.to_numeric(tested["optimality_gap_pct"], errors="coerce")
        finite = np.isfinite(gap)
        if (~finite).any():
            n_fl = tested.loc[~finite, "flight_id"].nunique()
            print(f"  {(~finite).sum()} solves in {n_fl} flights sit on the feasibility "
                  f"boundary (the fixed-mass path has an edge the exact mass rejects) "
                  f"— reported apart, not in the gap below")
        if finite.any():
            print(f"  worst optimality gap of the approximated path: "
                  f"{gap[finite].abs().max():.4f} %")

    print(f"\nflights where the approximation is worst:")
    print(per_flight.sort_values("cost_err_max_pct", ascending=False).head(8)
          .to_string(float_format=lambda v: f"{v:,.3f}"))
    print(f"\nfull table: {csv_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_argument_group("which flights (give exactly one)")
    sel.add_argument("--flight", metavar="ID",
                     help="one flight_id — writes the figure and the detailed print")
    sel.add_argument("--flights", metavar="ID,ID,...",
                     help="an explicit list")
    sel.add_argument("--flights-file", metavar="PATH",
                     help="one flight_id per line")
    sel.add_argument("--sample", type=int, metavar="N",
                     help="N flights per haul category (short/medium/long), drawn "
                          "from those that have a DP pickle at this sigma")
    sel.add_argument("--seed", type=int, default=0,
                     help="sampling seed (default 0) — quote it in the paper, the "
                          "sample is not reproducible without it")

    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--coefs", default="0,0.5,0.9",
                   help="Scalarisations to test (default 0,0.5,0.9). coef=1 makes the "
                        "cost error identically zero — fuel drops out of the objective "
                        "— so it says nothing about the cost, only about the mass.")
    p.add_argument("--no-refeed", action="store_true",
                   help="Skip the decision test (the expensive part: one edge-table "
                        "rebuild per coef). Roughly halves the run time.")
    p.add_argument("--edge-csv", action="store_true",
                   help="Also write one row per (flight, coef, edge): the mass the "
                        "solver charged vs the true mass, and the fuel/cost error it "
                        "caused. In a sweep it goes to <sweep-csv>_edges.csv.")
    p.add_argument("--figures", action="store_true",
                   help="Sweep only: write a figure per flight as well (a single "
                        "--flight always gets one).")
    p.add_argument("--sweep-csv", default=None, metavar="PATH",
                   help="Sweep only: where the table goes "
                        "(default <out-dir>/mass_profile_sweep_sigma<s>.csv).")
    p.add_argument("--overwrite", action="store_true",
                   help="Sweep only: start the table from scratch instead of resuming.")
    p.add_argument("--dp-dir", default=str(BENCH / "DP"),
                   help="The DP root; each flight's pickle is looked up in its m0 folder.")
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--global-interp", action="store_true",
                   help="Use the global interpolant (0_Data/flight_data/global_interp_"
                        "sigma<s>.casadi) shifted per flight, instead of fitting a "
                        "bspline per flight. ~164 s/flight faster; the field carries a "
                        "uniform ~2e-4 residual against the campaign's own fit.")
    p.add_argument("--grid-path", default=str(DEFAULT_GRID_PATH))
    p.add_argument("--interp", choices=["bspline", "linear"], default="bspline",
                   help="Must match the shape the stored run used.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args()

    sigma = args.sigma
    coefs = [float(c) for c in args.coefs.split(",") if c.strip() != ""]
    out_dir = Path(args.out_dir)
    ids = resolve_flights(args, sigma)

    if len(ids) > 1:
        sweep(ids, sigma, coefs, args, out_dir)
        return

    # ── single flight: the detailed report ───────────────────────────────────
    fid = ids[0]
    gi = None
    if args.global_interp:
        g, meta = load_global_interpolant(sigma=sigma, out_dir=Path(args.data_path).parent)
        gi = (g, meta)
    try:
        rows, details, ctx = run_flight(fid, sigma, coefs, args, global_interp=gi)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"{fid}: {exc}")

    print(f"\n{fid} · {ctx['actype']} · sigma={sigma:g} · m0={ctx['m0']:g} "
          f"({ctx['mass_init_kg']:,.0f} kg) · window "
          f"{ctx['min_alt_ft']:,.0f}-{ctx['max_alt_ft']:,.0f} ft")
    df = as_table(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"mass_profile_error_{fid}_sigma{sigma:g}.csv"
    df.to_csv(csv_path, index=False)

    if args.edge_csv:
        edge_csv = out_dir / f"mass_profile_edges_{fid}_sigma{sigma:g}.csv"
        edge_table(fid, details).to_csv(edge_csv, index=False)
        print(f"per-edge detail: {edge_csv}")

    fig_path = draw(fid, sigma, ctx["actype"], details, out_dir)

    line = "=" * 78
    print(f"\n{line}\nMASS PROFILE APPROXIMATION — {fid}, sigma={sigma:g}\n{line}")
    cols = ["coef", "mass_err_max_kg", "mass_err_max_pct", "fuel_err_kg",
            "fuel_err_pct", "cost_err_mean_pct", "cost_err_max_pct"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    if "optimality_gap_pct" in df:
        print("\ndecision test (DP re-solved with the exact mass of its own path):")
        print(df[["coef", "path_identical", "alt_diff_nodes", "alt_diff_max_ft",
                  "refeed_dfuel_kg", "optimality_gap_pct"]]
              .to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # The sentence the paper needs, with the numbers filled in.
    sign = "above" if df["mass_profile_heavier"].all() else "on both sides of"
    print(f"\nIn plain terms: the profile stays {sign} the true mass, by at most "
          f"{df.mass_err_max_kg.abs().max():,.0f} kg "
          f"({df.mass_err_max_pct.abs().max():.2f} %). That shifts each edge's cost by "
          f"{df.cost_err_mean_pct.mean():.3f} % on average, "
          f"{df.cost_err_max_pct.max():.3f} % at worst", end="")
    if "path_identical" in df:
        n_same = int(df["path_identical"].fillna(False).sum())
        print(f", and leaves the chosen trajectory unchanged at {n_same} of {len(df)} "
              f"scalarisations. The path the approximation picks is within "
              f"{df.optimality_gap_pct.abs().max():.3f} % of the exact-mass optimum.")
    else:
        print(".")

    print(f"\ntable:  {csv_path}\nfigure: {fig_path}")


if __name__ == "__main__":
    main()
