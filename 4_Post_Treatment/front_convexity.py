"""front_convexity.py -- how much of each Pareto front a weighted sum cannot reach.

THE QUESTION. A weighted-sum scalarisation minimises w*fuel + (1-w)*contrail. Every
weight optimises to a VERTEX OF THE LOWER CONVEX HULL of the achievable set, so a
non-dominated point sitting strictly ABOVE that hull is returned by no weight, ever. It
is not a matter of using a finer grid of weights. Those points are UNSUPPORTED, and the
epsilon-constraint sweep is the one that can reach them. This script counts them, locates
them and measures how far above the hull they sit, for every solver of the campaign:
OpenTOP singlestart and multistart, the DP, and the DP_MuO.

A CALIBRATION COMES FREE. The DP campaign read here is the weighted-sum-only run, and the
DP solves each scalarisation exactly on its graph. Its front is therefore convex BY
CONSTRUCTION, and whatever this script reports as non-convex for the DP is the test's own
false-positive rate. Read that row first: it is the yardstick for every other row.

WHY BOTH SWEEPS GO INTO ONE FRONT. The two sweeps are not competing answers to be
compared against each other; they are two samplings of the SAME achievable set. The hull
has to be built from every point the method found, or the test measures the sampling
instead of the geometry -- on the weighted-sum points alone the front is convex BY
CONSTRUCTION and the answer would always be "convex". So the pipeline pools fuel_only +
weighted_sum + epsilon_constraint, keeps the provenance of every point, and reports which
sweep the unsupported ones came from. That last number is the check on the whole
exercise: an unsupported point attributed to the weighted sum would mean the test is
wrong, not the solver.

THE PIPELINE, per flight and per method:

  1. collect every CONVERGED (fuel, contrail) point of the three sweeps, with its sweep
  2. drop exact duplicates, then dominated points -- both counted, not just removed
  3. sort by fuel ascending
  4. rescale both axes to the unit square (see NORMALISATION)
  5. lower convex hull, Andrew's monotone chain
  6. gap = vertical distance from each point down to the hull, >= 0 by construction
  7. local curvature = second difference of contrail against fuel
  8. contiguous runs of unsupported points -> the non-convex REGIONS

NORMALISATION. Fuel runs to tens of thousands of kg while contrail is signed and often
near zero, so one absolute tolerance cannot serve both axes, let alone both a short-haul
and a long-haul flight. Each front is rescaled to [0, 1] on both axes before the test,
which makes --tol a single number meaning "this fraction of the front's own height". The
price is that a front with no trade-off in it -- contrail varying by a millionth of its
own level -- gets that dust stretched across the unit square, and every rounding error
becomes a full-height bulge. --min-span rejects those as degenerate rather than reporting
them as non-convex.

TWO READINGS OF THE SAME THING. The hull gap and the curvature sign agree most of the
time. Where they do not, trust the gap: the curvature divides by the fuel spacing, so two
front points a gram apart turn a negligible bulge into a huge second difference. The
curvature is here because it LOCALISES -- it names the point where the front bends the
wrong way -- not because it is the better test.

    python 4_Post_Treatment/front_convexity.py
    python 4_Post_Treatment/front_convexity.py --sigma 2 --tol 1e-4
    python 4_Post_Treatment/front_convexity.py --campaign Linear

Writes <out-dir>/convexity_opentop_flights_sigma<s>.csv (one row per flight x method) and
<out-dir>/convexity_opentop_regions_sigma<s>.csv (one row per non-convex region).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import SELECTION_DIR, results_root  # noqa: E402

# Imported, not reimplemented. The dominance mask and the hull chain are the two places
# this analysis could silently disagree with every figure in the campaign, so they stay
# defined once, in the module the figures already use.
from plots_comparison import (  # noqa: E402
    FS, GRID, INK, INK_2, RULE, SURFACE, Plot, _lower_hull, fit,
    pareto_mask, paper_size,
)
from plots_comparison import METHODS as PC_METHODS  # noqa: E402

BENCH = results_root()
# The selection manifest: every flight stage 2 ran on, both masses.
DEFAULT_FLIGHTS = SELECTION_DIR / "selection" / "manifest.csv"
SWEEPS = ("fuel_only", "weighted_sum", "epsilon_constraint")

# One fuel-only solve + 6 weighted-sum + 6 epsilon-constraint. A flight with fewer
# records had a solve return no trajectory at all, which is a different failure from a
# solve that ran and did not converge -- both disqualify it from "all points converged".
EXPECTED_SOLVES = 13

# numpy 2 renamed trapz -> trapezoid and removed the old name. Bound once here so the
# script runs on whichever numpy the environment has: the cluster pins are older than
# the workstation's.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

OPENTOP_SWEEPS = ("fuel_only", "weighted_sum", "epsilon_constraint")

# (key, label, folder under results_root(), glob, sweeps, expected solves).
#
# Spelled out per campaign rather than derived: the directory names follow no single
# rule -- the linear OpenTOP run appended its own suffix, the bspline one did not, and
# the graph solvers live in their own trees entirely. Deriving the path silently pointed
# the linear campaign at a folder that does not exist, which reads as "no pickles"
# rather than as a bug.
#
# ``expected`` is how many solves a complete flight has, for --only-converged. None
# means the count is not fixed: the DP_MuO returns its whole label set, so
# the front length is a property of the flight, not of the sweep grid.
#
# The folder is the method's root under 3_Results/contrail/ and the glob reaches into
# its m0 folders (m_060/, m_080/). The OpenTOP singlestart is start 0 of the multistart
# run (extract_singlestart.py -> OpenTOP/Singlestart_start0/); its glob carries no
# suffix so it never picks up an _ms or _fuel pickle.
SOURCES = {
    "Bspline": [
        ("opentop_single", "OpenTOP singlestart",
         "OpenTOP/Singlestart_start0", "m_0*/*_sigma{s:g}.pkl.gz",
         OPENTOP_SWEEPS, 13),
        ("opentop_ms", "OpenTOP multistart",
         "OpenTOP", "m_0*/*_sigma{s:g}_ms.pkl.gz",
         OPENTOP_SWEEPS, 13),
        ("dp", "Dynamic Programming",
         "DP", "m_0*/*_sigma{s:g}_dp.pkl.gz", OPENTOP_SWEEPS, 13),
        ("dp_muo", "DP_MuO",
         "DP_MuO", "m_0*/*_sigma{s:g}_dp_muo.pkl.gz",
         ("pareto_dp_muo",), None),
    ],
    "Linear": [
        ("opentop_single", "OpenTOP singlestart",
         "open_top_run/Linear/contrails_linear", "*_sigma{s:g}.pkl.gz",
         OPENTOP_SWEEPS, 13),
        ("opentop_ms", "OpenTOP multistart",
         "open_top_run/Linear/contrails_ms_linear", "*_sigma{s:g}_ms.pkl.gz",
         OPENTOP_SWEEPS, 13),
        ("dp", "Dynamic Programming",
         "dp_run/Linear", "*_sigma{s:g}_dp.pkl.gz", OPENTOP_SWEEPS, None),
        ("dp_muo", "DP_MuO",
         "dp_muo_run/Linear", "*_sigma{s:g}_dp_muo.pkl.gz",
         ("pareto_dp_muo",), None),
    ],
}


def front_points(payload: dict, sweep_names=OPENTOP_SWEEPS):
    """Every converged (fuel, contrail) of the three sweeps, with the sweep that made it.

    Non-converged points are dropped rather than kept as losers: opentop hands back
    IPOPT's last iterate on failure, so a failed solve is not a trajectory the method
    delivered and has no business setting the shape of its front.
    """
    fuel, cont, sweeps, objs = [], [], [], []
    for sweep in sweep_names:
        for t in payload["results"].get(sweep, ()):
            if not t.get("success"):
                continue
            f, c = t.get("fuel_kg"), t.get("contrail")
            if f is None or c is None:
                continue
            f, c = float(f), float(c)
            if np.isfinite(f) and np.isfinite(c):
                fuel.append(f)
                cont.append(c)
                sweeps.append(sweep)
                # The SOLVER's own contrail for this point, where the pickle keeps it.
                # Two different fields, because the two families store it differently:
                #   label_contrail  graph solvers -- the label the search actually
                #                   minimised, summed over edges
                #   objective       OpenTOP, and only on epsilon_constraint records,
                #                   where the objective IS the pure grid cost
                # Everything else has no second opinion on record and gets NaN.
                meta = t.get("meta") or {}
                o = meta.get("label_contrail")
                if o is None and sweep == "epsilon_constraint":
                    o = meta.get("objective")
                objs.append(float(o) if o is not None else np.nan)
    return (np.asarray(fuel, float), np.asarray(cont, float), sweeps,
            np.asarray(objs, float))


def report_noise(cont, sweeps, objs, cs: float) -> float:
    """How far the plotted contrail can drift from the one the solver minimised.

    THE PROBLEM THIS MEASURES. ``contrail`` in the pickle is a post-hoc
    ``nansum(traj["grid_cost"])`` over the returned trajectory, while the solver
    minimised ``Cruise.obj_grid_cost``, a quadrature of the SAME field on the
    collocation mesh. Two discretisations of one integral, so they disagree -- and the
    convexity test runs on the first while the theory ("a weighted sum returns a hull
    vertex") holds for the second.

    A constant offset per flight would be harmless: it slides the whole front up and
    changes no hull. What breaks the test is the offset VARYING between points of the
    same front, so that is what is returned -- the within-front spread of
    (reported - optimised), in units of the front's own contrail span, the same unit
    the hull gap is in.

    Measured on the epsilon-constraint points only, because there the stored objective
    IS the pure grid cost. For a weighted-sum record the objective is
    cw*grid + fw*fuel and the grid part cannot be separated without the quadrature
    fuel, which is not stored.

    Returns NaN when the front carries too few usable epsilon-constraint points to say.
    """
    d = [c - o for c, o in zip(cont, objs) if np.isfinite(o)]
    if len(d) < 2 or cs <= 0:
        return float("nan")
    return float((max(d) - min(d)) / cs)


def analyse(fuel, cont, sweeps, objs, tol: float, min_span: float):
    """(row, regions, n_front), or (None, reason, n_front) when the test is undefined.

    ``n_front`` is returned on EVERY path, including the failures. A flight whose front
    has two points has no hull and no verdict, but its front size is still a fact about
    the run and the only place it is available -- such a flight never reaches the csv.
    """
    n_raw = fuel.size
    if n_raw == 0:
        return None, "no converged point", 0

    # 2a. exact duplicates. Counted apart from dominance because they mean something
    # different: a duplicate is two sweeps landing on the same trajectory, which is the
    # expected overlap at the ends of the front, not a solver failing to improve.
    _, uniq = np.unique(np.column_stack([fuel, cont]), axis=0, return_index=True)
    uniq = np.sort(uniq)
    n_dup = n_raw - uniq.size
    fuel, cont = fuel[uniq], cont[uniq]
    sweeps = [sweeps[i] for i in uniq]
    objs = objs[uniq]

    # 2b. dominated points
    keep = pareto_mask(fuel, cont)
    n_dominated = int((~keep).sum())
    fuel, cont = fuel[keep], cont[keep]
    sweeps = [s for s, k in zip(sweeps, keep) if k]
    objs = objs[keep]

    # 3. sort by fuel ascending
    order = np.argsort(fuel, kind="stable")
    fuel, cont = fuel[order], cont[order]
    sweeps = [sweeps[i] for i in order]
    objs = objs[order]

    if fuel.size < 3:
        return None, "fewer than 3 front points", int(fuel.size)  # no interior point

    fs = float(fuel[-1] - fuel[0])
    cs = float(cont.max() - cont.min())
    if fs <= 0.0 or cs <= 0.0:
        return None, "one axis constant", int(fuel.size)
    scale_f = max(float(np.max(np.abs(fuel))), 1e-30)
    scale_c = max(float(np.max(np.abs(cont))), 1e-30)
    if fs / scale_f < min_span or cs / scale_c < min_span:
        return None, "front is a point in disguise", int(fuel.size)

    # 4. unit square
    x = (fuel - fuel[0]) / fs
    y = (cont - cont.min()) / cs

    # 5-6. hull, and the distance from the front down to it
    hull = _lower_hull(x, y)
    gap = np.maximum(y - np.interp(x, x[hull], y[hull]), 0.0)   # >= 0; clip float dust
    unsup = gap > tol

    # 7. local curvature: the second difference of contrail against fuel on the
    # normalised front. Convex means the slopes dc/df increase towards zero as fuel
    # grows, so a NEGATIVE second difference is the front bending the wrong way.
    curv = np.full(x.size, np.nan)
    dx, dy = np.diff(x), np.diff(y)
    slope = dy / np.where(dx > 0, dx, np.nan)
    curv[1:-1] = np.diff(slope) / ((x[2:] - x[:-2]) / 2.0)
    concave = np.nan_to_num(curv, nan=0.0) < -tol

    # 8. contiguous runs of unsupported points
    sweep_arr = np.asarray(sweeps)
    regions, i = [], 0
    while i < unsup.size:
        if not unsup[i]:
            i += 1
            continue
        j = i
        while j + 1 < unsup.size and unsup[j + 1]:
            j += 1
        sl = slice(i, j + 1)
        # Area between front and hull over the region, trapezoid on the normalised x.
        # The neighbours are included so a one-point region still has width: the bulge is
        # bounded by the supported points either side, and those are where it returns to
        # zero.
        lo, hi = max(i - 1, 0), min(j + 2, x.size)
        area = float(trapezoid(gap[lo:hi], x[lo:hi]))
        regions.append({
            "i_start": i, "i_end": j, "n_points": j - i + 1,
            "fuel_lo_kg": float(fuel[i]), "fuel_hi_kg": float(fuel[j]),
            # Where along the front, as a fraction of its fuel span: 0 is the
            # fuel-optimal end, 1 the contrail-optimal end.
            "pos_lo": float(x[i]), "pos_hi": float(x[j]),
            "max_gap": float(gap[sl].max()),
            "max_gap_contrail": float(gap[sl].max() * cs),
            "area_norm": area,
            "area_contrail": area * cs,
            "sweeps": "+".join(sorted(set(sweep_arr[sl]))),
        })
        i = j + 1

    counts = pd.Series(sweep_arr[unsup]).value_counts() if unsup.any() \
        else pd.Series(dtype=int)
    total_area = float(trapezoid(gap, x))

    row = {
        "n_raw": int(n_raw), "n_duplicate": int(n_dup), "n_dominated": n_dominated,
        "n_front": int(fuel.size), "n_hull": int(hull.size),
        "convex": not bool(unsup.any()),
        "n_unsupported": int(unsup.sum()),
        "frac_unsupported": float(unsup.mean()),
        "n_regions": len(regions),
        "max_gap": float(gap.max()),
        "max_gap_contrail": float(gap.max() * cs),
        "area_norm": total_area,
        "area_contrail": total_area * cs,
        "span_frac": float(sum(r["pos_hi"] - r["pos_lo"] for r in regions)),
        "pos_at_max_gap": float(x[int(np.argmax(gap))]),
        "fuel_at_max_gap_kg": float(fuel[int(np.argmax(gap))]),
        "n_concave": int(concave.sum()),
        "min_curvature": float(np.nanmin(curv)) if np.isfinite(curv).any() else np.nan,
        "n_unsup_weighted_sum": int(counts.get("weighted_sum", 0)),
        "n_unsup_epsilon": int(counts.get("epsilon_constraint", 0)),
        "n_unsup_fuel_only": int(counts.get("fuel_only", 0)),
        "fuel_span_kg": fs, "contrail_span": cs,
    }
    # The verdict is only worth as much as the axis it was measured on. A bulge smaller
    # than the drift between the plotted contrail and the optimised one is not evidence
    # of a non-convex front -- it is inside the noise of the coordinate.
    # NOTE this only ever measures the CONTRAIL axis, the only one a pickle keeps a
    # second opinion on. The DP_MuO's label_contrail equals its reported
    # contrail exactly, so its noise comes out 0 and every verdict passes -- true of the
    # vertical axis, silent about the fuel axis, which its own labels show drifting by up
    # to ~0.5%. Read a 0 here as "no measurable contrail drift", not as "no error".
    noise = report_noise(cont, sweeps, objs, cs)
    row["report_noise"] = noise
    row["verdict_reliable"] = bool(row["max_gap"] > noise) if np.isfinite(noise) else False
    return row, regions, int(fuel.size)


def solve_tally(payload: dict, sweep_names) -> tuple[int, int]:
    """(solves recorded, solves that converged) for one flight."""
    tot = ok = 0
    for sweep in sweep_names:
        for t in payload["results"].get(sweep, ()):
            tot += 1
            ok += bool(t.get("success"))
    return tot, ok


def load(folder: Path, pattern: str, selection, tol: float, min_span: float,
         only_converged: bool = False, sweep_names=OPENTOP_SWEEPS, expected=None):
    files = sorted(folder.glob(pattern))
    rows, regs = [], []
    # Collected BEFORE the hull test, and kept even when that test is undefined:
    # "all 13 solves converged" is a property of the run, not of whether the front it
    # produced happens to have enough distinct points to fit a hull to.
    tally = {"files": len(files), "stub": 0, "unselected": 0,
             "partial": 0, "undefined": {}, "converged_ids": [], "front_size": {}}
    for path in files:
        with gzip.open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload["error"]:
            tally["stub"] += 1
            continue
        ctx = payload["context"] or {}
        fid = ctx.get("flight_id")
        if selection is not None and fid not in selection:
            tally["unselected"] += 1
            continue
        n_solves, n_ok = solve_tally(payload, sweep_names)
        # With no fixed count the completeness test is only "nothing failed": the
        # DP_MuO's front length is a property of the flight.
        full = (n_ok == n_solves) and (expected is None or n_solves == expected)
        if full:
            tally["converged_ids"].append(fid)
        if only_converged and not full:
            tally["partial"] += 1
            continue
        row, extra, n_front = analyse(*front_points(payload, sweep_names),
                                      tol, min_span)
        tally["front_size"][fid] = n_front
        if row is None:
            tally["undefined"][extra] = tally["undefined"].get(extra, 0) + 1
            continue
        head = {"flight_id": fid, "sigma": float(ctx.get("sigma")),
                "m0": ctx.get("m0"), "actype": ctx.get("actype"),
                "n_solves": n_solves, "n_solves_ok": n_ok, "fully_converged": full}
        rows.append({**head, **row})
        for k, r in enumerate(extra):
            regs.append({**head, "region": k, **r})
    return pd.DataFrame(rows), pd.DataFrame(regs), tally


def pct(n, d):
    return f"{n / d:.1%}" if d else "n/a"


def report(per: dict, tol: float, labels) -> None:
    print("\n" + "=" * 78)
    print("CONVEXITY OF THE PARETO FRONTS")
    print("=" * 78)
    print("!! READ THIS BEFORE QUOTING ANY NUMBER BELOW " + "!" * 31)
    print("The test runs on the contrail value STORED in the pickle, a post-hoc nansum")
    print("over the returned trajectory. The solver minimised a quadrature of the same")
    print("field on its collocation mesh, and the two disagree per point. Where that")
    print("disagreement varies more within a front than the bulge being measured, the")
    print("verdict is noise. Two independent checks of it are printed per method:")
    print("  (a) weighted-sum points found above the hull -- theory says zero")
    print("  (b) fronts whose bulge exceeds the coordinate drift -- 'survive' below")
    print("!" * 78)

    print("A point is UNSUPPORTED when it is non-dominated but sits more than tol above")
    print("the lower convex hull of its own front. No weighted sum reaches it at any")
    print(f"weight; only the epsilon-constraint sweep can.   tol = {tol:g} of the")
    print("front's own height, after both axes are rescaled to [0, 1].\n")

    for key, label in labels:
        df = per[key]
        if df.empty:
            print(f"{label}: nothing measured\n")
            continue
        n = len(df)
        nc = df[~df.convex]
        print(f"{label}   ({n:,} flights measured)")
        print(f"  convex fronts                 {n - len(nc):>6,}  {pct(n - len(nc), n)}")
        print(f"  NON-convex fronts             {len(nc):>6,}  {pct(len(nc), n)}")
        if nc.empty:
            print()
            continue
        print(f"  unsupported points            {int(df.n_unsupported.sum()):>6,}  "
              f"of {int(df.n_front.sum()):,} front points "
              f"({pct(df.n_unsupported.sum(), df.n_front.sum())})")
        print(f"    from epsilon-constraint     {int(df.n_unsup_epsilon.sum()):>6,}")
        print(f"    from weighted sum           {int(df.n_unsup_weighted_sum.sum()):>6,}"
              "   <- CHECK (a): theory says 0")
        print("        A weighted sum cannot return a point above the hull, so each of")
        print("        these is either a solve that stopped at a local optimum or, far")
        print("        more likely here, the coordinate drift described at the top.")
        print(f"    from fuel-only              {int(df.n_unsup_fuel_only.sum()):>6,}")
        print("  among the non-convex fronts:")
        print(f"    unsupported per front       median {nc.n_unsupported.median():.0f}"
              f"   p90 {nc.n_unsupported.quantile(.9):.0f}"
              f"   max {nc.n_unsupported.max():.0f}")
        print(f"    separate regions            median {nc.n_regions.median():.0f}"
              f"   max {nc.n_regions.max():.0f}")
        # The gap is a fraction of the front's own contrail range: 0.05 means the front
        # bulges a twentieth of its own height above the hull.
        print(f"    hull gap, normalised        median {nc.max_gap.median():.4f}"
              f"   p90 {nc.max_gap.quantile(.9):.4f}   max {nc.max_gap.max():.4f}")
        print(f"    hull gap, contrail units    median {nc.max_gap_contrail.median():.3f}"
              f"   p90 {nc.max_gap_contrail.quantile(.9):.3f}"
              f"   max {nc.max_gap_contrail.max():.3f}")
        print(f"    where along the front       median {nc.pos_at_max_gap.median():.2f}"
              "   (0 = fuel-optimal end, 1 = contrail-optimal end)")
        print(f"    fuel span it covers         median {nc.span_frac.median():.1%}"
              f"   p90 {nc.span_frac.quantile(.9):.1%}")
        rel = nc[nc.verdict_reliable]
        print(f"  CHECK (b) non-convex verdicts  {len(rel):>6,}  of {len(nc):,} survive the")
        print("            coordinate-drift test (bulge larger than the within-front")
        print("            drift between plotted and optimised contrail) = "
              f"{pct(len(rel), len(nc))}")
        print("            drift, in units of the front's own span:  median "
              f"{df.report_noise.median():.3f}   p90 {df.report_noise.quantile(.9):.3f}")
        print(f"  curvature test disagrees      "
              f"{int(((df.n_concave > 0) != (~df.convex)).sum()):>6,}  flights "
              "(trust the hull gap; the second\n"
              "                                        difference divides by the fuel "
              "spacing)")
        print()

    a = per.get("opentop_single", pd.DataFrame())
    b = per.get("opentop_ms", pd.DataFrame())
    if not a.empty and not b.empty:
        j = a[["flight_id", "convex"]].merge(b[["flight_id", "convex"]],
                                             on="flight_id", suffixes=("_s", "_m"))
        both = int((~j.convex_s & ~j.convex_m).sum())
        only_s = int((~j.convex_s & j.convex_m).sum())
        only_m = int((j.convex_s & ~j.convex_m).sum())
        neither = int((j.convex_s & j.convex_m).sum())
        print(f"PAIRED, on the {len(j):,} flights both modes measured")
        print(f"  non-convex in both               {both:>6,}  {pct(both, len(j))}")
        print(f"  non-convex in singlestart only   {only_s:>6,}  {pct(only_s, len(j))}")
        print(f"  non-convex in multistart only    {only_m:>6,}  {pct(only_m, len(j))}")
        print(f"  convex in both                   {neither:>6,}  {pct(neither, len(j))}")
        print("  A flight non-convex in one mode and not the other is a statement about")
        print("  what that run FOUND, not about the flight: an unsupported point can only")
        print("  be seen once some sweep has landed on it.\n")

    for key, label in labels:
        df = per[key]
        if df.empty or df.convex.all():
            continue
        print(f"worst 10 by hull gap -- {label}")
        print(f"  {'flight_id':<20} {'actype':>6} {'front':>6} {'unsup':>6} {'reg':>4} "
              f"{'gap':>8} {'gap (contrail)':>15} {'pos':>5} {'span':>7}")
        for _, r in df.nlargest(10, "max_gap").iterrows():
            print(f"  {r.flight_id:<20} {str(r.actype):>6} {r.n_front:>6.0f} "
                  f"{r.n_unsupported:>6.0f} {r.n_regions:>4.0f} {r.max_gap:>8.4f} "
                  f"{r.max_gap_contrail:>15.4f} {r.pos_at_max_gap:>5.2f} "
                  f"{r.span_frac:>7.1%}")
        print()


def plot_unsupported_share(frames: dict, key: str, out_dir: Path,
                           population: str, n_population: int,
                           campaign: str) -> None:
    """How much of a solver's front no weighted sum can reach, one panel per sigma.

    Violin + box, styled like solve_time_distribution and in the same method colour, so
    the two figures read as one family. Two deliberate departures from Plot.violin:

    LINEAR, NOT log10. That helper takes the log because solve times span decades. This
    quantity is a share of a front, bounded in [0, 100], and a third of the flights sit
    at exactly 0 -- log10 is undefined on them and would silently drop them.

    THE DENSITY IS CLIPPED to [0, 100]. A kernel density fitted to a bounded quantity
    puts mass outside the bounds, and here it would draw flights with a negative share
    of their front non-convex, which is not a thing. Clipping the drawn path is the
    honest rendering; the atom at zero is stated in text instead, because no violin can
    show a point mass -- a KDE spreads it into a bulge and understates it.

    The panels SHARE the y axis so the sigmas can be compared by eye, which is the only
    reason to put them side by side rather than in three files.
    """
    colour = next(c for k, _, c, _ in PC_METHODS if k == key)
    label = next(n for k, n, _, _ in PC_METHODS if k == key)
    sigmas = sorted(frames)

    fig, axes = plt.subplots(1, len(sigmas), figsize=paper_size(2.5, 3, 1.6, span="full"),
                             sharey=True, facecolor=SURFACE, layout="constrained",
                             squeeze=False)

    for ax, sg in zip(axes[0], sigmas):
        Plot.style(ax)
        ax.grid(axis="x", visible=False)
        pct = frames[sg].frac_unsupported.to_numpy() * 100.0
        zero = int((pct == 0).sum())

        if pct.size:
            body = ax.violinplot([pct], positions=[0], widths=0.78,
                                 showextrema=False)["bodies"][0]
            v = body.get_paths()[0].vertices
            v[:, 1] = np.clip(v[:, 1], 0.0, 100.0)
            body.set_facecolor(colour)
            body.set_alpha(0.30)
            body.set_edgecolor(colour)
            body.set_linewidth(1.4)
            body.set_zorder(2)

            box = ax.boxplot([pct], positions=[0], widths=0.13, showfliers=False,
                             patch_artist=True, whis=(5, 95), zorder=3)
            box["boxes"][0].set(facecolor=SURFACE, edgecolor=INK_2, linewidth=1.0)
            for line in box["whiskers"] + box["caps"]:
                line.set(color=INK_2, linewidth=1.0)
            box["medians"][0].set(color=INK, linewidth=1.8)

            med = float(np.median(pct))
            ax.text(0.10, med, f"{med:.0f}%", ha="left", va="center",
                    fontsize=FS["annot"], color=INK_2, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0, alpha=0.85))
            # Placed at mid-height rather than on the baseline: the KDE is widest
            # near zero, so a note sitting at y=0 was printed over the violin it
            # describes. x stays at the left margin, clear of the body either way.
            # The point mass at zero, said in words: a violin cannot render an atom, and
            # the KDE turns it into a low bulge reading as "a few flights near zero"
            # rather than "a third of them at exactly zero".
            ax.annotate(f"{zero:,} ({zero / len(pct):.0%})\nconvex", (0.03, 0.50),
                        xycoords="axes fraction", ha="left", va="center",
                        fontsize=FS["annot"], color=INK_2, zorder=5)

        # sigma UNDER the panel, matching front_size_distribution: with the title
        # gone there is nothing above the axes, and the reader meets the panel's
        # identity next to the count it applies to.
        # Only when there is more than one panel to tell apart. On a single-sigma
        # figure the label names something nothing contrasts with, and the sigma is
        # already in the file name.
        if len(sigmas) > 1:
            ax.set_xlabel(f"$\\sigma$ = {sg:g}", fontsize=FS["title"],
                          color=INK, fontweight="bold")
        ax.set_xlim(-0.6, 0.9)
        ax.set_xticks([0])
        ax.set_xticklabels([f"n = {len(pct):,}"], fontsize=FS["tick"], color=INK_2)

    axes[0][0].set_ylim(-3, 103)
    axes[0][0].set_yticks([0, 25, 50, 75, 100])
    axes[0][0].set_ylabel("non-convex Pareto (%)", fontsize=FS["label"], color=INK_2)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(f"{s:g}" for s in sigmas)
    # The CAMPAIGN is in the name. Without it the linear run writes the bspline
    # file's name into the bspline file's folder and the two are indistinguishable
    # afterwards -- which is exactly what happened the first time this was run.
    stem = f"nonconvex_pareto_{key}_{campaign.lower()}_sigma{tag}"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=600, bbox_inches="tight",
                    facecolor=SURFACE)
        print(f"wrote {stem}.{ext}")
    plt.close(fig)


def summary(per: dict) -> None:
    """The cross-campaign table: flights with at least one point off the hull.

    Two counts per cell, and both belong on the page. "flagged" is every front with a
    point above its own hull. "beyond the error" keeps only the fronts whose bulge is
    larger than that same front's drift between the plotted contrail and the optimised
    one, so it is the count that survives the coordinate problem. The truth for each
    cell is between them.
    """
    print("\n" + "=" * 78)
    print("AT LEAST ONE NON-CONVEX POINT ON THE FRONT")
    print("=" * 78)
    print(f"{'campaign':<10} {'sigma':>5}  {'mode':<14} {'measured':>9} {'flagged':>9} "
          f"{'%':>7} {'beyond err':>11} {'%':>7}")
    for (camp, sigma, key), df in per.items():
        label = next(l for k, l, *_ in SOURCES[camp] if k == key).replace("OpenTOP ", "")
        if df.empty:
            print(f"{camp:<10} {sigma:>5g}  {label:<14} {'-':>9} {'-':>9} {'-':>7} "
                  f"{'-':>11} {'-':>7}")
            continue
        nc = df[~df.convex]
        rel = nc[nc.verdict_reliable]
        print(f"{camp:<10} {sigma:>5g}  {label:<14} {len(df):>9,} {len(nc):>9,} "
              f"{len(nc) / len(df):>7.1%} {len(rel):>11,} {len(rel) / len(df):>7.1%}")
    print("\n'measured' excludes error stubs, flights outside the selection, and fronts")
    print("with fewer than 3 points or no spread -- a hull needs an interior point to bulge above.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--campaign", nargs="+", default=list(SOURCES),
                   choices=list(SOURCES),
                   help="subdirectory of open_top_run/ to read (Bspline or Linear). "
                        "Named explicitly because the pickle name does not encode the "
                        "interpolant, so mixing the two would be silent")
    p.add_argument("--sigma", nargs="+", type=float, default=[2.0], metavar="S",
                   help="one or more sigmas. Several means the whole analysis runs per "
                        "sigma and the plot gets one panel each; the file names carry "
                        "every sigma that went in, so a 3-sigma run cannot overwrite a "
                        "1-sigma one")
    p.add_argument("--flights-file", type=Path, default=DEFAULT_FLIGHTS,
                   help="restrict to these flight ids (one per line, or a csv with a flight_id "
                        "column); pass 'none' for every pickle")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="how far above the hull a point must sit to count, on a front "
                        "rescaled to the unit square")
    p.add_argument("--min-span", type=float, default=1e-6,
                   help="a front whose fuel or contrail varies by less than this "
                        "fraction of its own level is degenerate, not non-convex")
    p.add_argument("--only-converged", action="store_true",
                   help="keep only flights where the expected number of solves is "
                        "present AND every one converged. Off by default: a front is "
                        "built from the points that DID converge, so dropping the whole "
                        "flight is a different question from dropping a point")
    p.add_argument("--plot", nargs="*", metavar="METHOD",
                   help="draw the distribution of each flight's unsupported share, one "
                        "figure per method named. No value means the DP_MuO, "
                        "the only solver that enumerates its whole front and can "
                        "therefore exhibit unsupported points at all")
    p.add_argument("--small-fronts", type=int, default=None, metavar="N",
                   help="write, per method, the flights whose cleaned front has fewer "
                        "than N points. Read from the pickles rather than the csv "
                        "because a front under 3 points has no hull and never reaches "
                        "the csv at all -- those are exactly the ones this lists")
    p.add_argument("--out-dir", type=Path,
                   default=BENCH / "figures" / "convexity_fronts")
    args = p.parse_args()

    # The population fingerprint goes into every list this run writes. Without it a run
    # restricted to a subset silently OVERWRITES the full-population lists under the same
    # name, and nothing on disk says which population a given list was derived from --
    # the exact failure the campaign/sweep folders exist to prevent elsewhere. The
    # default selection is the campaign's own manifest and gets no tag, so the ordinary
    # run keeps the ordinary names.
    poptag = ""
    selection = None
    if str(args.flights_file).lower() != "none":
        if args.flights_file.suffix == ".csv":
            selection = set(pd.read_csv(args.flights_file, usecols=["flight_id"])
                            .flight_id.astype(str))
        else:
            selection = {ln.strip() for ln in args.flights_file.read_text().splitlines()
                         if ln.strip()}
        print(f"selection: {len(selection):,} flights from {args.flights_file.name}")
        if args.flights_file.resolve() != DEFAULT_FLIGHTS.resolve():
            digest = hashlib.sha1("|".join(sorted(selection)).encode()).hexdigest()[:8]
            poptag = f"_pop{len(selection)}-{digest}"
            print(f"           not the default manifest; lists tagged {poptag}")

    # Every key carries the sigma, because nothing else in a row does: two sigmas of the
    # same campaign and method would otherwise overwrite each other in these dicts and
    # the run would report the last one as if it were all of them.
    all_per, all_regions, conv_ids, sizes = {}, {}, {}, {}
    for camp in args.campaign:
        for sg in args.sigma:
            per, regions = {}, {}
            for key, label, rel, glob_, sweeps_, expected in SOURCES[camp]:
                folder = BENCH / rel
                df, rg, tally = load(folder, glob_.format(s=sg), selection,
                                     args.tol, args.min_span, args.only_converged,
                                     sweeps_, expected)
                per[key], regions[key] = df, rg
                conv_ids[(camp, sg, key)] = tally["converged_ids"]
                sizes[(camp, sg, key)] = tally["front_size"]
                undef = ("  ".join(f"{v} {k}" for k, v in tally["undefined"].items())
                         or "none")
                print(f"[{camp} s{sg:g}] {label:22s} {tally['files']:>5} pickles in "
                      f"{folder.name}: {len(df):>5} measured, "
                      f"{tally['stub']} error stubs, "
                      f"{tally['unselected']} outside the selection, "
                      f"{tally['partial']} not fully converged")
                print(f"{'':33s} undefined: {undef}")
            print(f"\n{chr(35) * 78}")
            print(f"### CAMPAIGN: {camp}   sigma = {sg:g}")
            print(f"{chr(35) * 78}")
            report(per, args.tol, [(k, l) for k, l, *_ in SOURCES[camp]])
            for k in per:
                all_per[(camp, sg, k)] = per[k]
                all_regions[(camp, sg, k)] = regions[k]

    summary(all_per)

    # Every sigma that went in is in the name: a 3-sigma run must not overwrite the csv
    # of a 1-sigma one, and nothing inside the file says which sigmas it holds.
    stag = "-".join(f"{s:g}" for s in sorted(args.sigma))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fl = pd.concat([d.assign(campaign=c, method=k)
                    for (c, _, k), d in all_per.items() if not d.empty],
                   ignore_index=True)
    have = [d.assign(campaign=c, method=k)
            for (c, _, k), d in all_regions.items() if not d.empty]
    rg = pd.concat(have, ignore_index=True) if have else pd.DataFrame()
    tag = "_converged" if args.only_converged else ""
    f1 = args.out_dir / f"convexity_flights_sigma{stag}{tag}.csv"
    f2 = args.out_dir / f"convexity_regions_sigma{stag}{tag}.csv"
    fl.to_csv(f1, index=False)
    rg.to_csv(f2, index=False)

    # The flight lists go in figures/ itself, not the convexity subfolder: they are a
    # property of the CAMPAIGN, not of this analysis, and anything else that wants "the
    # flights whose front is complete" should find them without knowing which script
    # produced them.
    listdir = BENCH / "figures"
    listdir.mkdir(parents=True, exist_ok=True)
    print()
    for (camp, sg, key), ids in conv_ids.items():
        mode = {"opentop_single": "singlestart", "opentop_ms": "multistart",
                "dp": "dp", "dp_muo": "dp_muo"}[key]
        if not ids:
            continue      # a method with no pickles at this sigma has no list to write
        out = listdir / f"converged13_{camp.lower()}_{mode}_sigma{sg:g}{poptag}.txt"
        out.write_text("".join(f"{i}" + chr(10) for i in sorted(ids)), encoding="utf-8")
        print(f"{len(ids):>6,} flights -> {out.name}")
    for camp in args.campaign:
        for sg in args.sigma:
            got = [v for (c, s, _), v in conv_ids.items() if c == camp and s == sg]
            if len(got) == 4 and all(got):
                common = set.intersection(*(set(v) for v in got))
                out = listdir / f"converged13_all4_{camp.lower()}_sigma{sg:g}{poptag}.txt"
                out.write_text("".join(f"{i}" + chr(10) for i in sorted(common)),
                               encoding="utf-8")
                print(f"{len(common):>6,} flights -> {out.name}   (complete in ALL FOUR)")
    print()

    if args.plot is not None:
        want = args.plot or ["dp_muo"]
        pop = args.flights_file.name if selection is not None else "the campaign"
        npop = len(selection) if selection is not None else 0
        for camp in args.campaign:
            for key in want:
                frames = {s: d for (c, s, k), d in all_per.items()
                          if c == camp and k == key and not d.empty}
                if not frames:
                    print(f"no rows to plot for {camp}/{key}")
                    continue
                plot_unsupported_share(frames, key, args.out_dir, pop,
                                       npop or max(len(d) for d in frames.values()),
                                       camp)

    if args.small_fronts is not None:
        for (camp, sg, key), sz in sizes.items():
            small = sorted(f for f, v in sz.items() if v < args.small_fronts)
            out = (listdir / f"small_front_lt{args.small_fronts}_{camp.lower()}_"
                             f"{key}_sigma{sg:g}{poptag}.txt")
            out.write_text("".join(f"{f}" + chr(10) for f in small), encoding="utf-8")
            hist = {}
            for f in small:
                hist[sz[f]] = hist.get(sz[f], 0) + 1
            print(f"{len(small):>6,} of {len(sz):,} flights with a front under "
                  f"{args.small_fronts} points -> {out.name}")
            print(f"{'':>8}by size: "
                  + "  ".join(f"{k}pt x{v}" for k, v in sorted(hist.items())))
        print()

    print(f"per-flight rows  -> {f1}   ({len(fl):,} rows)")
    print(f"per-region rows  -> {f2}   ({len(rg):,} rows)")


if __name__ == "__main__":
    main()
