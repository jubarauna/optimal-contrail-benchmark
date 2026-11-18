"""plots_comparison.py — solve-time comparison across the four methods.

Reads the pickles of the four campaigns into one table (one row per flight x sigma x
method) and draws four figures, one class each:

  RatioEcdf         paired per-flight time ratios, one panel per sigma
  TimeDistribution  violin + box of the raw solve time, ONE panel: sigma=2 on the
                    bspline campaign, over the flights that converged in all four
                    methods (so the DP and the DP_MuO are shown on those
                    flights and no others)
  TimeAndCount      the same, with a second row: how many trajectories that time bought
  Scalability       solve time vs problem size, one small multiple per method
  BudgetScatter     one figure per method: every flight's +5%-fuel trajectory, one
                    panel per sigma, coloured by the time that point cost
  JointParetoMultiVsSingle
                    sigma=2, OpenTOP multistart vs singlestart alone: the front the two
                    jointly define, with how many of the other mode's points each beats
  OwnPareto         one figure per method at sigma=2: that method's OWN front only,
                    contrail cost against extra fuel over the fuel-only trajectory.
                    Four files on one shared pair of axes, meant to be laid out 2x2

Read TimeAndCount before quoting any number from the others. The four campaigns do
NOT deliver the same thing per flight: DP and both OpenTOP modes always return 13
trajectories (1 fuel-only + 6 weighted-sum + 6 epsilon-constraint), while DP_MuO
returns the whole Pareto front -- median 54, up to several thousand. A ratio of two
solve times says nothing about that, so on its own it flatters DP_MuO on the flights
where the front is small and punishes it where the front is large. DP vs OpenTOP is
the one pair that is like-for-like.

Why ``solver_s`` and not the sum of the phases: ``interpolant_s`` measures building the
bspline, which is infrastructure SHARED between solvers, and in `g2d` the DP runs before
DP_MuO inside the same task -- the DP pays for the whole interpolant and DP_MuO
picks it up from the cache. Summing the phases would produce a "DP_MuO is 200x
cheaper" that is an artefact of the execution order, not of the algorithm. From
2026-09-28 the two run as separate jobs and a cache hit depends on the node; each
pickle then says which it was in ``interpolant_cache_hit``.

Two sweep sets, same figures, same file names, one folder each (see SWEEP_SETS):

    --sweeps all      every sweep pooled per method      -> figures/all/<campaign>/
    --sweeps eps_ws   OpenTOP by its epsilon-constraint,
                      DP by its weighted sum,
                      DP_MuO by its front     -> figures/eps_ws/<campaign>/

Under a restricted set the solve time is the chosen sweeps' own: MEASURED for the DP
(its per-sweep phases ws_s / eps_s / anchor_s), the recorded per-solve times for OpenTOP,
the whole phase for DP_MuO -- see sweep_time(). ``memory_and_disk`` and
``memory_and_disk_added`` are the exception -- process RSS and one pickle cannot be split
by sweep, so they are identical in both folders and say so on the page.

Three time bases, same figures, same file names, one folder each (see BASES):

    --time solve              solve_s + posthoc_s                -> <campaign>/
    --time setup_solve        + solver_setup_s                   -> <campaign>/setup_solve/
    --time meteo_setup_solve  + data_setup_s                     -> <campaign>/meteo_setup_solve/

(<campaign> is figures/<sweeps>/<campaign>/, so the two axes never share a folder.)

    python 4_Post_Treatment/plots_comparison.py
    python 4_Post_Treatment/plots_comparison.py --time setup_solve
    python 4_Post_Treatment/plots_comparison.py --time setup_solve --refresh

``--refresh`` re-reads the pickles into <campaign>/timings.csv, which ALL bases share, so
adding a campaign is: point SOURCES at the new directory, run once with --refresh, then
run each basis you want. Nothing else changes.

Everything a campaign produces -- figures AND csvs -- lands under its own folder,
figures/bspline_interp/ or figures/linear_interp/, so the two can never be read as
one set. Writes <campaign>/[<basis>/]*.{png,pdf} and <campaign>/timings.csv.

The figures themselves are then filed by paper section (each class's ``section``):
4_Post_Treatment/<section>/<sweeps>/<campaign>/[<basis>/], with <section> one of
Computing_time, Optimality, Fuel_study. The csvs and caches stay under figures/. The
opentop_*_vs_dp_buckets figures go straight to 4_Post_Treatment/Optimality/, as before.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import gzip
import pickle
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")     # the labels contain non-cp1252 characters

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.instrument import TIME_FIELDS, four_times  # noqa: E402
from src.paths import results_root  # noqa: E402

BENCH = results_root()
FIGDIR = BENCH / "figures"
# The OpenTOP-vs-DP optimality figures go here rather than to the campaign folder.
OPTIMALITY_DIR = ROOT / "4_Post_Treatment" / "Optimality"
# Every figure is saved under the paper section it belongs to (its class's ``section``):
# 4_Post_Treatment/<section>/<the same sub-path it has under FIGDIR>, so the
# sweep / campaign / time-basis folders are kept and two runs never overwrite each other.
SECTIONS_ROOT = ROOT / "4_Post_Treatment"
CACHE = FIGDIR / "timings.csv"          # shared by every basis; --refresh rebuilds it

# ── print size ───────────────────────────────────────────────────────────────
# The figures are drawn at the size they will be PRINTED, and the fonts are set in
# final points. That is the whole trick: a 12.6 in wide figure dropped into a 7.16 in
# column is scaled by 0.57, so its 9 pt label lands at 5.1 pt and disappears. Sizing
# the canvas instead of the text keeps line widths, markers and spacing right too.
#
# Widths are the usual two-column measures: 182 mm across both columns, 88 mm for one,
# and the type matches docs/paper_figures.py -- 8 pt for titles, labels, ticks and
# legends (10 pt titles), in the same Times-compatible serif -- so these figures and the paper's
# schematic ones read as one set when placed at 100 %. The figures are drawn at the
# size they will be printed: placed at any other width, the type scales with them.
_TYPE_SCALE = 1.0
FIG_W_FULL = 7.16 * _TYPE_SCALE     # in — spans both columns
FIG_W_COL = 3.50 * _TYPE_SCALE      # in — one column

# Final point sizes, as they appear on paper: 10 pt for titles, 8 pt for every other
# piece of text -- axis labels, ticks, legends, in-panel notes and value labels alike.
# Nothing is drawn below 8 pt: a size set as a fraction of these would undo the rule.
FS = {
    "suptitle": 10.0,
    "title": 10.0,
    "label": 8.0,
    "tick": 8.0,
    "legend": 8.0,
    "annot": 8.0,
}


@contextlib.contextmanager
def type_sizes(text: float, title: float):
    """Draw one figure at other type sizes than FS: every FS entry and the rc sizes
    take ``text`` (titles ``title``) inside the block, and are restored after it.

    For the figures asked for at a larger size than the module's. Text keeps the size
    it was created with, so the figure has to be BUILT inside the block; saving it
    afterwards is fine.
    """
    saved = dict(FS)
    FS.update({k: (title if k in ("title", "suptitle") else text) for k in FS})
    with plt.rc_context({"font.size": text, "axes.labelsize": text,
                         "xtick.labelsize": text, "ytick.labelsize": text,
                         "legend.fontsize": text, "axes.titlesize": title,
                         "figure.titlesize": title}):
        try:
            yield
        finally:
            FS.clear()
            FS.update(saved)


# The serif list of docs/paper_figures.py, in its order of preference.
SERIF_FONTS = ["Liberation Serif", "TeX Gyre Termes", "Times New Roman", "DejaVu Serif"]

# Matplotlib's own defaults (10 pt) apply to anything not sized explicitly — tick
# labels above all. Without this the axes would still print at ~5.7 pt.
plt.rcParams.update({
    # Same family and maths face as docs/paper_figures.py. Titles are not bold there,
    # so they are not bold here either.
    "font.family": "serif",
    "font.serif": SERIF_FONTS,
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": FS["title"],
    "axes.labelsize": FS["label"],
    "xtick.labelsize": FS["tick"],
    "ytick.labelsize": FS["tick"],
    "legend.fontsize": FS["legend"],
    "figure.titlesize": FS["suptitle"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "hatch.linewidth": 0.4,  # the 1.0 default is heavier than the bars at print size
    "pdf.fonttype": 42,      # embed TrueType, not Type 3 — most journals require it
    "ps.fonttype": 42,
})


def paper_size(per_panel_w: float, n_panels: int, height: float,
               span: str | None = None) -> tuple[float, float]:
    """The old (per_panel x n, height) canvas, rescaled to the printed width.

    The aspect ratio is preserved, so every panel keeps the proportions it was laid out
    with; only the absolute size changes. Passing span="col" targets a single column,
    which suits the one-panel figures.
    """
    # A single panel does not need both columns: the linear campaign only ran at sigma=2,
    # so its per-sigma figures have one panel each and would otherwise be blown up to
    # 7.16 in wide and proportionally tall. That is the DEFAULT (span=None), not a rule:
    # span="full" used to be downgraded here too, which made the documented override a
    # no-op and left the joint-Pareto scatter at 3.5 in with 9 pt type on top of it.
    if span is None:
        span = "col" if n_panels <= 1 else "full"
    target = FIG_W_FULL if span == "full" else FIG_W_COL
    drawn = per_panel_w * max(n_panels, 1)

    # CHROME ALLOWANCE. Height used to be pure aspect, which was right while the type was
    # small enough to ignore. It is not any more: a suptitle, a panel title, an axis label
    # and a row of ticks together occupy a fixed number of POINTS, and points do not
    # shrink when the panel does. At 30 pt the chrome ate two thirds of the canvas and
    # left the bars in a strip. So the panel keeps its aspect and the text's own space is
    # added on top, computed from the sizes actually in use -- which means this tracks FS
    # automatically instead of needing a hand-tuned height per figure.
    chrome_in = (FS["suptitle"] + FS["title"] + FS["label"] + FS["tick"]) / 72.0 * 1.7
    return (target, height * target / drawn + chrome_in)


def fit(fig, text: str, pt: float | None = None) -> str:
    """Wrap a caption to the width of ``fig`` so it cannot widen it.

    ``bbox_inches="tight"`` grows the saved canvas to contain everything drawn, so a
    single long centred caption silently undoes the print sizing: the worst one here
    stretched a 7.16 in figure to 12.78 in, and every font then shrank by that factor
    when the figure was placed in the column. Wrapping is what keeps the canvas honest.

    The width comes from the figure rather than from FIG_W_FULL because the one-panel
    figures (the linear campaign ran only sigma=2) are single-column, and wrapping those
    to the two-column measure would push them wide again.
    """
    pt = pt or FS["annot"]
    chars = max(28, int(fig.get_size_inches()[0] / (0.52 * pt / 72.0)))
    return "\n".join(textwrap.wrap(" ".join(text.split()), chars))


# What "time" means. Pick one with --time; the figures are identical otherwise and the
# folder is named after the choice. The columns are the four reported times
# (src.instrument.TIME_FIELDS, the same four for every method since 2026-10-19), summed,
# missing ones counting as zero:
#
#   data_setup_s    meteo slice + contrail interpolant build/load (meteo_s + interpolant_s)
#   solver_setup_s  window, flown route, DP cost map + the solver object (env_s + build_s):
#                   graph for DP/DP_MuO, Cruise + optimizer for OpenTOP
#   solve_s         every solve; DP_MuO up to its final front. Post-hoc excluded.
#   posthoc_s       the evaluation at the end: fuel + contrail (evaluate_path for the
#                   graph solvers, reprice for OpenTOP)
#
# Every basis counts the post-hoc: it is what turns a solve into the reported fuel and
# contrail, for all three methods. OpenTOP pickles before 2026-10-19 have no post-hoc
# term (posthoc_s = 0), and pickles before 2026-09-28 have no env_s, so do not mix those
# generations with the current one. opentop still assembles its NLP inside each
# trajectory() call, so part of its setup stays inside solve_s.
BASES = {
    "solve":             (["solve_s", "posthoc_s"], "solve + post-hoc"),
    "setup_solve":       (["solver_setup_s", "solve_s", "posthoc_s"],
                          "setup + solve + post-hoc"),
    "meteo_setup_solve": (["data_setup_s", "solver_setup_s", "solve_s", "posthoc_s"],
                          "data + setup + solve + post-hoc"),
}

# Pickle name = <flight_id>_sigma<sigma><suffix>.pkl.gz, one entry per SOURCES method.
# Only ParetoSpread needs it: the cache holds one budget point per flight, not the front.
PICKLE_SUFFIX = {"opentop_single": "", "opentop_ms": "_ms",
                 "dp": "_dp", "dp_muo": "_dp_muo"}
SPREADDIR = FIGDIR / "pareto_spread"     # ParetoSpread writes here, whatever --time says

TIME = "t"                       # the column every figure plots; built by add_time_column
TLABEL, OUTDIR = "solve", FIGDIR  # set by main()

# SIZE_SIGMA was here. Scalability now declares the sigma with the rest of its pinned
# slice (pin_sigma), so the sigma, the campaign and the flight list are one statement
# instead of a loose constant and a default nobody sees.

# The two campaigns: same population, same altitude window, same meteo grid, and the
# cost field interpolated differently. They are kept apart all the way down -- own
# directories, own cache, own output folder -- because the pickle name does NOT encode
# the interpolant, so a figure mixing the two would look exactly like a correct one.
#
# (method, directory, glob). The directory is the method's root under 3_Results/contrail_2d_cruise/
# and the glob reaches into its m0 folders (m_060/, m_080/), so both masses are read.
# The OpenTOP singlestart is start 0 of the multistart run, written by
# extract_singlestart.py to OpenTOP/Singlestart_start0/ -- not the older Singlestart/
# campaign, which started from opentop's default guess. Its glob is `_sigma[0-9]` with
# no suffix so it can never pick up an `_ms` or `_fuel` pickle.
TOL1000 = BENCH.parent / "contrail_tol1000_v2"
CRUISE2D = BENCH.parent / "contrail_2d_cruise"
CAMPAIGNS = {
    "bspline": {
        "figdir": "bspline_interp",
        "sigmas": [0.0, 1.0, 2.0],
        "sources": [
            ("opentop_single", BENCH / "OpenTOP" / "Singlestart_start0", "m_0*/*_sigma[0-9].pkl.gz"),
            ("opentop_ms", BENCH / "OpenTOP", "m_0*/*_sigma[0-9]_ms.pkl.gz"),
            ("dp", BENCH / "DP", "m_0*/*_dp.pkl.gz"),
            ("dp_muo", BENCH / "DP_MuO", "m_0*/*_dp_muo.pkl.gz"),
        ],
    },
    # sigma 2 only: the linear campaign was never run at sigma 0 or 1.
    "linear": {
        "figdir": "linear_interp",
        "sigmas": [2.0],
        "sources": [
            ("opentop_single", BENCH / "open_top_run" / "Linear" / "contrails_linear",
             "*_sigma[0-9].pkl.gz"),
            ("opentop_ms", BENCH / "open_top_run" / "Linear" / "contrails_ms_linear",
             "*_sigma[0-9]_ms.pkl.gz"),
            ("dp", BENCH / "dp_run" / "Linear", "*_dp.pkl.gz"),
            ("dp_muo", BENCH / "dp_muo_run" / "Linear", "*_dp_muo.pkl.gz"),
        ],
    },
    # The bspline campaign rerun with the lateral path imposed (track_fixed, track
    # tolerance 1000 m), in its own results root next to contrail/. Same layout, so the
    # same globs; its own population (the singlestart-converged flights OF THIS RUN, not
    # the bspline lists) and its own Optimality folder, so nothing of the first campaign
    # is overwritten or reused. Its population is stricter than bspline's: a flight is
    # in only if BOTH OpenTOP modes converged on all 13 points at sigma=2.
    "tol1000": {
        "figdir": "tol1000",
        "sigmas": [0.0, 1.0, 2.0],
        "sources": [
            ("opentop_single", TOL1000 / "OpenTOP" / "Singlestart_start0", "m_0*/*_sigma[0-9].pkl.gz"),
            ("opentop_ms", TOL1000 / "OpenTOP", "m_0*/*_sigma[0-9]_ms.pkl.gz"),
            ("dp", TOL1000 / "DP", "m_0*/*_dp.pkl.gz"),
            ("dp_muo", TOL1000 / "DP_MuO", "m_0*/*_dp_muo.pkl.gz"),
        ],
        "population": tuple(
            BENCH.parent / "convergence" / "tol1000"
            / f"opentop_single_and_ms_sigma2_tol1000_converged_{m}.txt"
            for m in ("m060", "m080")),
        "optimality": OPTIMALITY_DIR / "tol1000",
        # The DP of this run is weighted-sum only (no epsilon-constraint sweep), so a
        # complete DP flight is 1 + 6 solves, not 13.
        "expected_solves": {"all": {"dp": 7}},
    },
    # OpenTOP with TrackCruise (lateral path imposed exactly) and its DP, sigma 2 only,
    # in its own results root. Population: flights where BOTH OpenTOP modes converged on
    # all 13 points at sigma=2 (3_Results/convergence/2d_cruise/). DP is ws-only: 1 + 6.
    "2d_cruise": {
        "figdir": "2d_cruise",
        "sigmas": [2.0],
        "sources": [
            ("opentop_single", CRUISE2D / "OpenTOP" / "Singlestart_start0", "m_0*/*_sigma[0-9].pkl.gz"),
            ("opentop_ms", CRUISE2D / "OpenTOP", "m_0*/*_sigma[0-9]_ms.pkl.gz"),
            # DP re-priced with the re-timed post-hoc (reprice_dp_retimed.py)
            ("dp", CRUISE2D / "DP_retimed", "m_0*/*_dp.pkl.gz"),
        ],
        "population": tuple(
            BENCH.parent / "convergence" / "2d_cruise"
            / f"opentop_single_and_ms_sigma2_2d_cruise_converged_{m}.txt"
            for m in ("m060", "m080")),
        "optimality": OPTIMALITY_DIR / "2d_cruise",
        "expected_solves": {"all": {"dp": 7}},
    },
}

# The campaigns the pinned figures accept: both are bspline, and each brings its own
# DEFAULT_POPULATION (select_campaign), so a pinned list never crosses campaigns.
BSPLINE_CAMPAIGNS = ("bspline_interp", "tol1000", "2d_cruise")

# ── which sweeps of each method take part ───────────────────────────────
# A campaign pickle holds three sweeps per scalarising method (fuel_only,
# weighted_sum, epsilon_constraint) and one label-setting front for the
# DP_MuO. "all" pools them, which is what every figure did until now and
# what the dominance and convexity reports still assume.
#
# "eps_ws" is the article comparison: each method is represented by the ONE sweep it
# is being defended on -- OpenTOP by its epsilon-constraint, the DP by its weighted
# sum (the DP has no native constraint, so its epsilon-constraint is a bisection on
# the weight and reaches nothing the weighted sum cannot; it was removed from
# model_2D on 2026-10-19, so only older DP pickles carry it), and the DP_MuO by its whole front,
# which is not a sweep at all. Pooling three sweeps per method answers "what did this
# method deliver"; this set answers "how do these four algorithms compare at the one
# job each is meant for". Different question, so it gets its own folder: SWEEPTAG
# sits ABOVE the campaign in the output path.
SWEEP_SETS = {
    "all": {
        "dp_muo":       ("pareto_dp_muo",),
        "dp":             ("fuel_only", "weighted_sum", "epsilon_constraint"),
        "opentop_single": ("fuel_only", "weighted_sum", "epsilon_constraint"),
        "opentop_ms":     ("fuel_only", "weighted_sum", "epsilon_constraint"),
    },
    "eps_ws": {
        "dp_muo":       ("pareto_dp_muo",),
        "dp":             ("weighted_sum",),
        "opentop_single": ("epsilon_constraint",),
        "opentop_ms":     ("epsilon_constraint",),
    },
    # Weighted sum on BOTH sides. The DP campaign is the ws-only run, so this is the
    # one set in which OpenTOP and the DP are asked the same question with the same
    # scalarisation -- the eps-constraint points and the fuel-only anchor, which the
    # DP has no counterpart for, simply do not enter.
    "ws": {
        "dp_muo":       ("pareto_dp_muo",),
        "dp":             ("weighted_sum",),
        "opentop_single": ("weighted_sum",),
        "opentop_ms":     ("weighted_sum",),
    },
}

# How many solves a COMPLETE flight has, per sweep set and method. Spelled out rather
# than read off the data: --full-only exists to drop the flights whose sweep did not
# finish, and a count taken as "the most any flight has" would move with the very
# failures it is meant to exclude. 13 = 1 fuel-only + 6 weighted sums + 6 eps-constraint;
# the DP campaign is the ws-only run, so 6. None = not fixed -- the DP_MuO
# returns its whole label set, so its front length is a property of the flight and
# complete there means only that nothing failed, which front_arrays already ensures.
EXPECTED_SOLVES = {
    "all":    {"dp_muo": None, "dp": 13, "opentop_single": 13, "opentop_ms": 13},
    "eps_ws": {"dp_muo": None, "dp": 6, "opentop_single": 6,  "opentop_ms": 6},
    "ws":     {"dp_muo": None, "dp": 6, "opentop_single": 6,  "opentop_ms": 6},
}

# ── the population every figure is drawn on ──────────────────────────────────
# ONE definition, applied at every point a pickle is admitted. The campaigns hold every
# flight that ever ran; the paper is about a narrower set, and which set that is has to
# be a property of this module rather than of whoever remembers to pass a flag.
#
# THE DEFAULT IS OpenTOP SINGLESTART CONVERGENCE, at sigma=2 on the bspline campaign,
# all 13 solves. Every figure, table and csv is restricted to those flights, including
# the ones about the graph solvers: comparing four methods over four different
# populations answers no question, and OpenTOP singlestart is the strictest gate of the
# four, so it is the one that binds. The consequence to state when reporting: the graph
# solvers are NOT being shown on the flights they alone could handle.
#
# The lists live in 3_Results/convergence/, one per m0 (596 + 2 912 = 3 508 ids), and
# the population is their union. "Singlestart" there is start 0 of the multistart run
# (extract_singlestart.py), so a flight whose 13 solves only converged thanks to the
# altitude-shifted starts is NOT in it. A missing file is a hard error rather than a
# silent full-population run.
CONVERGENCE_DIR = BENCH.parent / "convergence"
DEFAULT_POPULATION = tuple(
    CONVERGENCE_DIR / f"opentop_singlestart_sigma2_bspline_converged_{m}.txt"
    for m in ("m060", "m080"))
POPULATION = None       # frozenset of flight_ids, or None for "every flight that ran"
POPTAG = ""             # "" for the default; a fingerprint otherwise, see select_population

def list_name(path) -> str:
    """A flight list's file name, or the names of several joined by ' + '."""
    if isinstance(path, (str, Path)):
        return Path(path).name
    return " + ".join(Path(p).name for p in path)

# --full-only. A second, narrower gate than the population: the population is a fixed
# list of ids, this one is a per-figure test on the points actually in the frame, so it
# follows the methods the figure pools instead of freezing one method's convergence.
FULL_ONLY = False
FULLTAG = ""            # "_full" when it is on, so the two caches never share a name


def select_full_only(flag: bool) -> None:
    """Bind the completeness gate. It tags every joint-front cache and figure."""
    global FULL_ONLY, FULLTAG
    FULL_ONLY, FULLTAG = bool(flag), "_full" if flag else ""


def read_flight_list(path) -> frozenset:
    """One flight id per line, as a frozenset. A missing file is a hard error.

    ``path`` may also be several files, read as their union (the per-m0 lists).
    Shared by the --population gate and by the figures that pin a list of their own, so
    "the file has to be there" is stated once: a list that silently fell back to the
    whole campaign is the failure nothing inside the png would show.
    """
    if not isinstance(path, (str, Path)):
        return frozenset().union(*(read_flight_list(p) for p in path))
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"flight list not found: {path}\n"
                         "Write it with front_convexity.py, or pass --population none.")
    return frozenset(ln.strip() for ln in path.read_text().splitlines() if ln.strip())


def select_population(path) -> None:
    """Bind the population every loader admits from. ``none`` disables it.

    A NON-default population tags every figure, cache and csv the run writes. The
    default does not, because it IS the campaign's population and tagging it would
    rename the whole figure set for no gain. Anything else has to be distinguishable on
    disk: nothing inside a png says which flights went into it, and two populations
    sharing a name is the failure the campaign and sweep folders exist to prevent.
    """
    global POPULATION, POPTAG
    paths = [path] if path is None or isinstance(path, (str, Path)) else list(path)
    if paths[0] is None or str(paths[0]).lower() == "none":
        POPULATION, POPTAG = None, ""
        print("population: EVERY flight in the campaign (--population none)")
        return
    paths = [Path(p) for p in paths]
    ids = read_flight_list(paths)
    POPULATION = ids
    if {p.resolve() for p in paths} == {p.resolve() for p in DEFAULT_POPULATION}:
        POPTAG = ""
    else:
        digest = hashlib.sha1("|".join(sorted(ids)).encode()).hexdigest()[:8]
        POPTAG = f"_pop{len(ids)}-{digest}"
    print(f"population: {len(ids):,} flights from {list_name(paths)}"
          + (f"   (tagged {POPTAG})" if POPTAG else "   (the default)"))


def _in_population(df, what: str = "cache"):
    """Drop rows outside the population. The guard for CACHED frames.

    A csv written before the population existed -- or by a --population none run -- holds
    every flight, and read_csv walks straight past the loaders' admission test. Filtering
    on the way out makes a stale cache merely slower to be right instead of silently
    wrong, and spares invalidating every cache by hand.
    """
    if POPULATION is None or df.empty or "flight_id" not in df:
        return df
    keep = df.flight_id.isin(POPULATION)
    if (~keep).any():
        print(f"  {int((~keep).sum()):,} of {len(df):,} rows in {what} are outside the "
              "population and were dropped")
    return df[keep].reset_index(drop=True)


def admitted(fid) -> bool:
    """Whether this flight is in scope. The only gate; every loader calls it."""
    return POPULATION is None or fid in POPULATION


def stamp_population(folder: Path) -> None:
    """Record beside the figures which flights produced them.

    A png carries no provenance. This is the one file in the folder that says what the
    population was, so a figure found a year later can be traced without rerunning.
    """
    n = "every flight in the campaign" if POPULATION is None else f"{len(POPULATION):,}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "POPULATION.txt").write_text(
        "Figures in this folder were drawn on:\n"
        f"  population : {n}\n"
        f"  source     : {list_name(DEFAULT_POPULATION) if not POPTAG else 'see tag ' + POPTAG}\n"
        # The one figure that does not use the population above. Stated here because
        # POPULATION.txt is what a reader consults to find out what a png stands on,
        # and solve_time_distribution.png stands on something narrower.
        f"\nExceptions -- drawn at sigma = 2 only, on the singlestart-converged\n"
        f"flights ({list_name(DEFAULT_POPULATION)}),\n"
        f"whatever the population above says:\n"
        f"  solve_time_distribution\n"
        f"  memory_and_disk\n"
        f"  memory_and_disk_added\n"
        f"  solve_time_vs_size\n",
        encoding="utf-8")


# Rebound by select_campaign() before anything reads them.
SWEEPTAG = "all"
EXPECTED_OVERRIDE = {}      # a campaign's own EXPECTED_SOLVES entries, see CAMPAIGNS
CAMPDIR, SIGMAS, SOURCES = FIGDIR, CAMPAIGNS["bspline"]["sigmas"], CAMPAIGNS["bspline"]["sources"]


def select_campaign(name: str, sweeps: str = "all") -> None:
    """Point every path and the sigma panels at one campaign and one sweep set.

    Rebinding module globals rather than threading a parameter through twenty call
    sites is deliberate: there is exactly one campaign per process, and the cache
    file moves with it, so a stale cache can never be read into the wrong figures.

    The sweep set is a folder ABOVE the campaign -- figures/<sweeps>/<campaign>/ --
    for the same reason the campaigns are kept apart: nothing inside a figure or a
    csv says which sweeps went into it, so two sets sharing a folder would be
    indistinguishable on disk. ``all`` therefore writes to figures/all/, which is
    where the existing set already sits.
    """
    global CAMPDIR, SIGMAS, SOURCES, CACHE, SPREADDIR, SWEEPTAG
    global DEFAULT_POPULATION, OPTIMALITY_DIR, EXPECTED_OVERRIDE
    c = CAMPAIGNS[name]
    EXPECTED_OVERRIDE = c.get("expected_solves", {})
    DEFAULT_POPULATION = c.get("population", DEFAULT_POPULATION)
    OPTIMALITY_DIR = c.get("optimality", OPTIMALITY_DIR)
    SWEEPTAG = sweeps
    CAMPDIR, SIGMAS, SOURCES = FIGDIR / sweeps / c["figdir"], c["sigmas"], c["sources"]
    CACHE = CAMPDIR / "timings.csv"
    SPREADDIR = CAMPDIR / "pareto_spread"


def sweeps_of(method: str) -> tuple:
    """The sweeps of ``method`` the active set counts. One lookup, in one place."""
    return SWEEP_SETS[SWEEPTAG][method]


def expected_solves(method: str):
    """How many converged solves a complete flight has here, or None if not fixed."""
    return EXPECTED_OVERRIDE.get(SWEEPTAG, {}).get(method, EXPECTED_SOLVES[SWEEPTAG][method])

# Fields copied out of each pickle's ``context``. ``n_points`` is the trajectory
# discretisation and all four campaigns write it for the same flight, so it is the one
# problem-size axis that is comparable across methods.
FIELDS = ["flight_id", "sigma", "m0", "actype", "n_points", "n_waypoints",
          "route_km", "meteo_s", "interpolant_s", "build_s", "solver_s",
          # Post-hoc evaluation. ``posthoc_in_solver`` is True only in old DP pickles,
          # whose solver_s includes it; four_times() takes it out. Missing keys -> NaN.
          "posthoc_s", "posthoc_calls", "posthoc_in_solver", "setup_s",
          # The four reported times are added by load() from four_times(), so they exist
          # for old pickles too.
          # Since 2026-09-28, all three methods.
          "env_s", "interpolant_cache_hit"]

# Fixed order, palette slots 1-4. Validated for side-by-side marks: worst adjacent
# CVD dE 9.1, normal-vision 22.9. The all-pairs list FAILS on yellow<->orange, which is
# why Scalability is a small multiple per method rather than one overplotted scatter.
# (key, full name, colour, short name). The short one is for axis ticks only: at print
# size a panel is ~2.3 in wide, and a long name stacked under four
# positions runs into its neighbours. The full name stays in titles and legends, where
# there is room, so nothing is lost — only the axis is abbreviated.
METHODS = [
    ("dp",             "Dynamic Programming",     "#2a78d6", "DP"),
    # The key "dp_muo" is the pickle suffix and the SOURCES key, not a display string.
    ("dp_muo",       "DP_MuO",                  "#eb6834", "DP_MuO"),
    ("opentop_single", "OpenTOP singlestart",     "#1baf7a", "OpenTOP\nsingle"),
    ("opentop_ms",     "OpenTOP multistart",      "#eda100", "OpenTOP\nmulti"),
]

# The method names spelled out in full, by request, for the two figures that draw ONE
# violin per method on a single panel and have the room. The four-panel layouts keep the
# METHODS abbreviations, which is what they are sized for.
#
# The line break is not cosmetic: one panel gives each method ~2.4 in of axis and
# "OpenTOP Singlestart" set at 24 pt is 3.2 in on one line, so it would run into its
# neighbour. Two lines fit; the ticks are not rotated, for the reason method_ticks gives.
LONG_LABELS = {"dp": "DP", "dp_muo": "DP_MuO",
               "opentop_single": "OpenTOP\nSinglestart",
               "opentop_ms": "OpenTOP\nMultistart"}

# Flat fill instead of the see-through default, for the same two figures: their violins
# never overlap. ``label_dx`` travels with it because an opaque body needs the median
# label pushed clear of it -- see Plot.violin.
FLAT_VIOLIN = dict(alpha=1.0, label_dx=0.44)

# Ratios drawn by RatioEcdf: (numerator, denominator, label, colour). Pair colours, not
# method colours -- a ratio is not a method, so it gets its own encoding.
PAIRS = [
    ("dp_muo",       "dp",             "DP_MuO / DP",          "#2a78d6"),
    ("opentop_single", "dp",             "OpenTOP / DP",           "#eb6834"),
    ("opentop_ms",     "opentop_single", "OpenTOP multi / single", "#1baf7a"),
]

# How a sweep reads in a title. "pareto_dp_muo" is not a sweep at all — one
# label-setting solve IS the front — so it is named for what it is.
SWEEP_LABEL = {"pareto_dp_muo": "whole front", "epsilon_constraint": "ε-constraint",
               "weighted_sum": "weighted sum", "fuel_only": "fuel-only"}


def sweep_note(keys=None) -> str:
    """What the active sweep set restricts to, as a clause for a title. Empty for "all".

    Which sweep each method is represented by is the whole point of the restriction, so
    it goes on the page: a figure that does not say it cannot be read a year from now.
    Grouped by sweep and using the short method names because the spelled-out version
    ran to four lines on the single-column figures, where it was taller than the axes.

    ``keys`` restricts it to the methods a figure actually draws — the three-solver
    joint front would otherwise announce a DP_MuO series it deliberately excludes.
    """
    if SWEEPTAG == "all":
        return ""
    groups: dict[tuple, list[str]] = {}
    for key, _, _, short in METHODS:
        if keys is not None and key not in keys:
            continue
        groups.setdefault(sweeps_of(key), []).append(" ".join(short.split()))
    return "  —  " + " · ".join(
        ", ".join(SWEEP_LABEL.get(x, x) for x in sw) + ": " + " + ".join(names)
        for sw, names in groups.items())


SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
# GRID was #e8e7e3, which is 1.21:1 against the surface -- too faint to follow across a
# panel this wide once the figures went to one panel at full width, and the first thing
# to disappear when a png is scaled back into a column. #cbc8c1 is 1.63:1: still
# recessive enough to sit behind the marks (set_axisbelow in style()), now actually
# readable. RULE is unchanged -- it is a drawn reference line, not chrome, and darkening
# the grid towards it is why the two are still 1.4x apart.
GRID, RULE = "#cbc8c1", "#b8b7b2"

BUDGET = 1.05        # BudgetScatter: the fuel a flight may spend above its cheapest option
FAILED = "#8a8985"   # neutral grey for a point that did not converge

# Ramp for the time colorbar: green = fast, through yellow, to red = slow. Built in
# OKLCH so the yellow midpoint is a real hue change rather than a bright spot: lightness
# is held at 0.75 across green-yellow-gold and then falls to 0.36 at the red end, so hue
# separates the fast half and lightness the slow half. A brighter yellow was rejected --
# at L 0.86 it contrasts 1.47:1 with the surface and 3 px dots in that band disappear.
# Every step here clears 2:1 (2.03:1 at the green end, 11.4:1 at the red).
# Caveat to state when using it: green-to-red is the pair red-green colour blindness
# collapses, and the flat-lightness first half means those readers lose the fast end.
TIME_RAMP = ["#40cc6d", "#7fc640", "#a8bc01", "#c1b10b", "#cda207", "#d19006",
             "#c77603", "#ba5b04", "#a83d02", "#941a00", "#77020f"]


def budget_point(results: dict, method: str) -> dict:
    """The one trajectory this flight would actually fly under a +5% fuel budget.

    Out of the flight's own front, takes every point the budget can afford and keeps the
    one with the LEAST contrail, breaking ties on the cheaper one. That is the question
    the budget actually asks: given this much fuel to spend, what is the best contrail
    it buys.

    Not the point that spends the most fuel: on a true front the two coincide, but the
    OpenTOP sweep is six independent optimisations that can land in different local
    optima, so its last affordable point is sometimes WORSE in contrail than an earlier
    one -- and worse than flying the cheapest trajectory. Taking the minimum picks the
    best result the method actually produced, instead of the last one it happened to
    return.

    One point per flight is what makes the population plottable at all: the raw axes are
    kg and contrail units, and a 900 km flight and a 6000 km flight share no scale.
    Against the flight's own cheapest trajectory they do: x is the extra fuel actually
    spent, y is the contrail that extra fuel avoided.

    Both contrail values are stored raw rather than as a ratio. ``contrail`` goes
    NEGATIVE in OpenTOP (down to -657, on 16 % of flights) and is ~0 on the cheapest
    trajectory of some DP flights, so any percentage built on it divides by zero or by a
    negative. The difference does not.

    The front is whatever the ACTIVE sweep set gives this method — ``epsilon_constraint``
    for DP and OpenTOP under "all", the one sweep each is defended on under "eps_ws".
    DP_MuO has no sweep at all: it returns the whole front, so the budget is applied
    to that.
    """
    # "all" keeps the documented choice -- epsilon_constraint alone -- rather than the
    # pooled three sweeps, so this figure does not change meaning when the sweep sets
    # were added. A restricted set has already chosen one sweep per method, and that is
    # the one whose budget point is the question.
    sel = (sweeps_of(method) if SWEEPTAG != "all" else
           ("pareto_dp_muo",) if method == "dp_muo" else ("epsilon_constraint",))
    front = [t for sw in sel for t in results.get(sw, ())]
    # FALLBACK, and the reason it exists: the DP campaign was re-run weighted-sum only
    # (dp_run/Bspline, 2026-09-08), so its pickles have no epsilon_constraint sweep and
    # the line above selected nothing -- min() over an empty front, which is what a
    # --refresh crashed on. A method whose documented sweep was not run still has a
    # front, and the budget question is answerable on it, so the fallback is every sweep
    # the pickle actually holds. It changes which sweep the DP's budget point comes from,
    # which is exactly the kind of drift that must not be silent: the sweeps used are
    # returned in ``b5_sweep`` and load() prints them per method on every read.
    if not front:
        sel = tuple(k for k, v in results.items() if v)
        front = [t for sw in sel for t in results[sw]]
    if not front:
        nan = float("nan")
        return {"b5_fuel_pct": nan, "b5_contrail": nan, "b5_contrail_ref": nan,
                "b5_time_s": nan, "b5_converged": False, "b5_sweep": ""}
    cheapest = min(front, key=lambda t: t["fuel_kg"])
    affordable = [t for t in front if t["fuel_kg"] <= cheapest["fuel_kg"] * BUDGET]
    pick = min(affordable, key=lambda t: (t["contrail"], t["fuel_kg"]))
    return {
        "b5_fuel_pct": (pick["fuel_kg"] / cheapest["fuel_kg"] - 1) * 100,
        "b5_contrail": pick["contrail"],
        "b5_contrail_ref": cheapest["contrail"],
        # For DP_MuO this is the whole search, repeated on every point of the front:
        # one run yields the entire front, so no point has a time of its own.
        "b5_time_s": pick["meta"]["time_s"],
        "b5_converged": pick["success"],
        "b5_sweep": "+".join(sel),
    }


# The DP's per-sweep stopwatches: batch_graph2d times each sweep in its own phase and
# writes <prefix>_s (the solve, post-hoc excluded) and <prefix>_posthoc_s beside it.
DP_SWEEP_PHASE = {"fuel_only": "anchor", "weighted_sum": "ws", "epsilon_constraint": "eps"}


def sweep_time(payload: dict, method: str, sel: tuple) -> tuple[float, float]:
    """(solve s, post-hoc s) of the sweeps ``sel`` in this flight's solve phase.

      DP        MEASURED: batch_graph2d wraps each sweep in its own phase, so the
                weighted sum's time is ``ws_s`` itself, not a share of ``solver_s``.
                anchor_s + ws_s + eps_s == solver_s. A pickle without the phase fields
                (written before 2026-09-08) is an error, not a zero.

      OpenTOP   ``time_s`` per record TILES the phase: the 13 records sum to 100.0 % of
                ``solver_s`` (measured, 40 flights), so the selected records' share of
                ``solver_s`` is the recorded sweep time, and the same share of
                ``posthoc_s`` (0 before 2026-10-19).

      DP_MuO    one label-setting solve is the whole front and the whole phase, so its
                one "sweep" is all of ``solver_s`` and all of ``posthoc_s``.
    """
    ctx = payload["context"]
    times = four_times(ctx)
    solver_s = times["solve_s"] if ctx.get("solver_s") is not None else np.nan
    if method == "dp":
        solve = posthoc = 0.0
        for sw in sel:
            if sw not in payload["results"]:
                continue                     # sweep not run for this flight
            p = DP_SWEEP_PHASE[sw]
            if f"{p}_s" not in ctx:
                raise KeyError(f"{ctx.get('flight_id')} sigma={ctx.get('sigma')}: DP pickle "
                               f"has no {p}_s -- written before the per-sweep timers")
            solve += float(ctx[f"{p}_s"])
            posthoc += float(ctx.get(f"{p}_posthoc_s") or 0.0)
        return solve, posthoc
    posthoc_s = times["posthoc_s"]
    if method == "dp_muo":
        return solver_s, posthoc_s
    work = {sw: float(sum(r["meta"].get("time_s") or 0.0 for r in recs))
            for sw, recs in payload["results"].items()}
    total = sum(work.values())
    share = sum(work.get(sw, 0.0) for sw in sel) / total if total else np.nan
    return solver_s * share, posthoc_s * share


def load(refresh: bool) -> pd.DataFrame:
    """One row per flight x sigma x method: the timing fields plus what was delivered."""
    cache = CACHE
    if cache.exists() and not refresh:
        print(f"reading cache {cache.name}  (--refresh to re-read the pickles)")
        return _in_population(pd.read_csv(cache), cache.name)

    rows = []
    n_excl = 0
    for method, folder, pattern in SOURCES:
        n = 0
        for f in sorted(glob.glob(str(folder / pattern))):
            with gzip.open(f, "rb") as fh:
                payload = pickle.load(fh)
            if payload["error"]:          # stub: this flight produced no solution
                continue
            context = payload["context"]
            if not admitted(context.get("flight_id")):
                n_excl += 1
                continue
            # n_traj is the point of the TimeAndCount figure: DP and OpenTOP always
            # return 13, DP_MuO returns however many Pareto points exist. Counted over
            # the ACTIVE sweeps, so it stays the count the drawn time bought.
            sel = sweeps_of(method)
            n_traj = sum(len(payload["results"].get(sw, ())) for sw in sel)
            # The solve time of those sweeps (measured for the DP, see sweep_time), and
            # its share of the whole phase, printed as a check against the "all" figures.
            sweep_s, sweep_posthoc_s = sweep_time(payload, method, sel)
            solver_s = four_times(context)["solve_s"]
            share = sweep_s / solver_s if solver_s else np.nan
            # Per-solve times INSIDE this flight's sweep -> ParetoSpread. DP_MuO
            # writes the same front_total_s on every point (one label-setting sweep
            # builds the whole front), so its three values are equal by construction.
            per = sorted(r["time_s"] for r in (payload.get("summary") or [])
                         if r.get("time_s") is not None)
            rows.append({"method": method, "n_traj": n_traj,
                         "sweep_share": share,
                         "sweep_s": sweep_s,
                         "sweep_posthoc_s": sweep_posthoc_s,
                         "t_min": per[0] if per else np.nan,
                         "t_med": per[len(per) // 2] if per else np.nan,
                         "t_max": per[-1] if per else np.nan,
                         **budget_point(payload["results"], method),
                         **{k: context.get(k) for k in FIELDS},
                         **four_times(context)})
            n += 1
        print(f"  {method:16s} {n:>6d} pickles from {folder.relative_to(BENCH.parent)}")

    df = pd.DataFrame(rows)
    # Which sweep the budget point (ParetoSpread, BudgetScatter) was taken from, per
    # method. Printed because budget_point falls back when the documented sweep is not
    # in the pickles, and a fallback nobody sees is a figure that quietly changed
    # meaning -- see the note in budget_point.
    if not df.empty and "b5_sweep" in df:
        print()
        print("  budget point taken from:")
        for key, name, _, _ in METHODS:
            v = df.loc[df.method == key, "b5_sweep"]
            if v.notna().any() and len(v):
                got = sorted(set(v.dropna()) - {""})
                print(f"    {name:24s} {', '.join(got) or 'nothing -- no front'}")
    if n_excl:
        print(f"  {n_excl} rows dropped: outside the population")
    # Printed, not buried: with --sweeps eps_ws the y axis is a FRACTION of the measured
    # solve phase, and the fraction is the one number a reader needs to check that
    # against the "all" figures.
    if not df.empty and SWEEPTAG != "all":
        print()
        print(f"  share of the solve phase charged to {SWEEPTAG} "
              f"(median over flights):")
        for key, name, _, _ in METHODS:
            v = df.loc[df.method == key, "sweep_share"]
            if v.notna().any():
                print(f"    {name:24s} {v.median():6.1%}   "
                      f"sweeps: {', '.join(sweeps_of(key))}")
    df.to_csv(cache, index=False)   # the cache stays FULL; filter on the way out
    print(f"cache written to {cache.name}")
    return _in_population(df, cache.name)


def add_time_column(df: pd.DataFrame, basis: str) -> tuple[pd.DataFrame, str]:
    """Add the column every figure plots, as the sum the chosen basis asks for.

    One correction, and only when ``data_setup_s`` is part of the sum: DP and DP_MuO
    run in the SAME task, DP first, so DP pays the whole bspline build and DP_MuO
    reads it from the cache -- median 191 s against 0.073 s, DP_MuO the cheaper one in
    7 204 of 7 204 pairs. Left alone that is a 2 451x head start handed to DP_MuO by
    the execution order, not by the algorithm, so both are charged the build once at the
    price the one that actually built it paid. ``build_s`` needs no such correction:
    each method constructs its own optimizer (DP 0.212 s, DP_MuO 0.205 s at sigma=2).
    """
    columns, label = BASES[basis]
    df = df.copy()
    # A cache written before the four times existed: derive them from its phase columns.
    if not set(TIME_FIELDS) <= set(df.columns):
        four = pd.DataFrame([four_times(r) for r in df.to_dict("records")], index=df.index)
        for k in TIME_FIELDS:
            df[k] = four[k]
    # Under a restricted sweep set the solve and post-hoc terms are the selected sweeps'
    # (sweep_time: measured for the DP). Swapped here rather than in BASES so the basis
    # definitions keep saying what they measure, and so "all" keeps reading the raw column.
    if SWEEPTAG != "all":
        swap = {"solve_s": "sweep_s", "posthoc_s": "sweep_posthoc_s"}
        columns = [swap.get(c, c) for c in columns]
    if "data_setup_s" in columns:
        pair = df.method.isin(("dp", "dp_muo"))
        interp = (df[pair].groupby(["flight_id", "sigma"])["interpolant_s"]
                  .transform("max").fillna(0.0))
        df.loc[pair, "data_setup_s"] = df.loc[pair, "meteo_s"].fillna(0.0) + interp
    df[TIME] = df[[c for c in columns if c in df]].fillna(0.0).sum(axis=1)
    return df, label


class Plot:
    """Base for the four figures: shared chrome, shared marks, shared saving.

    A subclass sets ``stem`` (the output file name) and implements ``draw``, which
    builds the figure and returns it.
    """

    stem = ""
    outdir = None          # None -> OUTDIR (the --time folder); set to pin a folder
    section = None         # paper section folder under 4_Post_Treatment/; set per figure

    # ── the pinned slice (opt-in, per figure) ────────────────────────────────
    # A figure that must NOT follow the run's flags declares it here: the one sigma it
    # draws, the campaign whose flights its list describes, and the list itself. None
    # means "follow the run", which is what every figure but the two violin ones does.
    # TimeDistribution says why the violin figures cannot: four distributions read side
    # by side have to stand on one population, and the run's own gate does not give
    # them one. Declared on the base so the two figures share one definition of it.
    pin_sigma = None
    pin_campaign = None        # CAMPDIR.names accepted; see select_campaign
    pin_population = None      # a flight-id list file, or "default": the campaign's own
    # 600, not the 300 that is the print standard: the pdf is vector and ignores this
    # entirely, so dpi only ever sets how the PNG reads on screen -- and at 300 a
    # four-panel strip lands at 2116 x 691 px, which looks soft at any zoom.
    dpi = 600

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> None:
        # A pinned flight list describes ONE campaign (the interpolant is not in the
        # pickle name), so outside it the figure draws nothing and says so rather than
        # putting other data under the same file name.
        if self.pin_campaign and CAMPDIR.name not in self.pin_campaign:
            print(f"skipping {self.stem}: pinned to the {self.pin_campaign} campaign "
                  f"(this run is {CAMPDIR.name})")
            return
        fig = self.draw()
        if fig.get_layout_engine() is None:      # a constrained-layout figure places itself
            fig.tight_layout()
        folder = self.figure_dir()
        folder.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"{self.stem}.{ext}", dpi=self.dpi,
                        bbox_inches="tight", facecolor=SURFACE)
            print(f"wrote {folder / self.stem}.{ext}")
        plt.close(fig)

    def figure_dir(self) -> Path:
        """Where this figure is saved: its paper section, same sub-path as under FIGDIR."""
        folder = self.outdir or OUTDIR
        try:
            rel = folder.relative_to(FIGDIR)
        except ValueError:          # already a section folder (OPTIMALITY_DIR)
            return folder
        return SECTIONS_ROOT / self.section / rel

    def draw(self):
        raise NotImplementedError

    def pinned(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """``df`` (default self.df) cut down to the pinned sigma and flight list.

        Prints what is left per method. That report is not decoration: the whole point
        of a pinned list is that every column stands on the same flights, the counts are
        no longer drawn under them, and a method that lost pickles would otherwise
        change the figure without changing anything visible on it.
        """
        d = self.df if df is None else df
        if self.pin_sigma is not None:
            d = d[d.sigma == self.pin_sigma]
        ids = None
        pop = DEFAULT_POPULATION if self.pin_population == "default" else self.pin_population
        if pop is not None:
            ids = read_flight_list(pop)
            d = d[d.flight_id.isin(ids)]
        if d.empty:
            raise SystemExit(f"{self.stem}: nothing left at sigma = {self.pin_sigma}"
                             + (f" for the {len(ids):,} flights of "
                                f"{list_name(pop)}" if ids else ""))
        print(f"  {self.stem}: sigma = {self.pin_sigma:g}"
              + (f", {len(ids):,} flights from {list_name(pop)}" if ids else ""))
        for key, name, _, _ in METHODS:
            n = d.loc[d.method == key, "flight_id"].nunique()
            print(f"    {name:24s} {n:>6,} flights"
                  + ("" if ids is None or n == len(ids)
                     else f"   ({len(ids) - n:,} without a pickle)"))
        return d

    def times(self, sigma: float) -> pd.DataFrame:
        """flight_id x method matrix of solve times at one sigma — the pairing step."""
        return (self.df[self.df.sigma == sigma]
                .pivot_table(index="flight_id", columns="method", values=TIME))

    @staticmethod
    def style(ax) -> None:
        """Recessive grid, two spines, muted ticks — the same on every panel."""
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d4d3cf")
        ax.tick_params(colors=INK_2, labelsize=FS["tick"])

    @staticmethod
    def violin(ax, x: float, values: np.ndarray, color: str, fmt: str,
               alpha: float = 0.30, label_dx: float = 0.10) -> None:
        """One violin + box at position x, drawn in log10 of the values.

        Drawn in log10 on a linear axis rather than on a log axis because the violin is
        a kernel density: a density fitted in seconds and then squashed by a log axis is
        not the density of the log, and grows a bulge wherever the axis compresses.
        The median is printed beside the box -- at the median, not at the top of the
        violin, which would read as a value at that height.

        ``alpha`` is the fill only; the outline is drawn at full strength either way.
        0.30 is the default because ParetoSpread puts three violins in one slot and
        they have to be seen through. TimeDistribution and MemoryDistribution ask for
        1.0: one violin per position, never overlapping, and the flat fill is what was
        wanted on the page. The box inside is painted in SURFACE and sits above the body, so
        it stays readable at any alpha.

        ``label_dx`` is how far right of the position the median text is set. The
        default 0.10 puts it against the box, which is where it belongs while the fill
        is see-through. At alpha 1.0 it does not: the text plate is drawn in SURFACE
        over an opaque body and reads as a smear across it, so a figure that asks for a
        flat fill also has to push the label clear of the body -- 0.44, just outside the
        0.39 half-width.
        """
        v = np.log10(values)
        if v.min() == v.max():
            # A constant series (DP and OpenTOP always return 13) has no density to fit.
            ax.plot([x - 0.30, x + 0.30], [v[0], v[0]], color=color, lw=4,
                    solid_capstyle="round", zorder=3)
        else:
            body = ax.violinplot([v], positions=[x], widths=0.78,
                                 showextrema=False)["bodies"][0]
            body.set_facecolor(color)
            body.set_alpha(alpha)
            body.set_edgecolor(color)
            body.set_linewidth(1.4)
            body.set_zorder(2)
            box = ax.boxplot([v], positions=[x], widths=0.13, showfliers=False,
                             patch_artist=True, whis=(5, 95), zorder=3)
            box["boxes"][0].set(facecolor=SURFACE, edgecolor=INK_2, linewidth=1.0)
            for line in box["whiskers"] + box["caps"]:
                line.set(color=INK_2, linewidth=1.0)
            box["medians"][0].set(color=INK, linewidth=1.8)

        median = float(np.median(v))
        # The text plate follows the fill: 0.85 lets the grid through while the violins
        # are see-through, but pushed clear of an opaque body the label can land on a
        # NEIGHBOUR's violin (memory_and_disk, "3174" against the multistart tail), and
        # a translucent plate there reads as a stain on that violin instead of a label.
        ax.text(x + label_dx, median, fmt.format(10 ** median), ha="left", va="center",
                fontsize=FS["annot"], color=INK_2, zorder=5,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0,
                          alpha=0.85 if alpha < 1.0 else 1.0))

    @staticmethod
    def decade_ticks(ax, values: np.ndarray) -> None:
        """Relabel a linear log10 axis as 10^k, bounded by the data.

        The kernel spills about a decade past the extremes; without the bound the axis
        buys two empty decades at each end.
        """
        lo, hi = np.log10(values).min() - 0.5, np.log10(values).max() + 0.7
        ax.set_ylim(lo, hi)
        decades = np.arange(np.ceil(lo), np.floor(hi) + 1)
        ax.set_yticks(decades)
        ax.set_yticklabels([f"$10^{{{int(d)}}}$" for d in decades])

    @staticmethod
    def method_ticks(ax, counts: list[int] | None, labels: dict | None = None,
                     keys: tuple | None = None) -> None:
        """Method name plus its n under each position.

        The abbreviation comes from METHODS; "OpenTOP single" and "OpenTOP multi" are
        0.9 units apart, so they carry their own line break there. The second line is the
        bare flight count — "n = " ahead of it is 8 characters in a 0.44 in slot and
        collides with the neighbouring method.

        ``counts=None`` drops that second line. Only TimeDistribution does it: it pins
        ONE population for all four methods, so four identical numbers under four
        columns said nothing the figure needed. They are printed to stdout instead, and
        a column that lost its pickles would show up there -- not on the page any more.

        ``labels`` overrides the METHODS short names for one figure, keyed by method.
        Per figure rather than in METHODS because the abbreviation there is sized for
        the four-panel layouts, where a longer name does not fit.
        """
        # ``keys`` draws a subset of METHODS, in that order, at positions 0..n-1.
        shown = [m for m in METHODS if keys is None or m[0] in keys]
        if keys is not None:
            shown.sort(key=lambda m: keys.index(m[0]))
        names = [(labels or {}).get(key, short) for key, _, _, short in shown]
        ticks = (names if counts is None else
                 [f"{name}\n{n}" if n else f"{name}\n—"
                  for name, n in zip(names, counts)])
        ax.set_xticks(range(len(names)))
        # 8 pt like all other text. On the three-panel figures the positions are 0.44 in
        # apart and "OpenTOP" at 8 pt is 0.42 in wide, so the two OpenTOP columns nearly
        # touch -- the price of not shrinking the type. Rotation was tried and is
        # worse here, because these labels are three lines tall and rotating them makes
        # the count of one column run into the name of the next.
        ax.set_xticklabels(ticks, fontsize=FS["annot"], color=INK)
        # Tight to the violins (half-width 0.39) on the left and to the rightmost median
        # label on the right: every 0.1 unit given back widens the slots the 8 pt names
        # have to fit in.
        ax.set_xlim(-0.45, len(names) - 0.2)


# class RatioEcdf(Plot):
#     """For each flight, one method's time divided by another's, sorted into a curve.
#
#     Dividing two times of the SAME flight cancels the flight length, which otherwise
#     dominates everything (time grows with distance, Spearman rho 0.65 to 0.91). An
#     aggregate would mix "fast method" with "short flight"; a ratio does not.
#
#     What it does NOT cancel is what each side delivered — see TimeAndCount.
#     """
#
#     stem = "timings_ratio_ecdf"
#
#     def draw(self):
#         # sharex: the panels exist to compare sigmas, so the same ratio must land at the
#         # same place on the page in all three.
#         fig, axes = plt.subplots(1, len(SIGMAS), figsize=paper_size(4.2, len(SIGMAS), 4.4),
#                                  sharey=True, sharex=True, facecolor=SURFACE,
#                                  squeeze=False)
#         drawn = {}
#
#         for ax, sigma in zip(axes[0], SIGMAS):
#             wide = self.times(sigma)
#             self.style(ax)
#             ax.axvline(1.0, color=RULE, lw=1.2, ls="--", zorder=1)   # 1.0 = same time
#
#             # Alternating offsets: at sigma=2 the multi/single (5.1x) and OpenTOP/DP
#             # (22.7x) medians sit half a decade apart and the labels collide.
#             offsets = [(6, -13), (6, 7), (-8, 7)]
#             for k, (num, den, label, color) in enumerate(PAIRS):
#                 if num not in wide.columns:
#                     continue                     # multistart only ran at sigma=2
#                 r = (wide[num] / wide[den]).dropna()
#                 x = np.sort(r.to_numpy())
#                 y = np.arange(1, len(x) + 1) / len(x)
#                 ax.plot(x, y, color=color, lw=2.0, zorder=3, solid_capstyle="round")
#                 drawn[label] = color
#
#                 # Direct label on the median: the aqua sits below 3:1 on this surface,
#                 # so identity cannot rely on colour alone.
#                 median = float(r.median())
#                 ax.plot([median], [0.5], "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=4)
#                 ax.annotate(f"{median:.3g}×", (median, 0.5), textcoords="offset points",
#                             xytext=offsets[k], fontsize=FS["annot"], color=INK_2,
#                             ha="right" if offsets[k][0] < 0 else "left")
#
#             faster = (wide["dp_muo"] < wide["dp"]).mean()
#             ax.set_title(f"$\\sigma$ = {sigma:g}   ({len(wide)} flights)", fontsize=FS["title"], color=INK)
#             ax.set_xscale("log")
#             ax.set_xlabel(f"{TLABEL}-time ratio, per flight", fontsize=FS["label"], color=INK_2)
#             ax.annotate(f"DP_MuO faster\non {100 * faster:.0f}% of flights",
#                         (0.03, 0.90), xycoords="axes fraction", fontsize=FS["annot"], color=INK_2)
#
#         axes[0][0].set_ylabel("fraction of flights", fontsize=FS["label"], color=INK_2)
#         axes[0][0].set_ylim(0, 1)
#
#         handles = [plt.Line2D([], [], color=c, lw=2.0, label=l) for l, c in drawn.items()]
#         fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
#                    fontsize=FS["label"], bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
#         fig.suptitle(fit(fig, f"{TLABEL.capitalize()} time, paired per-flight comparison  —  "
#                      "left of 1× the numerator is faster", FS["suptitle"]),
#                      fontsize=FS["suptitle"], color=INK, y=1.0)
#         return fig


class TimeDistribution(Plot):
    """Raw solve time per method, four violins on ONE panel. Time only, no context.

    Pinned, by request, to a single slice rather than following the run's flags:

      sigma = 2 only            the other two panels are gone. sigma=2 is the slice the
                                article is argued on and the only one where all four
                                methods ran everywhere; the DP at sigma 0 and 1 has 219
                                and 205 flights against 2 747, so those panels compared
                                four methods over two populations two orders of
                                magnitude apart.
      bspline campaign only     the interpolant is not in the pickle name, so a linear
                                run would produce a figure indistinguishable from this
                                one. Under --interp linear this figure SKIPS itself and
                                says so, instead of drawing linear data under a list
                                that describes the bspline campaign.
      DEFAULT_POPULATION        the 3 508 flights whose 13 OpenTOP singlestart (start 0)
                                solves converged, from 3_Results/convergence/. Pinned
                                rather than inherited so that --population cannot move
                                the four violins onto another set: four distributions
                                read side by side have to stand on one population.

    The population is narrowed here rather than by passing --population on the command
    line because it belongs to THIS figure: the other figures are drawn on the default
    and the flag is global, so a run that got this figure right would get the rest wrong.
    """

    section = "Computing_time"

    stem = "solve_time_distribution"
    pin_sigma, pin_campaign, pin_population = 2.0, BSPLINE_CAMPAIGNS, "default"

    # This figure does NOT follow --time either. It is the four-method comparison, and
    # "solve" is not a comparable quantity across the four any more: the DP's
    # weighted-sum-only run leaves its graph construction in build_s (0.25 s) and,
    # inside solver_s, spends nearly all of the phase on post-hoc evaluation rather than
    # on the DP recursion (0.181 s against 0.010 s on the first flight). OpenTOP has no
    # build term by construction -- the equivalent work happens inside each solve -- so
    # summing solver_setup_s + solve_s + posthoc_s charges every method for everything it
    # did to produce its trajectories, and charges OpenTOP nothing it did not do (see
    # the note under BASES).
    KEYS = ("dp", "dp_muo", "opentop_ms")     # drawn, in this order
    pin_basis = "setup_solve"

    TEXT_PT, TITLE_PT = 14.0, 14.0             # larger than FS, by request

    def draw(self):
        with type_sizes(self.TEXT_PT, self.TITLE_PT):
            return self._draw()

    def _draw(self):
        sub, label = add_time_column(self.pinned(), self.pin_basis)
        # One panel, but still the TWO-COLUMN width: span="col" is paper_size's default
        # for a single panel and it does not work here. Four method names at 24 pt need
        # about 6 in of axis between them (see method_ticks) and a 3.5 in column gives
        # 1.6 -- "DP_MuO", "OpenTOP single" and "OpenTOP multi" ran into each
        # other, the median labels crossed the neighbouring violin, and the suptitle
        # wrapped to five lines. The height is set low so the panel comes out landscape
        # at that width instead of a 13 in square; paper_size adds the chrome the 32 pt
        # type needs on top of it.
        #
        # constrained, not tight: the sigma used to be a panel title and the figure has
        # no title at all now, and tight_layout leaves the space that chrome had as a
        # band of white above the axes. The engine reserves only what is there.
        fig, ax = plt.subplots(figsize=paper_size(4.6, 1, 2.2, span="full"),
                               facecolor=SURFACE, layout="constrained")
        self.style(ax)

        # Three methods, by request: the singlestart is left out, and the multistart's
        # time is ALL five starts of every sweep point (ws_s + eps_s sum every
        # candidate's wall time), not the winning start's alone.
        shown = [m for m in METHODS if m[0] in self.KEYS]
        shown.sort(key=lambda m: self.KEYS.index(m[0]))
        # OpenTOP drawn in the singlestart's green, by request: with one OpenTOP violin
        # on the page, it takes the colour OpenTOP has in the other figures.
        ot_green = next(c for k, _, c, _ in METHODS if k == "opentop_single")
        for x, (key, name, color, _) in enumerate(shown):
            v = sub.loc[sub.method == key, TIME].to_numpy()
            if len(v):
                self.violin(ax, x, v, ot_green if key == "opentop_ms" else color,
                            "{:.3g} s", **FLAT_VIOLIN)
        # counts=None: pinned() prints each method's flight count, and that is where a
        # missing pickle shows.
        self.method_ticks(ax, None, {**LONG_LABELS, "opentop_ms":
                                     "OpenTOP Multistart" + chr(10) + "(5 starts)"}, keys=self.KEYS)

        self.decade_ticks(ax, sub.loc[sub.method.isin(self.KEYS), TIME].to_numpy())
        # The basis is the figure's own, so the label comes from add_time_column rather
        # than from the run's TLABEL -- otherwise a --time solve run would label a
        # setup + solve axis "solve time".
        ax.set_ylabel(f"{label} time per pareto (s)", fontsize=FS["label"], color=INK_2)

        # Both the title and the footnote are gone by request, so EVERYTHING this
        # figure is conditional on is now unstated on the page and has to be in the
        # caption: sigma = 2, the bspline interpolant, the 2 725 flights complete in
        # all four methods, the sweep set if it is not "all", and the right-censored
        # DP_MuO flights that never finished and are therefore absent from
        # that violin, which makes its tail optimistic. The run prints the first three
        # and POPULATION.txt records them beside the file.
        return fig


class TimeAndCount(Plot):
    """TimeDistribution with a second row: how many trajectories that time bought.

    The two rows share the x positions, so one column is one method: how long it took,
    and what came out. DP and both OpenTOP modes are a flat bar at 13 — they always
    return the same 13 trajectories. DP_MuO's row is a full distribution because it
    returns the entire Pareto front, whose size is a property of the flight, not a
    setting. Reading the top row alone is what makes a solve-time ratio misleading.
    """

    section = "Computing_time"

    stem = "solve_time_and_count"

    def draw(self):
        fig, axes = plt.subplots(2, len(SIGMAS), figsize=paper_size(4.6, len(SIGMAS), 7.8),
                                 sharey="row", facecolor=SURFACE, squeeze=False)

        for col, sigma in enumerate(SIGMAS):
            sub = self.df[self.df.sigma == sigma]
            for row, (column, fmt) in enumerate([(TIME, "{:.3g} s"),
                                                 ("n_traj", "{:.0f}")]):
                ax = axes[row][col]
                self.style(ax)
                counts = []
                for x, (key, name, color, _) in enumerate(METHODS):
                    v = sub.loc[sub.method == key, column].to_numpy()
                    counts.append(len(v))
                    if len(v):
                        self.violin(ax, x, v, color, fmt)
                self.method_ticks(ax, counts)
                if row == 0:
                    ax.set_xticklabels([])       # the names belong under the bottom row
            axes[0][col].set_title(f"$\\sigma$ = {sigma:g}", fontsize=FS["title"], color=INK)

        self.decade_ticks(axes[0][0], self.df[TIME].to_numpy())
        self.decade_ticks(axes[1][0], self.df.n_traj.to_numpy())
        axes[0][0].set_ylabel(f"{TLABEL} time per flight (s)", fontsize=FS["label"], color=INK_2)
        axes[1][0].set_ylabel("trajectories returned per flight", fontsize=FS["label"], color=INK_2)

        fig.suptitle(fit(fig, f"What the {TLABEL} time bought  —  top: time per flight, "
                     "bottom: trajectories that time produced", FS["suptitle"]),
                     fontsize=FS["suptitle"], color=INK, y=1.0)
        # va="top" at y=0: the caption hangs below the canvas, which bbox_inches="tight"
        # then includes, so it can never overlap the method labels whatever their height.
        fig.text(0.5, 0.0, fit(fig,
                 "The number under each method is its flight count.  "
                 "DP and both OpenTOP modes always return the same 13 trajectories "
                 "(1 fuel-only + 6 weighted-sum + 6 ε-constraint); DP_MuO returns the "
                 "whole Pareto front, whose size the flight decides.  "
                 "A ratio of two times in the top row ignores the bottom row."),
                 ha="center", va="top", fontsize=FS["annot"], color=INK_2)
        return fig


class FrontSizeDistribution(Plot):
    """How many trajectories one DP_MuO solve returns, over the population.

    The other figures treat the front size as a caveat — DP and both OpenTOP modes always
    return 13 trajectories, the DP_MuO returns however many exist, so a solve
    time is not comparable between them. This figure is that caveat drawn: x is the front
    a flight got, y is how many flights got a front that size.

    Log x, because the range is four and a half decades: 1 to 50 480 points at
    sigma = 2. Bins are one per integer below 10, where the counts are small exact
    integers and 60 % of the flights sit, and log-spaced above it, where an integer is no
    longer a meaningful resolution. Linear y, because the question is where the mass is,
    and the mass is the mode.

    What that costs, and what the annotation is for: the tail bins hold one to five
    flights each and are invisible against a mode of several hundred. They are the
    expensive flights — the ones that make the solve-time violin span six decades — so
    the quantiles and the maximum are printed in the corner rather than left to a bar
    one pixel tall.

    Reads the DP_MuO alone, so it carries no legend: one series, named in the
    title. The sweep set does not enter it — the front is one label-setting solve either
    way — but the flight count can differ by a pickle or two between campaigns, so n is
    printed per panel.
    """

    section = "Computing_time"

    stem = "front_size_distribution"
    key = "dp_muo"

    @staticmethod
    def bins(hi: int) -> np.ndarray:
        """One bin per integer to 10, log-spaced above it, closed on ``hi``.

        Plain log-spaced bins from 1 would put no integer at all in [1.33, 1.78) and the
        first decade would come out a comb. Plain integer bins would need 50 000 of them.
        The join at 10 is where a bin stops being one count wide and starts being a ratio
        wide, which is also where the reader stops counting and starts reading shape.
        """
        low = np.arange(1, 11)
        per_decade = 12
        n = int(np.ceil((np.log10(hi + 1) - 1) * per_decade)) + 1
        high = np.round(np.logspace(1, np.log10(hi + 1), max(n, 2))).astype(int)
        return np.unique(np.concatenate([low, high, [hi + 1]]))

    TEXT_PT = TITLE_PT = 12.0          # every text of this figure, by request

    def draw(self):
        with type_sizes(self.TEXT_PT, self.TITLE_PT):
            return self._draw()

    def _draw(self):
        F = FS
        # constrained, not tight: at this type size the suptitle and the shared x label
        # need the engine to reserve their space, and the hand-placed y= offsets the
        # tight-layout version used would drop the title onto the panels.
        # The canvas is fixed at the THREE-panel geometry whatever the campaign has,
        # so bspline (three sigmas) and linear (only sigma=2) come out the same size
        # and can be put side by side. Sizing it off len(SIGMAS) instead made the
        # linear figure a portrait 6.6 x 6.9 in with one tall panel -- the same data,
        # not the same graph. A single sigma simply gets one wide panel.
        fig, axes = plt.subplots(1, len(SIGMAS), figsize=paper_size(4.6, 3, 2.8, span="full"),
                                 sharey=True, sharex=True, facecolor=SURFACE,
                                 layout="constrained", squeeze=False)
        color = next(c for k, _, c, _ in METHODS if k == self.key)
        name = next(n for k, n, _, _ in METHODS if k == self.key)

        allv = self.df.loc[self.df.method == self.key, "n_traj"].dropna()
        edges = self.bins(int(allv.max()))

        for ax, sigma in zip(axes[0], SIGMAS):
            self.style(ax)
            v = self.df.loc[(self.df.method == self.key) & (self.df.sigma == sigma),
                            "n_traj"].dropna().to_numpy()
            # sigma goes UNDER the panel, not over it: with the suptitle at 32 pt,
            # two lines of heading stacked above the axes read as one block and it
            # was not obvious which belonged to which panel. set_xlabel puts it
            # below the ticks; the shared "Pareto front size" sits below all three.
            ax.set_xlabel(f"$\\sigma$ = {sigma:g}", fontsize=F["title"],
                          color=INK, fontweight="bold")
            if not len(v):
                ax.set_xscale("log")
                ax.text(0.5, 0.5, "campaign not run", transform=ax.transAxes,
                        ha="center", va="center", fontsize=F["title"], color=INK_2)
                continue

            counts, _ = np.histogram(v, bins=edges)
            # edgecolor SURFACE, not none: it is the 2 px gap that keeps adjacent bars
            # from reading as one filled region where the bins are narrow.
            ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
                   color=color, edgecolor=SURFACE, linewidth=0.35, zorder=3)

            med = float(np.median(v))
            ax.axvline(med, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
            ax.annotate(f"median {med:,.0f}", (med, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(4, -6), textcoords="offset points", ha="left", va="top",
                        fontsize=F["annot"], color=INK,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0,
                                  alpha=0.85))
            # The tail in words, because it is one to five flights per bin on a linear
            # count axis and therefore invisible as a bar.
            ax.annotate("\n".join([f"p99 = {np.percentile(v, 99):,.0f}",
                                   f"max = {v.max():,.0f}"]),
                        # Below the median label, not beside it: at 12 pt the median
                        # text reaches across the panel and the two met in the corner.
                        (0.97, 0.82), xycoords="axes fraction", ha="right", va="top",
                        fontsize=F["annot"], color=INK_2, linespacing=1.4)
            ax.set_xscale("log")

        axes[0][0].set_ylabel("flights", fontsize=F["label"], color=INK_2)
        # Fixed, not autoscaled: three round numbers are enough to read a shape from
        # and the tail bins are counted in the corner anyway. sharey=True means
        # setting them on the first axis sets them on all three.
        axes[0][0].set_yticks([0, 200, 400])
        fig.supxlabel("Pareto front size", fontsize=F["label"], color=INK_2)
        return fig


class FrontSizeDistributionSigma2(FrontSizeDistribution):
    """The same histogram, bspline campaign at sigma = 2 only, one column wide.

    sigma = 2 is the slice the article is argued on; the three-panel figure above stays
    for the other two. Under --interp linear this figure skips itself, as the pinned
    violins do, rather than drawing the linear campaign under the same name.
    """

    stem = "front_size_distribution_sigma2"
    pin_sigma, pin_campaign = 2.0, BSPLINE_CAMPAIGNS

    def draw(self):
        fig, ax = plt.subplots(figsize=(FIG_W_COL, 2.3), facecolor=SURFACE,
                               layout="constrained")
        self.style(ax)
        color = next(c for k, _, c, _ in METHODS if k == self.key)
        v = self.df.loc[(self.df.method == self.key) & (self.df.sigma == self.pin_sigma),
                        "n_traj"].dropna().to_numpy()
        if not len(v):
            raise SystemExit(f"{self.stem}: no DP_MuO flights at sigma = {self.pin_sigma:g}")
        edges = self.bins(int(v.max()))
        counts, _ = np.histogram(v, bins=edges)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               color=color, edgecolor=SURFACE, linewidth=0.35, zorder=3)

        med = float(np.median(v))
        ax.axvline(med, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
        ax.annotate(f"median {med:,.0f}", (med, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", ha="left", va="top",
                    fontsize=FS["annot"], color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0, alpha=0.85))
        # The tail in words: one to five flights per bin, invisible as bars.
        ax.annotate("\n".join([f"n = {len(v):,}",
                               f"p99 = {np.percentile(v, 99):,.0f}",
                               f"max = {v.max():,.0f}"]),
                    (0.97, 0.80), xycoords="axes fraction", ha="right", va="top",
                    fontsize=FS["annot"], color=INK_2, linespacing=1.3)
        ax.set_xscale("log")
        ax.set_xlabel("Pareto front size (trajectories per flight)", fontsize=FS["label"],
                      color=INK_2)
        ax.set_ylabel("flights", fontsize=FS["label"], color=INK_2)
        return fig


class Scalability(Plot):
    """Solve time against problem size, one small multiple per method.

    Small multiples rather than one four-colour scatter because a scatter needs the
    all-pairs colour gate and slot 4 (yellow) against slot 2 (orange) fails it. Each panel
    only ever asks the reader to identify one colour.

    The grey reference lines -- every other method's trend repeated behind each panel --
    were removed by request. They carried the cross-method comparison a facet otherwise
    costs, so the panels are now read one at a time and the shared log axes are all that
    is left to compare them by.
    """

    section = "Computing_time"

    stem = "solve_time_vs_size"
    # The same pinned slice as TimeDistribution and MemoryDistribution -- read the first
    # of those docstrings for why. This figure already drew sigma=2 alone (the old
    # SIZE_SIGMA); what is new is that the four panels now stand on ONE population, so
    # a growth rate read off one panel and compared with another is a comparison of
    # methods rather than of populations.
    pin_sigma, pin_campaign, pin_population = 2.0, BSPLINE_CAMPAIGNS, "default"

    def trend(self, sub: pd.DataFrame, key: str):
        """Median time per size bin, and the growth rate over the upper half.

        Quantile bins, not fixed width: n_points is far from uniform and equal-width bins
        would put most flights in one bin. Medians rather than the raw cloud so the
        DP_MuO tail -- six decades inside a single bin -- does not drag the line.

        The upper half because the curve is not one straight line: DP and both OpenTOP
        modes are flat below ~130 trajectory points, where fixed setup cost dominates,
        and only then climb. Fitting the whole range averages the floor with the climb
        and understates growth by 2-3x.
        """
        v = sub[sub.method == key]
        # A campaign may be missing a method entirely -- the linear multistart is still
        # running. Ten quantile bins over nothing raises, so say so and draw the panel
        # empty rather than inventing a trend line.
        if v.n_points.notna().sum() < 10:
            return None
        g = (v.groupby(pd.qcut(v.n_points, 10), observed=True)
              .agg(x=("n_points", "median"), y=(TIME, "median")))
        x, y = g.x.to_numpy(), g.y.to_numpy()
        half = len(g) // 2
        slope = np.polyfit(np.log10(x[half:]), np.log10(y[half:]), 1)[0]
        return x, y, float(slope)

    # Matched to TimeDistribution, by request: the same three methods, the multistart in
    # the singlestart's green and named with its five starts, the same setup + solve
    # time basis, and 14 pt type throughout.
    KEYS = TimeDistribution.KEYS
    TEXT_PT = TITLE_PT = 14.0

    def draw(self):
        with type_sizes(self.TEXT_PT, self.TITLE_PT):
            return self._draw()

    def _draw(self):
        sub, label = add_time_column(self.pinned(), TimeDistribution.pin_basis)
        ot_green = next(c for k, _, c, _ in METHODS if k == "opentop_single")
        shown = [(k, n, ot_green if k == "opentop_ms" else c, sh)
                 for k, n, c, sh in METHODS if k in self.KEYS]
        shown.sort(key=lambda m: self.KEYS.index(m[0]))
        names = {**LONG_LABELS,
                 "opentop_ms": "OpenTOP Multistart" + chr(10) + "(5 starts)"}
        trends = {key: self.trend(sub, key) for key, _, _, _ in shown}
        # constrained, like the other two pinned figures: with the suptitle gone,
        # tight_layout leaves the space it had as a band of white above the panels.
        fig, axes = plt.subplots(1, len(shown), figsize=paper_size(3.5, len(shown), 4.2),
                                 sharex=True, sharey=True, facecolor=SURFACE,
                                 layout="constrained")

        for ax, (key, name, color, _) in zip(axes, shown):
            self.style(ax)
            v = sub[sub.method == key]
            # Opaque, by request: at alpha 0.12 a single dot was invisible in the pdf
            # and only the dense core of each cloud read at all. The cost, to state in
            # the caption: an opaque cloud no longer shows density -- 2 725 dots per
            # panel overplot, so the visual weight of the core is no longer proportional
            # to how many flights are in it. The trend line is the density statement now.
            ax.scatter(v.n_points, v[TIME], s=5, color=color, linewidths=0, zorder=3)
            if trends[key] is None:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{names[key]}\n(no data)", fontsize=FS["title"],
                             color=INK_2)
                ax.text(0.5, 0.5, "campaign not run", transform=ax.transAxes,
                        ha="center", va="center", fontsize=FS["title"], color=INK_2)
                continue
            x, y, slope = trends[key]
            ax.plot(x, y, color=color, lw=2.4, zorder=4, solid_capstyle="round",
                    marker="o", ms=4.5, mec=SURFACE, mew=1.0)
            # Both axes log so the fitted growth rate IS a straight line on the page --
            # otherwise the annotated number is one the reader cannot check.
            ax.set_xscale("log")
            ax.set_yscale("log")
            # LONG_LABELS, not the METHODS long name: the same four names as the two
            # violin figures, so a reader moving between the three is not asked to match
            # "DP_MuO" here against another spelling there. The OpenTOP names
            # carry a line break, so the growth rate lands on a third line.
            ax.set_title(f"{names[key]}\n(growth {slope:.2f})",
                         fontsize=FS["title"], color=INK)

        # Y CAPPED AT 1e4, by request. sharey=True, so one axis sets all four. Three
        # DP_MuO flights sit above it -- the unbounded-tail cases, up to
        # 349 188 s -- and are drawn outside the axes and therefore not visible. Without
        # the cap those three stretched the shared axis over nine decades and flattened
        # every cloud into a line. It is a CLIP, not a filter: the trend lines and the
        # fitted growth rates are computed on all 2 725 flights, this hides three dots.
        axes[0].set_ylim(top=1e4)
        axes[0].set_ylabel(f"{label} time (s)", fontsize=FS["label"], color=INK_2)
        fig.supxlabel("problem size — trajectory points", fontsize=FS["label"], color=INK_2)
        # Title and footnote both removed by request, so the caption carries: sigma = 2,
        # the bspline interpolant, the 2 725 flights complete in all four methods, the
        # sweep set when it is not "all", that the line is the median per size bin, and
        # that "growth N" is fitted on the upper half of the size range (1 = doubling
        # the size doubles the time; below ~130 points fixed setup cost dominates, which
        # is why the curve is flat there).
        return fig


def memory_frame() -> pd.DataFrame:
    """The memory table, from the campaign's csv or rebuilt from the pickles.

    Prefers the csv --memory already wrote: the figure and the printed report must show
    the same numbers, and re-reading 28 000 pickles for a figure that has a csv sitting
    next to it costs five minutes for nothing.
    """
    # The sweep set does not enter this table -- RSS is a process high-water mark and
    # the pickle is one file -- so a set that has no csv of its own borrows the "all"
    # one instead of re-reading 28 000 pickles for numbers that cannot differ.
    csv = CAMPDIR / "memory_usage.csv"
    shared = FIGDIR / "all" / CAMPDIR.name / "memory_usage.csv"
    for path in (csv, shared):
        if path.exists():
            print(f"reading {path.relative_to(FIGDIR)}  (delete it to re-read the pickles)")
            return pd.read_csv(path)
    print(f"{csv.name} not found — reading the pickles")
    df = load_memory(SIGMAS)
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    return df


class MemoryDistribution(Plot):
    """Peak RAM and result-file size per flight, in the shape of TimeDistribution.

    Two rows because the two resources rank the methods differently and a reader who
    sees only one draws the wrong conclusion: OpenTOP dominates the RAM row and is
    unremarkable on disk, while DP_MuO sits on the floor of the RAM row and
    owns the entire upper tail of the disk row. Same x positions in both, so one column
    is one method.

    RAM here is the WHOLE PROCESS: GridWeather keeps the meteo grid resident for the
    life of the task, so every violin stands on that common floor. That is the number
    that sizes a SLURM task, and it is the right one for provisioning. It is the wrong
    one for comparing campaigns that ran as different jobs, because the floor is not
    calibrated between them -- see MemoryAdded for the floor-free version.

    Disk has no such floor: it is the gzip'd pickle and nothing else.
    """

    section = "Computing_time"

    stem = "memory_and_disk"
    # "RAM (MB)", by request, on BOTH figures. The consequence to be aware of: this row
    # is the process peak WITH the meteo floor in it and MemoryAdded's is the climb
    # ABOVE that floor -- two different quantities now carrying one label, with no title
    # left to separate them either. Only the file name and the caption tell them apart.
    ram_label = "RAM (MB)"
    # The same slice as TimeDistribution, and for the same reason -- read that docstring.
    # sigma=2 is also the only sigma where this csv HAS all four methods: the DP recorded
    # 216 and 202 flights at sigma 0 and 1 against 2 728, so those two columns compared
    # four methods over populations an order of magnitude apart.
    #
    # MemoryAdded inherits the pin. Deliberately: it is this figure with the meteo floor
    # removed, the two are meant to be read side by side, and leaving them on different
    # populations would put back exactly the mismatch the pin removes.
    pin_sigma, pin_campaign, pin_population = 2.0, BSPLINE_CAMPAIGNS, "default"
    # No sweep_note(): RSS is a per-PROCESS high-water mark and the pickle is one file,
    # so neither can be attributed to one sweep of the 13 solves that produced them.
    # This figure is therefore identical under every sweep set, by construction, and
    # stamping "OpenTOP: epsilon constraint" on it would claim a split nobody measured.
    # It is redrawn into each set's folder so that folder is a complete set.
    # Titles removed by request, on this figure AND on MemoryAdded. What they said, and
    # what therefore has to go in the caption along with the slice (sigma = 2, bspline,
    # the 2 725 flights complete in all four methods): top row is the RAM held per
    # flight, bottom row the size of the result written to disk, both over the whole
    # sweep, and neither can be split by sweep. For MemoryAdded, additionally, that its
    # top row is censored -- see its docstring, and report_rows() for the counts.
    headline = None

    # Pinned to the campaign folder, not the --time folder: neither RAM nor file size
    # depends on which phases the time basis happens to sum. A property, not a class
    # attribute, for the reason written on ParetoSpread.
    @property
    def outdir(self):
        return CAMPDIR

    def __init__(self, _df: pd.DataFrame):
        # Ignores the timings frame main() hands every Plot: RSS and file size are not
        # in that cache, and load() has no reason to carry eight more columns.
        super().__init__(memory_frame())

    def ram_values(self, sub: pd.DataFrame, key: str) -> np.ndarray:
        """The RAM series to draw for one method. Overridden by MemoryAdded."""
        return sub.loc[sub.method == key, "peak_mb"].dropna().to_numpy()

    def ram_axis(self, sub: pd.DataFrame) -> np.ndarray:
        """Every RAM value DRAWN, for bounding the y axis. Overridden by MemoryAdded.

        The slice, not the whole campaign: with one sigma on the page, bounding the axis
        by three sigmas' worth of values would buy empty decades at both ends -- the
        same reason decade_ticks bounds by the data at all.
        """
        return sub.peak_mb.dropna().to_numpy()

    def draw(self):
        sub = self.pinned()
        self.report_rows(sub)
        # ONE column now -- the pinned sigma -- so the two rows are the whole figure.
        # Full width and a low per-row height for the reason TimeDistribution gives:
        # four method names at 24 pt do not fit in a 3.5 in column, and the two rows are
        # read as a pair, so each keeps that figure's landscape proportion. Constrained
        # for the same reason too: with no panel title and no suptitle there is nothing
        # for tight_layout to leave a band of white for.
        fig, axes = plt.subplots(2, 1, figsize=paper_size(4.6, 1, 4.4, span="full"),
                                 facecolor=SURFACE, squeeze=False, layout="constrained")

        for row in (0, 1):
            ax = axes[row][0]
            self.style(ax)
            counts = []
            for x, (key, name, color, _) in enumerate(METHODS):
                if row == 0:
                    v, fmt = self.ram_values(sub, key), "{:.0f}"
                else:
                    # No unit in the label: the y axis carries it, and "1516 MB"
                    # beside "1506 MB" is two labels one x-unit apart that overlap.
                    v = sub.loc[sub.method == key, "disk_kb"].dropna().to_numpy()
                    fmt = "{:.3g}"
                counts.append(len(v))
                if len(v):
                    self.violin(ax, x, v, color, fmt, **FLAT_VIOLIN)
            self.method_ticks(ax, None, LONG_LABELS)
            if row == 0:
                ax.set_xticklabels([])      # the names belong under the bottom row

        self.decade_ticks(axes[0][0], self.ram_axis(sub))
        self.decade_ticks(axes[1][0], sub.disk_kb.dropna().to_numpy())
        axes[0][0].set_ylabel(self.ram_label, fontsize=FS["label"], color=INK_2)
        axes[1][0].set_ylabel("Result per flight (kB)", fontsize=FS["label"], color=INK_2)

        if self.headline:
            fig.suptitle(fit(fig, self.headline, FS["suptitle"]),
                         fontsize=FS["suptitle"], color=INK)
        return fig

    def report_rows(self, sub: pd.DataFrame) -> None:
        """What each violin is built from, per ROW, to stdout.

        pinned() reports the flights in the slice; this reports the flights in the two
        rows, which is not the same number: RAM and disk are separate columns of the csv
        and either can be missing, and MemoryAdded's RAM row drops every flight that set
        no new high-water mark. With the counts off the page this is where it is stated.
        """
        for key, name, _, _ in METHODS:
            n_ram = len(self.ram_values(sub, key))
            n_disk = int(sub.loc[sub.method == key, "disk_kb"].notna().sum())
            print(f"    {name:24s} RAM {n_ram:>6,}   disk {n_disk:>6,}")


class MemoryAdded(MemoryDistribution):
    """The same figure with the meteo floor subtracted: setup + solve only.

    ``peak_mb - meteo_peak_mb`` is how far the process climbed AFTER the meteo slice was
    resident, i.e. what the interpolant, the graph build and the solve added between
    them. It agrees with the sum of those three phases' deltas to within a megabyte,
    and it is the only RAM number that is comparable across campaigns, because the
    floor each job happened to run with cancels out.

    WHAT THIS FIGURE NO LONGER SAYS ON THE PAGE, and what its caption therefore has to:
    RSS is a per-process high-water mark, so a flight that does not exceed what an
    earlier flight in the same task already claimed records exactly 0 above the floor.
    That is censoring, not a measurement of zero, and those flights cannot go on a log
    axis, so the top row drops them -- 475 of 2 725 for the DP, 780 for the
    DP_MuO, 1 864 for OpenTOP singlestart and 1 652 for multistart, i.e.
    between 17 and 68 % of the population. The count used to be printed under each
    method and the title used to explain it; both are gone by request, so the top row
    now reads as though it were drawn on the same 2 725 flights as the row below it and
    is not. report_rows() prints the real counts on every run. Read the row as "when a
    flight did set a new high-water mark, how far above the floor did it go".

    The disk row is identical to MemoryDistribution's -- a result file has no meteo
    component to subtract -- and is repeated so the two figures can be read side by side.
    """

    stem = "memory_and_disk_added"
    # Explicit, unlike the parent's "RAM (MB)": this is the only one of the two figures
    # going into the paper, so the axis is the last line of defence against reading 41-48
    # MB as a total footprint when the real peak is 1.3-3.2 GB, nearly all of it the
    # resident weather field.
    ram_label = "RAM above meteo floor (MB)"
    # Everything else is the parent's, by request, so the two figures are coherent: the
    # same pinned slice, the same spelled-out method names, no flight counts, no title,
    # and the same two axis labels -- "RAM (MB)" here too, even though this row is the
    # climb above the meteo floor and the parent's is the peak including it.

    @staticmethod
    def _above(sub: pd.DataFrame, key: str) -> pd.Series:
        v = sub[sub.method == key]
        return (v.peak_mb - v.meteo_peak_mb).dropna()

    def ram_values(self, sub: pd.DataFrame, key: str) -> np.ndarray:
        a = self._above(sub, key)
        return a[a > 0].to_numpy()

    def ram_axis(self, sub: pd.DataFrame) -> np.ndarray:
        a = (sub.peak_mb - sub.meteo_peak_mb).dropna()
        return a[a > 0].to_numpy()


# ── joint Pareto front across all four solvers ───────────────────────────────
# Every converged trajectory the four methods produced for one flight, pooled, and
# only the points nothing else dominates kept. The colour of a surviving point is the
# method that produced it, so the figure answers "who actually owns the frontier"
# rather than "how does each method's own front look".
#
# Two normalisations make the population plottable on one pair of axes:
#
#   x   extra fuel over that flight's fuel-only warm start, in per cent. ONE reference
#       per flight, taken from the OpenTOP SINGLESTART fuel_only solve -- the warm start
#       the campaign actually ran. It has to be shared: points can only be pooled into a
#       joint front if they sit on the same fuel scale. The consequence, which belongs in
#       the caption: any systematic fuel offset between solvers (DP carries a fixed mass
#       profile and a different discretisation) shows up as a horizontal shift instead of
#       being hidden inside a per-method normalisation. That is the honest behaviour for
#       a joint front and the wrong one for a per-method plot.
#
#   y   contrail cost, raw. Not normalised, because a ratio divides by ~0 on the cheapest
#       trajectory of some flights. At sigma=2 that is nearly moot -- 3 flights of 2822
#       go negative, minimum -0.66 -- but at sigma=0 it is 44.6 % of OpenTOP flights, so
#       the raw axis is the one that survives a change of sigma.
#
# The DP_MuO tail is dropped, not capped: 407837_06141_0 and 43be8d_06988_0 return
# 50 480 and 20 958 trajectories, which is 25 % of every point in the figure from 2 of
# 2822 flights. Keeping them would let two flights colour the frontier. They are the
# same pair reported as right-censored everywhere else in this campaign; PARETO_MAX_FRONT
# is the knob, and the run prints exactly which flights it removed.
PARETO_MAX_FRONT = 10_000       # drop a flight whose pooled point count exceeds this


def dominance_path(sigma: float, methods=None) -> Path:
    """Where the tally written beside a joint-front cache lives."""
    tag = "" if methods is None else "_" + "_".join(sorted(methods))
    return CAMPDIR / f"joint_front_sigma{sigma:g}{tag}{POPTAG}{FULLTAG}_dominates.csv"


def dominance_tally(df: pd.DataFrame) -> pd.DataFrame:
    """Per ordered method pair: how many of the other's POOLED points it dominates.

    Pareto dominance, per flight: (f_a, c_a) beats (f_b, c_b) when f_a <= f_b,
    c_a <= c_b and at least one is strict. Counted over the pooled points rather than
    the non-dominated ones on purpose -- a point that was itself beaten still says what
    the method that produced it was able to beat, and that is the question the legend
    of the multistart figure asks.

    O(n log n) per flight, like pareto_mask: sort one method by fuel, take the running
    minimum of its contrail cost, then look every point of the other method up in it.
    The two searchsorted sides are the two ways of being strict.
    """
    rows = []
    for fid, g in df.groupby("flight_id", sort=False):
        arrs = {m: (sub.fuel_kg.to_numpy(dtype=float),
                    sub.contrail.to_numpy(dtype=float))
                for m, sub in g.groupby("method", sort=False)}
        for a, (fa, ca) in arrs.items():
            order = np.argsort(fa, kind="stable")
            fs, pref = fa[order], np.minimum.accumulate(ca[order])
            for b, (fb, cb) in arrs.items():
                if b == a:
                    continue
                le = np.searchsorted(fs, fb, side="right")   # how many f_a <= f_b
                lt = np.searchsorted(fs, fb, side="left")    # how many f_a <  f_b
                best_le = np.where(le > 0, pref[np.clip(le - 1, 0, None)], np.inf)
                best_lt = np.where(lt > 0, pref[np.clip(lt - 1, 0, None)], np.inf)
                beaten = (best_le < cb) | (best_lt <= cb)
                rows.append((fid, a, b, int(beaten.sum()), int(cb.size)))
    per_pair = pd.DataFrame(rows, columns=["flight_id", "method", "other",
                                           "points_dominated", "other_points"])
    return (per_pair.groupby(["method", "other"], as_index=False)
            .agg(points_dominated=("points_dominated", "sum"),
                 other_points=("other_points", "sum"),
                 flights=("flight_id", "nunique")))


def complete_only(df: pd.DataFrame, methods) -> pd.DataFrame:
    """Keep only the flights where EVERY pooled method delivered its full sweep.

    front_arrays has already dropped the failed solves, so a flight short of the
    expected count is one whose sweep did not finish. Applied to the POOLED points --
    before the mask and before the dominance tally -- because a half-finished sweep
    changes both what a method owns and what it beat, and gating one but not the other
    would put two different populations in the same legend.
    """
    want = [m for m, _, _, _ in METHODS if methods is None or m in methods]
    n = df.groupby(["flight_id", "method"]).size().unstack("method")
    ok = pd.Series(True, index=n.index)
    for m in want:
        # Absent method -> NaN column: the flight has no point at all from it, which is
        # not complete under either rule.
        col = n[m] if m in n.columns else pd.Series(np.nan, index=n.index)
        exp = expected_solves(m)
        ok &= col.notna() if exp is None else (col == exp)
        short = int((~(col.notna() if exp is None else (col == exp))).sum())
        print(f"  --full-only: {short:,} flights are short of "
              + (f"{exp} converged {m} solves" if exp else f"a {m} front"))
    keep = ok[ok].index
    print(f"  --full-only: {len(keep):,} of {len(ok):,} flights are complete in all "
          f"{len(want)} methods")
    return df[df.flight_id.isin(set(keep))]


def tie_flags(pooled: pd.DataFrame, nd: pd.DataFrame) -> np.ndarray:
    """Per surviving point: did another method produce the SAME (fuel, contrail)?

    Bit-identical, not within a tolerance: two solvers landing on the same optimum
    write the same float, and anything looser would silently call "nearly the same"
    the same, which is a judgement this figure should not make on its own.

    It matters because pareto_mask keeps the FIRST of two identical points and drops
    the rest, so a tie is credited in full to whichever method SOURCES reads first --
    on the front counts alone it reads as a win that method did not have.
    """
    key = ["flight_id", "fuel_kg", "contrail"]
    shared = pooled.groupby(key, sort=False).method.nunique()
    return nd.set_index(key).index.map(shared.gt(1)).to_numpy()


def joint_front_frame(sigma: float, methods=None,
                      max_front: int = PARETO_MAX_FRONT,
                      with_dominance: bool = False,
                      with_ties: bool = False) -> pd.DataFrame:
    """One row per NON-DOMINATED point of the pooled four-method front, per flight.

    Cached: this reads every pickle of all four campaigns and the DP_MuO fronts are
    large, so it is minutes of work for a figure that is otherwise redrawn in seconds.

    ``with_dominance`` also writes the tally beside the cache. It needs the POOLED
    points, which the cache does not keep, so a run that wants it and finds no tally
    re-reads the pickles even when the front cache is there -- the alternative is a
    legend silently drawn from a stale count.
    """
    tag = "" if methods is None else "_" + "_".join(sorted(methods))
    cache = CAMPDIR / f"joint_front_sigma{sigma:g}{tag}{POPTAG}{FULLTAG}.csv"
    tally = dominance_path(sigma, methods)
    if cache.exists() and (tally.exists() or not with_dominance):
        cached = pd.read_csv(cache)
        if not with_ties or "identical" in cached.columns:
            print(f"reading {cache.name}  (delete it to re-read the pickles)")
            return _in_population(cached, cache.name)
        print(f"re-reading the pickles: {cache.name} predates the identical column")
    if cache.exists():
        print(f"re-reading the pickles: {cache.name} has no {tally.name} beside it")

    rows, base = [], {}
    for method, folder, pattern in SOURCES:
        if methods is not None and method not in methods:
            continue
        n = 0
        for f in sorted(folder.glob(pattern.replace("[0-9]", f"{sigma:g}"))):
            with gzip.open(f, "rb") as fh:
                payload = pickle.load(fh)
            if payload["error"]:
                continue
            ctx = payload["context"]
            fid = ctx.get("flight_id")
            if not admitted(fid) or float(ctx.get("sigma")) != float(sigma):
                continue
            if method == "opentop_single":
                # The shared x reference. Only a CONVERGED fuel-only solve may serve as
                # it: a failed one is IPOPT's last iterate, and dividing the whole
                # population's x axis by a non-optimum would be silent nonsense.
                ok = [t for t in payload["results"].get("fuel_only", ())
                      if t.get("success") and t.get("fuel_kg")]
                if ok:
                    base[fid] = float(ok[0]["fuel_kg"])
            fuel, cont, sweeps = front_arrays(payload, method, with_sweep=True)
            for a, b, sw in zip(fuel, cont, sweeps):
                rows.append((fid, method, a, b, sw))
            n += 1
        print(f"  {method:16s} {n:>6d} pickles from {folder.relative_to(BENCH.parent)}")

    df = pd.DataFrame(rows, columns=["flight_id", "method", "fuel_kg",
                                     "contrail", "sweep"])
    # Both axes are raw, so no flight is dropped for lacking a reference. fuel_pct is
    # still carried where a converged fuel-only solve exists, for anyone who wants the
    # normalised view from the same cache, and is NaN for the rest.
    print(f"  {df.flight_id.nunique() - len(set(df.flight_id) & set(base))} flights "
          f"have no converged fuel-only solve (kept; their fuel_pct is empty)")

    if FULL_ONLY:
        df = complete_only(df, methods)

    big = df.groupby("flight_id").size()
    drop = set(big[big > max_front].index)
    if drop:
        print(f"  {len(drop)} flights dropped: pooled front over {max_front:,} points "
              f"({', '.join(sorted(drop))})")
        df = df[~df.flight_id.isin(drop)]

    if with_dominance:
        dom = dominance_tally(df)
        tally.parent.mkdir(parents=True, exist_ok=True)
        dom.to_csv(tally, index=False)
        print(f"  wrote {tally.name}")
        for _, r in dom.iterrows():
            print(f"    {r.method:16s} dominates {r.points_dominated:>8,d} of "
                  f"{r.other_points:>8,d} {r.other} points")

    keep = []
    for fid, g in df.groupby("flight_id", sort=False):
        m = pareto_mask(g.fuel_kg.to_numpy(), g.contrail.to_numpy())
        keep.append(g[m])
    nd = pd.concat(keep, ignore_index=True)
    nd["identical"] = tie_flags(df, nd)
    print(f"  {int(nd.identical.sum()):,} of them were produced identically by more "
          f"than one method")
    nd["fuel_pct"] = (nd.fuel_kg / nd.flight_id.map(base) - 1) * 100
    # The neutral reference: the cheapest trajectory ANY of the pooled methods found for
    # that flight. Unlike fuel_pct it privileges no solver -- fuel_pct divides by
    # OpenTOP's own warm start, which hands OpenTOP the origin by construction. Taken
    # over the pooled points before the mask, so it is the same number for every method.
    cheapest = df.groupby("flight_id").fuel_kg.min()
    nd["fuel_over_min_pct"] = (nd.fuel_kg / nd.flight_id.map(cheapest) - 1) * 100
    print(f"  {len(nd):,} non-dominated of {len(df):,} pooled points "
          f"over {nd.flight_id.nunique():,} flights")

    cache.parent.mkdir(parents=True, exist_ok=True)
    nd.to_csv(cache, index=False)
    return nd


def head_to_head_frame(sigma: float = 2.0) -> pd.DataFrame:
    """Per (flight, sweep, sweep point): the objective each OpenTOP mode reached.

    The two modes solve the SAME problem at each sweep point -- the same weighted sum,
    or minimum contrail under the same fuel slack -- so this is the only comparison in
    the campaign that is head to head rather than front against front. The objective is
    the one that solve was asked to minimise, rebuilt from the published pair:

        weighted sum        J = coef * contrail + (1 - coef) * fuel
        epsilon-constraint  J = contrail            (the budget is the constraint)

    Failed solves are dropped, so a flight short of its 13 has fewer rows; --full-only
    removes those flights entirely, which is what the figure is drawn on.
    """
    cache = CAMPDIR / f"head_to_head_sigma{sigma:g}{POPTAG}{FULLTAG}.csv"
    if cache.exists():
        cached = pd.read_csv(cache)
        if "dp" in cached.columns:
            print(f"reading {cache.name}  (delete it to re-read the pickles)")
            return _in_population(cached, cache.name)
        print(f"re-reading the pickles: {cache.name} predates the dp column")

    rows, n_ok = [], {}
    for method, folder, pattern in SOURCES:
        # The DP joins for its weighted sums only: it has no eps-constraint sweep of its
        # own in this campaign, so its rows are NaN there and every consumer says which
        # columns it needs.
        if method not in ("opentop_single", "opentop_ms", "dp"):
            continue
        for f in sorted(folder.glob(pattern.replace("[0-9]", f"{sigma:g}"))):
            with gzip.open(f, "rb") as fh:
                payload = pickle.load(fh)
            if payload["error"]:
                continue
            ctx = payload["context"]
            fid = ctx.get("flight_id")
            if not admitted(fid) or float(ctx.get("sigma")) != float(sigma):
                continue
            n = 0
            for sweep in sweeps_of(method):
                for t in payload["results"].get(sweep, ()):
                    if not t.get("success"):
                        continue
                    n += 1
                    meta = t.get("meta") or {}
                    if sweep == "weighted_sum":
                        coef = meta.get("coef")
                        if coef is None:
                            continue
                        obj = (float(coef) * float(t["contrail"])
                               + (1 - float(coef)) * float(t["fuel_kg"]))
                        point = float(coef)
                    elif sweep == "epsilon_constraint":
                        point = meta.get("fuel_slack_frac")
                        if point is None:
                            continue
                        obj, point = float(t["contrail"]), float(point)
                    else:
                        continue        # the fuel-only anchor has no sweep coordinate
                    rows.append((fid, method, sweep, point, obj))
            n_ok.setdefault(fid, {})[method] = n

    df = pd.DataFrame(rows, columns=["flight_id", "method", "sweep", "point", "obj"])
    if FULL_ONLY:
        # 13 = the whole sweep, fuel-only included: a mode that lost a solve anywhere
        # is a mode whose run did not finish, whichever point the loss fell on.
        full = {f for f, v in n_ok.items()
                if all(v.get(m) == expected_solves(m)
                       for m in ("opentop_single", "opentop_ms", "dp"))}
        print(f"  --full-only: {len(full):,} of {len(n_ok):,} flights complete in both modes")
        df = df[df.flight_id.isin(full)]

    # dropna on the two OpenTOP columns only: the DP column is empty on every
    # eps-constraint row by construction, and dropping those would silently halve the
    # frame the two-mode figure is drawn on.
    wide = (df.pivot_table(index=["flight_id", "sweep", "point"],
                           columns="method", values="obj")
              .dropna(subset=["opentop_single", "opentop_ms"]).reset_index())
    print(f"  {len(wide):,} head-to-head pairs over {wide.flight_id.nunique():,} flights")
    cache.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(cache, index=False)
    return wide


# ── each method's OWN Pareto front ───────────────────────────────────────────
# The same points as the joint front, masked differently and normalised differently.
# Both differences are the whole content of the figure, so neither is implicit:
#
#   MASK   per (flight, METHOD) instead of over the four pooled. A point survives here
#          if the method that produced it found nothing better ITSELF. That is what a
#          per-method front is; joint_front_frame asks the other question, whether any
#          of the four beat it, and answers "who owns the frontier".
#
#   x      extra fuel over the FUEL-ONLY trajectory, in per cent -- and which fuel-only
#          trajectory is a real choice, not a detail:
#
#          own   that method's own fuel-only solve. Each front then starts at its own
#                origin, so the shape of one solver's trade-off is not tilted by
#                another solver's mass profile or discretisation. This is the
#                normalisation the comment above joint_front_frame calls the right one
#                for a per-method plot, and it is the default here.
#          ws    the OpenTOP SINGLESTART fuel-only solve -- the warm start the campaign
#                actually ran -- shared by all four. Systematic fuel offsets between
#                solvers then show up as a horizontal shift instead of being normalised
#                away, which is what you want if the four figures are to be read as one
#                scale. It hands OpenTOP the origin by construction; say so if you use it.
#
#          The choice is in the FILE NAME, because nothing inside a scatter reveals which
#          reference it was divided by and the two look equally plausible.
#
#   y      contrail cost, raw, for the reason given above joint_front_frame: a ratio
#          divides by ~0 on the cheapest trajectory of some flights.

def fuel_only_ref(payload: dict) -> float | None:
    """The flight's fuel-optimal fuel burn according to THIS pickle, or None.

    Read straight from the ``fuel_only`` sweep rather than through ``front_arrays``,
    because the reference is a property of the flight and must not move when
    ``--sweeps`` changes which sweeps the figure draws. Only a CONVERGED solve counts:
    a failed one is IPOPT's last iterate, and dividing an axis by a non-optimum is
    silent nonsense.
    """
    ok = [float(t["fuel_kg"]) for t in payload["results"].get("fuel_only", ())
          if t.get("success") and t.get("fuel_kg")]
    return min(ok) if ok else None


def own_front_frame(sigma: float, max_front: int = PARETO_MAX_FRONT) -> pd.DataFrame:
    """One row per point of each method's own non-dominated front, per flight.

    Cached next to the joint one and read the same way: this walks every pickle of all
    four campaigns, which is minutes, for figures that are then redrawn in seconds.
    """
    cache = CAMPDIR / f"own_front_sigma{sigma:g}{POPTAG}.csv"
    if cache.exists():
        print(f"reading {cache.name}  (delete it to re-read the pickles)")
        return _in_population(pd.read_csv(cache), cache.name)

    rows, own_base, ws_base = [], {}, {}
    for method, folder, pattern in SOURCES:
        n = 0
        for f in sorted(folder.glob(pattern.replace("[0-9]", f"{sigma:g}"))):
            with gzip.open(f, "rb") as fh:
                payload = pickle.load(fh)
            if payload["error"]:
                continue
            ctx = payload["context"]
            fid = ctx.get("flight_id")
            if not admitted(fid) or float(ctx.get("sigma")) != float(sigma):
                continue
            ref = fuel_only_ref(payload)
            if ref is not None:
                own_base[(fid, method)] = ref
            if method == "opentop_single" and ref is not None:
                ws_base[fid] = ref
            fuel, cont, sweeps = front_arrays(payload, method, with_sweep=True)
            for a, b, sw in zip(fuel, cont, sweeps):
                rows.append((fid, method, a, b, sw))
            n += 1
        print(f"  {method:16s} {n:>6d} pickles from {folder.relative_to(BENCH.parent)}")

    df = pd.DataFrame(rows, columns=["flight_id", "method", "fuel_kg",
                                     "contrail", "sweep"])

    # Per (flight, method), not per flight: DP_MuO is the only method
    # that can blow past the cap, and dropping a flight from the other three because of
    # it would silently shrink their populations to match its worst case.
    size = df.groupby(["flight_id", "method"]).size()
    drop = set(size[size > max_front].index)
    if drop:
        print(f"  {len(drop)} (flight, method) fronts dropped, over {max_front:,} points: "
              + ", ".join(f"{fid}/{m}" for fid, m in sorted(drop)))
        df = df[~pd.MultiIndex.from_frame(df[["flight_id", "method"]]).isin(drop)]

    keep = []
    for _, g in df.groupby(["flight_id", "method"], sort=False):
        keep.append(g[pareto_mask(g.fuel_kg.to_numpy(), g.contrail.to_numpy())])
    nd = pd.concat(keep, ignore_index=True)

    # The DP_MuO runs no fuel-only sweep -- one label-setting solve IS its
    # whole front -- so its own reference is the cheapest point of that front, which is
    # the same trajectory a fuel-only sweep would have returned. Taken on the front
    # BEFORE the mask so it cannot depend on which points survived.
    cheapest = df.groupby(["flight_id", "method"]).fuel_kg.min()
    pairs = pd.MultiIndex.from_frame(nd[["flight_id", "method"]])
    from_sweep = pd.Series(pairs.map(own_base), index=nd.index)
    own = from_sweep.fillna(pd.Series(pairs.map(cheapest), index=nd.index))
    # Which of the two the reference actually came from, per row. Carried rather than
    # inferred later: a campaign run with --sweeps weighted_sum has no fuel_only sweep
    # at all, so EVERY row falls back, and an axis still labelled "fuel-only" would be
    # quietly wrong. The figure reads this and relabels itself.
    nd["ref_fuel_only"] = from_sweep.notna().to_numpy()
    nd["fuel_pct_own"] = (nd.fuel_kg / own - 1) * 100
    nd["fuel_pct_ws"] = (nd.fuel_kg / nd.flight_id.map(ws_base) - 1) * 100

    missing = nd.flight_id.nunique() - len(set(nd.flight_id) & set(ws_base))
    print(f"  {len(nd):,} non-dominated of {len(df):,} points over "
          f"{nd.flight_id.nunique():,} flights")
    print(f"  {missing} flights have no converged OpenTOP fuel-only solve "
          f"(kept; their fuel_pct_ws is empty)")

    cache.parent.mkdir(parents=True, exist_ok=True)
    nd.to_csv(cache, index=False)
    return nd


class JointPareto(Plot):
    """The frontier the four solvers jointly define, coloured by who owns each point.

    A point is drawn only if no point from ANY method, on the same flight, beats it on
    both axes. So a method's colour appearing at all means it produced something none of
    the other three could match -- which is a stronger claim than any per-method front
    comparison, and the reason the four are pooled before the mask rather than after.
    """

    section = "Optimality"

    stem = "joint_pareto_front"
    sigma = 2.0
    methods = None          # None -> every solver in SOURCES
    marker = dict(s=2.5, alpha=0.22)
    with_dominance = False  # subclasses that report what each method beat set this
    with_ties = False       # subclasses that draw the shared points apart set this
    tie_colour = INK       # black: neutral against every method colour on this surface
    legend_title = "owns the joint front"
    beats_note = "points no other solver beats on both axes"
    # The x reference is fuel_over_min_pct: the cheapest POOLED point of that flight.
    # Over the four solvers that is not always a fuel-only solve, so the default label
    # names what is actually divided by; the two-mode figure overrides it.
    xlabel = "extra fuel over the cheapest trajectory found for that flight (%)"
    # The contrail axis, and what it is drawn in. y_div rescales the DATA and the label
    # says so, rather than matplotlib's offset text: a "1e2" parked above the axis is
    # read as part of the top tick as often as not, and it disappears when the figure
    # is cropped into the paper.
    ylabel = "contrail cost"
    y_div = 1.0
    # Where the legend sits and how big its dots are. Both per figure: the corner that
    # is empty depends on how far the population spreads along x, and three entries of
    # 2.5 pt dots need a bigger swatch than four of them do.
    legend_loc = "upper left"
    legend_markerscale = 2.2
    # An inset that repeats the same series over a window of the same axes, or None.
    # (x_hi, y_hi, [left, bottom, width, height]) -- the window starts at the axis
    # minimum, so only its far corner is a choice, and the rectangle is in axes
    # fractions. y_hi is in the units of the axis, i.e. AFTER y_div.
    inset = None
    inset_marker = None     # None -> the main marker, at the same size
    inset_arrow = True      # one leader from the window to the inset
    titled = True           # False drops the suptitle; the caption then carries it
    # Per-figure colour overrides. METHODS is the campaign-wide encoding and stays as
    # it is: a colour changed there would silently repaint every other figure, and the
    # four-solver ones need the amber for multistart against the orange of DP_MuO.
    recolour: dict = {}

    @property
    def outdir(self):
        return CAMPDIR      # not a --time folder: no time basis enters this figure

    def __init__(self, _df: pd.DataFrame):
        super().__init__(joint_front_frame(self.sigma, self.methods,
                                           with_dominance=self.with_dominance,
                                           with_ties=self.with_ties))
        # Instance attribute over the class one: the restricted figure must not
        # overwrite the full-population figure of the same name.
        self.stem = self.stem + POPTAG + FULLTAG

    def legend_label(self, key: str, name: str, nfl: int, npts: int) -> str:
        """One legend entry. Overridden where the count is not just what was drawn."""
        return f"{name}  —  {nfl:,} flights, {npts:,} pts"

    def series(self):
        """(colour, label, rows) per scatter, most numerous first.

        A hook rather than the loop inline: the two-mode figure splits a third series
        out of the same frame, and everything else about the axes is shared.
        """
        counts = {k: int((self.df.method == k).sum()) for k, _, _, _ in METHODS
                  if (self.df.method == k).any()}
        # Most numerous first, so a method with few points is not buried under one with
        # many. DP_MuO alone owns 86 % of the points.
        for key in sorted(counts, key=counts.get, reverse=True):
            name, colour = next((n, c) for k, n, c, _ in METHODS if k == key)
            v = self.df[self.df.method == key]
            yield (self.recolour.get(key, colour),
                   self.legend_label(key, name, v.flight_id.nunique(), counts[key]), v)

    def zoom(self, ax):
        """Repeat the scatter over a window of the same axes, inside the panel.

        Drawn from series() like the main axes, so the inset can never end up showing a
        different split than the legend it sits under. The window is only a change of
        limits: no point is recomputed, and nothing outside it is redrawn -- the mask
        that decided who owns what was taken over the whole population, not over this
        corner of it.
        """
        x_hi, y_hi, rect = self.inset
        axin = ax.inset_axes(rect)
        self.style(axin)
        for colour, _, v in self.series():
            axin.scatter(v.fuel_over_min_pct, v.contrail / self.y_div, color=colour,
                         **(self.inset_marker or self.marker), linewidths=0, zorder=3)
        # Same small margin off the origin as the main axes, so the two read alike.
        axin.set_xlim(-0.02 * x_hi, x_hi)
        axin.set_ylim(-0.02 * y_hi, y_hi)
        axin.tick_params(labelsize=FS["tick"], colors=INK_2)
        # The window marked on the main axes, without the connector lines: it sits over
        # the dense corner, and two leaders crossing that cloud cost more than they say.
        _, connectors = ax.indicate_inset_zoom(axin, edgecolor=INK_2, alpha=0.55,
                                               lw=0.8)
        for c in connectors:
            c.set_visible(False)
        if self.inset_arrow:
            # ONE leader, from the far corner of the window to the near corner of the
            # inset, curved away from the cloud. The pair matplotlib draws would run
            # from two corners straight through the densest part of the scatter.
            ax.annotate("", xy=rect[:2], xycoords="axes fraction",
                        xytext=(x_hi, y_hi), textcoords="data", zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.0,
                                        alpha=0.75, shrinkA=3, shrinkB=3,
                                        connectionstyle="arc3,rad=-0.25"))
        return axin

    def draw(self):
        # span="full": one panel would otherwise be squeezed into a single column,
        # which is 3.6 in for a scatter carrying a quarter of a million points.
        fig, ax = plt.subplots(figsize=paper_size(9.2, 1, 5.6, span="full"),
                               facecolor=SURFACE, layout="constrained")
        self.style(ax)

        # Most numerous first, so a method with few points is not buried under one with
        # many. DP_MuO alone owns 86 % of the points.
        for colour, label, v in self.series():
            ax.scatter(v.fuel_over_min_pct, v.contrail / self.y_div, color=colour,
                       **self.marker, linewidths=0, zorder=3, label=label)

        # Linear x now: normalised per flight, the axis is a few per cent wide instead
        # of three decades of aircraft size, so the trade-off shape is visible and no
        # log is needed. Clipped at the 99th percentile -- a few flights run far out and
        # would flatten everything else against the origin.
        hi = np.percentile(self.df.fuel_over_min_pct, 99)
        ax.set_xlim(-hi * 0.02, hi)
        ax.set_xlabel(self.xlabel, fontsize=FS["label"], color=INK_2)
        ax.set_ylabel(self.ylabel, fontsize=FS["label"], color=INK_2)
        # y stays LINEAR. It spans 0 to ~3 550, not three decades, so it does not need a
        # log axis -- and log on both axes deforms the shape of the trade-off, which is
        # the one thing a Pareto figure exists to show. The 19 points at or below zero
        # (0.05 %, minimum -0.66, numerical dust in the OpenTOP interpolant) simply sit
        # just under the origin instead of forcing a symlog.
        top, bot = self.df.contrail.max() / self.y_div, self.df.contrail.min() / self.y_div
        ax.set_ylim(min(-0.05 * top, bot * 1.2), top * 1.04)

        # Inside the axes, in whichever corner the cloud leaves empty: an outside legend
        # was taking ~40 % of the figure width from the axes it was supposed to explain.
        leg = ax.legend(loc=self.legend_loc, frameon=True, fontsize=FS["legend"],
                        markerscale=self.legend_markerscale, labelspacing=0.5,
                        handletextpad=0.4,
                        borderpad=0.5, title=self.legend_title)
        if self.legend_title:
            leg.get_title().set_fontsize(FS["legend"])
        leg.get_frame().set(facecolor=SURFACE, edgecolor=RULE, linewidth=0.5, alpha=0.9)
        for h in leg.legend_handles:
            h.set_alpha(1.0)

        if self.inset:
            self.zoom(ax)

        # No y= with constrained layout: the engine reserves the space itself, and
        # pinning y=1.0 drops the title on top of the axes.
        pop = ("of a population of " + f"{len(POPULATION):,}" if POPULATION is not None
               else "of")
        if not self.titled:
            return fig
        fig.suptitle(fit(fig, f"Joint Pareto front, $\sigma$ = {self.sigma:g}  —  {self.beats_note}  "
                     f"({self.df.flight_id.nunique():,} flights {pop})"
                     + sweep_note(self.methods), FS["suptitle"]),
                     fontsize=FS["suptitle"], color=INK)
        return fig


class JointParetoNoMO(JointPareto):
    """The same joint front without DP_MuO.

    Dropping it removes the cardinality problem rather than working around it: DP and
    both OpenTOP modes return exactly 13 trajectories per flight, so no flight can
    swamp the figure and PARETO_MAX_FRONT never fires -- the whole 2 822 are in, and
    the "points owned" counts finally compare like with like instead of measuring how
    much output each method happens to produce.
    """

    stem = "joint_pareto_front_3solvers"
    methods = ("opentop_single", "opentop_ms", "dp")
    marker = dict(s=5.0, alpha=0.35)     # 20x fewer points than the four-solver version


class JointParetoMultiVsSingle(JointPareto):
    """OpenTOP multistart against singlestart, the two of them alone.

    Same mask as the joint figure, over a pool of two: a point is drawn only if the
    OTHER mode found nothing that beats it on both fuel and contrail cost, on the same
    flight. So the picture is the frontier the two modes jointly define and the colours
    say which mode owns each piece of it -- the direct answer to whether multistart is
    finding trajectories the single start cannot.

    The legend carries a second number the four-solver figures do not: how many of the
    other mode's POOLED points each one dominates. The front count alone is symmetric
    in a misleading way (both modes return 13 solves per flight, so neither can own many
    more points than the other), while the dominated count says by how much the winner
    wins -- a mode can own few points and still beat almost everything the other
    produced. Both are in the legend because either alone reads as the whole story.
    """

    stem = "joint_pareto_front_ms_vs_ss"
    methods = ("opentop_single", "opentop_ms")
    marker = dict(s=6.0, alpha=0.40)     # two 13-point sweeps per flight, not thousands
    with_dominance = True   # the tally csv stays a campaign output
    with_ties = True
    legend_title = None     # three entries, and the grey one owns nothing
    beats_note = "points the other mode does not beat on both axes"
    # The campaign encoding, unchanged: green singlestart, amber multistart, as in
    # every other figure of the set.
    recolour: dict = {}
    # With only the two OpenTOP modes pooled, the cheapest point of a flight is one of
    # the two fuel-only solves in all but rounding, so the axis can be named after it.
    xlabel = "extra fuel over fuel-only (%)"
    # Divided by 100 and said in the label: the raw axis runs to ~3 550 and five-digit
    # ticks crowd a column-width panel.
    ylabel = "contrail cost ($10^2$)"
    y_div = 100.0
    # Upper right: with only the two OpenTOP modes pooled the cloud hugs the left edge,
    # so the empty corner is the far one. Bigger dots because three entries of 6 pt
    # markers still read as specks at print size.
    legend_loc = "upper right"
    legend_markerscale = 4.5
    # Right-hand side, under the legend: the first 5 % of extra fuel and the first
    # 2 000 of contrail cost, which is where the two modes actually overlap.
    inset = (5.0, 20.0, [0.56, 0.40, 0.42, 0.36])
    inset_marker = dict(s=4.0, alpha=0.30)
    titled = False          # the two-mode figure goes in the text with a caption

    def series(self):
        """Three series: what each mode alone holds, and what both found identically.

        The shared points come OUT of the mode that happens to hold them instead of
        being drawn twice: pareto_mask already dropped the second copy of each identical
        pair, so leaving them in would credit every tie to whichever method SOURCES
        reads first -- singlestart -- and read as a win it did not have.
        """
        own = self.df[~self.df.identical]
        both = self.df[self.df.identical]
        counts = {k: int((own.method == k).sum()) for k, _, _, _ in METHODS
                  if (own.method == k).any()}
        out = [(self.recolour.get(k, next(c for j, _, c, _ in METHODS if j == k)),
                f"{next(n for j, n, _, _ in METHODS if j == k)}: {counts[k]:,} points",
                own[own.method == k])
               for k in sorted(counts, key=counts.get, reverse=True)]
        out.append((self.tie_colour, f"Identical points: {len(both):,}", both))
        return out


class JointParetoOpenTOPvsDP(JointPareto):
    """OpenTOP against the DP, with the two OpenTOP modes pooled into one series.

    The question here is not which OpenTOP mode is better -- that is the ms-vs-ss
    figure -- but whether the direct-collocation solver reaches trajectories the DP
    cannot, so the two modes enter as ONE competitor: their points are pooled before
    the mask, and a point survives when neither the other mode nor the DP beats it.
    Pooling before the mask, not after, is what makes the green count "what OpenTOP
    can do at its best" rather than "what one of its two modes happened to do".

    The DP campaign is the ws-only run: 6 weighted sums per flight against OpenTOP's
    2 x 13, so the raw counts are not a like-for-like measure of effort. What they do
    measure is how much of the joint frontier each solver holds, which is the
    question a Pareto figure is for.
    """

    stem = "joint_pareto_front_opentop_vs_dp"
    methods = ("opentop_single", "opentop_ms", "dp")
    marker = dict(s=5.0, alpha=0.35)
    with_dominance = True   # writes the per-pair tally beside the cache
    legend_title = None
    titled = False
    ylabel = "contrail cost ($10^2$)"
    y_div = 100.0
    legend_loc = "upper right"
    legend_markerscale = 4.5
    inset = (5.0, 20.0, [0.56, 0.40, 0.42, 0.36])
    inset_marker = dict(s=3.0, alpha=0.25)
    # x stays the base label: with the DP in the pool the cheapest pooled point is not
    # always an OpenTOP fuel-only solve, so "over fuel-only" would be a claim about
    # which solver defines the origin, and it is not always the same one.

    def series(self):
        ot = self.df[self.df.method.isin(("opentop_single", "opentop_ms"))]
        dp = self.df[self.df.method == "dp"]
        out = [("#1baf7a", f"OpenTOP (single + multistart): {len(ot):,} points", ot),
               ("#2a78d6", f"Dynamic Programming: {len(dp):,} points", dp)]
        return sorted(out, key=lambda r: len(r[2]), reverse=True)


class JointParetoOpenTOPvsBothDPs(JointParetoOpenTOPvsDP):
    """The same figure with the DP_MuO added as a third competitor.

    Three series, one mask: OpenTOP's two modes pooled, the weighted-sum DP, and
    DP_MuO, the label-setting solver. DP_MuO returns a whole label set
    rather than a fixed grid, so its point count per flight is a property of the flight
    -- its share of the front says how much of the frontier it reaches, never how much
    work it did, and the two must not be read as the same number.
    """

    stem = "joint_pareto_front_opentop_vs_dp_mo"
    methods = ("opentop_single", "opentop_ms", "dp", "dp_muo")
    marker = dict(s=4.0, alpha=0.30)
    inset_marker = dict(s=2.5, alpha=0.22)

    def series(self):
        ot = self.df[self.df.method.isin(("opentop_single", "opentop_ms"))]
        dp = self.df[self.df.method == "dp"]
        mo = self.df[self.df.method == "dp_muo"]
        out = [("#1baf7a", f"OpenTOP (single + multistart): {len(ot):,} points", ot),
               ("#2a78d6", f"Dynamic Programming: {len(dp):,} points", dp),
               ("#eb6834", f"DP_MuO: {len(mo):,} points", mo)]
        return sorted(out, key=lambda r: len(r[2]), reverse=True)


def perturbation_frame(sigma: float = 2.0) -> pd.DataFrame:
    """Per (flight, sweep, sweep point): start 0 against the best of starts 1-4.

    Read from the MULTISTART pickles, whose meta["candidates"] keeps every start of
    every sweep point with its own fuel and contrail. Start 0 is the fuel-only guess as
    given (the singlestart); starts 1-4 are the altitude-shifted copies of it. The
    perturbed side is the best CONVERGED of the four, by the same objective the head-to-
    head figure uses:

        weighted sum        J = coef * contrail + (1 - coef) * fuel
        epsilon-constraint  J = contrail

    Unlike multistart vs singlestart, either side can win here: the multistart's result
    is the best of all five, so it can never lose to start 0, but the four perturbed
    starts alone can. A point where start 0 or all four perturbed starts failed is
    dropped; with --full-only, a flight that lost any point is dropped whole.

    Columns: flight_id, sweep, point, start0, perturbed, perturbed_start (1-4).
    """
    cache = CAMPDIR / f"start0_vs_perturbed_sigma{sigma:g}{POPTAG}{FULLTAG}.csv"
    if cache.exists():
        print(f"reading {cache.name}  (delete it to re-read the pickles)")
        return _in_population(pd.read_csv(cache), cache.name)

    folder, pattern = next((f, pat) for m, f, pat in SOURCES if m == "opentop_ms")
    rows, complete, n_flights = [], set(), 0
    for f in sorted(folder.glob(pattern.replace("[0-9]", f"{sigma:g}"))):
        with gzip.open(f, "rb") as fh:
            payload = pickle.load(fh)
        if payload["error"]:
            continue
        ctx = payload["context"]
        fid = ctx.get("flight_id")
        if not admitted(fid) or float(ctx.get("sigma")) != float(sigma):
            continue
        n_flights += 1
        n_pts, n_ok = 0, 0
        for sweep in ("weighted_sum", "epsilon_constraint"):
            for t in payload["results"].get(sweep, ()):
                meta = t.get("meta") or {}
                cands = meta.get("candidates") or []
                if not cands:
                    continue
                n_pts += 1
                if sweep == "weighted_sum":
                    coef = float(meta["coef"])
                    point = coef
                    obj = lambda c: coef * c["grid_cost_exact"] + (1 - coef) * c["fuel"]
                else:
                    point = float(meta["fuel_slack_frac"])
                    obj = lambda c: c["grid_cost_exact"]
                c0 = [c for c in cands if c["start_index"] == 0 and c["success"]]
                pert = [c for c in cands if c["start_index"] != 0 and c["success"]]
                if not c0 or not pert:
                    continue
                best = min(pert, key=obj)
                rows.append((fid, sweep, point, obj(c0[0]), obj(best), best["start_index"]))
                n_ok += 1
        if n_pts and n_ok == n_pts:
            complete.add(fid)

    df = pd.DataFrame(rows, columns=["flight_id", "sweep", "point", "start0", "perturbed",
                                     "perturbed_start"])
    print(f"  {len(df):,} start-0 / perturbed pairs over {df.flight_id.nunique():,} of "
          f"{n_flights:,} flights")
    if FULL_ONLY:
        print(f"  --full-only: {len(complete):,} flights with both sides converged at "
              f"every sweep point")
        df = df[df.flight_id.isin(complete)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


class HeadToHeadBuckets(Plot):
    """Where the multistart actually wins, per sweep point, as shares of the population.

    Only the TAILS are drawn. The distribution has almost all of its mass at exactly
    zero -- between a quarter and a half of the pairs are bit-identical, and over 93 %
    agree to within 0.1 % -- so a stacked bar spends its whole height on a neutral block
    and pins the interesting slice against the 100 % edge, where a 4 % band reads as
    "nearly all". The agreement is printed as a number above each pair instead, and the
    axis is given to the part that varies: how often, and by how much, one start
    strategy beats the other.

    The comparison is on the objective THAT solve minimised, not on the front: two
    points of the same flight at the same sweep coordinate are the same problem solved
    twice, which is the only fair way to ask which start strategy is better.
    """

    section = "Optimality"

    stem = "ms_vs_ss_buckets"
    sigma = 2.0
    TIE = 0.1               # |rel| <= this is "the same answer", in % of the objective
    # Who is being compared. LOW is the one that wins when rel < 0 -- see relative() --
    # and its bars are the warm pair; HIGH is the denominator of the relative
    # difference, and gets the cool pair. (light, dark) = (0.1-1 %, > 1 %).
    LOW, LOW_COLOURS = "multistart", ("#f5cd66", "#eda100")
    HIGH, HIGH_COLOURS = "singlestart", ("#8fd9bd", "#0f8a5f")
    # (low, high, label, colour) in per cent of the singlestart objective, half-open
    # (low, high]. rel < 0 means the MULTISTART reached the lower objective. Each mode
    # gets its own bar, stacked by magnitude: the modest win at the bottom, the large
    # one on top, so the bar's height is "better at all" and its dark part is "better
    # by more than 1 %".
    SWEEPS = [("weighted_sum", "weighted sum"),
              ("epsilon_constraint", r"$\epsilon$-constraint")]
    # Per-panel height passed to paper_size, which rescales by the printed width: a
    # one-panel figure keeps the same drawn aspect only if it asks for half of it.
    PANEL_H = 3.0

    def relative(self, d: pd.DataFrame) -> pd.Series:
        """Signed difference in %, negative where LOW reached the lower objective."""
        return (d.opentop_ms - d.opentop_single) / d.opentop_single.abs() * 100

    def bands(self):
        """(low, high, label, colour) per bar segment, for the two bars of a group."""
        lo_l, lo_d = self.LOW_COLOURS
        hi_l, hi_d = self.HIGH_COLOURS
        return ([(-1.0, -self.TIE, f"{self.LOW} better by 0.1-1 %", lo_l),
                 (-np.inf, -1.0, f"{self.LOW} better by > 1 %", lo_d)],
                [(self.TIE, 1.0, f"{self.HIGH} better by 0.1-1 %", hi_l),
                 (1.0, np.inf, f"{self.HIGH} better by > 1 %", hi_d)])

    @property
    def outdir(self):
        return CAMPDIR

    def __init__(self, _df: pd.DataFrame):
        super().__init__(head_to_head_frame(self.sigma))
        self.stem = self.stem + POPTAG + FULLTAG

    @staticmethod
    def tick(sweep: str, point: float) -> str:
        """The sweep coordinate as it is set, not as it is stored."""
        return f"{point:g}" if sweep == "weighted_sum" else f"+{(point - 1) * 100:.0f} %"

    # Per-figure type sizes; None = the module's FS.
    TEXT_PT = None
    TITLE_PT = None

    def draw(self):
        txt = self.TEXT_PT or FS["label"]
        ttl = self.TITLE_PT or FS["title"]
        # per-panel width, not total: paper_size multiplies by n_panels itself.
        n = len(self.SWEEPS)
        fig, axes = plt.subplots(1, n, figsize=paper_size(4.6, n, self.PANEL_H,
                                                          span="full"),
                                 facecolor=SURFACE, layout="constrained", sharey=True,
                                 squeeze=False)
        axes = axes[0]
        # Signed, in per cent, and NEGATIVE means the multistart reached a lower
        # objective -- both sweeps minimise, so one sign convention covers them.
        d = self.df.copy()
        d["rel"] = self.relative(d)
        d = d.dropna(subset=["rel"])

        # One scale for both panels, from the data: the trend across the sweep is the
        # message, and two autoscaled panels would flatten it.
        tallest = 0.0
        for sweep, _ in self.SWEEPS:
            g = d[d.sweep == sweep]
            for p_ in g.point.unique():
                r = g[g.point == p_].rel
                tallest = max(tallest, (r < -self.TIE).mean() * 100,
                              (r > self.TIE).mean() * 100)

        for ax, (sweep, title) in zip(axes, self.SWEEPS):
            self.style(ax)
            ax.tick_params(labelsize=txt)
            g = d[d.sweep == sweep]
            points = sorted(g.point.unique())
            x = np.arange(len(points))
            low_bands, high_bands = self.bands()
            for bands, off in ((low_bands, +0.19), (high_bands, -0.19)):
                bottom = np.zeros(len(points))
                for lo, hi, label, colour in bands:
                    share = np.array([((g[g.point == p_].rel > lo)
                                       & (g[g.point == p_].rel <= hi)).mean() * 100
                                      for p_ in points])
                    ax.bar(x + off, share, bottom=bottom, width=0.34, color=colour,
                           linewidth=0, zorder=3,
                           label=label if ax is axes[0] else None)
                    bottom += share
                # The bar's total, on top of it: the two magnitudes are already the two
                # colours, and a number per segment would be four labels on a 4 % bar.
                for xi, tot in enumerate(bottom):
                    if tot >= 0.2:
                        ax.text(xi + off, tot + tallest * 0.03, f"{tot:.1f}",
                                ha="center", va="bottom", fontsize=txt,
                                color=INK_2, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels([self.tick(sweep, p_) for p_ in points])
            # Headroom for the legend, which sits over the low bars of this sweep's
            # left-hand end: every point's agreement share is the rest of the 100 % and
            # belongs in the caption, not in a row of numbers over the bars.
            ax.set_ylim(0, tallest * 1.55)
            ax.set_xlim(-0.7, len(points) - 0.3)
            ax.set_title(title, fontsize=ttl, color=INK_2, pad=10)
            ax.grid(axis="x", visible=False)

        axes[0].set_ylabel("flights (%)", fontsize=txt, color=INK_2)
        # Upper left of the weighted-sum panel: that corner is empty in both sweeps,
        # since the bars grow towards the contrail-heavy end of the grid.
        leg = axes[0].legend(loc="upper left", frameon=True, fontsize=txt,
                             handlelength=1.0, labelspacing=0.3, borderpad=0.45)
        leg.get_frame().set(facecolor=SURFACE, edgecolor=RULE, linewidth=0.5, alpha=0.92)
        for t in leg.get_texts():
            t.set_color(INK_2)
        return fig


class Start0VsPerturbedBuckets(HeadToHeadBuckets):
    """Start 0 (the singlestart) against the best of the four perturbed starts.

    The same buckets as ms_vs_ss_buckets, with the multistart split into its parts:
    the multistart result is the best of start 0 and starts 1-4, so it can never lose
    to start 0 -- this figure asks instead whether the perturbed starts ALONE beat the
    fuel-only guess, and where the fuel-only guess is the better start. Both sides can
    win, so all four bands carry bars.
    """

    stem = "start0_vs_perturbed_buckets"
    LOW, LOW_COLOURS = "perturbed starts", ("#f5cd66", "#eda100")
    HIGH, HIGH_COLOURS = "fuel-only start", ("#8fd9bd", "#0f8a5f")
    TEXT_PT = 10.0
    TITLE_PT = 12.0

    def __init__(self, _df: pd.DataFrame):
        Plot.__init__(self, perturbation_frame(self.sigma))
        self.stem = self.stem + POPTAG + FULLTAG

    def relative(self, d: pd.DataFrame) -> pd.Series:
        """Signed difference in % of start 0's objective, negative where the best
        perturbed start reached the lower objective."""
        return (d.perturbed - d.start0) / d.start0.abs() * 100

    def draw(self):
        """The yellow bars alone, by request: per sweep point, the share of flights
        where the best of the four perturbed starts reached a better optimum than the
        fuel-only start (start 0), split by how much -- 0.1-1 % (light) and > 1 %
        (dark) of start 0's objective. Below 0.1 % is a tie and not drawn; the
        flights where start 0 was the better one are no longer shown.
        Weighted sum and epsilon-constraint side by side, one panel each.
        """
        txt = self.TEXT_PT or FS["label"]
        ttl = self.TITLE_PT or FS["title"]
        # OpenTOP's green pair, as in the OpenTOP-vs-DP figures (light, dark).
        bands = [(self.TIE, 1.0, "by 0.1-1 %", "#8fd9bd"),
                 (1.0, np.inf, "by > 1 %", "#1baf7a")]
        sweeps = [("weighted_sum", "weighted sum"),
                  ("epsilon_constraint", "$" + chr(92) + "epsilon$-constraint")]
        fig, axes = plt.subplots(1, 2, figsize=paper_size(4.6, 2, self.PANEL_H,
                                                          span="full"),
                                 facecolor=SURFACE, layout="constrained", sharey=True)
        d = self.df.dropna(subset=["start0", "perturbed"]).copy()
        # improvement in % of start 0's objective, positive where a perturbed start won
        d["gain"] = (d.start0 - d.perturbed) / d.start0.abs() * 100
        tallest = max((d[d.sweep == sw].groupby("point").gain
                       .apply(lambda r: (r > self.TIE).mean() * 100).max())
                      for sw, _ in sweeps)
        for ax, (sweep, title) in zip(axes, sweeps):
            self.style(ax)
            ax.tick_params(labelsize=txt)
            g = d[d.sweep == sweep]
            pts = sorted(g.point.unique())
            x = np.arange(len(pts))
            bottom = np.zeros(len(pts))
            for lo, hi, label, colour in bands:
                share = np.array([((g[g.point == p_].gain > lo)
                                   & (g[g.point == p_].gain <= hi)).mean() * 100
                                  for p_ in pts])
                ax.bar(x, share, bottom=bottom, width=0.6, color=colour, linewidth=0,
                       zorder=3, label=label if ax is axes[0] else None)
                bottom += share
            for xi, tot in zip(x, bottom):
                ax.text(xi, tot + tallest * 0.02, f"{tot:.1f}", ha="center",
                        va="bottom", fontsize=txt, color=INK_2, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels([self.tick(sweep, p_) for p_ in pts])
            ax.set_xlim(-0.6, len(pts) - 0.4)
            ax.set_ylim(0, tallest * 1.18)
            ax.grid(axis="x", visible=False)
            ax.set_title(title, fontsize=ttl, color=INK_2)
        axes[0].set_ylabel("flights (%)", fontsize=txt, color=INK_2)
        # One line above both panels: inside a panel, the legend at this size is wider
        # than the panel and squeezed it.
        h, lab = axes[0].get_legend_handles_labels()
        leg = fig.legend(h, lab, loc="outside upper center", ncol=2, frameon=False,
                         fontsize=txt, handlelength=1.2, columnspacing=1.5,
                         title="perturbed starts better than the fuel-only start",
                         title_fontsize=txt)
        leg.get_title().set_color(INK_2)
        for tx in leg.get_texts():
            tx.set_color(INK_2)
        return fig


class Start0VsPerturbedAll(Start0VsPerturbedBuckets):
    """start0_vs_perturbed_buckets with the gains below 0.1 % drawn too: the share of
    flights where the best perturbed start beat the fuel-only start (start 0) by ANY
    amount, split 0-0.1 % / 0.1-1 % / > 1 % of start 0's objective.

    "Any amount" stops at SAME: |gap| <= 1e-6 % is one optimum reached twice and
    differing only by solver rounding, not a win. Flights where start 0 was the better
    one are not drawn, as in the parent figure.
    """

    stem = "start0_vs_perturbed_all"
    SAME = 1e-6          # %, below which the two are one optimum

    def draw(self):
        txt = self.TEXT_PT or FS["label"]
        ttl = self.TITLE_PT or FS["title"]
        bands = [(self.SAME, 0.1, "by < 0.1 %", "#d4f0e5"),
                 (0.1, 1.0, "by 0.1-1 %", "#8fd9bd"),
                 (1.0, np.inf, "by > 1 %", "#1baf7a")]
        sweeps = [("weighted_sum", "weighted sum"),
                  ("epsilon_constraint", "$" + chr(92) + "epsilon$-constraint")]
        fig, axes = plt.subplots(1, 2, figsize=paper_size(4.6, 2, self.PANEL_H,
                                                          span="full"),
                                 facecolor=SURFACE, layout="constrained", sharey=True)
        d = self.df.dropna(subset=["start0", "perturbed"]).copy()
        d["gain"] = (d.start0 - d.perturbed) / d.start0.abs() * 100
        tallest = max((d[d.sweep == sw].groupby("point").gain
                       .apply(lambda r: (r > self.SAME).mean() * 100).max())
                      for sw, _ in sweeps)
        for ax, (sweep, title) in zip(axes, sweeps):
            self.style(ax)
            ax.tick_params(labelsize=txt)
            g = d[d.sweep == sweep]
            pts = sorted(g.point.unique())
            x = np.arange(len(pts))
            bottom = np.zeros(len(pts))
            for lo, hi, label, colour in bands:
                share = np.array([((g[g.point == p_].gain > lo)
                                   & (g[g.point == p_].gain <= hi)).mean() * 100
                                  for p_ in pts])
                ax.bar(x, share, bottom=bottom, width=0.6, color=colour, linewidth=0,
                       zorder=3, label=label if ax is axes[0] else None)
                bottom += share
            for xi, tot in zip(x, bottom):
                ax.text(xi, tot + tallest * 0.02, f"{tot:.1f}", ha="center",
                        va="bottom", fontsize=txt, color=INK_2, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels([self.tick(sweep, p_) for p_ in pts])
            ax.set_xlim(-0.6, len(pts) - 0.4)
            ax.set_ylim(0, tallest * 1.18)
            ax.grid(axis="x", visible=False)
            ax.set_title(title, fontsize=ttl, color=INK_2)
        axes[0].set_ylabel("flights (%)", fontsize=txt, color=INK_2)
        h, lab = axes[0].get_legend_handles_labels()
        leg = fig.legend(h, lab, loc="outside upper center", ncol=3, frameon=False,
                         fontsize=txt, handlelength=1.2, columnspacing=1.5,
                         title="perturbed starts better than the fuel-only start",
                         title_fontsize=txt)
        leg.get_title().set_color(INK_2)
        for tx in leg.get_texts():
            tx.set_color(INK_2)
        return fig


class OpenTOPvsDPBuckets(HeadToHeadBuckets):
    """OpenTOP at its best against the DP, weighted sum only, point by point.

    OpenTOP enters as the better of its two modes at that same coefficient -- the
    question is what the solver family can do, not which start strategy did it -- and
    the DP is the only other solver that minimises the same scalarisation on the same
    grid, which is why the eps-constraint panel has no counterpart here: the DP
    campaign is the ws-only run.
    """

    stem = "opentop_vs_dp_buckets"
    LOW, LOW_COLOURS = "OpenTOP", ("#8fd9bd", "#1baf7a")
    HIGH, HIGH_COLOURS = "DP", ("#a8c8ee", "#2a78d6")
    SWEEPS = [("weighted_sum", "weighted sum")]
    PANEL_H = 1.5

    @property
    def outdir(self):
        return OPTIMALITY_DIR

    def relative(self, d: pd.DataFrame) -> pd.Series:
        best = d[["opentop_single", "opentop_ms"]].min(axis=1)
        return (best - d.dp) / d.dp.abs() * 100


class OpenTOPSingleVsDPBuckets(OpenTOPvsDPBuckets):
    """The same comparison with OpenTOP as its singlestart (start 0) alone."""

    stem = "opentop_single_vs_dp_buckets"
    LOW = "OpenTOP singlestart"

    def relative(self, d: pd.DataFrame) -> pd.Series:
        return (d.opentop_single - d.dp) / d.dp.abs() * 100


class OpenTOPMsVsDPBuckets(OpenTOPvsDPBuckets):
    """The same comparison with OpenTOP as its multistart alone."""

    stem = "opentop_ms_vs_dp_buckets"
    LOW = "OpenTOP multistart"

    def relative(self, d: pd.DataFrame) -> pd.Series:
        return (d.opentop_ms - d.dp) / d.dp.abs() * 100


class OpenTOPModesVsDPBuckets(OpenTOPvsDPBuckets):
    """Singlestart and multistart against the DP on one panel: what the extra starts buy.

    Per coefficient, four bars: the DP's wins against each mode, then each mode's wins
    against the DP, each the share of flights beyond the TIE. Magnitude is not split
    here -- the two single-mode figures carry it -- so light/dark can mean the mode:
    light + hatched = singlestart, dark = multistart. Same flights in both: start 0 is
    one of the multistart's candidates, so the pair differs only by the extra starts.
    """

    stem = "opentop_modes_vs_dp_buckets"
    MODES = (("opentop_single", "singlestart", "//"), ("opentop_ms", "multistart", None))
    # How much better counts as better, in % of the DP objective. TIE by default; the
    # subclass below raises it to 5 %.
    THRESHOLD = HeadToHeadBuckets.TIE
    LABEL_SUFFIX = ""
    # Per-figure overrides of the module type sizes and canvas; None = module default.
    TEXT_PT = None          # every text but the title
    TITLE_PT = None
    FIGSIZE = None          # (w, h) in inches
    LEGEND_NCOL = 2
    SHOW_TITLE = True
    BAR_W = 0.2             # one bar, in coefficient slots; 4 x BAR_W is the group

    def draw(self):
        txt = self.TEXT_PT or FS["label"]
        ttl = self.TITLE_PT or FS["title"]
        fig, ax = plt.subplots(figsize=self.FIGSIZE or paper_size(4.6, 1, self.PANEL_H,
                                                                  span="full"),
                               facecolor=SURFACE, layout="constrained")
        self.style(ax)
        ax.tick_params(labelsize=txt)
        d = self.df[self.df.sweep == "weighted_sum"].dropna(subset=["dp"])
        points = sorted(d.point.unique())
        x = np.arange(len(points))
        w = self.BAR_W
        bars = []      # (offset, mode column, wins_for_opentop, colour, hatch, label)
        for i, (col, mode, hatch) in enumerate(self.MODES):
            bars.append((-1.5 * w + i * w, col, False, self.HIGH_COLOURS[i], hatch,
                         f"DP better than {mode}{self.LABEL_SUFFIX}"))
        for i, (col, mode, hatch) in enumerate(self.MODES):
            bars.append((0.5 * w + i * w, col, True, self.LOW_COLOURS[i], hatch,
                         f"OpenTOP {mode} better than DP{self.LABEL_SUFFIX}"))
        tallest = 0.0
        for off, col, ot_wins, colour, hatch, label in bars:
            share = []
            for p_ in points:
                g = d[d.point == p_]
                rel = (g[col] - g.dp) / g.dp.abs() * 100
                share.append(((rel < -self.THRESHOLD) if ot_wins
                              else (rel > self.THRESHOLD)).mean() * 100)
            share = np.array(share)
            tallest = max(tallest, share.max())
            ax.bar(x + off, share, width=w * 0.92, color=colour, hatch=hatch,
                   edgecolor=SURFACE, linewidth=0, zorder=3, label=label)
            for xi, tot in zip(x, share):
                ax.text(xi + off, tot + 1.2, f"{tot:.0f}", ha="center", va="bottom",
                        fontsize=txt, color=INK_2, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([self.tick("weighted_sum", p_) for p_ in points])
        if self.FIGSIZE is None:
            # Room for the two-row legend over the bars, but no further: the y axis is
            # a share of flights, so it stops near 100 instead of growing with the data.
            ax.set_ylim(0, max(tallest * 1.3, 100) + 12)
            ax.set_yticks(np.arange(0, 101, 20))
        else:
            # A compact canvas has no height to spare: the axis stops above the tallest
            # bar, and the legend sits over the short bars on the left.
            ax.set_ylim(0, tallest * 1.2)
        # The outer margin follows the group width, so wider bars also mean less empty
        # space at the two ends.
        ax.set_xlim(-2 * w - 0.15, len(points) - 1 + 2 * w + 0.15)
        if self.SHOW_TITLE:
            ax.set_title("weighted sum", fontsize=ttl, color=INK_2, pad=6)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("flights (%)", fontsize=txt, color=INK_2)
        leg = ax.legend(loc="upper left", ncol=self.LEGEND_NCOL, frameon=True,
                        fontsize=txt, handlelength=1.4,
                        labelspacing=0.3, borderpad=0.45, columnspacing=1.0)
        leg.get_frame().set(facecolor=SURFACE, edgecolor=RULE, linewidth=0.5, alpha=0.92)
        for t in leg.get_texts():
            t.set_color(INK_2)
        return fig



class OpenTOPModesVsDPBuckets5pct(OpenTOPModesVsDPBuckets):
    """The same four bars, counting only wins by more than 5 % of the DP objective.

    Drawn on a shorter canvas with larger type than the module default, for the paper:
    the axis stops above the tallest bar instead of at 100 %, which is most of the
    space the default layout leaves empty here.
    """

    stem = "opentop_modes_vs_dp_buckets_gt5pct"
    THRESHOLD = 5.0
    LABEL_SUFFIX = " by > 5 %"
    TEXT_PT = 12.0
    SHOW_TITLE = False      # the caption says "weighted sum"
    FIGSIZE = (FIG_W_FULL, 2.9)
    LEGEND_NCOL = 1
    BAR_W = 0.235           # groups 0.94 wide: 0.06 between coefficients, not 0.2


class OwnPareto(Plot):
    """One figure per method: that method's own Pareto front, every flight overlaid.

    Four figures, one scale. The axis limits are computed over ALL FOUR methods and
    then applied to each panel, so the four files can be laid out 2x2 and read as one
    plate -- a per-method autoscale would give each its own zoom and make the
    differences between them impossible to see, which is the only reason to draw four.

    Sigma is fixed at 2: this is the front figure, not a sigma study, and the campaign
    that has all three sigmas already answers that question elsewhere.

    What a dot is: one non-dominated trajectory of one flight, as found by this method
    alone. What the spread means: the population, not a single trade-off curve --
    2 800 fronts drawn on top of each other. Read the LOWER-LEFT EDGE for the shape of
    the trade-off and the vertical extent for how much the flights differ.
    """

    section = "Optimality"

    sigma = 2.0

    @property
    def outdir(self):
        return CAMPDIR      # not a --time folder: no time basis enters this figure

    def __init__(self, df: pd.DataFrame, key: str, name: str, colour: str, ref: str):
        super().__init__(df)
        self.key, self.name, self.colour, self.ref = key, name, colour, ref
        # The reference is in the name because nothing inside the scatter reveals it.
        self.stem = f"own_pareto_{key}_{ref}{POPTAG}"

    def draw(self):
        col = "fuel_pct_own" if self.ref == "own" else "fuel_pct_ws"
        whole = self.df.dropna(subset=[col])
        sub = whole[whole.method == self.key]

        fig, ax = plt.subplots(figsize=paper_size(4.6, 1, 4.2, span="col"),
                               facecolor=SURFACE, layout="constrained")
        self.style(ax)

        # 2.5 pt at low alpha for the DP_MuO, which returns a median of 54
        # points per flight against the other three's 13: at s=5 its cloud saturates
        # into a solid block and the density stops being readable. The threshold is on
        # the count actually drawn, so it follows --sweeps rather than the method name.
        dense = len(sub) > 100_000
        ax.scatter(sub[col], sub.contrail, color=self.colour, linewidths=0, zorder=3,
                   s=2.5 if dense else 5.0, alpha=0.22 if dense else 0.35)

        # Shared limits, from the pooled frame. x is clipped at the 99th percentile for
        # the reason JointPareto clips: a few flights run far out and would flatten
        # everything else against the origin. The low end is NOT clamped to zero -- the
        # graph solvers do find trajectories below the OpenTOP warm start, and hiding
        # that would be the one thing this axis exists to show.
        lo, hi = np.percentile(whole[col], [1, 99])
        ax.set_xlim(min(lo, 0) - 0.02 * (hi - lo), hi)
        ax.set_ylim(min(-0.05 * whole.contrail.max(), whole.contrail.min() * 1.2),
                    whole.contrail.max() * 1.04)
        # The label follows the data, not the flag: with no fuel-only sweep in the
        # pickles the reference IS the cheapest point of the method's own front, and
        # saying "fuel-only" would name a solve that was never run.
        from_sweep = bool(sub.ref_fuel_only.any()) if len(sub) else True
        ref_label = ("the OpenTOP fuel-only warm start" if self.ref != "own"
                     else "its own fuel-only trajectory" if from_sweep
                     else "its own cheapest trajectory")
        # Wrapped, not just short: at 3.5 in the unwrapped label runs past the canvas and
        # bbox_inches="tight" does not rescue a label that constrained layout has already
        # centred on the axes -- it came out reading "... trajectory (%".
        ax.set_xlabel(fit(fig, f"extra fuel over {ref_label} (%)", FS["label"]),
                      fontsize=FS["label"], color=INK_2)
        ax.set_ylabel("contrail cost", fontsize=FS["label"], color=INK_2)

        # Counted on the page rather than left to the reader: a method missing flights
        # the other three have is a convergence result, and an empty corner of the
        # scatter does not say which of the two it is.
        outside = int((sub[col] > hi).sum())
        note = f"{sub.flight_id.nunique():,} flights · {len(sub):,} points"
        if outside:
            note += f"\n{outside} points beyond the axis"
        # A partial fallback is the ambiguous case: some flights measured against
        # a fuel-only solve and some against their cheapest point, on one axis.
        # Counted here rather than left to the label, which says only one of the two.
        if self.ref == "own" and len(sub) and 0 < int(sub.ref_fuel_only.sum()) < len(sub):
            n = int((~sub.ref_fuel_only).sum())
            note += f"\n{n:,} points measured against the cheapest point (no fuel-only solve)"
        ax.annotate(note, (0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                    fontsize=FS["annot"], color=INK_2)

        fig.suptitle(fit(fig, f"{self.name} — its own Pareto front, "
                     f"$\sigma$ = {self.sigma:g}" + sweep_note((self.key,)),
                     FS["suptitle"]), fontsize=FS["suptitle"], color=INK)
        return fig


class BudgetScatter(Plot):
    """One figure per method: the +5%-fuel trajectory of every flight, one panel per sigma.

    Each dot is one flight. Its position says what the fuel budget bought — how much
    extra fuel went in, how much contrail that avoided — and its colour says what it
    cost to compute. Grey crosses are flights whose solver did not converge on that
    point; they are drawn, not dropped, because dropping them would hide exactly the
    cases where the method fails.

    y is the share of the cheapest trajectory's contrail that the budget removed. It
    cannot go below 0: the cheapest trajectory is itself affordable, so the best
    affordable point is never worse than it. A dot on 0 is a flight where the budget
    bought nothing at all; a dot on 100 is one where it removed the contrail entirely.

    A percentage needs the denominator to be a positive quantity, and this metric is
    signed: on the cheapest trajectory it is <= 0 for a third of OpenTOP singlestart
    flights at sigma=0. Those flights have no percentage to plot and are counted in the
    panel corner rather than silently dropped. The few that land above 100 % -- the
    chosen point went net-negative -- are counted there too.

    x is the fuel that best point happened to cost, not the budget: it stops below 5 %
    when the front ends earlier, and also when less fuel already reaches the minimum
    contrail. Neither is a truncation.
    """

    section = "Fuel_study"

    def __init__(self, df: pd.DataFrame, key: str, name: str):
        super().__init__(df)
        self.key, self.name = key, name
        self.stem = f"pareto_budget_{key}"

    def draw(self):
        sub = self.df[self.df.method == self.key].copy()
        # Flights whose cheapest trajectory has no positive contrail have no percentage.
        sub["pct"] = np.where(sub.b5_contrail_ref > 0,
                              100 * (1 - sub.b5_contrail / sub.b5_contrail_ref), np.nan)
        cmap = LinearSegmentedColormap.from_list("time", TIME_RAMP)
        # One colour scale for the WHOLE campaign, not per method: the same colour is the
        # same number of seconds in every one of the four figures, which is what makes
        # them comparable. The price is that DP, whose points all take 0.1-5 ms, sits in
        # the bottom fifth of a scale that has to reach DP_MuO's 12.7 h.
        t = np.log10(self.df.b5_time_s)
        norm = Normalize(t.min(), t.max())

        # constrained rather than tight layout: tight_layout cannot place a colorbar
        # that is attached to several axes at once.
        fig, axes = plt.subplots(1, len(SIGMAS), figsize=paper_size(4.6, len(SIGMAS), 4.6),
                                 sharex=True, sharey=True, facecolor=SURFACE,
                                 layout="constrained", squeeze=False)

        for ax, sigma in zip(axes[0], SIGMAS):
            self.style(ax)
            whole = sub[sub.sigma == sigma]
            panel = whole.dropna(subset=["pct"])
            ok, bad = panel[panel.b5_converged], panel[~panel.b5_converged]
            ax.scatter(ok.b5_fuel_pct, ok.pct, c=np.log10(ok.b5_time_s),
                       cmap=cmap, norm=norm, s=9, alpha=0.65, linewidths=0, zorder=3)
            # Marker as well as colour: identity must not rest on grey alone.
            ax.scatter(bad.b5_fuel_pct, bad.pct, marker="x", s=26,
                       color=FAILED, linewidths=1.1, zorder=4,
                       label=f"not converged (n = {len(bad)})")
            ax.set_title(f"$\\sigma$ = {sigma:g}   ({len(panel)} flights)",
                         fontsize=FS["title"], color=INK)
            ax.set_xlabel("extra fuel spent (%)", fontsize=FS["label"], color=INK_2)
            if len(bad):
                # Upper left: the dense band sits on the 0 baseline, so a legend down
                # there lands on top of the data.
                ax.legend(loc="upper left", frameon=False, fontsize=FS["legend"],
                          labelcolor=INK_2)

            # What the axis cannot show, said out loud rather than dropped in silence.
            missing, over = len(whole) - len(panel), int((panel.pct > 100).sum())
            notes = ([f"{missing} without a percentage (contrail $\\leq$ 0 when cheapest)"] if missing else []) \
                  + ([f"{over} above 100% (contrail went negative)"] if over else [])
            if notes:
                # Upper right: the dense band lies along y=0, so the bottom corners are
                # the one place these lines would sit on top of the data.
                ax.annotate("\n".join(notes), (0.97, 0.97), xycoords="axes fraction",
                            ha="right", va="top", fontsize=FS["annot"], color=INK_2)

        axes[0][0].set_ylim(-3, 104)
        axes[0][0].set_ylabel("contrail removed, % of the cheapest trajectory's",
                           fontsize=FS["label"], color=INK_2)
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0.015)
        # Five evenly spaced ticks labelled in seconds, not whole decades: DP's per-point
        # times fit inside a single decade and a decade rule leaves it with one tick.
        ticks = np.linspace(t.min(), t.max(), 5)
        bar.set_ticks(ticks)
        bar.set_ticklabels([f"{10 ** v:.3g}" for v in ticks])
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=INK_2, labelsize=FS["tick"])
        bar.set_label("compute time of this point (s)", fontsize=FS["label"], color=INK_2)

        source = ("whole Pareto front — the colour is the time of the whole search, "
                  "the same for every point of a flight"
                  if self.key == "dp_muo" else "epsilon-constraint points only")
        # No explicit y: constrained layout places the suptitle itself, and pinning it
        # to 1.0 drops it on top of the panel titles.
        fig.suptitle(fit(fig, f"{self.name} — the trajectory each flight would fly on a "
                     f"+{100 * (BUDGET - 1):.0f}% fuel budget", FS["suptitle"]),
                     fontsize=FS["suptitle"], color=INK)
        fig.supxlabel(fit(fig, f"one dot per flight; {source}.  Both axes are measured against "
                      "that flight's own cheapest trajectory.  The colour scale is shared by "
                      "all four methods, so colours are comparable between figures"),
                      fontsize=FS["annot"], color=INK_2)
        return fig


class ParetoSpread(Plot):
    """Dispersion INSIDE each Pareto: how unequal are the solves of one sweep.

    Three violins per solver, over the whole population: for every flight, the
    fastest, the median and the slowest of its own solves. So the question is not
    "which flight was slow" but "within one flight's sweep, how much does one point
    cost against another".

    DP_MuO's three violins are identical on purpose, and that is the result: it
    does ONE label-setting sweep and the whole front falls out of it, so every point
    carries the same ``front_total_s``. DP and OpenTOP pay per Pareto point;
    DP_MuO pays once for the front.
    """

    section = "Computing_time"

    stem = "pareto_spread"
    # A property, not `outdir = SPREADDIR`: a class attribute is evaluated at import,
    # before select_campaign() rebinds the module globals, so it froze on whatever
    # FIGDIR/pareto_spread meant then. While bspline's figdir WAS FIGDIR that looked
    # correct, and the linear run silently overwrote the bspline figure with its own.
    @property
    def outdir(self):
        return SPREADDIR
    PICKS = [("fastest", "t_min", 0.72), ("median", "t_med", 0.42), ("slowest", "t_max", 1.0)]

    def draw(self):
        fig, axes = plt.subplots(1, len(SIGMAS), figsize=paper_size(5.6, len(SIGMAS), 5.4),
                                 sharey=True, facecolor=SURFACE, squeeze=False)
        for ax, sigma in zip(axes[0], SIGMAS):
            self.style(ax)
            sub = self.df[self.df.sigma == sigma]
            ticks, labels = [], []
            for m, (key, name, colour, _) in enumerate(METHODS):
                v = sub[sub.method == key]
                for k, (pick, column, shade) in enumerate(self.PICKS):
                    x = m * 3.6 + k
                    val = v[column].dropna().to_numpy()
                    val = val[val > 0]
                    if len(val):
                        self.violin(ax, x, val, self.fade(colour, shade), "{:.3g} s")
                    ticks.append(x)
                    labels.append(pick)
                ax.text(m * 3.6 + 1, 1.02, f"{name.replace(' ', chr(10))}",
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=FS["tick"], color=INK)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontsize=FS["annot"], color=INK_2, rotation=90)
            ax.set_xlim(-0.7, len(METHODS) * 3.6 - 1.4)
            ax.set_title(f"$\\sigma$ = {sigma:g}", fontsize=FS["title"], color=INK, pad=26)

        col = pd.concat([self.df[c] for _, c, _ in self.PICKS]).dropna().to_numpy()
        self.decade_ticks(axes[0][0], col[col > 0])
        axes[0][0].set_ylabel("time of ONE solve inside the sweep (s)",
                           fontsize=FS["label"], color=INK_2)
        fig.suptitle(fit(fig, "Dispersion inside a single Pareto: fastest, median and slowest "
                     "solve of each flight", FS["suptitle"]),
                     fontsize=FS["suptitle"], color=INK, y=1.0)
        fig.text(0.5, 0.0, fit(fig, "one flight contributes one value to each violin   ·   "
                 "DP_MuO's three are identical: it builds the whole front in a single "
                 "sweep, so every point carries the same time"), ha="center", va="top",
                 fontsize=FS["annot"], color=INK_2)
        return fig

    @staticmethod
    def fade(hex_colour: str, k: float) -> tuple:
        """Same hue, k of the way to white -- shades one method's three violins."""
        rgb = np.array([int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        return tuple(1 - k * (1 - rgb))


def report(df: pd.DataFrame) -> None:
    """The numbers behind the figures, for the tables in the text."""
    print("\nper-flight time ratios")
    print(f"{'sigma':>5} {'pair':24s} {'n':>6} {'p10':>10} {'median':>10} {'p90':>10}")
    for sigma in SIGMAS:
        wide = df[df.sigma == sigma].pivot_table(index="flight_id", columns="method",
                                                 values=TIME)
        for num, den, label, _ in PAIRS:
            # Either side can be absent at a sigma: the DP ran at sigma = 2 only.
            if num not in wide.columns or den not in wide.columns:
                continue
            r = (wide[num] / wide[den]).dropna()
            print(f"{sigma:>5g} {label:24s} {len(r):>6d} {r.quantile(.1):>10.4g} "
                  f"{r.median():>10.4g} {r.quantile(.9):>10.4g}")

    print("\n" + TLABEL + " time and what it delivered")
    print(f"{'sigma':>5} {'method':16s} {'n':>6} {'t_median':>10} {'t_p90':>10} "
          f"{'t_max':>10} {'total_h':>9} {'traj_med':>9} {'s/traj':>9}")
    for sigma in SIGMAS:
        sub = df[df.sigma == sigma]
        for key, name, _, _ in METHODS:
            v = sub[sub.method == key]
            if v.empty:
                continue
            t = v[TIME]
            print(f"{sigma:>5g} {name:16s} {len(v):>6d} {t.median():>10.4g} "
                  f"{t.quantile(.9):>10.4g} {t.max():>10.4g} "
                  f"{t.sum() / 3600:>9.2f} {v.n_traj.median():>9.0f} "
                  f"{(t / v.n_traj).median():>9.4g}")


# ── dominance: which method actually found the better trajectory ─────────────
# Answers one question and only one: after the SHARED post-hoc evaluation
# (evaluate_path, the same physics for all three solvers), does a graph solver
# deliver a trajectory that beats an OpenTOP one on BOTH axes at once — less fuel
# AND less contrail? A point that is cheaper in fuel but worse in contrail is not
# a win, it is a different place on the front, and mixing the two is how a
# comparison ends up saying nothing.
# One cache PER MARGIN. The margin is baked into the counts, not applied when
# reading, so a single file would hand yesterday's strict numbers to today's
# --margin 0.005 run without a word — the same silent-mixture trap the INTERP
# guard in batch_graph2d.sbatch exists to prevent.
def dom_cache(margin: float) -> Path:
    return CAMPDIR / f"dominance_m{margin * 1000:.0f}permille.csv"


# Which sweeps make up a method's front. The pooled union is what SWEEP_SETS["all"]
# holds: the question it answers is what the METHOD delivered for this flight, not
# what one sweep did. DP_MuO has no sweeps — one label-setting solve IS the front.
# Kept as a name of its own because it is the vocabulary of --method and the default
# every report reads; the ACTIVE set is sweeps_of(), which --sweeps chooses.
FRONT_SWEEPS = SWEEP_SETS["all"]

# (graph method, OpenTOP method) compared head to head.
DOM_PAIRS = [("dp", "opentop_single"), ("dp_muo", "opentop_single"),
             ("dp", "opentop_ms"),     ("dp_muo", "opentop_ms")]


def front_arrays(payload: dict, method: str, with_sweep: bool = False):
    """(fuel_kg, contrail) of every CONVERGED point the method delivered.

    Non-converged points are dropped rather than kept as losers: a solver failure
    is a convergence result, already reported elsewhere, and counting it here
    would let a method "win" by comparing against its rival's crashes.

    ``with_sweep`` adds which sweep produced each point (fuel_only, weighted_sum,
    epsilon_constraint, pareto_dp_muo). Opt-in rather than always returned so the
    three existing two-value call sites keep working unchanged.
    """
    fuel, contrail, sweeps = [], [], []
    for sweep in sweeps_of(method):
        for t in payload["results"].get(sweep, ()):
            if not t.get("success"):
                continue
            f, c = t.get("fuel_kg"), t.get("contrail")
            if f is None or c is None:
                continue
            f, c = float(f), float(c)
            if np.isfinite(f) and np.isfinite(c):
                fuel.append(f)
                contrail.append(c)
                sweeps.append(sweep)
    fu = np.asarray(fuel, dtype=float)
    co = np.asarray(contrail, dtype=float)
    return (fu, co, sweeps) if with_sweep else (fu, co)


def n_beaten(a: tuple, b: tuple, margin: float) -> int:
    """How many points of ``b`` are beaten by at least one point of ``a``.

    Beaten = strictly less fuel AND strictly less contrail. ``margin`` is relative
    and demanded on both axes so numerical dust is not counted as a win. Contrail
    is SIGNED — it goes negative on 16 % of OpenTOP flights — so the margin is
    taken on its magnitude instead of as a multiplicative factor, which would
    flip direction below zero.
    """
    af, ac = a
    bf, bc = b
    if af.size == 0 or bf.size == 0:
        return 0
    better_fuel = af[:, None] < bf[None, :] * (1.0 - margin)
    better_cont = ac[:, None] < bc[None, :] - margin * np.abs(bc[None, :])
    return int((better_fuel & better_cont).any(axis=0).sum())


def pareto_mask(fuel: np.ndarray, contrail: np.ndarray) -> np.ndarray:
    """Non-dominated mask, O(n log n): sort by fuel and keep every strict
    improvement in contrail. DP_MuO fronts run to thousands of points, so the
    quadratic version is not an option at population scale."""
    keep = np.zeros(fuel.size, dtype=bool)
    best = np.inf
    for i in np.lexsort((contrail, fuel)):
        if contrail[i] < best:
            keep[i] = True
            best = contrail[i]
    return keep


# ----------------------------------------------------------------------------
# Convexity of the front.
#
# A non-dominated point that lies strictly ABOVE the lower convex hull of its own
# front is UNSUPPORTED: no weighted-sum scalarisation returns it for any weight,
# because every weight vector optimises to a hull vertex. The epsilon-constraint
# sweep can reach it; the weighted sum never will. So this is not a remark about
# the shape of a curve — it is the exact set of trade-offs one family of sweeps
# is blind to, and DP_MuO is the only method here that
# enumerates the whole label set and can therefore exhibit them.
#
# Two equivalent readings, both reported because they fail differently under
# numerical dust: the hull gap (how far above the hull a point sits) and the
# slope kink (where the tangent stops flattening as fuel grows). On a convex
# front the slopes dc/df are negative and INCREASING towards zero.

def _lower_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the lower convex hull, x already sorted ascending and unique.

    Andrew's monotone chain, lower half only. Collinear points are dropped from
    the chain — they are still supported, and the gap test below scores them at
    zero, which is why membership is not used as the criterion.
    """
    hull: list[int] = []
    for i in range(x.size):
        while len(hull) >= 2:
            x0, y0 = x[hull[-2]], y[hull[-2]]
            x1, y1 = x[hull[-1]], y[hull[-1]]
            cross = (x1 - x0) * (y[i] - y0) - (y1 - y0) * (x[i] - x0)
            if cross <= 0.0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.asarray(hull, dtype=int)


def convexity_row(fuel: np.ndarray, contrail: np.ndarray,
                  tol: float, min_span: float) -> dict | None:
    """Non-convexity of one flight's front, or None when the test is undefined.

    Both axes are rescaled to [0, 1] over the front before the test, so ``tol``
    is one number for every flight regardless of units — fuel runs to tens of
    thousands of kg while contrail is signed and often near zero, and an absolute
    tolerance in raw units would mean something different on each flight.

    That same rescaling is what makes ``min_span`` necessary. A front whose
    contrail varies by a millionth of its own level has no trade-off in it, but
    normalising still stretches that dust across the unit square and every
    rounding error becomes a full-height bulge above the hull. Such fronts are
    reported as degenerate, not as non-convex.
    """
    keep = pareto_mask(fuel, contrail)
    f, c = fuel[keep], contrail[keep]
    order = np.argsort(f)
    f, c = f[order], c[order]
    if f.size < 3:
        return None               # a hull needs an interior point to bulge above
    fs, cs = float(f[-1] - f[0]), float(c.max() - c.min())
    if fs <= 0.0 or cs <= 0.0:
        return None               # degenerate front: one axis is constant
    scale_f = max(float(np.max(np.abs(f))), 1e-30)
    scale_c = max(float(np.max(np.abs(c))), 1e-30)
    if fs / scale_f < min_span or cs / scale_c < min_span:
        return None               # the front is a point in disguise

    x = (f - f[0]) / fs
    y = (c - c.min()) / cs

    hull = _lower_hull(x, y)
    gap = y - np.interp(x, x[hull], y[hull])      # >= 0 by construction
    unsupported = gap > tol

    slope = np.diff(y) / np.diff(x)
    kinks = int(np.sum(np.diff(slope) < -tol))    # tangent stopped flattening

    return {
        "n_raw": int(fuel.size),
        "n_front": int(f.size),
        "n_hull": int(hull.size),
        "n_unsupported": int(unsupported.sum()),
        "frac_unsupported": float(unsupported.mean()),
        "n_kinks": kinks,
        "max_gap": float(gap.max()),                    # normalised units
        "max_gap_contrail": float(gap.max() * cs),      # same, in contrail units
        "fuel_span_kg": fs,
        "contrail_span": cs,
        "contrail_span_rel": cs / scale_c,
        "fuel_at_max_gap": float(f[int(np.argmax(gap))]),
    }


def load_convexity(folder: Path, pattern: str, method: str,
                   tol: float, min_span: float) -> pd.DataFrame:
    """One row per pickle in ``folder``. Reads the pickles directly, no cache.

    Deliberately NOT using SOURCES: since the Bspline/Linear split the same method
    lives in more than one directory, and which one is being measured has to be an
    explicit argument rather than something the script decides.
    """
    files = sorted(folder.glob(pattern))
    if not files:
        subs = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
        print(f"no {pattern} directly in {folder}")
        if subs:
            print(f"  subdirectories present: {', '.join(subs)}")
            print("  point --dir at one of them; mixing campaigns would be silent")
        return pd.DataFrame()

    rows, skipped, undefined, excluded = [], 0, 0, 0
    for path in files:
        with gzip.open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload["error"]:
            skipped += 1
            continue
        if not admitted((payload["context"] or {}).get("flight_id")):
            excluded += 1
            continue
        row = convexity_row(*front_arrays(payload, method), tol, min_span)
        if row is None:
            undefined += 1
            continue
        ctx = payload["context"]
        rows.append({"flight_id": ctx["flight_id"], "sigma": float(ctx["sigma"]),
                     "m0": ctx.get("m0"), "actype": ctx.get("actype"), **row})

    print(f"{len(files)} pickles in {folder.name}: {len(rows)} measured, "
          f"{skipped} error stubs, {undefined} fronts too small or too flat to test, "
          f"{excluded} outside the population")
    return pd.DataFrame(rows)


def report_convexity(df: pd.DataFrame, tol: float, csv: Path) -> None:
    if df.empty:
        return
    print(f"\nnon-convex (unsupported) points on the front   tol={tol:g} "
          f"of the normalised front")
    print("unsupported = non-dominated but strictly above the lower convex hull "
          "-> unreachable by any weighted sum\n")
    print(f"{'sigma':>5} {'flights':>8} {'with >=1':>9} {'median n':>9} "
          f"{'p95 n':>7} {'max n':>7} {'median %':>9} {'max gap':>9}")
    for sigma in sorted(df.sigma.unique()):
        v = df[df.sigma == sigma]
        has = v.n_unsupported > 0
        print(f"{sigma:>5g} {len(v):>8d} {has.mean():>8.1%} "
              f"{v.n_unsupported.median():>9.0f} {v.n_unsupported.quantile(.95):>7.0f} "
              f"{v.n_unsupported.max():>7.0f} {v.frac_unsupported.median():>8.1%} "
              f"{v.max_gap.max():>9.4f}")

    # The two readings are the same statement, but the slope test divides by the
    # fuel spacing: two front points a gram apart turn a negligible bulge into a
    # huge slope jump. Where they disagree the hull gap is the one to trust — it
    # is measured in the same units as the objective.
    both = (df.n_kinks > 0) != (df.n_unsupported > 0)
    print(f"\nfronts with a slope kink: {(df.n_kinks > 0).mean():.1%}   "
          f"disagreeing with the hull test: {both.mean():.1%} "
          f"({int(both.sum())} flights)")

    worst = df.nlargest(10, "max_gap")
    print("\nworst 10 by hull gap")
    print(f"{'flight_id':22s} {'sig':>4} {'n_front':>8} {'unsup':>6} {'gap':>8} "
          f"{'gap (contrail)':>15}")
    for _, r in worst.iterrows():
        print(f"{r.flight_id:22s} {r.sigma:>4g} {r.n_front:>8.0f} "
              f"{r.n_unsupported:>6.0f} {r.max_gap:>8.4f} {r.max_gap_contrail:>15.4f}")

    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"\nper-flight rows written to {csv}")


# ── convergence audit (OpenTOP singlestart) ──────────────────────────────────
# The IPOPT return status is NOT in the result pickle. A record carries only
# ``success`` and ``meta['n_iter']``, and opentop hands back the last iterate on
# failure (``base.Base._solve`` swallows the RuntimeError), so a failed solve looks
# like a perfectly ordinary trajectory. The status survives in exactly one place:
# ``debug/<flight>_sigma<s>_debug.pkl.gz``, written by ``batch_opentop.save_debug``
# only for flights with at least one non-converged solve.
#
# Those two halves of the debug payload are joined by POSITION -- ``solves`` comes
# from ``opt.results`` while ``ipopt`` comes from the recorder wrapped around
# ``cruise._solve`` -- so the join is checked rather than trusted: ``success`` and the
# iteration count must agree on every solve, or the flight's statuses are dropped as
# unaligned instead of being silently mis-attributed to the wrong sweep point.
CONVERGENCE_METHODS = ["fuel_only", "weighted_sum", "epsilon_constraint"]


def sweep_point(rec_meta: dict):
    """The sweep coordinate of one solve: the weight for WS, the fuel slack for eps."""
    coef = rec_meta.get("coef")
    return coef if coef is not None else rec_meta.get("fuel_slack_frac")


def debug_statuses(dbg_path: Path):
    """Per-solve IPOPT return_status for one flight, or (None, reason) if unusable."""
    with gzip.open(dbg_path, "rb") as fh:
        dbg = pickle.load(fh)
    solves, ipopt = dbg.get("solves") or [], dbg.get("ipopt") or []
    if len(solves) != len(ipopt):
        return None, "length mismatch"
    for a, b in zip(solves, ipopt):
        if bool(a.get("success")) != bool(b.get("success")):
            return None, "success mismatch"
        if a.get("n_iter") != b.get("iter_count"):
            return None, "iteration count mismatch"
    return [str(b.get("return_status")) for b in ipopt], ""


def convergence_sources() -> list:
    """(label, folder, suffix) for the OpenTOP runs this campaign has on disk.

    Both singlestart and multistart, because "did this flight converge" is asked of the
    pair: multistart exists precisely to rescue what singlestart missed, and a list that
    covers only one of them cannot show whether it did.
    """
    label_of = {"opentop_single": "single", "opentop_ms": "multistart"}
    out = []
    for method, root, _pattern in SOURCES:
        if method not in label_of:
            continue
        # One entry per m0 folder: load_convergence globs a flat folder, and each m0
        # folder has its own debug/ next to its pickles.
        mass_dirs = [d for d in sorted(root.glob("m_0*"))
                     if d.is_dir() and any(d.glob("*.pkl.gz"))]
        for folder in mass_dirs:
            # PICKLE_SUFFIX, not the SOURCES pattern: load_convergence globs per sigma,
            # and "" is what keeps the singlestart glob off any _ms file.
            out.append((label_of[method], folder, PICKLE_SUFFIX[method]))
        if not mass_dirs:
            print(f"  [skip] {label_of[method]}: {root} missing or empty")
    return out


def load_convergence(sources, sigmas) -> tuple:
    """(one row per solve, one row per error stub) across the given OpenTOP runs.

    Reads the pickles directly, no cache: this is a triage question asked right after a
    campaign lands, and a cache that silently predates the last resume would answer the
    wrong question.
    """
    solve_rows, stub_rows, unaligned, n_excl = [], [], {}, 0
    for label, folder, suffix in sources:
        for sigma in sigmas:
            files = sorted(folder.glob(f"*_sigma{sigma:g}{suffix}.pkl.gz"))
            for path in files:
                with gzip.open(path, "rb") as fh:
                    payload = pickle.load(fh)
                ctx = payload.get("context") or {}
                fid = ctx.get("flight_id") or path.name.split("_sigma")[0]
                if not admitted(fid):
                    n_excl += 1
                    continue
                if payload["error"]:
                    # A stub never reached the solver, so it has no solves to classify.
                    # Counting it as non-convergence would be wrong: most stubs are
                    # an aircraft type OpenAP has no model for, not a failed NLP.
                    stub_rows.append({"run": label, "flight_id": fid,
                                      "sigma": float(sigma), "error": str(payload["error"])})
                    continue

                dbg = folder / "debug" / f"{fid}_sigma{sigma:g}_debug.pkl.gz"
                statuses, why = debug_statuses(dbg) if dbg.exists() else (None, "")
                if dbg.exists() and statuses is None:
                    unaligned[(label, why)] = unaligned.get((label, why), 0) + 1

                # save_debug walks opt.results in order and run_one_flight fills
                # payload["results"] from that same list, so this rebuilds its order.
                flat = [(m, r) for m in CONVERGENCE_METHODS
                        for r in payload["results"].get(m, [])]
                for i, (method, rec) in enumerate(flat):
                    meta = rec.get("meta") or {}
                    ok = bool(rec.get("success"))
                    if ok:
                        status = "Solve_Succeeded"
                    elif statuses is not None and len(statuses) == len(flat):
                        status = statuses[i]
                    elif dbg.exists():
                        # Multistart lands here by construction: it calls cruise._solve
                        # n_starts times per sweep point, so the ipopt log is ~5x longer
                        # than opt.results and the positional join does not hold. The
                        # failure is real; only its IPOPT status is unavailable.
                        status = "<status unavailable>"
                    else:
                        status = "<no debug file>"
                    solve_rows.append({
                        "run": label, "flight_id": fid, "sigma": float(sigma),
                        "method": method, "point": sweep_point(meta), "success": ok,
                        "n_iter": meta.get("n_iter"), "time_s": meta.get("time_s"),
                        "status": status, "actype": ctx.get("actype"), "m0": ctx.get("m0"),
                    })
            print(f"  {label:11s} sigma {sigma:g}: {len(files):>5d} pickles from {folder.relative_to(BENCH.parent)}")

    for (label, why), n in sorted(unaligned.items()):
        print(f"  [warn] {label}: {n} debug files unusable ({why})")
    if n_excl:
        print(f"  {n_excl} flight-sigmas outside the population")
    return pd.DataFrame(solve_rows), pd.DataFrame(stub_rows)


def nonconverged_flights(solves: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, sigma, method, flight) that has at least one failed solve.

    Flight level rather than solve level because that is the unit the question is asked
    in, and ``n_failed`` keeps the distinction the flight level would otherwise destroy:
    1 of 6 points lost is a usable Pareto front with a hole, 6 of 6 is no front at all.
    """
    bad = solves[~solves.success]
    if bad.empty:
        return pd.DataFrame()
    tot = (solves.groupby(["run", "sigma", "method", "flight_id"]).success
                 .agg(n_solves="size", n_ok="sum").reset_index())
    agg = (bad.groupby(["run", "sigma", "method", "flight_id"])
              .agg(n_failed=("success", "size"),
                   statuses=("status", lambda s: ";".join(sorted(set(s)))),
                   points_failed=("point", lambda s: ";".join(
                       f"{x:g}" for x in sorted(s.dropna().unique()))),
                   actype=("actype", "first"), m0=("m0", "first"))
              .reset_index())
    out = agg.merge(tot, on=["run", "sigma", "method", "flight_id"])
    out["all_failed"] = out.n_failed == out.n_solves
    return out.sort_values(["run", "sigma", "method", "n_failed", "flight_id"],
                           ascending=[True, True, True, False, True])


def report_convergence(solves: pd.DataFrame, stubs: pd.DataFrame,
                       folder, csv: Path) -> None:
    if solves.empty:
        print("no OpenTOP pickles read")
        return
    runs = [r for r in ("single", "multistart") if r in set(solves.run)]
    sweeps = ["weighted_sum", "epsilon_constraint"]

    print("\nOpenTOP convergence")
    print("converged = IPOPT Solve_Succeeded on that individual solve\n")

    print("FLIGHTS")
    print(f"{'run':>11} {'sigma':>5} {'ran':>7} {'stubs':>6} {'all 13':>8} "
          f"{'partial':>8} {'none':>6}")
    for run in runs:
        for sigma in sorted(solves[solves.run == run].sigma.unique()):
            v = solves[(solves.run == run) & (solves.sigma == sigma)]
            per = v.groupby("flight_id").success.agg(["sum", "count"])
            allok = int((per["sum"] == per["count"]).sum())
            none = int((per["sum"] == 0).sum())
            ns = int(((stubs.run == run) & (stubs.sigma == sigma)).sum()) \
                if not stubs.empty else 0
            print(f"{run:>11} {sigma:>5g} {len(per):>7d} {ns:>6d} {allok:>8d} "
                  f"{len(per) - allok - none:>8d} {none:>6d}")

    print("\nSOLVES BY METHOD")
    print(f"{'run':>11} {'sigma':>5} {'method':>20} {'solves':>8} {'converged':>10} "
          f"{'failed':>7} {'rate':>7}")
    for run in runs:
        for sigma in sorted(solves[solves.run == run].sigma.unique()):
            for method in CONVERGENCE_METHODS:
                v = solves[(solves.run == run) & (solves.sigma == sigma)
                           & (solves.method == method)]
                if v.empty:
                    continue
                ok = int(v.success.sum())
                print(f"{run:>11} {sigma:>5g} {method:>20} {len(v):>8d} {ok:>10d} "
                      f"{len(v) - ok:>7d} {ok / len(v):>6.1%}")

    print("\nWHY THE FAILURES FAILED   (IPOPT return_status, per solve)")
    bad = solves[~solves.success]
    if bad.empty:
        print("  nothing failed")
    for run in runs:
        for sigma in sorted(bad[bad.run == run].sigma.unique()):
            for method in sweeps:
                v = bad[(bad.run == run) & (bad.sigma == sigma) & (bad.method == method)]
                if v.empty:
                    continue
                print(f"  {run}  sigma {sigma:g}  {method}   {len(v)} failed")
                for status, n in v.status.value_counts().items():
                    print(f"      {n:>6d}  {status}")

    # The list itself: one row per flight that lost at least one sweep point.
    flights = nonconverged_flights(solves)
    if not flights.empty:
        print("\nFLIGHTS THAT LOST AT LEAST ONE SWEEP POINT")
        print(f"{'run':>11} {'sigma':>5} {'method':>20} {'flights':>8} "
              f"{'of which all 6':>15}")
        for run in runs:
            for sigma in sorted(flights[flights.run == run].sigma.unique()):
                for method in sweeps:
                    v = flights[(flights.run == run) & (flights.sigma == sigma)
                                & (flights.method == method)]
                    if v.empty:
                        continue
                    print(f"{run:>11} {sigma:>5g} {method:>20} {len(v):>8d} "
                          f"{int(v.all_failed.sum()):>15d}")
        out = csv.with_name("nonconverged_flights.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        flights.to_csv(out, index=False)
        print(f"\nflight list written to {out}")

    if not stubs.empty:
        print("\nERROR STUBS   (flight never reached the solver; not a failed NLP)")
        for run in runs:
            v = stubs[stubs.run == run]
            if v.empty:
                continue
            print(f"  {run}: {len(v)}")
            for err, n in v.error.str.slice(0, 70).value_counts().items():
                print(f"      {n:>6d}  {err}")

    csv.parent.mkdir(parents=True, exist_ok=True)
    solves.to_csv(csv, index=False)
    print(f"per-solve rows written to {csv}")


# ── memory and disk audit ────────────────────────────────────────────────────
# Two different quantities, and conflating them is the easy mistake:
#
#   <phase>_peak_rss_mb   the WHOLE PROCESS peak at the end of that phase. It carries
#                         the resident meteo grid -- GridWeather keeps one full parquet
#                         per distinct sigma file, ~1.5 GB each -- so the floor is ~1.9 GB
#                         no matter which solver ran. This is the number that sizes a
#                         SLURM task.
#   <phase>_delta_rss_mb  what that phase itself added. This is the number attributable
#                         to the solver, and it is the one to compare across methods.
#
# RSS is per process, so both are only meaningful at --workers 1, which is what the
# cluster used. Disk is the gzip'd pickle actually on disk, measured with stat.
MEM_PHASES = ["meteo", "interpolant", "build", "solver"]
MEM_METHODS = ["opentop_single", "opentop_ms", "dp", "dp_muo"]


def load_memory(sigmas) -> pd.DataFrame:
    """One row per flight x sigma x method: per-phase RSS and the pickle's size.

    Reads the pickles directly rather than the timings cache: the cache keeps only the
    four timing fields, and adding eight RSS columns to it would make every figure
    re-read for a question no figure asks.
    """
    rows = []
    for method, folder, pattern in SOURCES:
        if method not in MEM_METHODS:
            continue
        n = 0
        for f in sorted(glob.glob(str(folder / pattern))):
            path = Path(f)
            with gzip.open(path, "rb") as fh:
                payload = pickle.load(fh)
            if payload["error"]:
                continue
            ctx = payload["context"]
            if not admitted(ctx.get("flight_id")):
                continue
            if float(ctx.get("sigma")) not in [float(x) for x in sigmas]:
                continue
            row = {"flight_id": ctx.get("flight_id"), "sigma": float(ctx["sigma"]),
                   "method": method, "m0": ctx.get("m0"), "actype": ctx.get("actype"),
                   "n_points": ctx.get("n_points"),
                   "disk_kb": round(path.stat().st_size / 1024, 1)}
            for ph in MEM_PHASES:
                row[f"{ph}_peak_mb"] = ctx.get(f"{ph}_peak_rss_mb")
                row[f"{ph}_delta_mb"] = ctx.get(f"{ph}_delta_rss_mb")
            peaks = [row[f"{ph}_peak_mb"] for ph in MEM_PHASES
                     if row[f"{ph}_peak_mb"] is not None]
            deltas = [row[f"{ph}_delta_mb"] for ph in MEM_PHASES
                      if row[f"{ph}_delta_mb"] is not None]
            row["peak_mb"] = max(peaks) if peaks else None
            row["delta_mb"] = sum(deltas) if deltas else None
            rows.append(row)
            n += 1
        print(f"  {method:16s} {n:>6d} pickles from {folder.relative_to(BENCH.parent)}")
    return pd.DataFrame(rows)


def report_memory(df: pd.DataFrame, csv: Path) -> None:
    if df.empty:
        print("no pickles read")
        return

    def q(v, p):
        return float(np.percentile(v.dropna(), p)) if v.notna().any() else float("nan")

    print("\nRAM  —  process peak RSS (MB)")
    print("The peak carries the resident meteo grid, so no solver goes below it; what "
          "separates them\nis how far above it they climb. 'added' is what a "
          "phase grew the process by, and its\nMEDIAN IS 0 for most flights by "
          "construction: RSS is a high-water mark that never shrinks inside a task, so "
          "the\nsecond flight onward reuses what the first already claimed. "
          "Read the upper tail of 'added', never its middle.\n")
    print(f"{'sigma':>5} {'method':>16} {'n':>6} {'peak med':>9} {'peak p90':>9} "
          f"{'peak max':>9} {'added p90':>10} {'added max':>10}")
    for sigma in sorted(df.sigma.unique()):
        for method in MEM_METHODS:
            v = df[(df.sigma == sigma) & (df.method == method)]
            if v.empty:
                continue
            print(f"{sigma:>5g} {method:>16} {len(v):>6d} "
                  f"{q(v.peak_mb, 50):>9.0f} {q(v.peak_mb, 90):>9.0f} "
                  f"{v.peak_mb.max():>9.0f} {q(v.delta_mb, 90):>10.1f} "
                  f"{v.delta_mb.max():>10.0f}")

    print("\nRAM added, by phase   (p90 MB — the median is 0, see above; "
          "blank = no such phase)")
    print(f"{'sigma':>5} {'method':>16} " + " ".join(f"{ph:>12}" for ph in MEM_PHASES))
    for sigma in sorted(df.sigma.unique()):
        for method in MEM_METHODS:
            v = df[(df.sigma == sigma) & (df.method == method)]
            if v.empty:
                continue
            cells = []
            for ph in MEM_PHASES:
                col = v[f"{ph}_delta_mb"]
                cells.append(f"{q(col, 90):>12.1f}" if col.notna().any() else f"{'':>12}")
            print(f"{sigma:>5g} {method:>16} " + " ".join(cells))

    print("\nSSD  —  gzip'd pickle on disk")
    print(f"{'sigma':>5} {'method':>16} {'n':>6} {'med kB':>9} {'p90 kB':>9} "
          f"{'max kB':>10} {'total MB':>10}")
    for sigma in sorted(df.sigma.unique()):
        for method in MEM_METHODS:
            v = df[(df.sigma == sigma) & (df.method == method)]
            if v.empty:
                continue
            print(f"{sigma:>5g} {method:>16} {len(v):>6d} {q(v.disk_kb, 50):>9.1f} "
                  f"{q(v.disk_kb, 90):>9.1f} {v.disk_kb.max():>10.1f} "
                  f"{v.disk_kb.sum() / 1024:>10.1f}")
    print(f"\ncampaign total on disk: {df.disk_kb.sum() / 1048576:.2f} GB "
          f"over {len(df)} pickles")

    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"per-flight rows written to {csv}")


def load_fronts(refresh: bool, margin: float) -> pd.DataFrame:
    """One row per flight x sigma x (graph method, OpenTOP method).

    Reads the fronts themselves rather than the timings cache, which keeps only
    one budget point per flight.
    """
    cache = dom_cache(margin)
    if cache.exists() and not refresh:
        print(f"reading cache {cache.name}  (--refresh to re-read the pickles)")
        return _in_population(pd.read_csv(cache), cache.name)

    # (method, flight_id, sigma) -> front, read once and reused by every pair.
    fronts: dict[tuple, tuple] = {}
    meta: dict[tuple, dict] = {}
    for method, folder, pattern in SOURCES:
        n = 0
        for f in sorted(glob.glob(str(folder / pattern))):
            with gzip.open(f, "rb") as fh:
                payload = pickle.load(fh)
            if payload["error"]:
                continue
            ctx = payload["context"]
            if not admitted(ctx.get("flight_id")):
                continue
            key = (ctx["flight_id"], float(ctx["sigma"]))
            fronts[(method, *key)] = front_arrays(payload, method)
            meta.setdefault(key, {"actype": ctx.get("actype"), "m0": ctx.get("m0")})
            n += 1
        print(f"  {method:16s} {n:>6d} pickles from {folder.relative_to(BENCH.parent)}")

    rows = []
    for graph, ot in DOM_PAIRS:
        for (flight_id, sigma), info in meta.items():
            g = fronts.get((graph, flight_id, sigma))
            o = fronts.get((ot, flight_id, sigma))
            if g is None or o is None or g[0].size == 0 or o[0].size == 0:
                continue          # the pair is only defined where BOTH solved it
            # Pooled front: of the joint non-dominated set, how much does each
            # method own. The baseline-free version of the same question — a
            # method contributing nothing had everything it found beaten.
            fuel = np.concatenate([g[0], o[0]])
            cont = np.concatenate([g[1], o[1]])
            src = np.concatenate([np.zeros(g[0].size, bool), np.ones(o[0].size, bool)])
            keep = pareto_mask(fuel, cont)
            rows.append({
                "flight_id": flight_id, "sigma": sigma, "graph": graph, "ot": ot,
                "actype": info["actype"], "m0": info["m0"],
                "n_g": int(g[0].size), "n_ot": int(o[0].size),
                "g_beats": n_beaten(g, o, margin),
                "ot_beats": n_beaten(o, g, margin),
                "n_union": int(keep.sum()),
                "union_from_g": int((keep & ~src).sum()),
                "union_from_ot": int((keep & src).sum()),
                # Best single trajectory each side offered, for the size of the gap.
                "g_min_fuel": float(g[0].min()), "ot_min_fuel": float(o[0].min()),
                "g_min_cont": float(g[1].min()), "ot_min_cont": float(o[1].min()),
            })

    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)   # the cache stays FULL; filter on the way out
    print(f"cache written to {cache.name}")
    return _in_population(df, cache.name)


def report_dominance(df: pd.DataFrame, margin: float) -> None:
    """The counts, per sigma and per pair.

    ``graph wins`` is the headline the question asks for: at least one graph
    trajectory beats at least one OpenTOP trajectory on both axes. ``sweeps`` is
    the strong form — every OpenTOP point is beaten. They are reported together
    because the first is easy to satisfy when the two fronts merely overlap, and
    only the second says a method was outclassed on this flight.
    """
    print(f"\ndominance after the shared post-hoc evaluation  (margin={margin:.3%})")
    print("beaten = strictly less fuel AND strictly less contrail\n")
    print(f"{'sigma':>5} {'pair':28s} {'n':>6} {'graph wins':>11} {'sweeps':>9} "
          f"{'OT wins':>9} {'neither':>9} {'joint front':>12}")
    for sigma in SIGMAS:
        for graph, ot in DOM_PAIRS:
            v = df[(df.sigma == sigma) & (df.graph == graph) & (df.ot == ot)]
            if v.empty:
                continue
            g_win = v.g_beats > 0
            sweep = v.g_beats == v.n_ot
            o_win = v.ot_beats > 0
            share = (v.union_from_g / v.n_union.where(v.n_union > 0)).median()
            print(f"{sigma:>5g} {graph + ' vs ' + ot:28s} {len(v):>6d} "
                  f"{g_win.mean():>10.1%} {sweep.mean():>8.1%} {o_win.mean():>8.1%} "
                  f"{(~g_win & ~o_win).mean():>8.1%} {share:>11.1%}")
    print("\ngraph wins  = >=1 graph point beats >=1 OpenTOP point")
    print("sweeps      = EVERY OpenTOP point of that flight is beaten")
    print("OT wins     = >=1 OpenTOP point beats >=1 graph point (not exclusive)")
    print("joint front = median share of the pooled non-dominated set owned by the graph")


# --only picks from these. The keys are the CLI vocabulary; the stems they write are
# the class attributes, which is why "dist" and "size" do not read like file names.
FIGURES = {
    # "ratio":  RatioEcdf,        # timings_ratio_ecdf
    "dist":   TimeDistribution,  # solve_time_distribution
    "count":  TimeAndCount,      # solve_time_and_count
    "size":   Scalability,       # solve_time_vs_size
    "front":  FrontSizeDistribution,  # front_size_distribution, DP_MuO only
    "front2": FrontSizeDistributionSigma2,  # front_size_distribution_sigma2, bspline
    "spread": ParetoSpread,      # pareto_spread
    "memory": MemoryDistribution,  # memory_and_disk (campaign folder, not --time)
    "memory_added": MemoryAdded,   # memory_and_disk_added, meteo floor removed
    "pareto": JointPareto,         # joint_pareto_front, four solvers pooled
    "pareto3": JointParetoNoMO,    # joint_pareto_front_3solvers, no DP_MuO
    "pareto_ms": JointParetoMultiVsSingle,  # joint_pareto_front_ms_vs_ss, the two
                                            # OpenTOP modes alone, with what each beat
    "pareto_otdp": JointParetoOpenTOPvsDP,  # joint_pareto_front_opentop_vs_dp, the
                                            # two OpenTOP modes pooled against the DP
    "pareto_otdpmo": JointParetoOpenTOPvsBothDPs,  # joint_pareto_front_opentop_vs_dp_mo,
                                                   # the same with the DP_MuO in
    "ms_buckets": HeadToHeadBuckets,  # ms_vs_ss_buckets, head to head per sweep point
    "pert_buckets": Start0VsPerturbedBuckets,  # start0_vs_perturbed_buckets, start 0 vs starts 1-4
    "pert_all": Start0VsPerturbedAll,  # start0_vs_perturbed_all, gains from 0 % up
    # OpenTOPvsDPBuckets itself (better of the two modes) is not drawn: start 0 is one
    # of the multistart's candidates, so that figure was the multistart one again.
    "otdp_single_buckets": OpenTOPSingleVsDPBuckets,  # opentop_single_vs_dp_buckets
    "otdp_ms_buckets": OpenTOPMsVsDPBuckets,          # opentop_ms_vs_dp_buckets
    "otdp_modes_buckets": OpenTOPModesVsDPBuckets,    # opentop_modes_vs_dp_buckets, both
    "otdp_modes_5pct": OpenTOPModesVsDPBuckets5pct,   # ..._gt5pct, wins by > 5 % only
    "own":    OwnPareto,           # own_pareto_<method>_<ref>, one per method
    "budget": BudgetScatter,     # pareto_budget_<method>, one per method
}


def main() -> None:
    global TLABEL, OUTDIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="re-read the pickles instead of the cache")
    parser.add_argument("--interp", choices=list(CAMPAIGNS), default="bspline",
                        help="which campaign to read and where to write: bspline -> "
                             "figures/<sweeps>/bspline_interp/, linear -> "
                             "figures/<sweeps>/linear_interp/, tol1000 -> "
                             "figures/<sweeps>/tol1000/")
    parser.add_argument("--sweeps", choices=list(SWEEP_SETS), default="all",
                        help="which sweeps represent each method. all: every sweep "
                             "pooled (the existing figures). eps_ws: OpenTOP by its "
                             "epsilon-constraint, DP by its weighted sum, "
                             "DP_MuO by its whole front. The choice names "
                             "the folder; the DP is charged its measured sweep time")
    parser.add_argument("--time", choices=list(BASES), default="solve",
                        help="what to plot as the time; the folder is named after it "
                             "(solve -> <campaign>/, the rest -> <campaign>/<choice>/)")
    parser.add_argument("--memory", action="store_true",
                        help="report peak RSS per phase and pickle size on disk "
                             "for OpenTOP singlestart, DP and DP_MuO; stops, "
                             "draws no figure")
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), default=None,
                        help="draw only these figures (default: all). "
                             "e.g. --only dist size")
    parser.add_argument("--dominance", action="store_true",
                        help="count the flights where a graph solver beats OpenTOP on "
                             "BOTH axes at once, and stop; draws no figure")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="relative margin demanded on both axes before a point "
                             "counts as beaten (0.005 = half a percent)")
    parser.add_argument("--population", nargs="+",
                        default=None, metavar="PATH",
                        help="the flight-id list(s) EVERY figure, table and csv is "
                             "restricted to; several files are read as their union. "
                             "Defaults to the 3_Results/convergence/ lists: the flights "
                             "whose OpenTOP singlestart (start 0) converged on all 13 "
                             "points at sigma=2 on the bspline campaign, so the four "
                             "solvers are always compared over one population. Pass "
                             "other lists to use them (their fingerprint then tags "
                             "every output), or 'none' for every flight the campaign ran")
    parser.add_argument("--full-only", action="store_true",
                        help="joint-front figures only (--only pareto/pareto3/"
                             "pareto_ms): drop every flight whose sweep did not "
                             "converge IN FULL for each method the figure pools -- 13 "
                             "solves per OpenTOP mode, 6 for the DP, a non-empty front "
                             "for the DP_MuO. The default population "
                             "already imposes this on the OpenTOP SINGLESTART alone, "
                             "so the flag is what extends it to the other pooled "
                             "methods. Tags the cache and the png with _full")
    parser.add_argument("--pareto-ref", choices=("own", "ws"), default="own",
                        help="what the x axis of the per-method fronts (--only own) "
                             "is measured against: own = that method's own fuel-only "
                             "trajectory (each front starts at its own origin), "
                             "ws = the OpenTOP fuel-only warm start, shared by all "
                             "four. The choice is in the file name")
    parser.add_argument("--convergence", action="store_true",
                        help="audit the OpenTOP SINGLESTART solves of this campaign: "
                             "how many of the 13 per flight converged, and the IPOPT "
                             "return_status of every one that did not; stops, draws "
                             "no figure")
    parser.add_argument("--convexity", action="store_true",
                        help="read every pickle in --dir and report the non-dominated "
                             "points that sit above the lower convex hull, i.e. the "
                             "ones no weighted sum can reach; stops, draws no figure")
    parser.add_argument("--dir", type=Path, default=BENCH / "DP_MuO",
                        help="folder --convexity reads, the method's root; --pattern "
                             "reaches into its m0 folders")
    parser.add_argument("--pattern", default="m_0*/*_dp_muo.pkl.gz",
                        help="glob inside --dir")
    parser.add_argument("--method", choices=list(FRONT_SWEEPS), default="dp_muo",
                        help="which sweeps make up the front in those pickles")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="how far above the hull a point must sit to count, on a "
                             "front rescaled to the unit square")
    parser.add_argument("--min-span", type=float, default=1e-6,
                        help="a front whose fuel or contrail varies by less than this "
                             "fraction of its own level is reported as degenerate "
                             "instead of non-convex")
    parser.add_argument("--csv", type=Path, default=None,
                        help="where --convexity writes its per-flight rows")
    args = parser.parse_args()
    select_campaign(args.interp, args.sweeps)
    select_population(args.population or DEFAULT_POPULATION)
    select_full_only(args.full_only)
    CAMPDIR.mkdir(parents=True, exist_ok=True)
    stamp_population(CAMPDIR)

    # Reads the campaign's singlestart folder, not --dir: the status lives in that
    # folder's debug/ subdirectory, so the pair has to travel together.
    if args.convergence:
        srcs = convergence_sources()
        csv = args.csv or CAMPDIR / "convergence_opentop.csv"
        report_convergence(*load_convergence(srcs, SIGMAS), srcs, csv)
        return

    # Reads the pickles, not the timings cache: RSS is not in the cache, and this
    # question is asked once per campaign rather than on every plotting run.
    if args.memory:
        csv = args.csv or CAMPDIR / "memory_usage.csv"
        report_memory(load_memory(SIGMAS), csv)
        return

    if args.convexity:
        csv = args.csv or CAMPDIR / f"convexity_{args.dir.name.lower()}_{args.method}.csv"
        report_convexity(
            load_convexity(args.dir, args.pattern, args.method,
                           args.tol, args.min_span),
            args.tol, csv)
        return

    # Its own cache and its own question: no figure, no time basis, so it returns
    # before the plotting pipeline builds anything.
    if args.dominance:
        report_dominance(load_fronts(args.refresh, args.margin), args.margin)
        return

    OUTDIR = CAMPDIR if args.time == "solve" else CAMPDIR / args.time
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OPTIMALITY_DIR.mkdir(parents=True, exist_ok=True)   # per campaign, see CAMPAIGNS

    df, TLABEL = add_time_column(load(args.refresh), args.time)
    df = df[df[TIME] > 0]             # a log axis needs positive; there are no zeros

    # --only draws a subset. Named rather than positional so a figure added or
    # reordered later cannot silently change what an existing command produces.
    wanted = set(args.only) if args.only else set(FIGURES)
    if args.only:
        print("drawing only: " + ", ".join(sorted(wanted)))
    if "spread" in wanted:      # else --only leaves an empty folder behind
        SPREADDIR.mkdir(parents=True, exist_ok=True)
    for name, plot in FIGURES.items():
        if name not in wanted:
            continue
        if plot is BudgetScatter:
            for key, label, _, _ in METHODS:
                BudgetScatter(df, key, label).run()
        elif plot is OwnPareto:
            # Its own frame, not the timings table: the fronts live in the pickles and
            # nothing about a time basis enters this figure. Read once for all four.
            fronts = own_front_frame(OwnPareto.sigma)
            for key, label, colour, _ in METHODS:
                OwnPareto(fronts, key, label, colour, args.pareto_ref).run()
        else:
            plot(df).run()
    report(df)


if __name__ == "__main__":
    main()
