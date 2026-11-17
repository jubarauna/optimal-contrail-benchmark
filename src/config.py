"""Benchmark configuration shared by every stage.
"""
from __future__ import annotations

import math

from src.paths import DATA_DIR

DEFAULT_DATA_PATH = DATA_DIR / "eu_flights_2022feb20_persistent_all.parquet"
DEFAULT_GRID_PATH = DATA_DIR / "grid_era5_smoothed.parquet.gz"

WS_GRID  = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]      # weighted-sum contrail weights
EPS_GRID = [1.01, 1.02, 1.03, 1.04, 1.05, 1.10]  # epsilon-constraint fuel-slack fractions

M0            = 0.80    # initial mass as a FRACTION OF MTOW for top.Cruise (not a Mach:
                        # the Mach goes in separately via mach_value). Override per run
                        # with --m0. Narrow bodies run at M0, wide bodies at M0_WIDEBODY.
M0_WIDEBODY   = 0.60

# The types that run at M0_WIDEBODY. Rule: the OpenAP model the solver uses for the
# type (openap.prop.aircraft(t, use_synonym=True)) has a fuselage at least 5 m wide.
# The gap is clean -- the widest narrow body is the 757 at 4.10 m, the narrowest wide
# body the 767 at 5.03 m. Written out rather than computed so the population does
# not move with the OpenAP version. Consequences of the rule, on purpose: A124, MD11
# and A306 are here through their OpenAP synonyms (747-400, 777-300, A330-200), and
# the A310 is not, because OpenAP models it as an A318. Types with no OpenAP model
# (B78X, A35K, A339, B764, A346, ...) cannot run at any mass and are not listed.
WIDEBODY_TYPES = frozenset({
    "A388",                          # A380-800   7.14 m
    "B748", "B744", "A124",          # 747        6.50 m
    "B77W", "B77L", "B772", "MD11",  # 777        6.20 m
    "A359",                          # A350-900   5.96 m
    "B789", "B788",                  # 787        5.77 m
    "A343",                          # A340-300   5.64 m
    "A333", "A332", "A306",          # A330       5.64 m
    "B763",                          # 767-300    5.03 m
})
ALT_BUFFER_FT_MAX = 4000.0  # DEFAULT optimization window = real flight mean altitude
                        # -MIN/+MAX (both the fuel-only warm start and the contrail
                        # sweeps). Override per run with --alt-buffer MIN MAX.
ALT_BUFFER_FT_MIN = 4000.0
                        # These were edited in place between campaigns and the drift
                        # was invisible: opentop_batch ran -1000/+1000, m0_060 and
                        # m0_060_wb2 ran -1000/+4000, the contrail phase and the
                        # graph-2D runs -4000/+4000. Because the filename does not
                        # encode the window and nothing echoed it, three phase-1
                        # convergence rates ended up not comparable to each other.
                        # Hence the flag: pair every window with its own --out-dir,
                        # exactly as for --m0, and read it back from
                        # context['alt_buffer_ft_min'/'_max'] instead of inferring it.
MAX_ITER      = 3000    # IPOPT max iterations per solve (caps runaway non-converging solves)

# Interpolation of the contrail cost grid. "bspline" has smooth derivatives, which is
# what IPOPT's line search needs here; "linear" is C0 — zero curvature inside a cell and
# a gradient jump at every cell face — but builds in a fraction of a second instead of
# minutes. It is NOT encoded in the output filename, so give each shape its own --out-dir.
INTERP_SHAPE  = "bspline"
TRACK_TOLERANCE_M = 10000.0  # how far (m) the optimised ground track may deviate from the
                        # real one: opentop's follow_track checks it at every collocation
                        # node and quarter-interval sample. opentop 2.7.0's own default is
                        # 1000 m, but at 1 node / 50 km (the node spacing then) that made
                        # most real tracks infeasible: 0/10 converged in the 2026-09-25 debug run, against
                        # 6/10 at 10 km. The filename does NOT encode it, so give each
                        # tolerance its own --out-dir. Override per run with
                        # --track-tolerance.
WARM_START    = False   # False: singlestart contrail sweeps solve WITHOUT the fuel-only guess as
                        # a seed (the ε-constraint still solves the fuel-only once for its F0
                        # budget). Multistart ignores it: its starts always perturb the fuel-only.
MULTISTART_ALT_OFFSETS_FT = (-2000.0, -1000.0, 1000.0, 2000.0)
                        # --multistart: every sweep solve starts once from the fuel-only
                        # trajectory, then once from its shape recentred on the real
                        # flight's mean altitude + each of these offsets (the window is
                        # cruise +/-4000 ft). The fuel-only anchor itself is one solve from
                        # opentop's default guess. The lateral track is never moved. Fixed,
                        # not random, so every flight and solve sees the same starts.
                        # Recorded per pickle as altitude_offsets_ft.
N_STARTS      = 1 + len(MULTISTART_ALT_OFFSETS_FT)
                        # starts per solve: each of the 13 sweep solves becomes N_STARTS
                        # solves, so the run costs ~N_STARTS x the singlestart one.

# ── graph-2D discretisation (DP and DP_MuO) ──────────────────────────────────
ALT_STEP_FT = 1000.0
MAX_ROC_FPM = 500.0


def level_window(cruise_alt_ft: float, ceiling_ft: float,
                 alt_buffer_ft: tuple[float, float] = (ALT_BUFFER_FT_MIN,
                                                       ALT_BUFFER_FT_MAX),
                 step_ft: float = ALT_STEP_FT) -> tuple[float, float, float]:
    """The contrail-phase altitude window, snapped outward to whole flight levels.

    The graph's altitudes are discrete (one node every ``step_ft`` from the lower
    bound), so a fractional bound would put every node between the 1000 ft levels
    the cost grid was built on. OpenTOP is continuous and has no such need, but its
    contrail sweeps use this same window so all three methods search the same
    altitude range. The stage-1 fuel-only selection does NOT: it keeps the exact
    cruise_alt -MIN/+MAX window it was run with.

    Returns ``(min_alt_ft, max_alt_ft_requested, max_alt_ft)``: the upper bound
    before and after the aircraft ceiling, itself floored to a whole level.
    """
    buf_min, buf_max = alt_buffer_ft
    min_alt_ft = math.floor((cruise_alt_ft - buf_min) / step_ft) * step_ft
    max_requested = math.ceil((cruise_alt_ft + buf_max) / step_ft) * step_ft
    max_alt_ft = min(max_requested, math.floor(ceiling_ft / step_ft) * step_ft)
    return float(min_alt_ft), float(max_requested), float(max_alt_ft)
