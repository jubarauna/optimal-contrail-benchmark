"""Filesystem anchors, resolved once.

Every script lives exactly one level below the repository root (0_Data/,
1_Flight_Selection/, 2_Run/, 4_Post_Treatment/), so none of them needs to count
``parents[]`` for itself. 3_Results/ holds output only, no scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Flight parquets, meteo grids and the prebuilt global interpolant. Not in git:
#: see 0_Data/README.md for the Zenodo deposit and how to regenerate them.
#: OPTIMAL_DATA points at a copy you already have instead of the default location.
DATA_DIR = Path(os.environ.get("OPTIMAL_DATA", ROOT / "0_Data" / "flight_data"))

#: Solver output (pickles), all under 3_Results/. Also not in git.
#:   flight_selection_2d_cruise/  stage 1: fuel_narrow_m080/, fuel_wide_m060/, selection/
#:   contrail_2d_cruise/          stage 2: DP/, DP_MuO/, OpenTOP/, each with m_060/ and m_080/
RESULTS_DIR = ROOT / "3_Results"
SELECTION_DIR = RESULTS_DIR / "flight_selection_2d_cruise"
RUN_DIR = RESULTS_DIR / "contrail_2d_cruise"
FIGURE_DIR = ROOT / "4_Post_Treatment" / "figures"


def results_root() -> Path:
    """Where the solvers' pickles actually live.

    They are not in git -- they are 11.5 GB, see 0_Data/README.md and the Zenodo deposit.
    The default is 3_Results/contrail_2d_cruise/, and OPTIMAL_RESULTS points somewhere else:

        OPTIMAL_RESULTS=/path/to/results python 4_Post_Treatment/post_treat.py

    The folder pointed to is the one containing DP/, DP_MuO/ and OpenTOP/ (the Zenodo
    deposit of the earlier campaigns has open_top_run/, dp_run/ and dijkstra_run/).
    """
    return Path(os.environ.get("OPTIMAL_RESULTS", RUN_DIR))
