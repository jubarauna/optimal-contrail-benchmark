"""
build_global_interpolant.py — build ONE contrail-cost interpolant for the whole
batch, so the expensive bspline fit is paid once instead of per flight.

Why one interpolant works for all flights
-----------------------------------------
The per-flight code builds a fresh interpolant because it references time to each
flight's departure (``ts = timestamp − departure``), and opentop queries the grid
at elapsed flight time (``ts`` state starts at 0 at departure). Two flights that
depart at different wall-clock times therefore need different ``ts→field``
alignment — so a departure-relative interpolant cannot be shared.

The fix: build the interpolant ONCE in **absolute time**, with ``ts`` referenced
to a fixed global ``t0`` (the grid's first timestamp) instead of any flight. The
grid values are the same field for everyone, so this single interpolant covers
every flight in the domain. Per flight you then don't refit — you compose the
cached bspline with a constant time shift ``Δ = departure − t0`` (see
:func:`shift_interpolant`), which costs microseconds.

Outputs (next to the grid, in 0_Data/flight_data/)
------------------------------------------------
    global_interp_sigma<σ>.casadi            # the serialized bspline (reused via cache)
    global_interp_sigma<σ>.meta.json         # t0 + domain bounds, for the Δ shift + checks
    global_linear_interp_sigma<σ>.casadi     # the same with --shape linear, side by side
    global_linear_interp_sigma<σ>.meta.json

Usage
-----
    # the three sigmas of the campaign, from grid_era5_smoothed.parquet.gz
    python 0_Data/build_global_interpolant.py --sigma 0 1 2 --rebuild

    # the linear one, same three sigmas, beside the bspline
    python 0_Data/build_global_interpolant.py --sigma 0 1 2 --shape linear

    # on the cluster
    sbatch 0_Data/build_global_interpolant.sbatch

The grid column follows from sigma, exactly as for the per-flight fit
(src.tools.graph_based_tools.sigma_sources): 0 -> 'persistent', 2 -> 'cost',
1 -> 'cost_sigma1'. --cost-col overrides it for a single sigma.

Later, in an optimization
-------------------------
    from src.contrails.interpolant import (
        load_global_interpolant, shift_interpolant)

    interp, meta = load_global_interpolant(sigma=2)                    # bspline
    interp, meta = load_global_interpolant(sigma=2, shape="linear")    # linear
    per_flight = shift_interpolant(interp, departure=fd.start, meta=meta)
    # hand `per_flight` to OpenTOPOptimizer(interpolant=per_flight, ...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The importable half lives in src/interpolant.py: both solver batches load the
# interpolant this script writes, and neither can import 0_Data (not an identifier).
from src.contrails.interpolant import DEFAULT_GRID_PATH, build_global_interpolant  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-path", default=str(DEFAULT_GRID_PATH))
    p.add_argument("--sigma", type=float, nargs="+", default=[2.0],
                   help="One or more sigmas, built one after the other.")
    p.add_argument("--cost-col", default=None,
                   help="Grid column used as cost. Default: the one the per-flight fit "
                        "uses for that sigma (sigma_sources). Only with a single --sigma.")
    p.add_argument("--out-dir", default=None, help="Default: next to the grid file.")
    p.add_argument("--shape", default="bspline", choices=["bspline", "linear"])
    p.add_argument("--rebuild", action="store_true", help="Ignore any existing cache.")
    args = p.parse_args()
    if args.cost_col and len(args.sigma) > 1:
        p.error("--cost-col names one column, so it takes a single --sigma")

    from src.tools.graph_based_tools import sigma_sources
    for sigma in args.sigma:
        cost_col = args.cost_col or sigma_sources(Path(args.grid_path), [sigma])[float(sigma)][1]
        print(f"\n=== sigma={sigma:g}  column={cost_col} ===")
        interp, meta = build_global_interpolant(
            grid_path=Path(args.grid_path), sigma=sigma, cost_col=cost_col,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            shape=args.shape, rebuild=args.rebuild,
        )

        # Sanity: evaluate at the domain centre and show how a per-flight shift is used.
        b = meta["bounds"]
        mid = [sum(b["lon"]) / 2, sum(b["lat"]) / 2, sum(b["height"]) / 2, sum(b["ts"]) / 2]
        print(f"[check] interp at domain-centre {['%.1f' % v for v in mid]} = "
              f"{float(interp(mid)):.4f}")
        print(f"[check] t0 = {meta['t0']}  |  domain lon{b['lon']} lat{b['lat']} "
              f"h{b['height']} ts{b['ts']}  |  build {meta.get('build_s')} s")


if __name__ == "__main__":
    main()
