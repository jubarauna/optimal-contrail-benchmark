"""
Audit result directories: what configuration did the pickles ACTUALLY run with?

None of the settings below are in the file names, so a directory can silently mix
two campaigns (is_done() skips flights already present). This reads every pickle's
context and prints, per directory, each setting's distinct values and counts.
A campaign-wide setting with more than one value is flagged MIXED.

    python 2_Run/audit_out.py 3_Results/contrail/OpenTOP/m_080 3_Results/contrail/DP/m_080
    python 2_Run/audit_out.py 3_Results/flight_selection/fuel_narrow_m080

Pass the OpenTOP and DP/DP_MuO directories of one campaign together and compare
the columns: the benchmark is only fair if they agree.
"""
from __future__ import annotations

import gzip
import pickle
import sys
from collections import Counter
from pathlib import Path

# Must be ONE value per directory. Per-flight fields (actype, mach, cruise_alt_ft...)
# are not listed: they vary by design.
CAMPAIGN_FIELDS = (
    "solver", "method", "mode", "m0", "sigma",
    "alt_buffer_ft_min", "alt_buffer_ft_max", "alt_window_snap_ft", "alt_step_ft",
    "track_mode", "track_tolerance_m",
    "interpolant_mode", "interpolant_shape", "ipopt_hessian",
    "nodes_rule", "wind_enabled",
)
PER_SIGMA_OK = {"sigma"}   # several sigmas in one directory is normal


def audit(directory: Path) -> None:
    files = sorted(directory.glob("*_sigma*.pkl.gz"))
    print(f"\n=== {directory}  ({len(files)} pickles)")
    if not files:
        return
    values = {k: Counter() for k in CAMPAIGN_FIELDS}
    # Convergence per method (fuel_only, weighted_sum, ...): solves that converged,
    # and pickles where every solve of that method converged.
    points_ok, points_all = Counter(), Counter()
    runs_ok, runs_all = Counter(), Counter()
    errors, unreadable, flights = 0, 0, set()
    for f in files:
        try:
            with gzip.open(f, "rb") as fh:
                payload = pickle.load(fh)
        except Exception:  # noqa: BLE001 - truncated pickle from a killed task
            unreadable += 1
            continue
        ctx = payload.get("context", {})
        flights.add(ctx.get("flight_id"))
        errors += payload.get("error") is not None
        for k in CAMPAIGN_FIELDS:
            if k in ctx:
                values[k][repr(ctx[k])] += 1
        for method, items in (payload.get("results") or {}).items():
            if not items:
                continue
            ok = sum(bool(r.get("success")) for r in items)
            points_ok[method] += ok
            points_all[method] += len(items)
            runs_ok[method] += ok == len(items)
            runs_all[method] += 1
    print(f"  flights {len(flights)}   with error {errors}   unreadable {unreadable}")
    if points_all:
        print("  convergence          solves converged          pickles all converged")
        for m in points_all:
            print(f"    {m:<18} {points_ok[m]:>6}/{points_all[m]:<6} {100 * points_ok[m] / points_all[m]:5.1f}%"
                  f"   {runs_ok[m]:>6}/{runs_all[m]:<6} {100 * runs_ok[m] / runs_all[m]:5.1f}%")
    for k, c in values.items():
        if not c:
            continue
        flag = "  <-- MIXED" if len(c) > 1 and k not in PER_SIGMA_OK else ""
        print(f"  {k:<20} " + "  ".join(f"{v} x{n}" for v, n in c.most_common()) + flag)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        audit(Path(d))
