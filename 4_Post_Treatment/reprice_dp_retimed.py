"""
reprice_dp_retimed.py — re-price existing DP pickles with the re-timed post-hoc.

The DP's waypoint times come from one TAS (the fixed Mach at the real mean cruise
altitude), whatever altitude the path takes. ``evaluate_path`` now re-times each path
at its own altitudes (``flown_ts``) before pricing it; this script applies the same
fix to pickles written before that, without re-solving: the altitude profiles are
the DP's, only their times, fuel and contrail change.

For every <id>_sigma<s>_dp.pkl.gz in --src/m_0*/ it writes the same file under --dst:
  results[*]  ts_s, fuel_kg, contrail re-priced; the old values kept in
              meta["fuel_kg_gridtime"] / meta["contrail_gridtime"] / meta["ts_s_gridtime"]
  summary     fuel_kg, contrail re-priced
  context     the re-pricing time ADDED to posthoc_s and to the sweep's
              <phase>_posthoc_s (anchor = fuel-only, ws = weighted sum);
              posthoc_retimed=True, posthoc_retime_s = the added time

Before re-pricing, each record is priced once with its ORIGINAL times and checked
against the stored fuel / contrail, so a route rebuilt differently from the batch's
is caught rather than silently mixed in (that check is not timed).

    python 4_Post_Treatment/reprice_dp_retimed.py
    python 4_Post_Treatment/reprice_dp_retimed.py --src 3_Results/contrail_2d_cruise/DP \
        --dst 3_Results/contrail_2d_cruise/DP_retimed
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from openap.aero import ft  # noqa: E402

from src.config import DEFAULT_DATA_PATH  # noqa: E402
from src.data.traffic import FlightData  # noqa: E402
from src.data.flights import ensure_flight_slice  # noqa: E402
from src.contrails.interpolant import load_global_interpolant, shift_interpolant  # noqa: E402
from src.data.route import real_flight_route  # noqa: E402
from src.postprocess.evaluation import (  # noqa: E402
    contrail_integral, flown_ts, route_latlon,
)
from src.performance.aircraft import fuel_burn  # noqa: E402

SRC = PROJECT_ROOT / "3_Results" / "contrail_2d_cruise" / "DP"
PHASE = {"fuel_only": "anchor", "weighted_sum": "ws", "epsilon_constraint": "eps"}
CHECK_RTOL = 1e-6

_INTERP: dict = {}


def global_interp(sigma: float):
    if sigma not in _INTERP:
        _INTERP[sigma] = load_global_interpolant(sigma, out_dir=PROJECT_ROOT / "0_Data" / "flight_data")
    return _INTERP[sigma]


def build_route(ctx: dict, slice_path: Path):
    fd = FlightData(start=str(ctx["begin"]), stop=str(ctx["end"]), path=str(slice_path),
                    base_id=ctx["flight_id"], aircraft_type=ctx["actype"])
    route, total_km = real_flight_route(fd.flight, mach=ctx["mach"],
                                        avg_alt_m=ctx["cruise_alt_ft"] * ft,
                                        roc=ctx["max_roc_fpm"], alt_step_ft=ctx["alt_step_ft"])
    if abs(total_km - ctx["route_km"]) > 1e-6 * ctx["route_km"]:
        raise ValueError(f"route {total_km:.3f} km != stored {ctx['route_km']:.3f} km")
    return route


def reprice_flight(job) -> list[str]:
    files, slice_path, dst_root = job
    msgs, route = [], None
    for f in files:
        f = Path(f)
        with gzip.open(f, "rb") as fh:
            payload = pickle.load(fh)
        out = Path(dst_root) / f.parent.name / f.name
        out.parent.mkdir(parents=True, exist_ok=True)
        ctx = payload.get("context") or {}
        if payload.get("error") or ctx.get("posthoc_retimed"):
            with gzip.open(out, "wb") as fh:
                pickle.dump(payload, fh)
            continue
        try:
            if route is None:
                with open(os.devnull, "w") as dn:      # real_flight_route prints a line
                    so, sys.stdout = sys.stdout, dn
                    try:
                        route = build_route(ctx, Path(slice_path))
                    finally:
                        sys.stdout = so
            g, gmeta = global_interp(float(ctx["sigma"]))
            interp = shift_interpolant(g, ctx["begin"], gmeta)
            mach, m0, actype = ctx["mach"], ctx["mass_init_kg"], ctx["actype"]

            added = defaultdict(float)
            new_vals = {}
            for sweep, recs in payload["results"].items():
                for r in recs:
                    if not r.get("success") or r.get("alt_ft") is None:
                        continue
                    alt, dist, ts = (np.asarray(r[k], dtype=float) for k in ("alt_ft", "dist_km", "ts_s"))
                    lat, lon = route_latlon(dist, route)
                    # check: original times reproduce the stored numbers (untimed)
                    f0, _ = fuel_burn(alt, ts, mach, m0, actype)
                    c0 = contrail_integral(lat, lon, alt, ts, interp)
                    if (abs(f0 - r["fuel_kg"]) > CHECK_RTOL * abs(r["fuel_kg"])
                            or abs(c0 - r["contrail"]) > max(CHECK_RTOL * abs(r["contrail"]), 1e-6)):
                        raise ValueError(f"check failed: fuel {f0:.4f} vs {r['fuel_kg']:.4f}, "
                                         f"contrail {c0:.6f} vs {r['contrail']:.6f}")
                    # the re-timed post-hoc (timed)
                    t0 = time.perf_counter()
                    ts2 = flown_ts(alt, dist, mach)
                    lat, lon = route_latlon(dist, route)
                    f2, _ = fuel_burn(alt, ts2, mach, m0, actype)
                    c2 = contrail_integral(lat, lon, alt, ts2, interp)
                    added[PHASE.get(sweep, sweep)] += time.perf_counter() - t0
                    r["meta"]["fuel_kg_gridtime"] = r["fuel_kg"]
                    r["meta"]["contrail_gridtime"] = r["contrail"]
                    r["meta"]["ts_s_gridtime"] = r["ts_s"]
                    r["fuel_kg"], r["contrail"], r["ts_s"] = float(f2), float(c2), ts2
                    new_vals[(sweep, r["meta"].get("coef", r["meta"].get("point")))] = (float(f2), float(c2))

            for s in payload.get("summary") or []:
                key = (s.get("method"), s.get("coef", s.get("point")) if s.get("method") != "fuel_only" else None)
                if key[0] == "fuel_only":
                    key = ("fuel_only", next((k[1] for k in new_vals if k[0] == "fuel_only"), None))
                if key in new_vals:
                    s["fuel_kg"], s["contrail"] = new_vals[key]

            total = float(sum(added.values()))
            for p, t in added.items():
                if p not in PHASE.values():
                    continue
                ctx[f"{p}_posthoc_s"] = float(ctx.get(f"{p}_posthoc_s") or 0.0) + t
            ctx["posthoc_s"] = float(ctx.get("posthoc_s") or 0.0) + total
            ctx["posthoc_retimed"] = True
            ctx["posthoc_retime_s"] = total
            with gzip.open(out, "wb") as fh:
                pickle.dump(payload, fh)
        except Exception as e:  # noqa: BLE001 — report and go on
            msgs.append(f"{f.name}: {type(e).__name__}: {e}")
    return msgs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, default=SRC)
    p.add_argument("--dst", type=Path, default=None, help="default: <src>_retimed")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--subdir", default="m_0*", help="e.g. m_060")
    p.add_argument("--pattern", default="*_dp.pkl.gz", help="e.g. *_dp_muo.pkl.gz")
    p.add_argument("--limit", type=int, default=None, help="first N flights only")
    args = p.parse_args()
    dst = args.dst or args.src.with_name(args.src.name + "_retimed")

    by_flight = defaultdict(list)
    for f in sorted(args.src.glob(f"{args.subdir}/{args.pattern}")):
        by_flight[(f.parent.name, f.name.split("_sigma")[0])].append(str(f))
    keys = sorted(by_flight)[:args.limit] if args.limit else sorted(by_flight)
    print(f"{sum(len(by_flight[k]) for k in keys)} pickles, {len(keys)} flights -> {dst}")

    full_df = pd.read_parquet(DEFAULT_DATA_PATH)
    cache = args.src / "_flight_cache"
    jobs = [(by_flight[k], str(ensure_flight_slice(full_df, cache, k[1])), str(dst)) for k in keys]
    del full_df

    errors = []
    with Pool(args.workers) as pool:
        for i, msgs in enumerate(pool.imap_unordered(reprice_flight, jobs), 1):
            errors += msgs
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} flights   ({len(errors)} errors)", flush=True)
    for m in errors[:20]:
        print("  [error]", m)
    if errors:
        print(f"{len(errors)} pickles not re-priced (not written)")


if __name__ == "__main__":
    main()
