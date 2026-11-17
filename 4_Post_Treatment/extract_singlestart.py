"""
extract_singlestart.py — the single-start campaign, read out of the multistart one.

Start 0 of every multistart solve IS a single-start solve: it starts from the
fuel-only trajectory as given (initial_guess "fuel_only", altitude offset 0), and the
other starts only add altitude-shifted copies of it. So the single-start result of
each WS / EC sweep point is already on disk, as candidate start_index 0, and needs no
re-run. Reading it from the same pickles also makes single vs multi a comparison of the
SAME solves, with and without the extra starts.

Not the separate Singlestart/ campaign: that one ran with WARM_START=False, i.e. from
opentop's default guess, not from the fuel-only trajectory.

For every <id>_sigma<s>_ms.pkl.gz in a multistart folder this writes
<id>_sigma<s>.pkl.gz (no _ms, so the singlestart loaders read it) with the same payload
as batch_opentop: each WS / EC record is rebuilt from start 0 exactly as
OpenTOPOptimizer._traj_to_result builds one, and summary is rebuilt from the records.
The fuel-only record is copied: it is a single solve under multistart too.

Timing in context is start 0's, not the multistart run's: ws_s / eps_s are the sums of
start 0's wall times, solver_s = anchor_s + ws_s + eps_s. Everything else in context
(meteo, interpolant, memory peaks) is the multistart run's, which did that work once.

Usage
-----
    python 4_Post_Treatment/extract_singlestart.py
    python 4_Post_Treatment/extract_singlestart.py --src 3_Results/contrail/OpenTOP_linear \
        --dst 3_Results/contrail/OpenTOP_linear/Singlestart_start0
"""

from __future__ import annotations
import argparse
import gzip
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openap.aero import ft  # noqa: E402
from src.optimization.base import OptResult  # noqa: E402

OPENTOP = PROJECT_ROOT / "3_Results" / "contrail" / "OpenTOP"
SWEEPS = ("weighted_sum", "epsilon_constraint")


def start0_record(rec: dict) -> dict:
    """One WS / EC record, rebuilt from its start-0 candidate."""
    meta = rec["meta"]
    (c0,) = [c for c in meta["candidates"] if c["start_index"] == 0]
    if c0["initial_guess"] != "fuel_only":
        raise ValueError(f"start 0 was seeded from {c0['initial_guess']!r}, not fuel_only")
    traj = c0["trajectory"]
    x, y = traj["x"].to_numpy(), traj["y"].to_numpy()
    dist_km = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))]) / 1000.0
    # Sweep coordinates (coef, fuel_slack_frac, fuel_budget_kg) belong to the point, not
    # to the start, so they carry over; the solve's own fields come from start 0.
    new_meta = {k: v for k, v in meta.items()
                if k in ("optimizer", "method", "coef", "fuel_slack_frac", "fuel_budget_kg")}
    new_meta.update({
        "time_s": c0["wall_time_s"], "objective": c0["objective"], "n_iter": c0["iters"],
        "trajectory": traj,
        # grid_cost_exact is what _multi_start leaves in cruise.grid_cost_value, i.e.
        # the quadrature _traj_to_result reports as the contrail.
        "contrail_source": "quadrature",
        "contrail_left_endpoint": float(np.nansum(traj["grid_cost"])),
        "multistart": False, "candidates": None,
        "return_status": c0.get("return_status"),
    })
    return OptResult(
        alt_ft=traj["h"].to_numpy() / ft, dist_km=dist_km, ts_s=traj["ts"].to_numpy(),
        lat=traj["latitude"].to_numpy(), lon=traj["longitude"].to_numpy(),
        fuel_kg=float(traj["mass"].iloc[0] - traj["mass"].iloc[-1]),
        contrail=float(c0["grid_cost_exact"]), cost=c0["objective"],
        success=bool(c0["success"]), meta=new_meta,
    ).as_dict()


def convert(task) -> str:
    src, dst = task
    with gzip.open(src, "rb") as f:
        payload = pickle.load(f)
    ctx = dict(payload["context"])
    if payload["error"] is None:
        results = {"fuel_only": [dict(r, meta=dict(r["meta"], multistart=False))
                                 for r in payload["results"].get("fuel_only", [])]}
        for m in SWEEPS:
            results[m] = [start0_record(r) for r in payload["results"].get(m, [])]
        summary = [OptResult(**r).to_row() for m in ("fuel_only", *SWEEPS) for r in results[m]]
        ws_s = sum(r["meta"]["time_s"] for r in results["weighted_sum"])
        eps_s = sum(r["meta"]["time_s"] for r in results["epsilon_constraint"])
        ctx.update(ws_s=ws_s, eps_s=eps_s, solver_s=(ctx.get("anchor_s") or 0.0) + ws_s + eps_s)
    else:
        results, summary = payload["results"], payload["summary"]
    ctx.update(multistart=False, n_starts=1, singlestart_source="multistart_start0",
               multistart_file=src.name)
    out = {"context": ctx, "results": results, "summary": summary, "error": payload["error"]}
    tmp = dst.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=5) as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst)
    return dst.name


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", default=str(OPENTOP), help="folder holding the m_0* multistart folders")
    p.add_argument("--dst", default=str(OPENTOP / "Singlestart_start0"))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    src_root, dst_root = Path(a.src), Path(a.dst)
    tasks = []
    for mdir in sorted(d for d in src_root.glob("m_0*") if d.is_dir()):
        out = dst_root / mdir.name
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(mdir.glob("*_sigma*_ms.pkl.gz")):
            dst = out / f.name.replace("_ms.pkl.gz", ".pkl.gz")
            if a.overwrite or not dst.exists():
                tasks.append((f, dst))
    print(f"{len(tasks)} pickles to convert -> {dst_root}")
    with Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(convert, tasks, chunksize=8), 1):
            if i % 1000 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}")


if __name__ == "__main__":
    main()
