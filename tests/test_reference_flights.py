"""Regression test: re-run the reference flights and compare with the saved reference.

The reference pickles are written once, BEFORE a refactor, by the commands below
(same flags as RUNS here), into 3_Results/reference/. The test re-runs the same
batches into a temporary folder and checks that every point of every front has the
same success flag, fuel and contrail.

    .venv\\Scripts\\python.exe -m pytest tests -v
"""
from __future__ import annotations

import gzip
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "3_Results" / "reference"
SIGMA = 2
REL_TOL = 1e-6          # fuel / contrail, relative

# flight_id -> m0 folder, m0 value
FLIGHTS = {
    "3950c7_02436_0": ("m_080", "0.80"),   # A318, narrow body
    "4066ab_05279_0": ("m_060", "0.60"),   # A333, wide body
}

# solver -> (batch script, extra flags, reference subfolder, pickle suffixes)
RUNS = {
    "opentop": ("2_Run/batch_opentop.py",
                ["--mode", "full", "--track-mode", "exact", "--global-interp"],
                "OpenTOP", [""]),
    "graph2d": ("2_Run/batch_graph2d.py",
                ["--methods", "dp", "dp_muo", "--global-interp"],
                "DP", ["_dp", "_dp_muo"]),
}

CASES = [(fid, solver, suffix)
         for fid in FLIGHTS
         for solver, (*_, suffixes) in RUNS.items()
         for suffix in suffixes]


def _load(path: Path) -> dict:
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def _key(rec: dict) -> tuple:
    """Identifies a point of a front: method + its parameter."""
    return (rec.get("method"), rec.get("coef"), rec.get("fuel_slack_frac"),
            rec.get("point"))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Run each (solver, flight) batch once; return a lookup of its output folder."""
    cache: dict[tuple[str, str], Path] = {}

    def _run(solver: str, fid: str) -> Path:
        if (solver, fid) not in cache:
            script, flags, _, _ = RUNS[solver]
            m_dir, m0 = FLIGHTS[fid]
            out = tmp_path_factory.mktemp(f"{solver}_{fid}")
            cmd = [sys.executable, script, *flags, "--sigmas", str(SIGMA),
                   "--m0", m0, "--flights", fid, "--out-dir", str(out)]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            assert proc.returncode == 0, f"{' '.join(cmd)}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
            cache[(solver, fid)] = out
        return cache[(solver, fid)]

    return _run


@pytest.mark.parametrize("fid, solver, suffix", CASES)
def test_matches_reference(run_dir, fid, solver, suffix):
    m_dir, _ = FLIGHTS[fid]
    name = f"{fid}_sigma{SIGMA}{suffix}.pkl.gz"
    ref_path = REFERENCE / RUNS[solver][2] / m_dir / name
    if not ref_path.exists():
        pytest.skip(f"no reference {ref_path.relative_to(ROOT)}: run the reference commands first")

    new_path = run_dir(solver, fid) / name
    assert new_path.exists(), f"batch did not write {name}"

    ref, new = _load(ref_path), _load(new_path)
    assert new["error"] is None, new["error"]

    ref_pts = {_key(r): r for r in ref["summary"]}
    new_pts = {_key(r): r for r in new["summary"]}
    assert new_pts.keys() == ref_pts.keys(), "different set of front points"

    for k, r in ref_pts.items():
        n = new_pts[k]
        assert n["success"] == r["success"], f"{k}: success"
        if r["success"]:
            assert n["fuel_kg"] == pytest.approx(r["fuel_kg"], rel=REL_TOL), f"{k}: fuel_kg"
            assert n["contrail"] == pytest.approx(r["contrail"], rel=REL_TOL, abs=1e-9), f"{k}: contrail"
