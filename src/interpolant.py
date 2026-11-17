"""The whole-domain cost interpolant: build it once, shift it per flight.

The CLI that writes the .casadi/.meta.json pair is 0_Data/build_global_interpolant.py;
this module is the importable half, used by both solver batches.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import casadi as ca
import pandas as pd
import opentop as top

# Windows consoles default to cp1252 and choke on the → / σ characters printed
# below — force UTF-8 so a normal run never crashes on encoding.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 — best effort; older/foreign stream types
        pass

from src.paths import DATA_DIR

_DATA_ROOT = DATA_DIR
DEFAULT_GRID_PATH = _DATA_ROOT / "grid_era5_smoothed.parquet.gz"


SHAPES = ("bspline", "linear")


def _paths(sigma: float, out_dir: Path, shape: str = "bspline") -> tuple[Path, Path]:
    """Cache (.casadi) and metadata (.meta.json) paths for a given sigma and shape.

    The bspline keeps its original name (global_interp_sigma<s>), which every batch
    and sbatch already reads; any other shape carries it in the stem
    (global_linear_interp_sigma<s>), so the two live side by side in one folder.
    Without the shape in the name a linear build found the bspline file, took it as
    a cache hit and handed it back under the wrong label.
    """
    if shape not in SHAPES:
        raise ValueError(f"shape={shape!r}; expected one of {SHAPES}")
    prefix = "global" if shape == "bspline" else f"global_{shape}"
    stem = f"{prefix}_interp_sigma{sigma:g}"
    return out_dir / f"{stem}.casadi", out_dir / f"{stem}.meta.json"


def build_global_interpolant(
    grid_path: Path = DEFAULT_GRID_PATH,
    sigma: float = 2.0,
    cost_col: str = "cost",
    out_dir: Path | None = None,
    shape: str = "bspline",
    rebuild: bool = False,
) -> tuple[ca.Function, dict]:
    """Build (or load from cache) the single whole-batch interpolant.

    Reads the fixed-lattice grid, references ``ts`` to the grid's first timestamp
    (absolute time), fits one bspline over the whole domain and caches it. The
    returned ``meta`` carries ``t0`` and the domain bounds needed to build the
    per-flight time shift and to check a flight is in-domain.

    Args:
        grid_path: The precomputed grid (grid_era5_smoothed.parquet.gz).
        sigma: Only labels the output files — the smoothing itself must already
            be baked into ``cost_col`` of ``grid_path``.
        cost_col: Grid column to use as the cost value. ``"cost"`` is the σ=2
            smoothed field; ``"persistent"`` is the raw σ=0 boolean.
        out_dir: Where to write the .casadi + .meta.json (default: next to grid).
        shape: "bspline" (recommended) or "linear".
        rebuild: If True, ignore any existing cache and rebuild.

    Returns:
        (interpolant, meta) — a CasADi Function `(lon, lat, height_m, ts_abs_s)`
        → cost, and the metadata dict.
    """
    out_dir = Path(out_dir) if out_dir is not None else _DATA_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path, meta_path = _paths(sigma, out_dir, shape)

    if cache_path.exists() and meta_path.exists() and not rebuild:
        print(f"[build] cache hit → {cache_path.name}")
        return load_global_interpolant(sigma=sigma, out_dir=out_dir, shape=shape)
    if rebuild and cache_path.exists():
        # cached_interpolant_from_dataframe below LOADS any file at cache_path without
        # checking it (opentop/tools.py), so a rebuild must remove it first or it
        # returns the old fit under a fresh meta.json.
        cache_path.unlink()

    print(f"[build] reading grid {grid_path} ...")
    df = pd.read_parquet(grid_path)
    if cost_col not in df.columns:
        raise KeyError(f"cost_col={cost_col!r} not in grid columns {list(df.columns)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    t0 = df["timestamp"].min()
    # Absolute time axis: ts=0 is the grid's first timestamp, shared by ALL flights.
    df["ts"] = (df["timestamp"] - t0).dt.total_seconds()
    # interpolant_from_dataframe reads a 'cost' column; map the chosen source into it.
    df["cost"] = df[cost_col].astype(float)

    n = len(df)
    dims = (df["longitude"].nunique(), df["latitude"].nunique(),
            df["height"].nunique(), df["ts"].nunique())
    expected = dims[0] * dims[1] * dims[2] * dims[3]
    if expected != n:
        raise ValueError(
            f"grid is not a complete rectangular lattice: "
            f"{dims[0]}×{dims[1]}×{dims[2]}×{dims[3]} = {expected} != {n} rows. "
            "interpolant_from_dataframe needs a full grid.")

    print(f"[build] fitting {shape} over {n:,} nodes "
          f"(lon×lat×h×ts = {dims[0]}×{dims[1]}×{dims[2]}×{dims[3]}) — this can take minutes ...")
    t_start = time.perf_counter()
    interp = top.tools.cached_interpolant_from_dataframe(df, cache_path, shape=shape)
    build_s = time.perf_counter() - t_start
    print(f"[build] done in {build_s:.1f}s → {cache_path.name}")

    meta = {
        "sigma": sigma,
        "cost_col": cost_col,
        "shape": shape,
        # The one-off fit, recorded because it replaces every per-flight build: the
        # campaign's interpolant cost is this, once per sigma, plus the per-flight
        # shift in interpolant_s.
        "build_s": round(build_s, 1),
        "grid_path": str(grid_path),
        "cache_path": str(cache_path),
        "t0": t0.isoformat(),                 # absolute-time origin for the Δ shift
        "n_nodes": int(n),
        "dims": {"lon": dims[0], "lat": dims[1], "height": dims[2], "ts": dims[3]},
        "bounds": {
            "lon":    [float(df["longitude"].min()), float(df["longitude"].max())],
            "lat":    [float(df["latitude"].min()),  float(df["latitude"].max())],
            "height": [float(df["height"].min()),    float(df["height"].max())],
            "ts":     [float(df["ts"].min()),        float(df["ts"].max())],
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[build] wrote metadata → {meta_path.name}")
    return interp, meta


def load_global_interpolant(
    sigma: float = 2.0, out_dir: Path | None = None, shape: str = "bspline"
) -> tuple[ca.Function, dict]:
    """Load a previously built interpolant + its metadata from disk."""
    out_dir = Path(out_dir) if out_dir is not None else _DATA_ROOT
    cache_path, meta_path = _paths(sigma, out_dir, shape)
    if not (cache_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"No global {shape} interpolant for σ={sigma:g} in {out_dir}. "
            f"Run build_global_interpolant(sigma={sigma:g}, shape={shape!r}) first.")
    meta = json.loads(meta_path.read_text())
    # The file name says which shape it should be; the meta says which one was fitted.
    # A disagreement is a file renamed or copied by hand, and loading it would put one
    # field under the other's label.
    if meta.get("shape", "bspline") != shape:
        raise ValueError(f"{meta_path.name} records shape={meta.get('shape')!r}, "
                         f"but shape={shape!r} was asked for")
    interp = top.tools.load_interpolant(cache_path)
    return interp, meta


def departure_offset_s(departure, meta: dict) -> float:
    """Δ = departure − t0 (seconds): the shift that turns opentop's elapsed-time
    ``ts`` state into the absolute time the global interpolant is indexed on."""
    t0 = pd.Timestamp(meta["t0"])
    dep = pd.Timestamp(departure)
    dep = dep.tz_localize("UTC") if dep.tzinfo is None else dep.tz_convert("UTC")
    return float((dep - t0).total_seconds())


def shift_interpolant(interp: ca.Function, departure, meta: dict) -> ca.Function:
    """Wrap the global (absolute-time) interpolant for ONE flight.

    opentop's ``ts`` state is 0 at departure; the global interpolant is indexed
    on absolute time. This returns a Function with the SAME 4-vector interface
    `[lon, lat, h, ts_elapsed]` that internally queries the cached bspline at
    ``ts_elapsed + Δ``. It is a symbolic composition — no refit — so it is
    effectively free to build per flight.
    """
    delta = departure_offset_s(departure, meta)
    X = ca.MX.sym("x", 4)                                  # [lon, lat, h, ts_elapsed]
    Xa = ca.vertcat(X[0], X[1], X[2], X[3] + delta)        # → absolute time
    return ca.Function("grid_cost", [X], [interp(Xa)])


