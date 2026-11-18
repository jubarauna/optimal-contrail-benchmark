"""Flight index and per-flight parquet cache, shared by every solver batch."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_flight_index(data_path: Path) -> pd.DataFrame:
    """One row per flight: begin/end timestamps, typecode, point count."""
    print(f"Loading flight index from {data_path} ...")
    df = pd.read_parquet(data_path, columns=["flight_id", "timestamp", "typecode"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    g = df.groupby("flight_id")
    idx = pd.DataFrame({
        "begin":    g["timestamp"].min(),
        "end":      g["timestamp"].max(),
        "typecode": g["typecode"].first(),
        "n_points": g["timestamp"].count(),
    }).reset_index().sort_values("flight_id").reset_index(drop=True)
    print(f"  {len(idx)} flights found.\n")
    return idx


def ensure_flight_slice(full_df: pd.DataFrame, cache_dir: Path, flight_id: str) -> Path:
    """Write (once) a single-flight parquet so FlightData(base_id=...) is cheap
    instead of re-loading the full multi-flight parquet for every flight."""
    path = cache_dir / f"{flight_id}.parquet"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        full_df[full_df["flight_id"] == flight_id].to_parquet(path)
    return path

