"""
filter_benchmarking.py — extract benchmark flights from a pre-processed
local parquet that already carries contrail conditions (persistent, issr, sac).

Reads eu_flights_2022feb20_filter_resample_meteo.parquet.gz, keeps only
flights where at least one trajectory point has persistent == True, and writes
them all to one parquet (plus a _summary.csv).

Usage (the benchmark input, 5 919 flights):
    python 0_Data/filter_benchmarking.py --recover-typecodes \
        --out-path 0_Data/flight_data/eu_flights_2022feb20_persistent_all.parquet

    # the flights the typecode lookup miss threw away, as their own parquet
    python 0_Data/filter_benchmarking.py --only-recovered
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows consoles default to cp1252 and choke on the → / % characters printed
# below — force UTF-8 so a normal run never crashes on encoding.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 — best effort; older/foreign stream types
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DATA_DIR  # noqa: E402

_DATA_ROOT = DATA_DIR
DEFAULT_DATA_PATH = _DATA_ROOT / "eu_flights_2022feb20_filter_resample_meteo.parquet.gz"
DEFAULT_OUT_PATH  = _DATA_ROOT / "eu_flights_2022feb20_persistent.parquet"
RECOVERED_OUT_PATH = _DATA_ROOT / "eu_flights_2022feb20_persistent_recovered.parquet"

# Columns the downstream optimization actually consumes. A flight with a NaN in
# ANY of these is unusable (bad trajectory point, missing wind, or an unknown
# aircraft type that we no longer want to silently default to B738), so the whole
# flight is dropped before benchmarking.
REQUIRED_COLS = [
    "latitude", "longitude", "altitude", "timestamp",   # trajectory
    "u_component_of_wind", "v_component_of_wind",        # wind (enable_wind)
    "persistent",                                        # contrail condition
    "typecode",                                          # aircraft type (no defaulting)
]


def drop_nan_flights(
    df: pd.DataFrame, required_cols: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Drop every flight that has a NaN in any ``required_cols`` column.

    A single bad point disqualifies the whole flight_id (the optimizer runs on
    the entire trajectory, so a mid-flight NaN would break it). Returns the
    cleaned frame and the list of dropped flight_ids.
    """
    cols = required_cols if required_cols is not None else REQUIRED_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"required columns not in data: {missing}")

    bad_mask = df[cols].isna().any(axis=1)
    bad_ids = df.loc[bad_mask, "flight_id"].unique().tolist()
    kept = df[~df["flight_id"].isin(bad_ids)].copy()
    print(f"  NaN filter: dropped {len(bad_ids)} of {df['flight_id'].nunique()} "
          f"flights with NaN in {cols}")
    return kept, bad_ids


def fill_missing_typecodes(df: pd.DataFrame) -> list[str]:
    """Fill NaN ``typecode`` from ``traffic.data.aircraft``, in place.

    ``typecode`` is the only required column that ever drops a flight (3 149 of
    14 000), and it is a lookup miss rather than missing OpenSky data: the merge
    upstream used ``aircraft_db.csv`` (129 228 records) while traffic ships ~520 000.
    Those flights have zero NaN in trajectory, wind and ``persistent``, so nothing
    has to be re-fetched or re-processed to bring them back.

    Returns the flight_ids that were filled.
    """
    from traffic.data import aircraft

    db = aircraft.data[["icao24", "typecode"]].dropna(subset=["typecode"])
    lookup = (db.assign(icao24=db["icao24"].str.lower().str.strip(),
                        typecode=db["typecode"].str.upper().str.strip())
                .query("typecode != ''")
                .drop_duplicates("icao24")
                .set_index("icao24")["typecode"])

    missing = df["typecode"].isna()
    filled = df.loc[missing, "icao24"].str.lower().map(lookup)
    df.loc[missing, "typecode"] = filled
    recovered = sorted(df.loc[missing & df["typecode"].notna(), "flight_id"].unique())
    still_missing = df.loc[df["typecode"].isna(), "flight_id"].nunique()
    print(f"  typecode recovery: {len(recovered)} flights filled from "
          f"traffic.data.aircraft, {still_missing} still unknown")
    return recovered


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _haul_category(dist_km: float) -> str:
    if dist_km < 1500:
        return "short"
    if dist_km < 3500:
        return "medium"
    return "long"


def load_persistent_flights(
    data_path: str | Path = DEFAULT_DATA_PATH,
    min_persistent_frac: float = 0.0,
    recover_typecodes: bool = False,
    only_recovered: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the subset of the parquet that contains persistent-contrail flights.

    A flight qualifies when the fraction of its points with persistent == True
    is >= min_persistent_frac (default 0.0 means at least one point).

    ``recover_typecodes`` looks the missing aircraft types up in traffic's database
    first, so the flights they would otherwise disqualify stay in. ``only_recovered``
    then keeps ONLY those flights, which is how the recovered population is written
    to its own parquet: everything already computed from the original one stays
    valid and the new flights are additive.
    """
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path)

    recovered = fill_missing_typecodes(df) if recover_typecodes else []

    # Remove flights with NaN in any required optimization input (incl. unknown
    # aircraft type) before selecting on persistent contrails.
    df, _ = drop_nan_flights(df)

    if only_recovered:
        df = df[df["flight_id"].isin(set(recovered))].copy()
        print(f"  keeping the {df['flight_id'].nunique()} recovered flights only")

    stats = (
        df.groupby("flight_id")["persistent"]
        .agg(persistent_frac="mean", n_points="count")
        .reset_index()
    )
    qualifying = stats[stats["persistent_frac"] >= min_persistent_frac]["flight_id"].tolist()
    # exclude flights where every point is False (frac == 0 when min is 0)
    qualifying = stats[stats["persistent_frac"] > 0]["flight_id"].tolist()
    if min_persistent_frac > 0:
        qualifying = stats[stats["persistent_frac"] >= min_persistent_frac]["flight_id"].tolist()

    print(f"  {len(df['flight_id'].unique())} total flights → "
          f"{len(qualifying)} with persistent contrails "
          f"(min_frac={min_persistent_frac:.0%})\n")

    return df[df["flight_id"].isin(qualifying)].copy(), qualifying


def build_scenarios(df: pd.DataFrame, flight_ids: list[str]) -> list[dict]:
    scenarios = []
    for fid in flight_ids:
        fdata = df[df["flight_id"] == fid].sort_values("timestamp")
        if len(fdata) < 2:
            continue

        lat0, lon0 = fdata["latitude"].iloc[0], fdata["longitude"].iloc[0]
        lat1, lon1 = fdata["latitude"].iloc[-1], fdata["longitude"].iloc[-1]
        dist_km = _haversine_km(lat0, lon0, lat1, lon1)
        typecode = fdata["typecode"].iloc[0]
        if pd.isna(typecode):
            typecode = "B738"

        scenarios.append(dict(
            scenario_id=fid,
            flight_id=fid,
            haul_category=_haul_category(dist_km),
            distance_km=round(dist_km, 1),
            typecode=typecode,
            origin_lat=lat0,
            origin_lon=lon0,
            dest_lat=lat1,
            dest_lon=lon1,
            persistent_frac=round(float(fdata["persistent"].mean()), 4),
            n_points=len(fdata),
        ))
    return scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--out-path", default=None,
                        help="Output parquet file (all qualifying flights combined). "
                             f"Default: {DEFAULT_OUT_PATH.name}, or "
                             f"{RECOVERED_OUT_PATH.name} with --only-recovered.")
    parser.add_argument(
        "--min-persistent-frac", type=float, default=0.0,
        help="Min fraction of trajectory points with persistent==True (default: any)",
    )
    parser.add_argument(
        "--recover-typecodes", action="store_true",
        help="Fill missing typecodes from traffic.data.aircraft before the NaN "
             "filter, instead of dropping those flights.",
    )
    parser.add_argument(
        "--only-recovered", action="store_true",
        help="Keep only the flights --recover-typecodes brought back, and write "
             "them to their own parquet (implies --recover-typecodes).",
    )
    args = parser.parse_args()

    recover = args.recover_typecodes or args.only_recovered
    out_path = Path(args.out_path) if args.out_path else (
        RECOVERED_OUT_PATH if args.only_recovered else DEFAULT_OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df, flight_ids = load_persistent_flights(
        args.data_path, args.min_persistent_frac,
        recover_typecodes=recover, only_recovered=args.only_recovered)
    scenarios = build_scenarios(df, flight_ids)

    summary_rows = []
    for sc in scenarios:
        print(f"  [{sc['flight_id']}]  {sc['distance_km']:.0f} km  "
              f"{sc['persistent_frac']*100:.1f}% persistent  {sc['n_points']} pts")
        summary_rows.append(sc)

    df.to_parquet(out_path)
    print(f"\nSaved {len(flight_ids)} flights ({len(df)} rows) → {out_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path}")
    print(summary_df[["scenario_id", "haul_category", "distance_km",
                       "typecode", "persistent_frac", "n_points"]].to_string(index=False))


if __name__ == "__main__":
    main()
