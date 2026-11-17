#  This script samples ERA5 weather over Europe for one day
#  marks each grid point where a persistent contrail would form
#  and smooths that into cost fields
#  Was used to create 0_Data/flight_data/grid_era5_smoothed.parquet.gz

from datetime import timedelta
from glob import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from fastmeteo.source import ArcoEra5
from openap import aero, contrail
from scipy.ndimage import gaussian_filter

# Repo root on sys.path so the paths resolve no matter the working directory this
# script is launched from (relative paths break when cwd isn't the repo root).
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DATA_DIR  # noqa: E402

#
# %%
LAT0, LAT1 = 26, 66
LON0, LON1 = -15, 40
DATE = pd.to_datetime("2022-02-20", utc=True)

# The ERA5 cache is ~14 GB and is NOT in git. Anchored to a fixed path rather than
# to the working directory so an existing download is reused, never re-fetched;
# set OPTIMAL_ERA5_STORE to point at a copy you already have. A cache interrupted
# mid-write comes back as "conflicting sizes for dimension 'time'" — delete the
# store and let it re-download, because re-running alone will not repair it.
ERA5_STORE = os.environ.get("OPTIMAL_ERA5_STORE", str(DATA_DIR.parent / "era5-zarr"))
ARCO_GRID = ArcoEra5(
    local_store=ERA5_STORE,
    model_levels=137,
)


# %%
def agg_contrail_conditions(flight):
    flight = flight.assign(
        rhi=lambda d: contrail.relative_humidity(
            d.specific_humidity,
            aero.pressure(d.altitude * aero.ft),
            d.temperature,
            to="ice",
        ),
        crit_temp=lambda d: contrail.critical_temperature_water(
            aero.pressure(d.altitude * aero.ft)
        ),
        sac=lambda d: d.temperature < d.crit_temp,
        issr=lambda d: d.rhi > 1,
        persistent=lambda d: d.sac & d.issr,
    )
    return flight


def smooth_cost_grid(df_cost, sigma, out_col, src_col="persistent"):
    """Add a Gaussian-smoothed cost column to ``df_cost``.

    Smooths the raw ``src_col`` field (the persistent-contrail flag as float)
    over (height, latitude, longitude) with the given ``sigma`` — no smoothing
    across time — and writes it to ``out_col``. Smoothing always starts from the
    raw field, so columns for different σ are independent of call order.
    """
    print(f"Smoothing grid: sigma={sigma} -> '{out_col}'")
    raw = df_cost[src_col].astype(float).to_numpy().reshape(
        df_cost.ts.nunique(),
        df_cost.height.nunique(),
        df_cost.latitude.nunique(),
        df_cost.longitude.nunique(),
    )
    raw = np.nan_to_num(raw)
    smoothed = gaussian_filter(raw, sigma=(0, sigma, sigma, sigma), mode="nearest")
    df_cost[out_col] = smoothed.reshape(-1)
    return df_cost


# %%
print("Generating cost grid coordinates")

lats = np.arange(LAT0, LAT1, 0.5)
lons = np.arange(LON0, LON1, 0.5)
alts = np.arange(20_000, 45_000, 1000)
timestamps = pd.date_range(DATE, DATE + pd.Timedelta("1d"), freq="1h")

Timestamps, Alts, Lats, Lons = np.meshgrid(timestamps, alts, lats, lons)

GRID = pd.DataFrame(
    {
        "timestamp": Timestamps.ravel(),
        "altitude": Alts.ravel(),
        "latitude": Lats.ravel(),
        "longitude": Lons.ravel(),
    }
)


# %%
print("Generating ERA5 cost grid")


def generate_era5_cost():
    era5_grid = ARCO_GRID.interpolate(GRID)

    era5_grid = agg_contrail_conditions(era5_grid)

    df_cost = (
        era5_grid.assign(
            height=lambda x: x.altitude * aero.ft,
            cost=lambda x: x.persistent.astype(float),
            ts=lambda x: (x.timestamp - x.timestamp.iloc[0]).dt.total_seconds(),
        )
        .rename(columns={"u_component_of_wind": "u", "v_component_of_wind": "v"})
        .sort_values(["ts", "height", "latitude", "longitude"])
    )

    return df_cost


file_era5_cost = str(DATA_DIR / "grid_era5_smoothed.parquet.gz")

df_cost_era5 = generate_era5_cost()
df_cost_era5 = smooth_cost_grid(df_cost_era5, sigma=2, out_col="cost")
df_cost_era5 = smooth_cost_grid(df_cost_era5, sigma=1, out_col="cost_sigma1")
df_cost_era5.to_parquet(file_era5_cost, index=False)



# %%