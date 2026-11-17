from pathlib import Path
from math import ceil

from src.paths import DATA_DIR, RUN_DIR

# ── configuration ────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = (
    DATA_DIR / "eu_flights_2022feb20_persistent_all.parquet"
)
DEFAULT_OUT_DIR = RUN_DIR / "eu_flights_2022feb20"

DEFAULT_GRID_PATH = DATA_DIR / "grid_era5_smoothed.parquet.gz"

def sigma_sources(base_path: Path, sigmas: list[float]) -> dict[float, tuple[Path, str]]:
    """Map each σ to (grid file, cost column), all in the one base grid.
    σ=0 → 'persistent' (the raw boolean field), σ=2 → 'cost', any other σ →
    'cost_sigma<σ>' (e.g. 'cost_sigma1'), written by 0_Data/regenerate_grid.py."""
    base = Path(base_path)
    src: dict[float, tuple[Path, str]] = {}
    for s in sigmas:
        s = float(s)
        if s == 0.0:
            src[s] = (base, "persistent")   # raw boolean field == σ=0 cost
        elif s == 2.0:
            src[s] = (base, "cost")         # base grid is smoothed at σ=2
        else:
            src[s] = (base, f"cost_sigma{s:g}")
    return src

# NOTE: the precomputed-grid meteo backend formerly named GridMeteo now lives
# in data/data.py as GridWeather, alongside the live WeatherData backend, and
# is consumed through FlightData.attach_meteo("grid", grid=GridWeather(...)).
# sigma_sources above stays here — GridWeather imports it.
