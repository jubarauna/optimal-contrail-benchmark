import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from fastmeteo.source import ArcoEra5
from traffic.core import Flight


class WeatherData:
    def __init__(self):
        TMP = Path(tempfile.gettempdir())
        self.era5 = ArcoEra5(local_store=str(TMP / "era5-zarr"))
    def interpolate(
            self,
            flight: Flight | None = None,
            origin_lat: np.ndarray | None = None,
            origin_lon: np.ndarray | None = None,
            destination_lat: np.ndarray | None = None,
            destination_lon: np.ndarray | None = None,
            start: pd.DatetimeIndex | None = None,
            stop: pd.DatetimeIndex | None = None,
            time_buffer_hours: float = 1.0,
    ):
        """
        Interpolate ERA5 weather either:
        - from a Flight object
        - or from explicit lat/lon/alt/time arrays (simulation mode)
        """

        # -----------------------------
        # Case 1: Real flight
        # -----------------------------
        if flight is not None:
            if flight.data.empty:
                raise ValueError("Flight has no trajectory data.")

            origin_lat = flight.data.latitude.iloc[0]
            origin_lon = flight.data.longitude.iloc[0]
            destination_lat = flight.data.latitude.iloc[-1]
            destination_lon = flight.data.longitude.iloc[-1]
            start = flight.data.timestamp.iloc[0]
            stop = flight.data.timestamp.iloc[-1]
        # -----------------------------
        # Case 2: Simulation input
        # -----------------------------
        else:
            if any(v is None for v in [origin_lon, origin_lat, destination_lat, destination_lon, start, stop]):
                raise ValueError(
                    "Provide either a flight OR latitudes, longitudes and timestamps."
                )
        # Pad time range to cover longer optimized trajectories
        # Reference script uses minimum 4h to guarantee enough B-spline points
        safe_buffer = max(time_buffer_hours, 2.0)
        start_padded = (start - pd.Timedelta(hours=1)).floor("1h")
        stop_padded = (stop + pd.Timedelta(hours=safe_buffer)).ceil("1h")
        timestamps = pd.date_range(start_padded, stop_padded, freq="1h")

        # -----------------------------
        # Build DataFrame
        # -----------------------------

        latmin = math.ceil(min(origin_lat, destination_lat)) - 2
        latmax = math.ceil(max(origin_lat, destination_lat)) + 2
        lonmin = math.ceil(min(origin_lon, destination_lon)) - 4
        lonmax = math.ceil(max(origin_lon, destination_lon)) + 4

        # creates numpy arrays
        latitudes = np.arange(latmin, latmax, 1)
        longitudes = np.arange(lonmin, lonmax, 1)
        altitudes = np.arange(20000, 46000, 1000)

        latitudes, longitudes, altitudes, times = np.meshgrid(
            latitudes, longitudes, altitudes, timestamps
        )

        grid = pd.DataFrame().assign(
            latitude=latitudes.flatten(),
            longitude=longitudes.flatten(),
            altitude=altitudes.flatten(),
            timestamp=times.flatten(),
        )

        meteo = self.era5.interpolate(grid)

        # Normalize ERA5 short variable names to the full names expected downstream
        _col_map = {
            "q": "specific_humidity",
            "t": "temperature",
            "u": "u_component_of_wind",
            "v": "v_component_of_wind"
        }
        meteo = meteo.rename(columns={k: v for k, v in _col_map.items() if k in meteo.columns})

        missing = [c for c in ("specific_humidity", "temperature") if c not in meteo.columns]
        if missing:
            raise ValueError(
                f"ERA5 output is missing required columns {missing}. "
                f"Available columns: {list(meteo.columns)}"
            )

        # Validate enough time steps for downstream B-spline
        n_ts = meteo.timestamp.nunique()
        if n_ts < 2:
            raise ValueError(
                f"Only {n_ts} ERA5 time steps retrieved — B-spline needs ≥2. "
                f"Increase time_buffer_hours (currently {time_buffer_hours})."
            )

        return meteo


class GridWeather:
    """Precomputed-grid meteo backend — the offline counterpart to
    :class:`WeatherData`.

    Reads the contrail-cost grid from parquet (every σ is a pre-smoothed column
    of the same file, no runtime ``gaussian_filter``), and slices the relevant
    lat/lon/time box for a given flight. Built once and reused across many
    flights (e.g. the benchmarking batch) so the big parquet is read a single
    time. Pass the instance to :meth:`FlightData.attach_meteo` as
    ``attach_meteo("grid", grid=<this>, sigma=...)``.

    ``sigmas`` lists which σ columns will be requested (see ``sigma_sources``);
    a σ whose column is missing is an error — add it in 0_Data/regenerate_grid.py.
    """

    _GRID_SORT = ["ts", "height", "latitude", "longitude"]

    def __init__(self, base_path, sigmas: list[float]):
        # Imported here (not at module top) so data.py has no import-time
        # dependency on the tools layer; these are pure grid-file helpers.
        from src.tools.graph_based_tools import sigma_sources

        srcs = sigma_sources(base_path, sigmas)

        self._by_path: dict[Path, tuple] = {}   # path -> (df, lat, lon, tindex)
        self._sigma: dict[float, tuple] = {}     # σ -> (path, cost array)
        for s, (path, col) in srcs.items():
            path = Path(path)
            if path not in self._by_path:
                print(f"Loading meteo grid {path.name} ...")
                df = pd.read_parquet(path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.sort_values(self._GRID_SORT).reset_index(drop=True)
                self._by_path[path] = (
                    df, df["latitude"].to_numpy(), df["longitude"].to_numpy(),
                    pd.DatetimeIndex(df["timestamp"]),
                )
            df = self._by_path[path][0]
            if col not in df.columns:
                raise KeyError(f"'{col}' not in {path.name} (σ={s}); add it with "
                               f"smooth_cost_grid in 0_Data/regenerate_grid.py")
            self._sigma[float(s)] = (path, df[col].astype(float).to_numpy())
        print("  σ sources: " + ", ".join(
            f"{s:g}→{p.name}:{c}" for s, (p, c) in srcs.items()))

    def slice_for_flight(self, o_lat, o_lon, d_lat, d_lon, start, stop, sigma,
                         buffer_hours: float = 2.0) -> pd.DataFrame:
        """Return the grid rows in the flight's lat/lon/time box, with the chosen
        σ's cost in a ``cost`` column and a departure-relative ``ts``.

        The bbox/time padding mirrors ``WeatherData.interpolate`` so the two
        backends select comparable domains. Output columns are a superset of what
        the cost grid and plots need (``timestamp, altitude, latitude, longitude,
        u, v, temperature, specific_humidity, persistent, height, cost, ts``)."""
        if float(sigma) not in self._sigma:
            raise KeyError(f"σ={sigma} not loaded (have {sorted(self._sigma)}).")
        path, cost = self._sigma[float(sigma)]
        df, lat, lon, tindex = self._by_path[path]

        latmin = math.ceil(min(o_lat, d_lat)) - 2
        latmax = math.ceil(max(o_lat, d_lat)) + 2
        lonmin = math.ceil(min(o_lon, d_lon)) - 4
        lonmax = math.ceil(max(o_lon, d_lon)) + 4
        from src.data.traffic import FlightData  # local: traffic imports meteo
        start = FlightData._to_utc(start)
        stop = FlightData._to_utc(stop)
        safe = max(buffer_hours, 2.0)
        t0 = (start - pd.Timedelta(hours=1)).floor("1h")
        t1 = (stop + pd.Timedelta(hours=safe)).ceil("1h")

        mask = (
            (lat >= latmin) & (lat <= latmax)
            & (lon >= lonmin) & (lon <= lonmax)
            & (tindex >= t0) & (tindex <= t1)
        )
        if not mask.any():
            raise ValueError(
                f"Flight bbox lat[{latmin},{latmax}] lon[{lonmin},{lonmax}] "
                f"time[{t0}..{t1}] has no cells in {path.name}.")

        sub = df.loc[mask].copy()
        sub["cost"] = cost[mask]
        sub["ts"] = (sub["timestamp"] - start).dt.total_seconds()
        return sub.sort_values(self._GRID_SORT).reset_index(drop=True)
