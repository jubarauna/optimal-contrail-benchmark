import pandas as pd
from fastmeteo.source import ArcoEra5
import math
from traffic.data import opensky
import numpy as np
from traffic.core import Flight, Traffic
import tempfile, os
from pathlib import Path
from src.paths import DATA_DIR
from openap import aero
from scipy.interpolate import RegularGridInterpolator


def typecode_for(icao24: str) -> str | None:
    """Aircraft type for an icao24 address, from traffic's aircraft database.

    Only the callsign / single-file constructors need it: flights read from the
    benchmark parquet carry their own ``typecode`` column."""
    from traffic.data import aircraft

    rec = aircraft.data.loc[aircraft.data["icao24"].str.lower() == str(icao24).lower(), "typecode"]
    rec = rec.dropna()
    return str(rec.iloc[0]).upper() if not rec.empty and str(rec.iloc[0]).strip() else None

RENAME_MAP = {
    "lat": "latitude",
    "lon": "longitude",
    "velocity": "groundspeed",
    "heading": "track",
    "vertrate": "vertical_rate",
    "baroaltitude": "altitude",
}

class FlightData:
    def __init__(self, start: str = None, stop: str = None, callsign: str = None, origin: str = None, destination: str = None,
                 origin_lat: float = None,
                 origin_lon: float = None,
                 destination_lat: float = None,
                 destination_lon: float = None,
                 path: str = None,
                 base_id: str = None,
                 aircraft_type = "B738"
                 ):
        self.start = start
        self.stop = stop
        self.callsign = callsign
        self.origin = origin
        self.destination = destination
        self.airports = [
            self.origin,
            self.destination
        ]
        self.flight = None
        self.meteo = None
        self.initial_latitude = origin_lat
        self.initial_longitude = origin_lon
        self.final_latitude = destination_lat
        self.final_longitude = destination_lon
        self.coords = [
            self.initial_latitude,
            self.initial_longitude,
            self.final_latitude,
            self.final_longitude
        ]
        self.path = path
        self.flight_id = base_id
        self.aircraft_type = aircraft_type

        if self.start is None or self.stop is None:
            raise ValueError("Please provide begin and end flight time.")

        if self.callsign is not None:
            self.from_callsign()
        elif all(v is not None for v in self.coords):
            self.from_simulation()
        elif all(v is not None for v in self.airports):
            self.from_airports()
        elif self.path is not None:
            if self.flight_id is not None:
                self.from_file_flights()
            else:
                self.from_file_oneflight()
        # Live-ERA5 backend built lazily: WeatherData() → ArcoEra5() opens the
        # remote ARCO-ERA5 zarr over GCS in its constructor, so building it here
        # unconditionally would make EVERY FlightData (incl. grid-only / offline
        # ones) depend on that remote — and fail if it is unreachable. Only
        # attach_meteo("era5") needs it, so it is created there on demand.
        self.wd = None
        self.meteo = None
        self.wind = None
        # Meteo bookkeeping set by attach_meteo(...). `_meteo_source` records how
        # self.meteo was produced ("era5" | "grid"); the grid path also stores the
        # GridWeather handle and the σ its `cost` column is baked at. `_wind_t0` is
        # the timestamp that self.wind's `ts` axis (and the TAS wind query) are
        # measured from — always departure, so it matches opentop's elapsed-time
        # state and the departure-relative cost grid.
        self._meteo_source = None
        self._grid = None
        self._grid_sigma = None
        self._wind_t0 = None
        self._df_cost = None
        self._interpolant = None
    #------------------------------
    # Create flight for simulation flight
    #------------------------------
    def from_simulation(
            self):

        timestamps = pd.date_range(self.start, self.stop, freq="1h")
        latitudes = np.linspace(self.initial_latitude, self.final_latitude, len(timestamps))
        longitudes = np.linspace(self.initial_longitude, self.final_longitude, len(timestamps))

        self.flight = Flight(
            pd.DataFrame().assign(
                timestamp=timestamps,
                latitude=latitudes,
                longitude=longitudes,
                altitude=35000,  # Constant cruise altitude for simulation
            )
        )
    # -----------------------------
    # Load flight from file
    # -----------------------------
    def from_file_oneflight(self):
        df = pd.read_parquet(self.path)

        if "timestamp" not in df.columns and "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        if df["timestamp"].dtype in [np.int64, np.float64]:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.rename(columns=RENAME_MAP)
        if "altitude" in df.columns:
            df["altitude"] = df["altitude"] * 3.28084

        t = Traffic(df)
        self.flight = next(iter(t))

        self.flight = self.flight.sort_values("timestamp")

        self.flight = (
            self.flight
            .drop(["onground", "last_position"], axis=1, errors="ignore")
            .query("latitude.notnull()")
        )
        if self.flight.data.empty:
            raise ValueError("Flight has no valid trajectory points.")

        counts = self.flight.data["callsign"].value_counts()
        main_callsign = counts.idxmax()
        self.flight = Traffic(self.flight.data[self.flight.data["callsign"] == main_callsign].copy())

        icao24 = self.flight.data["icao24"].iloc[0]
        typecode = typecode_for(icao24)
        if typecode is not None:
            self.aircraft_type = typecode

        self.filter_cruise()

        filtered = self.flight.assign(
            ts=lambda d: (
                    pd.to_datetime(d.timestamp, utc=True)
                    - pd.to_datetime(d.timestamp.iloc[0], utc=True)
            ).dt.total_seconds()
        )

        self.flight = filtered
        self.initial_latitude = self.flight.data.latitude.iloc[0]
        self.initial_longitude = self.flight.data.longitude.iloc[0]
        self.final_latitude = self.flight.data.latitude.iloc[-1]
        self.final_longitude = self.flight.data.longitude.iloc[-1]
        self.start = self.flight.data.timestamp.iloc[0]
        self.stop = self.flight.data.timestamp.iloc[-1]

    def from_file_flights(self):
        df = pd.read_parquet(self.path)
        t_opt = Traffic(df)
        icao24 = self.flight_id.split("_")[0]
        t_example = t_opt.query(f"icao24=='{icao24}'")

        flight = t_example[self.flight_id].assign(
            ts=lambda d: (d.timestamp - d.timestamp.iloc[0]).dt.total_seconds()
        )

        self.flight = (
            flight
            .drop(["last_position", "onground"], axis=1, errors="ignore")
            .query("latitude.notnull()")
        )

        # Aircraft type straight from the parquet's typecode column. Unlike
        # from_file_oneflight / from_callsign (which look up traffic's aircraft database by
        # icao24), from_file_flights has no lookup — without this, aircraft_type
        # stays at the "B738" default and mis-configures the fuel/thrust models.
        if "typecode" in self.flight.data.columns:
            tc = self.flight.data["typecode"].dropna()
            if not tc.empty:
                self.aircraft_type = str(tc.iloc[0]).upper()

        self.filter_cruise()

        if self.flight.data.empty:
            raise ValueError("Flight has no valid trajectory points.")
        # self.flight = (
        #     self.flight.iterate_lazy(iterate_kw=dict(by="45min"))
        #     .assign_id(name="{self.icao24}_{idx:>05}")
        #     .pipe(self.split_flight_legs)
        #     .longer_than("30 minutes")
        #     .shorter_than("8 hours")
        #     .eval(max_workers=16, desc="processing")
        # )
        # self.flight = self.flight.pipe(self.flight.filter_cruise).eval(max_workers=16, desc="filtering")
        # self.flight = self.flight.resample("30s").eval(max_workers=16, desc="resampling")

        self.initial_latitude = self.flight.data.latitude.iloc[0]
        self.initial_longitude = self.flight.data.longitude.iloc[0]
        self.final_latitude = self.flight.data.latitude.iloc[-1]
        self.final_longitude = self.flight.data.longitude.iloc[-1]
        self.start = self.flight.data.timestamp.iloc[0]
        self.stop = self.flight.data.timestamp.iloc[-1]

    # -----------------------------
    # Load flight from OpenSky
    # -----------------------------
    # Todo: make this more robust to handle airports data too
    def from_callsign(self):
        flight_0 = opensky.history(
            self.start, self.stop,
            callsign=self.callsign,
            return_flight=True
        )

        if flight_0 is None or flight_0.data.empty:
            raise ValueError("No flight data retrieved.")

        # Select main icao24 if multiple
        counts = flight_0.data["icao24"].value_counts()
        main_icao24 = counts.idxmax()

        flight_main = Flight(
            flight_0.data[flight_0.data["icao24"] == main_icao24].copy()
        )

        self.flight = (
            flight_main
            .drop(["last_position", "onground"], axis=1, errors="ignore")
            .query("latitude.notnull()")
        )
        self.filter_cruise()
        if self.flight.data.empty:
            raise ValueError("Flight has no valid trajectory points.")

        typecode = typecode_for(main_icao24)
        if typecode is not None:
            self.aircraft_type = typecode

        flight_main = flight_main.drop(columns=['serials'])
        flight_main.data.to_csv('output.csv', index=False)

        self.initial_latitude = self.flight.data.latitude.iloc[0]
        self.initial_longitude = self.flight.data.longitude.iloc[0]
        self.final_latitude = self.flight.data.latitude.iloc[-1]
        self.final_longitude = self.flight.data.longitude.iloc[-1]
    # -----------------------------
    # Load flight from OpenSky
    # -----------------------------
    # Todo: make this more robust to handle airports data too
    def from_airports(self):
        flight_0 = opensky.history(
            self.start, self.stop,
            callsign=self.callsign,
            return_flight=True
        )

        if flight_0 is None or flight_0.data.empty:
            raise ValueError("No flight data retrieved.")

        # Select main icao24 if multiple
        counts = flight_0.data["icao24"].value_counts()
        main_icao24 = counts.idxmax()

        flight_main = Flight(
            flight_0.data[flight_0.data["icao24"] == main_icao24].copy()
        )

        self.flight = (
            flight_main
            .drop(["last_position", "onground"], axis=1, errors="ignore")
            .query("latitude.notnull()")
        )

        if self.flight.data.empty:
            raise ValueError("Flight has no valid trajectory points.")

        self.initial_latitude = self.flight.data.latitude.iloc[0]
        self.initial_longitude = self.flight.data.longitude.iloc[0]
        self.final_latitude = self.flight.data.latitude.iloc[-1]
        self.final_longitude = self.flight.data.longitude.iloc[-1]
    # -----------------------------
    # Cruise filtering
    # -----------------------------
    def filter_cruise(self):
        if self.flight is None:
            raise ValueError("Load flight before filtering.")

        cruise = self.flight.phases().query("phase == 'CRUISE'")
        if cruise is None or cruise.data.empty:
            return

        self.flight = self.flight.between(cruise.start, cruise.stop)
        self.flight = self.flight.filter("kalman")

    # -----------------------------
    # Meteo attachment (source-agnostic)
    # -----------------------------
    def attach_meteo(
        self,
        source: str = "era5",
        *,
        wind: bool = False,
        buffer_hours: float = 2.0,
        sigma: float | None = None,
        grid: "GridWeather | None" = None,
        grid_path: str | None = None,
    ) -> pd.DataFrame:
        """Populate ``self.meteo`` from a chosen meteo source, then optionally
        ``self.wind``. Returns the standardized meteo frame.

        This is the single entry point for both meteo backends; the downstream
        products (``self.wind``, :meth:`cost_grid`, :meth:`interpolant`) are then
        built the same way regardless of where the data came from.

        Parameters
        ----------
        source : {"era5", "grid"}
            ``"era5"``  — live ERA5 fetch over the flight's bbox/time (via
            :class:`WeatherData`); cost must still be computed per σ.
            ``"grid"``  — slice a precomputed, already-σ-smoothed grid (via
            :class:`GridWeather`); ``cost`` comes ready-made.
        wind : bool
            Also build :attr:`wind` (needed for the TAS wind correction in
            :meth:`compute_cruise_mach`; the OpenTOP batches do not pass it to
            ``enable_wind``, so their solves have no wind).
        buffer_hours : float
            Padding beyond the flown window (both backends honour a ≥2 h floor).
        sigma : float
            Required for ``source="grid"`` — which σ column to read. For
            ``source="era5"`` it is ignored here (σ is applied later in
            :meth:`cost_grid`, which can re-smooth to any σ).
        grid : GridWeather, optional
            A preloaded grid handle (load the parquet once, reuse across many
            flights — this is what the batch runner passes). If omitted for
            ``source="grid"``, ``grid_path`` is loaded on the spot.
        grid_path : str, optional
            Parquet path used to construct a :class:`GridWeather` when ``grid`` is
            not supplied.
        """
        if self.flight is None:
            raise ValueError("Load flight before interpolating meteo.")

        source = source.lower()
        if source in ("era5", "live"):
            if self.wd is None:                       # build the live-ERA5 backend on demand
                self.wd = WeatherData()
            raw = self.wd.interpolate(flight=self.flight, time_buffer_hours=buffer_hours)
            self._meteo_source = "era5"
            self._grid = None
            self._grid_sigma = None
        elif source == "grid":
            if sigma is None:
                raise ValueError("attach_meteo(source='grid') requires sigma=.")
            gw = grid if grid is not None else GridWeather(grid_path, sigmas=[sigma])
            if grid is None and grid_path is None:
                raise ValueError(
                    "attach_meteo(source='grid') needs either grid= or grid_path=."
                )
            raw = gw.slice_for_flight(
                self.initial_latitude, self.initial_longitude,
                self.final_latitude, self.final_longitude,
                start=self.start, stop=self.stop, sigma=sigma,
                buffer_hours=buffer_hours,
            )
            self._meteo_source = "grid"
            self._grid = gw
            self._grid_sigma = float(sigma)
        else:
            raise ValueError(f"unknown meteo source {source!r} (use 'era5' or 'grid').")

        # B-spline over the time axis needs ≥2 distinct ERA5 hours; guard here so
        # the message is the same whichever backend produced the frame.
        n_ts = raw["timestamp"].nunique()
        if n_ts < 2:
            raise ValueError(
                f"Only {n_ts} meteo time steps retrieved — B-spline needs ≥2. "
                f"Increase buffer_hours (currently {buffer_hours})."
            )

        self.meteo = self._standardize_meteo(raw)
        # Invalidate cached derived products from a previous attach.
        self._df_cost = None
        self._interpolant = None
        self.wind = None

        if wind:
            self._build_wind()
        return self.meteo

    def interpolate_meteo(self, time_buffer_hours: float = 3.0, wind: bool = False):
        """Backward-compatible ERA5 shortcut. Equivalent to
        ``attach_meteo("era5", wind=wind, buffer_hours=time_buffer_hours)``."""
        return self.attach_meteo(
            "era5", wind=wind, buffer_hours=time_buffer_hours
        )

    @staticmethod
    def _to_utc(ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    def _standardize_meteo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise a raw meteo frame so both backends look identical downstream:
        UTC ``timestamp`` and wind columns named ``u``/``v`` (ERA5's long
        ``*_component_of_wind`` names are collapsed). Other columns pass through,
        so the grid frame keeps its ready-made ``cost``/``persistent`` and the
        ERA5 frame keeps ``specific_humidity``/``temperature`` for
        :func:`agg_conditions`."""
        df = df.rename(columns={
            "u_component_of_wind": "u",
            "v_component_of_wind": "v",
        }).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def _build_wind(self) -> None:
        """Build :attr:`wind` = ``[ts, latitude, longitude, h, u, v]`` from
        :attr:`meteo`, with ``ts`` measured from departure (:attr:`start`).

        Anchoring on departure — rather than the padded meteo start — is what
        makes the field consistent with opentop's elapsed-time state and with the
        departure-relative cost grid, and it lets :meth:`_along_track_wind_ms`
        query without needing ``self.meteo`` to still be around."""
        if self.meteo is None:
            raise ValueError("Attach meteo before building the wind field.")
        if not {"u", "v"}.issubset(self.meteo.columns):
            raise ValueError("meteo has no u/v wind columns; cannot build wind.")
        t0 = self._to_utc(self.start)
        self._wind_t0 = t0
        self.wind = (
            self.meteo.assign(
                h=lambda x: x.altitude * aero.ft,
                ts=lambda x: (x.timestamp - t0).dt.total_seconds(),
            )[["ts", "latitude", "longitude", "h", "u", "v"]]
        )

    # -----------------------------
    # Cost grid + interpolant (source-agnostic)
    # -----------------------------
    def cost_grid(self, sigma: float | None = None, t_reference=None) -> pd.DataFrame:
        """Return the opentop/A* cost grid ``[longitude, latitude, height, cost,
        ts]`` for the attached meteo, ``ts`` relative to ``t_reference`` (default
        departure).

        For ``source="grid"`` the ``cost`` column is already σ-smoothed, so this
        just reshapes it (and asserts the requested σ matches the baked one). For
        ``source="era5"`` it runs :func:`agg_conditions` + :func:`prepare_grid` to
        compute and smooth the cost at the requested σ."""
        if self.meteo is None:
            raise ValueError("Call attach_meteo(...) before cost_grid(...).")
        t_ref = self.start if t_reference is None else t_reference

        if self._meteo_source == "grid":
            if sigma is not None and float(sigma) != self._grid_sigma:
                raise ValueError(
                    f"grid meteo is smoothed at σ={self._grid_sigma}; cannot serve "
                    f"σ={sigma}. Re-attach with attach_meteo(sigma={sigma})."
                )
            t0 = self._to_utc(t_ref)
            df_cost = (
                self.meteo.assign(
                    height=lambda x: x.altitude * aero.ft,
                    cost=lambda x: x["cost"].astype(float),
                    ts=lambda x: (x.timestamp - t0).dt.total_seconds(),
                )
                .sort_values(["ts", "height", "latitude", "longitude"])
                .reset_index(drop=True)
            )
        else:
            if sigma is None:
                raise ValueError(
                    "cost_grid(sigma=...) is required for ERA5 meteo (σ is applied here)."
                )
            # Lazy imports: keep data.py free of a hard dependency on the model
            # layer (which imports data.py) — avoids an import cycle.
            from src.models.climate.contrails import agg_conditions
            from src.models.grid.contrail_grid import prepare_grid
            df_cost = prepare_grid(agg_conditions(self.meteo), sigma, t_reference=t_ref)

        self._df_cost = df_cost
        return df_cost

    def interpolant(self, sigma: float | None = None, cache_path=None,
                    shape: str = "bspline"):
        """Cached CasADi interpolant of the cost grid (:meth:`cost_grid`).

        ``cache_path`` defaults to a per-(source, flight, σ) file in the system
        tempdir. NB: the underlying cache does no invalidation, so the default key
        includes the meteo source and σ to keep grid- and ERA5-derived
        interpolants from colliding."""
        import opentop as top  # lazy: opentop is heavy and not always needed

        df_cost = self.cost_grid(sigma) if (self._df_cost is None or sigma is not None) \
            else self._df_cost
        if cache_path is None:
            cache_path = self._interp_cache_path(sigma)
        self._interpolant = top.tools.cached_interpolant_from_dataframe(
            df_cost, str(cache_path), shape=shape
        )
        return self._interpolant

    def _interp_cache_path(self, sigma) -> Path:
        """A stable-enough default interpolant cache path keyed by meteo source,
        flight identity and σ."""
        ident = self.callsign or self.flight_id or f"{self.initial_latitude:.2f}_{self.initial_longitude:.2f}"
        start_tag = str(self.start).replace(" ", "_").replace(":", "-")
        src = self._meteo_source or "era5"
        name = f"fd_interp_{src}_{ident}_{start_tag}_sigma{sigma}.casadi"
        return Path(tempfile.gettempdir()) / name

    def compute_cruise_mach(self, wind_correction: bool = True) -> float:
        """
        Representative cruise Mach number of the real flight.

        TAS is derived from the recorded groundspeed; when ``wind_correction`` is
        on (default) the along-track ERA5 wind component is removed so TAS is the
        *true airspeed actually flown* (groundspeed alone is ground-referenced and
        gives a wind-biased Mach). Mach is then ``tas2mach(TAS, altitude)``
        averaged over the trace.

        Call :meth:`filter_cruise` first so the mean covers only the cruise
        segment, and :meth:`interpolate_meteo` with ``wind=True`` so the wind
        field is available (required when ``wind_correction`` is on).

        Returns
        -------
        float : mean cruise Mach.
        """
        if self.flight is None:
            raise ValueError("Load flight before computing mach.")

        df     = self.flight.data
        gs_ms  = df["groundspeed"].values.astype(float) * aero.kts   # kt → m/s
        alt_m  = df["altitude"].values.astype(float) * aero.ft       # ft → m

        if wind_correction:
            tailwind_ms = self._along_track_wind_ms(df)
        else:
            tailwind_ms = 0.0

        tas_ms = gs_ms - tailwind_ms
        mach   = aero.tas2mach(tas_ms, alt_m)
        return float(np.nanmean(mach))

    def _along_track_wind_ms(self, flight_df: pd.DataFrame) -> np.ndarray:
        """
        Along-track (tail) wind component [m/s] at each flight point, sampled from
        the wind field ``self.wind`` (built by ``attach_meteo(..., wind=True)``).

        Works for either meteo source — the field's ``ts`` axis and the query time
        below are both measured from departure (``self._wind_t0``), so the ERA5
        and grid backends are interchangeable here.

        Positive = tailwind (adds to groundspeed, so TAS = GS − tailwind).
        """

        if self.wind is None:
            raise ValueError(
                "Wind field not available. Call attach_meteo(..., wind=True) first, "
                "or use compute_cruise_mach(wind_correction=False)."
            )
        if "track" not in flight_df.columns:
            raise ValueError("Flight data has no 'track' column; cannot project wind.")

        wind_df = self.wind.dropna()
        lats = np.sort(wind_df["latitude"].unique())
        lons = np.sort(wind_df["longitude"].unique())
        hs   = np.sort(wind_df["h"].unique())    # metres
        tss  = np.sort(wind_df["ts"].unique())   # seconds since departure
        shape = (len(lats), len(lons), len(hs), len(tss))

        ordered  = wind_df.sort_values(["latitude", "longitude", "h", "ts"])
        interp_u = RegularGridInterpolator(
            (lats, lons, hs, tss), ordered["u"].values.reshape(shape),
            method="linear", bounds_error=False, fill_value=None,
        )
        interp_v = RegularGridInterpolator(
            (lats, lons, hs, tss), ordered["v"].values.reshape(shape),
            method="linear", bounds_error=False, fill_value=None,
        )

        # Query time in the wind field's reference (seconds since departure),
        # computed straight from timestamps so it is robust to filter_cruise()
        # re-slicing the flight (unlike the flight's own 'ts' column). Uses the
        # stored departure origin rather than self.meteo, so the correction works
        # even when meteo came from a grid slice (self.meteo may not span the pad).
        t0 = self._wind_t0 if self._wind_t0 is not None else self._to_utc(self.start)
        t_query = (
            pd.to_datetime(flight_df["timestamp"], utc=True) - t0
        ).dt.total_seconds().values
        query_pts = np.column_stack([
            flight_df["latitude"].values,
            flight_df["longitude"].values,
            flight_df["altitude"].values * aero.ft,   # ft → m to match wind 'h'
            t_query,
        ])

        wu = interp_u(query_pts)   # m/s eastward
        wv = interp_v(query_pts)   # m/s northward
        track_rad = np.deg2rad(flight_df["track"].values)
        return wu * np.sin(track_rad) + wv * np.cos(track_rad)   # along-track [m/s]

    def split_flight_legs(self) -> None | Flight:
        # recreate flight ids with different legs
        above_fl150 = self.flight.query("altitude > 25000")
        if above_fl150 is not None:
            self.flight=above_fl150.split("30 min").all("{self.flight_id}_{i}")
    def head(self):
        print(self.flight.data.head())


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


if __name__ == "__main__":
    # Example usage
    # fd = FlightData(start="2023-01-01 00:00:00", stop="2023-01-01 06:00:00",
    #                 origin_lat=40.6413, origin_lon=-73.7781,  # JFK
    #                 destination_lat=51.4700, destination_lon=-0.4543  # LHR
    #                 )
    # fd.interpolate_meteo(time_buffer_hours=3.0)
    begin = "2026-01-10 05:00"
    end = "2026-01-10 07:00"
    DATA_PATH_ONEFLIGHT = str(
        DATA_DIR / "ostk_export_2026-06-18T08-06-11.parquet"
    )
    fd2 = FlightData(start=begin, stop=end, path=DATA_PATH_ONEFLIGHT)
    fd2.head()