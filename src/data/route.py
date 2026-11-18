import os
from pathlib import Path
from src.data.traffic import FlightData
from traffic.core import Flight
import tempfile
import math
from openap import aero
from openap.aero import ft
import numpy as np


def great_circle_route(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    mach: float = 0.78,
    avg_alt_m: float = 34000 / ft,
    roc: float = 500, # ft/min
    alt_step_ft: float = 500, # ft
    # num_points: int = 240,
) -> tuple:
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    total_distance_km = 6371 * c

    tas_ms = aero.mach2tas(mach, avg_alt_m)
    cruise_speed_kmh = tas_ms * 3.6

    num_points = math.floor(roc * total_distance_km *1000 / (tas_ms * alt_step_ft * 60))

    fractions = np.linspace(0, 1, num_points)
    route = []
    # haversine formula to calculate arc distance
    for f in fractions:
        A = np.sin((1 - f) * c) / np.sin(c)
        B = np.sin(f * c) / np.sin(c)
        x = A * np.cos(lat1r) * np.cos(lon1r) + B * np.cos(lat2r) * np.cos(lon2r)
        y = A * np.cos(lat1r) * np.sin(lon1r) + B * np.cos(lat2r) * np.sin(lon2r)
        z = A * np.sin(lat1r) + B * np.sin(lat2r)
        lat = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
        lon = np.degrees(np.arctan2(y, x))
        distance = f * total_distance_km
        ts_s = (distance / cruise_speed_kmh) * 3600
        route.append((lat, lon, distance, ts_s))

    print(f"Great Circle: {lat1:.2f},{lon1:.2f} → {lat2:.2f},{lon2:.2f} | {total_distance_km:.1f} km")
    return route, total_distance_km

def cumulative_gc_km(lat_deg, lon_deg) -> np.ndarray:
    """
    Cumulative great-circle (haversine) distance [km] along a lat/lon trace.

    Same length as the input; entry ``i`` is the distance from the first point to
    point ``i`` (so it starts at 0 and is non-decreasing). Shared by
    :func:`real_flight_route` and :func:`real_flight_baseline` so the distance
    axis is computed exactly once, one way.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    a = (np.sin(np.diff(lat) / 2) ** 2
         + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2)
    seg_km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    return np.concatenate([[0.0], np.cumsum(seg_km)])

def real_flight_route(
    flight: Flight,
    mach: float,
    avg_alt_m: float,
    roc: float = 500,          # ft/min
    alt_step_ft: float = 500,  # ft
) -> tuple:
    """
    Build an equally-spaced cruise route that follows the REAL lateral path.

    Same output contract as :func:`great_circle_route` — a list of
    ``(lat, lon, distance_km, ts_s)`` tuples plus ``total_distance_km`` — so it is
    a drop-in input for :class:`ContrailEnvironmentFromGrid` and the ``model_2D``
    optimizers.  Only the lateral path differs: instead of a great circle between
    the endpoints, the real flight's ``(lat, lon)`` trace is resampled onto
    equally-spaced distance points (the altitude-only optimization keeps the
    lateral path fixed, so real and optimized flights share this exact path).

    The point count and the timestamps mirror :func:`great_circle_route` exactly
    (``num_points`` from ROC / cruise TAS / ``alt_step_ft``; ``ts_s`` from a
    constant cruise speed), so a route built here is directly comparable to one
    built by ``great_circle_route``.

    Parameters
    ----------
    flight : traffic Flight
        Object exposing ``.data`` with ``latitude`` / ``longitude`` columns
        (e.g. ``FlightData.flight``).
    mach, avg_alt_m :
        Precalculated from the real flight; together they set the cruise TAS used
        for both the point count and the timestamps.  ``avg_alt_m`` is in metres.
    roc : float
        Max climb rate (ft/min) — only used to size ``num_points``, as in
        ``great_circle_route``.
    alt_step_ft : float
        Altitude grid step (ft) — only used to size ``num_points``.

    Returns
    -------
    route : list[tuple[float, float, float, float]]
        ``(lat, lon, distance_km, ts_s)`` per equally-spaced point.
    total_distance_km : float
    """
    lat_real = np.asarray(flight.data.latitude.values, dtype=float)
    lon_real = np.asarray(flight.data.longitude.values, dtype=float)

    # Cumulative great-circle distance along the real trace [km].
    cum_km = cumulative_gc_km(lat_real, lon_real)
    total_distance_km = float(cum_km[-1])

    # np.interp needs a strictly increasing sample axis: drop stationary points
    # (duplicated cumulative distance) before interpolating.
    keep = np.concatenate([[True], np.diff(cum_km) > 0])
    cum_km_u, lat_u, lon_u = cum_km[keep], lat_real[keep], lon_real[keep]

    # Same number of points as great_circle_route (ROC / tas / alt_step bound).
    tas_ms = aero.mach2tas(mach, avg_alt_m)
    cruise_speed_kmh = tas_ms * 3.6
    num_points = route_point_count(total_distance_km, mach, avg_alt_m, roc, alt_step_ft)

    # Resample the real lateral path onto equally-spaced distances.
    # (Linear interp between the dense real trackpoints; fine away from the
    # antimeridian, which these mid-latitude flights do not cross.)
    new_dist = np.linspace(0.0, total_distance_km, num_points)
    new_lat = np.interp(new_dist, cum_km_u, lat_u)
    new_lon = np.interp(new_dist, cum_km_u, lon_u)

    route = []
    for lat, lon, distance in zip(new_lat, new_lon, new_dist):
        ts_s = (distance / cruise_speed_kmh) * 3600
        route.append((float(lat), float(lon), float(distance), float(ts_s)))

    print(f"Real Flight Route: {total_distance_km:.1f} km | {num_points} pts (follows real path)")
    return route, total_distance_km

def route_point_count(total_distance_km: float, mach: float, avg_alt_m: float,
                      roc: float, alt_step_ft: float) -> int:
    """Number of points :func:`real_flight_route` puts on a route.

    One point per ``alt_step_ft / roc`` minutes of flight at the cruise TAS, so that
    climbing one altitude step at ``roc`` ft/min takes exactly one segment; at least 2.
    The graph solvers' waypoints come from this, and the OpenTOP batch sizes its
    collocation grid from it too, so both methods discretise each flight alike.
    """
    tas_ms = aero.mach2tas(mach, avg_alt_m)
    n = math.floor(roc * total_distance_km * 1000 / (tas_ms * alt_step_ft * 60))
    return max(n, 2)


def real_flight_baseline(flight: Flight, distances) -> np.ndarray:
    """
    Real cruise altitude [ft] resampled onto ``distances`` (typically
    ``env.distances``), so the real flight can be overlaid as a *baseline* against
    the optimized vertical profiles.

    Uses the exact same cumulative-distance axis as :func:`real_flight_route`
    (via :func:`cumulative_gc_km`), so the returned altitudes line up 1-to-1 with
    the optimized paths' distance grid.

    Returns
    -------
    np.ndarray : altitude in ft, one value per entry of ``distances``.
    """
    lat = np.asarray(flight.data.latitude.values, dtype=float)
    lon = np.asarray(flight.data.longitude.values, dtype=float)
    alt = np.asarray(flight.data.altitude.values, dtype=float)

    cum_km = cumulative_gc_km(lat, lon)
    # np.interp needs a strictly increasing sample axis: drop stationary points.
    keep = np.concatenate([[True], np.diff(cum_km) > 0])
    return np.interp(np.asarray(distances, dtype=float), cum_km[keep], alt[keep])

if __name__ == "__main__":
    sigma=2
    flight_name = "Berlin-Madrid"
    begin = "2023-01-05 09:48"
    end = "2023-01-05 11:00"
    callsign = "RYR880W"
    actype = "B738"
    print(f"Preparing flight: {flight_name} | {begin} → {end} | σ={sigma}")
    # 1) Flight setup — same identity as 2_Run/batch_opentop.py.
    #    Reuse _ParetoBase to load the flight, build the cost grid + interpolant,
    #    and hand back a configured top.Cruise + mach / mass / altitude bounds.
    TMP = Path(tempfile.gettempdir())
    interpolant_cache = os.environ.get(
        "INTERPOLANT_CACHE",
        str(TMP / f"opentop_interp_{flight_name}_sigma{sigma}.casadi"),
    )
    fd = FlightData(start=begin, stop=end, callsign=callsign)
    fd.attach_meteo("era5", buffer_hours=2.0)
    df_cost = fd.cost_grid(sigma)
    interpolant = fd.interpolant(sigma, cache_path=interpolant_cache)

    fd = FlightData(start=begin, stop=end, callsign=callsign)

    # mach / average cruise altitude / actype precalculated from the real flight.
    fd.attach_meteo("era5", wind=True)                  # wind field for TAS correction
    avg_alt_m = float(fd.flight.data.altitude.mean()) * ft
    mach = fd.compute_cruise_mach()
    actype = fd.aircraft_type                           # read from parquet typecode
    print(f"actype={actype} | mach={mach:.3f} | avg_alt={avg_alt_m / ft:.0f} ft")
    route, total_distance_km = real_flight_route(
        fd.flight, mach=mach, avg_alt_m=avg_alt_m, roc=500, alt_step_ft=500
    )
    print(route[:3], "...", f"{total_distance_km:.1f} km", f"{len(route)} pts")
