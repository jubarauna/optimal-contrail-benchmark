import numpy as np
from src.optimization.environment import ContrailEnvironmentFromGrid
from openap.aero import ft
from openap import aero
from src.performance.aircraft import fuel_burn

# ──────────────────────────────────────────────────────────────────────────────
# POST-HOC EVALUATION  (shared between A* and OpenAP)
# ──────────────────────────────────────────────────────────────────────────────

# 3-point Gauss-Legendre on [0, 1]: nodes and weights.
_GL_X = np.array([0.5 - np.sqrt(0.15), 0.5, 0.5 + np.sqrt(0.15)])
_GL_W = np.array([5.0, 8.0, 5.0]) / 18.0


def contrail_integral(lat, lon, alt_ft, ts_s, interpolant, ts_offset_s=0.0):
    """∫ cost dt along a trajectory, on the contrail interpolant itself.

    The ONE contrail measure for every solver family. The graph solvers optimise over
    ``env.cost_map`` (the interpolant sampled once per route node and flight level,
    each edge charged its END node) and OpenTOP over its collocation quadrature; both
    are their own model of the cost. This prices any trajectory the same way, so the
    numbers can be compared across families.

    Per segment, 3 Gauss-Legendre points, with lat, lon, altitude and time linear
    between the nodes, at the time the trajectory itself passes. Values are NOT
    clipped at zero (a bspline can undershoot), as in OpenTOP.
    """
    lat, lon, alt_ft, ts = (np.asarray(a, dtype=float) for a in (lat, lon, alt_ft, ts_s))
    f = _GL_X[None, :]                                         # (1, 3)

    def along(a):                                              # (n_seg, 3)
        return a[:-1, None] + f * np.diff(a)[:, None]

    T = along(ts)
    X = np.vstack([along(lon).ravel(), along(lat).ravel(),
                   along(alt_ft).ravel() * ft, T.ravel() + ts_offset_s])
    c = np.asarray(interpolant.map(X.shape[1])(X)).reshape(T.shape)
    return float(((c * _GL_W).sum(axis=1) * np.diff(ts)).sum())


def route_latlon(distances_km, route):
    """(lat, lon) at ``distances_km`` along ``route`` [(lat, lon, km, ts), ...]."""
    r_lat, r_lon, r_km = (np.array([p[i] for p in route], dtype=float) for i in range(3))
    d = np.asarray(distances_km, dtype=float)
    return np.interp(d, r_km, r_lat), np.interp(d, r_km, r_lon)


def flown_ts(alt_ft, distances_km, mach: float) -> np.ndarray:
    """Node times (s from 0) of a path flown at ``mach`` at its OWN altitudes.

    The graph solvers' waypoint times come from one TAS, the fixed Mach at the real
    flight's mean cruise altitude, whatever altitude the path takes. Each segment is
    re-timed here with the TAS at its mid altitude, as OpenTOP's dynamics time it (no
    wind on either side), so both families are priced with consistent flight times.
    """
    alt_ft = np.asarray(alt_ft, dtype=float)
    mid = (alt_ft[:-1] + alt_ft[1:]) / 2.0
    dt = np.diff(np.asarray(distances_km, dtype=float)) * 1000.0 / aero.mach2tas(mach, mid * ft)
    return np.concatenate([[0.0], np.cumsum(dt)])


def reprice(res, interpolant, mach: float, mass_init_kg: float, actype: str) -> None:
    """Re-price an OpenTOP result's fuel and contrail with the graph solvers' evaluator.

    In place. ``fuel_kg`` becomes :func:`fuel_burn` and ``contrail``
    :func:`contrail_integral`, exactly what :func:`evaluate_path` gives a DP/DP_MuO
    path. The solver's own values (the mass its dynamics burnt, the quadrature it
    minimised) are kept in ``meta["fuel_solver"]`` / ``meta["contrail_solver"]``. A
    result without a trajectory is left untouched.
    """
    if res is None or res.lat is None or res.lon is None:
        return
    res.meta["fuel_solver"] = res.fuel_kg
    res.meta["contrail_solver"] = res.contrail
    res.fuel_kg, _ = fuel_burn(res.alt_ft, res.ts_s, mach, mass_init_kg, actype)
    res.contrail = contrail_integral(res.lat, res.lon, res.alt_ft, res.ts_s, interpolant)


def evaluate_path(
    alt_ft: np.ndarray,
    distances_km: np.ndarray,
    ts_s: np.ndarray,
    mach: float,
    mass_init_kg: float,
    env: ContrailEnvironmentFromGrid,
    actype: str = "A320",
    retime: bool = True,
) -> dict:
    """
    Evaluate a path using identical physics for both A* and OpenAP.
    alt_ft, distances_km, ts_s must be same length (node values).
    Returns physical metrics only — no internal optimizer costs.

    ``total_contrail`` is the path's integral on the contrail interpolant
    (:func:`contrail_integral`, position from the route by distance), the measure
    OpenTOP results are re-priced with too. ``total_contrail_table`` is the env's
    cost_map, end node per segment -- what the graph solvers optimise over.

    ``retime`` (default): fuel and ``total_contrail`` use :func:`flown_ts`, the times
    of the path flown at its own altitudes, returned as ``ts_s``; the solver's ``ts_s``
    only enters ``total_contrail_table``, which is the solver's own view of the path.
    """
    ts_eval = flown_ts(alt_ft, distances_km, mach) if retime else np.asarray(ts_s, dtype=float)
    total_fuel_kg, mass = fuel_burn(alt_ft, ts_eval, mach, mass_init_kg, actype)
    total_contrail = 0.0
    for i in range(len(alt_ft) - 1):
        avg_alt = (alt_ft[i] + alt_ft[i + 1]) / 2.0
        avg_ts = (ts_s[i] + ts_s[i + 1]) / 2.0
        dt = ts_s[i + 1] - ts_s[i]
        total_contrail += env.get_contrail_cost_by_ts(avg_ts, avg_alt) * dt

    lat, lon = route_latlon(distances_km, env.route)
    exact = contrail_integral(lat, lon, alt_ft, ts_eval, env.interpolant,
                              getattr(env, "_ts_offset_s", 0.0))
    return {
        "total_fuel_kg": total_fuel_kg,
        "total_contrail": exact,
        "total_contrail_table": total_contrail,
        "final_mass_kg": mass,
        "ts_s": ts_eval,
        "flight_time_s": ts_eval[-1] - ts_eval[0],
        "total_alt_changes_ft": float(np.abs(np.diff(alt_ft)).sum()),
    }
