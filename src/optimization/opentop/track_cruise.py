"""
TrackCruise
===========
An opentop ``Cruise`` whose lateral path is IMPOSED instead of optimised.

``Cruise.follow_track`` keeps the full state ``[xp, yp, h, m, ts]`` and the heading
control, and pulls the solution towards the track with a progress variable per node
plus ~6 tolerance inequalities per interval (``opentop/_track.py:constrain_track``).
The aircraft is still free laterally inside ``tolerance_m``, which is why the track
tolerance changes convergence so much.

Here the position is ONE state, the arc length ``s`` along the track spline:

    states   x = [s, h, m, ts]          controls  u = [vs]          (mach_value set)
                                                  u = [mach, vs]    (mach_value None)
    ds/dt = v·cos(gamma) / |r'(s)|      (no wind; enable_wind is refused)
    xp, yp = x_ref(s), y_ref(s)         heading = atan2(x_ref'(s), y_ref'(s))

so the solution lies ON the track by construction: no tolerance, no progress
variables, and the same lateral problem the graph solvers solve
(``real_flight_route``).

With ``mach_value`` the Mach is a CONSTANT of the model, not a decision variable
pinned by equalities as in Cruise: the controls are the vertical speed alone, the
same fixed-Mach problem the graph solvers solve. Without it Mach stays a control
(free, or constant via fix_mach_number) as in Cruise.

Base is not modified. The objectives (``_objectives.obj_fuel``, ``obj_grid_cost``)
index the 5-state vector by position, so ``init_model`` hands them a VIRTUAL
``[x_ref(s), y_ref(s), h, m, ts]`` / ``[mach, vs, psi_ref(s)]`` (mach the constant
when fixed) instead of being
rewritten, and ``to_trajectory`` rebuilds the same arrays numerically, so the output
DataFrame has the usual columns (plus ``s``).

What changes against ``Cruise._add_formulation`` (cruise.py:208, opentop 2.7.0):
  * h, m, ts move from state index 2, 3, 4 to 1, 2, 3;
  * the heading-guess, turn-rate, fix_track and constrain_track blocks are gone —
    there is no heading control, the track is the curvature;
  * waypoints are refused (the track already pins every point).

Depends on opentop internals (private methods, the copied formulation), hence the
version check below: an upgrade must fail loudly, not solve something else.
"""
from __future__ import annotations

from collections.abc import Callable
from math import pi
from typing import Any

import casadi as ca
import numpy as np
import openap.casadi as oc
import opentop as top
from openap.aero import fpm, ft, kts
from opentop import _objectives
from opentop._track import project_track, validate_track

OPENTOP_VERSION = "2.7.0"   # the formulation below is a copy of this version's
if top.__version__ != OPENTOP_VERSION:
    raise ImportError(f"TrackCruise is written against opentop {OPENTOP_VERSION}, "
                      f"found {top.__version__}: re-check _add_formulation first.")


class TrackCruise(top.Cruise):
    """Cruise on a prescribed ground track: lateral position reduced to ``s``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._track_fn = None   # s -> (x_ref, y_ref, psi_ref), built by follow_track

    # ── track ────────────────────────────────────────────────────────────────

    def follow_track(self, lat: Any, lon: Any, *, tolerance_m: float | None = None) -> None:
        """Impose the ground track through (lat, lon).

        Same call as ``Cruise.follow_track`` so a caller can swap the class only.
        ``tolerance_m`` is accepted and IGNORED: the track is exact here.
        """
        coordinates = validate_track(lat, lon, (self.lat1, self.lon1), (self.lat2, self.lon2))
        self._track_coordinates = coordinates
        self._build_track()

    def _build_track(self) -> None:
        # Called from follow_track
        self.track_ref = project_track(*self._track_coordinates, self.proj)
        x_ref, y_ref, length = self.track_ref
        sc = ca.MX.sym("s")  # type: ignore[arg-type]
        xr, yr = x_ref(sc), y_ref(sc)
        # Heading from the spline's tangent, with opentop's convention (0 = north,
        # clockwise): atan2(dx, dy), as in Cruise.init_conditions.
        dxr, dyr = ca.jacobian(xr, sc), ca.jacobian(yr, sc)
        psi = ca.atan2(dxr, dyr)
        inner = ca.Function("track_in", [sc], [xr, yr, psi, ca.hypot(dxr, dyr)])
        # s clamped to [0, length]: the CasADi bspline returns 0 outside its knots, and
        # IPOPT relaxes bounds slightly, so an end node at s = length + 1e-9 came out
        # at (0, 0), the projection centre (42.2N 4.8E on 02a1b3_00400_0).
        s = ca.MX.sym("s")  # type: ignore[arg-type]
        x_c, y_c, psi_c, dl_ds = inner(ca.fmin(ca.fmax(s, 0.0), length))
        self._track_fn = ca.Function("track", [s], [x_c, y_c, psi_c], ["s"], ["x", "y", "psi"])
        # project_track parameterises by cumulative CHORD length, not the spline's
        # arc length, so |r'(s)| != 1: metres flown per metre of s. xdot divides by it.
        self._track_speed = ca.Function("track_speed", [s], [dl_ds], ["s"], ["dl_ds"])

    def fix_track_angle(self):
        raise ValueError("TrackCruise imposes the track; fix_track_angle does not apply")

    def enable_wind(self, windfield: Any) -> None:
        # With wind, ds/dt needs the along-track wind and a crab angle
        # (w_along + sqrt(v_h^2 - w_cross^2)), which xdot below does not model.
        raise NotImplementedError("TrackCruise has no wind model")

    def multi_start_trajectory(self, objective: str | Callable = "fuel", **kwargs: Any):
        # opentop's extra starts shift the guess in altitude AND bulge it laterally
        # (_multi_start._perturb_guess, lateral_jitter_km=100 by default); a fixed track
        # cannot take the lateral part. Altitude-only starts are fine, and so is
        # OpenTOPOptimizer's multistart: it shifts the altitude itself, one start each.
        if kwargs.get("n_starts", 5) > 1 and kwargs.get("lateral_jitter_km", 100.0) != 0:
            raise NotImplementedError("TrackCruise: pass lateral_jitter_km=0 for n_starts > 1 "
                                      "(the track is imposed; only altitude can be perturbed)")
        return super().multi_start_trajectory(objective, **kwargs)

    @property
    def _mach_fixed(self) -> bool:
        """Mach is a model constant (u = [vs]) rather than a control."""
        return self.mach_value is not None

    @property
    def _i_vs(self) -> int:
        """Index of vs in u."""
        return 0 if self._mach_fixed else 1

    def _mach(self, u: Any) -> Any:
        return self.mach_value if self._mach_fixed else u[0]

    def _virtual(self, x: Any, u: Any) -> tuple[Any, Any]:
        """The 5-state / 3-control vectors the opentop objectives index into."""
        xr, yr, psi = self._track_fn(x[0])
        return (ca.vertcat(xr, yr, x[1], x[2], x[3]),
                ca.vertcat(self._mach(u), u[self._i_vs], psi))

    # ── model ────────────────────────────────────────────────────────────────

    def xdot(self, x, u) -> ca.MX:
        """[ds, dh, dm, dts] for x = [s, h, m, ts], u = [vs] or [mach, vs]."""
        s, h, m, _ = x[0], x[1], x[2], x[3]
        mach, vs = self._mach(u), u[self._i_vs]
        v = oc.aero.mach2tas(mach, h, dT=self.dT)
        gamma = ca.arctan2(vs, v)
        dm = -self.fuelflow.enroute(m, v / kts, h / ft, vs / fpm, dT=self.dT)
        # Ground speed v·cos(gamma) along the curve, converted to a rate of s.
        return ca.vertcat(v * ca.cos(gamma) / self._track_speed(s), vs, dm, 1)

    def init_model(self, objective, *, function_name: str = "f", **kwargs):
        """Base.init_model with x = [s, h, m, ts], u = [vs] or [mach, vs]."""
        s = ca.MX.sym("s")  # type: ignore[arg-type]
        h = ca.MX.sym("h")  # type: ignore[arg-type]
        m = ca.MX.sym("m")  # type: ignore[arg-type]
        ts = ca.MX.sym("ts")  # type: ignore[arg-type]
        vs = ca.MX.sym("vs")  # type: ignore[arg-type]

        self.x = ca.vertcat(s, h, m, ts)
        self.u = vs if self._mach_fixed else ca.vertcat(ca.MX.sym("mach"), vs)  # type: ignore[arg-type]
        xv, uv = self._virtual(self.x, self.u)

        interval_dt = ca.MX.sym("dt")  # type: ignore[arg-type]
        if isinstance(objective, Callable):
            self.objective = objective
            L = self.objective(xv, uv, interval_dt, **kwargs)
        else:
            resolved = _objectives.resolve_objective(objective)
            ctx = self._objective_ctx()
            ctx.update(kwargs)
            self.objective = lambda x, u, dt, **kw: resolved(x, u, dt, **{**ctx, **kw})
            L = resolved(xv, uv, interval_dt, **ctx)

        self.func_dynamics = ca.Function(
            function_name,
            [self.x, self.u, interval_dt],
            [self.xdot(self.x, self.u), L],
            ["x", "u", "dt"],
            ["xdot", "L"],
            {"allow_free": True},
        )

        self.func_grid_cost = None
        if kwargs.get("interpolant") is not None:
            grid_L = _objectives.obj_grid_cost(
                xv,
                uv,
                interval_dt,
                proj=self.proj,
                interpolant=kwargs["interpolant"],
                n_dim=kwargs.get("n_dim"),
                time_dependent=kwargs.get("time_dependent", True),
                symbolic=True,
            )
            self.func_grid_cost = ca.Function(
                f"grid_{function_name}",
                [self.x, self.u, interval_dt],
                [grid_L],
                ["x", "u", "dt"],
                ["L"],
                {"allow_free": True},
            )

    def _constrain_performance_model_domain(self, xdot: ca.MX) -> None:
        # Base hard-codes xdot[3] as the mass rate; it is xdot[2] here.
        if self.performance_model == "bada4":
            self._opti.subject_to(-xdot[2] >= self.BADA4_MIN_FUELFLOW_KG_S)

    # ── bounds and guesses ───────────────────────────────────────────────────

    def initial_guess(self, flight: Any = None) -> np.ndarray:
        """(nodes+1, 4) guess with columns [s, h, m, ts].

        A DataFrame guess (warm start, multistart's shifted copies) is a trajectory
        that already follows the track, so ``s`` is its own cumulative projected
        distance rescaled onto [0, length]. Resampled to nodes+1 rows if needed.
        """
        length = self.track_ref[2]
        n = self.nodes + 1
        if flight is None:
            return np.column_stack([
                np.linspace(0.0, length, n),
                np.full(n, self.aircraft["cruise"]["height"]),
                np.full(n, self.mass_init),
                np.linspace(0.0, 6 * 3600, n),
            ])

        xp, yp = self.proj(np.asarray(flight.longitude), np.asarray(flight.latitude))
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xp), np.diff(yp)))])
        frac = d / d[-1] if d[-1] > 0 else np.linspace(0.0, 1.0, len(d))
        h = np.asarray(flight.altitude, dtype=float) * ft
        m = (np.asarray(flight.mass, dtype=float) if "mass" in flight
             else np.full(len(d), self.mass_init))
        if "ts" in flight:
            t = np.asarray(flight.ts, dtype=float)
        elif "timestamp" in flight:
            t = (flight.timestamp - flight.timestamp.min()).dt.total_seconds().to_numpy()
        else:
            t = np.linspace(0.0, 6 * 3600, len(d))

        target = np.linspace(0.0, 1.0, n)
        return np.column_stack([target * length] +
                               [np.interp(target, frac, col) for col in (h, m, t)])

    def init_conditions(self, **kwargs: Any) -> None:
        """Bounds and guesses for x = [s, h, m, ts], u = [vs] or [mach, vs]."""
        if self._track_coordinates is None:
            raise ValueError("TrackCruise needs follow_track(lat, lon) before solving")
        # self._build_track()
        length = self.track_ref[2]

        ts_max = max(5, length / 1000 / 500) * 3600
        h_max = kwargs.get(
            "h_max",
            self.h_max if self.h_max is not None else self.aircraft["limits"]["ceiling"],
        )
        h_min = kwargs.get("h_min", self.h_min if self.h_min is not None else 15_000 * ft)

        # s is pinned at both ends, so origin and destination are exact.
        self.x_0_lb = [0.0, h_min, self.mass_init_lb, 0]
        self.x_0_ub = [0.0, h_max, self.mass_init_ub, 0]
        self.x_f_lb = [length, h_min, self.mass_min, 0]
        self.x_f_ub = [length, h_max, self.mass_init_ub, ts_max]
        self.x_lb = [0.0, h_min, self.mass_min, 0]
        self.x_ub = [length, h_max, self.mass_init_ub, ts_max]

        if self._mach_fixed:
            self.u_0_lb = self.u_f_lb = self.u_lb = [-500 * fpm]
            self.u_0_ub = self.u_f_ub = self.u_ub = [500 * fpm]
        else:
            self.u_0_lb = self.u_f_lb = self.u_lb = [0.5, -500 * fpm]
            self.u_0_ub = self.u_f_ub = self.u_ub = [self.mach_max, 500 * fpm]

        initial_guess = kwargs.get("initial_guess", None)
        self.x_guess = self.initial_guess(initial_guess)
        if initial_guess is None:
            # Base._add_transcription seeds ts itself, but in COLUMN 4 (base.py:584),
            # which does not exist here. It is done here instead, with the same
            # flight-duration seed (ts_final_guess = length / 200), and
            # _add_formulation then always hands _add_transcription a guess so that
            # line is skipped.
            self.x_guess[:, 3] = np.linspace(0.0, length / 200, self.nodes + 1)

        self.u_guess = [0] if self._mach_fixed else [0.7, 0]

    # ── formulation (Cruise._add_formulation, re-indexed) ────────────────────

    def _add_formulation(
        self,
        opti: ca.Opti,
        objective: str | Callable = "fuel",
        *,
        max_fuel: float | None = None,
        initial_guess: Any = None,
        h_min: float | None = None,
        h_max: float | None = None,
        interpolant: Any = None,
        n_dim: int | None = None,
        time_dependent: bool = False,
        auto_rescale_objective: bool = False,
        exact_hessian: bool = False,
        waypoints: Any = None,
        waypoint_tolerance_m: float = 2_000.0,
        waypoint_node_indices: list[int] | None = None,
        variable_timestep: bool | None = None,
        dt_min: float | None = None,
        dt_max: float | None = None,
        route_margin_m: float = 10_000.0,
        name_prefix: str = "flight",
        minimize: bool = False,
    ):
        if waypoints is not None:
            raise ValueError("TrackCruise imposes the track; waypoints do not apply")
        init_kwargs: dict[str, Any] = {"initial_guess": initial_guess}
        if h_min is not None:
            init_kwargs["h_min"] = h_min
        if h_max is not None:
            init_kwargs["h_max"] = h_max
        self.init_conditions(**init_kwargs)

        kwargs = {
            # Never None, so _add_transcription skips its column-4 ts seed (see
            # init_conditions). It reads nothing else from it.
            "initial_guess": initial_guess if initial_guess is not None else self.x_guess,
            "interpolant": interpolant,
            "n_dim": n_dim,
            "time_dependent": time_dependent,
            "auto_rescale_objective": auto_rescale_objective,
            # The track spline enters the dynamics and every objective: exact Hessian,
            # as Cruise does whenever follow_track is set.
            "exact_hessian": True,
            "variable_timestep": bool(variable_timestep),
            "dt_min": dt_min,
            "dt_max": dt_max,
            "route_margin_m": route_margin_m,
        }
        if dt_max is None:
            kwargs.pop("dt_max")

        transcription = self._add_transcription(
            opti,
            objective,
            ts_final_guess=self.track_ref[2] / 200,
            minimize=minimize,
            name_prefix=name_prefix,
            **kwargs,
        )
        X, U = transcription.X, transcription.U
        i_vs = self._i_vs

        # Nonlinear performance constraints need checks inside each interval.
        for state, control in transcription.path_points():
            v = oc.aero.mach2tas(self._mach(control), state[1], dT=self.dT)
            tas, alt = v / kts, state[1] / ft
            self._constrain_clean_performance(
                opti, state[2], tas, alt, self._thrust_climb(tas, alt)
            )

        # ts and dt consistency
        for k in range(self.nodes):
            opti.subject_to(
                opti.bounded(-1, X[k + 1][3] - X[k][3] - self._interval_dt(k), 1)  # type: ignore[arg-type]
            )

        # Limit vertical acceleration independently of interval duration
        for k in range(self.nodes):
            vertical_acceleration = self._control_change_rate(U, k, i_vs)
            opti.subject_to(
                opti.bounded(
                    -self.MAX_VERTICAL_ACCELERATION,
                    vertical_acceleration,
                    self.MAX_VERTICAL_ACCELERATION,
                )  # type: ignore[arg-type]
            )

        # mach_value: already a model constant, nothing to pin. Otherwise
        # fix_mach keeps the Mach control constant, as in Cruise.
        if not self._mach_fixed and self.fix_mach:
            for k in range(self.nodes):
                opti.subject_to(U[k + 1][0] == U[k][0])

        if self.fix_alt:
            for k in range(self.nodes + 1):
                opti.subject_to(U[k][i_vs] == 0)

        if not self.allow_descent:
            for k in range(self.nodes + 1):
                opti.subject_to(U[k][i_vs] >= 0)

        opti.subject_to(opti.bounded(0, X[0][2] - X[-1][2], self.fuel_max))  # type: ignore[arg-type]
        if max_fuel is not None:
            opti.subject_to(X[0][2] - X[-1][2] <= max_fuel)

        return transcription

    # ── output ───────────────────────────────────────────────────────────────

    def to_trajectory(self, ts_final: float, x_opt: Any, u_opt: Any, **kwargs: Any):
        """Rebuild [xp, yp, h, m, ts] / [mach, vs, psi] from s, then Base's export.

        The DataFrame therefore has every usual column (latitude, longitude,
        heading, grid_cost, ...), plus ``s``: the arc length along the track, m.
        """
        X = np.atleast_2d(np.asarray(x_opt, dtype=float))
        U = np.atleast_2d(np.asarray(u_opt, dtype=float))
        s = X[0]
        xr, yr, psi = (v.full().ravel()
                       for v in self._track_fn.map(s.size)(s.reshape(1, -1)))
        X5 = np.vstack([xr, yr, X[1], X[2], X[3]])
        mach = np.full(s.size, self.mach_value) if self._mach_fixed else U[0]
        U5 = np.vstack([mach, U[self._i_vs], psi % (2 * pi)])
        df = super().to_trajectory(ts_final, X5, U5, **kwargs)
        return df.assign(s=s)
