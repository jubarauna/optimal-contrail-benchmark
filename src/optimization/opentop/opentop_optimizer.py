"""
OpenTOPOptimizer
================
Wraps an opentop continuous optimal-control solve behind the common
:class:`BaseOptimizer` contract, so it is interchangeable with the graph models
in a ``ParetoSweep`` or any driver.

Mapping to the universal interface:
  optimize()          → weighted solve  (contrail_weight·grid + fuel_weight·fuel)
  solve_epsilon(b)    → native constrained solve (minimise contrail s.t. max_fuel=b)
  pareto_front(m, pts)→ a *sweep* over the grid (one solve per point): opentop has
                        no single-solve multi-objective mode, so the front is built
                        by re-solving at each weight / fuel budget.

Unlike the graph models, opentop integrates fuel/contrail itself, so ``evaluate``
uses those native trajectory metrics — unless a grid ``environment`` is supplied,
in which case it defers to the shared post-hoc ``evaluate_path`` for cross-family
consistency.
"""
from __future__ import annotations
import tempfile
import opentop as top
import time
from pathlib import Path
import os
from src.models.grid.contrail_grid import prepare_grid
import numpy as np
from openap.aero import ft
from src.data import FlightData
from src.models.climate.contrails import agg_conditions
from src.optimization.base import BaseOptimizer, OptResult


class OpenTOPOptimizer(BaseOptimizer):
    name = "opentop"

    def __init__(
        self,
        cruise,                       # configured top.Cruise (track/mach fixed, cruise-descent)
        interpolant,                  # cached contrail interpolant
        *,
        h_min_ft: float | None = None,
        h_max_ft: float | None = None,
        recenter_window_ft: float | None = None,   # see warm_start()
        mach: float,
        mass_init_kg: float,
        actype: str,
        environment=None,             # optional grid env; None → opentop's native metrics
        fuel_weight: float = 1.0,
        contrail_weight: float = 0.0,
        initial_guess=None,           # fuel-only warm start (see warm_start())
        warm_start: bool = True,      # seed singlestart WS/EC solves with the fuel-only guess
                                      # (multistart always starts from it, see _solve)
        n_dim: int = 4,
        time_dependent: bool = True,
        multistart: bool = False,     # solve from several altitude-shifted initial guesses
        altitude_offsets_ft: tuple = (-2000.0, -1000.0, 1000.0, 2000.0),
        centre_alt_ft: float | None = None,   # multistart: altitude the perturbed starts
                                              # are centred on (the real flight's mean);
                                              # None = the window's centre
        multistart_fuel_only: bool = False,   # also multistart the fuel-only solve (see
                                              # warm_start); independent of multistart
    ):
        super().__init__(
            environment=environment,
            mach=mach, mass_init_kg=mass_init_kg, actype=actype,
            fuel_weight=fuel_weight, contrail_weight=contrail_weight,
        )
        self.cruise = cruise
        self.interpolant = interpolant
        self.h_min_ft = h_min_ft
        self.h_max_ft = h_max_ft
        # When set, the fuel-only warm start runs UNBOUNDED in altitude (full
        # aircraft envelope) and the contrail solves are then confined to the
        # fuel-only trajectory's mean altitude ± this many feet. See warm_start().
        self.recenter_window_ft = recenter_window_ft
        self._recentered = False
        if recenter_window_ft is None and (h_min_ft is None or h_max_ft is None):
            raise ValueError(
                "OpenTOPOptimizer needs an altitude window: pass h_min_ft/h_max_ft, "
                "or set recenter_window_ft to derive it from the fuel-only warm start."
            )
        self.n_dim = n_dim
        self.time_dependent = time_dependent
        self._initial_guess = initial_guess
        self._use_warm_start = warm_start

        # Multistart config (used by warm_start() and every solve when enabled): one
        # start from the basis trajectory, then one per offset. See _multi_start().
        self.multistart = multistart
        self.altitude_offsets_ft = tuple(float(o) for o in altitude_offsets_ft)
        self.n_starts = 1 + len(self.altitude_offsets_ft)
        self.centre_alt_ft = centre_alt_ft
        self.multistart_fuel_only = multistart_fuel_only
        self._candidates = None       # failed/other starts from the last multistart solve
        self.results: list[OptResult] = []   # populated by pareto_front()
        # Sweep-level setup cost, excluded from every per-point time_s. Set by
        # warm_start() when it actually solves.
        self.warm_start_s = 0.0
        # The fuel-only solve's own outcome, set by warm_start() when it solves. Read
        # these, not cruise.success/stats: after a multistart those describe the last
        # start, not the winner, and after a sweep they describe the sweep's last solve.
        self.warm_start_success = None
        self.warm_start_objective = None
        self.warm_start_n_iter = -1

    # ── objectives ───────────────────────────────────────────────────────────

    def _weighted_objective(self):
        """contrail_weight·grid_cost + fuel_weight·fuel (the weighted-sum scalarisation)."""
        cr, cw, fw = self.cruise, self.contrail_weight, self.fuel_weight
        nd, td = self.n_dim, self.time_dependent

        def objective(x, u, dt, **kwargs):
            kwargs.pop("n_dim", None)
            kwargs.pop("time_dependent", None)
            grid = cr.obj_grid_cost(x, u, dt, time_dependent=td, n_dim=nd, **kwargs)
            fuel = cr.obj_fuel(x, u, dt, **kwargs)
            return cw * grid + fw * fuel
        return objective

    def _contrail_objective(self):
        """Pure contrail objective (used under a fuel budget in solve_epsilon)."""
        cr, nd, td = self.cruise, self.n_dim, self.time_dependent

        def objective(x, u, dt, **kwargs):
            kwargs.pop("n_dim", None)
            kwargs.pop("time_dependent", None)
            return cr.obj_grid_cost(x, u, dt, time_dependent=td, n_dim=nd, **kwargs)
        return objective

    # ── solves ───────────────────────────────────────────────────────────────

    def _multi_start(self, kw: dict, basis=None):
        """Start 0 from ``basis``, then one start per altitude offset → (traj, candidates).

        ``basis`` is the initial guess of start 0; None means opentop's default guess,
        and the offsets are then applied to start 0's solution instead. Start k keeps
        the basis's SHAPE but has its mean altitude moved to ``centre_alt_ft`` (the real
        flight's mean altitude; the window's centre when not given) plus offset k; its
        lateral track is untouched. Not offsets OF the basis: the fuel-only trajectory
        often sits near one end of the window (37 400 ft in 31 000-39 000 on
        396670_02638_0), and offsets of it then explore only that half -- there, the
        low-contrail solution 5 000 ft below was never reached.

        Each candidate says which initial guess it came from: ``initial_guess`` is
        "fuel_only" (start 0 from the basis), "default" (start 0 without one) or
        "fuel_only_shape_at_centre" (starts 1..), with ``guess_mean_alt_ft`` the guess's
        mean altitude and ``perturbation["altitude_ft"]`` its offset from the centre.

        Every start goes through opentop's own multi_start_trajectory with one start,
        so each candidate has opentop's fields (objective, success, return_status,
        iters, perturbation, trajectory, ...). The ranking is opentop's: converged
        before not converged, then lowest objective. ``cruise.objective_value`` and
        ``grid_cost_value`` are set to the winner's, as opentop does.
        """
        def one(index, guess, offset_ft, kind):
            k = dict(kw, return_failed=True)
            if guess is not None:
                k["initial_guess"] = guess
            traj, (cand,) = self.cruise.multi_start_trajectory(n_starts=1, **k)
            cand["start_index"] = index
            cand["initial_guess"] = kind
            cand["guess_mean_alt_ft"] = (float(np.mean(guess["altitude"]))
                                         if guess is not None else None)
            cand["perturbation"] = {"lateral_km": 0.0, "altitude_ft": float(offset_ft)}
            return traj, cand

        traj0, cand0 = one(0, basis, 0.0, "fuel_only" if basis is not None else "default")
        candidates = [cand0]
        base = basis if basis is not None else traj0
        centre_ft = (self.centre_alt_ft if self.centre_alt_ft is not None
                     else 0.5 * (self.h_min_ft + self.h_max_ft))
        recentred = base["altitude"] - float(np.mean(base["altitude"])) + centre_ft
        for i, offset in enumerate(self.altitude_offsets_ft, start=1):
            guess = base.copy()
            guess["altitude"] = recentred + offset
            candidates.append(one(i, guess, offset, "fuel_only_shape_at_centre")[1])

        candidates.sort(key=lambda c: (not c["success"], c["objective"]))
        self.cruise.objective_value = candidates[0]["objective"]
        self.cruise.grid_cost_value = candidates[0]["grid_cost_exact"]
        return candidates[0]["trajectory"], candidates

    def warm_start(self):
        """Fuel-only trajectory → cached as the initial guess for the contrail sweeps.

        Idempotent: a guess already provided (constructor ``initial_guess=``) or
        computed by an earlier call is reused, so the WS and EC sweeps on one
        instance share a single fuel-only solve instead of recomputing it.

        Altitude window:
          * default — the fuel-only solve uses the configured ``h_min_ft``/
            ``h_max_ft``, and the contrail sweeps use the same window.
          * ``recenter_window_ft`` set — the fuel-only solve is UNBOUNDED in
            altitude (opentop's full envelope, ~15 000 ft → ceiling), so its
            optimum altitude is unconstrained; the window for the contrail sweeps
            is then set to that trajectory's mean altitude ± ``recenter_window_ft``.

        This is a full solve, and it sits OUTSIDE the per-point timers in
        ``_solve`` — the sweep pays for it once. Its cost is recorded in
        ``self.warm_start_s`` (0.0 when a guess was supplied and no solve ran)
        so a caller can report it alongside the per-point times. The EC sweep
        needs it even with ``warm_start=False``, for the F0 baseline.
        """
        if self._initial_guess is not None:
            traj = self._initial_guess
        else:
            base = dict(
                objective="fuel", interpolant=self.interpolant,
                n_dim=self.n_dim, time_dependent=self.time_dependent,
            )
            # Bound the fuel-only solve only when NOT recentering; otherwise let it
            # roam the full envelope so its optimum altitude drives the window.
            if self.recenter_window_ft is None:
                base["h_min"] = self.h_min_ft * ft
                base["h_max"] = self.h_max_ft * ft
            t0 = time.perf_counter()
            if self.multistart_fuel_only:
                # Start 0 from opentop's default guess, then start 0's solution
                # recentred on centre_alt_ft + each altitude offset (_multi_start with
                # no basis). The winner is the fuel-only trajectory; its candidates go
                # into the fuel-only OptResult.
                traj, self._candidates = self._multi_start(base, basis=None)
                best = self._candidates[0]
                self.warm_start_success = best["success"]
                self.warm_start_n_iter = int(best["iters"]) if best["iters"] is not None else -1
                self.warm_start_objective = best["objective"]
            else:
                # ONE solve from opentop's default initial guess: by default the
                # fuel-only anchor is not multistarted, only the contrail sweeps are.
                # _candidates is cleared so its OptResult does not carry a previous solve's.
                traj = self.cruise.trajectory(**base)
                self._candidates = None
                self.warm_start_success = self.cruise.success
                try:
                    self.warm_start_n_iter = int(self.cruise.stats.get("iter_count", -1))
                except Exception:  # noqa: BLE001 — failed solve → unknown count
                    self.warm_start_n_iter = -1
                self.warm_start_objective = self.cruise.objective_value
            self.warm_start_s = time.perf_counter() - t0
            print(f"[OpenTOPOptimizer] warm start (fuel-only baseline): {self.warm_start_s:.2f} s")
            self._initial_guess = traj

        # Confine the contrail sweeps to the fuel-only trajectory's altitude band.
        if self.recenter_window_ft is not None and not self._recentered:
            if traj is None:
                raise RuntimeError(
                    "fuel-only warm start returned no trajectory; cannot recenter "
                    "the altitude window (pass explicit h_min_ft/h_max_ft instead)."
                )
            ref_alt = float(np.nanmean(traj["altitude"]))    # ft
            self.h_min_ft = ref_alt - self.recenter_window_ft
            self.h_max_ft = ref_alt + self.recenter_window_ft
            self._recentered = True
            print(f"[OpenTOPOptimizer] contrail altitude window recentered on the "
                  f"fuel-only trajectory: {ref_alt:.0f} ± {self.recenter_window_ft:.0f} ft "
                  f"→ [{self.h_min_ft:.0f}, {self.h_max_ft:.0f}] ft")
        return traj

    def _solve(self, objective, extra_kwargs: dict | None = None):
        """Run one solve → (traj, elapsed, success, objective_value, n_iter).
        ``objective_value`` is the solver's physical objective at the solution
        (opentop ``Cruise.objective_value``); ``n_iter`` is the IPOPT iteration
        count from the solver stats. Stores candidates on multistart."""
        kw = dict(
            objective=objective, interpolant=self.interpolant,
            h_min=self.h_min_ft * ft, h_max=self.h_max_ft * ft,
            n_dim=self.n_dim, time_dependent=self.time_dependent,
        )
        if extra_kwargs:
            kw.update(extra_kwargs)
        t0 = time.perf_counter()
        if self.multistart:
            # The fuel-only trajectory is the basis of every start: start 0 solves
            # from it as given, the others from it shifted by each altitude offset.
            # Independent of warm_start, which only governs singlestart. warm_start()
            # is idempotent, so this reuses the fuel-only solve rather than repeating it.
            traj, self._candidates = self._multi_start(kw, basis=self.warm_start())
            # The winner's status and iterations. cruise.success and cruise.stats
            # describe the LAST start solved, which is not necessarily the winner.
            best = self._candidates[0]
            return (traj, time.perf_counter() - t0, best["success"],
                    best["objective"], best["iters"])
        if self._use_warm_start and self._initial_guess is not None:
            kw["initial_guess"] = self._initial_guess
        traj = self.cruise.trajectory(**kw)
        self._candidates = None
        try:
            n_iter = int(self.cruise.stats.get("iter_count", -1))
        except Exception:  # noqa: BLE001 — no/failed solve → unknown iteration count
            n_iter = -1
        return (traj, time.perf_counter() - t0, self.cruise.success,
                self.cruise.objective_value, n_iter)

    # ── BaseOptimizer contract ───────────────────────────────────────────────

    def optimize(self) -> OptResult | None:
        traj, elapsed, success, objective, n_iter = self._solve(self._weighted_objective())
        return self._traj_to_result(traj, elapsed, success, method="weighted_sum",
                                    objective=objective, n_iter=n_iter)

    def solve_epsilon(self, fuel_budget_kg: float) -> OptResult | None:
        """Native epsilon-constraint: minimise contrail subject to fuel ≤ budget."""
        traj, elapsed, success, objective, n_iter = self._solve(
            self._contrail_objective(), extra_kwargs={"max_fuel": fuel_budget_kg},
        )
        r = self._traj_to_result(traj, elapsed, success, method="epsilon_constraint",
                                 objective=objective, n_iter=n_iter)
        if r is not None:
            r.meta["fuel_budget_kg"] = fuel_budget_kg
        return r

    def pareto_front(self, method: str, points: list[float]):
        """
        Build a Pareto front by *sweeping* the grid — opentop has no single-solve
        multi-objective mode, so each point is its own solve (weighted-sum, or a
        native ``max_fuel`` epsilon-constraint). Returns (and stores) the results.
        """
        if method == "weighted_sum":
            self._run_weighted_sum(points)
        elif method == "epsilon_constraint":
            self._run_epsilon_constraint(points)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'weighted_sum' or 'epsilon_constraint'.")

    def _run_weighted_sum(self, coef_grid: list[float]):
        if self._use_warm_start:
            self.warm_start()
        tag = "multistart" if self.multistart else "singlestart"
        print(f"\n[OpenTOPOptimizer] weighted-sum sweep ({tag}) — {len(coef_grid)} coef values")
        for coef in coef_grid:
            # coef weights the contrail objective; (1-coef) weights fuel.
            self.contrail_weight = coef
            self.fuel_weight = 1.0 - coef
            res = self.optimize()
            if res is None:
                print(f"  [coef={coef:.3f}] no trajectory returned — skipping.")
                continue
            # Non-converged solves are retained (success=False) with their trajectory.
            print(
                f"  [coef={coef:.3f}] "
                f"fuel={res.fuel_kg:.1f} kg  contrail={res.contrail:.4f}  "
                f"success={res.success}  time={res.meta['time_s']:.2f} s"
            )
            self._tag_and_store(res, coef=coef)

    def _run_epsilon_constraint(self, fuel_slack_fractions: list[float]):
        """Minimise contrail subject to fuel ≤ F0 * fraction via native ``max_fuel``."""
        # Always needed here for the fuel-only baseline F0 that defines the
        # budgets — even when warm_start=False (then _solve just won't use it as a
        # seed). Idempotent, so it reuses the WS sweep's guess on the same instance.
        traj = self.warm_start()                                          # fuel-only baseline
        F0   = float(traj["mass"].iloc[0] - traj["mass"].iloc[-1])        # baseline fuel (kg)
        tag  = "multistart" if self.multistart else "singlestart"
        print(f"\n[OpenTOPOptimizer] epsilon-constraint sweep ({tag}) — {len(fuel_slack_fractions)} budget points")
        for frac in fuel_slack_fractions:
            budget = F0 * frac
            res = self.solve_epsilon(budget)
            if res is None:
                print(f"  [frac={frac:.3f}, budget={budget:.1f} kg] no trajectory returned — skipping.")
                continue
            print(
                f"  [frac={frac:.3f}, budget={budget:.1f} kg] "
                f"fuel={res.fuel_kg:.1f} kg  contrail={res.contrail:.4f}  "
                f"success={res.success}  time={res.meta['time_s']:.2f} s"
            )
            self._tag_and_store(res, coef=None, fuel_slack_frac=frac)  # fuel_budget_kg set by solve_epsilon

    def _tag_and_store(self, res: OptResult, **meta) -> None:
        """Attach sweep metadata (coef / fuel_slack_frac) and keep the point."""
        res.meta.update(meta)
        self.results.append(res)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _traj_to_result(self, traj, elapsed, success, method, objective=None,
                        n_iter=None) -> OptResult | None:
        # Retain non-converged solves (success=False) so a Pareto sweep keeps the
        # point; only a genuinely absent trajectory yields None.
        if traj is None:
            return None
        x, y = traj["x"].to_numpy(), traj["y"].to_numpy()
        dist_km = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))]) / 1000.0
        # Contrail = the solver's own integral (Legendre quadrature at the collocation
        # points, the J_grid it minimised), which opentop leaves in grid_cost_value
        # after each solve -- for multistart, the winner's (set in _multi_start).
        # traj["grid_cost"] is NOT that: opentop's per-node diagnostic, a left-endpoint
        # rectangle over each collocation interval, which over a field that falls along the
        # route charges every interval its start's cost. Kept beside it for comparison,
        # and as the fallback when no solve left a value (a reused initial guess).
        left_endpoint = float(np.nansum(traj["grid_cost"]))
        quadrature = getattr(self.cruise, "grid_cost_value", None)
        return OptResult(
            alt_ft   = traj["h"].to_numpy() / ft,
            dist_km  = dist_km,
            ts_s     = traj["ts"].to_numpy(),
            lat      = traj["latitude"].to_numpy(),
            lon      = traj["longitude"].to_numpy(),
            fuel_kg  = float(traj["mass"].iloc[0] - traj["mass"].iloc[-1]),   # native
            contrail = float(quadrature) if quadrature is not None else left_endpoint,
            cost     = objective,   # native solver objective value (Cruise.objective_value)
            success  = bool(success),
            meta     = {"optimizer": self.name, "method": method,
                        "time_s": elapsed, "objective": objective, "n_iter": n_iter,
                        "trajectory": traj,
                        "contrail_source": ("quadrature" if quadrature is not None
                                            else "left_endpoint"),
                        "contrail_left_endpoint": left_endpoint,
                        "multistart": self.multistart or (method == "fuel_only"
                                                          and self.multistart_fuel_only),
                        "candidates": self._candidates},
        )


# -- example / smoke test -----------------------------------------------------
# Runs a real opentop solve for the Berlin-Madrid flight. Slow (fetches meteo,
# builds the interpolant, then solves) — run deliberately from the repo root:
#     python -m src.optimization.opentop.optimize
if __name__ == "__main__":
    ALT_STEP_FT = 1000.0
    MIN_ALT_FT  = 29000.0
    MAX_ALT_FT  = 45000.0
    MAX_ROC     = 500.0
    sigma       = 2
    mach        = 0.78
    flight_name = "Berlin-Madrid"
    begin = "2023-01-05 09:48"
    end = "2023-01-05 11:00"
    callsign = "RYR880W"
    actype = "B738"
    # 1) Flight setup — same identity as 2_Run/batch_opentop.py.
    #    Reuse _ParetoBase to load the flight, build the cost grid + interpolant,
    #    and hand back a configured top.Cruise + mach / mass / altitude bounds.
    TMP = Path(tempfile.gettempdir())
    interpolant_cache = os.environ.get(
        "INTERPOLANT_CACHE",
        str(TMP / f"opentop_interp_{flight_name}_sigma{sigma}.casadi"),
    )
    fd = FlightData(start=begin, stop=end, callsign=callsign)
    fd.interpolate_meteo(time_buffer_hours=2.0)
    meteo = agg_conditions(fd.meteo)
    df_cost = prepare_grid(meteo, sigma, t_reference=fd.start)
    interpolant = top.tools.cached_interpolant_from_dataframe(
        df_cost, interpolant_cache, shape="bspline"
    )
    optimizer = top.Cruise(
        actype,
        (fd.initial_latitude, fd.initial_longitude),
        (fd.final_latitude, fd.final_longitude),
        m0=0.85
    )
    optimizer.fix_track_angle()
    optimizer.fix_mach_number()
    optimizer.allow_cruise_descent()

    # 2) The optimizer under test. environment=None keeps opentop's native
    #    fuel/contrail; pass setup.env to score it on the shared graph physics.
    opt = OpenTOPOptimizer(
        cruise       = optimizer,
        interpolant  = interpolant,
        h_min_ft     = MIN_ALT_FT,
        h_max_ft     = MAX_ALT_FT,
        mach         = mach,
        mass_init_kg = optimizer.mass_init,
        actype       = actype
    )

    # 4) Weighted-sum solves at a few contrail weights (BaseOptimizer.optimize).
    print("\n-- weighted-sum (optimize) --")
    opt.pareto_front(method="weighted_sum", points=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    # 5) Native epsilon-constraint: minimise contrail s.t. fuel ≤ 1.05·F0.
    print("\n-- epsilon-constraint (solve_epsilon) --")
    opt.pareto_front(method="epsilon_constraint", points=[1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.10])

    print("\n-- results --")
    for res in opt.results:
        print(f"  method={res.meta['method']}  "
              f"fuel={res.fuel_kg:.1f} kg  contrail={res.contrail:.4f}  " )


