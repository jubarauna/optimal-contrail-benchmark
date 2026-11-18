"""
No-oscillation trajectory optimizers on the 4-tuple anti-oscillation graph.

State: (dist, alt, prev_alt, ts) — no zigzag by construction.

No-oscillation rule (same as a_star_no_osc.py):
  alt > prev_alt  →  next_alt >= alt   (continuing climb or level)
  alt < prev_alt  →  next_alt <= alt   (continuing descent or level)
  alt == prev_alt →  any next_alt      (level move → full freedom)

Direction index convention used in DP tables:
  0 = descent  (alt < prev_alt)
  1 = level    (alt == prev_alt)
  2 = climb    (alt > prev_alt)

Structure
---------
``_NoOscBase`` holds everything shared: physics, mass profile, edge-cost
precomputation, graph construction, heuristics and path utilities.  Each
algorithm lives in its own subclass with a single ``optimize()`` method (the
original public name — e.g. ``optimize_dp`` — is kept as an alias):

  DagOptimizer              DAG topological relaxation           O(V+E)
  DijkstraOptimizer         NetworkX Dijkstra
  AstarSuffixSumOptimizer   A* with admissible suffix-sum heuristic
  AstarDpOptimizer          A* with exact backward-DP heuristic
  DpOptimizer               Vectorized DP over (waypoint, alt, direction)
  DpMuoOptimizer   Multi-objective label-setting Pareto front

``TrajectoryOptimizerNoOscAll`` is a facade that inherits all of them, so a
single instance can run every method while sharing one built graph (used by the
comparison/benchmarking drivers).
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import dijkstra_path
from src.data.route import real_flight_route, real_flight_baseline
from openap import aero
from src.performance.aircraft import aircraft as ac_props, fuel_flow_model, thrust_model, drag_model
from openap.aero import ft, kts
import tempfile
import opentop as top
from src.data.traffic import FlightData
from src.contrails.conditions import agg_conditions
from pathlib import Path
import os
from src.contrails.cost_grid import prepare_grid
from src.optimization.environment import ContrailEnvironmentFromGrid
from src.optimization.base import BaseOptimizer, OptResult
from src.postprocess.evaluation import evaluate_path
import time
import matplotlib.pyplot as plt
from src.postprocess.plots import plot_cost_map

g0 = 9.80665


# ── Pareto labels ────────────────────────────────────────────────────────────

@dataclass
class Label:
    """A non-dominated (fuel, contrail) cost reaching ``node`` = (step, alt_idx)."""
    fuel: float
    contrail: float
    node: tuple                       # (step_index, altitude_index)
    parent: Optional["Label"] = None


def dominates(a: Label, b: Label) -> bool:
    """True if ``a`` Pareto-dominates ``b`` (minimisation of both objectives)."""
    return (a.fuel <= b.fuel and a.contrail <= b.contrail
            and (a.fuel < b.fuel or a.contrail < b.contrail))


def add_if_nondominated(new_label: Label, pareto_list: list[Label]) -> bool:
    """
    Insert ``new_label`` into ``pareto_list`` iff it is not dominated by (nor a
    duplicate of) an existing label; prune the labels it dominates in place.
    Returns True if inserted.
    """
    for existing in pareto_list:
        if dominates(existing, new_label):
            return False
        if existing.fuel == new_label.fuel and existing.contrail == new_label.contrail:
            return False  # identical objective vector already present
    pareto_list[:] = [l for l in pareto_list if not dominates(new_label, l)]
    pareto_list.append(new_label)
    return True


# ── Shared base ──────────────────────────────────────────────────────────────

class _model_2D_base(BaseOptimizer):
    """
    Physics, profiles, edge-cost precomputation, graph construction, heuristics
    and path utilities shared by every no-oscillation solver.

    Now a :class:`BaseOptimizer` so the 2D models share the universal contract:
    every concrete solver aliases ``optimize`` to its own method, and ``evaluate``
    (fuel/contrail + input stamping) is inherited unchanged.

    Subclasses set ``_DEFAULT_OPTIMIZER`` to declare which precomputation their
    ``optimize()`` needs; it can be overridden per instance via ``optimizer=``.
    """

    name: str = "DP"
    _DEFAULT_OPTIMIZER: Optional[str] = "dp"

    def __init__(
        self,
        environment: ContrailEnvironmentFromGrid,
        cruise_alt_ft: float = 34000,
        mach: float = 0.78,
        mass_init_kg: float = 65000.0,
        fuel_weight: float = 1.0,
        contrail_weight: float = 0.0,
        max_climb_rate_fpm: float = 500.0,
        max_descent_rate_fpm: float = 500.0,
        actype: str = "A320",
        optimizer: Optional[str] = None,
    ):
        self.env = environment
        self.mach = mach
        self.mass_init_kg = mass_init_kg
        self.fuel_weight = fuel_weight
        self.contrail_weight = contrail_weight
        self.cruise_alt_ft = cruise_alt_ft
        self.max_climb_rate_fpm = max_climb_rate_fpm
        self.max_descent_rate_fpm = max_descent_rate_fpm
        self.actype = actype

        self.fuel_model  = fuel_flow_model(actype)
        self.thrust_model = thrust_model(actype)
        self.drag_model  = drag_model(actype)
        aircraft = ac_props(actype)
        self.S   = aircraft["wing"]["area"]
        polar    = self.drag_model.polar["clean"]
        self.cd0 = polar["cd0"]
        self.ck  = polar["k"]

        opt = optimizer if optimizer is not None else self._DEFAULT_OPTIMIZER
        self._optimizer = opt

        # Construction-time precompute is the openap-heavy part (mass profile +
        # per-edge fuel/thrust/drag), and it is deliberately NOT inside any
        # solver timer: the tables are coef-independent, so a Pareto sweep pays
        # for them once and every point after that is pure recursion. Timing it
        # here lets a caller report "setup + N × solve" instead of silently
        # amortising the setup to zero.
        # Defaults first: _compute_heuristic_tables below may set its own timer,
        # and these must not clobber it afterwards.
        self.graph_build_s = 0.0
        self.dp_heuristic_s = 0.0
        self.suffix_heuristic_s = 0.0
        # Post-hoc evaluation (``evaluate_path``) accumulated over every point this
        # instance produces. It is INSTRUMENTATION, not solving: both families pay it
        # only so their fuel/contrail are reported with identical physics. It is
        # measured apart because it dominates — ~90% of a DP "solve" is this call, not
        # the recursion — and because the two solvers charge it to different phases.
        self.posthoc_s = 0.0
        self.posthoc_calls = 0
        # One entry per DP solve (_solve_dp_point), infeasible ones included: the
        # per-record time_s covers only the accepted points.
        self.dp_solve_log: list[dict] = []

        _t0 = time.perf_counter()
        self._compute_mass_profile()
        if opt == "astar_suffix_sum":
            self._compute_heuristic_tables()   # suffix sums  → A* suffix_sum
        elif opt in ("astar_dp", "dp", "pareto"):
            self._precompute_edge_costs()      # edge cost arrays → DP / DP heuristic / Pareto
        self.setup_s = time.perf_counter() - _t0

        # Graph is built lazily on first call to a graph-based method.
        self.graph = None

        print(f"\n✓ {type(self).__name__} initialized (graph not yet built)")
        print(f"  contrail_weight={contrail_weight}  fuel_weight={fuel_weight}")
        print(f"  setup (mass profile + precompute): {self.setup_s:.2f}s")
        if hasattr(self, "_edge_cost_array"):
            n_alts  = len(self.env.altitudes)
            n, _, span = self._edge_cost_array.shape
            print(f"  edge table: {n}×{n_alts}×{span}  ({np.isfinite(self._edge_cost_array).sum()} feasible)")

    # ── Physics ────────────────────────────────────────────────────────────────

    def _is_feasible(self, mass_kg: float, alt_ft: float, vs_fpm: float) -> bool:
        alt_m     = alt_ft * ft
        v         = aero.mach2tas(self.mach, alt_m)
        rho       = aero.density(alt_m)
        q         = 0.5 * rho * v ** 2
        tas_kts   = v / kts
        thrust_max = self.thrust_model.cruise(tas=tas_kts, alt=alt_ft)
        drag      = self.drag_model.clean(mass=mass_kg, tas=tas_kts, alt=alt_ft, vs=vs_fpm)
        if thrust_max * 0.95 < drag:
            return False
        drag_max = thrust_max * 0.9
        cd_max   = drag_max / (q * self.S + 1e-10)
        cl2      = (cd_max - self.cd0) / self.ck
        if cl2 <= 1e-10:
            return False
        if np.sqrt(cl2) * q * self.S * 0.80 < mass_kg * g0:
            return False
        return True

    def _edge_cost(self, dist1, alt1, dist2, alt2, ts1, ts2):
        """Edge cost: (dist1,alt1,ts1) → (dist2,alt2,ts2)."""
        dt_s     = ts2 - ts1
        avg_alt  = (alt1 + alt2) / 2.0
        avg_ts   = (ts1 + ts2) / 2.0
        vs_fpm   = ((alt2 - alt1) / dt_s) * 60.0 if dt_s > 0 else 0.0
        tas      = aero.mach2tas(self.mach, avg_alt * ft) / kts
        # Mass at the edge's START node — the same left-endpoint convention as
        # ``_precompute_edge_costs`` (``mass = self._mass_at_dist[i]``) and as
        # ``evaluate_path``, which burns fuel at the incoming mass and only then
        # decrements it. Looking it up on ``avg_dist`` instead put this path one
        # waypoint ahead of the DP tables, so the same trajectory priced through the
        # graph and through the tables did not agree.
        # ``side="right" - 1`` rather than the default: dist1 IS a node value, and a
        # left search would step to the next node on the slightest float excess.
        dist_idx = int(np.clip(np.searchsorted(self.env.distances, dist1, side="right") - 1,
                               0, len(self._mass_at_dist) - 1))
        mass = self._mass_at_dist[dist_idx]
        ff   = self.fuel_model.enroute(mass=mass, tas=tas, alt=avg_alt, vs=vs_fpm)
        fuel_cost     = self.fuel_weight * max(0.0, ff) * dt_s
        contrail_cost = self.contrail_weight * self.env.get_contrail_cost_by_ts(avg_ts, avg_alt) * dt_s
        return fuel_cost + contrail_cost

    # ── Profiles ───────────────────────────────────────────────────────────────

    def _compute_mass_profile(self):
        n    = len(self.env.distances)
        dt_s = float(self.env.ts[1] - self.env.ts[0])
        self._mass_at_dist    = np.zeros(n)
        self._mass_at_dist[0] = self.mass_init_kg
        for i in range(n - 1):
            mass    = self._mass_at_dist[i]
            best_ff = float('inf')
            for alt in self.env.altitudes:
                if not self._is_feasible(mass, alt, 0.0):
                    continue
                tas_kts = aero.mach2tas(self.mach, alt * ft) / kts
                ff = max(0.0, self.fuel_model.enroute(mass=mass, tas=tas_kts, alt=alt, vs=0))
                if ff < best_ff:
                    best_ff = ff
            self._mass_at_dist[i + 1] = mass - (best_ff if best_ff < float('inf') else 0.0) * dt_s

    def _compute_heuristic_tables(self):
        """
        Admissible suffix-sum lower bounds on fuel and contrail cost-to-go.
        Uses the lightest possible mass profile (depleted at max burn rate) so
        the heuristic never over-estimates.

        Cost recorded in ``self.suffix_heuristic_s`` (it is openap-heavy, and is
        called out of __init__ by callers that want A* suffix_sum alongside
        another optimizer).
        """
        _t0  = time.perf_counter()
        n    = len(self.env.distances)
        dt_s = float(self.env.ts[1] - self.env.ts[0])

        # Minimum possible mass profile: deplete at maximum fuel flow each step.
        self._min_mass = np.zeros(n)
        self._min_mass[0] = self.mass_init_kg
        max_ff = np.zeros(n)
        for i in range(n):
            mass     = self._min_mass[i]
            worst_ff = 0.0
            for alt in self.env.altitudes:
                if not self._is_feasible(mass, alt, 0.0):
                    continue
                tas_kts = aero.mach2tas(self.mach, alt * ft) / kts
                vs_up   = (self.env.alt_step_ft / dt_s) * 60.0
                ff_up   = max(0.0, self.fuel_model.enroute(mass=mass, tas=tas_kts, alt=alt, vs=vs_up))
                if ff_up > worst_ff:
                    worst_ff = ff_up
            max_ff[i] = worst_ff
            if i < n - 1:
                self._min_mass[i + 1] = max(0.0, self._min_mass[i] - max_ff[i] * dt_s)

        # Fuel suffix sum: re-evaluate min ff at min-mass lower bound.
        self._min_fuel_from = np.zeros(n + 1)
        for i in range(n - 2, -1, -1):
            mass_lb = self._min_mass[i]
            best_ff = float('inf')
            for alt in self.env.altitudes:
                if not self._is_feasible(mass_lb, alt, 0.0):
                    continue
                tas_kts = aero.mach2tas(self.mach, alt * ft) / kts
                ff = max(0.0, self.fuel_model.enroute(mass=mass_lb, tas=tas_kts, alt=alt, vs=0))
                if ff < best_ff:
                    best_ff = ff
                next_alt = alt - self.env.alt_step_ft
                if next_alt >= self.env.min_alt_ft:
                    vs  = (-self.env.alt_step_ft / dt_s) * 60.0
                    ffd = max(0.0, self.fuel_model.enroute(mass=mass_lb, tas=tas_kts, alt=alt, vs=vs))
                    if ffd < best_ff:
                        best_ff = ffd
            if best_ff == float('inf'):
                best_ff = 0.0
            self._min_fuel_from[i] = self.fuel_weight * best_ff * dt_s + self._min_fuel_from[i + 1]

        # Contrail suffix sum.
        min_contrail = np.min(self.env.cost_map, axis=1)
        self._min_contrail_from = np.zeros(n + 1)
        for i in range(n - 2, -1, -1):
            self._min_contrail_from[i] = min_contrail[i] * dt_s + self._min_contrail_from[i + 1]

        self.suffix_heuristic_s = time.perf_counter() - _t0

    # ── Edge cost precomputation (vectorized, used by DP and Pareto) ──────────

    def _precompute_edge_costs(self):
        """
        Vectorized edge cost tables: entry [i, j, s] = cost of going from
        (waypoint i, alt j) to (waypoint i+1, alt j+dk) where dk = s - max_steps.
        Infeasible / out-of-bounds entries are np.inf.

        Three arrays are filled in one pass:
          _fuel_edge_array      unweighted fuel cost      (for Pareto)
          _contrail_edge_array  unweighted contrail cost  (for Pareto)
          _edge_cost_array      fuel_weight·fuel + contrail_weight·contrail
        """
        n      = len(self.env.distances)
        n_alts = len(self.env.altitudes)
        alts   = self.env.altitudes

        dt0_s        = float(self.env.ts[1] - self.env.ts[0])
        self._max_steps = int(self.max_climb_rate_fpm * (dt0_s / 60.0) / self.env.alt_step_ft)
        max_steps    = self._max_steps
        span         = 2 * max_steps + 1

        self._edge_cost_array     = np.full((n - 1, n_alts, span), np.inf)
        self._fuel_edge_array     = np.full((n - 1, n_alts, span), np.inf)
        self._contrail_edge_array = np.full((n - 1, n_alts, span), np.inf)

        # Static geometry (same for every waypoint).
        j_from_2d   = np.arange(n_alts)[:, None]                      # (n_alts, 1)
        dk_arr      = np.arange(-max_steps, max_steps + 1)[None, :]   # (1, span)
        j_to_2d     = j_from_2d + dk_arr                              # (n_alts, span)
        bounds_ok   = (j_to_2d >= 0) & (j_to_2d < n_alts)
        j_to_safe   = np.clip(j_to_2d, 0, n_alts - 1)

        alt_from_2d = alts[j_from_2d]                                 # (n_alts, 1)
        alt_to_2d   = alts[j_to_safe]                                 # (n_alts, span)
        dalt_2d     = alt_to_2d - alt_from_2d
        avg_alt_2d  = (alt_from_2d + alt_to_2d) / 2.0

        alt_idx_2d  = np.clip(
            np.round((avg_alt_2d - self.env.min_alt_ft) / self.env.alt_step_ft),
            0, n_alts - 1,
        ).astype(int)

        for i in range(n - 1):
            dt_s  = float(self.env.ts[i + 1] - self.env.ts[i])
            dt_min = dt_s / 60.0
            mass  = self._mass_at_dist[i]
            avg_ts = (self.env.ts[i] + self.env.ts[i + 1]) / 2.0
            ts_idx = int(np.clip(np.searchsorted(self.env.ts, avg_ts), 0, len(self.env.ts) - 1))

            vs_fpm_2d   = dalt_2d * (60.0 / dt_s)
            rate_fpm_2d = np.abs(dalt_2d) / dt_min
            roc_ok = (
                ~((dalt_2d > 0) & (rate_fpm_2d > self.max_climb_rate_fpm)) &
                ~((dalt_2d < 0) & (rate_fpm_2d > self.max_descent_rate_fpm))
            )
            candidate = bounds_ok & roc_ok
            if not candidate.any():
                continue

            flat_avg_alt = avg_alt_2d[candidate]
            flat_vs      = vs_fpm_2d[candidate]
            v            = aero.mach2tas(self.mach, flat_avg_alt * ft)
            rho          = aero.density(flat_avg_alt * ft)
            q            = 0.5 * rho * v ** 2
            tas_kts      = v / kts

            thrust_max = self.thrust_model.cruise(tas=tas_kts, alt=flat_avg_alt)
            drag_val   = self.drag_model.clean(mass=mass, tas=tas_kts, alt=flat_avg_alt, vs=flat_vs)
            feas       = thrust_max * 0.95 >= drag_val
            cd_max     = thrust_max * 0.9 / (q * self.S + 1e-10)
            cl2        = (cd_max - self.cd0) / self.ck
            feas      &= cl2 > 1e-10
            feas      &= np.sqrt(np.maximum(cl2, 0.0)) * q * self.S * 0.80 >= mass * g0

            feas_2d = np.zeros((n_alts, span), dtype=bool)
            feas_2d[candidate] = feas
            if not feas_2d.any():
                continue

            flat_avg_alt_f = avg_alt_2d[feas_2d]
            flat_vs_f      = vs_fpm_2d[feas_2d]
            tas_f          = aero.mach2tas(self.mach, flat_avg_alt_f * ft) / kts
            ff             = self.fuel_model.enroute(mass=mass, tas=tas_f, alt=flat_avg_alt_f, vs=flat_vs_f)
            fuel_unw       = np.maximum(0.0, ff) * dt_s
            contrail_unw   = self.env.cost_map[ts_idx, alt_idx_2d[feas_2d]] * dt_s
            self._fuel_edge_array[i][feas_2d]     = fuel_unw
            self._contrail_edge_array[i][feas_2d] = contrail_unw
            self._edge_cost_array[i][feas_2d]     = (
                self.fuel_weight * fuel_unw + self.contrail_weight * contrail_unw
            )

    # ── Graph construction (same anti-oscillation edges as a_star_no_osc) ─────

    def _build_graph(self):
        env      = self.env
        dt0_s    = float(env.ts[1] - env.ts[0])
        dt0_min  = dt0_s / 60.0
        max_steps = int(self.max_climb_rate_fpm * dt0_min / env.alt_step_ft)
        alt_step  = int(env.alt_step_ft)

        reachable_pairs = [
            (alt, prev_alt)
            for alt in env.altitudes
            for prev_alt in env.altitudes
            if abs(float(alt) - float(prev_alt)) <= max_steps * alt_step
        ]

        for dist, ts in zip(env.distances, env.ts):
            for alt, prev_alt in reachable_pairs:
                self.graph.add_node((dist, alt, prev_alt, ts))

        edges_added = 0
        for i, (dist, ts) in enumerate(zip(env.distances[:-1], env.ts[:-1])):
            next_dist = env.distances[i + 1]
            next_ts   = env.ts[i + 1]
            dt_s      = next_ts - ts
            dt_min    = dt_s / 60.0
            mass      = self._mass_at_dist[i]

            for alt, prev_alt in reachable_pairs:
                alt      = float(alt)
                prev_alt = float(prev_alt)

                # Anti-oscillation: constrain allowed next altitudes.
                if alt > prev_alt:
                    allowed_next = [a for a in env.altitudes if a >= alt]
                elif alt < prev_alt:
                    allowed_next = [a for a in env.altitudes if a <= alt]
                else:
                    allowed_next = list(env.altitudes)

                for next_alt in allowed_next:
                    next_alt = float(next_alt)
                    dalt     = next_alt - alt
                    if dt_min > 0:
                        rate_fpm = abs(dalt) / dt_min
                        if dalt > 0 and rate_fpm > self.max_climb_rate_fpm:
                            continue
                        if dalt < 0 and rate_fpm > self.max_descent_rate_fpm:
                            continue
                    if not (env.min_alt_ft <= next_alt <= env.max_alt_ft):
                        continue
                    avg_alt = (alt + next_alt) / 2.0
                    vs_fpm  = (dalt / dt_s) * 60.0 if dt_s > 0 else 0.0
                    if not self._is_feasible(mass, avg_alt, vs_fpm):
                        continue
                    cost = self._edge_cost(dist, alt, next_dist, next_alt, ts, next_ts)
                    self.graph.add_edge(
                        (dist, alt, prev_alt, ts),
                        (next_dist, next_alt, alt, next_ts),
                        weight=cost,
                    )
                    edges_added += 1

        start_dist = float(env.distances[0])
        start_ts   = float(env.ts[0])
        end_dist   = float(env.distances[-1])
        end_ts     = float(env.ts[-1])

        # SOURCE: all starting altitudes with prev_alt = alt (level start → full freedom).
        for alt in env.altitudes:
            self.graph.add_edge("SOURCE", (start_dist, float(alt), float(alt), start_ts), weight=0.0)

        # SINK: connect all final-waypoint states.
        for alt, prev_alt in reachable_pairs:
            self.graph.add_edge((end_dist, float(alt), float(prev_alt), end_ts), "SINK", weight=0.0)

        print(f"  graph edges_added={edges_added}")

    def _ensure_graph(self):
        """Build the 4-tuple graph on first use by a graph-based method.

        Records the build cost in ``self.graph_build_s``. Because the build is
        lazy, whichever solver runs FIRST pays for it inside its own timing
        unless the caller hoists this call out and times it separately — see
        ``__main__`` below and ``Benchmarking._run_all_solvers``.
        """
        if self.graph is None:
            _t0 = time.perf_counter()
            self.graph = nx.DiGraph()
            self._build_graph()
            self.graph_build_s = time.perf_counter() - _t0
            print(f"  Graph ready: {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges "
                  f"({self.graph_build_s:.2f}s)")

    # ── Heuristics ─────────────────────────────────────────────────────────────

    def _heuristic_suffix_sum(self, node, goal):
        if node in ("SOURCE", "SINK"):
            return 0.0
        dist, alt, prev_alt, ts = node
        i = int(np.clip(np.searchsorted(self.env.distances, dist), 0, len(self.env.distances) - 1))
        if i >= len(self.env.distances) - 1:
            return 0.0
        return self._min_fuel_from[i] + self.contrail_weight * self._min_contrail_from[i]

    def compute_dp_heuristic(self):
        """
        Vectorized backward DP over (waypoint, alt, direction) state space.
        _dp_cost_to_go[i, j, d] = exact min cost-to-go from waypoint i,
        altitude altitudes[j], arriving with direction d (0=desc, 1=level, 2=climb).
        This is a perfect heuristic for the 4-tuple graph: A* expands only
        nodes on optimal paths.
        """
        _t0       = time.perf_counter()
        n         = len(self.env.distances)
        n_alts    = len(self.env.altitudes)
        max_steps = self._max_steps
        span      = 2 * max_steps + 1

        dk_arr     = np.arange(-max_steps, max_steps + 1)                      # (span,)
        d_next_arr = np.where(dk_arr > 0, 2, np.where(dk_arr < 0, 0, 1))      # (span,)

        j_arr    = np.arange(n_alts)
        j_to_2d  = j_arr[:, None] + dk_arr[None, :]                           # (n_alts, span)
        bounds_ok = (j_to_2d >= 0) & (j_to_2d < n_alts)
        j_to_safe = np.clip(j_to_2d, 0, n_alts - 1)

        # dir_masks[d] is a (span,) boolean: which dk values are allowed from direction d.
        dir_masks = np.stack([
            dk_arr <= 0,                   # d=0 descent: only level or continue descending
            np.ones(span, dtype=bool),     # d=1 level:   full freedom
            dk_arr >= 0,                   # d=2 climb:   only level or continue climbing
        ])                                 # (3, span)

        dp = np.full((n, n_alts, 3), np.inf)
        dp[n - 1, :, :] = 0.0

        for i in range(n - 2, -1, -1):
            # dp_next_2d[j, s] = dp[i+1, j_to_safe[j,s], d_next_arr[s]]
            dp_next_2d = dp[i + 1, j_to_safe, d_next_arr]   # (n_alts, span)
            ec = self._edge_cost_array[i]                    # (n_alts, span)
            for d_idx in range(3):
                valid = bounds_ok & dir_masks[d_idx][None, :]
                total = np.where(valid, ec + dp_next_2d, np.inf)
                dp[i, :, d_idx] = np.min(total, axis=1)

        self._dp_cost_to_go = dp
        self.dp_heuristic_s = time.perf_counter() - _t0
        finite = dp[np.isfinite(dp)]
        print(f"\n  DP heuristic table: {n}×{n_alts}×3  ({self.dp_heuristic_s:.2f}s)")
        if finite.size:
            print(f"  dp[0] range: [{finite.min():.4f}, {finite.max():.4f}]")

    def _heuristic_dp(self, node, goal):
        if node in ("SOURCE", "SINK"):
            return 0.0
        dist, alt, prev_alt, ts = node
        i = int(np.clip(np.searchsorted(self.env.distances, dist), 0, len(self.env.distances) - 1))
        if i >= len(self.env.distances) - 1:
            return 0.0
        j = int(np.argmin(np.abs(self.env.altitudes - alt)))
        if   alt > prev_alt: d_idx = 2
        elif alt < prev_alt: d_idx = 0
        else:                d_idx = 1
        val = self._dp_cost_to_go[i, j, d_idx]
        return 0.0 if not np.isfinite(val) else float(val)

    # ── Utilities ──────────────────────────────────────────────────────────────

    def path_to_arrays(self, path: list) -> tuple:
        """Accept 4-tuple (dist, alt, prev_alt, ts) or 3-tuple (dist, alt, ts)."""
        if path and len(path[0]) == 4:
            return (
                np.array([p[1] for p in path]),
                np.array([p[0] for p in path]),
                np.array([p[3] for p in path]),
            )
        return (
            np.array([p[1] for p in path]),
            np.array([p[0] for p in path]),
            np.array([p[2] for p in path]),
        )

    def graph_cost(self, path: list) -> float:
        """Sum of edge costs along a 3- or 4-tuple path."""
        p3 = [(p[0], p[1], p[3]) for p in path] if path and len(path[0]) == 4 else path
        total = 0.0
        for i in range(len(p3) - 1):
            d1, a1, t1 = p3[i]
            d2, a2, t2 = p3[i + 1]
            total += self._edge_cost(d1, a1, d2, a2, t1, t2)
        return total

    def path_objectives(self, path: list) -> tuple[float, float]:
        """
        Unweighted ``(fuel, contrail)`` of a path, summed from ``_fuel_edge_array``
        / ``_contrail_edge_array``.

        These are the very arrays :meth:`DpMuoOptimizer.build_pareto_front`
        accumulates into its labels, so a scalarised solve and a label-setting solve
        can be compared in ONE objective space. Do not substitute
        :func:`evaluate_path` here: it re-integrates with a propagating mass while
        the edge tables use the fixed ``_mass_at_dist`` profile, so the two disagree
        slightly — enough to blur which points are on the convex hull.

        Requires the edge tables (``optimizer`` in dp / astar_dp / pareto).
        """
        if not hasattr(self, "_fuel_edge_array"):
            raise RuntimeError(
                "path_objectives needs the edge tables — construct with "
                "optimizer='dp' (or 'pareto'/'astar_dp')."
            )
        alt, _, _ = self.path_to_arrays(path)
        n_alts = len(self.env.altitudes)
        j = np.clip(
            np.round((alt - self.env.min_alt_ft) / self.env.alt_step_ft), 0, n_alts - 1
        ).astype(int)
        fuel = contrail = 0.0
        for i in range(len(j) - 1):
            s = int(j[i + 1] - j[i]) + self._max_steps
            fuel     += float(self._fuel_edge_array[i, j[i], s])
            contrail += float(self._contrail_edge_array[i, j[i], s])
        return fuel, contrail

    def _to_result(self, path: list, cost: Optional[float] = None) -> OptResult:
        """
        Normalise a solved node-list path into an :class:`OptResult` (2D → vertical
        profile only; ``lat``/``lon`` stay None). ``cost`` is the native graph edge
        sum; ``fuel_kg``/``contrail`` are left None for ``evaluate()`` to fill.
        """
        alt, dist, ts = self.path_to_arrays(path)
        if cost is None:
            cost = self.graph_cost(path)
        return OptResult(
            alt_ft=alt, dist_km=dist, ts_s=ts, cost=cost,
            meta={"optimizer": self.name, "algorithm": self._optimizer},
        )


# ── Single-objective solvers (one algorithm per class) ───────────────────────

class DagOptimizer(_model_2D_base):
    """DAG shortest path via topological relaxation — O(V+E)."""

    _DEFAULT_OPTIMIZER = "dag"

    def optimize_dag(self) -> dict | None:
        self._ensure_graph()
        try:
            topo = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            print("DAG: graph contains a cycle.")
            return None
        dist_to = {node: float('inf') for node in self.graph.nodes}
        pred    = {node: None         for node in self.graph.nodes}
        dist_to["SOURCE"] = 0.0
        for u in topo:
            if not np.isfinite(dist_to[u]):
                continue
            for v, data in self.graph[u].items():
                relaxed = dist_to[u] + data.get("weight", 0.0)
                if relaxed < dist_to[v]:
                    dist_to[v] = relaxed
                    pred[v] = u
        if not np.isfinite(dist_to["SINK"]):
            print("DAG: no path found.")
            return None
        path, node = [], "SINK"
        while node is not None:
            path.append(node)
            node = pred[node]
        path.reverse()
        return {"path": path[1:-1], "cost": dist_to["SINK"]}

    optimize = optimize_dag


class DijkstraOptimizer(_model_2D_base):
    """Dijkstra on the 4-tuple graph."""

    _DEFAULT_OPTIMIZER = "dp_muo"

    def optimize_dijkstra(self) -> dict | None:
        self._ensure_graph()
        try:
            path = dijkstra_path(self.graph, "SOURCE", "SINK", weight="weight")
            return {"path": path[1:-1]}
        except nx.NetworkXNoPath:
            print("Dijkstra: no path found.")
            return None

    optimize = optimize_dijkstra


class AstarSuffixSumOptimizer(_model_2D_base):
    """A* with admissible suffix-sum lower-bound heuristic."""

    _DEFAULT_OPTIMIZER = "astar_suffix_sum"

    def optimize_astar_suffix_sum(self) -> dict | None:
        self._ensure_graph()
        try:
            path = nx.astar_path(self.graph, "SOURCE", "SINK",
                                 heuristic=self._heuristic_suffix_sum, weight="weight")
            return {"path": path[1:-1]}
        except nx.NetworkXNoPath:
            print("A* suffix_sum: no path found.")
            return None

    optimize = optimize_astar_suffix_sum


class AstarDpOptimizer(_model_2D_base):
    """A* with exact backward-DP heuristic (call compute_dp_heuristic() first)."""

    _DEFAULT_OPTIMIZER = "astar_dp"

    def optimize_astar_dp(self) -> dict | None:
        self._ensure_graph()
        if not hasattr(self, "_dp_cost_to_go"):
            raise RuntimeError("Call compute_dp_heuristic() before optimize_astar_dp().")
        try:
            path = nx.astar_path(self.graph, "SOURCE", "SINK",
                                 heuristic=self._heuristic_dp, weight="weight")
            return {"path": path[1:-1]}
        except nx.NetworkXNoPath:
            print("A* dp: no path found.")
            return None

    optimize = optimize_astar_dp


class DpOptimizer(_model_2D_base):
    """Vectorized DP over (waypoint, alt, direction) — no graph needed."""

    _DEFAULT_OPTIMIZER = "dp"

    def optimize_dp(self) -> dict | None:
        """
        Uses the precomputed edge cost array; the no-oscillation constraint is
        enforced via direction masks on the allowed dk values at each step.
        """
        n         = len(self.env.distances)
        n_alts    = len(self.env.altitudes)
        max_steps = self._max_steps
        span      = 2 * max_steps + 1

        dk_arr     = np.arange(-max_steps, max_steps + 1)
        d_next_arr = np.where(dk_arr > 0, 2, np.where(dk_arr < 0, 0, 1))

        j_arr    = np.arange(n_alts)
        j_to_2d  = j_arr[:, None] + dk_arr[None, :]
        bounds_ok = (j_to_2d >= 0) & (j_to_2d < n_alts)
        j_to_safe = np.clip(j_to_2d, 0, n_alts - 1)

        dir_masks = np.stack([
            dk_arr <= 0,
            np.ones(span, dtype=bool),
            dk_arr >= 0,
        ])

        dp       = np.full((n, n_alts, 3), np.inf)
        policy_j = np.full((n, n_alts, 3), -1, dtype=int)
        policy_d = np.full((n, n_alts, 3), -1, dtype=int)
        dp[n - 1, :, :] = 0.0

        for i in range(n - 2, -1, -1):
            dp_next_2d = dp[i + 1, j_to_safe, d_next_arr]   # (n_alts, span)
            ec = self._edge_cost_array[i]
            for d_idx in range(3):
                valid  = bounds_ok & dir_masks[d_idx][None, :]
                total  = np.where(valid, ec + dp_next_2d, np.inf)
                best_s = np.argmin(total, axis=1)
                dp[i, :, d_idx]       = total[j_arr, best_s]
                policy_j[i, :, d_idx] = j_to_safe[j_arr, best_s]
                policy_d[i, :, d_idx] = d_next_arr[best_s]

        # Start with d=1 (level): prev_alt == alt at SOURCE for all nodes.
        best_j0 = int(np.argmin(dp[0, :, 1]))
        if not np.isfinite(dp[0, best_j0, 1]):
            print("DP NoOsc: no feasible path found.")
            return None

        path, j, d = [], best_j0, 1
        for i in range(n):
            path.append((float(self.env.distances[i]),
                         float(self.env.altitudes[j]),
                         float(self.env.ts[i])))
            if i < n - 1:
                j_next = int(policy_j[i, j, d])
                d_next = int(policy_d[i, j, d])
                if j_next < 0:
                    print(f"DP NoOsc: broken policy at step {i}.")
                    return None
                j, d = j_next, d_next

        return {"path": path, "cost": float(dp[0, best_j0, 1])}

    optimize = optimize_dp

    # ── DP-based Pareto sweep (migrated from graph_based/pareto.py::DPPareto) ──

    def _reweight_edge_costs(self, coef: float) -> None:
        """
        Re-scalarise the (already precomputed) per-edge fuel/contrail costs for a
        new trade-off weight ``coef`` (contrail_weight = coef, fuel_weight =
        1 - coef), rewriting the combined ``_edge_cost_array`` that
        ``optimize_dp`` consumes.

        Cheap on purpose: the physics — mass profile and the *unweighted*
        ``_fuel_edge_array`` / ``_contrail_edge_array`` — is coef-independent and
        computed once at construction, so a whole Pareto sweep re-runs only the
        vectorised DP, never openap. Infeasible edges (inf in the fuel array) stay
        inf; contrail is finite exactly where fuel is, so no 0·inf = nan arises.
        """
        self.fuel_weight = 1.0 - coef
        self.contrail_weight = coef
        finite = np.isfinite(self._fuel_edge_array)
        combined = np.full_like(self._fuel_edge_array, np.inf)
        combined[finite] = (
            self.fuel_weight * self._fuel_edge_array[finite]
            + self.contrail_weight * self._contrail_edge_array[finite]
        )
        self._edge_cost_array = combined

    def _solve_dp_point(self, coef: float) -> Optional[OptResult]:
        """
        Solve the DP at scalarisation ``coef`` and return an evaluated
        :class:`OptResult` (``fuel_kg`` / ``contrail`` filled by the shared
        post-hoc physics in :func:`evaluate_path`), or ``None`` if infeasible.
        """
        t0 = time.perf_counter()
        self._reweight_edge_costs(coef)
        result = self.optimize_dp()
        elapsed = time.perf_counter() - t0
        if result is None:
            self.dp_solve_log.append({"coef": coef, "time_s": elapsed,
                                      "posthoc_s": 0.0, "feasible": False})
            return None

        res = self._to_result(result["path"], cost=result.get("cost"))
        _tp = time.perf_counter()
        metrics = evaluate_path(
            res.alt_ft, res.dist_km, res.ts_s,
            self.mach, self.mass_init_kg, self.env, self.actype,
        )
        posthoc = time.perf_counter() - _tp
        self.posthoc_s += posthoc
        self.posthoc_calls += 1
        self.dp_solve_log.append({"coef": coef, "time_s": elapsed,
                                  "posthoc_s": posthoc, "feasible": True})
        res.fuel_kg = metrics["total_fuel_kg"]
        res.contrail = metrics["total_contrail"]
        res.ts_s = metrics["ts_s"]          # flown at its own altitudes (flown_ts)
        res.meta.update({
            "optimizer": "DP",
            "coef": coef,
            "time_s": elapsed,
            "total_alt_changes_ft": metrics["total_alt_changes_ft"],
            # The contrail as the DP's own cost map scores this path (what it minimised);
            # res.contrail is the path's integral on the interpolant.
            "contrail_table": metrics["total_contrail_table"],
        })
        return res

    def pareto_dp(
        self,
        grid: Optional[list[float]] = None,
        method: str = "weighted_sum",
    ) -> list[OptResult]:
        """
        Fuel-vs-contrail Pareto front built by *sweeping* the DP.

        Unlike :class:`DpMuoOptimizer` (one label-setting solve returns
        the whole front), this scalarises the two objectives and re-solves the
        fast DP once per point:

          - ``weighted_sum``: coef in [0, 1] blends fuel/contrail directly.

        The only method. The epsilon-constraint was removed on 2026-10-19: the DP has
        no native constraint, so each point cost a bisection on coef (~12 DP solves)
        and reached only points of the weighted-sum front. Pickles written before
        then still carry ``epsilon_constraint`` records.

        Returns the accepted points as :class:`OptResult`s (also stored on
        ``self.pareto_results``). Note this **overwrites** ``fuel_weight`` /
        ``contrail_weight`` as it sweeps.
        """
        if method == "weighted_sum":
            if grid is None:
                grid = [0.0, 0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]
            self.pareto_results = self._pareto_weighted_sum(grid)
        else:
            raise ValueError(f"Unknown method '{method}'. The DP sweeps 'weighted_sum' only.")
        return self.pareto_results

    def _pareto_weighted_sum(self, coef_grid: list[float]) -> list[OptResult]:
        print(f"\n[pareto_dp] weighted-sum sweep — {len(coef_grid)} coef values")
        out: list[OptResult] = []
        for coef in coef_grid:
            res = self._solve_dp_point(coef)
            if res is None:
                print(f"  [coef={coef:.3f}] no path — skipping.")
                continue
            res.meta["method"] = "weighted_sum"
            print(
                f"  [coef={coef:.3f}] fuel={res.fuel_kg:.1f} kg  "
                f"contrail={res.contrail:.4f}  time={res.meta['time_s']:.2f} s"
            )
            out.append(res)
        return out


# ── Multi-objective solver (Pareto front via label setting) ──────────────────

class DpMuoOptimizer(_model_2D_base):
    """
    Multi-objective (fuel vs contrail) Pareto front on the no-oscillation graph.

    A label-setting sweep over the layered DAG: labels are propagated waypoint by
    waypoint (step 0 → n-1), so every predecessor layer is finalised before its
    successors are expanded.  Each state is (step, alt_index, direction); it holds
    a *non-dominated set* of ``Label``s carrying the running (fuel, contrail) cost
    and a parent pointer for path reconstruction.

    Fuel and contrail are kept as separate objectives (from ``_fuel_edge_array`` /
    ``_contrail_edge_array``), so ``fuel_weight`` / ``contrail_weight`` are ignored
    here — the whole trade-off curve is returned at once.
    """

    _DEFAULT_OPTIMIZER = "pareto"

    def build_pareto_front(self) -> dict | None:
        env       = self.env
        n         = len(env.distances)
        n_alts    = len(env.altitudes)
        max_steps = self._max_steps
        span      = 2 * max_steps + 1

        dk_arr   = np.arange(-max_steps, max_steps + 1)
        d_of_dk  = np.where(dk_arr > 0, 2, np.where(dk_arr < 0, 0, 1))   # arrival direction per dk
        # Allowed dk per current direction — the anti-oscillation constraint
        # (NOT encoded in the edge arrays, which only enforce rate/bounds).
        dir_masks = [
            dk_arr <= 0,                    # d=0 descent
            np.ones(span, dtype=bool),      # d=1 level
            dk_arr >= 0,                    # d=2 climb
        ]

        fuel_arr = self._fuel_edge_array        # (n-1, n_alts, span)
        cont_arr = self._contrail_edge_array

        # layers[i][(alt_idx, direction)] -> non-dominated list[Label]
        layers = [dict() for _ in range(n)]
        # Seed step 0: level start (d=1) at every altitude, zero cost.
        for j in range(n_alts):
            layers[0][(j, 1)] = [Label(0.0, 0.0, (0, j), None)]

        # Work counters. Label setting does not iterate, so "number of iterations"
        # has no meaning here; the comparable quantity against IPOPT iterations or
        # DP solves is how many labels were created and how many survived pruning.
        n_expanded = 0

        for i in range(n - 1):
            fuel_i = fuel_arr[i]
            cont_i = cont_arr[i]
            nxt    = layers[i + 1]
            for (j, d), bucket in layers[i].items():
                if not bucket:
                    continue
                for s in range(span):
                    if not dir_masks[d][s]:
                        continue
                    fc = fuel_i[j, s]
                    if not np.isfinite(fc):
                        continue
                    j2 = j + dk_arr[s]
                    if j2 < 0 or j2 >= n_alts:
                        continue
                    d2 = int(d_of_dk[s])
                    cc = float(cont_i[j, s])
                    dst = nxt.setdefault((j2, d2), [])
                    for lbl in bucket:
                        n_expanded += 1
                        add_if_nondominated(
                            Label(lbl.fuel + float(fc), lbl.contrail + cc, (i + 1, j2), lbl),
                            dst,
                        )

        # Merge the final layer across all arrival directions into one front.
        front: list[Label] = []
        for bucket in layers[n - 1].values():
            for lbl in bucket:
                add_if_nondominated(lbl, front)

        if not front:
            print("DP_MuO: no feasible path found.")
            return None

        self.n_labels_expanded = n_expanded
        self.n_labels_kept = sum(len(b) for layer in layers for b in layer.values())

        front.sort(key=lambda l: l.fuel)   # ascending fuel → descending contrail
        paths = [self._label_to_path(l) for l in front]
        print(f"\n  Pareto front: {len(front)} non-dominated trajectories")
        return {
            "front":  [{"fuel": l.fuel, "contrail": l.contrail} for l in front],
            "labels": front,
            "paths":  paths,
        }

    optimize = build_pareto_front

    def _label_to_path(self, label: Label) -> list:
        """Walk the parent chain into a 3-tuple (dist, alt, ts) path."""
        chain = []
        node = label
        while node is not None:
            chain.append(node.node)   # (step_index, alt_index)
            node = node.parent
        chain.reverse()
        env = self.env
        return [
            (float(env.distances[i]), float(env.altitudes[j]), float(env.ts[i]))
            for (i, j) in chain
        ]


# ── Facade: every solver on one instance (shared graph) ──────────────────────

class model_2D_graph_based(
    DagOptimizer,
    DijkstraOptimizer,
    AstarSuffixSumOptimizer,
    AstarDpOptimizer,
    DpOptimizer,
    DpMuoOptimizer,
):
    """
    Backward-compatible facade exposing every optimizer method on a single
    instance, so one built graph is shared across all graph-based methods
    (used by the comparison and benchmarking drivers).
    """

    _DEFAULT_OPTIMIZER = "dp"

    def optimize(self, *args, **kwargs):
        raise NotImplementedError(
            "model_2D_graph_based is a facade — call a specific method: "
            "optimize_dp(), optimize_dijkstra(), optimize_dag(), "
            "optimize_astar_suffix_sum(), optimize_astar_dp(), or build_pareto_front()."
        )


# Backward-compatible alias: the facade was renamed model_2D_graph_based, but
# several importers (pareto.py, driver_comparison.py, benchmarking.py) still use
# the old name.
TrajectoryOptimizerNoOscAll = model_2D_graph_based

if __name__ == "__main__":
    ALT_STEP_FT = 1000.0
    MIN_ALT_FT  = 31000.0
    MAX_ALT_FT  = 40000.0
    MAX_ROC     = 500.0
    CONTRAIL_WEIGHT = 0.1      # 0 = fuel-only; raise to bend around contrails
    sigma       = 2
    # Cruise-entry mass fraction of MTOW (NOT takeoff): ~0.85·MTOW is way too
    # heavy for cruise and makes every altitude fail the lift feasibility check.
    # A medium-haul widebody enters cruise around 0.6–0.7·MTOW.
    m0=0.8
    flight_name = "Berlin-Madrid"
    begin = "2023-01-05 09:48"
    end = "2023-01-05 11:00"
    callsign = "RYR880W"
    # flight_name="Ireland-Libya"
    # begin="2022-02-20 00:00"
    # end="2022-02-20 03:29"
    # callsign = "SVA020"
    # 1) Flight setup: load the flight, build the contrail cost grid + interpolant,
    #    and take mach / cruise altitude / actype straight from the real flight.
    TMP = Path(tempfile.gettempdir())
    interpolant_cache = os.environ.get(
        "INTERPOLANT_CACHE",
        str(TMP / f"opentop_interp_{flight_name}_sigma{sigma}.casadi"),
    )
    fd = FlightData(start=begin, stop=end, callsign=callsign)
    fd.filter_cruise()                                       # keep only the cruise segment
    fd.interpolate_meteo(time_buffer_hours=2.0, wind=True)   # meteo + wind (TAS correction)
    meteo = agg_conditions(fd.meteo)
    df_cost = prepare_grid(meteo, sigma, t_reference=fd.start)
    interpolant = top.tools.cached_interpolant_from_dataframe(
        df_cost, interpolant_cache, shape="bspline"
    )

    # mach / average cruise altitude / actype precalculated from the real flight.
    avg_alt_m     = float(fd.flight.data.altitude.mean()) * ft
    cruise_alt_ft = avg_alt_m / ft
    mach          = fd.compute_cruise_mach()
    actype        = fd.aircraft_type                    # from traffic's aircraft DB (callsign path)
    print(f"actype={actype} | mach={mach:.3f} | cruise_alt={cruise_alt_ft:.0f} ft")
    route, total_distance_km = real_flight_route(
        fd.flight, mach=mach, avg_alt_m=avg_alt_m, roc=MAX_ROC, alt_step_ft=ALT_STEP_FT
    )

    env = ContrailEnvironmentFromGrid(
        interpolant=interpolant, grid_data=df_cost,
        route=(route, total_distance_km),
        alt_step_ft=ALT_STEP_FT,
        min_alt_ft=MIN_ALT_FT,
        max_alt_ft=MAX_ALT_FT,
    )
    aircraft = ac_props(actype)

    opt = model_2D_graph_based(
    environment = env,
    cruise_alt_ft = cruise_alt_ft, mach = mach, mass_init_kg = m0*aircraft["mtow"], actype = actype,
    max_climb_rate_fpm = MAX_ROC, max_descent_rate_fpm = MAX_ROC,
    fuel_weight = 1.0 - CONTRAIL_WEIGHT, contrail_weight = CONTRAIL_WEIGHT,
    )
    opt._compute_heuristic_tables()  # needed by A* suffix_sum
    opt.compute_dp_heuristic()  # needed by A* dp-heuristic

    # 2) Build the graph ONCE, up front, timed separately. The graph-based
    #    solvers (DAG/Dijkstra/A*) share it; DP doesn't use it. Building it here
    #    keeps the per-edge openap fuel-flow cost out of the first solver's timing.
    t0 = time.time()
    opt._ensure_graph()
    graph_build_s = time.time() - t0
    print(f"\ngraph build: {graph_build_s:6.2f}s  "
          f"({opt.graph.number_of_nodes()} nodes, {opt.graph.number_of_edges()} edges)")

    # 3) Run every single-objective solver on the one shared graph/tables.
    SOLVERS = [
        # ("A* suffix_sum",   "optimize_astar_suffix_sum"),
        # ("A* dp-heuristic", "optimize_astar_dp"),
        ("DP standalone",   "optimize_dp"),
        # ("Dijkstra",        "optimize_dijkstra"),
        ("DAG shortest",    "optimize_dag"),
    ]

    paths = {}
    print("running solvers (cost via graph_cost → same basis for all):")
    for label, method in SOLVERS:
        t0  = time.time()
        res = getattr(opt, method)()
        dt  = time.time() - t0
        if res is None:
            print(f"  [{label:16s}] no path found — skipping.")
            continue
        # Score every path the same way (graph edge-cost sum) so DP and the graph
        # solvers are directly comparable, and no solver reports 'n/a'.
        cost = opt.graph_cost(res["path"])
        print(f"  [{label:16s}] cost={cost:8.2f}  solve={dt:6.2f}s  pts={len(res['path'])}")
        paths[label] = res["path"]

    # 4) Real ("Baseline") altitude profile resampled onto the env distances.
    paths["Baseline"] = (env.distances, real_flight_baseline(fd.flight, env.distances))

    print("\nNote: the five single-objective solvers are exact optimizers of the same "
          "problem, so their optimal paths coincide (a cross-validation). "
          "Only 'Baseline' (the real flight) differs.")

    # 5) Overlay every trajectory on the contrail cost map (dict form avoids the
    #    isinstance(ContrailEnvironmentFromGrid) branch in plot_cost_map).
    plot_cost_map(
        {"env_distances": env.distances,
         "env_altitudes": env.altitudes,
         "env_cost_map":  env.cost_map},
        paths=paths,
    )
    out = Path("all_algorithms_cost_map.png")
    # plt.savefig(out, dpi=150)
    # print(f"saved {out.resolve()}")
    # plt.show()

    # 6) Multi-objective example: DpMuoOptimizer.
    #    Unlike the single-objective solvers above (which need fuel_weight /
    #    contrail_weight), the Pareto solver keeps fuel and contrail as two
    #    separate objectives and returns the WHOLE trade-off curve at once, so the
    #    weights are ignored. It reuses the same env/actype/mach as everything
    #    above; instantiating it (_DEFAULT_OPTIMIZER="pareto") precomputes the
    #    fuel/contrail edge arrays it needs — no graph required.
    pareto_opt = DpMuoOptimizer(
        environment=env,
        cruise_alt_ft=cruise_alt_ft, mach=mach,
        mass_init_kg=m0 * aircraft["mtow"], actype=actype,
        max_climb_rate_fpm=MAX_ROC, max_descent_rate_fpm=MAX_ROC,
    )

    t0 = time.time()
    pareto = pareto_opt.build_pareto_front()
    print(f"\nPareto front solve: {time.time() - t0:6.2f}s")

    if pareto is not None:
        front = pareto["front"]                 # sorted ascending fuel
        fuel = np.array([p["fuel"] for p in front])
        cont = np.array([p["contrail"] for p in front])
        print(f"  {len(front)} non-dominated trajectories")
        print(f"  min-fuel     : fuel={fuel[0]:.1f}  contrail={cont[0]:.4f}")
        print(f"  min-contrail : fuel={fuel[-1]:.1f}  contrail={cont[-1]:.4f}")

        # (a) The Pareto front itself: fuel vs contrail trade-off.
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fuel, cont, "-o", color="tab:purple", ms=4)
        ax.scatter(fuel[0], cont[0], color="tab:green", zorder=5, label="min fuel")
        ax.scatter(fuel[-1], cont[-1], color="tab:red", zorder=5, label="min contrail")
        ax.set_xlabel("Fuel cost")
        ax.set_ylabel("Contrail cost")
        ax.set_title(f"Pareto front — {flight_name} (σ={sigma})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        # (b) Overlay the two extreme trajectories (+ a mid one) on the cost map.
        extremes = {
            "Pareto min-fuel":     pareto["paths"][0],
            "Pareto mid":          pareto["paths"][len(pareto["paths"]) // 2],
            "Pareto min-contrail": pareto["paths"][-1],
            "Baseline":            paths["Baseline"],
        }
        plot_cost_map(
            {"env_distances": env.distances,
             "env_altitudes": env.altitudes,
             "env_cost_map":  env.cost_map},
            paths=extremes,
        )

    # 7) DP-based Pareto sweep — migrated from pareto.py::DPPareto, now a method
    #    of DpOptimizer (inherited by the facade). It scalarises fuel/contrail and
    #    re-solves the fast DP once per coef, reusing the precomputed edge arrays
    #    (no openap re-eval per point). Contrast with (6): the label-setting solver
    #    returns the exact front in one pass; the sweep samples it at chosen coefs.
    ws_front = opt.pareto_dp(grid=[0.0, 0.25, 0.5, 0.75, 1.0], method="weighted_sum")
    print(f"\nDP weighted-sum Pareto sweep: {len(ws_front)} points")
    for res in ws_front:
        print(f"  coef={res.meta['coef']:.2f}  fuel={res.fuel_kg:.1f} kg  "
              f"contrail={res.contrail:.4f}")

    plt.show()