"""
BaseOptimizer
=============
The universal contract shared by every optimizer family — graph-based
(2D/4D/5D) and opentop alike.

The only things true of *every* optimizer are:
  1. it is configured from a flight/environment plus fuel/contrail weights, and
  2. it produces a trajectory (``OptResult``) that can be evaluated to physical
     (fuel, contrail) metrics with identical post-hoc physics.

Anything family-specific — the five graph algorithms, ``path_to_arrays`` /
``graph_cost``, opentop warm starts / multistart — lives in the middle tiers
(``GraphOptimizerBase``, ``OpenTOPOptimizer``), NOT here.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
"""
OptResult
=========
The single trajectory type every optimizer returns, regardless of family
(graph-based 2D/4D/5D or opentop).  It normalises the output so downstream code
— evaluation, plotting, Pareto sweeps, saving — never has to care which solver
produced it.

Conventions
-----------
- ``alt_ft`` / ``dist_km`` / ``ts_s`` are the vertical profile, always present.
- ``lat`` / ``lon`` are only set by laterally-resolved solvers (4D/5D/opentop);
  they stay ``None`` for the 2D vertical-profile models.
- ``fuel_kg`` / ``contrail`` start as ``None`` and are filled in by
  ``BaseOptimizer.evaluate()`` (post-hoc physical metrics), so a raw solve and
  its evaluated form share the same object.
- ``cost`` is the solver's *native* objective value (graph edge-cost sum, or the
  opentop objective), kept separate from the post-hoc physical metrics.
"""

from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class OptResult:
    # Vertical profile — always present.
    alt_ft:   np.ndarray
    dist_km:  np.ndarray
    ts_s:     np.ndarray

    # Lateral track — 4D / 5D / opentop only.
    lat:      np.ndarray | None = None
    lon:      np.ndarray | None = None

    # Physical metrics (filled by BaseOptimizer.evaluate()).
    fuel_kg:  float | None = None
    contrail: float | None = None

    # Native solver objective value (graph_cost / opentop objective).
    cost:     float | None = None

    success:  bool = True
    meta:     dict = field(default_factory=dict)   # optimizer, algorithm, coef, time_s…

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def alt_points(self) -> int:
        return len(self.alt_ft)

    @property
    def dis_points(self) -> int:
        return len(self.dist_km)

    def to_row(self) -> dict:
        """
        Flat scalar record for tabulating a sweep (no trajectory arrays).
        ``meta`` is merged in so optimizer/coef/time land as columns.
        """
        return {
            "fuel_kg":  self.fuel_kg,
            "contrail": self.contrail,
            "cost":     self.cost,
            "success":  self.success,
            **self.meta,
        }

    def as_dict(self) -> dict:
        """Full dict including trajectory arrays — for pickling."""
        return asdict(self)


# -- example of use -----------------------------------------------------------
if __name__ == "__main__":
    # 1) A 2D vertical-profile solve (e.g. A*): no lateral track, so lat/lon
    #    stay None. cost is the solver's native graph edge-cost sum.
    n = 6
    vertical = OptResult(
        alt_ft=np.linspace(30000, 40000, 10),
        dist_km=np.linspace(0, 900, n),
        ts_s=np.linspace(0, 3_600, n),
        cost=1234.5,
        meta={"optimizer": "astar", "algorithm": "graph-2D"},
    )

    print("-- 2D vertical-profile result --")
    print(f"alt_points        : {vertical.alt_points}")
    print(f"dis_points        : {vertical.dis_points}")
    print(f"lateral track set : {vertical.lat is not None}")   # False
    print(f"fuel/contrail yet : {vertical.fuel_kg}, {vertical.contrail}")  # None, None (pre-evaluate)
    print(f"to_row()          : {vertical.to_row()}")

    # 2) A laterally-resolved solve (e.g. opentop), post-evaluate: lat/lon and
    #    the physical metrics are now filled in.
    lateral = OptResult(
        alt_ft=np.linspace(0, 39_000, n),
        dist_km=np.linspace(0, 950, n),
        ts_s=np.linspace(0, 3_800, n),
        lat=np.linspace(51.5, 32.9, n),
        lon=np.linspace(-0.5, 13.2, n),
        fuel_kg=5_820.0,
        contrail=142.0,
        cost=0.87,
        meta={"optimizer": "opentop", "coef": 0.5, "time_s": 12.3},
    )

    print("\n-- opentop result (evaluated) --")
    print(f"lateral track set : {lateral.lat is not None}")    # True
    print(f"to_row()          : {lateral.to_row()}")

    # 3) as_dict() gives the full record (arrays included) -- ready to pickle.
    full = lateral.as_dict()
    print(f"\nas_dict() keys    : {list(full.keys())}")
    print(f"alt_ft is ndarray : {isinstance(full['alt_ft'], np.ndarray)}")


class BaseOptimizer(ABC):
    """config → ``optimize()`` → ``OptResult`` → ``evaluate()``."""

    #: short identifier used in results / registry (e.g. "dp", "opentop").
    name: str = "base"

    def __init__(
        self,
        environment,
        *,
        mach: float,
        mass_init_kg: float,
        actype: str,
        fuel_weight: float = 1.0,
        contrail_weight: float = 0.0,
    ):
        self.env = environment
        self.mach = mach
        self.mass_init_kg = mass_init_kg
        self.actype = actype
        self.fuel_weight = fuel_weight
        self.contrail_weight = contrail_weight

    # ── contract ─────────────────────────────────────────────────────────────

    @abstractmethod
    def optimize(self) -> OptResult | None:
        """Solve the problem and return a normalized trajectory (None if infeasible)."""

    def pareto_front(self, **kwargs) -> list[OptResult]:
        """
        Return a set of non-dominated trajectories in a *single* solve.

        Only solvers with a native multi-objective mode (the graph label-setting
        Pareto) override this.  Optimizers without one — e.g. opentop — cannot
        build a front in one call: the caller must instead run a ``ParetoSweep``
        and pick a scalarisation strategy.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no Pareto front. Run a ParetoSweep "
            f"and choose a method: 'weighted_sum' (vary the fuel/contrail weights) "
            f"or 'epsilon_constraint' (vary the fuel budget)."
        )


# -- example of use -----------------------------------------------------------
# Run from the repo root with:  python -m src.optimization.optimizer
if __name__ == "__main__":
    import numpy as np

    # BaseOptimizer is abstract: every real family (graph 2D/4D/5D, opentop)
    # subclasses it and implements optimize(). Here is the smallest possible
    # concrete optimizer -- it just returns a hand-built trajectory so we can
    # exercise the shared contract without a real environment / physics stack.
    class DummyOptimizer(BaseOptimizer):
        name = "dummy"

        def optimize(self) -> OptResult | None:
            n = 6
            return OptResult(
                alt_ft=np.linspace(0, self.env["cruise_alt_ft"], n),
                dist_km=np.linspace(0, 900, n),
                ts_s=np.linspace(0, 3_600, n),
                cost=42.0,
                meta={"optimizer": self.name},
            )

    # 1) The base class cannot be instantiated on its own -- optimize() is
    #    abstract, so Python refuses.
    try:
        BaseOptimizer(environment=None, mach=0.78, mass_init_kg=65_000, actype="A320")
    except TypeError as exc:
        print("-- BaseOptimizer is abstract --")
        print(f"  {exc}")

    # 2) Configure a concrete optimizer: environment + cruise Mach + initial
    #    mass + aircraft type, plus the fuel / contrail objective weights.
    opt = DummyOptimizer(
        environment={"cruise_alt_ft": 37_000},   # a real solver takes a ContrailEnvironmentFromGrid
        mach=0.78,
        mass_init_kg=65_000.0,
        actype="A320",
        fuel_weight=1.0,
        contrail_weight=0.0,
    )

    # 3) Solve -> a normalized OptResult (fuel_kg / contrail still None here;
    #    evaluate() would fill them from the shared post-hoc physics, but that
    #    needs a real environment so we don't call it in this toy example).
    result = opt.optimize()
    print("\n-- optimize() result --")
    print(f"  cost      : {result.cost}")
    print(f"  n_points  : {result.alt_ft.size}")
    print(f"  to_row()  : {result.to_row()}")

    # 4) The two multi-objective hooks are optional: a solver that has no native
    #    Pareto mode / fuel-budget constraint raises an informative error telling
    #    the caller to use a ParetoSweep instead.
    print("\n-- optional multi-objective hooks --")
    for call in (opt.pareto_front, lambda: opt.solve_epsilon(fuel_budget_kg=5_000)):
        try:
            call()
        except NotImplementedError as exc:
            print(f"  {exc}")
