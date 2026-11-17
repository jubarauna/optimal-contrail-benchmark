"""Wall-time and peak-memory instrumentation shared by the solver batches.

Exists so the OpenTOP, DP and DP_MuO runs are measured the SAME way.
The three allocate in different places — CasADi/IPOPT in native memory, the DP in
NumPy arrays, the label-setting sweep in plain Python objects — so ``tracemalloc``
is not an option: it sees Python allocations only and would understate the
optimal-control solver by most of its footprint. This samples the process RSS
instead, which counts all three equally.

    with phase("solve") as m:
        ...
    m.seconds, m.peak_rss_mb, m.delta_rss_mb

Caveats that belong in any table built from this:

* **RSS is per PROCESS.** With ``--workers 1`` (what the SLURM tasks use) one
  flight owns the process and the numbers are clean. With N>1 workers a pool
  process handles flights back to back, so ``peak_rss_mb`` is that process's high
  water mark, not this flight's — only ``delta_rss_mb`` stays meaningful, and
  even that is noisy under a shared allocator.
* **Peak is sampled, not exact.** A spike shorter than the sampling interval can
  be missed. The interval is deliberately small relative to a solve (which takes
  seconds), but a sub-10 ms allocation burst is invisible.
* **The allocator does not return memory promptly.** A negative or zero delta
  after a big phase means the pages were reused, not that nothing was allocated;
  compare peaks across methods rather than deltas.
"""

from __future__ import annotations

import threading
import time
import csv
from pathlib import Path
from dataclasses import dataclass, field

try:
    import psutil
    _PROC = psutil.Process()
except Exception:  # noqa: BLE001 — instrumentation must never break a run
    psutil = None
    _PROC = None

SAMPLE_INTERVAL_S = 0.01
_MB = 1024 * 1024


def rss_mb() -> float:
    """Resident set size of this process, MB. 0.0 when psutil is unavailable."""
    if _PROC is None:
        return 0.0
    try:
        return _PROC.memory_info().rss / _MB
    except Exception:  # noqa: BLE001
        return 0.0


@dataclass
class Measurement:
    name: str
    seconds: float = 0.0
    start_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    delta_rss_mb: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self, prefix: str | None = None) -> dict:
        p = f"{prefix or self.name}_"
        out = {
            f"{p}s": round(self.seconds, 4),
            f"{p}peak_rss_mb": round(self.peak_rss_mb, 1),
            f"{p}delta_rss_mb": round(self.delta_rss_mb, 1),
        }
        out.update({f"{p}{k}": v for k, v in self.extra.items()})
        return out


class phase:
    """Context manager timing a block and tracking this process's peak RSS.

    The sampler runs in a daemon thread so a crash inside the block cannot leave
    it running, and it is stopped in ``__exit__`` even when an exception unwinds.
    """

    def __init__(self, name: str, sample: bool = True):
        self.m = Measurement(name)
        self._sample = sample and _PROC is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        peak = self.m.start_rss_mb
        while not self._stop.is_set():
            cur = rss_mb()
            if cur > peak:
                peak = cur
            self.m.peak_rss_mb = peak
            self._stop.wait(SAMPLE_INTERVAL_S)
        cur = rss_mb()
        self.m.peak_rss_mb = max(peak, cur)

    def __enter__(self) -> Measurement:
        self.m.start_rss_mb = rss_mb()
        self.m.peak_rss_mb = self.m.start_rss_mb
        if self._sample:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self._t0 = time.perf_counter()
        return self.m

    def __exit__(self, *exc) -> bool:
        self.m.seconds = time.perf_counter() - self._t0
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.m.delta_rss_mb = self.m.peak_rss_mb - self.m.start_rss_mb
        return False   # never swallow an exception


# ── the four reported times ──────────────────────────────────────────────────
# The benchmark reports four times per (flight, sigma, method), the same four for
# DP, DP_MuO and OpenTOP (since 2026-10-19):
#
#   data_setup_s    meteo slice + contrail interpolant build/load    meteo_s + interpolant_s
#   solver_setup_s  altitude window, flown route (+ the DP cost map) env_s + build_s
#                   and the solver object: DP/DP_MuO graph, OpenTOP Cruise + optimizer
#   solve_s         every solve of the flight; for DP_MuO the label  solver_s, post-hoc out
#                   search up to the final Pareto front
#   posthoc_s       the evaluation at the end: fuel + contrail       posthoc_s
#                   (evaluate_path for DP/DP_MuO, reprice for OpenTOP)
#
# The phase fields stay in the pickle as the breakdown; these are sums of them.
TIME_FIELDS = ("data_setup_s", "solver_setup_s", "solve_s", "posthoc_s")


def four_times(ctx: dict) -> dict:
    """The four reported times from a pickle's ``context``, old pickles included.

    A missing phase counts as zero (pickles before 2026-09-28 have no env_s, and
    OpenTOP before 2026-10-19 no posthoc_s). Old DP pickles with
    ``posthoc_in_solver=True`` carry the post-hoc inside solver_s; it is taken out
    here so solve_s means the same thing for every pickle.
    """
    def g(k):
        v = ctx.get(k)
        return 0.0 if v is None or v != v else float(v)    # None / NaN -> 0

    posthoc = g("posthoc_s")
    solve = g("solver_s")
    flag = ctx.get("posthoc_in_solver")
    if flag is not None and flag == flag and str(flag).lower() == "true":   # CSV-safe
        solve -= posthoc
    return {"data_setup_s": round(g("meteo_s") + g("interpolant_s"), 4),
            "solver_setup_s": round(g("env_s") + g("build_s"), 4),
            "solve_s": round(solve, 4),
            "posthoc_s": round(posthoc, 4)}


# ── progress log ─────────────────────────────────────────────────────────────
_PROGRESS_FIELDS = ["utc_time", "flight_id", "sigma", "status",
                    "elapsed_s", "n_solves", "n_success", "fail_status", "error"]


def log_progress(out_dir: Path, row: dict) -> None:
    path = out_dir / "progress.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROGRESS_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _PROGRESS_FIELDS})
