# optimal-contrail-benchmark

Benchmark: OpenTOP (single/multistart) vs DP vs DP_MuO on fuel/contrail trade-off.
Pipeline = numbered stages 0_Data → 1_Flight_Selection → 2_Run → 4_Post_Treatment;
`src/` is the shared library. Data/results are NOT in git (Zenodo).

## Run
- Python 3.11, opentop pinned to 2.7.0 (TrackCruise refuses other versions).
- Scripts run from repo root: `python 2_Run/batch_opentop.py --limit 2 --sigmas 2`
- Cluster: follow docs/CLUSTER_CHECKLIST.md.

## Invariants (break these and results stop being comparable)
- Campaign settings live ONLY in `src/config.py`; batch_graph2d imports from batch_opentop — never restate constants.
- Settings not encoded in filenames (alt buffer, interp shape, track tolerance, m0, multistart) → each needs its own `--out-dir`.
- Narrow bodies m0=0.80, WIDEBODY_TYPES m0=0.60. M0 is a fraction of MTOW, not Mach.
- All solvers are priced post-hoc with `src/postprocess/evaluation.evaluate_path` (re-timed at flown altitudes via `flown_ts`). Old DP pickles: re-price with `4_Post_Treatment/reprice_dp_retimed.py`.
- Only converged fuel-only solves enter stage 2.

## Key types
- `OptResult` / `BaseOptimizer` (src/optimization/base.py): common contract for all solvers.
- `TrackCruise`: OpenTOP with lateral path imposed exactly (state = arc length s).
- `DpOptimizer`: fuel-only anchor + weighted-sum sweep (no ε since 2026-10-19). `DpMuoOptimizer`: whole front in one solve.

## Output
One pickle per (flight, σ, method): `{context, results, summary, error}`.
Names: `<id>_sigma<s>[ _fuel | _ms | _dp | _dp_muo ].pkl.gz`. Batches are resumable (skip existing).

## Conventions
- (your preferences: style, comments, commit rules, what not to touch…)
