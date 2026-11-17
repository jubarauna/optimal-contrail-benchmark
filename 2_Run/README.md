# 2_Run — the benchmark itself

Four solvers over the same 2 409 flights, the same grid
(`grid_era5_smoothed.parquet.gz`) and the same per-flight initial mass
that stage 1 assigned. σ ∈ {0, 1, 2} is the smoothing of the contrail cost field.

| Method | Runner | How it is selected |
|---|---|---|
| OpenTOP singlestart | `batch_opentop.py` | default |
| OpenTOP multistart (5 starts) | `batch_opentop.py` | `--multistart` |
| DP | `batch_graph2d.py` | `--methods dp` |
| **DP_MuO** | `batch_graph2d.py` | `--methods dp_muo` |

**DP_MuO** is the multi-objective label-setting solver — one solve returns the
whole Pareto front. The CLI value, the `context['method']` field, the
`*_sigma<σ>_dp_muo.pkl.gz` filenames and the `pareto_dp_muo` record method all say
`dp_muo` since 2026-09-28. Earlier campaigns wrote `dijkstra` there
(`*_dijkstra.pkl.gz`, `dijkstra_run/`); the post-processing no longer reads them.

## Running

```bash
sbatch 2_Run/run_opentop.sbatch    # OpenTOP, --mode full
METHOD=dp     sbatch --job-name=dp     2_Run/run_graph2d.sbatch   # DP
METHOD=dp_muo sbatch --job-name=dp_muo 2_Run/run_graph2d.sbatch   # DP_MuO, its own job

# locally, a couple of flights
python 2_Run/batch_opentop.py --limit 2 --sigmas 2
python 2_Run/batch_graph2d.py --limit 2 --methods dp dp_muo
```

Both runners are resumable: an existing output file means that flight is
skipped, and `progress.csv` is appended after every flight.

## Before you size a job

- `GridWeather` keeps the grid file resident in every task — ~1.5 GB, peaking
  at 2× while loading. All three σ are columns of that one file, so `--sigmas
  0 1 2` costs no more grid memory than a single σ.
- The per-flight CasADi bspline over the cost slice dominates memory and scales
  with slice volume. Measured peak RSS: 34.5 GB on the worst OpenTOP task,
  26.65 GB on a single graph-2D flight (which then OOMed at 27.3 GB).
- Passing an interpolant to a fuel-only OpenTOP solve switches IPOPT to an exact
  Hessian: ~23× slower, and it shifts the fuel baseline. `--mode fuel` avoids it.
- DP_MuO has an **unbounded tail**. It is ~185× faster than the DP at the median
  and uses ~17× more CPU in total; the worst flight measured 24.3 h with 20 958
  front points. Budget wall clock for the tail, not for the median.

`--global-interp` swaps the per-flight fit for the whole-domain one from
`0_Data/`. Results are not point-by-point comparable with a per-flight run, so
the `.sbatch` refuses it unless the output directory name says so.
