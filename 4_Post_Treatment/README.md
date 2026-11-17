# 4_Post_Treatment — figures, tables and the convexity analysis

Every script here reads the pickles that stage 2 wrote. Point them at a run
directory; none of them re-solves anything (except `mass_profile_error.py`,
which rebuilds one flight's environment on purpose).

| Script | What it produces |
|---|---|
| `plots_comparison.py` | Solve-time comparison across the four methods; Pareto-front overlays; the shared `timings.csv` cache (`--refresh` rebuilds it) |
| `front_convexity.py` | How much of each Pareto front a weighted sum cannot reach — the non-convexity measure |
| `mass_profile_error.py` | Mass-profile error between the graph model and the OpenTOP trajectory |
| `post_treat.py` | Per-flight figures: Pareto comparison and altitude profiles |

## Reading timings correctly

- Use `solver_s`, not the sum of the phases. In ~20 % of the instrumented
  pickles `interpolant_s` is a cache hit under 1 s, not the real build cost, so
  summing phases invents a speed-up that did not happen.
- The DP evaluates paths **inside** its solve phase, DP_MuO evaluates its front
  **after** that phase closes. `posthoc_in_solver` in the context says which, and
  the numbers alone cannot tell you.
- Phase-1 convergence rates are **not** comparable across campaigns: the
  altitude window changed and the filename does not encode it. Phase 2 is fine.

## Naming

The multi-objective label-setting solver is **DP_MuO** in every label, title and
comment, and `dp_muo` in every machine name — the `--methods dp_muo` value,
`context['method']`, the `*_dp_muo.pkl.gz` files, the `pareto_dp_muo` records and
`dp_muo_run/`. Results written before 2026-09-28 say `dijkstra` there and are not
read by these scripts.
