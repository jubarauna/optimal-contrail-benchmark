# 1_Flight_Selection — which flights enter the benchmark

The contrail benchmark only admits flights whose **fuel-only** solve converged,
because that trajectory is the warm start and the ε-constraint's fuel budget F0.
This stage runs that fuel-only sweep and assembles the population.

## The two initial masses

`m0` is the initial mass as a fraction of MTOW, and it is set by aircraft type:
narrow bodies run at 0.80, wide bodies at 0.60. Wide body means a type in
`WIDEBODY_TYPES` (`src/config.py`): its OpenAP model has a fuselage at least 5 m
wide. The two populations never overlap, so every flight has exactly one m0, and
`select_flights.py` refuses any result solved at the other mass.

## Order of operations

One command, on the login node, from anywhere in the clone:

```bash
bash 1_Flight_Selection/submit_selection.sh
```

It submits three jobs:

| Job | Script | Writes to `3_Results/flight_selection_2d_cruise/` |
|---|---|---|
| 1. fuel-only, narrow bodies, m0 = 0.80 | `run_fuel_only.sbatch`, `FLEET=narrow` | `fuel_narrow_m080/` |
| 2. fuel-only, wide bodies, m0 = 0.60 | `run_fuel_only.sbatch`, `FLEET=wide` | `fuel_wide_m060/` |
| 3. flight lists, after 1 and 2 finish | `prepare_contrails_input.sbatch` | `selection/` |

Both fleets run under the same altitude window, −1000/+1000 ft around the real
flight's mean cruise altitude, and the same ground-track tolerance, 10 000 m (how
far the optimised track may deviate from the flown one). opentop's own default,
1000 m, left 0 of 10 debug flights feasible against 6 of 10 at 10 km.
`ALT_BUFFER="MIN MAX"` or `TRACK_TOL=<m>` in front of the command picks others.
Neither is in the directory names: the campaign runs one configuration, and each
pickle records its window and tolerance in its context. Move `3_Results/flight_selection_2d_cruise/` away before
running with other settings, or the finished flights are skipped.

If a sweep hits its time limit, job 3 never starts. The script prints the lines
to resubmit a sweep (it resumes) and to run the selection by hand.

Step 3 writes `flights.txt`, `flights_m008.txt`, `flights_m006.txt`,
`manifest.csv` and `rejected.csv`: every flight of the dataset that is NOT in
`flights.txt`, with its reason — `not_converged`, `error`, `unreadable`,
`wrong_mass`, or `no_result` (no result file at all, e.g. the job ran out of
time before reaching it). Every flight whose fuel-only solve converged at its
fleet's mass is selected; no flight is held out by ID. Selected + rejected =
all 5 919 flights.
All stage-1 output lives in `3_Results/flight_selection_2d_cruise/`. Stage 2 reads the lists
from `3_Results/flight_selection_2d_cruise/selection/` (the `SEL` default in both `2_Run`
sbatch files).

`run_fuel_only.sbatch` calls `2_Run/batch_opentop.py --mode fuel`. The runner is
shared with stage 2 on purpose: one code path means the altitude window, the
mass and the solver settings cannot drift between the warm start and the sweeps
that use it.

## Traps

- **Pair every altitude window with its own `--out-dir`.** `ALT_BUFFER_FT_MIN` /
  `ALT_BUFFER_FT_MAX` (now in `src/config.py`) were edited in place between
  campaigns and the filename encodes neither, so three phase-1 convergence rates
  ended up not comparable. Read the window back from
  `context['alt_buffer_ft_min'/'_max']`, never infer it.
- Same rule for `--m0`: the filename does not carry it either.

## Analysis scripts

| Script | What it answers |
|---|---|
| `select_flights.py` | Builds the contrail-phase input set; `--coverage` reports it |
