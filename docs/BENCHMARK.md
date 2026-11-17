# Benchmarking batches — OpenTOP and graph-2D

Two batch runners over the **same** flight population, the **same** mass
assumption and the **same** contrail cost field, so the solvers are comparable:

| Batch | Solver(s) | Method |
|---|---|---|
| `batch_opentop.py` | OpenTOP | continuous optimal control, direct collocation + IPOPT |
| `batch_graph2d.py` | DP, DP_MuO | discrete graph over (waypoint, flight level) |

Both read `0_Data/flight_data/eu_flights_2022feb20_persistent_all.parquet` and the
precomputed grid `grid_era5_smoothed.parquet.gz`, and both write **one
gzip'd pickle per unit of work** that you post-treat directly.

---

## Files

| File | Role |
|------|------|
| `batch_opentop.py` | OpenTOP batch runner. Flights × σ; slices meteo, builds a per-flight interpolant + cruise (no wind in the solve), runs the fuel-only warm start and/or the two Pareto sweeps. Resumable. |
| `batch_opentop_full.sbatch` | **The OpenTOP submission script in use.** `--mode full`, sized for memory, `OUT`/`FLIGHTS` hardcoded. |
| `batch_opentop.sbatch`, `batch_opentop_m0.sbatch` | Older submission scripts. **Not sized for memory** — see [Memory](#memory-read-this-before-sizing-any-job). |
| `batch_graph2d.py` | DP + DP_MuO batch runner. Flights × σ × method. Shares `ensure_flight_slice` / `load_flight_index` (`src/flights.py`), `log_progress` (`src/instrument.py`) and `WS_GRID` / `EPS_GRID` / `M0` / `ALT_BUFFER_*` (`src/config.py`) with the OpenTOP batch, so the two have the same flight handling and sweep grids by construction. |
| `batch_graph2d.sbatch` | The graph-2D submission script. |
| `instrument.py` | `phase()` context manager — per-phase wall time and RSS, recorded into every pickle. Imported by both runners. |
| `prepare_contrails_input.sbatch` | Builds the flight lists `flights_m006.txt` / `flights_m008.txt`. Run this first; everything else consumes them. |
| `open_top_run/drive_opentop.py` | Single-flight driver the batch mirrors. For inspecting one flight. |
| `select_flights.py` | Warm-start selection: the flight lists and `rejected.csv`. |
| `mass_profile_error.py` | How much the fixed mass profile (`model_2D._compute_mass_profile`) costs — mass error, per-edge cost error, and whether re-solving with the path's true mass moves the trajectory. The justification for not carrying mass as a state. `--flight ID` for one flight plus its figure; `--sample N` for a stratified sample (N per haul category) written to one resumable CSV. |
| `build_global_interpolant.py` | One interpolant over the whole domain, for a future shared-interpolant mode. Not used by either batch. |
| `Per_flight/benchmarking.py` | The old `Benchmarking` facade. Kept for reference; **not** used. |

---

## Launching on the cluster

Submit from the repository root: every path in the `.sbatch` files is relative
to it (`cd "$SLURM_SUBMIT_DIR"`).

### One variable to set

Both `.sbatch` files now derive everything from `M0`. `OUT` defaults to a folder
under `3_Results/contrail_2d_cruise/` and `FLIGHTS` is chosen from `M0` by a `case`:

```bash
case "$M0" in
    0.80|0.8) export FLIGHTS="$SEL/flights_m008.txt" ;;
    0.60|0.6) export FLIGHTS="$SEL/flights_m006.txt" ;;
    *) echo "no flight list defined for M0=$M0" >&2; exit 1 ;;
esac
```

This replaced `export OUT=... FLIGHTS=...` on the command line, which cost two
jobs on 2026-08-03: a fresh login shell had dropped the exports, `$OUT` expanded
to nothing, `FLIGHTS` became the absolute path `/flights_m006.txt`, and both jobs
died at the guard. Deriving `FLIGHTS` from `M0` also removes the possibility of
pairing `m0=0.60` with the `m008` list and silently benchmarking the wrong
subpopulation.

### OpenTOP

```bash
export M0=0.60 && sbatch --export=ALL studies/benchmarking/batch_opentop_full.sbatch
export M0=0.80 && sbatch --export=ALL studies/benchmarking/batch_opentop_full.sbatch
```

### Graph-2D

```bash
export M0=0.60 && sbatch --export=ALL studies/benchmarking/batch_graph2d.sbatch
export M0=0.80 && sbatch --export=ALL studies/benchmarking/batch_graph2d.sbatch
```

Second pass for whatever OOM'd, same file, flags overridden — no edit, no flight
list, because `is_done()` skips everything the first pass wrote:

```bash
export M0=0.60 && sbatch --nodes=16 --ntasks=16 --ntasks-per-node=1 \
    --cpus-per-task=22 --export=ALL studies/benchmarking/batch_graph2d.sbatch
```

### Changing σ

`--sigmas` is `nargs="+"` in both runners, so `--sigmas 0 1 2` runs all three in
one job — the pickles are named per σ and never collide. It is edited **by hand**
in the `srun` body of the `.sbatch`; it is deliberately not an environment
variable.

`sbatch` snapshots the script at submission, so editing the file afterwards does
not affect jobs already queued or running. The normal workflow is therefore:
submit, edit the σ line, submit again.

**A `.py` file is NOT snapshotted** — it is read when each task starts. Never
transfer a runner while jobs using it are queued or running.

### Testing before committing a big allocation

The `debug` partition (`MaxNodes=2`, `MaxTime=02:00:00`, includes bigmem nodes)
runs the **real** script against a small flight list, by pointing `OUT` (and, for
graph-2D, `SEL`) at a test directory:

```bash
TEST=$PWD/studies/benchmarking/open_top_run/contrails_debug
mkdir -p $TEST
head -4 studies/benchmarking/open_top_run/contrails_full/flights_m006.txt \
    > $TEST/flights_m006.txt

sbatch --partition=debug --nodes=1 --ntasks=2 --ntasks-per-node=2 \
       --cpus-per-task=11 --mem-per-cpu=3900M --time=00:30:00 \
       --job-name=dbg --mail-type=NONE \
       --export=ALL,M0=0.60,OUT=$TEST \
       studies/benchmarking/batch_opentop_full.sbatch
```

This exercises the exact `OUT`/`FLIGHTS` logic, the grid loading and the shard
split. Use `--mem-per-cpu=3900M` on `debug`: it is a mixed partition and a task
can land on a 96 GB node, where `7500M` exceeds `MaxMemPerCPU` and is rejected or
clamped.

**Python's stdout is block-buffered when redirected to a file**, so the runner's
`print()` output does not appear in the `.out` in real time — only the `echo`
lines from bash do. Do not read a missing "Loading meteo grid …" as a failure;
check the filesystem instead.

---

## Memory: read this before sizing any job

This is where every failure in this batch has come from.

### What actually allocates

Per task, per flight:

| Component | Size |
|---|---|
| `GridWeather` — the full grid parquet (every σ is a column of it), kept resident | ~1.5 GB, and `sort_values` peaks at 2× while loading |
| CasADi `bspline` interpolant over the flight's cost slice | **dominant**, and scales with slice volume |
| OpenTOP only: IPOPT with an exact Hessian (selected by the interpolant's presence) | large |
| graph-2D only: `df_cost` plus the `(n−1, n_alts, span)` edge tables | modest |

`GridWeather.__init__` (`src/data.py`) loads the grid file once and keeps it, in
every task, regardless of which flights that task handles. σ=0, 1 and 2 are all
columns of that one file, so the σ list does not change grid memory. The peaks
measured below predate this: σ=1 then came from a second file, ~1.5 GB more per
task whenever it was in the list.

### Measured peaks

From `/usr/bin/time -v` — the kernel's true, unsampled peak, printed per task by
both `.sbatch` files:

| Job | What | Peak RSS |
|---|---|---|
| 3999869 / 3999870 | OpenTOP `--mode full`, worst of 48 tasks | **34.5 GB** |
| 4000732 | graph-2D, 3 flights × 2 methods, σ=2 | 6.06 GB |
| 4000787 | graph-2D, flight `010154_00026_0` alone | **26.65 GB → OOM at 27.3 GB** |

Two lessons, both learned the hard way:

- **`sacct`'s `MaxRSS` under-reports.** For the same OpenTOP jobs it said
  29.15 GB against the true 34.5 GB — it samples every `JobAcctGatherFrequency`
  seconds (30 by default) and the bspline build's peak is a short transient.
  Size on `/usr/bin/time -v`, cross-check with `sacct`, never the reverse.
- **The median is not the population.** Sizing graph-2D on its 6.06 GB (three
  flights picked with `head -4`) put it at 27.3 GB per task, and a single flight
  went straight through that. The tail is several times the median because slice
  volume grows with flight length.

### How to buy memory

Memory and cores are **tied per node**: you cannot ask for memory without asking
for the cores that go with it.

| Partition | Nodes | Cores/node | Memory/node | Ceiling per core |
|---|---|---|---|---|
| `medium` | 5–16 | 24 | 96 GB | 4000 MB |
| `bigmem` | 1–4 (of 12) | 24 | 192077 MB | 8003 MB |
| `gigamem` | 1–4 | 24 | 772600 MB | ~32 GB |

`--mem-per-cpu` **above the ceiling is rejected or silently clamped**. That is why
a 45 GB task is written `--cpus-per-task=6 --mem-per-cpu=7500M` and not
`--cpus-per-task=1 --mem-per-cpu=45000M`.

The three constraints, all of which SLURM enforces:

```
ntasks-per-node × cpus-per-task  ≤  24                    cores available
ntasks-per-node × cpus-per-task × mem-per-cpu  ≤  ~90%    leave room for slurmd
nodes × ntasks-per-node  ==  ntasks                       or the job is rejected
```

**The partition is not what caps per-task memory — tasks-per-node is.** One task
alone on a 96 GB `medium` node gets 85.8 GB, more than the currently-free
`bigmem` nodes would give at the same task count. Choose the partition on node
*availability*, then choose tasks-per-node on the memory the tail needs.

### `SRUN_CPUS_PER_TASK` is mandatory

SLURM ≥ 22.05 stopped propagating `--cpus-per-task` from `sbatch` to `srun`.
Without this line the step launches 1 CPU per task and each cgroup gets
`mem-per-cpu` instead of `cpus-per-task × mem-per-cpu` — the entire memory
allocation is silently undone:

```bash
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
```

### Current sizing

| Job | Partition | Layout | Per task |
|---|---|---|---|
| `batch_opentop_full.sbatch` | `bigmem` | 4 nodes × 4 tasks × 6 cores × 7500M | 45 GB (16 tasks) |
| `batch_graph2d.sbatch` | `medium` | 16 nodes × 2 tasks × 11 cores × 3900M | 42.9 GB (32 tasks) |
| graph-2D pass 2 | `medium` | 16 nodes × 1 task × 22 cores × 3900M | 85.8 GB (16 tasks) |

**Expect a few flights to OOM anyway.** That is contained and deliberate: the
kill takes one task, the rest keep going, and the flight simply has no pickle.
Re-running the leftovers with more memory is cheaper than sizing every task for
the worst flight in the population. The per-task cap (rather than a shared
`--mem=<node>` pool) is what makes it contained — with a shared pool the OOM
killer takes every task on the node.

### The configuration that fails

`--ntasks-per-node=24` on a 96 GB node gives each task 4 GB. This is what killed
the first full run:

```
terminate called after throwing an instance of 'std::bad_alloc'
srun: error: nodeNNN: task N: Killed
```

`batch_opentop.sbatch` and `batch_opentop_m0.sbatch` still carry that layout.

---

## Options

### Shared by both runners

| Flag | Default | Meaning |
|------|---------|---------|
| `--sigmas S [S ...]` | `2` | σ values; several allowed |
| `--flights ID [ID ...]` | all | only these `flight_id`s |
| `--flights-file PATH` | — | ids from a file, one per line (`#` comments ignored). Use this, not `--flights`, for large sets — thousands of ids do not survive the quoting in an `srun bash -c '…'` body |
| `--limit N` | — | first N flights only (smoke test) |
| `--retry-failed` | off | re-attempt flights saved with an error. **Required when resuming after an OOM**: without it `is_done()` counts an error stub as finished and that flight is skipped forever |
| `--m0 F` | `0.80` | initial mass as a fraction of MTOW |
| `--actypes ICAO [...]` | all | only these aircraft types |
| `--workers N` | `1` | worker processes **within** one task; each builds its own `GridWeather` |
| `--shard i/N` | — | handle only shard `i` of `N` (round-robin) |
| `--grid-path`, `--data-path`, `--out-dir` | see below | inputs and output directory |

### OpenTOP only

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode {fuel,contrails,full}` | `full` | see [Run modes](#run-modes-opentop) |
| `--multistart` | off | several starts per solve instead of one: the fuel-only trajectory, then it shifted by each `MULTISTART_ALT_OFFSETS_FT` altitude offset (lateral track unchanged) |
| `--n-starts N` | `5` | starts per solve (1–5): the fuel-only trajectory and the first N−1 offsets; only with `--multistart` |

Multistart costs **~5.2×** single start, measured end-to-end on the same flight
(104.5 s against 16.0 s). It writes to `*_ms.pkl.gz` — without that suffix a
multistart run would collide with the single-start filenames and `is_done()`
would skip every flight that already had a result, burning the whole allocation
doing nothing.

### Graph-2D only

| Flag | Default | Meaning |
|------|---------|---------|
| `--methods {dp,dp_muo} [...]` | both | which solvers to run; each writes its own pickle, so they are independently resumable |

---

## Configuration constants

### `batch_opentop.py`

| Constant | Value | Meaning |
|---|---|---|
| `ALT_BUFFER_FT_MIN` | `4000.0` | window floor: real mean cruise altitude − this |
| `ALT_BUFFER_FT_MAX` | `4000.0` | window ceiling: real mean cruise altitude + this |
| `M0` | `0.80` | initial mass as a **fraction of MTOW** (not a Mach) |
| `MAX_ITER` | `3000` | IPOPT max iterations per solve |
| `WARM_START` | `False` | whether the sweeps **seed** from the fuel-only trajectory — see below |
| `MULTISTART_ALT_OFFSETS_FT` | `(-2000, -1000, 1000, 2000)` | altitude shifts of the fuel-only trajectory, one start each, under `--multistart` |
| `N_STARTS` | `5` | starts per solve under `--multistart`: 1 + the number of offsets |
| `WS_GRID` | `[0.1,0.3,0.5,0.7,0.9,1.0]` | weighted-sum contrail weights |
| `EPS_GRID` | `[1.01,1.02,1.03,1.04,1.05,1.10]` | ε-constraint fuel-slack fractions |

13 solves per flight in `--mode full`: 1 fuel-only + 6 weighted-sum + 6
ε-constraint. That is `n_solves = 13` in `progress.csv`.

### `WARM_START` — what it does and does not control

The fuel-only solve **always runs** in `--mode full`, independently of this flag.
`batch_opentop.py:468` calls `opt.warm_start()` explicitly before the sweeps,
because it is both the F0 the ε-constraint budgets are relative to and the
minimum-fuel anchor of the Pareto front — and `pareto_front()` does not record it
in `opt.results`, so leaving it to be triggered implicitly would lose it.

That one solve is then used in two independent ways:

| use | what is consumed | controlled by | in `--mode full` |
|---|---|---|---|
| **numerical seed** | the whole *trajectory*, handed to IPOPT as its starting point | `WARM_START` | **off** |
| **F0 baseline** | just a *number*, the fuel burnt | nothing — structural | **on** |

`WARM_START` only touches the first (`opentop_optimizer.py:211-212`):

```python
if self._use_warm_start and self._initial_guess is not None:
    kw["initial_guess"] = self._initial_guess
```

With `False`, the 12 sweep solves start from IPOPT's own default initialisation.
The second is unavoidable: `_run_epsilon_constraint` computes
`F0 = traj["mass"][0] − traj["mass"][−1]` and the budgets are `F0 × EPS_GRID`.

`warm_start()` is idempotent, so the fuel-only problem is solved **once** per
flight either way — the flag changes what is done with the answer, not how many
solves run.

Under `--multistart` the flag has no effect at all: random restarts supply their
own initial guesses and the seed is never passed
(`opentop_optimizer.py:205-208`).

**The collocation grid matches DP's waypoints, flight by flight.** `batch_opentop.py`
sets `nodes` to DP's segment count, `route_point_count(...) - 1` with DP's own
`MAX_ROC_FPM` / `ALT_STEP_FT` (one point per 2 min of flight, ~26 km), so both
methods have the same number of points along each route. opentop 2.7.0 uses an
explicit count as-is: no floor of 20 and no cap of 120. The count is saved in the
context (`nodes`, `nodes_rule="dp_waypoints"`). Results before 2026-10-08 used one
interval per 50 km of origin-destination distance (`NODE_KM = 50`, removed), as few
as 1 interval on short flights; their context has no `nodes` key.

### `batch_graph2d.py`

| Constant | Value | Meaning |
|---|---|---|
| `ALT_STEP_FT` | `1000.0` | altitude lattice step |
| `MAX_ROC_FPM` | `500.0` | max climb and descent rate |

Both `ALT_BUFFER_*`, `M0`, `WS_GRID` and `EPS_GRID` are imported from
`batch_opentop.py`, so the altitude window and the sweep grids cannot drift
between the two batches.

**These two constants pin the time step, and it cannot be made finer.** The route
point count (`contrail_grid.py:132`) is

```python
num_points = floor(roc * total_distance_km * 1000 / (tas_ms * alt_step_ft * 60))
```

Substituting the flight time `T = D/tas`, the step comes out at
`alt_step × 60 / roc = 120 s`, independent of distance and speed. That is the
*minimum*: `model_2D.py:314` computes

```python
self._max_steps = int(self.max_climb_rate_fpm * (dt0_s / 60.0) / self.env.alt_step_ft)
```

and below 120 s this is `0` — `span = 1`, the only allowed transition is to stay
level, and the optimisation disappears. At exactly 120 s, one level per step is
exactly 500 fpm: no climb capability is wasted to integer truncation.

Coarser is allowed, but only in integer multiples: `dt = k × 120 s` gives
`max_steps = k` and preserves the rate. `dt = 240 s` halves the point count, puts
a node every ~56 km (against OpenTOP's 50 km) and keeps the physics exact.
Matching OpenTOP's node spacing *exactly* (213 s) would leave `max_steps = 1` and
silently reduce the effective climb rate to 282 fpm.

---

## Run modes (OpenTOP)

| Mode | What it solves | Output file |
|---|---|---|
| `fuel` | ONLY the fuel-only warm start | `<id>_sigma<σ>_fuel.pkl.gz` |
| `contrails` | the WS + ε-constraint sweeps, **reusing** the saved fuel-only warm start | `<id>_sigma<σ>.pkl.gz` |
| `full` (default) | fuel-only warm start **+** both sweeps in one go | `<id>_sigma<σ>.pkl.gz` |

`--mode full` is what the cluster script uses, and it matters for correctness:
F0 — the ε-constraint's fuel budget — then necessarily shares the same altitude
window, mass and solver settings as the sweeps that consume it.

`--mode contrails` errors for any flight whose `*_fuel.pkl.gz` is missing or
recorded an error, so run `--mode fuel` first if you split them.

---

## What runs per flight

### OpenTOP

1. **Flight geometry** — `FlightData(base_id, path)`; origin/dest, mean cruise
   altitude and Mach from the real flight. The grid slice carries wind, so the
   cruise Mach is wind-corrected (TAS, not ground-referenced).
2. **Meteo + cost grid** — sliced from the precomputed grid via `GridWeather`,
   with departure-relative `ts` matching opentop's elapsed-time state.
3. **Interpolant** — a bspline over that flight's cost slice, cached in `%TEMP%`.
4. **Cruise, without wind** — `top.Cruise` with `fix_mach_number()`,
   `allow_cruise_descent()` and `follow_track(...)`. `enable_wind` is **not**
   called: the wind only corrects the cruise Mach (step 1), so flight time is
   distance / TAS, as in the graph solvers. Pickles written before 2026-10-08 say
   `wind_enabled=True` in their context; the solve was the same, without wind.
5. **Fuel-only warm start**, then the two Pareto sweeps via
   `OpenTOPOptimizer.pareto_front`.

### Graph-2D

Steps 1–3 are identical — the same interpolant object, which is why the two
batches have the same memory profile. Then:

4. **Route** — `real_flight_route` resamples the real lateral path onto equally
   spaced points (the lateral path is fixed; only altitude is optimised).
5. **Edge cost tables** — `(n−1, n_alts, span)` arrays of unweighted fuel and
   contrail cost, built **once** with one openap thrust/drag/fuel call per
   waypoint. The arrays are tiny (9 levels × span 3), so what is paid is openap's
   per-call overhead, linear in the number of waypoints.
6. **13 sweep points** — these only *reweight* the cached tables
   (`model_2D.py:795-812`); the DP recursion itself is trivial. The sweep is not
   what costs.

Measured at σ=2: DP 52–256 s per flight, DP_MuO 0.6–61 s (cheap unless its
Pareto front explodes — 535 points on one flight), plus a long tail: one flight
exceeded 24 minutes.

---

## Output layout

```
<out-dir>/
    progress.csv                          # live log, one row per unit of work
    _flight_cache/<flight_id>.parquet     # per-flight slice (perf cache)
    debug/<flight_id>_sigma<σ>_debug.pkl.gz
    <flight_id>_sigma<σ>.pkl.gz           # OpenTOP full / contrails
    <flight_id>_sigma<σ>_fuel.pkl.gz      # OpenTOP --mode fuel
    <flight_id>_sigma<σ>_ms.pkl.gz        # OpenTOP --multistart
    <flight_id>_sigma<σ>_dp.pkl.gz        # graph-2D DP
    <flight_id>_sigma<σ>_dp_muo.pkl.gz  # graph-2D DP_MuO
```

Defaults: `open_top_run/opentop_batch` and `graph2d_run` (override `--out-dir`).
On the cluster both `.sbatch` files override these with absolute paths.

**Resumable.** A flight whose pickle exists with no recorded error is skipped, so
an interrupted run (wall-time cut, node failure, OOM) continues on resubmit.
`--retry-failed` additionally re-attempts the errored ones and the truncated
pickles left by tasks that were SIGKILLed mid-write.

Each pickle is one dict:

```python
{
    "context": {...},          # flight identity, begin/end, origin/dest, mach, mass,
                               #   alt window, sigma, multistart/n_starts/seed,
                               #   per-phase timings and RSS
    "results": {               # full OptResult dicts, grouped by method
        "fuel_only":          [rec],
        "weighted_sum":       [rec, rec, ...],
        "epsilon_constraint": [rec, rec, ...],
    },
    "summary": [ {...}, ... ], # scalar rows only (no trajectory arrays)
    "error":   None,           # str if the flight failed
}
```

Each record carries the profile arrays (incl. `lat`/`lon`) and, in `meta`, the
trajectory DataFrame, the objective value and the IPOPT
iteration count. **Non-converged solves are kept with `success=False`** — filter,
don't lose them.

### `progress.csv`

```
utc_time, flight_id, sigma, status, elapsed_s, n_solves, n_success, fail_status, error
```

- `status` — `done` (the solve ran) or `error` (exception around the solve)
- `fail_status` — IPOPT status when it did not converge cleanly; **empty means
  converged**. Seen: `Infeasible_Problem_Detected`, `Maximum_Iterations_Exceeded`
  (`n_iter == MAX_ITER`), `Restoration_Failed`

```bash
awk -F, 'NR>1 && $4=="done" && $8=="" {n++} END{print n" converged"}' progress.csv
```

**No write lock.** All tasks append concurrently, so lines can interleave and a
few header rows get mixed in. This does not affect correctness — resumability
keys off the per-flight pickle filenames, not the CSV. Treat it as a best-effort
triage log; the `pkl.gz` count is the number that matters.

---

## Loading results

```python
import pickle, gzip
d = pickle.load(gzip.open("…/0100e0_00011_1_sigma2.pkl.gz", "rb"))

print(d["error"])                 # None if the flight ran
d["summary"]                      # scalar rows
rec = d["results"]["weighted_sum"][0]
rec["lat"], rec["lon"]            # optimised lateral track
rec["meta"]["trajectory"]         # full trajectory DataFrame
```

One table across all flights (small — summaries only):

```python
import glob, gzip, pickle, pandas as pd
rows = []
for f in glob.glob("…/*_sigma*.pkl.gz"):
    d = pickle.load(gzip.open(f, "rb"))
    for r in d["summary"]:
        rows.append({"flight_id": d["context"].get("flight_id"), **r})
df = pd.DataFrame(rows)
```

---

## Meteo: the precomputed grid

Both batches read the precomputed contrail-cost + wind grid and slice each
flight's cost grid from it (`GridWeather`, `data/data.py`) — no per-flight ERA5
fetch, no runtime smoothing. The cost for each σ:

| σ | source |
|---|------|
| 0 | `persistent` column of the base grid (raw boolean → cost) |
| 2 | `cost` column of the base grid (already smoothed at σ=2) |
| other (e.g. 1) | `cost_sigma<σ>` column of the base grid (`cost_sigma1`) |

The same grid supplies the wind field (`u`/`v`), used only to correct the cruise
Mach; no solver flies with it.

> A σ with no column in the grid stops `GridWeather` with a `KeyError` at startup,
> before any flight runs. Add it with `smooth_cost_grid` in
> `0_Data/regenerate_grid.py` and regenerate the grid once, outside the job.

---

## Parallelism

Two independent levels:

1. **`--shard i/N`** — split work *across* processes/nodes. Shard `i` handles
   items `i, i+N, i+2N, …`. Disjoint, so any number run at once with no
   coordination. Driven by `$SLURM_PROCID`.
2. **`--workers W`** — `W` items at once *within* one process. Each worker builds
   its **own** `GridWeather`, so W multiplies the memory footprint. On the cluster
   this is always `1`; per-task memory is bought with `--cpus-per-task` instead.

Keep every solve single-threaded (`OMP_NUM_THREADS=1` etc., set in both `.sbatch`
files) so tasks don't oversubscribe cores. The extra cores of a task exist to
raise its memory quota, **not** to be used.

**Why `bash -c '…'` with single quotes.** `$SLURM_PROCID` only exists *inside*
each `srun` task. Written directly, the batch shell would expand it to nothing
before `srun` runs (and abort under `set -u`). Single quotes defer expansion to
each task; because they also block `$OUT`, `$GRID`, `$DATA`, those are `export`ed
so the child `bash` inherits them.

---

## Gotchas

- **`python` on the compute nodes is system Python 2.** For one-off commands
  outside the batch, use `.conda-optimal/bin/python` explicitly.
- **Transfer every module the runner needs, and check they are current.**
  `batch_graph2d.py` additionally needs `instrument.py`, `contrail_grid.py`,
  `model_2D.py` and `evaluation.py`. A *stale* file is worse than a missing one:
  a missing module raises `ModuleNotFoundError`, but an old `model_2D.py` runs
  silently against an outdated solver. Compare sizes and dates, not just presence.
- **`ensure_flight_slice` is not atomic** (`batch_opentop.py:140-147`):
  `if not path.exists(): … to_parquet(path)`. Two jobs sharing an `_flight_cache`
  can write the same slice simultaneously and corrupt it — and this does **not**
  self-heal, because the corrupt file exists, so `--retry-failed` reuses it
  forever. If a flight fails with a parquet/pyarrow read error, delete
  `_flight_cache/<id>.parquet` before resubmitting.
- **CasADi needs ≥ 4 points per axis.** Flights whose slice spans fewer than 4
  time steps (late-night departures near the grid's temporal edge) fail with
  `Need more data points`. They are recorded as error stubs and skipped.
- **Line endings.** `dos2unix` any `.sbatch` edited on Windows, or bash chokes
  on `\r`.
- **These files are not in git.** `batch_opentop.py`, `batch_opentop_full.sbatch`,
  `batch_graph2d.sbatch` and `instrument.py` were untracked as of 2026-08-04, so
  there is no history to diff against. Commit them.

---

## Known open questions

- **18 OpenTOP flights write a `debug/` pickle** (IPOPT never converged at any
  sweep point): 83 `Infeasible_Problem_Detected` + 53 `Maximum_Iterations_Exceeded`
  at `n_iter == 3000`, spread uniformly across all 13 sweep points. For 9 of the
  18 the `fuel_only` baseline itself fails, which poisons F0 for their whole
  ε-constraint sweep. Not yet attributed to aircraft type, distance or m0.
- **4 further flights are dropped without a debug file.** `n_bad` counts stored
  results only, so a flight whose solves all returned `None` writes nothing; their
  IPOPT status survives only in the slurm `.out` logs. `None` comes from
  `opentop/cruise.py:303-314`, which discards a trajectory whose
  `altitude.max() < 5000` or whose final mass is below OEW — both of which are
  also NLP bounds, so they can only be violated by IPOPT's infeasible-start
  iterates being returned as `opti.debug` after a `RuntimeError`.
- **Widebodies do not converge at `m0 = 0.80`** (0% success); `m0 = 0.55` gives
  ~70%.
- **Discretisation is not matched between solvers**, by design — see
  [Configuration constants](#batch_graph2dpy). What would settle the comparison is
  a convergence check per method (refine until the objective stops moving), not
  equal point counts. For graph-2D the refinement axis is `(alt_step, dt)`
  together: 500 ft with 60 s keeps `max_steps = 1` at ~4× the cost. For OpenTOP it
  is raising `max_nodes` above 120.
