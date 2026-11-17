#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Fuel-only sweep of ONE fleet, the first step of flight selection. The same
# script runs both fleets, so their altitude window and solver settings cannot
# drift apart:
#
#   FLEET=narrow   narrow bodies at m0=0.80 (src.config.M0)
#   FLEET=wide     wide bodies at m0=0.60 (src.config.M0_WIDEBODY), i.e. the
#                  types in src.config.WIDEBODY_TYPES
#
# The two populations are disjoint, so every flight has exactly one m0, decided
# by its aircraft type. Normally submitted by submit_selection.sh, which runs
# both fleets and chains the selection after them:
#
#   bash 1_Flight_Selection/submit_selection.sh
#
# By hand, one fleet at a time (both fleets run at the #SBATCH size below):
#
#   FLEET=narrow sbatch --job-name=fuel_narrow 1_Flight_Selection/run_fuel_only.sbatch
#   FLEET=wide   sbatch --job-name=fuel_wide   1_Flight_Selection/run_fuel_only.sbatch
#
# Results go to 3_Results/flight_selection/fuel_<fleet>_m<m0>/.
# The run is resumable: resubmit the same line and finished flights are skipped.
#
# Count converged flights afterwards, per type, without writing anything:
#   python 1_Flight_Selection/select_flights.py --dry-run --rejected \
#       --source 3_Results/flight_selection/fuel_wide_m060 --out-dir /tmp/unused
# ─────────────────────────────────────────────────────────────────────────────

# Size, 2026-09-25: 2 nodes on `short` for both fleets. 10 nodes on `medium`
# (which refuses fewer than 5) waited most of a day in the queue, while 2 nodes on
# `short` started within hours. ~5 200 narrow bodies over 48 tasks is ~110 flights
# per task at ~1 min each, so --time is the partition maximum: the wall clock must
# not be what stops a sweep (it resumes if it does, but the selection job then
# never starts).
#SBATCH --nodes=2
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH --begin=now
#SBATCH --partition=short
#SBATCH --mail-user=juliana-cristina.cortez-barauna@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=fuel_only
#SBATCH -o logs/%x.%j.out # STDOUT, %x = job name; logs/ must exist (it is in git)
#SBATCH -e logs/%x.%j.err # STDERR

set -euo pipefail

# ── environment ──────────────────────────────────────────────────────────────
# conda env built once on the cluster (see README.md).
# conda-forge supplies the heavy stack (numpy/scipy/xarray/traffic/casadi/…);
# openap/opentop/fastmeteo come from a local wheelhouse (PyPI is firewalled).
source /softs/Utils/anaconda3-2023/etc/profile.d/conda.sh
conda activate "$SLURM_SUBMIT_DIR/.conda-optimal"
cd "$SLURM_SUBMIT_DIR"   # every path below is relative to the repo root

# One thread per solve so the N workers don't oversubscribe the cores. The batch
# also sets these by default, but exporting here is explicit and covers children.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── sweep parameters ─────────────────────────────────────────────────────────
# Exported so each srun task's `bash -c` (single-quoted) can still see them.
#
# FLEET is required: there is no sensible default, and guessing one would run
# the wrong half of the population at the wrong mass.
export FLEET="${FLEET:?set FLEET=narrow or FLEET=wide}"

# m0 follows from the fleet. It is deliberately NOT read from M0: stage 2 is
# submitted with `export M0=...`, and sbatch propagates the submitting shell, so a
# leftover M0 would silently solve a fleet at the other fleet's mass. FUEL_M0
# overrides it on purpose, e.g. to sweep the mass of one type (with ACTYPES).
case "$FLEET" in
    narrow) DEFAULT_M0=0.80 ;;
    wide)   DEFAULT_M0=0.60 ;;
    *) echo "FLEET=$FLEET is invalid (expected narrow or wide)" >&2; exit 1 ;;
esac
export FUEL_M0="${FUEL_M0:-$DEFAULT_M0}"

# ALT_BUFFER is "MIN MAX" in feet: the optimisation window is the real flight's
# mean cruise altitude -MIN / +MAX. The default, -1000/+1000, is the selection
# window.
#
# It is a variable rather than hard-coded because the earlier phase-1 campaigns
# each used a DIFFERENT window and nobody noticed: opentop_batch ran -1000/+1000,
# m0_060 and m0_060_wb2 ran -1000/+4000. A wider window is an easier NLP, so
# convergence rates are only comparable between runs that used the SAME window.
export ALT_BUFFER="${ALT_BUFFER:-1000 1000}"
read -r AB_MIN AB_MAX <<< "$ALT_BUFFER"

# TRACK_TOL is how far, in metres, the optimised ground track may deviate from the
# real one (batch_opentop --track-tolerance; default 10000, opentop's own is 1000). It
# decides convergence as much as the window does: at 1 km most real tracks are
# infeasible, at 10 km most converge.
export TRACK_TOL="${TRACK_TOL:-10000}"

# TRACK_MODE: "exact" (default) is TrackCruise (src/optimization/opentop/track_cruise.py),
# the solution ON the flown track, TRACK_TOL ignored; "tolerance" is
# top.Cruise.follow_track within TRACK_TOL. Every pickle records it as context['track_mode'].
export TRACK_MODE="${TRACK_MODE:-exact}"

# MULTISTART: non-empty multistarts each fuel-only solve: opentop's default guess, then
# its solution recentred on the real mean altitude + MULTISTART_ALT_OFFSETS_FT (src/
# config.py; N_STARTS uses the first N-1). Pickles get _fuel_ms, so they never collide
# with singlestart ones; ~N_STARTS x the wall clock.
export MULTISTART="${MULTISTART:-}"
export N_STARTS="${N_STARTS:-}"

# FUEL_OUT is named after the fleet and m0 only. The window and the track tolerance
# are NOT in the name: the campaign runs one configuration, and every pickle records
# both in its context (alt_buffer_ft_min/_max, track_tolerance_m). A sweep with
# other settings must not reuse this directory, or is_done() would skip the flights
# already there. Not named OUT, which is the stage-2 variable — a leftover export
# would redirect this job.
export FUEL_OUT="${FUEL_OUT:-3_Results/flight_selection_2d_cruise/fuel_${FLEET}_m${FUEL_M0/./}}"
# The mode is not in the name either: an exact sweep written over a tolerance one would
# be skipped flight by flight by is_done() and silently mix the two.
case "$TRACK_MODE:$FUEL_OUT" in
    exact:*exact*|exact:*2d_cruise*) ;;
    exact:*) echo "TRACK_MODE=exact requires its own FUEL_OUT with 'exact' or '2d_cruise' in the name; FUEL_OUT=$FUEL_OUT" >&2
             exit 1 ;;
    tolerance:*exact*|tolerance:*2d_cruise*)
             echo "TRACK_MODE=tolerance must not write into an exact folder ('exact' or '2d_cruise' in the name); FUEL_OUT=$FUEL_OUT" >&2
             exit 1 ;;
    tolerance:*) ;;
    *) echo "TRACK_MODE=$TRACK_MODE is invalid (expected tolerance or exact)" >&2; exit 1 ;;
esac

# Optional filters, both empty by default (= every flight of the fleet).
# ACTYPES narrows the fleet to named ICAO codes; it is word-split on purpose
# (--actypes takes several). FUEL_FLIGHTS is a flight-id list, to re-run a named
# subset, e.g. a handful of flights a job never reached.
export ACTYPES="${ACTYPES-}"
export FUEL_FLIGHTS="${FUEL_FLIGHTS-}"

export GRID="0_Data/flight_data/grid_era5_smoothed.parquet.gz"
# DATA is the single flight parquet (all 5 919 flights, typecodes recovered); the
# population actually run is chosen by FLEET and the filters, not by the file.
export DATA="0_Data/flight_data/eu_flights_2022feb20_persistent_all.parquet"

# The only record of what this job ran with: none of it is in the filenames.
echo "fleet=$FLEET  m0=$FUEL_M0  alt_buffer=-$AB_MIN/+$AB_MAX ft  track_mode=$TRACK_MODE  track_tol=$TRACK_TOL m  out=$FUEL_OUT"
echo "actypes=${ACTYPES:-<all>}  flights=${FUEL_FLIGHTS:-<all in DATA>}"
if [ -n "$MULTISTART" ]; then echo "multistart=on (n_starts=${N_STARTS:-5}, pickles *_fuel_ms)"; else echo "multistart=off"; fi
echo "data=$DATA"

# ── launch one task per flight-shard across the whole allocation ─────────────
# srun spawns $SLURM_NTASKS tasks. Each gets a distinct rank $SLURM_PROCID and
# handles shard PROCID/NTASKS, so together they cover every flight, disjointly.
# --workers 1 because each task already owns one core (24 tasks fill a node).
#
# Single quotes on the bash -c body are REQUIRED: they defer $SLURM_PROCID
# expansion to each task (where it is set), not the batch shell (where it isn't).
# $ACTYPES and $ALT_BUFFER are deliberately unquoted inside so they split into
# several arguments.
srun bash -c 'python 2_Run/batch_opentop.py \
    --mode fuel \
    --fleet "$FLEET" \
    --m0 "$FUEL_M0" \
    --alt-buffer $ALT_BUFFER \
    --track-tolerance "$TRACK_TOL" \
    --track-mode "$TRACK_MODE" \
    ${MULTISTART:+--multistart} \
    ${N_STARTS:+--n-starts $N_STARTS} \
    ${ACTYPES:+--actypes $ACTYPES} \
    ${FUEL_FLIGHTS:+--flights-file "$FUEL_FLIGHTS"} \
    --sigmas 2 \
    --out-dir "$FUEL_OUT" \
    --grid-path "$GRID" \
    --data-path "$DATA" \
    --shard "${SLURM_PROCID}/${SLURM_NTASKS}" \
    --workers 1 \
    --retry-failed'
