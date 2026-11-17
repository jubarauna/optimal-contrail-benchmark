#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Flight selection in one command, run on the LOGIN node (not with sbatch):
#
#   bash 1_Flight_Selection/submit_selection.sh
#
# Submits three jobs:
#   1. fuel-only sweep, narrow bodies (m0=0.80)   run_fuel_only.sbatch, FLEET=narrow
#   2. fuel-only sweep, wide bodies   (m0=0.60)   run_fuel_only.sbatch, FLEET=wide
#   3. the selection, which writes the flight lists once 1 AND 2 have finished
#      without error                              prepare_contrails_input.sbatch
#
# The altitude window and the track tolerance are set HERE, once, and handed to all
# three jobs, so the two fleets and the selection cannot disagree about them.
# Defaults -1000/+1000 ft and 10000 m; others go in front of the command, e.g.
#   TRACK_TOL=5000 bash 1_Flight_Selection/submit_selection.sh
# Neither is in the directory names: the campaign runs ONE configuration, and each
# result records its own in its context. A run with other settings needs the old
# 3_Results/flight_selection/ moved away first, or is_done() skips the flights already there
# -- or its own RES (below).
#
# If a sweep hits its time limit, the selection never starts (squeue shows it as
# DependencyNeverSatisfied). Cancel it, resubmit the sweep with the line printed
# below (it resumes), then submit the selection by hand with the printed line.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# sbatch records the directory it is called from as SLURM_SUBMIT_DIR, and every
# job cds there, so submit from the repo root whatever the caller's cwd.
cd "$(dirname "$0")/.."

# Leftovers from an earlier session would reach the jobs: sbatch propagates this
# shell's environment. Only ALT_BUFFER, TRACK_TOL, TRACK_MODE, MULTISTART, N_STARTS
# and RES are meant to come from outside.
unset M0 OUT SEL FLIGHTS FLEET FUEL_M0 FUEL_OUT FUEL_FLIGHTS ACTYPES \
      WB_SRC NB_SRC LISTS_OUT

ALT_BUFFER="${ALT_BUFFER:-1000 1000}"
read -r AB_MIN AB_MAX <<< "$ALT_BUFFER"
TRACK_TOL="${TRACK_TOL:-10000}"
# Where the two sweeps and the lists go. Override it to run a new configuration
# beside an old one instead of moving the old one away, e.g.
#   RES=3_Results/flight_selection_tol1000_v2 TRACK_TOL=1000 bash 1_Flight_Selection/submit_selection.sh
RES="${RES:-3_Results/flight_selection_2d_cruise}"
# TRACK_MODE: "exact" (default) is TrackCruise, the solution ON the track (TRACK_TOL is
# ignored); "tolerance" is top.Cruise held within TRACK_TOL of the flown track, e.g.
#   RES=3_Results/flight_selection_tol10000 TRACK_MODE=tolerance bash 1_Flight_Selection/submit_selection.sh
# Not in the directory names either, so a tolerance selection needs its own RES.
TRACK_MODE="${TRACK_MODE:-exact}"
# MULTISTART: non-empty multistarts every fuel-only solve (pickles *_fuel_ms), e.g.
#   RES=3_Results/flight_selection_ms MULTISTART=1 bash 1_Flight_Selection/submit_selection.sh
# N_STARTS: starts per solve with MULTISTART (default 5).
MULTISTART="${MULTISTART:-}"
N_STARTS="${N_STARTS:-}"
case "$TRACK_MODE:$RES" in
    exact:*exact*|exact:*2d_cruise*) ;;
    exact:*) echo "TRACK_MODE=exact requires its own RES with 'exact' or '2d_cruise' in the name; RES=$RES" >&2
             exit 1 ;;
    tolerance:*exact*|tolerance:*2d_cruise*)
             echo "TRACK_MODE=tolerance must not write into an exact folder ('exact' or '2d_cruise' in the name); RES=$RES" >&2
             exit 1 ;;
    tolerance:*) ;;
    *) echo "TRACK_MODE=$TRACK_MODE is invalid (expected tolerance or exact)" >&2; exit 1 ;;
esac
NB_OUT="$RES/fuel_narrow_m080"
WB_OUT="$RES/fuel_wide_m060"
LISTS="$RES/selection"

# SLURM does not create the log directory; a job whose -o path is missing dies
# without writing a single line. logs/ is in git, but a clone made before it was
# added would not have it.
mkdir -p logs

# A fresh clone has neither: both are outside git. Failing here costs nothing;
# failing inside the job costs the queue wait.
for need in .conda-optimal \
            0_Data/flight_data/eu_flights_2022feb20_persistent_all.parquet \
            0_Data/flight_data/grid_era5_smoothed.parquet.gz; do
    [ -e "$need" ] || { echo "missing $need (see the README)" >&2; exit 1; }
done

# --parsable prints "<jobid>" or "<jobid>;<cluster>"; keep the id.
submit() { sbatch --parsable "$@" | cut -d';' -f1; }

# Both fleets run at the #SBATCH size of run_fuel_only.sbatch (2 nodes, `short`,
# 24 h); only the job name differs.
NB_ARGS=(--job-name=fuel_narrow)
WB_ARGS=(--job-name=fuel_wide)

MS_VARS="MULTISTART=$MULTISTART N_STARTS=$N_STARTS"
NB_JOB=$(FLEET=narrow ALT_BUFFER="$ALT_BUFFER" TRACK_TOL="$TRACK_TOL" TRACK_MODE="$TRACK_MODE" \
         MULTISTART="$MULTISTART" N_STARTS="$N_STARTS" \
         FUEL_OUT="$NB_OUT" submit "${NB_ARGS[@]}" 1_Flight_Selection/run_fuel_only.sbatch)
WB_JOB=$(FLEET=wide ALT_BUFFER="$ALT_BUFFER" TRACK_TOL="$TRACK_TOL" TRACK_MODE="$TRACK_MODE" \
         MULTISTART="$MULTISTART" N_STARTS="$N_STARTS" \
         FUEL_OUT="$WB_OUT" submit "${WB_ARGS[@]}" 1_Flight_Selection/run_fuel_only.sbatch)
SEL_JOB=$(NB_SRC="$NB_OUT" WB_SRC="$WB_OUT" LISTS_OUT="$LISTS" MULTISTART="$MULTISTART" \
          submit --dependency=afterok:${NB_JOB}:${WB_JOB} \
          1_Flight_Selection/prepare_contrails_input.sbatch)

cat <<EOF
window  -${AB_MIN}/+${AB_MAX} ft   track mode  ${TRACK_MODE}   track tolerance  ${TRACK_TOL} m
multistart  $([ -n "$MULTISTART" ] && echo "on, n_starts=${N_STARTS:-5}" || echo off)

  $NB_JOB  narrow bodies, m0=0.80  → $NB_OUT
  $WB_JOB  wide bodies,   m0=0.60  → $WB_OUT
  $SEL_JOB  selection, after both   → $LISTS

To resubmit one sweep (it resumes where it stopped):
  FLEET=narrow ALT_BUFFER="$ALT_BUFFER" TRACK_TOL=$TRACK_TOL TRACK_MODE=$TRACK_MODE $MS_VARS FUEL_OUT=$NB_OUT sbatch ${NB_ARGS[*]} 1_Flight_Selection/run_fuel_only.sbatch
  FLEET=wide ALT_BUFFER="$ALT_BUFFER" TRACK_TOL=$TRACK_TOL TRACK_MODE=$TRACK_MODE $MS_VARS FUEL_OUT=$WB_OUT sbatch ${WB_ARGS[*]} 1_Flight_Selection/run_fuel_only.sbatch
To run the selection by hand:
  NB_SRC=$NB_OUT WB_SRC=$WB_OUT LISTS_OUT=$LISTS MULTISTART=$MULTISTART sbatch 1_Flight_Selection/prepare_contrails_input.sbatch
EOF
