#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Assemble the input set for the contrail phase, on a compute node.
#
# Gathers one CONVERGED fuel-only warm start per flight from the two fuel-only
# sweeps (run_fuel_only.sbatch with FLEET=wide at m0=0.60 and FLEET=narrow at
# m0=0.80), then sanity-checks what it produced. select_flights refuses any
# result solved at the wrong mass for its type (src.config.WIDEBODY_TYPES).
#
# Normally submitted by submit_selection.sh, chained after both sweeps. By hand:
#   sbatch 1_Flight_Selection/prepare_contrails_input.sbatch
#
# When it finishes, launch the sweeps with SEL pointing at the new lists:
#   (SEL defaults to 3_Results/flight_selection_2d_cruise/selection)
#   export M0=0.80 && sbatch --export=ALL 2_Run/run_opentop.sbatch
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --begin=now
#SBATCH --partition=short
#SBATCH --mail-user=juliana-cristina.cortez-barauna@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=prep_contrails
#SBATCH -o logs/prep.%j.out # STDOUT; logs/ must exist (it is in git)
#SBATCH -e logs/prep.%j.err # STDERR

set -euo pipefail

source /softs/Utils/anaconda3-2023/etc/profile.d/conda.sh
conda activate "$SLURM_SUBMIT_DIR/.conda-optimal"

cd "$SLURM_SUBMIT_DIR"

RUNS="3_Results/flight_selection_2d_cruise"   # every stage-1 output lives here
# The two fuel-only sweeps to select from — the default output directories of
# run_fuel_only.sbatch for FLEET=wide and FLEET=narrow.
# Both must have been run under the SAME window and track tolerance; each pickle
# records them in its context.
WB_SRC="${WB_SRC:-$RUNS/fuel_wide_m060}"
NB_SRC="${NB_SRC:-$RUNS/fuel_narrow_m080}"
# Where the flight lists go. Deliberately a FRESH directory, and deliberately not
# named OUT or SEL: those are the stage-2 variables, and sbatch propagates the
# submitting shell, so a leftover export would silently redirect this job. It is
# also not contrails_full, which holds the lists AND results of the earlier
# campaign. Pass this directory as SEL to the stage-2 sbatch files.
LISTS_OUT="${LISTS_OUT:-$RUNS/selection}"
echo "wide:   $WB_SRC"
echo "narrow: $NB_SRC"
echo "lists → $LISTS_OUT"
# MULTISTART: non-empty when the sweeps ran with it, so their *_fuel_ms pickles are read.
MULTISTART="${MULTISTART:-}"
echo "multistart: ${MULTISTART:-off}"

# The two sources hold disjoint fleets, so their order does not decide anything.
# --ntasks=1: without it srun inherits the node's CPU count and would run this
# single-threaded assembly step once per core, all writing the same directory.
# --lists-only: the sweeps run with --mode full, which re-solves the warm start
# in-run, so the saved pickles are only a flight filter and copying them would
# achieve nothing. This writes flights.txt, manifest.csv and one list per m0.
# --rejected also writes rejected.csv: every flight that did not converge, and why.
srun --ntasks=1 python 1_Flight_Selection/select_flights.py \
    --source "$WB_SRC" \
    --source "$NB_SRC" \
    --out-dir "$LISTS_OUT" \
    --lists-only \
    ${MULTISTART:+--multistart} \
    --rejected

# ── verify ───────────────────────────────────────────────────────────────────
# The per-m0 lists are what the sweeps are submitted with; if they do not add up
# to flights.txt, some flights would silently never be optimised.
n_ids=$(wc -l < "$LISTS_OUT/flights.txt")
n_split=$(cat "$LISTS_OUT"/flights_m0*.txt | wc -l)
echo "flights.txt: $n_ids ids | per-m0 lists total: $n_split"

if [ "$n_ids" -ne "$n_split" ]; then
    echo "MISMATCH between flights.txt and the per-m0 lists" >&2
    exit 1
fi

echo
echo "next, one job per m0 (see the run_opentop.sbatch header):"
ls "$LISTS_OUT"/flights_m0*.txt
