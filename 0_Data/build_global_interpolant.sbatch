#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build the whole-domain interpolant (bspline, or linear with SHAPE) for sigma = 0, 1 and 2, from the
# current grid (grid_era5_smoothed.parquet.gz), before the contrail campaign:
#
#   dos2unix 0_Data/build_global_interpolant.sbatch
#   sbatch 0_Data/build_global_interpolant.sbatch
#
# SIGMAS picks a subset (default "0 1 2"), e.g. sigma=2 first so its campaign can
# start while the others build. Two builds may run at once: they write different files.
#   SIGMAS=2   sbatch 0_Data/build_global_interpolant.sbatch
#   SIGMAS="0 1" sbatch 0_Data/build_global_interpolant.sbatch
#
# SHAPE picks the interpolant (default bspline). The linear one is written beside the
# bspline, as global_linear_interp_sigma<S>, so the two never overwrite each other:
#   SHAPE=linear sbatch 0_Data/build_global_interpolant.sbatch
#
# Writes 0_Data/flight_data/global_interp_sigma<S>.casadi + .meta.json. --rebuild
# replaces any earlier file: the sigma=2 one on disk was fitted to an older grid
# file (grid_era5_smoothed_1000ft), and every sigma of the campaign must come from
# the grid the flights are sliced from. Each .meta.json records the fit's build_s.
#
# Size: the fit is over all 5.5M grid nodes, against 1.6M for the largest
# per-flight slice, which peaked at 34.5 GB. Hence a whole bigmem node for one
# task; the three sigmas run one after the other, so the peak is one fit's.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=24:00:00
#SBATCH --partition=bigmem
#SBATCH --mail-user=juliana-cristina.cortez-barauna@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=global_interp
#SBATCH -o logs/%x.%j.out # STDOUT; logs/ must exist (it is in git)
#SBATCH -e logs/%x.%j.err # STDERR

set -euo pipefail

source /softs/Utils/anaconda3-2023/etc/profile.d/conda.sh
conda activate "$SLURM_SUBMIT_DIR/.conda-optimal"
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SIGMAS="${SIGMAS:-0 1 2}"
SHAPE="${SHAPE:-bspline}"
echo "sigmas=$SIGMAS shape=$SHAPE"
/usr/bin/time -v python 0_Data/build_global_interpolant.py --sigma $SIGMAS --shape "$SHAPE" --rebuild

# Same naming rule as src/contrails/interpolant.py:_paths.
if [ "$SHAPE" = bspline ]; then PREFIX=global; else PREFIX="global_${SHAPE}"; fi
echo
for S in $SIGMAS; do
    ls -l "0_Data/flight_data/${PREFIX}_interp_sigma${S}.casadi"
    grep -h '"build_s"\|"cost_col"\|"grid_path"\|"shape"' "0_Data/flight_data/${PREFIX}_interp_sigma${S}.meta.json"
done
