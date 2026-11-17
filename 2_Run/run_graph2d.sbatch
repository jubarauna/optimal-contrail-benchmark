#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DP + DP_MuO (2D) over the SAME flights and the SAME mass
# assumption as the OpenTOP run, so the three solvers are comparable.
#
# One job per m0, exactly as for OpenTOP — the mass is a property of the flight
# population, not of the solver, and the lists come from the same manifest.
#
# DP and DP_MuO run as SEPARATE jobs, one method each, so each job's wall clock,
# memory report and log belong to one solver. METHOD is required; the job name
# names the log (logs/<job-name>.<jobid>.out):
#
#   dos2unix 2_Run/run_graph2d.sbatch
#   export M0=0.80 && METHOD=dp     sbatch --export=ALL --job-name=dp     2_Run/run_graph2d.sbatch
#   export M0=0.80 && METHOD=dp_muo sbatch --export=ALL --job-name=dp_muo 2_Run/run_graph2d.sbatch
#   (and the same two lines with M0=0.60)
#
# OUT and FLIGHTS are no longer passed by hand — see the block below for why.
#
# Results: 3_Results/contrail_2d_cruise/DP/m_080, DP/m_060, DP_MuO/m_080, DP_MuO/m_060 — one
# folder per (method, m0), derived from METHOD and M0. Each is resumable on its own.
#
# The per-flight interpolant is cached in the node's /tmp. The two jobs do not
# share a node reliably, so DP_MuO may rebuild what DP built, or load it; each
# pickle records which as context['interpolant_cache_hit'].
# ─────────────────────────────────────────────────────────────────────────────

# MEMORY — sized for the GLOBAL interpolant (GLOBAL_INTERP=1, the default). The
# whole-domain fit is built once, offline, by 0_Data/build_global_interpolant.sbatch;
# a task only loads it and shifts it per flight, so the per-flight CasADi bspline
# build (batch_graph2d.py:222-224) that used to dominate memory no longer happens
# here. What is left is the grid parquet (~1.5 GB, GridWeather), df_cost and the
# edge tables.
#
# Measured on the 2026-10 campaign, global bspline, all three sigmas — the
# instrument.py peaks recorded in every pickle (context['*_peak_rss_mb']):
#
#   DP       3761 pickles    median 1.4 GB   p99 1.8 GB   max 2.2 GB
#   DP_MuO   11281 pickles   median 1.6 GB   p99 2.5 GB   max 6.1 GB
#
# Those peaks are SAMPLED, so a short spike can be missed (see the docstring of
# src/instrument.py). The job uses the same size as run_opentop.sbatch, 11.7 GB
# per task, ~2x the worst. Check the true per-task peaks in the /usr/bin/time -v
# report at the end of the log after the first run. If a task comes close,
# resubmit with 3 tasks per node instead of 7 (finished flights are skipped):
#   --ntasks=15 --ntasks-per-node=3 --cpus-per-task=7       (27.3 GB per task)
#
# GLOBAL_INTERP= (empty, the per-flight fit) needs the OLD sizing, not this one.
# Debug jobs on 2026-08-04 measured 6.06 GB for 3 flights x 2 methods (job
# 4000732), but flight 010154_00026_0 alone peaked at 26.65 GB and was OOM-killed
# at 27.3 GB (job 4000787); at 42.9 GB it did not OOM, but it was cut off by the
# wall clock, so its true peak is still unbounded. Cost scales with the flight's
# lat/lon/time slice volume, the same object OpenTOP builds, whose worst task
# measured 34.5 GB. Submit it with 42.9 GB per task, and with 85.8 GB per task for
# a second pass over what OOM'd:
#   --nodes=16 --ntasks=32 --ntasks-per-node=2 --cpus-per-task=11
#   --nodes=16 --ntasks=16 --ntasks-per-node=1 --cpus-per-task=22
# Anything still failing at 85.8 GB belongs on gigamem, one flight at a time.
#
# Memory and cores are TIED per node (medium: 96 GB over 24 cores = 4000 MB per
# core), so per-task memory is bought in cores: 3 cores x 3900M = 11700M.
# --mem-per-cpu must stay BELOW 4000 or it is rejected/clamped.
#
# THROUGHPUT, measured at sigma=2: DP 52-256 s per flight, DP_MuO 0.6-61 s (cheap
# unless its Pareto front explodes — 535 points on 02a0ac_00346_0), plus the long
# tail above. The table is built once and the 13 sweep points only reweight it
# (model_2D.py:795-812), so the sweep is not what costs.
#
# Sizing rule if you change it (`medium` accepts 5-16 nodes):
#   ntasks-per-node x cpus-per-task <= 24                      cores
#   ntasks-per-node x cpus-per-task x mem-per-cpu <= ~86 GB    leave room for slurmd
#   nodes x ntasks-per-node == ntasks                          or SLURM rejects the job
# 7 tasks x 3 cores x 3900M = 81900M, 21 of 24 cores, 35 tasks over 5 nodes (the
# `medium` minimum; 8 tasks would be 93.6 GB, over the slurmd margin).
#
# --time is the partition maximum and the run is RESUMABLE (is_done() skips
# finished flights), so a timeout is not a loss — resubmit the same line and it
# continues. --requeue likewise makes a node failure cost only the flights in
# flight at that moment.
#SBATCH --nodes=5
#SBATCH --ntasks=35
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --begin=now
#SBATCH --partition=medium
#SBATCH --mail-user=juliana-cristina.cortez-barauna@isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=graph2d
#SBATCH -o logs/%x.%j.out # STDOUT, %x = job name; logs/ must exist (it is in git)
#SBATCH -e logs/%x.%j.err # STDERR

set -euo pipefail

source /softs/Utils/anaconda3-2023/etc/profile.d/conda.sh
conda activate "$SLURM_SUBMIT_DIR/.conda-optimal"
cd "$SLURM_SUBMIT_DIR"   # every path below is relative to the repo root

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Unbuffered stdout: srun writes to a file, so Python block-buffers its prints and
# a job killed at TIMEOUT loses everything after the startup lines.
export PYTHONUNBUFFERED=1

# SLURM >= 22.05 stopped propagating --cpus-per-task from sbatch to srun. Without
# this, the step would launch 1 CPU per task and each task's cgroup would get
# 3900M instead of 42900M — silently undoing the whole memory fix above.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

# OUT and FLIGHTS have DEFAULTS instead of being required. They used to be
# "${OUT:?...}" / "${FLIGHTS:?...}", and on 2026-08-03 that lost two OpenTOP jobs:
# a fresh login shell had dropped the exports, $OUT expanded to nothing, FLIGHTS
# became the absolute path "/flights_m006.txt" and both jobs died at the guard
# below. The defaults are absolute so they do not depend on the submit directory.
export M0="${M0:?set M0 to the initial mass fraction}"

# FLIGHTS is DERIVED from M0, not passed separately — each mass fraction has its
# own list, so passing both by hand made it possible to pair m0=0.60 with the m008
# list and silently benchmark the wrong subpopulation. Note the lists live under
# the OpenTOP selection directory, NOT under OUT: the whole point is that the
# three solvers run over exactly the same flights.
SEL="${SEL:-3_Results/flight_selection_2d_cruise/selection}"   # written by prepare_contrails_input.sbatch
case "$M0" in
    0.80|0.8) export FLIGHTS="$SEL/flights_m008.txt"; MASS_DIR=m_080 ;;
    0.60|0.6) export FLIGHTS="$SEL/flights_m006.txt"; MASS_DIR=m_060 ;;
    *) echo "no flight list defined for M0=$M0 (expected 0.80 or 0.60)" >&2; exit 1 ;;
esac

# One method per job. METHOD is also the --methods value and the pickle suffix.
export METHOD="${METHOD:?set METHOD=dp or METHOD=dp_muo}"
case "$METHOD" in
    dp)     METHOD_DIR=DP ;;
    dp_muo) METHOD_DIR=DP_MuO ;;
    *) echo "METHOD=$METHOD is invalid (expected dp or dp_muo)" >&2; exit 1 ;;
esac

# Results go to 3_Results/contrail_2d_cruise/<method>/<mass>/, e.g. DP_MuO/m_080 — the same layout
# as OpenTOP/m_080, so each (method, m0) has its own folder.
export OUT="${OUT:-3_Results/contrail_2d_cruise/$METHOD_DIR/$MASS_DIR}"

# Before GLOBAL_INTERP below, which checks one file per sigma.
export SIGMAS="${SIGMAS:-0 1 2}"

# INTERP chooses how the cost grid is interpolated: "bspline" (default, what every
# previous campaign used) or "linear". It is not an implementation detail — it
# changes the field the solver sees, and the output file name does NOT encode it.
# Hence the guard below: a linear campaign written into a bspline directory would
# be indistinguishable from the old content, and is_done() would still skip the
# flights already present, producing a silent mix of the two.
export INTERP="${INTERP:-bspline}"
case "$INTERP" in
    bspline|linear) ;;
    *) echo "INTERP=$INTERP is invalid (expected bspline or linear)" >&2; exit 1 ;;
esac
case "$INTERP:$OUT" in
    linear:*linear*) ;;
    linear:*) echo "INTERP=linear requires its own OUT with 'linear' in the name; OUT=$OUT" >&2
              exit 1 ;;
esac
# GLOBAL_INTERP: the campaign uses the GLOBAL bspline (default "1"): one fit of the
# whole domain in ABSOLUTE time per sigma, composed per flight with a constant
# offset, instead of a per-flight fit (median 164 s, worst case 5 655 s). The field
# carries a uniform ~2e-4 residual against a per-flight fit (0.34 % mean relative).
# GLOBAL_INTERP= (empty) goes back to the per-flight fit. Not in the folder name:
# the campaign runs one configuration, and every pickle records it as
# context['interpolant_shape'] ("global_bspline" or "bspline").
#
# Requires 0_Data/flight_data/global_interp_sigma<S>.casadi + .meta.json for EACH sigma
# in the list — built by 0_Data/build_global_interpolant.sbatch.
export GLOBAL_INTERP="${GLOBAL_INTERP-1}"
# Exported: it is expanded inside the single-quoted bash -c, i.e. in the TASK's
# shell. Without export it would arrive empty there and the flag would be silently ignored.
export GI_FLAG=""
if [ -n "$GLOBAL_INTERP" ]; then
    # The global fit is a bspline (shape="bspline" in the .meta.json). Asking for
    # INTERP=linear alongside it would be asking for two incompatible things: the flag
    # wins, the field comes out bspline, and the directory would be labelled "linear"
    # with bspline content. Abort rather than lie.
    if [ "$INTERP" != "bspline" ]; then
        echo "GLOBAL_INTERP=1 requires INTERP=bspline (the global fit is a bspline); INTERP=$INTERP" >&2
        exit 1
    fi
    for S in $SIGMAS; do
        for EXT in casadi meta.json; do
            F="0_Data/flight_data/global_interp_sigma${S}.${EXT}"
            [ -s "$F" ] || { echo "GLOBAL_INTERP=1 but $F is missing" >&2; exit 1; }
        done
    done
    export GI_FLAG="--global-interp"
fi

# SWEEPS chooses which DP sweeps run: "fuel_only weighted_sum" (default) or one of
# them. The DP has no epsilon-constraint since 2026-10-19; campaigns before that also
# ran it, so their OUT directories hold a third sweep.
# The file name does not encode the choice, hence the guard below.
# Ignored by --methods dp_muo, which has no sweep.
export SWEEPS="${SWEEPS:-fuel_only weighted_sum}"
case "$SWEEPS" in
    "fuel_only weighted_sum") ;;
    *) case "$OUT" in
           *sweep*|*ws_only*|*wsonly*) ;;
           *) echo "SWEEPS=$SWEEPS requires its own OUT (with 'ws_only' or 'sweep' in the name); OUT=$OUT" >&2
              exit 1 ;;
       esac ;;
esac

export GRID="0_Data/flight_data/grid_era5_smoothed.parquet.gz"
# DATA is the single parquet with all 5 919 flights (typecodes recovered).
# The population that runs is defined
# by the flight list, not by the file. A listed flight missing from DATA does NOT
# error: the batch only prints "WARNING: N flight ids not found and skipped" and carries on.
export DATA="${DATA:-0_Data/flight_data/eu_flights_2022feb20_persistent_all.parquet}"

# SIGMAS became overridable on 2026-08-20. It was hard-coded, and the value there
# was the leftover of a partial resume, not the campaign's intent: this file said
# "0 1" and batch_graph2d said "2", while both target directories hold all three
# sigmas. Submitting without looking produced an incomplete set WITH NO ERROR. The
# default is now the whole campaign; the line echoed below is the safeguard.
# (SIGMAS itself is set above, next to METHOD.)
# Memory: the three sigmas are columns of the SAME grid file, which GridWeather
# loads only once (~1.5 GB per task). Until 23/09/2026 sigma=1 came from a second
# file and added another ~1.5 GB; the peaks measured in the header still include
# that copy.


if [ ! -s "$FLIGHTS" ]; then
    echo "missing or empty $FLIGHTS" >&2
    exit 1
fi
echo "graph-2D  method=$METHOD  m0=$M0  flights=$(wc -l < "$FLIGHTS")  out=$OUT"
echo "data=$DATA"
echo "sweeps=$SWEEPS"
echo "sigmas=$SIGMAS"
echo "interp=$INTERP${GI_FLAG:+  (GLOBAL: single fit of the whole domain, no per-flight fit)}"
# Printed before the work starts: is_done() skips each of these, so this is how much
# of the list this job will NOT do. Informational, as in run_opentop.sbatch.
echo "  pickles already in OUT for $METHOD: $(find "$OUT" -maxdepth 1 -name "*_sigma*_${METHOD}.pkl.gz" 2>/dev/null | wc -l)"

# Single quotes defer $SLURM_PROCID to each task.
#
# --sigmas takes several values (nargs="+"): "--sigmas 0 1 2" runs all three in
# one job, since the pickles are named per σ and never collide. All three σ are
# columns of one grid file, loaded once per task, so adding σ costs no grid memory.
#
# --retry-failed so a resume re-runs the error stubs: without it is_done() counts
# a stub as finished and that flight is skipped forever.
#
# /usr/bin/time -v wraps EACH task and prints its "Maximum resident set size" —
# the kernel's true, unsampled peak. This job has never been sized on a
# measurement, only on OpenTOP's, so this is how you get its real number.
#
# `|| rc=$?` because of `set -e`: one failed task would otherwise abort here and
# you would lose the memory report. The code is re-raised at the end.
rc=0
srun /usr/bin/time -v bash -c 'python 2_Run/batch_graph2d.py \
    --methods "$METHOD" \
    --sigmas $SIGMAS \
    --interp "$INTERP" $GI_FLAG \
    --sweeps $SWEEPS \
    --m0 "$M0" \
    --out-dir "$OUT" \
    --flights-file "$FLIGHTS" \
    --grid-path "$GRID" \
    --data-path "$DATA" \
    --shard "${SLURM_PROCID}/${SLURM_NTASKS}" \
    --workers 1 \
    --retry-failed' || rc=$?

# ── memory report, in the job's own log ──────────────────────────────────────
echo
echo "===================== PEAK MEMORY PER TASK ====================="
echo "true peaks (kernel, unsampled) — highest first:"
grep -h "Maximum resident set size" "logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err" \
    | awk '{printf "  %.1f GB\n", $NF/1048576}' | sort -rn | head -20 || true
echo
echo "slurm accounting (MaxRSS is sampled — may under-report the peak):"
sacct -j "$SLURM_JOB_ID" --units=G \
    --format=JobID%20,State,Elapsed,MaxRSS,MaxVMSize,ReqMem,ExitCode || true
echo "==============================================================="

# How far it got, so "finished" is distinguishable from "died quietly". One pickle
# per (flight, σ) for this job's method. progress.csv is shared by both methods.
echo
echo "==================== COVERAGE ===================="
echo "flights requested:  $(wc -l < "$FLIGHTS")"
echo "result pkl.gz:      $(find "$OUT" -maxdepth 1 -name "*_sigma*_${METHOD}.pkl.gz" | wc -l)  ($METHOD)"
if [ -s "$OUT/progress.csv" ]; then
    echo "progress.csv rows:  $(( $(wc -l < "$OUT/progress.csv") - 1 ))"
    echo "rows with error:    $(grep -c ',error,' "$OUT/progress.csv" || true)"
else
    echo "progress.csv:       MISSING — no flight finished at all"
fi
echo "srun exit code:     $rc"
echo "================================================="

exit "$rc"
