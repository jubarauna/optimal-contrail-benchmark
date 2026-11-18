#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Contrail optimisation in ONE run per flight: --mode full solves the fuel-only
# warm start and both Pareto sweeps together, so F0 (the epsilon-constraint's
# fuel budget) and the sweeps necessarily share the same altitude window, mass
# and solver settings. The saved fuel-only pickles from earlier sweeps are then
# only a flight filter — they are not read here.
#
# Submit ONE job per m0, each with its own flight list; both may share OUT
# because the lists are disjoint and every pickle records its own m0. FLIGHTS is
# derived from M0 and no longer exported by hand — see the SEL/OUT block below.
#
#   dos2unix 2_Run/run_opentop.sbatch
#   export M0=0.80 && sbatch --export=ALL 2_Run/run_opentop.sbatch
#   export M0=0.60 && sbatch --export=ALL 2_Run/run_opentop.sbatch
#
# RE-MEASURING (e.g. to get the instrument.py timing fields into the pickles)
# needs a FRESH OUT, because is_done() (batch_opentop.py:184-195) counts any
# existing error-free pickle as finished. Resubmitted over a complete OUT this
# job skips every flight and spends its whole 24 h allocation doing nothing —
# and --retry-failed does not help, it only re-runs the error stubs. SEL keeps
# pointing at the flight lists, so a fresh OUT no longer means a missing list:
#
#   export OUT=3_Results/contrail_2d_cruise/OpenTOP/m_080_retime
#   export M0=0.80 && sbatch --export=ALL 2_Run/run_opentop.sbatch
#
# By default results go to 3_Results/contrail_2d_cruise/OpenTOP/m_080 or m_060, derived from M0 —
# the same layout as DP/ and DP_MuO/.
#
# 13 solves per flight (1 fuel-only + 6 weighted-sum + 6 epsilon-constraint),
# each carrying the contrail interpolant and therefore an exact Hessian.
# ─────────────────────────────────────────────────────────────────────────────

# MEMORY — sized for the GLOBAL interpolant (GLOBAL_INTERP=1, the default). The
# whole-domain fit is built once, offline, by 0_Data/build_global_interpolant.sbatch;
# a task only loads it and shifts it per flight, so the per-flight CasADi bspline
# build that used to dominate memory no longer happens here. What is left is the
# grid parquet (~1.5 GB, GridWeather) plus IPOPT with an exact Hessian.
#
# Measured on the 2026-10 campaign, global bspline, multistart, all three sigmas —
# the instrument.py peaks recorded in every pickle (context['*_peak_rss_mb']):
#
#   OpenTOP m0=0.80   9291 pickles   median 3.8 GB   p99 4.7 GB   max 5.1 GB
#   OpenTOP m0=0.60   1992 pickles   median 3.3 GB   p99 5.1 GB   max 6.0 GB
#
# Those peaks are SAMPLED, so a short spike can be missed (see the docstring of
# src/instrument.py); 11.7 GB per task is ~2x the worst. Check the true per-task
# peaks in the /usr/bin/time -v report at the end of the log after the first run.
# If a task comes close, resubmit with 3 tasks per node instead of 7:
#   --ntasks=15 --ntasks-per-node=3 --cpus-per-task=7       (27.3 GB per task)
#
# GLOBAL_INTERP= (empty, the per-flight fit) needs the OLD sizing, not this one:
# its bspline build peaked at 34.5 GB per task (/usr/bin/time -v, jobs
# 3999869/3999870; sacct's MaxRSS said 29.15 GB, it samples every 30 s). Submit it
# on bigmem with 45 GB per task:
#   --partition=bigmem --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=6 --mem-per-cpu=7500M
#
# Memory and cores are TIED on pando (medium: 96 GB over 24 cores = 4000 MB per
# core), so per-task memory is bought in cores: 3 cores x 3900M = 11700M.
# --mem-per-cpu must stay BELOW 4000 or it is rejected/clamped.
#
# Sizing rule if you change it (`medium` accepts 5-16 nodes):
#   ntasks-per-node x cpus-per-task <= 24                      cores
#   ntasks-per-node x cpus-per-task x mem-per-cpu <= ~86 GB    leave room for slurmd
#   nodes x ntasks-per-node == ntasks                          or SLURM rejects the job
# 7 tasks x 3 cores x 3900M = 81900M, 21 of 24 cores, 35 tasks over 5 nodes (the
# `medium` minimum; 8 tasks would be 93.6 GB, over the slurmd margin). A hard per-task cap (rather than a shared node pool) means one greedy
# flight is killed alone and _run_task logs its error stub, instead of the OOM
# killer taking every task on the node with it.
#
# THROUGHPUT, same campaign, multistart, all three sigmas per flight: median ~27 min
# per flight at m0=0.80 and ~53 min at m0=0.60, with a tail of single flights
# running for many hours. The tail, not the median, decides when the job ends.
#
# --time is the partition maximum: wall clock must not be what stops this either.
# --requeue so a node failure restarts the job rather than losing it; the run is
# resumable (is_done() skips finished flights), so a requeue costs almost nothing.
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
#SBATCH --job-name=opentop_full
#SBATCH -o logs/full.%j.out # STDOUT; logs/ must exist (it is in git)
#SBATCH -e logs/full.%j.err # STDERR

set -euo pipefail

source /softs/Utils/anaconda3-2023/etc/profile.d/conda.sh
conda activate "$SLURM_SUBMIT_DIR/.conda-optimal"
cd "$SLURM_SUBMIT_DIR"   # every path below is relative to the repo root

# One thread per task: the extra 2 cores of each task exist to raise the memory
# quota (see the header), NOT to be used. Threading here would only fight for cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Unbuffered stdout: srun writes to a file, so Python block-buffers its prints and
# a job killed at TIMEOUT loses everything after the startup lines — the log then
# stops at "N flights found." with no trace of which flights ran.
export PYTHONUNBUFFERED=1

# SLURM >= 22.05 stopped propagating --cpus-per-task from sbatch to srun. Without
# this, the step would launch 1 CPU per task and each task's cgroup would get
# 7500M instead of 45000M — silently undoing the whole memory fix.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

# OUT defaults instead of being required. It used to be "${OUT:?...}", but a fresh
# login shell loses exports: $OUT expanded to nothing, FLIGHTS became the absolute
# path "/flights_m006.txt", and the job died at the guard below. The default is
# absolute so it does not depend on where sbatch was run from either. Override it
# for a different sweep (e.g. OUT=.../contrails_ms for the multistart run).
export M0="${M0:?set M0 to the initial mass fraction}"

# SEL is where the flight LISTS live; OUT is where results GO. They used to be
# the same variable — FLIGHTS was "$OUT/flights_m008.txt" — which quietly tied the
# two together: pointing OUT at a fresh directory for a re-run also pointed FLIGHTS
# at a list that was not there, and the job died at the guard below. Splitting them
# is what batch_graph2d.sbatch already does, and for the same reason: the flight
# population is a property of the SELECTION, not of the run writing the results.
#
# SEL defaults to the original run, so the existing invocations are unchanged.
SEL="${SEL:-3_Results/flight_selection_2d_cruise/selection}"   # written by prepare_contrails_input.sbatch

# TRACK_TOL: how far (m) the optimised ground track may leave the real one. It has
# to be the tolerance the flights were SELECTED with (10000 m, the stage-1 default)
# -- --mode full re-solves the fuel-only here, and a flight selected at 10 km can be
# infeasible at 1 km. Before this variable existed the stage-2 runs never passed
# --track-tolerance and silently solved at the 1000 m default. It is not in any
# directory name: the campaign runs one configuration, and every pickle records it
# as context['track_tolerance_m'].
export TRACK_TOL="${TRACK_TOL:-10000}"

# TRACK_MODE: "exact" (default) is TrackCruise (src/optimization/opentop/track_cruise.py):
# the lateral position is the distance along the track, so the solution lies ON it and
# TRACK_TOL is ignored. "tolerance" is top.Cruise.follow_track, held within TRACK_TOL
# of the flown track. The pickle names do NOT encode it, hence the guard below, as for
# INTERP=linear: an exact campaign written over a tolerance one would be skipped flight
# by flight by is_done() and silently mix the two. The guard runs once OUT is set,
# below. Every pickle records it as context['track_mode'].
export TRACK_MODE="${TRACK_MODE:-exact}"

# FLIGHTS is DERIVED from M0, not passed separately. The two are not independent —
# each mass fraction has its own flight list — so passing them by hand made it
# possible to pair m0=0.60 with the m008 list and silently run the wrong
# subpopulation. Now there is one variable to get right instead of two.
case "$M0" in
    0.80|0.8) export FLIGHTS="$SEL/flights_m008.txt"; MASS_DIR=m_080 ;;
    0.60|0.6) export FLIGHTS="$SEL/flights_m006.txt"; MASS_DIR=m_060 ;;
    *) echo "no flight list defined for M0=$M0 (expected 0.80 or 0.60)" >&2; exit 1 ;;
esac
# 3_Results/contrail_2d_cruise/OpenTOP/<mass>/, the same layout as DP/ and DP_MuO/. Multistart
# pickles carry an _ms suffix, so they can share it.
export OUT="${OUT:-3_Results/contrail_2d_cruise/OpenTOP/$MASS_DIR}"
case "$TRACK_MODE:$OUT" in
    exact:*exact*|exact:*2d_cruise*) ;;
    exact:*) echo "TRACK_MODE=exact requires its own OUT with 'exact' or '2d_cruise' in the name; OUT=$OUT" >&2
             exit 1 ;;
    tolerance:*exact*|tolerance:*2d_cruise*)
             echo "TRACK_MODE=tolerance must not write into an exact folder ('exact' or '2d_cruise' in the name); OUT=$OUT" >&2
             exit 1 ;;
    tolerance:*) ;;
    *) echo "TRACK_MODE=$TRACK_MODE is invalid (expected tolerance or exact)" >&2; exit 1 ;;
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
export SIGMAS="${SIGMAS:-0 1 2}"

# MULTISTART / N_STARTS: non-empty turns on --multistart: each solve starts from the
# fuel-only trajectory and from it shifted by the altitude offsets in src/config.py
# (MULTISTART_ALT_OFFSETS_FT; N_STARTS uses the first N-1). The fuel-only anchor is
# multistarted too (opentop's default guess, then its solution shifted the same way);
# before 2026-11-03 it was one solve. Each solve becomes n_starts
# solves, and the pickles get the _ms suffix, so they do NOT collide with the
# single-start ones and can share the same OUT. The existing multistart campaign
# is sigma=2 only:
#
#   MULTISTART=1 SIGMAS=2 OUT=.../contrails_ms
#
# Budget the wall clock by n_starts: ~5x the single-start with the default of 5.
export MULTISTART="${MULTISTART:-}"
export N_STARTS="${N_STARTS:-}"

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


# GLOBAL_INTERP: the campaign uses the GLOBAL bspline (default "1"), one fit per sigma
# in GLOBAL_INTERP_DIR, composed per flight with a constant offset instead of being
# refitted. The field carries a uniform ~2e-4 residual against a per-flight fit.
# GLOBAL_INTERP= (empty) goes back to the per-flight fit. Not in the folder name:
# every pickle records it as context['interpolant_mode'] ("global_bspline" or
# "per_flight").
#
# INTERP picks which global fit: global_interp_sigma<S> (bspline) or
# global_linear_interp_sigma<S> (linear), side by side in GLOBAL_INTERP_DIR.
# Empty = next to $GRID. A linear run still needs an OUT with 'linear' in it (above):
# the pickle names do not carry the shape.
#
#   INTERP=linear MULTISTART=1 SIGMAS=2 M0=0.80 #   OUT=3_Results/contrail/OpenTOP_linear/m_080 sbatch --export=ALL 2_Run/run_opentop.sbatch
export GLOBAL_INTERP="${GLOBAL_INTERP-1}"
export GLOBAL_INTERP_DIR="${GLOBAL_INTERP_DIR:-}"
if [ -n "$GLOBAL_INTERP" ]; then
    # Every sigma needs its file for this shape — found missing here, not after the
    # queue wait. Same naming rule as src/contrails/interpolant.py:_paths.
    if [ "$INTERP" = bspline ]; then GI_PREFIX=global; else GI_PREFIX="global_${INTERP}"; fi
    GI_DIR="${GLOBAL_INTERP_DIR:-$(dirname "$GRID")}"
    for S in $SIGMAS; do
        for EXT in casadi meta.json; do
            F="$GI_DIR/${GI_PREFIX}_interp_sigma${S}.${EXT}"
            [ -s "$F" ] || { echo "GLOBAL_INTERP=1 but $F is missing" >&2; exit 1; }
        done
    done
fi

if [ ! -s "$FLIGHTS" ]; then
    echo "missing or empty $FLIGHTS — run prepare_contrails_input.sbatch first" >&2
    exit 1
fi
echo "mode=full  m0=$M0  flights=$(wc -l < "$FLIGHTS")  out=$OUT"
echo "data=$DATA"
if [ "$TRACK_MODE" = exact ]; then TOL_ECHO="ignored"; else TOL_ECHO="$TRACK_TOL m"; fi
echo "interp=$INTERP  track_mode=$TRACK_MODE  track_tol=$TOL_ECHO"
echo "global_interp=${GLOBAL_INTERP:-off}  dir=${GLOBAL_INTERP_DIR:-<next to GRID>}"
if [ -n "$MULTISTART" ]; then MS_ECHO="on (n_starts=${N_STARTS:-5}, pickles *_ms)"; else MS_ECHO="off"; fi
echo "sigmas=$SIGMAS  multistart=$MS_ECHO"
echo "  sel=$SEL"
echo "  out=$OUT"

# How many results OUT already holds, printed BEFORE the work starts. is_done()
# skips each of these, so this number is how much of the flight list this job will
# NOT do — the difference between a resume (intended) and a re-measurement that
# silently finds everything finished (not intended). It is informational on
# purpose: reusing OUT is exactly right when resuming, so this must not be a guard.
echo "  pickles already in OUT: $(find "$OUT" -maxdepth 1 -name '*_sigma*.pkl.gz' 2>/dev/null | wc -l)"

# Single quotes defer $SLURM_PROCID to each task. --flights-file, not --flights:
# thousands of ids would not survive the quoting here.
#
# --retry-failed is REQUIRED when resuming after the OOM job: without it,
# is_done() counts an error stub as done, so every flight whose std::bad_alloc
# was caught by _run_task would be skipped forever. It also re-runs the
# truncated pickles left behind by tasks that were SIGKILLed mid-write.
#
# /usr/bin/time -v wraps EACH task and prints its "Maximum resident set size" to
# stderr when that task finishes — the kernel's true peak, which is what to size
# the next run with. Prefer it over sacct's MaxRSS: that one is SAMPLED (every
# JobAcctGatherFrequency seconds, 30 by default) and the bspline build's peak is a
# short transient it can miss entirely, i.e. sacct can UNDER-report.
#
# `|| rc=$?` because of `set -e`: one failed task would otherwise abort the script
# here and you would lose the memory report below — the one thing you need on your
# return. The code is re-raised at the very end so the job still records FAIL.
rc=0
srun /usr/bin/time -v bash -c 'python 2_Run/batch_opentop.py \
    --mode full \
    --sigmas $SIGMAS \
    --interp "$INTERP" \
    ${GLOBAL_INTERP:+--global-interp} \
    ${GLOBAL_INTERP_DIR:+--global-interp-dir "$GLOBAL_INTERP_DIR"} \
    ${MULTISTART:+--multistart} \
    ${N_STARTS:+--n-starts $N_STARTS} \
    --m0 "$M0" \
    --track-tolerance "$TRACK_TOL" \
    --track-mode "$TRACK_MODE" \
    --out-dir "$OUT" \
    --flights-file "$FLIGHTS" \
    --grid-path "$GRID" \
    --data-path "$DATA" \
    --shard "${SLURM_PROCID}/${SLURM_NTASKS}" \
    --workers 1 \
    --retry-failed' || rc=$?

# ── memory report, in the job's own log ──────────────────────────────────────
# So the numbers are waiting in logs/full.$SLURM_JOB_ID.out on your return,
# with no post-hoc command to remember. The grep pulls the 16 per-task peaks out
# of the .err; sacct gives the accounting view for cross-checking.
echo
echo "===================== PEAK MEMORY PER TASK ====================="
echo "true peaks (kernel, unsampled) — highest first:"
grep -h "Maximum resident set size" "logs/full.${SLURM_JOB_ID}.err" \
    | awk '{printf "  %.1f GB\n", $NF/1048576}' | sort -rn | head -20 || true
echo
echo "slurm accounting (MaxRSS is sampled — may under-report the peak):"
sacct -j "$SLURM_JOB_ID" --units=G \
    --format=JobID%20,State,Elapsed,MaxRSS,MaxVMSize,ReqMem,ExitCode || true
echo "==============================================================="

# How far it actually got, so you can tell "finished" from "died quietly" at a
# glance. Counts are approximate: all 16 tasks append to the one progress.csv, so
# a few header lines get interleaved. The pkl.gz count is the number that matters.
echo
echo "==================== COVERAGE ===================="
echo "flights requested:  $(wc -l < "$FLIGHTS")"
echo "result pkl.gz:      $(find "$OUT" -maxdepth 1 -name '*_sigma*.pkl.gz' | wc -l)"
if [ -s "$OUT/progress.csv" ]; then
    echo "progress.csv rows:  $(( $(wc -l < "$OUT/progress.csv") - 1 ))"
    echo "rows with error:    $(grep -c ',error,' "$OUT/progress.csv" || true)"
else
    echo "progress.csv:       MISSING — no flight finished at all"
fi
echo "srun exit code:     $rc"
echo "================================================="

exit "$rc"
