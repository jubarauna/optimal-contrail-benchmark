# Cluster run checklist

How to use: copy this file for each submission

    cp docs/CLUSTER_CHECKLIST.md logs/checklist_$(date +%F)_<what>.md

then edit the copy (nano / vim):
  - tick a box:      [ ]  ->  [x]
  - fill a value:    replace the ____
  - pick an option:  keep the one you chose, delete the others

The SUBMIT blocks (sections 1 and 2) are shell code: fill them, then paste the
whole block into the cluster terminal, from the repo root.
  - to keep a variable's default, DELETE its line (or put # in front of it)
List what is still unticked with:  grep -n -- '- \[ \]' <file>

Why: nothing below is in the result file names, and is_done() skips every flight
already in an output directory, so a wrong setting never errors - it silently mixes
or skips results.

    RUN NAME  = ____
    DATE      = ____

---

## 0. Every run: code, environment, data

- [ ] Local changes committed and pushed; nothing relevant left in `git status`.
- [ ] dos2unix run on every .sbatch / .sh edited on Windows.
- [ ] Env activated in the login shell before any `python` command below (the
      bare `python` there is Python 2 and fails with SyntaxError):

        source /softs/Utils/anaconda3-2023/etc/profile.d/conda.sh
        conda activate ./.conda-optimal       # from the repo root
        python --version                      # must print 3.x

- [ ] Data present:
      0_Data/flight_data/eu_flights_2022feb20_persistent_all.parquet
      0_Data/flight_data/grid_era5_smoothed.parquet.gz

        ALT_BUFFER_FT_MIN / MAX = ____ / ____
        ALT_STEP_FT             = ____
        M0 / M0_WIDEBODY        = ____ / ____
        TRACK_TOLERANCE_M       = ____

---

## 1. Flight selection

### SUBMIT (fill, then paste)

    optimal-contrail-benchmark          # your clone: ____
    unset TRACK_MODE TRACK_TOL ALT_BUFFER RES MULTISTART N_STARTS OPTS
    # submit_selection.sh reads ONLY these; it unsets everything else itself.
    export TRACK_MODE=____    # exact | tolerance     default exact
                              #   exact = TrackCruise, the solution ON the flown track
    export TRACK_TOL=____     # metres                default 10000 (ignored if exact)
                              #   stage 2 must use the SAME value
    export ALT_BUFFER="____ ____"   # "MIN MAX" ft    default "1000 1000"
    export RES=____           # results root          default 3_Results/flight_selection
                              #   NEW directory for any new configuration;
                              #   exact: must contain 'exact' or '2d_cruise'
    # export MULTISTART=1     # multistart each fuel-only solve (pickles *_fuel_ms); ~5x the wall clock
    # export N_STARTS=____    # with MULTISTART; default 5
    bash 1_Flight_Selection/submit_selection.sh

Resources: fixed in run_fuel_only.sbatch (2 nodes x 24 tasks, short, 24 h) and
prepare_contrails_input.sbatch (1 task, short, 30 min).

### Checks

- [ ] The summary printed by the script matches the block above (window, mode,
      tolerance) and shows 3 job ids.
- [ ] Both sweeps finished without TIMEOUT. If not, the selection job shows
      DependencyNeverSatisfied: cancel it, resubmit the printed sweep line (it
      resumes), then run the printed selection line by hand.
- [ ] Audit shows no MIXED; m0 is 0.8 / 0.6; track_mode, track_tolerance_m and
      alt_buffer_* are the values above:

        python 2_Run/audit_out.py $RES/fuel_narrow_m080 $RES/fuel_wide_m060

- [ ] Converged flights (the rest are in $RES/selection/rejected.csv):

        NARROW = ____     WIDE = ____

---

## 2. Contrail campaign

One OpenTOP job and two graph-2D jobs (dp, dp_muo) PER m0: 6 jobs for 0.80 + 0.60.
M0, SEL, SIGMAS, INTERP and GLOBAL_INTERP must be the SAME in all of them.

- [ ] SEL is the selection made in section 1 for this campaign
      (flight lists: $SEL/flights_m008.txt and $SEL/flights_m006.txt).
- [ ] Global interpolant files exist for EVERY sigma and the chosen shape, built by
      0_Data/build_global_interpolant.sbatch:
      0_Data/flight_data/global_interp_sigma<S>.casadi + .meta.json          (bspline)
      0_Data/flight_data/global_linear_interp_sigma<S>.casadi + .meta.json   (linear)
- [ ] CAVEAT linear: run_opentop.sbatch accepts INTERP=linear with GLOBAL_INTERP=1
      (global_linear_* files); run_graph2d.sbatch refuses it and needs GLOBAL_INTERP=
      (a per-flight fit). The solvers would then see different fields. Decided: ____
- [ ] CAVEAT GLOBAL_INTERP_DIR: only run_opentop.sbatch reads it; run_graph2d.sbatch
      always loads from 0_Data/flight_data/. Keep the files there.

### 2a. OpenTOP - SUBMIT (fill, then paste; once per m0)

    optimal-contrail-benchmark
    unset M0 SEL OUT FLIGHTS DATA SIGMAS INTERP GLOBAL_INTERP GLOBAL_INTERP_DIR \
          TRACK_MODE TRACK_TOL MULTISTART N_STARTS METHOD SWEEPS OPTS
    export M0=____            # 0.80 | 0.60           REQUIRED; picks the flight list
    export SEL=____           # flight lists          default 3_Results/flight_selection_2d_cruise/selection
    export OUT=____           # results               default 3_Results/contrail_2d_cruise/OpenTOP/m_080 (or m_060)
                              #   NEW directory per configuration; a resume reuses it
    export TRACK_MODE=____    # exact | tolerance     default exact; SAME as the selection
                              #   exact: OUT must contain 'exact' or '2d_cruise'
    export TRACK_TOL=____     # metres                default 10000; SAME as the selection
    export SIGMAS="____"      # e.g. "0 1 2" | "2"    default "0 1 2"
    export INTERP=____        # bspline | linear      default bspline
                              #   linear: OUT must contain 'linear'
    export GLOBAL_INTERP=____ # 1 = global fit (default); "export GLOBAL_INTERP=" (empty) = per-flight fit
    # export GLOBAL_INTERP_DIR=____   # only if the global files are not next to the grid
    # export MULTISTART=1             # on: pickles get _ms; ~5x the wall clock
    # export N_STARTS=____            # with MULTISTART; default 5
    # export DATA=____                # default 0_Data/flight_data/eu_flights_2022feb20_persistent_all.parquet
    sbatch --export=ALL 2_Run/run_opentop.sbatch

- [ ] Submitted for M0=0.80 (job ____) and M0=0.60 (job ____).
- [ ] Stage 2 uses config.py's ALT_BUFFER_FT_* (--alt-buffer is not passed), NOT the
      selection's ALT_BUFFER. Intended.

### 2b. DP / DP_MuO - SUBMIT (fill, then paste; once per method and m0)

    cd ~/optimal-contrail-benchmark
    unset M0 SEL OUT FLIGHTS DATA SIGMAS INTERP GLOBAL_INTERP GLOBAL_INTERP_DIR \
          TRACK_MODE TRACK_TOL MULTISTART N_STARTS METHOD SWEEPS OPTS
    export M0=____            # 0.80 | 0.60           REQUIRED; SAME as OpenTOP
    export METHOD=____        # dp | dp_muo           REQUIRED
    export SEL=____           # SAME as OpenTOP       default 3_Results/flight_selection_2d_cruise/selection
    export OUT=____           # results               default 3_Results/contrail_2d_cruise/DP/m_080, DP_MuO/m_060, ...
    export SIGMAS="____"      # SAME as OpenTOP       default "0 1 2"
    export INTERP=____        # SAME as OpenTOP       default bspline (linear needs GLOBAL_INTERP=)
    export GLOBAL_INTERP=____ # SAME as OpenTOP       default 1; files read from 0_Data/flight_data/
    # export SWEEPS="____"            # default "fuel_only weighted_sum" (dp only);
    #                                 # anything else: OUT must contain 'ws_only' or 'sweep'
    # export DATA=____                # default as for OpenTOP
     sbatch --export=ALL --job-name=$METHOD 2_Run/run_graph2d.sbatch

- [ ] Submitted: dp 0.80 (job ____), dp_muo 0.80 (job ____),
                 dp 0.60 (job ____), dp_muo 0.60 (job ____).

---

## 3. Right after submitting: the first lines of each log

    head -15 logs/full.<jobid>.out        (OpenTOP)
    head -15 logs/dp.<jobid>.out          (DP; dp_muo likewise)
    head -5  logs/fuel_narrow.<jobid>.out (selection)

- [ ] m0, flights=N (the list's length), out=, sel=, data= match the SUBMIT block.
- [ ] interp=, global_interp=, track_mode=, track_tol=, sigmas=, multistart=, sweeps=
      match the SUBMIT block.
- [ ] "pickles already in OUT:" is 0 for a new run, or the expected count for a
      resume. (A re-measurement showing the full count will do nothing for 24 h.)
- [ ] No "WARNING: N flight ids not found and skipped".

---

## 4. When it finishes

- [ ] COVERAGE block at the end of the log: result pkl.gz = flights x sigmas.

        ROWS WITH ERROR = ____

- [ ] "srun exit code: 0", or the failures are understood. TIMEOUT / requeue:
      paste the same SUBMIT block again (it resumes).
- [ ] PEAK MEMORY block: the highest peak is well below the 11.7 GB per task.

        HIGHEST PEAK = ____ GB

- [ ] Audit across the three solvers shows no MIXED, the same m0, sigmas and alt
      buffers in all three, OpenTOP's interpolant_mode equal to DP's
      interpolant_shape, and track_mode / track_tolerance_m as intended:

        python 2_Run/audit_out.py <OpenTOP OUT> <DP OUT> <DP_MuO OUT>

- [ ] This file, the commit hash and the job ids kept next to the results.
