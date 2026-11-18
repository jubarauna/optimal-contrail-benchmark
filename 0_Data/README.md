# 0_Data — flight and meteo inputs

Nothing in this folder is versioned except the scripts. The benchmark reads
**one input**, the flight parquet, which lives in the Zenodo deposit (DOI: *to be
filled at submission*). Download it into `0_Data/flight_data/`, which is where
`src/paths.py` points. Everything else here is rebuilt by the scripts below.

## Where the flight parquet comes from

| Step | Script | Output |
|---|---|---|
| 1. OpenSky ADS-B, Europe, 2022-02-20 | not scripted here; taken from the figshare deposit of Sun et al., *contrail-or-not* ([10.6084/m9.figshare.29400650](https://doi.org/10.6084/m9.figshare.29400650)) | `eu_flights_2022feb20_raw.parquet.gz` |
| 2. Split into flights (30 min–8 h), Kalman filter, resample 30 s, ERA5 (ARCO, 137 levels), SAC/ISSR/`persistent` | `01_pre_process.py` of [junzis/contrail-or-not](https://github.com/junzis/contrail-or-not) | `eu_flights_2022feb20_filter_resample_meteo.parquet.gz` |
| 3. Recover missing typecodes, drop flights with any NaN in the required columns, keep flights with ≥ 1 persistent point | `0_Data/filter_benchmarking.py` (this repo) | `eu_flights_2022feb20_persistent_all.parquet` + `_summary.csv` |

```bash
python 0_Data/filter_benchmarking.py --recover-typecodes \
    --out-path 0_Data/flight_data/eu_flights_2022feb20_persistent_all.parquet
```

Step 3 fills missing `typecode`s from `traffic.data.aircraft`, which is
downloaded at run time and can change. A rebuild may therefore admit a slightly
different set than the 5 919 flights in the Zenodo parquet, which stays the
reference.

## Files

| File | Size | Origin |
|---|---:|---|
| `eu_flights_2022feb20_persistent_all.parquet` | 81 MB | **The input.** OpenSky Network, 2022-02-20, via `traffic`, with ERA5 conditions per point. 5 919 flights: those with no NaN in trajectory, wind, `persistent` or `typecode` (missing types filled from `traffic`'s aircraft database) and at least one point in persistent-contrail conditions. Built by the chain above. |
| `grid_era5_smoothed.parquet.gz` | 293 MB | ERA5 (Copernicus C3S), resampled to 1000 ft levels. One file for every σ: `persistent` is σ=0, `cost` σ=2, `cost_sigma1` σ=1. Rebuilt by `regenerate_grid.py`. |
| `global_interp_sigma2.casadi` + `.meta.json` | 176 MB | **Derived, not an input.** Rebuild with `build_global_interpolant.py`. |

ERA5 requires attribution to the Copernicus Climate Change Service; OpenSky data
is subject to the OpenSky Network terms. Neither is redistributed here.

## Scripts

```bash
python 0_Data/filter_benchmarking.py --recover-typecodes --out-path 0_Data/flight_data/eu_flights_2022feb20_persistent_all.parquet
python 0_Data/regenerate_grid.py           # ERA5 -> cost grid
python 0_Data/build_global_interpolant.py  # cost grid -> whole-domain interpolant
```

`build_global_interpolant.py` is only the CLI. The two functions the solver
batches call — `load_global_interpolant` and `shift_interpolant` — live in
`src/contrails/interpolant.py`, because a folder named `0_Data` is not a Python
identifier and cannot be imported.

A run that passes `--global-interp` is **not comparable point by point** with a
per-flight interpolant run: the single whole-domain fit carries a uniform
residual of ~2e-4 in cost units (0.34 % relative on average). Give it its own
output directory.
