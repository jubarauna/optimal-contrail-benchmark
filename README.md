# Contrail-aware trajectory optimisation — benchmark code

Code and campaign documentation for the benchmark comparing four solvers for
the fuel/contrail trade-off on real European flights of 2022-02-20:
**OpenTOP singlestart**, **OpenTOP multistart**, **DP**, and **DP_MuO** (a
multi-objective label-setting solver that returns the whole Pareto front in one
solve).

No data or results are in this repository — 11.5 GB of flight parquets, meteo
grids and solver pickles live in the Zenodo deposit (DOI: 10.5281/zenodo.24344602). Each stage's README says what to download and where to put it.

## Pipeline

The numbered folders are the stages, in order. Each has its own README.

| Stage | What it does |
|---|---|
| [`0_Data/`](0_Data/) | OpenSky flights and the ERA5 cost grid; the whole-domain interpolant |
| [`1_Flight_Selection/`](1_Flight_Selection/) | Fuel-only sweeps at m0 = 0.80 and 0.60; which flights enter the benchmark |
| [`2_Run/`](2_Run/) | The four solvers, on the cluster or locally |
| [`3_Results/`](3_Results/) | Every result: `flight_selection_2d_cruise/` (stage 1) and `contrail_2d_cruise/` (stage 2), not in git |
| [`4_Post_Treatment/`](4_Post_Treatment/) | Figures, tables and the front-convexity analysis |

`src/` is the shared library the stages import — flight and meteo access, the
performance and contrail models, the two optimisers, the benchmark
configuration. It is a package because the stage folders are not: a directory
called `0_Data` is not a Python identifier and cannot appear in an `import`.

```
src/
├── paths.py         ROOT and the data/result directories, resolved once
├── config.py        the settings that define a campaign (m0, altitude window,
│                    sweep grids, the 26 excluded flights)
├── instrument.py    wall-time and peak-RSS phases, progress.csv
├── data/            flights.py  flight index and per-flight parquet cache
│                    traffic.py  FlightData (OpenSky / parquet flights)
│                    meteo.py    WeatherData (live ERA5), GridWeather (precomputed grid)
│                    route.py    great-circle and real-flight routes
├── contrails/       conditions.py   SAC / ISSR persistence
│                    cost_grid.py    prepare_grid (σ-smoothed cost grid)
│                    interpolant.py  the whole-domain cost interpolant (build / load / shift)
├── performance/     aircraft.py  OpenAP aircraft data, fuel/thrust/drag models, fuel_burn
├── optimization/    environment.py (ContrailEnvironmentFromGrid),
│                    graph_based/model_2D.py (DP, DP_MuO), opentop/
└── postprocess/     evaluation and plotting helpers
```

Every script lives exactly one level below the root and starts by putting the
root on `sys.path`, so all of them run the same way:

```bash
python 2_Run/batch_opentop.py --limit 2 --sigmas 2
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # or requirements.lock.txt, pinned
```

Python 3.12 (tested with 3.12.11). On the cluster use `requirements-cluster.txt` (the same
packages, installed offline from the prebuilt `wheelhouse_linux/`, which is built for
Python 3.11).

## Credentials

The benchmark itself needs none: it starts from the flight parquet in the Zenodo
deposit, and ERA5 is read from the public ARCO-ERA5 store without a key.

Only querying OpenSky directly needs an account (`FlightData.from_callsign` /
`from_airports` in `src/data/traffic.py`, or re-downloading raw flights). Request
access to the OpenSky Trino database, then put **your own** credentials in
pyopensky's config file, never in the code:

```ini
[default]
username = <your OpenSky username>
password = <your OpenSky password>
```

The file is created on first import; this prints where it is:

```bash
python -c "import pyopensky.config as c; print(c.opensky_config_file)"
```

## Documentation

- [`docs/BENCHMARK.md`](docs/BENCHMARK.md) — what runs per flight, where the
  time and the memory go, how to size a job.
- [`docs/problem_formulation.tex`](docs/problem_formulation.tex) — the formulation.

## Figures

*Fill this in at submission: one row per figure and table in the paper.*

| Figure | Command | Input |
|---|---|---|
| — | `python 4_Post_Treatment/plots_comparison.py …` | Zenodo: `dp_muo_run/Bspline/` |

## Third-party code and data

`openap` and `opentop` are LGPL-3.0 and are used as dependencies, not vendored.
ERA5 data requires
attribution to the Copernicus Climate Change Service. Flight data comes from the
OpenSky Network under its terms. Neither dataset is redistributed here.

## License

The code is released under the [MIT License](LICENSE).
Copyright (c) 2026 ISAE-SUPAERO. Author: Juliana Cristina Cortez Barauna.
