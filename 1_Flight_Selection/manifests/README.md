# Manifests

The small text artefacts that define the benchmark population. These *are*
versioned: they are kilobytes, and without them the population cannot be
reconstructed from the pickles alone. Everything large lives in the Zenodo deposit.

| File | Flights | What |
|---|---:|---|
| `flights.txt` | 3 888 | every selected flight (converged fuel-only solve at its fleet's m0) |
| `flights_m008.txt` | 3 192 | narrow bodies, m0 = 0.80 |
| `flights_m006.txt` | 696 | wide bodies, m0 = 0.60 |

Copied from `3_Results/flight_selection_2d_cruise/selection/`, written by
`prepare_contrails_input.sbatch` (`select_flights.py`) for the TrackCruise
(`TRACK_MODE=exact`) campaign.
