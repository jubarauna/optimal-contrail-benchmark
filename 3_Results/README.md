# 3_Results — every result the pipeline writes

Output only: no scripts live here, and nothing below this README is in git
(the pickles go to the Zenodo deposit).

```
3_Results/
├── flight_selection_2d_cruise/  stage 1, 1_Flight_Selection/
│   ├── fuel_narrow_m080/        fuel-only sweep, narrow bodies, m0 = 0.80
│   ├── fuel_wide_m060/          fuel-only sweep, wide bodies,   m0 = 0.60
│   └── selection/               flights.txt, flights_m008.txt, flights_m006.txt,
│                                manifest.csv, rejected.csv
└── contrail_2d_cruise/          stage 2, 2_Run/
    ├── DP/        m_060/  m_080/
    ├── DP_MuO/    m_060/  m_080/
    └── OpenTOP/   m_060/  m_080/
```

The folder names carry neither the altitude window nor the track tolerance: the
campaign runs one configuration, and every pickle records both in its context.
Running with other settings needs these folders moved away first, or the
finished flights are skipped.
