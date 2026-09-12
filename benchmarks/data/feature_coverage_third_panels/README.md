# Third-panel validation across frequencies

One public Monash panel per non-hourly frequency, scored with the same
four-fold, three-seed protocol as the M4 cross-frequency benchmark, shows that
**coverage gaps are panel-specific rather than confined to Hourly**. The
unchanged pools cover tourism yearly and traffic weekly about as well as M4,
but **tourism quarterly and Australian weather daily show material gaps** of
31–48 and 13–44 points per fold, and hospital monthly a minor gap. On the
seasonal panels the real series are more strongly seasonal than their nearest
synthetic neighbours; on weather the difference is first-difference
autocorrelation and level shifts. Presets are unchanged.

## Protocol

Panels were declared before any result, chosen for size (at least 352 series
for four fresh query groups of 64) and for domains other than the M4
competition data. Hourly is not rerun: its third panel is Monash
traffic_hourly, reported with the composition rounds.

| Frequency | Panel | Series | Used | Length | Recipe (period/window) |
|---|---|---:|---:|---|---|
| Yearly | tourism_yearly | 518 | 518 | 11–47 | none/5 |
| Quarterly | tourism_quarterly | 427 | 427 | 30–130 | 4/4 |
| Monthly | hospital | 767 | 640 | 84 | 12/12 |
| Weekly | traffic_weekly | 862 | 640 | 104 | none/10 |
| Daily | weather | 3010 | 640 | 512 windows | 7/7 |

Sample at most 640 series with one new seed, keep whole series up to 512
observations and one seeded, uniformly placed 512-observation window for
longer ones (weather only). Extract `native_v1` with the same audited recipes
as the M4 benchmark; every sampled series is complete. Form four disjoint
groups of 64 fresh queries, each with 64 fitting references, 32 calibration
queries, and 256 real candidates. Synthetic draws match each fold's real
candidate lengths exactly; request 256 per source and match the first 240
shared complete slots (245–252 available). Radii are calibrated on real
candidates per fold, draw, and metric. Flags are the declared **material** (25
points) and **minor** (10 points) median gaps in at least three of four folds.
Archive hashes are recorded.

## Results

Median q90 coverage across four folds × three draws, full standardized
Euclidean distance, 240 candidates; the M4 column repeats the balanced pool's
cross-frequency median for comparison:

| Frequency | Panel | Balanced pool | Pretraining pool | Random MAR | Real baseline | Balanced on M4 | Per-fold median gap (points) | Flag |
|---|---|---:|---:|---:|---:|---:|---|---|
| Yearly | tourism | 82.8% | 82.0% | 18.8% | 90.6% | 81.2% | 10.9, 7.8, 15.6, 1.6 | none |
| Quarterly | tourism | 46.9% | 42.2% | 19.5% | 91.4% | 87.5% | 31.3, 39.1, 46.9, 48.4 | **material** |
| Monthly | hospital | 57.8% | 53.9% | 31.2% | 85.2% | 70.3% | 29.7, 17.2, 23.4, 37.5 | minor |
| Weekly | traffic | 71.1% | 70.3% | 54.7% | 84.4% | 61.7% | 9.4, 32.8, 3.1, 20.3 | none |
| Daily | weather | 47.7% | 51.6% | 14.1% | 85.2% | 56.2% | 26.6, 43.8, 12.5, 39.1 | **material** |
| Hourly (prior) | traffic | 0.0% | 0.0% | 0.0% | 85.2% | 0.8% | 90.6, 84.4, 85.9, 79.7 | material |

Tourism yearly and traffic weekly are covered as well as, or better than, the
M4 panels of the same frequency. Tourism quarterly is the sharpest change:
balanced-pool coverage falls from 87.5% on M4 Quarterly to 46.9%, with every
fold–draw pair between 34% and 59%, while the real baseline stays above 91%.
Median nearest distances are 2.5 standardized units for the pools against 1.1
for real candidates. Hospital monthly sits 27 points below its baseline and is
flagged minor; weather daily is 34–38 points below and flagged material, with
fold–draw coverage ranging from 25% to 84%. Manhattan distance and omitting
trend strength move medians by at most six points on these panels. Random MAR
covers 55% of traffic weekly queries, its best result anywhere.

The remaining distances point in two directions:

- **Seasonal strength on tourism quarterly and hospital monthly.** It is the
  largest contributor on both panels (23% and 14% of the mean squared
  distance), and real queries are more seasonal than their nearest neighbours
  by a median 0.2–0.6 standardized units. Seasonal ACF adds another 10% on
  hospital. The nearest neighbours are dominated by the multiplicative
  Holt-Winters ETS prototype (26% of tourism assignments) and the IoT sensor
  preset, so the pools contain seasonal series but too few with seasonality
  as strong and regular as these domains show.
- **First-difference autocorrelation and level shifts on weather daily.** Real
  weather queries have higher first-difference ACF (+0.3) and smaller maximum
  level shifts (−0.2) than their nearest neighbours, which are fractional
  Brownian motion and Gaussian-process series. The panel mixes rain (many
  zeros), temperatures, and solar radiation, and no single prototype family
  dominates.

## Implications

Together with the M4 results, coverage is now measured on eleven real panels.
The pools are within a few points of the real baseline on four (M4 Yearly and
Quarterly, tourism yearly, traffic weekly), 10–20 points short on four (M4
Monthly, Weekly, Daily, and traffic weekly's counterpart hospital monthly),
and 35 points or more short on three (tourism quarterly, weather daily, and
both Hourly panels). The earlier reading that the gap is frequency-specific
holds only within M4: across domains, strongly seasonal quarterly and monthly
panels and heterogeneous daily weather are also under-covered. The recurring
theme is not one frequency but **strong, regular seasonality combined with
level movement**, which the Hourly composition rounds also found. That
suggests a seasonal-strength investigation of the same kind as the Hourly
chain, on a further held-out panel, before any preset change. Production
defaults remain unchanged.

## Limits

One panel per frequency; hospital and traffic weekly have fully aligned
calendars, so folds share the same period. Twelve fold–draw measurements per
panel reuse overlapping real roles and are not independent replications; no
confidence intervals. Weather windows are seeded and random. Coverage depends
on the declared recipes, the 512-observation cap, the distance definition, and
fold radii. Forecasting utility is not measured.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/synforecast-matplotlib .venv/bin/python \
  benchmarks/benchmark_third_panels.py --download
```

Downloads five Zenodo archives (about 40 MB in total) into
`.cache/feature_coverage/monash/` and checks their hashes. `summary.json`
records the protocol, panel metadata, window offsets, splits, per-fold scores
and retention, aggregates, attribution, nearest-prototype counts, flags, and
trace hashes. The fifteen `*_details.json.gz` files hold per-query distances
and neighbours, fitted parameters, candidate IDs, and generator provenance and
configurations; they are reproducible, local, and gitignored.
