# Multi-seed cross-frequency coverage on fresh queries

With three generation seeds, four disjoint query groups, and 240 matched
candidates per panel, **no M4 frequency other than Hourly shows a material
coverage gap** for the unchanged balanced and pretraining pools. Yearly and
Quarterly match the real baseline within a few points; Monthly, Weekly, and
Daily sit 5–20 points below it, with Weekly flagged as a minor gap and Daily
split between two folds with large gaps and two with small ones. The Hourly
gap of 83–89 points is therefore frequency-specific, not a general pool
deficiency. Random MAR is far behind on every panel. Presets are unchanged.

## Protocol

Sample 640 training series per frequency (all 359 for Weekly) with one new
seed, keep at most the last 512 observations, and extract `native_v1` with the
[audited recipes](../feature_coverage_availability/README.md): Yearly none/5,
Quarterly 4/4, Monthly 12/12, Weekly none/10, Daily 7/7 (period/window). All
sampled real rows are complete. Nonseasonal recipes make seasonal strength and
seasonal ACF constant, so those panels fit ten variable features.

Form four disjoint groups of 64 fresh evaluation queries; each fold selects
256 real candidates and 32 calibration queries outside its evaluation group,
with 64 of the candidates fitting the scaler. Because lengths vary, synthetic
draws match each fold's real candidate lengths exactly and are fold-specific:
fold seeds are the generation seed plus 1000 × fold. Request 256 series per
source, take the first 240 shared complete slots in candidate order (244–249
were available in every fold), and match real candidates one-to-one. Radii
are calibrated on real candidates per fold, draw, and metric.

Two declared flags on the median over seeds of the real-minus-best-source
full-feature q90 gap, each requiring three of four folds: **material** at 25
points (the Hourly persistence gate) and **minor** at 10 points. Both are
operational triggers, not significance tests. Hourly is not rerun; its row
below comes from the [fresh-split validation](../feature_coverage_hourly_splits/README.md).

## Results

Median q90 coverage across four folds × three draws, full standardized
Euclidean distance, 240 candidates:

| Frequency | Balanced pool | Pretraining pool | Random MAR | Real baseline | Per-fold median gap (points) | Flag |
|---|---:|---:|---:|---:|---|---|
| Yearly | 81.2% | 79.7% | 9.4% | 85.2% | 1.6, −1.6, −1.6, 12.5 | none |
| Quarterly | 87.5% | 85.9% | 34.4% | 90.6% | 3.1, 3.1, 9.4, −1.6 | none |
| Monthly | 70.3% | 72.7% | 14.8% | 85.9% | 4.7, 15.6, 14.1, 9.4 | none |
| Weekly | 61.7% | 57.8% | 1.6% | 79.7% | 12.5, 17.2, 7.8, 32.8 | minor |
| Daily | 56.2% | 51.6% | 0.0% | 81.2% | 42.2, 9.4, 34.4, 7.8 | none |
| Hourly (prior) | 0.8% | 1.6% | 0.0% | 90.6% | 89.1, 89.1, 85.9, 82.8 | material |

Balanced-pool coverage ranges across the twelve fold–draw pairs from 64–92%
(Yearly), 66–98% (Quarterly), 67–88% (Monthly), 36–84% (Weekly), and 20–86%
(Daily). Manhattan distance moves the medians by up to nine points, and
omitting trend strength lowers pool coverage by up to ten points on Yearly and
Weekly rather than raising it, unlike Hourly, where trend strength dominated
the gap. Median nearest-neighbour distances are 0.8–1.5 standardized units for
the pools against 0.5–0.9 for real candidates, versus about 4 for Hourly.

The two pools are within five points of each other on every panel, and the
pretraining pool is slightly lower on Weekly and Daily; these differences are
a few queries and do not favour either preset. Random MAR reaches 34% only on
Quarterly.

Daily is the panel to watch. Two folds have median gaps above 25 points and
two below 10, so the flag is not raised, but coverage depends strongly on which
real series serve as queries. On Daily and Weekly, the nearest pool neighbours
are dominated by the fractional Brownian motion (41% of Daily assignments) and
Lévy prototypes, and the remaining distance is spread across first-difference
ACF, crossing points, seasonal strength, and spectral entropy, with no
dominant coordinate. On Monthly, seasonal strength leads the remaining distance
(real queries are more seasonal than their neighbours by a median 0.17
standardized units), followed by first-difference ACF. These are distance
decompositions on the twelve fold–draw pairs, not causal estimates.

## Implications

Within M4, the Hourly gap addressed by the investigation, fresh splits, and
two composition rounds is specific to that frequency. The
[third-panel validation](../feature_coverage_third_panels/README.md) qualifies
this: on Monash panels from other domains, tourism quarterly and weather daily
show material gaps and hospital monthly a minor one, so gaps are
panel-specific rather than confined to Hourly, with strong regular seasonality
as a recurring theme. Any preset response should be evaluated on held-out
panels rather than on M4 alone. Production defaults remain unchanged.

## Limits

Each panel's twelve measurements reuse overlapping real roles across folds and
are not independent replications; there are no confidence intervals. Draws are
fold-specific, so seed and split variation are not separated as they were for
Hourly. The Weekly panel uses all 359 series, so its folds share most
candidates. Coverage depends on the declared recipes, the 512-observation cap,
the distance definition, and fold radii. Forecasting utility is not measured.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/synforecast-matplotlib .venv/bin/python \
  benchmarks/benchmark_cross_frequency.py
```

Requires the cached M4 training CSVs (pass `--download` to fetch them).
`summary.json` records the protocol, recipes, sampled and complete counts,
splits, per-fold scores and retention for every source, aggregate medians and
ranges, attribution summaries, nearest-prototype counts, flags, and trace
hashes. The fifteen `*_details.json.gz` files hold per-query distances and
neighbours, fitted parameters, candidate IDs, and generator provenance,
configurations, and failures; they are reproducible, local, and gitignored.
