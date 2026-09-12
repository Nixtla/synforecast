# Small six-frequency native coverage benchmark

`native_v1` now freezes the twelve feature definitions without changing the
candidate formulas. Saved candidate pilot artifacts retain their original
labels and remain accepted by the validator. Period/window settings are
explicit extraction parameters, not different feature schemas.

The first small comparison across all six M4 frequencies is complete. It uses
balanced and pretraining pools plus random SynForecast MAR. The pools cover
more held-out queries than MAR on five panels under the q90 distance criterion;
pretraining is not consistently ahead of the balanced pool. All three sources
show an Hourly gap under that criterion. These are exploratory results from
one seed and 32 evaluation queries per panel, not a stable generator ranking.

## Protocol

Uniformly sample 128 real training series per frequency, keeping at most their
last 512 observations. Split them into 64 fitting/generation-length references,
32 calibration queries, and 32 evaluation queries with no overlap. Use the
[audited extraction recipes](../feature_coverage_availability/README.md):
Yearly None/5, Quarterly 4/4, Monthly 12/12, Weekly None/10, Daily 7/7,
Hourly 24/24 (period/window). Weekly therefore does not measure annual
seasonality, and Hourly measures only the declared daily cycle.

Each source generates one series for each of the same 64 reference lengths.
Pool allocation cycles through shuffled prototypes before reusing one, with
a randomized remainder and allocation independent of reference lengths.
Pretraining uses its default balanced-plus-meta composition, not a meta-only
pool. Public preset configurations are recorded; its `seasonal_period`
argument controls included balanced variants, not every meta-generator.
Random MAR is configured with the declared extraction period. Corpus seeds
use separate offsets; all frequencies reuse the same seed schedule.

All real queries have complete features. In each panel, the balanced pool
retains 62/64 outputs, pretraining 63/64, and MAR 64/64. The exclusions are
intentional non-finite observations from the IoT failure preset. They are
recorded, not imputed or replaced. Restrict **all three sources and the real
candidate baseline to the same 61 reference IDs** that succeeded everywhere.
This preserves exact retained lengths and candidate counts. Scores are
conditional on that common complete cohort; the input failures remain in
the separate 64-draw retention denominator. Removing/adding a source could
change this benchmark's matched cohort, even though the real-fitted API
itself leaves existing scores unchanged when another corpus is projected.

Fit feature scaling, constant-column selection, and PCA on all 64 fitting
references. Full-feature distances are Euclidean on all retained standardized
features, without whitening. Calibrate q50/q90/q95 radii from the 32 calibration
queries' nearest distances to the 61 real candidates. Evaluate the 32 held-out
queries against those candidates and each synthetic source. Also report PCA
distances with their own real calibration and exploratory grid scores at
15/30/45 bins. The grid's real population is the 64 fitting references; the
distance coverage population is the separate 32 evaluation queries.

## Results

Fraction of held-out real queries covered at the **full-feature q90 radius**:

| Frequency | Balanced pool | Pretraining pool | Random MAR | Real baseline |
|---|---:|---:|---:|---:|
| Yearly | 81.2% | 87.5% | 9.4% | 90.6% |
| Quarterly | 71.9% | 68.8% | 6.2% | 87.5% |
| Monthly | 71.9% | 71.9% | 34.4% | 78.1% |
| Weekly | 75.0% | 84.4% | 0.0% | 96.9% |
| Daily | 62.5% | 65.6% | 0.0% | 90.6% |
| Hourly | 0.0% | 0.0% | 0.0% | 90.6% |

The small balanced/pretraining differences represent only one to three queries.
They should not drive preset changes. Threshold choice also matters: Hourly's
balanced pool covers 2/32 queries at q95, despite none at q90. Its median
full-feature nearest distance is 4.10, versus 1.01 for the real baseline;
pretraining's is 4.23 and MAR's 10.17. The Hourly fitting sample combines strong
seasonality with strong lag-one/seasonal dependence, but these aggregates
alone do not identify which generator mechanisms need changing.

PCA retains 67–84% of fitting-feature variance across these six panels. Both
the projection and the grid keep the limitations established by the prior
validation. Standardized distances still weight correlated features separately
and depend on the declared period, windows, lengths, reference, and sample
size. This benchmark establishes neither forecasting utility nor privacy.

**Next:** expand real samples and independent generation seeds, investigate
the Hourly gap using full-feature neighbours, and build the interpretation
notebook. Keep definitions fixed; do not tune them or presets to this one seed.

The [Hourly follow-up investigation](../feature_coverage_hourly/README.md) is
now complete on the same real query split with ten fresh generation seeds
and 60/240 matched candidates. It finds a persistent measured gap and material
trend-strength/metric sensitivity. Real-sample expansion and the broader
cross-frequency benchmark remain outstanding.

## Reproduce and artifacts

```bash
# Add --download if the public training CSVs have not been cached locally.
.venv/bin/python benchmarks/benchmark_feature_coverage.py --smoke --seed 20260913
```

`summary.json` records schema, extraction settings, source hashes, sampled
original/effective lengths, generation failures, retained/matched counts,
distance curves and quantiles, grid scores, constant-feature departures, and
runtime environment metadata. The six `*_details.json.gz` files are ignored
local artifacts containing split IDs, original generation order, pool
configurations and per-series provenance, fitted parameters, per-query
distances/nearest indices, and fitted grids. Rerunning regenerates them; no
Parquet is produced by this mode. The checkout is recorded as dirty development
state, not a released version. Earlier historical summaries are not rewritten.
