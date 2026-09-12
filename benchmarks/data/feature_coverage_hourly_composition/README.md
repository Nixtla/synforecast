# Native generator composition for Hourly coverage

Existing public generators, configured (not modified) to combine a daily cycle
with slow level changes and amplitude variation, **reproduce the diagnostic
controls' native-feature gains on the fresh M4 Hourly splits**. They do not
reproduce the controls' daily-profile shape gain. On an independent Hourly
panel (Monash traffic_hourly), the existing pools show the same gap, and the
composition improves coverage in every paired comparison, but the gain is
small in three of four folds and **fails the declared transfer criterion**.
Production presets, generators, and feature definitions are unchanged.

## Protocol

Reuse the [fresh-split validation](../feature_coverage_hourly_splits/README.md)
exactly: the same four disjoint 64-query groups, 64 fitting references, 32
calibration queries, 256 real candidates per fold, `native_v1` with period 24
and window 24, trailing 512 observations, and the same three generation seeds.
The existing pools and random MAR are regenerated and their recorded fold
scores are replayed bit-for-bit before the composition is scored. The
composition uses the same 240 retained slots and real candidates as its paired
existing draw; it completed all 256 requested slots in every seed, so no slot
was replaced. Radii are calibrated on real candidates per fold, draw, and metric.

The composition corpus is eight prototypes cycled evenly across slots (32
each), each a public class with non-default settings declared before any
result was inspected. The seed offset differs from every existing corpus.

| Prototype | Class | Composition |
|---|---|---|
| tsi_daily_shaped | TSIGenerator | one period-24 harmonic, 2f/3f overtones, slow amplitude envelope, drifting trend, noise 2–40% of signal |
| tsi_daily_weekly | TSIGenerator | two harmonics from periods {24, 168}, overtones and envelopes with probability 0.5, drifting trend |
| ets_AAdM_24 | ETSGenerator | ETS(A,Ad,M), period 24; random-walk level, damped trend, multiplicative season |
| ets_MAdM_24 | ETSGenerator | ETS(M,Ad,M), period 24 |
| seasonal_trend_breaks | SeasonalGenerator | sine, period 24, two random-location slope changes |
| seasonal_drift_level_breaks | SeasonalGenerator | sine, period 24, linear drift, two random-location level shifts |
| energy_residential_drift | EnergyLoadGenerator | residential daily/weekly profile plus one slope change |
| energy_commercial_drift | EnergyLoadGenerator | commercial daily/weekly profile plus one slope change |

Transfer panel: 862 hourly freeway occupancy series (San Francisco Bay Area,
2015–2016; Lai et al., 2017) from the Monash Forecasting Repository, Zenodo
record 4656132, with the archive hash recorded. Each series contributes one
seeded, uniformly placed 512-observation window, so the panel is not tied to a
single calendar interval. The same four-fold protocol is applied with no
exclusions. Because generation depends only on length and seed, the identical
synthetic draws and slots are scored against both real panels.

Declared criterion, per panel: the median over draws of the composition's
full-feature q90 coverage minus the best existing source's must reach 25
percentage points in at least three of four folds. This reuses the persistence
gate's scale and is an operational trigger, not a significance test.

## M4 Hourly: fresh splits

Median q90 coverage across four folds × three draws, 240 candidates:

| Source | Full features | Manhattan | Omit trend strength | Daily-profile shape |
|---|---:|---:|---:|---:|
| Balanced pool | 0.8% | 18.0% | 43.8% | 68.8% |
| Pretraining pool | 1.6% | 18.8% | 44.5% | 68.8% |
| Random MAR | 0.0% | 0.0% | 0.0% | 0.0% |
| **Native composition** | **67.2%** | **66.4%** | **71.9%** | 68.0% |
| Shaped control (prior run) | 68.0% | 66.4% | 68.8% | 75.8% |
| Matched real baseline | 90.6% | 90.6% | 91.4% | 89.8% |

Per-fold median gains over the best existing source are 76.6, 53.1, 67.2, and
53.1 points, so the criterion passes in all four folds. Paired against the best
existing source, the composition raises full-feature q90 coverage by a median
59.4 points (range 12.5–76.6; higher in all twelve comparisons), Manhattan by
45.3, and coverage without trend strength by 30.5 points. The nearest
composition neighbour is a median 1.87 standardized units closer than the
nearest existing neighbour. Full-feature coverage ranges from 12.5% to 81.2%
across fold–draw pairs; the lowest values occur in fold 3 of one draw.

Against the paired shaped control, the composition is within one point on full
features (median +0.8, higher in six of twelve comparisons, range −48.4 to
+6.3 because of the single weak fold–draw pair) and 4.7 points higher without
trend strength. It is a median 9.4 points **lower** on the daily-profile shape
score, never higher in any comparison, and a median 2.3 points below the
existing pools there. The native composition therefore matches the controls'
feature-space gains but not their improvement in average daily shape.

Nearest full-feature neighbours come from the two TSI prototypes in 590 of 768
query–draw assignments (77%), the sine-plus-breaks prototypes in 126, and
EnergyLoad in 52. Neither ETS prototype is ever the nearest neighbour in the
full space. Remaining distance is spread across features: first-difference
ACF, trend strength, and the ten-lag ACF sum each contribute 12–16% of the mean
per-query squared distance, with no dominant coordinate.

## Transfer panel: Monash traffic_hourly

| Source | Full features | Manhattan | Omit trend strength | Daily-profile shape |
|---|---:|---:|---:|---:|
| Balanced pool | 0.0% | 0.0% | 0.0% | 6.2% |
| Pretraining pool | 0.0% | 0.0% | 0.0% | 6.2% |
| Random MAR | 0.0% | 0.0% | 0.0% | 0.0% |
| **Native composition** | **10.2%** | 7.8% | **21.1%** | 6.2% |
| Matched real baseline | 88.3% | 85.2% | 85.9% | 82.8% |

The existing gap persists on traffic: the median real-minus-best-existing q90
gap is 75.0–89.1 points in every fold, and the existing pools' nearest
neighbours again differ mostly in trend strength, crossing points, and
first-difference ACF.

The composition is higher than the best existing source in all twelve paired
comparisons, but the per-fold median gains are 6.2, 12.5, 43.8, and 3.1
points, so the transfer criterion fails. The one large fold also has a
calibrated q90 radius of 3.9 standardized units, about twice the other folds'
1.7–2.0, so its gain is partly a looser real-calibrated threshold. Without
trend strength, the median paired gain is 15.6 points. Daily-profile coverage
is unchanged at 6.2%: no source reproduces the traffic panel's average daily
shape, whereas real candidates cover 82.8% of queries.

Traffic neighbours are TSI prototypes in 97% of assignments. Their remaining
differences are unlike M4's: real traffic queries have higher spectral
entropy, lower ten-lag and seasonal autocorrelation, fewer crossing points, and
larger maximum level shifts than their nearest composition series (median
standardized differences of about ±1.0–1.3). The composition is too smooth and
too regular for this panel, and its slow level changes are too gentle relative
to weekend and outage effects. The
[targeted follow-up](../feature_coverage_traffic_composition/README.md)
addresses these coordinates on fresh traffic queries.

## Implications

The M4 result supports the hypothesis that the Hourly gap reflects missing
*combinations* of daily cycles with slow level and amplitude changes, and shows
that the shipped TSI, Seasonal, and EnergyLoad classes can express them with
configuration only. The traffic result shows that this specific configuration
does not generalize to a second Hourly panel at the declared threshold. The
evidence justifies continuing evaluation, not changing public presets:
production defaults remain unchanged. Forecasting utility was not measured.

## Limits

All M4 caveats carry over: real roles overlap across folds, the three draws
are reused across folds and panels, and the twelve fold–draw measurements are
not independent replications. The composition's parameter ranges were chosen
to mirror the controls' relative magnitudes before any result was inspected
and were not retuned afterward; a single configuration was evaluated. The
traffic windows are seeded and random, so each fold mixes calendar periods,
and a differently placed window set could give different absolute coverage.
Coverage remains sensitive to the distance definition and to fold-specific
radii.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/synforecast-matplotlib .venv/bin/python \
  benchmarks/validate_hourly_composition.py
```

Requires the completed fresh-split `summary.json`, the cached M4 Hourly
training CSV, and the cached Monash `traffic_hourly_dataset.zip` in
`.cache/feature_coverage/monash/`; pass `--download` to fetch either. Source
hashes are checked. `summary.json` records the protocol, recipe, both panels'
splits, traffic window offsets, per-fold scores, paired effects, criterion
results, nearest-prototype counts, and trace hashes. The six
`*_details.json.gz` files hold per-query distances and neighbours, fitted
parameters, candidate IDs, and composition configurations, provenance, and
failures; they are reproducible, local, and gitignored.
