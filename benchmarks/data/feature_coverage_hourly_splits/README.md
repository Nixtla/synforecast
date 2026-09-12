# Hourly coverage on fresh query splits

The Hourly feature-space gap persists on **256 additional evaluation series**
across four disjoint query groups and three new generator draws. Paired controls
show that combining daily cycles with slow level changes, amplitude variation,
and a second harmonic substantially reduces it. Gains survive omitting trend
strength, and the harmonic improves a separate daily-profile shape score.
This supports further evaluation of compositional generators. No native feature
definitions or production presets were changed.

Follow-up: [native generator composition and a traffic_hourly transfer test](../feature_coverage_hourly_composition/README.md)
now checks whether existing generators reproduce these gains.

## Protocol

Use the same public M4 Hourly training panel and trailing 512 observations,
with `native_v1`, period 24, and window 24. Exclude the original investigation's
32 evaluation IDs from every role. Split 256 of the remaining 382 IDs into
four disjoint groups of 64 evaluation queries. Each fold independently selects
256 candidate slots and 32 calibration queries outside its evaluation group;
64 of its candidates fit the scaler/PCA. Calibration and evaluation queries
never enter that fold's fitting or candidate sets.

Request 256 synthetic series per existing source for each of three new seeds.
Take the first 240 shared complete slots in generation order, without replacing
failures. Match real and synthetic candidate lengths and counts exactly. Reuse
the same synthetic draws across folds to isolate real-split variation. Fit
scaling on real references and calibrate q50/q90/q95 distance radii against real
candidates separately for every fold, draw, and metric.

The controls were gated on a declared criterion: the median over generation
seeds of the real-minus-best-existing-source q90 gap must exceed or equal
25 percentage points in at least three folds. All four passed, with gaps of
89.1, 89.1, 85.9, and 82.8 points. This is an operational trigger, not a
significance test. Control formulas and parameter ranges were specified before
inspecting these results and were not retuned.

## Existing generators

Median q90 coverage across four folds × three draws, with 240 candidates:

| Source | Full features | Omit trend strength | Daily-profile shape |
|---|---:|---:|---:|
| Balanced pool | 0.8% | 43.8% | 68.8% |
| Pretraining pool | 1.6% | 44.5% | 68.8% |
| Random MAR | 0.0% | 0.0% | 0.0% |
| Matched real baseline | 90.6% | 91.4% | 89.8% |

Full-feature coverage ranges from 0–10.9% for balanced and 0–15.6% for
pretraining; MAR remains zero. Manhattan distance raises the pool medians to
18.0% and 18.8%. Absolute coverage remains sensitive to the distance definition.
Trend strength contributes a median 47.5% and 42.5% of the mean per-query
squared-distance fractions for balanced and pretraining, respectively.

The separate shape check averages 21 complete 24-hour cycles, centers and
normalizes the resulting daily profile, and minimizes RMS distance over 24
circular hourly shifts. It discards the final eight observations, level,
amplitude, and much day-to-day variation. Its relatively high pool coverage
supports the presence of similar daily shapes; it does not establish complete
time-series fidelity. Each table column uses its own real-calibrated radius.

## Paired diagnostic controls

Start with a unit daily sine and Gaussian noise. Add slow level changes
(linear drift plus a weekly sine), an exponential weekly amplitude envelope,
and a second daily harmonic. Phases and component parameters are randomized;
noise and components are shared between variants within each slot. The recipe
uses no real series values. Each control draw retains exactly the same 240
slots and real baseline as its paired existing-generator draw.

| Control | Full features | Omit trend strength | Daily-profile shape |
|---|---:|---:|---:|
| Cycle | 1.6% | 46.9% | 67.2% |
| Cycle + drift | 57.0% | 60.2% | 66.4% |
| Cycle + amplitude | 3.1% | 50.0% | 67.2% |
| Cycle + drift + amplitude | 63.3% | 64.8% | 66.4% |
| Shaped cycle + drift + amplitude | 68.0% | 68.8% | 75.8% |

Drift improves full-feature q90 coverage in all twelve paired comparisons,
with a median gain of 48.4 percentage points over the cycle control. Omitting
trend strength preserves a 15.6-point median gain, also positive in all twelve
comparisons. Amplitude variation alone has little effect on the primary score.
The complete shaped control improves full-feature coverage by a paired median
59.4 points, coverage without trend strength by 21.9 points, and daily-profile
coverage by 13.3 points. All three gains are positive in every comparison.
These are medians of paired changes; they need not equal differences between
the table's separately aggregated medians.

The evidence supports a gap in combinations of seasonal shape and slower
behaviours, with improvement extending beyond the trend-strength coordinate.
It does not justify changing feature definitions to increase coverage, or
immediately replacing public presets with this diagnostic recipe.

## Limits and next work

“Fresh” means excluded from the original scored query set, not a new dataset.
All observations come from the same M4 training panel. Reference, candidate,
and calibration roles overlap across folds; an evaluation ID in one fold may
serve another role in another fold. The three synthetic draws are reused.
The twelve measurements therefore are not independent replications or
confidence intervals. Findings are conditional on complete-case retention,
this length and calibration scheme, and exploratory control choices motivated
by the earlier investigation. Forecasting utility has not been measured.

The [follow-up](../feature_coverage_hourly_composition/README.md) evaluated
these combinations through existing generator configuration on these splits
and on the Monash traffic_hourly panel: the M4 gain is reproduced, the
daily-shape gain is not, and the transfer gain is small. Production presets
remain unchanged. The broader cross-frequency benchmark and interpretation
notebook remain outstanding.

## Reproduce

```bash
.venv/bin/python benchmarks/validate_hourly_splits.py
```

Requires the prior Hourly investigation's completed `summary.json` and the
cached public training CSV; pass `--download` if the CSV is absent. Source
hashes are checked. `summary.json` records the protocol, exact splits, seeds,
retention, run summaries, aggregate scores, paired effects, and trace hashes.
The six `*_details.json.gz` files contain per-query distances/neighbours,
fitted parameters, actual candidate IDs, and existing-generator provenance
and failures. They are reproducible, local, and gitignored. Only this report
and summary JSON are intended for version control; no Parquet is added.
