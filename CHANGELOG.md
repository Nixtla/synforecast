# Changelog

## Unreleased

- Added native panel feature extraction and feature-space coverage evaluation,
  with real-anchored PCA/grid defaults, explicit missing-feature diagnostics,
  and out-of-range counts. The twelve-feature schema is frozen as `native_v1`;
  it preserves the existing private MAR-targeting helper.
- Added a reproducible native feature pilot on M4 Monthly/Yearly training
  samples and controlled changes. Added explicit `window_size` for short
  panels without changing the default window or seasonal decomposition.
- Added offline full-feature distance corroboration of the native pilot,
  with held-out real calibration, paired reference/corpus-size experiments,
  and grid-boundary sensitivity at fixed PCA coordinates.
- Audited native feature availability on all M4 training panels and documented
  explicit period/window recipes, including the short-series limitation of
  annual Weekly seasonality. Reproducible feature caches and detailed traces
  are now excluded from version control; summaries and reports are retained.
- Finalized `native_v1` without changing the candidate formulas; benchmark
  validation remains compatible with saved candidate artifacts. Added a small
  six-frequency benchmark comparing balanced/pretraining pools and random MAR
  with exactly matched retained lengths and separate real calibration queries.
- Added an Hourly coverage investigation with fixed real queries, ten generation
  redraws, nested candidate sizes, nearest-series examples, and distance/feature
  sensitivity diagnostics. Native feature definitions and presets are unchanged.
- Validated the Hourly gap on four fresh query groups and three new generation
  draws, with gated paired cycle/drift/amplitude/shape controls and a separate
  daily-profile shape diagnostic. Detailed traces remain outside version control.
- Tested a benchmark-only composition of existing TSI, ETS, Seasonal, and
  EnergyLoad generators on the fresh Hourly splits and on the Monash
  traffic_hourly panel. It reproduces the controls' native-feature gains on M4
  but not their daily-shape gain, and transfers only weakly; presets unchanged.
- Added a traffic-targeted second composition round on fresh traffic_hourly
  queries with an M4 retention check. It roughly doubles the first round's
  traffic coverage but stays below the declared threshold; presets unchanged.
- Added a three-seed, four-fold cross-frequency coverage benchmark on fresh M4
  Yearly/Quarterly/Monthly/Weekly/Daily queries. No frequency other than
  Hourly shows a material gap; Weekly is flagged minor and Daily is
  query-dependent. Presets unchanged.
- Added the `feature_coverage` capabilities notebook, which interprets the
  coverage benchmark chain from committed summaries, plots real queries next
  to their nearest synthetic neighbours from a small committed example set,
  and states the preset recommendation (keep defaults; treat Hourly as
  frequency-specific).
- Added a third-panel coverage validation on Monash tourism yearly/quarterly,
  hospital monthly, traffic weekly, and weather daily. Tourism quarterly and
  weather daily show material gaps, hospital monthly a minor one, so coverage
  gaps are panel-specific, not confined to Hourly. Presets unchanged.
- Added a seasonal-strength composition round with held-out validation
  (M3 quarterly/monthly, tourism monthly). A moderated corpus fails the
  replacement criterion but, added to the balanced pool at equal candidate
  count, raises coverage on all five panels; presets unchanged pending a
  pre-declared augmented-pool test and a forecasting-utility check.
- Added the opt-in `seasonal_pool()` preset: eight configured TSI, ETS, and
  Seasonal generators with a pronounced seasonal cycle on a moving level,
  meant to be added to `balanced_pool` or `pretraining_pool`. Exported from
  `synforecast`, documented in the composition reference and GENERATORS.md.
- Tested `seasonal_pool` with a pre-declared augmented-pool criterion on four
  never-used panels: passes on M1 monthly, neutral on fresh M4 Quarterly and
  Monthly, fails the no-harm gate on intermittent car parts when it replaces
  the pretraining share. Added on top of both pools it never hurts. Opt-in only.
- Renamed `balanced_pool` to `interpretable_pool`; the pool is balanced across
  niches, but so is `pretraining_pool`, and the distinguishing property is that
  every instance is an interpretable single-mechanism process. `balanced_pool`
  remains as a deprecated alias that warns. Benchmark artifacts keep the old
  label.
- Added `pretraining_pool(include_seasonal=...)`, which appends the eight
  `seasonal_pool` instances at the derived period (skipped for yearly data).
  A pre-declared fifteen-panel test found gains on every quarterly and
  monthly panel but small losses on weekly, daily, and intermittent panels, so
  the default stays False; the flag is recommended for quarterly and monthly
  targets.

- `balanced_pool` derives the seasonal period of its seasonal variants from
  `freq` (hourly 24, daily 7, weekly 52, monthly 12, quarterly 4) instead of
  hardcoding 12, and accepts an explicit `seasonal_period` override that
  `pretraining_pool` forwards.
- Added `MARGenerator` for GRATIS-style MAR simulation, including a validated
  fixed-parameter mode.
- Added `MARGenerator.tune_to_features` and a minimal feature module for
  feature-targeted generation, with `tuning_diagnostics` reporting convergence
  and search budget usage.
- Added non-parametric `SynAugment.mbb` remainder block bootstrapping.
- Added panel-aware `SynAugment.dba` with banded DTW.
- Added `MARGenerator` to `pretraining_pool` after the existing meta-generators.
- Added native Rust paths for MAR batch generation, feature computation,
  decomposition, MBB sampling, DTW, and DBA updates. Banded DTW stores only
  the band, spectral entropy uses cached RealFFT plans for all lengths,
  and `SynAugment.dba` computes pairwise panel
  distances in bounded parallel chunks, retaining only the requested nearest
  neighbors. Moving-average decomposition is linear-time and reused across
  MBB copies.
- Feature search scores populations in parallel native Rust, isolates candidate
  failures, and caches stationarity checks. Covariance certificates with a NumPy
  dense eigenvalue fallback keep validation free of a SciPy runtime dependency.
  It accepts output, standardization and innovation options and `n_jobs`.
- Panel methods sort and partition once. DBA copies share initial alignments
  and refine in parallel, with allocation/iteration limits and interrupt checks.
- Removed the unused dense pairwise-DTW API and moved the convolution test
  oracle out of the package. Fixed native MAR coefficient-offset overflow.
- `MARGenerator` rejects non-finite or oversized parameters, checks
  second-order stationarity of fixed mixtures (Wong and Li 2000), and raises
  instead of substituting noise when a fixed model fails the output guards.
- MAR random-mode fallback noise now honors `standardize=True` in both paths.
- Fixed MAR mixtures no longer bypass stationarity validation above AR order
  32. Trailing zeros are ignored, and larger mixtures require a sufficient
  stability condition; uncertified mixtures raise an explicit error.
- `SynAugment.mbb` and `SynAugment.dba` skip unusable series with a logged
  warning instead of raising or silently dropping them.
- MBB reports explicit seasonal periods that cannot fit two cycles, DBA
  reports near-zero reference scales, and infinity errors identify the source.

## 0.1.0 - Initial release
