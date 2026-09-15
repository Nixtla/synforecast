# SynForecast Roadmap

## Augmentation

- **Optional moment matching** — augmented series are pinned to the source mean,
  std, and lag-1 ACF by construction; add `match_moments=False` and test whether
  unpinned should be the default. Highest priority here.
- **Strategy selection** — fit-and-simulate, `mixup`, `mbb`, and `dba` now ship
  as separate methods; add a unified `SynAugment(strategy=...)` switch spanning
  them and the jitter/scale/warp baselines.
- **Wider fitter coverage** — 9 fitters against 32 generators, so only a third of
  the library is reachable; add ETS, StateSpace, INAR, multi-period seasonal.
- **Panel-aware augmentation** — augment jointly to preserve cross-series
  correlation instead of series-by-series.
- **Exogenous passthrough** — carry `X` columns through augmentation.
- **Fit diagnostics** — return ranked candidates with fit scores so a poor
  best-fit is visible rather than silent.
- **Rust batch path** — route fitted-generator simulation through it so
  augmentation scales like generation.

## Evaluation

- **Feature-space coverage** — native panel extraction and real-anchored
  PCA/grid evaluation are implemented in `synforecast.evaluation`, with a
  frozen twelve-feature `native_v1` schema. Monthly/Yearly pilot and full-feature
  corroboration are complete, including paired reference/sample-size checks
  and grid sensitivity with fixed PCA. Use full-feature distances alongside
  the exploratory map. Availability is now audited on all 100,000 M4 training
  series with explicit period/window recipes; whole-panel Weekly uses the
  nonseasonal convention because annual seasonality excludes 65/359 series.
  A small six-frequency comparison of balanced/pretraining pools and random
  MAR is complete, with matched lengths and held-out distance calibration.
  The Hourly investigation now covers ten redraw seeds and 60/240 matched
  candidates: the measured gap persists, with material trend-strength and
  metric sensitivity. Four fresh query groups (256 series) confirm the gap;
  paired cycle/drift/amplitude/shape controls improve both native-feature and
  daily-profile coverage. A benchmark-only composition of existing TSI,
  Seasonal, and EnergyLoad generators reproduces the full-feature gain on M4
  (median q90 coverage 67% vs 1–2%) but not the daily-shape gain, and gives
  only small gains on Monash traffic_hourly. A targeted second round on fresh
  traffic queries roughly doubles that gain (median q90 coverage 19% vs 0%)
  while retaining the M4 result, but stays below the declared 25-point
  threshold, so presets stay unchanged. Remaining traffic differences are
  crossing points and ten-lag autocorrelation. A three-seed, four-fold
  cross-frequency benchmark on fresh queries finds no material gap on Yearly,
  Quarterly, Monthly, Weekly, or Daily (Weekly minor, Daily query-dependent),
  so within M4 the Hourly gap is frequency-specific. A third-panel validation
  on Monash tourism, hospital, traffic weekly, and weather panels finds
  material gaps on tourism quarterly and weather daily and a minor one on
  hospital monthly: gaps are panel-specific, with strong regular seasonality
  as the recurring theme. The interpretation notebook
  (`nbs/docs/capabilities/feature_coverage.ipynb`) presents the chain and the
  recommendation. A seasonal-strength composition round (two declared
  recipes; design panels tourism quarterly and hospital, validation M3
  quarterly/monthly and tourism monthly) fails the replacement criterion but
  the moderated recipe, added to the balanced pool at equal count, raises
  coverage on all five panels (+3 to +32 points over adding the pretraining
  pool). The corpus now ships as the opt-in `seasonal_pool()` preset; a
  pre-declared augmented-pool test on four new panels passes on M1 monthly,
  is neutral on M4, and fails no-harm on intermittent car parts when it
  displaces the pretraining share, so it is documented as an addition on top
  of the pools and stays out of the defaults. `pretraining_pool` gained an
  `include_seasonal` flag; a pre-declared fifteen-panel default-flip test
  passed on all quarterly/monthly panels but failed no-harm on weekly, daily,
  and intermittent ones, so the default stays False and the flag is
  recommended for quarterly/monthly targets. `balanced_pool` was renamed
  `interpretable_pool` with a deprecated alias. Next: measure forecasting
  utility of pool + seasonal_pool pretraining on a seasonal panel; consider a
  pre-declared test of a period-aware default (on only for periods 4 and 12). Generated feature caches and detailed traces stay
  outside version control.
- **Nearest-neighbour distance to real data** — promote the memorization check
  from notebook to API.
- **Fidelity scores** — discriminative and predictive scores for comparing
  generation methods.
- **Wider benchmarks** — extend `when_synthetic_helps` beyond one model and
  dataset.

## Performance

- **Rust SARIMA exogenous support**, then drop the Python SARIMA path.
- **Rust batch dispatch** for `Copula`, `VAR`, `DailyActiveUsers`, `IoTSensor`.
- **Streaming generation** — chunked Parquet/Arrow output for corpora larger than
  memory.

## Integrations and docs

- **Hierarchical panels** — panels that aggregate coherently, for testing
  reconciliation.
- **Generator selection guide** — a decision path to a generator or pool, instead
  of reading all 32 pages.

## Under consideration

- **Deep generative models** (GAN/VAE/diffusion) — heavy deps, GPU time, and
  per-dataset training conflict with cheap deterministic CPU generation;
- **Irregular sampling** — non-uniform time grids and channels observed at
  different times.
