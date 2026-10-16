# Changelog

## Unreleased

### Added

- `BaseGenerator.generate_values(n_series, n_jobs)`: the series `generate`
  would draw, as value arrays without building a DataFrame (native Rust batch
  path where available).
- `members=` on `pretraining_pool`, `interpretable_pool` and `seasonal_pool`:
  build only the selected pool entries, each identical to the full pool's
  entry. Drawing one member per series no longer constructs the whole pool
  (0.11 ms instead of 3.6 ms per draw).

### Changed

- `KernelSynthGenerator` composes stationary kernels as functions of the lag
  and samples them with an O(n^2) Durbin-Levinson recursion (the same Cholesky
  factor). Compositions with a linear kernel keep the dense path, with a
  cheaper Gram matrix. At length 2,273 a draw takes about 12 ms instead of
  280-970 ms.
- `GaussianProcessGenerator(kernel="periodic")` samples exactly from the
  periodic kernel's Fourier series in O(n J) instead of an O(n^3) Cholesky:
  about 1 ms instead of 560 ms at length 2,273.

### Reproducibility

- Same-seed output of periodic `GaussianProcessGenerator` differs from 0.2.0;
  the sampled distribution is unchanged.
- `KernelSynthGenerator` output equals 0.2.0 to floating-point rounding
  (within 1e-7 of the series' standard deviation), not bit for bit.
- Every other generator, and `pretraining_pool` with `members=None`, is
  bit-identical to 0.2.0.

## 0.2.0 - 2026-10-12

### Added

- `synforecast.evaluation`: `compute_features` (twelve scale-free features per
  series) and `feature_coverage` / `compare_feature_coverage`, which score how
  well a synthetic corpus covers real data.
- Opt-in `seasonal_pool(freq=...)` preset and
  `pretraining_pool(include_seasonal=...)` for strongly seasonal data.
- `MARGenerator` for GRATIS-style mixture-autoregressive simulation, with
  `tune_to_features` for feature-targeted generation. It is also part of
  `pretraining_pool`.
- `SynAugment.mbb` (moving-block bootstrap) and `SynAugment.dba` (DTW
  barycenter averaging).
- Native Rust paths for MAR generation, features, decomposition, MBB, DTW, and
  DBA.

### Changed

- Renamed `balanced_pool` to `interpretable_pool` and `include_balanced` to
  `include_interpretable`. The old names still work but warn.
- `interpretable_pool` derives its seasonal period from `freq` instead of
  hardcoding 12, and accepts a `seasonal_period` override. Its cyclic
  variants take their mean period from it instead of a fixed 50 steps.
- `SynAugment.mbb` and `SynAugment.dba` skip unusable series with a warning.
- Removed the unused dense pairwise-DTW API.

### Fixed

- `MARGenerator` validates parameters and stationarity, including mixtures
  above AR order 32, and raises instead of silently substituting noise.
- MAR fallback noise honors `standardize=True`.
- Native MAR coefficient-offset overflow.
- Overlays (changepoints, anomalies, missingness) no longer affect each
  other or the base series. Previously, enabling one could move another, and
  for IoTSensor and DailyActiveUsers change later series.

### Reproducibility

- Same-seed output differs from 0.1.0 for the cyclic pool variants,
  IoTSensor, DailyActiveUsers, and multi-series ETS/StateSpace
  `generate_with_states`; with overlays enabled also for KernelSynth, Copula,
  VAR and TCM.

## 0.1.0 - Initial release
