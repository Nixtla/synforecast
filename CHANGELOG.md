# Changelog

## Unreleased

- Added `MARGenerator` for GRATIS-style MAR simulation, including a validated
  fixed-parameter mode.
- Added `MARGenerator.tune_to_features` and a minimal feature module for
  feature-targeted generation.
- Added non-parametric `SynAugment.mbb` remainder block bootstrapping.
- Added panel-aware `SynAugment.dba` with banded DTW.
- Added `MARGenerator` to `pretraining_pool` after the existing meta-generators.
- Added native Rust paths for MAR batch generation, feature computation,
  decomposition, MBB sampling, DTW, and DBA updates. Banded DTW stores only
  the band, spectral entropy uses a cached RealFFT path for
  non-power-of-two lengths, and `SynAugment.dba` computes pairwise panel
  distances in parallel.
- `MARGenerator` rejects non-finite or oversized parameters, checks
  second-order stationarity of fixed mixtures (Wong and Li 2000), and raises
  instead of substituting noise when a fixed model fails the output guards.
- `SynAugment.mbb` and `SynAugment.dba` skip unusable series with a logged
  warning instead of raising or silently dropping them.

## 0.1.0 - Initial release
