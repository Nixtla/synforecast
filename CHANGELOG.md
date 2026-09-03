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
  decomposition, MBB sampling, DTW, and DBA updates.

## 0.1.0 - Initial release
