# MAR and augmentation correctness checks

Python tests call the compiled Rust kernels as well as Python entry points.
Agreement between wrappers alone is insufficient; the checks below use small
independent oracles, known answers, and model-implied statistics. Bibliographic
references and implementation differences are in [GENERATORS.md](../GENERATORS.md).

| Algorithm | Meaningful checks | Location |
|---|---|---|
| DTW distance/path | Exhaustive enumeration of admissible paths on 1,500 small cases, including unequal lengths, bands and ties; reconstructed path cost | `test_dtw.py` |
| Weighted DBA | Explicit association lists from exhaustive alignments; unequal weights/lengths; 1, 2 and 4 updates on a fixture that keeps changing | `test_dtw.py` |
| Trend, seasonal components and strengths | Hand-calculated odd/even-period components and variance ratios; direct convolution, endpoint and large-offset checks | `test_features.py` |
| FFT / entropy | Naive DFT coefficients, NumPy periodograms, odd/even/prime lengths, constants and round trips | `../rust/src/fft.rs`, `test_features.py` |
| MBB | Recover unique source residual indices to verify contiguous requested blocks, partial ends and both eligible start endpoints; single/batch paths and public per-source cap | `test_mbb.py` |
| MAR | Hand-expanded lag recurrence and burn-in in Rust/Python; PACF and seasonal polynomials; full Kronecker stationarity oracle; unequal-weight analytic mean, variance and ACF in both paths | `test_mar.py`, `../rust/src/generators/mar.rs` |
| Feature search | Mean-feature L2 objective; best-candidate retention; stopping at equality; exhausted budget, invalid candidates, reproducibility and fresh-draw behavior | `test_mar.py` |

DTW uses squared local cost and returns its square root; ties prefer diagonal,
then up, then left. Decomposition extends the nearest computed trend value at
each endpoint. These conventions matter to the expected values above.
Seeded MAR statistical checks use unstandardized draws; Python and Rust are
checked against the same identities without requiring identical RNG streams.
Innovation propagation is checked against all five reference distributions in
`test_innovations.py`. MAR checks also cover unnormalized weights, mixed AR
orders, and high-order validation, including trailing-zero bypass prevention
and conservative rejection when stability cannot be certified.

During this review, temporary wrapper mutations were run in isolated processes:
forcing MBB block size to 2 failed 8 assertions; replacing DBA weights with ones
failed 6; reversing Python MAR coefficients failed its recurrence assertion;
moving all seasonality into the remainder failed both hand-calculated fixtures.
These demonstrate sensitivity to those mistakes, not exhaustive mutation coverage.

## Feature-search experiment

Run `python benchmarks/benchmark_mar_targeting.py --save /tmp/mar-targeting.json`.
Four targets, seeds 31–35, 60 candidate evaluations per method, three training
draws per candidate at lengths 64/96/128, and 24 fresh native draws per fitted
generator were used. Both methods use the same candidate prior and objective;
the baseline samples independently rather than evolving candidates. Evaluation
seeds are disjoint from training seeds. Seasonal period is 12 throughout.

Mean held-out L2 distances from the review run (lower is better):

| Target | Evolutionary | Random | Evolutionary wins |
|---|---:|---:|---:|
| ACF1 = 0.7 | 0.0327 | 0.0483 | 3/5 |
| Spectral entropy = 0.5 | 0.0602 | 0.0368 | 1/5 |
| Trend strength = 0.5 | 0.0625 | 0.0598 | 4/5 |
| Seasonal strength = 0.5 | 0.1163 | 0.1395 | 4/5 |

Per-seed measurements and environment metadata are saved in
[mar_targeting_results.json](../benchmarks/data/mar_targeting_results.json).

This small experiment does not establish consistent superiority over random
search or guarantee the requested features on new draws. In particular, the
best training distance is not a held-out accuracy estimate.

## Reproduce checks

Build the extension first (`maturin develop --release` in an activated environment),
then run:

```bash
pytest tests/test_dtw.py tests/test_features.py tests/test_mbb.py tests/test_mar.py --no-cov
cargo test --manifest-path rust/Cargo.toml --lib
```

Full test/lint commands are in [CONTRIBUTING.md](../CONTRIBUTING.md).
Source provenance and the unresolved upstream RealFFT copyright notice are
recorded separately in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
