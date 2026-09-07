# Review: PR #55 — [FEAT] Add generators

**Source:** https://github.com/Nixtla/synforecast/pull/55
**Branch:** `feat/roadmap_generators` → `main`
**Scope:** 52 files, +6227/−87. All 30 CI checks pass. No prior reviews at time of writing.

Adds GRATIS-style mixture autoregressive generation, feature-targeted parameter search, and
moving-block bootstrap / weighted DBA augmentation, with Rust kernels for bulk generation and
numerical operations.

---

## General observations

This is high-quality work, and the parts that were easiest to get wrong are the parts that are
most carefully done:

- **Python/Rust MAR parity is genuinely correct.** The PACF→AR mapping, seasonal polynomial
  multiplication, lag ordering, burn-in discard, population std (ddof=0 on both sides), and the
  fixed-mode-raises / random-mode-falls-back-to-noise split all agree between
  [mar.py](synforecast/generators/mar.py) and [mar.rs](rust/src/generators/mar.rs). The `lo == hi`
  noise-scale case is correctly special-cased in Rust where `Uniform::new(a, a)` would panic.
- **Banded DTW is memory-frugal and index-safe.** [augmentation.rs:70](rust/src/augmentation.rs#L70)
  keeps two rolling rows; `band_width`'s `.max(n.abs_diff(m))` is what guarantees a feasible path
  exists for arbitrary length ratios.
- **`nearest_dtw_neighbors` uses the right granularity** — bounded k-heaps, 4096-pair parallel
  chunks, parallelism over pairs rather than series.
- **Test rigor is above average where it counts.** [test_dtw.py:46](tests/test_dtw.py#L46)
  exhaustively enumerates admissible warping paths for all `n,m ∈ [1,5] × band ∈ [0,5]` against an
  independent oracle; [test_features.py:87](tests/test_features.py#L87) pins Rust-vs-Python feature
  parity at 63/64/121/1000/1001/4093/4095/4096.
- **Docs are unusually well cross-checked.** Every MAR/MBB/DBA parameter name, default, and range in
  GENERATORS.md matches the code; `_lib.pyi` matches the PyO3 bindings exactly; the fixed-mode
  example in `mar.qmd` actually validates (spectral radius ≈ 0.427).

The findings below are mostly about *reachable but unexercised* paths, not about the core algorithms.

---

## Correctness

### 1. `dba` produces exact duplicates of constant series
[dataset.py:978](synforecast/dataset.py#L978) / [dataset.py:1041](synforecast/dataset.py#L1041)

The forward transform divides by `divisor = std if std >= 1e-8 else 1.0`, but `moments` stores `std`
and the inverse multiplies by `std`. When `std < 1e-8` these disagree and the barycenter is
annihilated — output collapses to `np.full(n, mean)`, a byte-identical copy of the source, with no
`_warn_skipped` entry. `test_constant_reference_keeps_zero_scale` pins this, so it's intentional,
but flat series are common (dead SKUs, saturated sensors) and `n_augment` duplicate rows in an
augmented corpus is a leakage/weighting hazard. Either use `divisor` on both sides, or route
near-constant references through the existing skip warning.

### 2. Mixture-stationarity check is silently bypassed above AR order 32
[mar.py:196](synforecast/generators/mar.py#L196)

`_apply_seasonal_factor` inflates component order to `base_order + seasonal_period`, so
`tune_to_features(..., seasonal_period=28)` already exceeds the cutoff and `seasonal_period=168`
(weekly-at-hourly, a headline use case) reaches ~173. Above 32 the Wong–Li check returns `True`
unconditionally. The failure mode is asymmetric and user-hostile: a configuration that would be
*rejected at construction* at order ≤32 is instead accepted, passes its training draws by luck,
then raises `ValueError` from `generate()` in production. Worth either a warning on the skip path or
a cheaper sufficient condition. Untested in both directions.

### 3. A single `inf` aborts an entire panel
[dataset.py:591](synforecast/dataset.py#L591)

`_interpolate_missing` raises unconditionally on `±inf`, and `mbb`/`dba` call it inside the
per-series loop with no `try`. NaN is interpolated, all-NaN is skipped with a warning, but one `inf`
in one series of a 10,000-series panel kills the whole call — and the message doesn't name the
offending series. The docstring's skip clause covers "series without a finite observation", so this
isn't strictly a contradiction, but the three-way asymmetry deserves a deliberate decision and a
test either way.

### 4. `mbb` silently discards an explicitly requested `seasonal_period`
[augmentation.rs:380](rust/src/augmentation.rs#L380)

`period.filter(|p| *p <= values.len() / 2)` drops the user's `seasonal_period=24` on a 40-point
series; the seasonal component becomes all zeros with no indication. On a mixed-length panel this
silently applies to a subset. A `logger.warning` at the Python level (where per-series length is
known) would make it visible.

---

## Performance

### 5. Feature targeting runs entirely on the slow paths

Two compounding issues:

- [mar.py:641](synforecast/generators/mar.py#L641) calls `generate_single_series` — the pure-Python
  per-timestep reference loop — for all `15 × 30 × 3 = 1350` default-budget draws. The PR's own
  benchmark measures the native path at **30.7×** faster single-threaded
  ([native_paths_summary.json](benchmarks/data/native_paths_summary.json)), plus rayon parallelism
  the search never gets. The whole population per generation is independent, so one
  `generate_multi_batch` call per generation would collapse 90 Python simulations into one parallel
  call. The docstring already concedes the two paths have different RNG streams, so the only cost is
  stream identity.
- [mar.py:209](synforecast/generators/mar.py#L209) runs `np.linalg.eigvals` on a `p(p+1)/2` square
  operator inside the pydantic validator, and `_candidate_fitness` constructs one generator per
  candidate → **450 constructions** at default budget. At `seasonal_period=24` that's a 435×435
  non-symmetric eigendecomposition each time. Note the perverse shape: cost peaks just below the
  order-32 cutoff, above which it's free because it's skipped. Only the spectral radius is needed —
  power iteration on `X → Σ wₖAₖXAₖᵀ` is O(K·p³) and would also remove the need for the cutoff.

### 6. The power-of-two FFT "fast path" is slower than the path it bypasses
[fft.rs:159](rust/src/fft.rs#L159)

Power-of-two lengths route into `fft_radix2` on a full complex buffer (2× memory, 2× arithmetic,
twiddles accumulated in the inner loop) instead of the cached RealFFT plan. The PR's own numbers
show the inversion: `feature_kernel` at n=4096 is **0.129 ms**, `feature_kernel_odd` at n=4095 is
**0.0854 ms** — the mixed-radix length is 51% *faster* at effectively identical size. Sending
power-of-two lengths through `real_fft_plan` too removes a ~1.5× penalty on `spectral_entropy`,
which is on the targeting hot path.

### 7. `mbb`/`dba` extract series with one full-panel scan each
[dataset.py:854](synforecast/dataset.py#L854), [dataset.py:969](synforecast/dataset.py#L969)

`df_nw.filter(nw.col(id_col) == series_id)` per series is O(S·N), plus S separate sorts. For
1000×500 that's 5×10⁸ row comparisons before any kernel work — against a **0.067 ms** native MBB
kernel, so ≥99% of wall time is frame slicing. The pattern is pre-existing in
`analyze`/`augment`/`mixup`, but these are the new bulk entry points. Sort once by
`(id_col, time_col)` and split with numpy offsets. Same section: `nw.concat` over
`S·n_augment + 1` single-copy frames with three casts each.

### 8. DBA barycenter refinement is entirely single-threaded
[augmentation.rs:292](rust/src/augmentation.rs#L292)

Iterations × series loop serially in Rust while the neighbor search immediately before it uses every
core. Alignments within one iteration are independent. At L=2048 a 1000-series panel is ~3 minutes
serial vs ~10 s across 20 cores.

Related: [dataset.py:1020](synforecast/dataset.py#L1020) re-calls `dba_barycenter` per copy with
identical reference and neighbors — since `barycenter` initializes to `reference` regardless of
weights, iteration 1's alignment paths are *identical across copies*, so ~20% of DBA work is
recomputed at default `n_iterations=5`.

---

## Test coverage

The suite is strong, but four gaps let real bugs through silently:

- **Native standardization on the accepted path is never asserted** —
  [mar.rs:228](rust/src/generators/mar.rs#L228). `standardize=True` is the default and `generate()`
  always goes native, yet every native test either sets `standardize=False` or asserts only
  scale-invariant quantities (ACF, shape, determinism). Changing the guard from `sp[8]` to `sp[9]`
  would return unstandardized series from every random-mode `generate()` with nothing failing. One
  `mean≈0, std≈1` assertion closes it.
- **The 12-scalar batch contract is checked by shape only** —
  [test_mar.py:43](tests/test_mar.py#L43) asserts `scalars.shape == (12,)` and nothing about values.
  `MARGenerator` also appears nowhere in [test_innovations.py](tests/test_innovations.py) (which
  parametrizes GARCH/SARIMA/ETS). Swapping `_rs_innov_dist`/`_rs_innov_param` makes Rust read
  `dist = 5`, fall through to `_ => normal`, and produce Gaussian innovations for
  `innovation_distribution="t"` with no error — masked by `standardize`.
- **Weight normalization is never exercised unnormalized.** All fixed-mode tests pass weights
  already summing to 1. If the `object.__setattr__` normalization regressed, Python would raise from
  `np.random.choice`, but Rust's `choose_component` walks a cumulative sum against `uniform01()` —
  weights `[1.0, 3.0]` would select component 0 every step, silently yielding a single-component AR
  series.
- **The mixture-stationarity oracle never sees mixed AR orders** —
  [test_mar.py:343](tests/test_mar.py#L343) draws `order` once for all components, so the
  zero-padding of a short component into a larger companion is never compared against anything.
  Drawing `order` per component is a one-line fix.

Also: [test_mar.py:630](tests/test_mar.py#L630) is seed-fragile (depends on `seed=5` happening to
draw a failing candidate); monkeypatching `_candidate_fitness` as line 510 already does would make
it deterministic.

---

## Documentation

- [dataset.py:252](synforecast/dataset.py#L252) — "mixup forms convex combinations of
  **z-normalized** series". Default is `scaling="mean"` (divide by mean absolute value);
  z-normalization is the opt-in `"std"`. Contradicts the correct argument docs at line 663. This is
  the class summary rendered at the top of the API page.
- [GENERATORS.md:139](GENERATORS.md#L139) — the DBA section claims "Decomposition, block sampling,
  DTW, and barycenter updates execute in native Rust." DBA does neither decomposition nor block
  sampling; copy-paste from the MBB section above.
- [index.html.mdx:170](docs/mintlify/index.html.mdx#L170) still says "31 generators" where
  README.md says 32. It's regenerated by `make all_docs`, so the site will be right — but the PR
  hand-updated the other `docs/*.html.md` sources, so this artifact is just stale.
- [generators_pretraining.html.md:22](docs/generators_pretraining.html.md#L22) —
  `tuning_diagnostics` is absent from the mkdocstrings `members` list, yet `mar.qmd:99` tells users
  to print it and GENERATORS.md:230 documents `best_distance`.
- [dataset.py:801](synforecast/dataset.py#L801) / [dataset.py:915](synforecast/dataset.py#L915) —
  `mbb` and `dba` are published reference members but document their algorithms in prose only, with
  no `Args`/`Returns`/`Raises`. Every other public `SynAugment` method has them; these two raise 6–7
  distinct `ValueError`s that go unlisted.

---

## Minor

- **`n_iterations` is unbounded and uninterruptible** —
  [dataset.py:949](synforecast/dataset.py#L949) checks only `>= 1`, while `n_augment` is explicitly
  capped at 1000 "to prevent resource exhaustion" in three places. `dba_barycenter` releases the
  GIL, so `n_iterations=10**9` spins in native code with `KeyboardInterrupt` undeliverable.
- **`dtw_alignment` allocates an unbounded parent matrix** —
  [augmentation.rs:124](rust/src/augmentation.rs#L124). `vec![u8::MAX; n * row_width]` is ~`0.2·L²`
  bytes at default `window_fraction`; L=300k → ~18 GB. A failed Rust allocation calls
  `handle_alloc_error`, which *aborts* rather than panics, so PyO3 can't convert it to an exception.
  `Vec::try_reserve` or an explicit budget check would surface it as `ValueError`.
- **Unchecked `usize` overflow in fixed-param parsing** —
  [mar.rs:109](rust/src/generators/mar.rs#L109). `offset + order` wraps (release profile has
  `overflow-checks` off), so a crafted `ap[1]` slips past the guard into a slice panic. Only
  reachable via the raw `_lib` binding — the public `MARGenerator` path derives orders from real
  `len()` — and `panic = "unwind"` makes it a catchable `PanicException`, so impact is bounded.
  Still worth `checked_add`.
- **`pairwise_dtw_distances` is dead code across four layers** —
  [augmentation.rs:158](rust/src/augmentation.rs#L158),
  [bindings/augmentation.rs:49](rust/src/bindings/augmentation.rs#L49),
  [_dtw.py:48](synforecast/_dtw.py#L48), [_lib.pyi:20](synforecast/_lib.pyi#L20). No caller in
  `synforecast/`; superseded by `nearest_dtw_neighbors`, and it allocates the exact O(S²) matrix
  that function was written to avoid.
- **`_moving_average` is a test oracle shipped in the package** —
  [_features.py:26](synforecast/_features.py#L26). Sole consumer is
  [test_features.py:62](tests/test_features.py#L62), and it omits the `period <= n//2` fallback that
  Rust applies — the test compensates by recomputing `usable_period` itself, so the rule lives in
  neither implementation and the oracle can go stale silently.
- **`tune_to_features` has no generator-kwarg passthrough** —
  [mar.py:373](synforecast/generators/mar.py#L373). `BaseGenerator` sets `extra="forbid"`, so anyone
  needing `engine`, `standardize`, `innovation_distribution`, or custom column names must rebuild
  `MARGenerator` by hand from the four fixed-parameter lists — losing `tuning_diagnostics` in the
  process.

---

**Overall:** the algorithmic core is sound and well-tested. Findings 1, 2, and 5 are the ones worth
addressing before merge.
