# Pre-declared test of the shipped `seasonal_pool` preset

The opt-in `synforecast.seasonal_pool()` preset was tested with the
augmented-pool criterion declared **before** running, on four panels never
used in any earlier step. Adding the preset to the balanced pool in place of
the pretraining pool's share **passes both gates on M1 monthly, is neutral on
fresh M4 Quarterly and Monthly samples, and fails the no-harm gate on the
intermittent car-parts panel** and, narrowly, on M4 Quarterly. Added on top of
both pools instead, a post-hoc comparison against adding random MAR shows it
never hurts and helps most where seasonal gaps exist. The preset stays
opt-in and is not part of any default pool.

## Protocol

Four panels, one new split seed, three new generation seeds, the same
four-fold fresh-query protocol, matched candidates, and real-only radii as the
earlier benchmarks:

| Panel | Series used | Length | Role |
|---|---:|---|---|
| m4_quarterly_fresh | 640 of 24,000, new sample seed | 22–512 | never-sampled M4 Quarterly series |
| m4_monthly_fresh | 640 of 48,000, new sample seed | 54–512 | never-sampled M4 Monthly series |
| m1_monthly | 617 | 48–150 | independent competition panel |
| car_parts | 640 of 2,674 | 51 | intermittent, 76% zeros; a no-harm check where seasonality is weak |

The preset is scored exactly as the benchmark corpora were, as a fourth source
with the same 240 matched slots per fold. **Declared criterion**, per panel,
on the 480-candidate unions balanced + preset versus balanced + pretraining,
at each fold's real-calibrated q90 radius: *gain* requires a median paired
change of at least +5 points in three of four folds; *no harm* requires no
fold's median paired change below −2 points. The control is therefore not
"the pool alone" but "the same number of extra candidates from the existing
corpus", which is the choice a user faces.

## Results

Median q90 coverage across four folds × three draws:

| Panel | Balanced alone | Preset alone | Balanced + pretraining (480) | Balanced + preset (480) | Paired change | Gain | No harm |
|---|---:|---:|---:|---:|---:|---|---|
| m4_quarterly_fresh | 78.1% | 32.8% | 84.4% | 81.2% | +0.0 | no | **no** (fold 3 at −3.1) |
| m4_monthly_fresh | 89.1% | 56.2% | 91.4% | 96.9% | +2.3 | no | yes |
| m1_monthly | 66.4% | 66.4% | 76.6% | 89.8% | **+12.5** | **yes** | yes |
| car_parts | 43.8% | 21.1% | 50.0% | 46.9% | −3.1 | no | **no** |

Per-fold median paired changes are 0.0, +1.6, 0.0, −3.1 (M4 Quarterly);
+9.4, +3.1, −1.6, 0.0 (M4 Monthly); +12.5, +7.8, +20.3, +9.4 (M1 monthly,
higher in all twelve pairs); and −9.4, −4.7, −3.1, −3.1 (car parts, range
−25 to +1.6). The M4 panels have no existing gap to fill (their pools already
cover 78–89% against baselines of 84–95%), so a neutral result there is
expected; on M1 monthly, where the pools sit 24 points below the baseline
with a minor flag, the preset closes about half of the shortfall and is as
good as either pool even standalone. On car parts the pools themselves have
a material gap (44% against 85%), but it is an intermittent-demand gap, and
the seasonal preset contributes nothing to it while the pretraining pool's
meta-generators contribute a little; replacing that share therefore costs
coverage.

Remaining preset distances on the M4 panels are led by seasonal strength in
the opposite direction from the tourism case: real M4 queries are **less**
seasonal than the preset's nearest neighbours by 0.6–0.8 standardized units.
The persistent and weakly seasonal TSI variants supply most of the preset's
nearest neighbours everywhere; on car parts the weakly seasonal variant alone
supplies 65%.

## Post-hoc three-way union

Specified after the results above were seen, so labelled post hoc: add the
preset **on top of** both pools (720 candidates) and compare with adding the
same number of random MAR series instead.

| Panel | Balanced + pretraining (480) | + MAR (720) | + preset (720) | Paired change vs + MAR | Pairs higher |
|---|---:|---:|---:|---:|---:|
| m4_quarterly_fresh | 84.4% | 85.2% | 85.9% | +1.6 | 11/12 |
| m4_monthly_fresh | 91.4% | 92.2% | 96.9% | +3.9 | 6/12 |
| m1_monthly | 76.6% | 78.1% | 89.8% | +7.8 | 12/12 |
| car_parts | 50.0% | 54.7% | 53.1% | −0.8 | 1/12 |

Added on top of both pools, the preset is never worse than the 480-candidate
union and is ahead of the MAR control on three panels, within one point of
it on car parts. This is the configuration the preset's documentation
recommends.

## Implications

The declared gates give a mixed verdict: one clear pass, two neutral panels
where there was no gap to close, and one failure on data whose gap the preset
was never meant to address. Read together with the earlier five panels, the
preset reliably raises coverage where real series are strongly seasonal on a
moving level (tourism, hospital, M1 monthly) and is neutral to slightly
negative as a **substitute** for existing breadth on panels without that
behaviour. It should therefore be shipped as an **opt-in addition on top of**
`balanced_pool` or `pretraining_pool`, documented with these results, and not
folded into either default. Forecasting utility remains unmeasured.

## Limits

One preset configuration, four panels, twelve overlapping fold–draw pairs per
panel, no confidence intervals. The pretraining-pool control shares most of
its prototypes with the balanced pool; the MAR control is a weak source. Car
parts has aligned calendars and 76% zeros, at the edge of what the twelve
features describe. Coverage is conditional on the features, distance, and
radii. Forecasting utility is not measured.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/synforecast-matplotlib .venv/bin/python \
  benchmarks/validate_seasonal_preset.py --download
```

Downloads the two Monash archives (under 1 MB) and requires the cached M4
Quarterly and Monthly training CSVs. `summary.json` records the protocol,
declared criterion, panel metadata, splits, per-fold scores and union
coverage for all four sources, aggregates, attribution, nearest-prototype
counts, gate results, and the post-hoc three-way union. The twelve
`*_details.json.gz` files hold per-query distances and neighbours, fitted
parameters, candidate IDs, and generator provenance and configurations; they
are reproducible, local, and gitignored.
