# Targeted composition on fresh traffic_hourly queries

A second benchmark-only composition, targeted at the feature differences the
[first round](../feature_coverage_hourly_composition/README.md) left on the
traffic panel, **raises fresh-traffic coverage over the first round in nine of
twelve paired comparisons but still fails the declared transfer criterion**.
Median full-feature q90 coverage on 256 never-used traffic queries moves from
0% (existing pools) to 10.9% (first round) to 18.8% (targeted). The first
round's earlier traffic result replicates on these fresh queries. On the M4
splits the targeted corpus retains, and slightly improves, the first round's
gains. Production presets, generators, and features are unchanged.

## Protocol

The traffic panel, its seeded 512-observation windows, the M4 Hourly sample,
the three generation seeds, and the 240 paired slots are identical to the first
round; the archive and CSV hashes and window offsets are checked. The 256
traffic IDs that served as first-round evaluation queries are excluded from
every role, and a new split seed forms four disjoint groups of 64 fresh
evaluation queries from the remaining 606 series, each with 64 fitting
references, 32 calibration queries, and 256 real candidates. The first-round
traffic attribution motivated the targeted variants, so those queries are not
reused. The M4 evaluation uses the first round's splits as a **retention
check only**: those queries informed both rounds and are not fresh evidence.
The first round's recorded M4 scores are replayed bit-for-bit for all four
earlier sources before the targeted corpus is scored.

Five sources are scored: the unchanged balanced and pretraining pools, random
MAR, the first-round composition, and the targeted composition. Radii are
calibrated on real candidates per fold, draw, and metric.

The targeted corpus keeps three first-round prototypes and replaces the never-
nearest ETS pair and the remaining variants with noisier or level-shifting
configurations, all declared before any result was inspected:

| Prototype | Class | Change from the first round |
|---|---|---|
| tsi_daily_shaped | TSIGenerator | unchanged |
| tsi_daily_weekly | TSIGenerator | unchanged |
| tsi_daily_noisy | TSIGenerator | noise 30–150% of signal; all five irregular processes, heavy tails (t with 2.5–6 df), GARCH-like |
| tsi_daily_weekly_noisy | TSIGenerator | daily/weekly harmonics with the same noisier irregular component |
| seasonal_trend_breaks | SeasonalGenerator | unchanged |
| seasonal_level_outages | SeasonalGenerator | two level shifts plus injected 12-hour drops and single-hour dips |
| energy_residential_weekend | EnergyLoadGenerator | sharper commuting peaks, weekend reduction 40, noise 15, slope change, rare deep dips |
| energy_commercial_weekend | EnergyLoadGenerator | weekend reduction 40, noise 15, extreme-weather multipliers, slope change, rare 8-hour drops |

Declared primary criterion, unchanged: the median over draws of the targeted
corpus's full-feature q90 coverage minus the best existing source's must reach
25 points in at least three of four fresh traffic folds. Declared secondary
checks: the targeted corpus must exceed the first round in more than half of
the twelve paired fresh-traffic comparisons, and its median paired change on
the M4 splits must not fall below −10 points.

## Fresh traffic queries

Median q90 coverage across four folds × three draws, 240 candidates:

| Source | Full features | Manhattan | Omit trend strength | Daily-profile shape |
|---|---:|---:|---:|---:|
| Balanced pool | 0.0% | 0.0% | 0.0% | 13.3% |
| Pretraining pool | 0.0% | 0.0% | 0.0% | 9.4% |
| Random MAR | 0.0% | 0.0% | 0.0% | 0.0% |
| First-round composition | 10.9% | 10.2% | 25.0% | 13.3% |
| **Targeted composition** | **18.8%** | **15.6%** | **27.3%** | 14.1% |
| Matched real baseline | 85.2% | 89.1% | 86.7% | 93.8% |

The existing gap holds on these queries too: the median real-minus-best-existing
q90 gap is 79.7–90.6 points per fold. The first round's coverage on fresh
queries (10.9%, per-fold gains 21.9, 10.9, 7.8, and 9.4 points) closely
replicates its earlier 10.2%, so that transfer result was not specific to the
first query set.

The targeted corpus is higher than the best existing source in all twelve
paired comparisons, by a median 18.8 points (range 7.8–26.6). Its per-fold
median gains are 26.6, 18.8, 15.6, and 14.1 points, so only one fold reaches
the 25-point threshold and the **primary criterion fails**. Paired against the
first round it gains a median 7.0 points on full features (range −1.6 to
+17.2; higher in nine of twelve comparisons), 6.2 on Manhattan, and 3.9
without trend strength; daily-profile coverage is unchanged within noise. The
secondary fresh-traffic check therefore passes. Fold radii are 1.7–2.2
standardized units, without the loose fold seen in the first round's traffic
split.

The noisy daily TSI variant is the nearest full-feature neighbour in 343 of 768
query–draw assignments (45%), the unchanged shaped daily TSI in 243 (32%), and
the sine-with-outages prototype in 90 (12%). Neither EnergyLoad variant is
nearest more than three times in either round. The targeted changes narrowed
the differences they aimed at: the median standardized real-minus-neighbour
gap fell from +1.11 to +0.67 for spectral entropy and from +0.83 to +0.39 for
maximum level shift. The remaining distance is now led by crossing points (real
queries cross their mean about one standardized unit less often; 17% of the
mean squared distance) and the ten-lag ACF sum (real lower by about one unit;
13%), with trend strength third. Real traffic series combine low within-day
autocorrelation with long stretches on one side of the mean, which none of the
present configurations produce together.

## M4 retention check

| Source | Full features | Manhattan | Omit trend strength | Daily-profile shape |
|---|---:|---:|---:|---:|
| Best existing pool | 1.6% | 18.8% | 44.5% | 68.8% |
| First-round composition | 67.2% | 66.4% | 71.9% | 68.0% |
| Targeted composition | 69.5% | 71.9% | 77.3% | 71.1% |
| Matched real baseline | 90.6% | 90.6% | 91.4% | 89.8% |

The targeted corpus passes the earlier M4 criterion (per-fold median gains
81.3, 37.5, 75.0, and 29.7 points) and changes paired full-feature coverage
against the first round by a median +5.5 points, higher in nine of twelve
comparisons, so the retention check passes. The range is wide, −32.8 to +17.2:
in one draw, folds 1 and 3 fall from 54.7% and 53.1% to 26.6% and 20.3%, so
the noisier variants cost coverage on some M4 queries while helping others.
Daily-profile coverage rises by a median 3.1 points (higher in ten of twelve),
mostly through the sine-with-outages prototype. These M4 numbers are on
previously used queries and do not add independent evidence.

## Implications

Two rounds of configuration-only composition now show a consistent picture.
The M4 Hourly gap is largely closed by combining daily cycles with slow level
changes. The traffic panel's gap is only partly closed: each targeted change
moved the coordinates it aimed at, and coverage roughly doubled over the first
round, but the corpus remains far from the real baseline and below the declared
threshold. The remaining traffic differences concern within-day irregularity
combined with sustained excursions from the mean, not the components tested so
far. This is not sufficient evidence for a public preset change. Production
defaults stay unchanged.

## Limits

One targeted configuration was evaluated, chosen from the first round's
traffic attribution and not retuned afterwards. Fresh traffic queries share
candidates, references, and calibration series with the first round's roles,
and the same three synthetic draws are reused across panels and rounds; the
twelve fold–draw measurements are not independent replications. Coverage
remains sensitive to the distance definition and fold radii. Forecasting
utility is not measured.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/synforecast-matplotlib .venv/bin/python \
  benchmarks/validate_traffic_composition.py
```

Requires the completed first-round `summary.json`, the cached M4 Hourly
training CSV, and the cached Monash `traffic_hourly_dataset.zip`; pass
`--download` to fetch either. `summary.json` records the protocol, recipe,
excluded IDs, fresh traffic splits, per-fold scores for all five sources,
paired effects, criterion and secondary results, nearest-prototype counts, and
trace hashes. The six `*_details.json.gz` files hold per-query distances and
neighbours, fitted parameters, candidate IDs, and both compositions'
configurations and provenance; they are reproducible, local, and gitignored.
