# Seasonal-strength composition with held-out validation panels

Two benchmark-only corpora of configured public generators were scored against
the unchanged pools on the two panels that motivated them (tourism quarterly,
hospital monthly) and on three panels that did not (M3 quarterly, M3 monthly,
tourism monthly). The first, strongly seasonal recipe is **worse than the
pools everywhere**. The moderated second recipe **fails the declared
replacement criterion on every validation panel**, but as an **addition to the
balanced pool at equal candidate count it raises coverage on all five panels**,
by 3 to 32 points over adding the pretraining pool instead, and most on the
tourism panels where the pools are weakest. Presets remain unchanged; this is
the first positive held-out evidence for a preset *addition* rather than a
replacement, and forecasting utility is still unmeasured.

## Protocol

Design panels reuse the [third-panel validation](../feature_coverage_third_panels/README.md)
exactly: its splits, generation seeds, fold-specific draws, and 240 matched
slots. The three existing sources' recorded fold scores are replayed
bit-for-bit before the compositions are scored. Validation panels use a new
split seed and new generation seeds with the same four-fold protocol: at most
640 series, complete-case rows, 64 references, 32 calibration queries, 64
fresh queries and 256 real candidates per fold, real-only radii. Archive
hashes are recorded.

| Role | Panel | Series | Used | Length |
|---|---|---:|---:|---|
| design | tourism_quarterly | 427 | 427 | 30–130 |
| design | hospital (monthly) | 767 | 640 | 84 |
| validation | m3_quarterly | 756 | 640 | 24–72 |
| validation | m3_monthly | 1428 | 640 | 66–144 |
| validation | tourism_monthly | 366 | 366 | 91–333 |

**Round 1** was declared before any result: eight prototypes with strong
period-4 or period-12 seasonality, multiplicative and additive ETS with fixed
seasonal factors, TSI with amplitude 1–4 and low noise, and sines with level
or slope breaks. On the design panels it produced near-deterministic
seasonality (seasonal strength 0.9–1.0, spectral entropy below 0.1 on monthly,
lag-1 ACF near zero at period 4), far from the real distribution (median
seasonal strength 0.56 on tourism quarterly, 0.22 on hospital, under
substantial noise and with persistent lag-1 dependence).

**Round 2** was declared after inspecting round 1 on the design panels only:
the same classes with noise at 30–600% of the seasonal signal, AR(1) and
Student-t irregular components, a persistent-noise variant, a weakly seasonal
variant, noisier ETS with faster level adaptation, and noisier sines with
breaks. Its feature ranges overlap the real ones on both design panels. Both
rounds are scored on every panel; nothing was changed after the validation
panels were run.

Declared primary criterion, per validation panel: the median over draws of
the composition's full-feature q90 coverage minus the best existing source's
must reach 25 points in three of four folds, with the composition scored as a
**replacement** corpus of 240 candidates. The **augmented-pool analysis** was
added after the replacement results were seen and is therefore post hoc: it
compares the balanced pool plus a composition (480 candidates) with the
balanced pool plus the pretraining pool (480 candidates), using the saved
per-query distances and each fold's real-calibrated radius, so both sides have
equal density and the control is simply more of the existing corpus.

## Replacement scores

Median full-feature q90 coverage across four folds × three draws, 240
candidates per source:

| Panel | Role | Balanced | Pretraining | Round 1 | Round 2 | Real baseline | Existing gap flag |
|---|---|---:|---:|---:|---:|---:|---|
| tourism_quarterly | design | 46.9% | 42.2% | 20.3% | **57.8%** | 91.4% | material |
| hospital | design | 57.8% | 53.9% | 2.3% | 45.3% | 85.2% | minor |
| m3_quarterly | validation | 82.8% | 80.5% | 4.7% | 26.6% | 86.7% | none |
| m3_monthly | validation | 68.0% | 66.4% | 3.1% | 42.2% | 85.9% | minor |
| tourism_monthly | validation | 10.9% | 8.6% | 5.5% | **37.5%** | 85.9% | material |

Round 1 is below the best pool in all sixty paired fold–draw comparisons.
Round 2 beats the best pool on tourism quarterly by a median 11.7 points
(higher in nine of twelve) and on tourism monthly by 30.5 points (higher in
all twelve, per-fold median gains 37.5, 12.5, 21.9, and 35.9), but only two
tourism-monthly folds reach 25 points, so the **primary criterion fails on
every validation panel**. On M3 quarterly, where the pools have no gap, and
on M3 monthly and hospital, the narrow round-2 corpus covers 11–52 points
fewer queries than the pools when it replaces them. Tourism monthly is a new
finding in itself: the pools cover 11% of its queries against a 86% real
baseline, a 70–73 point gap in every fold.

Remaining round-2 distances are still led by seasonal strength (real queries
more seasonal by 0.4–0.5 standardized units on the validation panels) and by
first-difference ACF; the persistent-noise TSI variant is the nearest neighbour
in 59% of M3 quarterly assignments and the noisy additive ETS in 38% of
tourism quarterly ones.

## Augmented pool at equal candidate count

Median q90 coverage of 480-candidate unions and the paired change against the
balanced-plus-pretraining control:

| Panel | Role | Balanced + pretraining | Balanced + round 1 | Balanced + round 2 | Paired change, round 2 | Pairs higher |
|---|---|---:|---:|---:|---:|---:|
| tourism_quarterly | design | 53.1% | 59.4% | **78.1%** | +29.7 | 12/12 |
| hospital | design | 62.5% | 58.6% | **72.7%** | +7.8 | 8/12 |
| m3_quarterly | validation | 83.6% | 83.6% | **86.7%** | +3.1 | 9/12 |
| m3_monthly | validation | 75.0% | 70.3% | **84.4%** | +9.4 | 12/12 |
| tourism_monthly | validation | 13.3% | 18.0% | **42.2%** | +32.0 | 12/12 |

Adding the round-2 corpus to the balanced pool is never worse than adding the
pretraining pool in the median and is higher in 53 of 60 fold–draw pairs. The
gain is largest where the pools were weakest and small where they were already
near the real baseline. Adding round 1 is roughly neutral or slightly negative.
A union can never lower coverage relative to the balanced pool alone; the
comparison shows what the extra 240 candidates buy when they are configured
compositions rather than more of the existing corpus.

## Implications

The replacement criterion was the wrong yardstick for a corpus meant to fill a
gap: a narrow seasonal corpus cannot stand in for the pools on broad panels,
and the declared test says so. The post-hoc augmented comparison is the
question a preset change actually poses, and its answer is positive on all
five panels, including the three that did not inform the recipe. Together
with the Hourly rounds, the evidence now supports proposing an **additive,
opt-in "strongly seasonal" preset** built from configured `TSIGenerator`,
`ETSGenerator`, and `SeasonalGenerator` instances, provided a pre-declared
augmented-pool criterion is passed on a further held-out panel and its
forecasting utility is measured. Production defaults are unchanged.

## Limits

Round 2 was informed by the design panels, and the augmented analysis was
specified after the replacement results were known; both are disclosed above
and neither touched the validation panels' data before scoring. The union
comparison uses the pretraining pool as a same-size control, which shares
most of its prototypes with the balanced pool. Fold–draw measurements overlap
in real roles and are not independent replications; there are no confidence
intervals. Tourism monthly has only 366 series, so its folds share most
candidates. Coverage remains conditional on the twelve features, the distance,
and fold radii. Forecasting utility is not measured.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/synforecast-matplotlib .venv/bin/python \
  benchmarks/validate_seasonal_composition.py --download
# recompute only the union analysis from saved details
.venv/bin/python benchmarks/validate_seasonal_composition.py --augment-only
```

Requires the completed third-panel `summary.json` and the Monash archives
(about 1 MB for the three validation panels). `summary.json` records both
recipes, the round protocol, panel metadata, splits, per-fold scores and
retention for all five sources, aggregates, attribution, nearest-prototype
counts, criteria, paired effects, and the augmented-pool analysis. The fifteen
`*_details.json.gz` files hold per-query distances and neighbours, fitted
parameters, candidate IDs, and generator configurations and provenance; they
are reproducible, local, and gitignored.
