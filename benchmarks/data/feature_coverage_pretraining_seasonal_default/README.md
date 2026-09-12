# Should `include_seasonal` be the `pretraining_pool` default?

**No, not on this evidence.** The pre-declared rule required no harm on every
panel and a gain on at least three of four strongly seasonal target panels.
Adding the eight `seasonal_pool` instances to `pretraining_pool` **raises
coverage on every quarterly and monthly panel** (by 2 to 14 points, with
large gains on tourism and M1) but **lowers it by 3 to 6 points on weekly,
daily, and intermittent panels**, where the extra instances displace share
without adding matching behaviour. Six panels fail no-harm and two of four
targets pass the gain gate, so the default stays `False`. The flag is the
right tool for quarterly and monthly targets and is documented as such.

## Protocol

Two versions of the public pool are compared as equal-count corpora, exactly
as every earlier benchmark draws its sources: shuffled cycling across all
prototypes, 256 requested series per fold matched to real candidate lengths,
the first 240 shared complete slots, and real-only q90 radii. Both are the
shipped function: `pretraining_pool()` with 54 instances and
`pretraining_pool(include_seasonal=True)` with 62, at the period each panel's
recipe declares. Fifteen panels cover every frequency except yearly, where the
flag leaves the pool unchanged: fresh M4 samples with a new seed at Quarterly,
Monthly, Weekly, Daily, and Hourly, and the ten Monash panels used earlier
(tourism quarterly and monthly, hospital, traffic weekly and hourly, weather,
M3 quarterly and monthly, M1 monthly, car parts). New split and generation
seeds throughout; four folds × three draws per panel.

**Declared rule.** Paired change is coverage with the flag minus coverage
without, per fold–draw pair, summarised as the median over draws per fold.
*No harm*: no fold below −2 points. *Gain*: at least +5 points in three of
four folds. *Flip the default* only if no-harm holds on every panel and gain
holds on at least three of tourism quarterly, tourism monthly, hospital, and
M1 monthly.

## Results

Median full-feature q90 coverage across four folds × three draws:

| Panel | Frequency | Without flag | With flag | Real baseline | Paired change | Pairs higher | No harm | Gain |
|---|---|---:|---:|---:|---:|---:|---|---|
| M4 Quarterly (fresh) | Q | 77.3% | 84.4% | 88.3% | +1.6 | 7/12 | **no** (−3.1) | no |
| M4 Monthly (fresh) | M | 73.4% | 82.8% | 91.4% | +6.2 | 11/12 | yes | no |
| M4 Weekly | W | 74.2% | 71.9% | 93.0% | −5.5 | 4/12 | **no** | no |
| M4 Daily | D | 43.0% | 42.2% | 89.1% | −4.7 | 2/12 | **no** | no |
| M4 Hourly | H | 0.8% | 2.3% | 85.9% | +0.8 | 6/12 | yes | no |
| tourism quarterly | Q | 46.9% | 65.6% | 89.1% | +10.2 | 11/12 | yes | **yes** |
| hospital | M | 61.7% | 65.6% | 89.8% | +2.3 | 8/12 | yes | no |
| traffic weekly | W | 72.7% | 71.9% | 88.3% | −0.8 | 5/12 | **no** (−3.1) | no |
| weather | D | 47.7% | 46.1% | 89.8% | 0.0 | 4/12 | **no** (−10.9 in one fold) | no |
| traffic hourly | H | 0.0% | 0.0% | 82.8% | 0.0 | 3/12 | yes | no |
| M3 quarterly | Q | 76.6% | 79.7% | 85.9% | +2.3 | 8/12 | yes | no |
| M3 monthly | M | 73.4% | 76.6% | 91.4% | +4.7 | 10/12 | yes | no |
| tourism monthly | M | 13.3% | 42.2% | 92.2% | +14.1 | 11/12 | yes | no (two folds at +4.7) |
| M1 monthly | M | 60.2% | 72.7% | 90.6% | +8.6 | 12/12 | yes | **yes** |
| car parts | M | 52.3% | 50.8% | 91.4% | −3.1 | 3/12 | **no** | no |

The pattern is clean. Where the panel's period is 4 or 12 and its series are
seasonal, the flag helps: the median change is positive on all eight
quarterly and monthly panels except car parts, and the seasonal instances
supply 17–72% of nearest neighbours there. Where the panel is weekly or daily,
the flag builds a 52-step or 7-step cycle that these panels do not exhibit;
the instances are nearest neighbours for 0–16% of queries but take 13% of the
corpus, so the pool loses a few points. On hourly panels the flag is neutral
(the instances are close to many queries but the pools' hourly gap is a
different problem). Car parts, intermittent with 76% zeros, loses 3 points
for the same displacement reason.

Against the declared rule: six panels fail no-harm (four of them weekly,
daily, or intermittent; M4 Quarterly and traffic weekly miss by about one
point in a single fold), and two of the four target panels pass the gain
gate, tourism monthly missing on fold heterogeneity despite a +14 median.
The default is therefore **not flipped**.

## Implications

- `include_seasonal` stays `False` by default. The documentation recommends
  setting it for quarterly and monthly targets, where every panel tested
  improved, and leaving it off for weekly, daily, hourly, and intermittent
  targets, where it displaces useful breadth.
- A frequency-aware default that enables the flag only for periods 4 and 12
  would have passed no-harm on seven of eight such panels and gain on two of
  four; that rule was not declared in advance, so it is noted as a candidate
  for a future pre-declared test, not adopted.
- Forecasting utility remains unmeasured; a default change should also clear
  that bar.

## Limits

Coverage only, twelve overlapping fold–draw pairs per panel, no confidence
intervals. Both pools were drawn with the same allocation scheme, so the
comparison includes the share displacement a default change causes, which is
the intended question; it does not measure the union of the two pools. The
weekly recipe is nonseasonal by the audited convention while the flag derives
a 52-step period from the frequency. Yearly is excluded because the flag has
no effect there.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/synforecast-matplotlib .venv/bin/python \
  benchmarks/validate_pretraining_seasonal_default.py --download
```

Requires the cached M4 training CSVs and the Monash archives used by the
earlier benchmarks. `summary.json` records the protocol, rule, panel
metadata, splits, per-fold scores, paired changes, aggregates,
nearest-prototype counts, per-panel gates, and the decision. The 45
`*_details.json.gz` files hold per-query distances and neighbours, fitted
parameters, candidate IDs, and generator provenance; they are reproducible,
local, and gitignored.
