# Native feature availability across M4

All 100,000 M4 training series were checked with the twelve native candidates,
using at most the last 512 observations per series, as in the coverage pilot.
This is a complete real-panel availability audit, not a synthetic coverage
benchmark. The recommended explicit recipes retain every real series.

| Frequency | Training series | Seasonal period | Window size | Complete rows |
|---|---:|---:|---:|---:|
| Yearly | 23,000 | `None` | 5 | 23,000 |
| Quarterly | 24,000 | 4 | 4 | 24,000 |
| Monthly | 48,000 | 12 | 12 | 48,000 |
| Weekly | 359 | `None` | 10 | 359 |
| Daily | 4,227 | 7 | 7 | 4,227 |
| Hourly | 414 | 24 | 24 | 414 |

These are panel-level configurations. No period or window changes per series,
no missing values are imputed, and raw validation rejects internal gaps or
non-finite observations. Period and window must be passed explicitly to the
native extraction API; its defaults are unchanged.

## What the audit changed

The proposed 52-week annual recipe with a 52-observation window retains only
294/359 Weekly series (81.9%). All 65 exclusions are shorter than 104
observations; the shortest Weekly series has 80. Seasonal strength, lumpiness,
maximum level shift, and maximum variance shift are undefined for those
series. Retention in the shortest length quartile is only 33.0%, failing the
predeclared 95% overall / 90% shortest-quartile pilot criteria. Reducing the
window alone cannot supply the second annual cycle seasonal strength needs.

Use the separately audited M4 nonseasonal convention (`seasonal_period=None`,
`window_size=10`) for a whole-panel Weekly comparison. It retains all rows.
This deliberately does **not measure annual Weekly seasonality**: seasonal
strength and seasonal ACF are zero and constant, so they are omitted from
the fitted space. For annual-seasonality questions, report a separate
52-week analysis of the 294 eligible series, with all 65 exclusions visible
and synthetic lengths matched to that eligible population. Do not mix its
scores with the nonseasonal whole-panel scores. The initial annual recipe's
failure remains in `summary.json`; the recommendation was selected after
this availability check, not before it.

The full Yearly audit confirms why the explicit window of 5 is needed:
the unchanged default window of 10 retains 17,824/23,000 (77.5%), excluding
5,176 short series. Window 5 retains all 23,000. Yearly's two seasonality
summaries are also constant by declaration.

Daily uses a declared seven-day cycle, consistent with the generator recipe.
M4's metadata declares Daily nonseasonal; that alternative (`None`, window
10) was audited separately and also retains all 4,227 rows. Availability
cannot establish which representation is more informative. Keep the selected
seven-day assumption in every benchmark's extraction metadata. Hourly's
24-hour recipe measures daily seasonality only, not multiple seasonal cycles.

## Scope and decision

The remaining-frequency availability step is complete. Keep the twelve feature
definitions and these explicit recipes for the next benchmark. No native
feature implementation changed. This audit originally recorded
`native_candidate_v1`; the unchanged definitions have since been finalized
as `native_v1` and used in the [small six-frequency comparison](../feature_coverage_smoke/README.md).
The audit itself does not establish feature fidelity or synthetic coverage.

The 512-observation cap affects 12 Yearly, 43 Quarterly, 1,199 Monthly,
249 Weekly, 2,944 Daily, and all 414 Hourly series. Results describe those
trailing windows, not the entire history of every series. All raw observations
were checked for finite values before truncation. Each recommended recipe
also retains 100% of its shortest quartile under this cap.

## Reproduce

```bash
# Public training CSVs are downloaded only if missing; cached CSVs stay local.
.venv/bin/python benchmarks/benchmark_feature_coverage.py \
  --availability-only --download

# Fully offline once the six training CSVs are cached.
.venv/bin/python benchmarks/benchmark_feature_coverage.py --availability-only
```

The audit streams all training rows and produces only `summary.json`. It
records source URLs/hashes, extraction recipes, original/effective length
quantiles, cap counts, per-feature availability and value quantiles, constant
features, overall/shortest-quartile retention, and criterion outcomes. To keep
the summary small, exclusions have counts, the first ten examples in source
order, and a SHA-256 of sorted excluded IDs serialized as compact JSON;
rerunning the audit reconstructs the full set. No Parquet is written.

The existing pilot's Parquet caches, detailed fitted-grid JSON, and compressed
validation traces are generated local artifacts excluded by `.gitignore`.
Reports, summary JSON, and small figures remain suitable for version control.
