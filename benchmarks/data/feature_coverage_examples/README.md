# Example series for the coverage notebook

`examples.json` holds a few real series and their nearest synthetic neighbours
so the [feature-coverage notebook](../../../nbs/docs/capabilities/feature_coverage.ipynb)
can show what covered and uncovered queries look like without network access.
It is illustration, not a benchmark result.

## Selection

For each panel, take the first fold and the first generation seed of the
corresponding benchmark, rank the evaluation queries by their full-feature
nearest distance to the balanced pool, and keep the queries at the 10th, 50th,
and 90th rank percentiles. Selection is by rank, never by appearance.

| Panel | Benchmark | Neighbours stored |
|---|---|---|
| Monthly, Daily | cross-frequency, fold 0, seed 51261913 | balanced pool, pretraining pool |
| M4 Hourly | fresh splits, fold 0, seed 31261913 | balanced pool, first-round composition, targeted composition |
| Traffic | targeted round's fresh traffic splits, fold 0, seed 31261913 | balanced pool, first-round composition, targeted composition |

Synthetic neighbours are regenerated from the recorded seeds, so their values
are exactly those scored in the benchmarks. Real values are the trailing 512
(at most) training observations used by the benchmarks; M4 and the Monash
traffic panel are public datasets. Values are rounded to six significant
digits. The notebook standardizes each series before plotting.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/synforecast-matplotlib .venv/bin/python \
  benchmarks/export_coverage_examples.py
```

Requires the cached M4 training CSVs and the Monash traffic archive (pass
`--download` to fetch), plus the committed fresh-split and first-round
composition summaries for the excluded query IDs.
