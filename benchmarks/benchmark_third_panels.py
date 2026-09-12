"""Third-panel coverage validation for every non-hourly frequency.

The cross-frequency benchmark and the Hourly chain both use M4 training data,
and Hourly was then checked on Monash traffic_hourly. This script applies the
same four-fold, three-seed protocol to one public Monash panel per remaining
frequency, so every frequency has a real panel that did not inform any design
decision. Sources, presets, recipes, and features are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlopen

import benchmark_cross_frequency as cross
import numpy as np
from _env import environment_metadata
from benchmark_feature_coverage import save_json

from synforecast._features import FEATURE_SCHEMA

# Declared before any result. One public Monash Forecasting Repository panel per
# frequency, chosen for size (>= 352 series for four fresh query groups) and for
# a domain different from the M4 competition data. Hourly is not rerun: its
# third panel is Monash traffic_hourly, reported in the composition rounds.
PANELS = {
    "Yearly": {
        "dataset": "tourism_yearly",
        "record": 4656103,
        "file": "tourism_yearly_dataset.zip",
        "sha256": "20238ddb9b11465b061bbfadcc398ef01f0896a8791199381b3edc706abf0263",
        "description": "518 yearly tourism demand series (Athanasopoulos et al., 2011)",
    },
    "Quarterly": {
        "dataset": "tourism_quarterly",
        "record": 4656093,
        "file": "tourism_quarterly_dataset.zip",
        "sha256": "fa16da64f29aa04df447756cb65a5d58f8bc3d860fc22751c76850a3ffb048c3",
        "description": "427 quarterly tourism demand series (Athanasopoulos et al., 2011)",
    },
    "Monthly": {
        "dataset": "hospital",
        "record": 4656014,
        "file": "hospital_dataset.zip",
        "sha256": "8fd4085efb4517508e30fc69247eee9b74b300765bc6e94482c5d250efba66e5",
        "description": "767 monthly hospital patient-count series, 84 observations each (Hyndman, expsmooth)",
    },
    "Weekly": {
        "dataset": "traffic_weekly",
        "record": 4656135,
        "file": "traffic_weekly_dataset.zip",
        "sha256": "df3fc458fb1d6f770451fc45c43b930c091e69e86d508410dae4843918851525",
        "description": "862 weekly San Francisco freeway occupancy series, 104 observations each (Lai et al., 2017)",
    },
    "Daily": {
        "dataset": "weather",
        "record": 4654822,
        "file": "weather_dataset.zip",
        "sha256": "adc9195b4b842673892fec1ba22cced9c42d5c4cba9a261f455ffbe96db7847b",
        "description": "3010 daily Australian weather series: rain, minimum and maximum temperature, solar radiation (Sparks et al., bomrang)",
    },
}
SPLIT_SEED = 20260917
GENERATION_SEEDS = (61261913, 62261913, 63261913)
N_SAMPLE = 640
LENGTH = 512


def read_tsf(path):
    """Monash .tsf reader: any equal or unequal lengths, no missing values."""
    series = {}
    in_data = False
    with Path(path).open(encoding="latin-1") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.startswith("@missing") and line.split()[1] != "false":
                    raise ValueError("Only complete .tsf panels are supported")
                in_data = line == "@data"
                continue
            head, payload = line.rsplit(":", 1)
            name = head.split(":", 1)[0]
            if "?" in payload:
                raise ValueError(f"Missing values in series {name}")
            values = np.array(payload.split(","), dtype=float)
            if name in series or not np.isfinite(values).all():
                raise ValueError(f"Duplicate or non-finite series {name}")
            series[name] = values
    if not in_data or not series:
        raise ValueError("No @data section found")
    return series


def fetch_panel(cache, spec, download):
    archive = cache / spec["file"]
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f"Missing {archive}; use --download to fetch it")
        archive.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://zenodo.org/records/{spec['record']}/files/{spec['file']}?download=1"
        partial = archive.with_suffix(".partial")
        print(f"Downloading {url}", flush=True)
        with urlopen(url, timeout=300) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        partial.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"Unexpected archive hash for {spec['file']}: {digest}")
    member = archive.with_suffix(".tsf").name
    if not (cache / member).exists():
        with zipfile.ZipFile(archive) as archive_file:
            archive_file.extract(member, cache)
    return read_tsf(cache / member)


def prepare_real(series, seed, n_sample=N_SAMPLE, length=LENGTH):
    """Seeded sample of at most n_sample series; one seeded window when longer."""
    rng = np.random.default_rng(seed)
    ids = list(series)
    if len(ids) > n_sample:
        ids = sorted(rng.choice(ids, n_sample, replace=False).tolist())
    real, offsets = {}, {}
    for uid in ids:
        values = series[uid]
        if len(values) > length:
            start = int(rng.integers(0, len(values) - length + 1))
            real[uid] = values[start : start + length]
            offsets[uid] = start
        else:
            real[uid] = values
            offsets[uid] = 0
    return real, offsets


def run_panel(group, cache, download, output):
    spec = PANELS[group]
    frequency, period, window = cross.panel_recipes()[group]
    series = fetch_panel(cache, spec, download)
    real, offsets = prepare_real(series, SPLIT_SEED)
    rows, dropped = cross.complete_real(cross.feature_rows(real, period, window))
    if len(rows) < 352:
        raise ValueError(f"{group}: only {len(rows)} complete series; need 352")
    splits = cross.fresh_splits(list(rows), [], seed=SPLIT_SEED)
    lengths = [len(real[uid]) for uid in rows]
    panel = {
        "source": {
            k: spec[k] for k in ("dataset", "record", "file", "sha256", "description")
        },
        "frequency": frequency,
        "seasonal_period": period,
        "window_size": window,
        "n_series_in_panel": len(series),
        "n_sampled": len(real),
        "n_complete": len(rows),
        "dropped_incomplete_ids": dropped,
        "n_windowed": int(sum(len(series[uid]) > LENGTH for uid in real)),
        "window_offsets": offsets,
        "effective_length_quantiles_min_p25_p50_p75_max": np.quantile(
            lengths, [0, 0.25, 0.5, 0.75, 1]
        ),
        "splits": splits,
    }
    panel.update(
        cross.evaluate_splits(
            real,
            rows,
            splits,
            period,
            frequency,
            window,
            GENERATION_SEEDS,
            output,
            group,
        )
    )
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/feature_coverage/monash")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_third_panels"),
    )
    parser.add_argument("--groups", nargs="+", choices=list(PANELS))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "completed": False,
        "schema": FEATURE_SCHEMA,
        "environment": environment_metadata(),
        "protocol": {
            "split_seed": SPLIT_SEED,
            "generation_seeds": GENERATION_SEEDS,
            "n_sample": N_SAMPLE,
            "length": f"at most {LENGTH} observations; one seeded uniformly placed window per longer series, whole series otherwise",
            "recipes": {
                group: {"frequency": f, "seasonal_period": p, "window_size": w}
                for group, (f, p, w) in cross.panel_recipes().items()
            },
            "flags": cross.FLAGS,
            "folds": "same four-fold fresh-query protocol, matched candidates, and real-only calibration as the cross-frequency benchmark",
            "hourly": "not rerun; Monash traffic_hourly results are in feature_coverage_hourly_composition and feature_coverage_traffic_composition",
            "limits": "one panel per frequency; aligned calendars for hospital and traffic_weekly; real roles overlap across folds; no confidence intervals; forecasting utility not measured",
        },
        "panels": {},
    }
    for group in args.groups or list(PANELS):
        report["panels"][group] = run_panel(
            group, args.cache, args.download, args.output
        )
        save_json(args.output / "summary.json", report)
    report["completed"] = True
    save_json(args.output / "summary.json", report)
    print(f"Saved third-panel validation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
