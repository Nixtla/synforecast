"""Test whether existing native generators reproduce the Hourly control gains.

A benchmark-only composition corpus is assembled from public generator classes
with non-default configurations. No generator, preset, or feature definition is
changed. The corpus is scored against the unchanged pools on the same fresh M4
Hourly splits, slots, and radii, then on an independent Hourly panel (Monash
traffic_hourly). Only summaries and this report are intended for Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from _env import environment_metadata
from benchmark_feature_coverage import sample_training, save_json, training_file
from investigate_hourly_coverage import CORPORA, draw_candidates, feature_rows
from validate_feature_coverage import save_details
from validate_hourly_splits import (
    GATE,
    GENERATION_SEEDS,
    SPLIT_SEED,
    aggregate_runs,
    compact,
    fresh_splits,
    persistence_gate,
    scores_for_split,
)

from synforecast import interpretable_pool
from synforecast._features import FEATURE_NAMES, FEATURE_SCHEMA
from synforecast.generators import (
    EnergyLoadGenerator,
    ETSGenerator,
    SeasonalGenerator,
    TSIGenerator,
)

COMPOSITION = "native_composition"
SOURCES = (*CORPORA, COMPOSITION)
METRICS = ("full", "manhattan", "omit_trend_strength", "profile")
# Distinct from the existing corpus offsets 0/100000/200000 used by
# generate_matched, so composition draws never share a slot seed.
COMPOSITION_SEED_OFFSET = 300000
# Declared before any composition result was inspected. Criterion reuses the
# persistence gate's scale: median over draws of composition-minus-best-existing
# full-feature q90 coverage must reach 25 points in at least three of four folds.
CRITERION = {
    "minimum_median_composition_minus_best_existing_q90_gain": GATE[
        "minimum_median_real_minus_best_source_q90_gap"
    ],
    "minimum_splits": GATE["minimum_splits"],
}
TRANSFER = {
    "dataset": "monash_traffic_hourly",
    "description": "862 hourly San Francisco Bay Area freeway occupancy rates, 2015-2016 (Lai et al., 2017), Monash Forecasting Repository",
    "url": "https://zenodo.org/records/4656132/files/traffic_hourly_dataset.zip?download=1",
    "zip_sha256": "3db12ba866a9c9d3c8109b7b6d189a990c38d0e5002fa2617022157358d08299",
    "member": "traffic_hourly_dataset.tsf",
    "window": "one seeded uniform 512-observation window per series, so the panel is not tied to a single calendar interval",
}
# Benchmark-only configuration of public generators. Each prototype combines a
# 24-step daily cycle with slow level changes; amplitude variation and richer
# daily shapes are included where the class supports them. Ranges were chosen
# to mirror the declared control recipe's relative magnitudes, not fitted.
COMPOSITION_RECIPE = {
    "tsi_daily_shaped": "TSI: one period-24 harmonic with 2f/3f overtones and a slow amplitude envelope, drifting trend, low noise",
    "tsi_daily_weekly": "TSI: two harmonics drawn from periods {24, 168}, overtones and envelopes with probability 0.5, drifting trend",
    "ets_AAdM_24": "ETS(A,Ad,M) period 24: random-walk level with damped trend; multiplicative seasonality scales amplitude with level",
    "ets_MAdM_24": "ETS(M,Ad,M) period 24: as above with multiplicative error",
    "seasonal_trend_breaks": "Sine period 24 plus two random-location slope changes",
    "seasonal_drift_level_breaks": "Sine period 24 plus linear drift and two random-location level shifts",
    "energy_residential_drift": "EnergyLoad residential daily/weekly profile plus one slope change",
    "energy_commercial_drift": "EnergyLoad commercial daily/weekly profile plus one slope change",
}


def native_composition_pool(length=512, freq="h", seed=None, engine="polars"):
    """Public generator classes with non-default settings; presets untouched."""
    base = {
        "min_length": length,
        "max_length": length,
        "freq": freq,
        "engine": engine,
    }

    def _seed(i):
        return None if seed is None else seed + i

    tsi = {
        "seasonal_periods": [24.0],
        "n_seasonal_range": (1, 1),
        "harmonics_prob": 1.0,
        "amplitude_modulation_prob": 1.0,
        "trend_types": ["linear", "piecewise_linear", "damped", "logistic"],
        "trend_slope_range": (-2.0, 2.0),
        "seasonal_amplitude_range": (0.5, 2.0),
        "noise_scale_range": (0.02, 0.4),
        "irregular_types": ["gaussian", "ar1"],
        "ar1_phi_range": (0.3, 0.9),
        "level_range": (5.0, 10.0),
        "scale_range": (1.0, 1.0),
    }
    ets = {
        "trend_type": "add",
        "seasonal_type": "mul",
        "seasonal_period": 24,
        "level": 10.0,
        "trend": 0.0,
        "alpha": 0.15,
        "beta": 0.02,
        "gamma": 0.02,
        "phi": 0.9,
        "damped": True,
    }
    return [
        TSIGenerator(**base, seed=_seed(0), alias="tsi_daily_shaped", **tsi),
        TSIGenerator(
            **base,
            seed=_seed(1),
            alias="tsi_daily_weekly",
            **{
                **tsi,
                "seasonal_periods": [24.0, 168.0],
                "n_seasonal_range": (2, 2),
                "harmonics_prob": 0.5,
                "amplitude_modulation_prob": 0.5,
            },
        ),
        ETSGenerator(
            **base,
            seed=_seed(2),
            alias="ets_AAdM_24",
            error_type="add",
            noise_std=0.3,
            **ets,
        ),
        ETSGenerator(
            **base,
            seed=_seed(3),
            alias="ets_MAdM_24",
            error_type="mul",
            noise_std=0.03,
            **ets,
        ),
        SeasonalGenerator(
            **base,
            seed=_seed(4),
            alias="seasonal_trend_breaks",
            seasonality_period=24,
            seasonality_amplitude=1.0,
            noise_level=0.1,
            base_level=10.0,
            changepoints=True,
            num_changepoints=2,
            changepoint_type="trend",
            changepoint_trend_changes=[0.003, -0.005],
        ),
        SeasonalGenerator(
            **base,
            seed=_seed(5),
            alias="seasonal_drift_level_breaks",
            seasonality_period=24,
            seasonality_amplitude=1.0,
            trend=0.0015,
            noise_level=0.1,
            base_level=10.0,
            changepoints=True,
            num_changepoints=2,
            changepoint_type="level",
            changepoint_level_changes=[0.5, -0.4],
        ),
        EnergyLoadGenerator(
            **base,
            seed=_seed(6),
            alias="energy_residential_drift",
            load_type="residential",
            changepoints=True,
            num_changepoints=1,
            changepoint_type="trend",
            changepoint_trend_changes=[0.1],
        ),
        EnergyLoadGenerator(
            **base,
            seed=_seed(7),
            alias="energy_commercial_drift",
            load_type="commercial",
            changepoints=True,
            num_changepoints=1,
            changepoint_type="trend",
            changepoint_trend_changes=[-0.1],
        ),
    ]


def generate_composition(
    lengths,
    seed,
    pool_factory=native_composition_pool,
    recipe=COMPOSITION_RECIPE,
    seed_offset=COMPOSITION_SEED_OFFSET,
):
    """Same per-slot protocol as the existing pools: one public generate() call."""
    corpus_seed = seed + seed_offset
    pool = pool_factory(seed=corpus_seed)
    if [p.alias for p in pool] != list(recipe):
        raise ValueError("Composition pool aliases must match the declared recipe")
    rng = np.random.default_rng(corpus_seed)
    assigned = np.concatenate(
        [
            rng.permutation(len(pool))
            for _ in range((len(lengths) + len(pool) - 1) // len(pool))
        ]
    )[: len(lengths)]
    rng.shuffle(assigned)
    values, provenance, failures = {}, {}, {}
    for i, (uid, length) in enumerate(lengths.items()):
        prototype = pool[int(assigned[i])]
        config = prototype.model_dump(
            by_alias=True, include=set(type(prototype).model_fields)
        )
        config.update(min_length=length, max_length=length, seed=corpus_seed + i)
        provenance[uid] = prototype.alias
        try:
            generated = type(prototype)(**config).generate(1, n_jobs=1)["y"].to_numpy()
            if len(generated) != length or not np.isfinite(generated).all():
                raise ValueError(
                    f"Generated length={len(generated)}, expected={length}; "
                    f"non-finite observations={int((~np.isfinite(generated)).sum())}"
                )
            values[uid] = generated
        except (ValueError, RuntimeError) as exc:
            failures[uid] = str(exc)
    configs = [
        p.model_dump(mode="json", by_alias=True, include=set(type(p).model_fields))
        for p in pool
    ]
    return values, provenance, failures, configs


def read_tsf_hourly(path):
    """Minimal Monash .tsf reader for equal-length, complete hourly panels."""
    series = {}
    in_data = False
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.startswith("@missing") and line.split()[1] != "false":
                    raise ValueError("Only complete .tsf panels are supported")
                in_data = line == "@data"
                continue
            name, _, payload = line.split(":", 2)
            values = np.array(payload.split(","), dtype=float)
            if name in series or not np.isfinite(values).all():
                raise ValueError(f"Duplicate or non-finite series {name}")
            series[name] = values
    if not in_data or not series:
        raise ValueError("No @data section found")
    lengths = {len(v) for v in series.values()}
    if len(lengths) != 1:
        raise ValueError(f"Expected equal lengths, found {sorted(lengths)}")
    return series


def seeded_windows(series, length, seed):
    """One uniformly placed window per series; offsets are recorded."""
    rng = np.random.default_rng(seed)
    windows, offsets = {}, {}
    for uid, values in series.items():
        if len(values) < length:
            raise ValueError(f"{uid} is shorter than {length}")
        start = int(rng.integers(0, len(values) - length + 1))
        windows[uid] = values[start : start + length]
        offsets[uid] = start
    return windows, offsets


def transfer_panel(cache, download):
    archive = cache / "traffic_hourly_dataset.zip"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(
                f"Missing {archive}; use --download to fetch the Monash traffic_hourly zip"
            )
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".partial")
        print(f"Downloading {TRANSFER['url']}", flush=True)
        with (
            urlopen(TRANSFER["url"], timeout=120) as response,
            partial.open("wb") as out,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        partial.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != TRANSFER["zip_sha256"]:
        raise ValueError(f"Unexpected traffic_hourly archive hash {digest}")
    member = cache / TRANSFER["member"]
    if not member.exists():
        with zipfile.ZipFile(archive) as archive_file:
            archive_file.extract(TRANSFER["member"], cache)
    series = read_tsf_hourly(member)
    if len(series) != 862:
        raise ValueError(f"Expected 862 traffic series, found {len(series)}")
    return series, {
        **TRANSFER,
        "n_series": len(series),
        "series_length": len(next(iter(series.values()))),
    }


def best_existing(fold_scores, metric):
    return max(CORPORA, key=lambda name: fold_scores[name][metric]["coverage"][1])


def paired_effects(runs, baseline, name=COMPOSITION):
    """Median paired q90 change of the composition against a baseline scorer.

    ``baseline(run, fold, metric)`` returns the baseline metric scores for that
    fold; medians of paired changes need not equal differences of medians.
    """
    results = {}
    for metric in METRICS:
        differences, distance_changes = [], []
        for run in runs:
            for fold in run["folds"]:
                base = baseline(run, fold, metric)
                changed = fold["scores"][name][metric]
                differences.append(changed["coverage"][1] - base["coverage"][1])
                distance_changes.append(
                    changed["distance_quantiles_p10_p50_p90"][1]
                    - base["distance_quantiles_p10_p50_p90"][1]
                )
        results[metric] = {
            "median_q90_coverage_change": np.median(differences),
            "min_max_q90_change": [min(differences), max(differences)],
            "fraction_runs_with_higher_q90_coverage": np.mean(
                np.array(differences) > 0
            ),
            "median_nearest_distance_change": np.median(distance_changes),
        }
    return results


def composition_criterion(runs, name=COMPOSITION):
    gaps = []
    for fold in range(4):
        values = []
        for run in runs:
            scores = run["folds"][fold]["scores"]
            values.append(
                scores[name]["full"]["coverage"][1]
                - max(scores[other]["full"]["coverage"][1] for other in CORPORA)
            )
        gaps.append(float(np.median(values)))
    threshold = CRITERION["minimum_median_composition_minus_best_existing_q90_gain"]
    return {
        "per_split_median_gain": gaps,
        "passed": sum(gap >= threshold for gap in gaps) >= CRITERION["minimum_splits"],
    }


def prototype_neighbour_counts(scores, provenance, selected, name=COMPOSITION):
    counts = {}
    for metric in ("full", "profile"):
        counts[metric] = dict(
            Counter(
                provenance[selected[i]]
                for i in scores[name][metric]["neighbour_indices"]
            )
        )
    return counts


def check_replay(folds, prior_run, names=CORPORA):
    """Recorded sources must reproduce their recorded scores exactly."""
    for fold, recorded in zip(folds, prior_run["folds"], strict=True):
        for name in names:
            for metric in METRICS:
                for key in ("coverage", "radii", "distance_quantiles_p10_p50_p90"):
                    np.testing.assert_allclose(
                        fold["scores"][name][metric][key],
                        recorded["scores"][name][metric][key],
                        rtol=1e-12,
                        atol=0,
                    )


def evaluate_panel(real, real_rows, splits, draws, output, prefix):
    runs = []
    for seed, (synthetic, rows, selected, generation) in draws.items():
        folds, details = [], []
        for split in splits:
            scores, params, candidates = scores_for_split(
                real, real_rows, split, selected, synthetic, rows
            )
            folds.append(
                {
                    "fold": split["fold"],
                    "scores": compact(scores),
                    "composition_nearest_prototype_counts": prototype_neighbour_counts(
                        scores, generation["composition_provenance"], selected
                    ),
                }
            )
            details.append(
                {
                    "fold": split["fold"],
                    "parameters": params,
                    "real_candidate_ids": candidates,
                    "scores": scores,
                }
            )
        detail_file = save_details(
            output / f"{prefix}_{seed}_details.json.gz",
            {"generation": generation, "selected_slots": selected, "folds": details},
        )
        runs.append(
            {
                "seed": seed,
                "n_shared_complete_existing": generation["n_shared_complete"],
                "n_retained_by_source": generation["n_retained_by_source"],
                "details": detail_file,
                "folds": folds,
            }
        )
        print(
            f"{prefix} seed {seed}: full-feature q90 by split "
            + str(
                {
                    name: [f["scores"][name]["full"]["coverage"][1] for f in folds]
                    for name in SOURCES
                }
            ),
            flush=True,
        )
    return runs


def panel_summary(runs, prior_controls=None):
    summary = {
        "aggregate": aggregate_runs(runs, SOURCES),
        "existing_gate": persistence_gate(runs),
        "criterion": composition_criterion(runs),
        "paired_vs_best_existing": paired_effects(
            runs,
            lambda _run, fold, metric: fold["scores"][
                best_existing(fold["scores"], metric)
            ][metric],
        ),
        "paired_vs_each_existing": {
            name: paired_effects(
                runs, lambda _run, fold, metric, name=name: fold["scores"][name][metric]
            )
            for name in CORPORA
        },
        "composition_nearest_prototype_counts": {
            metric: dict(
                sum(
                    (
                        Counter(fold["composition_nearest_prototype_counts"][metric])
                        for run in runs
                        for fold in run["folds"]
                    ),
                    Counter(),
                )
            )
            for metric in ("full", "profile")
        },
        "composition_attribution_summary": {
            feature: {
                key: np.median(
                    [
                        fold["scores"][COMPOSITION]["attribution"][feature][key]
                        for run in runs
                        for fold in run["folds"]
                    ]
                )
                for key in (
                    "mean_squared_distance_fraction",
                    "median_real_minus_neighbour_standardized",
                )
            }
            for feature in FEATURE_NAMES
        },
    }
    if prior_controls is not None:
        by_seed = {run["paired_existing_seed"]: run for run in prior_controls}

        def control(run, fold, metric, variant):
            return by_seed[run["seed"]]["folds"][fold["fold"]]["scores"][variant][
                metric
            ]

        summary["paired_vs_controls"] = {
            variant: paired_effects(
                runs, lambda run, fold, metric, v=variant: control(run, fold, metric, v)
            )
            for variant in ("cycle", "cycle_drift_amplitude", "shape_drift_amplitude")
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/feature_coverage/m4")
    )
    parser.add_argument(
        "--transfer-cache", type=Path, default=Path(".cache/feature_coverage/monash")
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_hourly_splits/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_hourly_composition"),
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prior = json.loads(args.prior.read_text())
    if (
        prior["schema"] != FEATURE_SCHEMA
        or not prior["completed"]
        or not prior["gate_result"]["trigger_controls"]
    ):
        raise ValueError("A completed, gated fresh-split Hourly validation is required")
    if len(interpretable_pool(freq="h")) != 42:
        raise ValueError("Production interpretable_pool must remain unchanged")
    source, source_metadata = training_file(args.cache, "Hourly", args.download)
    if source_metadata["sha256"] != prior["source"]["sha256"]:
        raise ValueError("Hourly source hash differs from the fresh-split validation")
    traffic, transfer_metadata = transfer_panel(args.transfer_cache, args.download)
    real, _ = sample_training(source, "Hourly", 414, SPLIT_SEED, 512)
    excluded = prior["protocol"]["excluded_old_evaluation_ids"]
    splits = fresh_splits(list(real), excluded)
    if splits != prior["splits"]:
        raise ValueError("Fresh M4 splits do not match the prior validation")
    real_rows = feature_rows(real)
    windows, offsets = seeded_windows(traffic, 512, SPLIT_SEED)
    traffic_rows = feature_rows(windows)
    traffic_splits = fresh_splits(list(windows), [], seed=SPLIT_SEED)
    slots = [f"slot{i:03d}" for i in range(256)]
    template = {uid: np.zeros(512) for uid in slots}
    report = {
        "completed": False,
        "schema": FEATURE_SCHEMA,
        "environment": environment_metadata(),
        "source": source_metadata,
        "transfer_source": transfer_metadata,
        "prior_sha256": hashlib.sha256(args.prior.read_bytes()).hexdigest(),
        "protocol": {
            "split_seed": SPLIT_SEED,
            "generation_seeds": GENERATION_SEEDS,
            "composition_seed_offset": COMPOSITION_SEED_OFFSET,
            "same_m4_splits_slots_and_radii_as_prior": True,
            "existing_replay_checked": True,
            "n_requested_per_source": 256,
            "n_matched_candidates": 240,
            "period": 24,
            "window": 24,
            "length": 512,
            "composition_recipe": COMPOSITION_RECIPE,
            "criterion": CRITERION,
            "transfer_protocol": "same four-fold fresh-split protocol with no exclusions; same synthetic draws and slots as the M4 panel because generation depends only on length and seed",
            "limits": "benchmark-only configuration; real roles overlap across folds; three draws reused across panels; no confidence intervals; forecasting utility not measured",
        },
        "splits": splits,
        "transfer_splits": traffic_splits,
        "transfer_window_offsets": offsets,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    prior_by_seed = {run["seed"]: run for run in prior["existing_runs"]}
    draws = {}
    for seed in GENERATION_SEEDS:
        generated, rows, common, generation = draw_candidates(template, slots, seed)
        if len(common) != prior_by_seed[seed]["n_shared_complete"]:
            raise ValueError(
                f"Seed {seed} shared-complete count differs from the prior run"
            )
        selected = common[:240]
        values, provenance, failures, configs = generate_composition(
            dict.fromkeys(slots, 512), seed
        )
        frame_rows = feature_rows(values)
        incomplete = [
            uid
            for uid in selected
            if uid not in frame_rows or not np.isfinite(frame_rows[uid]).all()
        ]
        if incomplete:
            raise ValueError(
                f"Composition incomplete on paired slots {incomplete[:5]}; no replacement allowed"
            )
        generated[COMPOSITION] = values
        rows[COMPOSITION] = frame_rows
        generation.update(
            {
                "composition_provenance": provenance,
                "composition_failures": failures,
                "composition_configs": configs,
            }
        )
        generation["n_retained_by_source"][COMPOSITION] = int(
            sum(np.isfinite(frame_rows[uid]).all() for uid in frame_rows)
        )
        draws[seed] = (generated, rows, selected, generation)
        print(
            f"Seed {seed}: composition complete on {generation['n_retained_by_source'][COMPOSITION]}/256 slots, "
            f"{len(failures)} failures; prototypes {dict(Counter(provenance.values()))}",
            flush=True,
        )
    report["m4_runs"] = evaluate_panel(
        real, real_rows, splits, draws, args.output, "m4"
    )
    for run in report["m4_runs"]:
        check_replay(run["folds"], prior_by_seed[run["seed"]])
    report["m4_summary"] = panel_summary(report["m4_runs"], prior["control_runs"])
    save_json(args.output / "summary.json", report)
    print(f"M4 criterion: {report['m4_summary']['criterion']}", flush=True)
    report["transfer_runs"] = evaluate_panel(
        windows, traffic_rows, traffic_splits, draws, args.output, "traffic"
    )
    report["transfer_summary"] = panel_summary(report["transfer_runs"])
    print(
        f"Transfer existing gate: {report['transfer_summary']['existing_gate']}; "
        f"criterion: {report['transfer_summary']['criterion']}",
        flush=True,
    )
    report["completed"] = True
    for run in report["m4_runs"] + report["transfer_runs"]:
        for fold in run["folds"]:
            for scores in fold["scores"].values():
                del scores["attribution"]
    save_json(args.output / "summary.json", report)
    print(f"Saved composition validation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
