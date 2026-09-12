"""Multi-seed cross-frequency coverage of the unchanged pools on fresh queries.

Extends the one-seed six-frequency smoke benchmark to Yearly, Quarterly,
Monthly, Weekly, and Daily with the protocol used for the Hourly fresh splits:
four disjoint query groups, three generation seeds, 256 requested and 240
matched candidates per source, and real-only calibration. It asks whether the
Hourly gap is frequency-specific. Sources, presets, and features are unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from _env import environment_metadata
from benchmark_feature_coverage import (
    AVAILABILITY_RECIPES,
    RECOMMENDED_RECIPES,
    extract,
    generate_matched,
    matched_complete_ids,
    sample_training,
    save_json,
    training_file,
)
from datasetsforecast.m4 import M4Info
from investigate_hourly_coverage import CORPORA, distance_attribution, metric_scores
from validate_feature_coverage import fit_spaces, save_details
from validate_hourly_splits import GATE, fresh_splits

from synforecast._features import FEATURE_NAMES, FEATURE_SCHEMA

FREQUENCIES = {
    "Yearly": "YS",
    "Quarterly": "QS",
    "Monthly": "MS",
    "Weekly": "W",
    "Daily": "D",
}
# Declared before any result: new split and generation seeds; Hourly is not
# rerun because its fresh-split validation already used this protocol.
SPLIT_SEED = 20260916
GENERATION_SEEDS = (51261913, 52261913, 53261913)
N_SAMPLE = 640
N_MATCHED = 240
MINIMUM_MATCHED = 200
# Two operational flags on the median-over-seeds real-minus-best-source q90 gap:
# the Hourly persistence gate's 25 points and a 10-point minor flag. Neither
# is a significance test.
FLAGS = {
    "material": GATE["minimum_median_real_minus_best_source_q90_gap"],
    "minor": 0.10,
    "minimum_splits": GATE["minimum_splits"],
}
METRICS = ("full", "manhattan", "omit_trend_strength")


def panel_recipes():
    """Audited period/window recipes recommended for whole-panel extraction."""
    return {
        group: (
            FREQUENCIES[group],
            *AVAILABILITY_RECIPES[group][RECOMMENDED_RECIPES[group]],
        )
        for group in FREQUENCIES
    }


def feature_rows(series, period, window):
    frame = extract(series, period, window_size=window)
    return dict(
        zip(
            frame["unique_id"].to_list(),
            frame.select(FEATURE_NAMES).to_numpy(),
            strict=True,
        )
    )


def complete_real(rows):
    """Complete-case real series; incomplete IDs are reported, never imputed."""
    complete = {uid: row for uid, row in rows.items() if np.isfinite(row).all()}
    dropped = sorted(set(rows) - set(complete))
    return complete, dropped


def project(values, parameters):
    kept = parameters["kept_columns"]
    mean, scale = np.asarray(parameters["mean"]), np.asarray(parameters["scale"])
    return (np.asarray(values)[:, kept] - mean[kept]) / scale[kept]


def projections(parameters):
    kept = [FEATURE_NAMES[i] for i in parameters["kept_columns"]]
    identity = np.eye(len(kept))
    schemes = {"full": (identity, "euclidean"), "manhattan": (identity, "manhattan")}
    if "trend_strength" in kept:
        schemes["omit_trend_strength"] = (
            np.delete(identity, kept.index("trend_strength"), axis=1),
            "euclidean",
        )
    return kept, schemes


def draw_fold(real, split, period, frequency, window, seed):
    """Exact per-slot length matching against this fold's real candidates."""
    slots = [f"slot{i:03d}" for i in range(len(split["candidates"]))]
    template = {
        slot: real[uid] for slot, uid in zip(slots, split["candidates"], strict=True)
    }
    corpora, provenance, failures, configs = generate_matched(
        template, period, frequency, seed, include_pretraining=True
    )
    frames = {
        name: extract(values, period, failures[name], window_size=window)
        for name, values in corpora.items()
    }
    valid = set(matched_complete_ids(frames))
    common = [slot for slot in slots if slot in valid]
    rows = {
        name: dict(
            zip(
                frame["unique_id"].to_list(),
                frame.select(FEATURE_NAMES).to_numpy(),
                strict=True,
            )
        )
        for name, frame in frames.items()
    }
    return (
        rows,
        common,
        {
            "provenance": provenance,
            "failures": {name: dict(errors) for name, errors in failures.items()},
            "configs": configs,
            "n_shared_complete": len(common),
            "n_retained_by_source": {
                name: int(sum(np.isfinite(row).all() for row in rows[name].values()))
                for name in rows
            },
        },
    )


def score_fold(real_rows, split, selected, rows, names=CORPORA):
    ref, cal, query = [
        np.array([real_rows[uid] for uid in split[key]])
        for key in ("reference", "calibration", "evaluation")
    ]
    _, parameters = fit_spaces(ref, [cal, query])
    kept, schemes = projections(parameters)
    real_ids = [split["candidates"][int(slot[4:])] for slot in selected]
    reference, calibration, evaluation, candidates = [
        project(x, parameters)
        for x in (ref, cal, query, np.array([real_rows[uid] for uid in real_ids]))
    ]
    scores = {}
    for name in names:
        values = project(np.array([rows[name][slot] for slot in selected]), parameters)
        if not np.isfinite(values).all():
            raise ValueError(f"Incomplete standardized features for {name}")
        scores[name] = {
            scheme: metric_scores(
                calibration @ p,
                evaluation @ p,
                candidates @ p,
                values @ p,
                reference @ p,
                metric,
            )
            for scheme, (p, metric) in schemes.items()
        }
        scores[name]["attribution"] = distance_attribution(
            evaluation, values, scores[name]["full"]["neighbour_indices"], kept
        )
    return scores, parameters, real_ids


def compact(scores):
    return {
        name: {
            scheme: {
                k: v
                for k, v in row.items()
                if k not in ("distances", "neighbour_indices")
            }
            for scheme, row in schemes.items()
        }
        for name, schemes in scores.items()
    }


def gap_flags(runs):
    gaps = []
    for fold in range(4):
        values = []
        for run in runs:
            sources = run["folds"][fold]["scores"]
            baseline = sources[CORPORA[0]]["full"]["real_baseline_coverage"][1]
            values.append(
                baseline - max(sources[name]["full"]["coverage"][1] for name in CORPORA)
            )
        gaps.append(float(np.median(values)))
    return {
        "per_split_median_gap": gaps,
        **{
            f"{level}_gap": sum(gap >= FLAGS[level] for gap in gaps)
            >= FLAGS["minimum_splits"]
            for level in ("material", "minor")
        },
    }


def aggregate(runs, names=CORPORA):
    summary = {}
    for name in names:
        rows = [fold["scores"][name] for run in runs for fold in run["folds"]]
        summary[name] = {}
        for metric in METRICS:
            if any(metric not in r for r in rows):
                continue
            summary[name][metric] = {
                key: {
                    "median": np.median([r[metric][key] for r in rows], axis=0),
                    "min_max": [
                        np.min([r[metric][key] for r in rows], axis=0),
                        np.max([r[metric][key] for r in rows], axis=0),
                    ],
                }
                for key in (
                    "coverage",
                    "real_baseline_coverage",
                    "distance_quantiles_p10_p50_p90",
                    "real_distance_quantiles_p10_p50_p90",
                )
            }
        features = rows[0]["attribution"].keys()
        summary[name]["attribution"] = {
            feature: {
                key: np.median([r["attribution"][feature][key] for r in rows])
                for key in (
                    "mean_squared_distance_fraction",
                    "median_real_minus_neighbour_standardized",
                )
            }
            for feature in features
        }
        summary[name]["nearest_prototype_counts"] = dict(
            sum(
                (
                    Counter(fold["nearest_prototype_counts"][name])
                    for run in runs
                    for fold in run["folds"]
                ),
                Counter(),
            )
        )
    return summary


def run_panel(args, group, output):
    frequency, period, window = panel_recipes()[group]
    source, metadata = training_file(args.cache, group, args.download)
    n_sample = min(args.n_sample, M4Info[group].n_ts)
    real, original_lengths = sample_training(source, group, n_sample, SPLIT_SEED, 512)
    rows, dropped = complete_real(feature_rows(real, period, window))
    splits = fresh_splits(list(rows), [], seed=SPLIT_SEED)
    lengths = {uid: len(real[uid]) for uid in rows}
    panel = {
        "source": metadata,
        "frequency": frequency,
        "seasonal_period": period,
        "window_size": window,
        "n_sampled": n_sample,
        "n_complete": len(rows),
        "dropped_incomplete_ids": dropped,
        "n_capped_at_512": int(sum(n > 512 for n in original_lengths.values())),
        "effective_length_quantiles_min_p25_p50_p75_max": np.quantile(
            list(lengths.values()), [0, 0.25, 0.5, 0.75, 1]
        ),
        "splits": splits,
    }
    panel.update(
        evaluate_splits(
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


def evaluate_splits(
    real, rows, splits, period, frequency, window, seeds, output, label
):
    """Score the unchanged sources on given real splits; reusable across panels."""
    panel = {"runs": []}
    for seed in seeds:
        folds, details = [], []
        for split in splits:
            fold_seed = seed + 1000 * split["fold"]
            synthetic_rows, common, generation = draw_fold(
                real, split, period, frequency, window, fold_seed
            )
            if len(common) < MINIMUM_MATCHED:
                raise ValueError(
                    f"{label} seed {fold_seed}: only {len(common)} shared complete slots"
                )
            selected = common[:N_MATCHED]
            scores, parameters, real_ids = score_fold(
                rows, split, selected, synthetic_rows
            )
            folds.append(
                {
                    "fold": split["fold"],
                    "seed": fold_seed,
                    "n_shared_complete": len(common),
                    "n_matched": len(selected),
                    "n_retained_by_source": generation["n_retained_by_source"],
                    "kept_features": [
                        FEATURE_NAMES[i] for i in parameters["kept_columns"]
                    ],
                    "nearest_prototype_counts": {
                        name: dict(
                            Counter(
                                generation["provenance"][name][selected[i]]
                                for i in scores[name]["full"]["neighbour_indices"]
                            )
                        )
                        for name in CORPORA
                    },
                    "scores": compact(scores),
                }
            )
            details.append(
                {
                    "fold": split["fold"],
                    "seed": fold_seed,
                    "generation": generation,
                    "selected_slots": selected,
                    "real_candidate_ids": real_ids,
                    "parameters": parameters,
                    "scores": scores,
                }
            )
        detail_file = save_details(output / f"{label}_{seed}_details.json.gz", details)
        panel["runs"].append({"seed": seed, "details": detail_file, "folds": folds})
        print(
            f"{label} seed {seed}: full-feature q90 by fold "
            + str(
                {
                    name: [
                        round(float(f["scores"][name]["full"]["coverage"][1]), 4)
                        for f in folds
                    ]
                    for name in CORPORA
                }
            )
            + f"; real {[round(float(f['scores'][CORPORA[0]]['full']['real_baseline_coverage'][1]), 4) for f in folds]}",
            flush=True,
        )
    panel["summary"] = aggregate(panel["runs"])
    panel["gap_flags"] = gap_flags(panel["runs"])
    print(f"{label}: {panel['gap_flags']}", flush=True)
    for run in panel["runs"]:
        for fold in run["folds"]:
            for scores in fold["scores"].values():
                del scores["attribution"]
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/feature_coverage/m4")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_cross_frequency"),
    )
    parser.add_argument("--groups", nargs="+", choices=list(FREQUENCIES))
    parser.add_argument("--n-sample", type=int, default=N_SAMPLE)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.n_sample < 352:
        parser.error("--n-sample must be at least 352 for four fresh query groups")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "completed": False,
        "schema": FEATURE_SCHEMA,
        "environment": environment_metadata(),
        "protocol": {
            "split_seed": SPLIT_SEED,
            "generation_seeds": GENERATION_SEEDS,
            "fold_seed": "generation seed + 1000 * fold; corpus offsets as in the smoke benchmark",
            "n_sample": args.n_sample,
            "n_folds": 4,
            "n_reference": 64,
            "n_calibration": 32,
            "n_evaluation_per_fold": 64,
            "n_requested_per_source": 256,
            "n_matched_candidates": N_MATCHED,
            "minimum_matched": MINIMUM_MATCHED,
            "length": "trailing 512 training observations; exact per-slot length matching to each fold's real candidates, so draws are fold-specific",
            "recipes": {
                group: {"frequency": f, "seasonal_period": p, "window_size": w}
                for group, (f, p, w) in panel_recipes().items()
            },
            "real_filter": "complete-case native_v1 rows only; dropped IDs recorded",
            "flags": FLAGS,
            "hourly": "not rerun; see feature_coverage_hourly_splits",
            "limits": "real roles overlap across folds; twelve fold-draw measurements per panel are not independent; no confidence intervals; forecasting utility not measured",
        },
        "panels": {},
    }
    for group in args.groups or list(FREQUENCIES):
        report["panels"][group] = run_panel(args, group, args.output)
        save_json(args.output / "summary.json", report)
    report["completed"] = True
    save_json(args.output / "summary.json", report)
    print(f"Saved cross-frequency benchmark to {args.output}", flush=True)


if __name__ == "__main__":
    main()
