"""Investigate Hourly coverage with fixed real queries, redraws, and metric sensitivity.

Reuses native_v1 and public generator configurations without tuning either.
Only a compact summary and illustrative figure are intended for version control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from _env import environment_metadata
from benchmark_feature_coverage import (
    extract,
    generate_matched,
    matched_complete_ids,
    sample_training,
    save_json,
    training_file,
)
from validate_feature_coverage import fit_spaces, save_details

from synforecast._features import FEATURE_NAMES, FEATURE_SCHEMA, classical_decompose

CORPORA = ("balanced_pool", "pretraining_pool", "synforecast_mar")
SIZES = (60, 240)
QUANTILES = (0.5, 0.9, 0.95)


def nearest_metric(query, candidates, metric="euclidean"):
    delta = query[:, None, :] - candidates[None, :, :]
    if metric == "euclidean":
        distances = np.sqrt(np.sum(delta**2, axis=2))
    elif metric == "manhattan":
        distances = np.sum(np.abs(delta), axis=2)
    else:
        raise ValueError(f"Unknown metric {metric}")
    indices = np.argmin(distances, axis=1)
    return distances[np.arange(len(query)), indices], indices


def metric_scores(calibration, queries, real_candidates, synthetic, anchor, metric):
    calibration_distances, _ = nearest_metric(calibration, real_candidates, metric)
    radii = np.quantile(calibration_distances, QUANTILES)
    anchor_radii = np.quantile(
        nearest_metric(calibration, anchor, metric)[0], QUANTILES
    )
    real_distances, _ = nearest_metric(queries, real_candidates, metric)
    distances, indices = nearest_metric(queries, synthetic, metric)
    return {
        "radii": radii,
        "anchor_radii": anchor_radii,
        "coverage": [np.mean(distances <= r) for r in radii],
        "real_baseline_coverage": [np.mean(real_distances <= r) for r in radii],
        "fixed_anchor_coverage": [np.mean(distances <= r) for r in anchor_radii],
        "distance_quantiles_p10_p50_p90": np.quantile(distances, [0.1, 0.5, 0.9]),
        "real_distance_quantiles_p10_p50_p90": np.quantile(
            real_distances, [0.1, 0.5, 0.9]
        ),
        "distances": distances,
        "neighbour_indices": indices,
    }


def distance_attribution(queries, synthetic, indices, names):
    """Decompose actual nearest-neighbour squared distances, not causal effects."""
    delta = queries - synthetic[indices]
    squared = delta**2
    total = squared.sum(axis=1, keepdims=True)
    fractions = np.divide(squared, total, out=np.zeros_like(squared), where=total > 0)
    return {
        name: {
            "mean_squared_distance_fraction": np.mean(fractions[:, j]),
            "median_real_minus_neighbour_standardized": np.median(delta[:, j]),
        }
        for j, name in enumerate(names)
    }


def feature_rows(series):
    frame = extract(series, 24, window_size=24)
    return dict(
        zip(
            frame["unique_id"].to_list(),
            frame.select(FEATURE_NAMES).to_numpy(),
            strict=True,
        )
    )


def component_variance_ratios(values):
    """Context for strength ratios; correlated components need not sum to one."""
    values = np.asarray(values)
    total = np.var(values)
    components = classical_decompose(values, 24)
    return {
        name: float(np.var(component) / total) if total > 0 else 0.0
        for name, component in zip(
            ("trend", "seasonal", "remainder"), components, strict=True
        )
    }


def prepare_real(args):
    source, metadata = training_file(args.cache, "Hourly", args.download)
    sampled, _ = sample_training(source, "Hourly", 128, args.reference_seed, 512)
    ids = np.asarray(list(sampled))
    order = np.random.default_rng(args.reference_seed).permutation(128)
    generation_order = ids[order[:64]].tolist()
    split = {
        "reference": sorted(generation_order),
        "calibration": sorted(ids[order[64:96]].tolist()),
        "evaluation": sorted(ids[order[96:]].tolist()),
    }
    all_series, _ = sample_training(source, "Hourly", 414, args.reference_seed, 512)
    unused = [uid for uid in all_series if uid not in sampled]
    extra = (
        np.random.default_rng(args.reference_seed + 1)
        .permutation(unused)[:192]
        .tolist()
    )
    candidate_ids = generation_order + extra
    rows = feature_rows(all_series)

    def matrix(selected):
        return np.array([rows[uid] for uid in selected])

    reference, calibration, evaluation = [
        matrix(split[name]) for name in ("reference", "calibration", "evaluation")
    ]
    _, parameters = fit_spaces(reference, [calibration, evaluation])
    if len(parameters["kept_columns"]) != len(FEATURE_NAMES):
        raise ValueError(
            "Hourly investigation expects all twelve real-reference features to vary"
        )
    if any(len(values) != 512 for values in all_series.values()):
        raise ValueError(
            "Hourly investigation expects 512-observation trailing windows"
        )
    return all_series, rows, split, candidate_ids, parameters, metadata


def standardized(values, parameters):
    return (values - np.asarray(parameters["mean"])) / np.asarray(parameters["scale"])


def evaluate_candidates(real_rows, split, candidate_ids, synthetic_rows, parameters):
    reference, cal, query = [
        standardized(np.array([real_rows[uid] for uid in split[name]]), parameters)
        for name in ("reference", "calibration", "evaluation")
    ]
    real_candidates = standardized(
        np.array([real_rows[uid] for uid in candidate_ids]), parameters
    )
    components = np.asarray(parameters["components"])
    reports = {}
    for name, rows in synthetic_rows.items():
        synthetic = standardized(
            np.array([rows[uid] for uid in candidate_ids]), parameters
        )
        schemes = {
            "full": (np.eye(len(FEATURE_NAMES)), "euclidean"),
            "manhattan": (np.eye(len(FEATURE_NAMES)), "manhattan"),
            "pca2": (components.T, "euclidean"),
        }
        schemes.update(
            {
                f"omit_{feature}": (
                    np.delete(np.eye(len(FEATURE_NAMES)), j, axis=1),
                    "euclidean",
                )
                for j, feature in enumerate(FEATURE_NAMES)
            }
        )
        reports[name] = {
            scheme: metric_scores(
                cal @ projection,
                query @ projection,
                real_candidates @ projection,
                synthetic @ projection,
                reference @ projection,
                metric,
            )
            for scheme, (projection, metric) in schemes.items()
        }
        reports[name]["attribution"] = distance_attribution(
            query, synthetic, reports[name]["full"]["neighbour_indices"], FEATURE_NAMES
        )
    return reports


def draw_candidates(real, ids, seed):
    generated, provenance, failures, configs = generate_matched(
        {uid: real[uid] for uid in ids}, 24, "h", seed, include_pretraining=True
    )
    frames = {
        name: extract(values, 24, failures[name], window_size=24)
        for name, values in generated.items()
    }
    valid = set(matched_complete_ids(frames))
    common = [uid for uid in ids if uid in valid]
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
        generated,
        rows,
        common,
        {
            "provenance": provenance,
            "failures": failures,
            "configs": configs,
            "n_requested_per_corpus": len(ids),
            "n_shared_complete": len(common),
            "n_retained_by_source": {
                name: int(
                    np.isfinite(frame.select(FEATURE_NAMES).to_numpy())
                    .all(axis=1)
                    .sum()
                )
                for name, frame in frames.items()
            },
        },
    )


def replay_original(args, real, real_rows, split, candidate_ids, parameters):
    generated, rows, common, generation = draw_candidates(
        real, candidate_ids[:64], args.reference_seed + 1000
    )
    # The original benchmark sorts the shared cohort by ID.
    common = sorted(common)
    scores = evaluate_candidates(real_rows, split, common, rows, parameters)
    prior = json.loads((args.smoke / "summary.json").read_text())
    if prior["seed"] != args.reference_seed:
        raise ValueError("Reference seed must match the original smoke benchmark")
    original = prior["groups"]["Hourly"]["distances"]
    for name in CORPORA:
        for space in ("full", "pca2"):
            np.testing.assert_allclose(
                scores[name][space]["coverage"],
                original[space][name]["synthetic_covered"],
                rtol=0,
                atol=0,
            )
            np.testing.assert_allclose(
                scores[name][space]["radii"],
                original[space][name]["calibration_radii"],
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                scores[name][space]["distance_quantiles_p10_p50_p90"][1],
                original[space][name]["median_distance"],
                rtol=1e-12,
            )
    baseline = scores["balanced_pool"]["full"]["distances"]
    order = np.argsort(baseline, kind="stable")
    selected = order[np.rint(np.array([0.1, 0.5, 0.9]) * (len(order) - 1)).astype(int)]
    examples = []
    for q in selected:
        uid = split["evaluation"][q]
        examples.append(
            {
                "real_id": uid,
                "selection": "p10/p50/p90 rank of original balanced full-feature distance",
                "real_values": real[uid],
                "real_features": real_rows[uid],
                "real_component_variance_ratios": component_variance_ratios(real[uid]),
                "neighbours": {
                    name: {
                        "candidate_id": common[
                            scores[name]["full"]["neighbour_indices"][q]
                        ],
                        "prototype": generation["provenance"][name][
                            common[scores[name]["full"]["neighbour_indices"][q]]
                        ],
                        "distance": scores[name]["full"]["distances"][q],
                        "values": generated[name][
                            common[scores[name]["full"]["neighbour_indices"][q]]
                        ],
                        "features": rows[name][
                            common[scores[name]["full"]["neighbour_indices"][q]]
                        ],
                        "component_variance_ratios": component_variance_ratios(
                            generated[name][
                                common[scores[name]["full"]["neighbour_indices"][q]]
                            ]
                        ),
                    }
                    for name in CORPORA
                },
            }
        )
    return {
        "reproduces_original": True,
        "scores": scores,
        "generation": generation,
        "candidate_ids": common,
        "examples": examples,
    }


def compact_scores(scores):
    return {
        name: {
            scheme: {
                key: value
                for key, value in values.items()
                if key not in ("distances", "neighbour_indices")
            }
            for scheme, values in schemes.items()
        }
        for name, schemes in scores.items()
    }


def summarize_repeats(runs):
    summary = {}
    for size in SIZES:
        summary[size] = {}
        for source in CORPORA:
            selected = [r["sizes"][size][source] for r in runs]
            summary[size][source] = {}
            for scheme in (
                "full",
                "manhattan",
                "pca2",
                *(f"omit_{f}" for f in FEATURE_NAMES),
            ):
                summary[size][source][scheme] = {
                    key: {
                        "median": np.median([r[scheme][key] for r in selected], axis=0),
                        "seed_min_max": [
                            np.min([r[scheme][key] for r in selected], axis=0),
                            np.max([r[scheme][key] for r in selected], axis=0),
                        ],
                    }
                    for key in (
                        "coverage",
                        "real_baseline_coverage",
                        "fixed_anchor_coverage",
                        "distance_quantiles_p10_p50_p90",
                    )
                }
            summary[size][source]["attribution"] = {
                feature: {
                    key: np.median([r["attribution"][feature][key] for r in selected])
                    for key in (
                        "mean_squared_distance_fraction",
                        "median_real_minus_neighbour_standardized",
                    )
                }
                for feature in FEATURE_NAMES
            }
    return summary


def plot_examples(examples, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(examples), 4, figsize=(15, 8), sharex=True, squeeze=False
    )
    for row, example in enumerate(examples):
        signals = [("Real " + example["real_id"], example["real_values"])]
        signals.extend(
            (f"{name}\n{n['prototype']} · distance {n['distance']:.2f}", n["values"])
            for name, n in example["neighbours"].items()
        )
        for col, (label, values) in enumerate(signals):
            values = np.asarray(values)
            scale = values.std()
            normalized = (values - values.mean()) / scale if scale > 0 else values * 0
            ax = axes[row, col]
            ax.plot(np.arange(344, 512), normalized[-168:], linewidth=1)
            ax.set_title(label, fontsize=8)
            ax.set_xlabel("Hour within 512-observation window", fontsize=8)
        axes[row, 0].set_ylabel("Standardized value")
    fig.suptitle(
        "Original Hourly queries and full-feature nearest synthetic examples\nLast 168 hours shown; normalization and features use all 512; each panel has its own y-axis"
    )
    fig.tight_layout()
    fig.savefig(output / "nearest_examples.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/feature_coverage/m4")
    )
    parser.add_argument(
        "--smoke", type=Path, default=Path("benchmarks/data/feature_coverage_smoke")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/data/feature_coverage_hourly")
    )
    parser.add_argument("--reference-seed", type=int, default=20260913)
    parser.add_argument("--redraws", type=int, default=10)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.redraws < 2:
        parser.error("--redraws must be at least 2")
    real, real_rows, split, candidate_ids, parameters, source = prepare_real(args)
    original = replay_original(args, real, real_rows, split, candidate_ids, parameters)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_examples(original["examples"], args.output)
    original_details = save_details(args.output / "original_details.json.gz", original)
    report = {
        "completed": False,
        "environment": environment_metadata(),
        "schema": FEATURE_SCHEMA,
        "source": source,
        "original_summary_sha256": hashlib.sha256(
            (args.smoke / "summary.json").read_bytes()
        ).hexdigest(),
        "protocol": {
            "reference_seed": args.reference_seed,
            "redraws": args.redraws,
            "seasonal_period": 24,
            "window_size": 24,
            "length": 512,
            "n_reference_fit": 64,
            "n_calibration": 32,
            "n_evaluation": 32,
            "n_requested_per_corpus_per_seed": 256,
            "nested_complete_candidate_sizes": SIZES,
            "real_queries_fixed": True,
            "feature_definitions_fixed": True,
            "metric_sensitivity": "full Euclidean primary; Manhattan, PCA2, and 12 individual feature omissions, each with its own real calibration",
            "radii": "q50/q90/q95 size-matched real calibration plus a fixed 64-reference anchor; no synthetic threshold fitting",
            "selection": "first 60/240 shared complete IDs in generation order, with exactly matching real candidates; remaining successes unused; no replacement draws",
            "variability": "generator seed range conditional on the fixed real sample, not a confidence interval",
        },
        "split_ids": split,
        "generation_order_ids": candidate_ids,
        "parameters": parameters,
        "original": {
            "reproduces_original": True,
            "scores": compact_scores(original["scores"]),
            "details": original_details,
            "examples": [
                {
                    k: v
                    for k, v in example.items()
                    if k not in ("real_values", "neighbours")
                }
                | {
                    "neighbours": {
                        name: {k: v for k, v in neighbour.items() if k != "values"}
                        for name, neighbour in example["neighbours"].items()
                    }
                }
                for example in original["examples"]
            ],
        },
        "runs": [],
    }
    for repeat in range(args.redraws):
        seed = args.reference_seed + 1000 + 1000000 * (repeat + 1)
        _, rows, common, generation = draw_candidates(real, candidate_ids, seed)
        if len(common) < max(SIZES):
            raise ValueError(
                f"Seed {seed} retained only {len(common)} shared rows; cannot compare declared sizes without replacement"
            )
        sizes = {
            n: evaluate_candidates(real_rows, split, common[:n], rows, parameters)
            for n in SIZES
        }
        for name in CORPORA:
            # Same queries, metric, and nested candidate sets: distances cannot grow.
            assert np.all(
                sizes[240][name]["full"]["distances"]
                <= sizes[60][name]["full"]["distances"]
            )
        details = save_details(
            args.output / f"seed_{seed}_details.json.gz",
            {"generation": generation, "shared_complete_ids": common, "scores": sizes},
        )
        report["runs"].append(
            {
                "seed": seed,
                "n_shared_complete": len(common),
                "n_retained_by_source": generation["n_retained_by_source"],
                "failure_counts": {
                    name: len(errors) for name, errors in generation["failures"].items()
                },
                "nearest_prototype_counts": {
                    n: {
                        name: dict(
                            Counter(
                                generation["provenance"][name][common[i]]
                                for i in scores[name]["full"]["neighbour_indices"]
                            )
                        )
                        for name in CORPORA
                    }
                    for n, scores in sizes.items()
                },
                "sizes": {n: compact_scores(scores) for n, scores in sizes.items()},
                "details": details,
            }
        )
        save_json(args.output / "summary.json", report)
        print(
            f"Hourly seed {seed}: shared {len(common)}/256; q90 at 240 candidates "
            + ", ".join(
                f"{name}={sizes[240][name]['full']['coverage'][1]:.1%}"
                for name in CORPORA
            ),
            flush=True,
        )
    report["aggregate"] = summarize_repeats(report["runs"])
    report["completed"] = True
    # Per-seed detailed curves remain local; keep primary per-seed comparisons in Git.
    for run in report["runs"]:
        run["sizes"] = {
            n: {
                name: {
                    scheme: scores[scheme] for scheme in ("full", "manhattan", "pca2")
                }
                for name, scores in corpora.items()
            }
            for n, corpora in run["sizes"].items()
        }
    save_json(args.output / "summary.json", report)
    print(f"Saved Hourly investigation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
