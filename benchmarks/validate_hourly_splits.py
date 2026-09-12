"""Validate the Hourly finding on fresh query roles, then gated paired controls.

No native definitions or public presets are changed. Controls are diagnostic
series, not a proposed preset update. Only summaries are intended for Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from _env import environment_metadata
from benchmark_feature_coverage import sample_training, save_json, training_file
from investigate_hourly_coverage import (
    CORPORA,
    distance_attribution,
    draw_candidates,
    feature_rows,
    metric_scores,
    standardized,
)
from validate_feature_coverage import fit_spaces, save_details

from synforecast._features import FEATURE_NAMES, FEATURE_SCHEMA

# Declared before inspecting fresh query results. Reuse draws across real splits.
SPLIT_SEED = 20260914
GENERATION_SEEDS = (31261913, 32261913, 33261913)
CONTROL_RECIPE = {
    "period": 24,
    "length": 512,
    "noise_sd_uniform": [0.01, 0.2],
    "linear_total_change_uniform": [-0.8, 0.8],
    "slow_level_amplitude_uniform": [0.1, 0.5],
    "slow_period": 168,
    "log_amplitude_uniform": [0.0, 0.4],
    "second_harmonic_amplitude_uniform": [0.1, 0.4],
    "phase": "independent uniform 0..2pi; all variants share each draw's components",
}
GATE = {"minimum_median_real_minus_best_source_q90_gap": 0.25, "minimum_splits": 3}
VARIANTS = (
    "cycle",
    "cycle_drift",
    "cycle_amplitude",
    "cycle_drift_amplitude",
    "shape_drift_amplitude",
)


def fresh_splits(ids, excluded, seed=SPLIT_SEED):
    if not set(excluded).issubset(ids):
        raise ValueError("Excluded old query IDs must belong to the real panel")
    eligible = [uid for uid in ids if uid not in set(excluded)]
    if len(set(ids)) != len(ids) or len(eligible) < 352:
        raise ValueError("Need unique IDs and at least 352 eligible Hourly series")
    order = np.random.default_rng(seed).permutation(eligible).tolist()
    splits = []
    for fold in range(4):
        evaluation = order[64 * fold : 64 * (fold + 1)]
        remaining = [uid for uid in order if uid not in set(evaluation)]
        shuffled = (
            np.random.default_rng(seed + fold + 1).permutation(remaining).tolist()
        )
        splits.append(
            {
                "fold": fold,
                "reference": shuffled[:64],
                "candidates": shuffled[:256],
                "calibration": shuffled[256:288],
                "evaluation": evaluation,
            }
        )
    return splits


def daily_profile(values):
    """Phase-insensitive shape uses a normalized average of complete 24-hour cycles."""
    values = np.asarray(values)
    if len(values) < 24 or not np.isfinite(values).all():
        raise ValueError("Daily profile needs at least 24 finite observations")
    profile = values[: len(values) // 24 * 24].reshape(-1, 24).mean(axis=0)
    profile -= profile.mean()
    scale = np.sqrt(np.mean(profile**2))
    return profile / scale if scale > 1e-12 else np.zeros(24)


def profile_nearest(query, candidates):
    # One global phase offset per average daily profile, no warping or fitting.
    best = np.full((len(query), len(candidates)), np.inf)
    for offset in range(24):
        delta = query[:, None, :] - np.roll(candidates, offset, axis=1)[None, :, :]
        best = np.minimum(best, np.sqrt(np.mean(delta**2, axis=2)))
    indices = np.argmin(best, axis=1)
    return best[np.arange(len(query)), indices], indices


def profile_scores(calibration, query, candidates, synthetic):
    radii = np.quantile(profile_nearest(calibration, candidates)[0], [0.5, 0.9, 0.95])
    distances, neighbours = profile_nearest(query, synthetic)
    real_distances = profile_nearest(query, candidates)[0]
    return {
        "radii": radii,
        "coverage": [np.mean(distances <= r) for r in radii],
        "real_baseline_coverage": [np.mean(real_distances <= r) for r in radii],
        "distance_quantiles_p10_p50_p90": np.quantile(distances, [0.1, 0.5, 0.9]),
        "real_distance_quantiles_p10_p50_p90": np.quantile(
            real_distances, [0.1, 0.5, 0.9]
        ),
        "distances": distances,
        "neighbour_indices": neighbours,
    }


def controlled_variants(seed, count=256):
    """Paired components specified by CONTROL_RECIPE; no real values are inputs."""
    rng = np.random.default_rng(seed)
    time = np.arange(512)
    variants = {name: {} for name in VARIANTS}
    for i in range(count):
        uid = f"slot{i:03d}"
        phase, slow_phase, amplitude_phase, harmonic_phase = rng.uniform(
            0, 2 * np.pi, 4
        )
        wave = np.sin(2 * np.pi * time / 24 + phase)
        noise = rng.normal(size=512) * rng.uniform(*CONTROL_RECIPE["noise_sd_uniform"])
        drift = rng.uniform(
            *CONTROL_RECIPE["linear_total_change_uniform"]
        ) * np.linspace(-0.5, 0.5, 512)
        drift += rng.uniform(*CONTROL_RECIPE["slow_level_amplitude_uniform"]) * np.sin(
            2 * np.pi * time / 168 + slow_phase
        )
        envelope = np.exp(
            rng.uniform(*CONTROL_RECIPE["log_amplitude_uniform"])
            * np.sin(2 * np.pi * time / 168 + amplitude_phase)
        )
        harmonic = rng.uniform(
            *CONTROL_RECIPE["second_harmonic_amplitude_uniform"]
        ) * np.sin(4 * np.pi * time / 24 + harmonic_phase)
        variants["cycle"][uid] = wave + noise
        variants["cycle_drift"][uid] = wave + drift + noise
        variants["cycle_amplitude"][uid] = envelope * wave + noise
        variants["cycle_drift_amplitude"][uid] = envelope * wave + drift + noise
        variants["shape_drift_amplitude"][uid] = (
            envelope * (wave + harmonic) + drift + noise
        )
    return variants


def scores_for_split(real, real_rows, split, slot_ids, synthetic, synthetic_rows):
    ref, cal, query = [
        np.array([real_rows[uid] for uid in split[key]])
        for key in ("reference", "calibration", "evaluation")
    ]
    _, parameters = fit_spaces(ref, [cal, query])
    if parameters["kept_columns"] != list(range(len(FEATURE_NAMES))):
        raise ValueError("Fresh Hourly fit must retain all twelve variable features")
    real_ids = [split["candidates"][int(slot[4:])] for slot in slot_ids]
    reference, calibration, evaluation, candidates = [
        standardized(x, parameters)
        for x in (ref, cal, query, np.array([real_rows[uid] for uid in real_ids]))
    ]
    profiles = [
        np.array([daily_profile(real[uid]) for uid in ids])
        for ids in (split["calibration"], split["evaluation"], real_ids)
    ]
    projections = {
        "full": (np.eye(12), "euclidean"),
        "manhattan": (np.eye(12), "manhattan"),
        "omit_trend_strength": (
            np.delete(np.eye(12), FEATURE_NAMES.index("trend_strength"), axis=1),
            "euclidean",
        ),
    }
    scores = {}
    for name, rows in synthetic_rows.items():
        values = standardized(np.array([rows[uid] for uid in slot_ids]), parameters)
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
            for scheme, (p, metric) in projections.items()
        }
        scores[name]["profile"] = profile_scores(
            *profiles,
            np.array([daily_profile(synthetic[name][uid]) for uid in slot_ids]),
        )
        scores[name]["attribution"] = distance_attribution(
            evaluation, values, scores[name]["full"]["neighbour_indices"], FEATURE_NAMES
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


def persistence_gate(runs):
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
    passed = (
        sum(
            gap >= GATE["minimum_median_real_minus_best_source_q90_gap"] for gap in gaps
        )
        >= GATE["minimum_splits"]
    )
    return {"per_split_median_gap": gaps, "trigger_controls": passed}


def aggregate_runs(runs, names):
    summary = {}
    for name in names:
        rows = [fold["scores"][name] for run in runs for fold in run["folds"]]
        summary[name] = {}
        for metric in ("full", "manhattan", "omit_trend_strength", "profile"):
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
                )
            }
    return summary


def paired_control_effects(runs):
    results = {}
    for variant in VARIANTS[1:]:
        results[variant] = {}
        for metric in ("full", "manhattan", "omit_trend_strength", "profile"):
            differences, distance_changes = [], []
            for run in runs:
                for fold in run["folds"]:
                    base, changed = [
                        fold["scores"][name][metric] for name in ("cycle", variant)
                    ]
                    differences.append(changed["coverage"][1] - base["coverage"][1])
                    distance_changes.append(
                        changed["distance_quantiles_p10_p50_p90"][1]
                        - base["distance_quantiles_p10_p50_p90"][1]
                    )
            results[variant][metric] = {
                "median_q90_coverage_change": np.median(differences),
                "min_max_q90_change": [min(differences), max(differences)],
                "fraction_runs_with_higher_q90_coverage": np.mean(
                    np.array(differences) > 0
                ),
                "median_nearest_distance_change": np.median(distance_changes),
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/feature_coverage/m4")
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_hourly/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_hourly_splits"),
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prior = json.loads(args.prior.read_text())
    if prior["schema"] != FEATURE_SCHEMA or not prior["completed"]:
        raise ValueError("A completed native_v1 Hourly investigation is required")
    source, source_metadata = training_file(args.cache, "Hourly", args.download)
    if source_metadata["sha256"] != prior["source"]["sha256"]:
        raise ValueError("Hourly source hash differs from the prior investigation")
    real, _ = sample_training(source, "Hourly", 414, SPLIT_SEED, 512)
    if any(len(values) != 512 for values in real.values()):
        raise ValueError("Every Hourly series must supply 512 observations")
    real_rows = feature_rows(real)
    excluded = prior["split_ids"]["evaluation"]
    splits = fresh_splits(list(real), excluded)
    slots = [f"slot{i:03d}" for i in range(256)]
    template = {uid: np.zeros(512) for uid in slots}
    report = {
        "completed": False,
        "schema": FEATURE_SCHEMA,
        "environment": environment_metadata(),
        "source": source_metadata,
        "prior_sha256": hashlib.sha256(args.prior.read_bytes()).hexdigest(),
        "protocol": {
            "split_seed": SPLIT_SEED,
            "generation_seeds": GENERATION_SEEDS,
            "excluded_old_evaluation_ids": excluded,
            "n_folds": 4,
            "n_unique_new_queries": 256,
            "n_reference": 64,
            "n_calibration": 32,
            "n_evaluation_per_fold": 64,
            "n_requested_per_source": 256,
            "n_matched_candidates": 240,
            "period": 24,
            "window": 24,
            "length": 512,
            "same_draws_reused_across_splits": True,
            "persistence_gate": GATE,
            "declared_control_recipe": CONTROL_RECIPE,
            "profile": "mean of first 21 complete days; center/unit-variance normalize 24-hour profile; min RMS over 24 circular hourly shifts",
            "limits": "disjoint new evaluation groups; reference/candidate roles overlap across folds; same three generator draws reused; no confidence intervals",
        },
        "splits": splits,
        "existing_runs": [],
        "control_runs": [],
    }
    retained_draws = []
    for seed in GENERATION_SEEDS:
        generated, rows, common, generation = draw_candidates(template, slots, seed)
        if len(common) < 240:
            raise ValueError(
                "Fewer than 240 shared complete candidates; no replacement allowed"
            )
        selected = common[:240]
        retained_draws.append((seed, selected))
        folds, details = [], []
        for split in splits:
            scores, params, candidates = scores_for_split(
                real, real_rows, split, selected, generated, rows
            )
            folds.append({"fold": split["fold"], "scores": compact(scores)})
            details.append(
                {
                    "fold": split["fold"],
                    "parameters": params,
                    "real_candidate_ids": candidates,
                    "scores": scores,
                }
            )
        detail_file = save_details(
            args.output / f"existing_{seed}_details.json.gz",
            {"generation": generation, "selected_slots": selected, "folds": details},
        )
        report["existing_runs"].append(
            {
                "seed": seed,
                "n_shared_complete": len(common),
                "n_retained_by_source": generation["n_retained_by_source"],
                "details": detail_file,
                "folds": folds,
            }
        )
        save_json(args.output / "summary.json", report)
        print(
            f"Existing seed {seed}: q90 by split "
            + str(
                {
                    name: [f["scores"][name]["full"]["coverage"][1] for f in folds]
                    for name in CORPORA
                }
            ),
            flush=True,
        )
    report["existing_summary"] = aggregate_runs(report["existing_runs"], CORPORA)
    report["gate_result"] = persistence_gate(report["existing_runs"])
    print(f"Persistence gate: {report['gate_result']}", flush=True)
    if report["gate_result"]["trigger_controls"]:
        for seed, selected in retained_draws:
            control_seed = seed + 10000000
            variants = controlled_variants(control_seed)
            rows = {name: feature_rows(values) for name, values in variants.items()}
            folds, details = [], []
            for split in splits:
                scores, params, candidates = scores_for_split(
                    real, real_rows, split, selected, variants, rows
                )
                folds.append({"fold": split["fold"], "scores": compact(scores)})
                details.append(
                    {
                        "fold": split["fold"],
                        "parameters": params,
                        "real_candidate_ids": candidates,
                        "scores": scores,
                    }
                )
            detail_file = save_details(
                args.output / f"controls_{control_seed}_details.json.gz",
                {"selected_slots": selected, "folds": details},
            )
            report["control_runs"].append(
                {
                    "seed": control_seed,
                    "paired_existing_seed": seed,
                    "details": detail_file,
                    "folds": folds,
                }
            )
            save_json(args.output / "summary.json", report)
            print(
                f"Controls seed {control_seed}: evaluated all four fresh splits",
                flush=True,
            )
        report["control_summary"] = aggregate_runs(report["control_runs"], VARIANTS)
        report["paired_control_effects"] = paired_control_effects(
            report["control_runs"]
        )
    report["completed"] = True
    report["existing_attribution_summary"] = {
        name: {
            feature: {
                key: np.median(
                    [
                        fold["scores"][name]["attribution"][feature][key]
                        for run in report["existing_runs"]
                        for fold in run["folds"]
                    ]
                )
                for key in (
                    "mean_squared_distance_fraction",
                    "median_real_minus_neighbour_standardized",
                )
            }
            for feature in FEATURE_NAMES
        }
        for name in CORPORA
    }
    # Detailed per-run attributions remain in ignored traces; keep their summary.
    for run in report["existing_runs"] + report["control_runs"]:
        for fold in run["folds"]:
            for scores in fold["scores"].values():
                del scores["attribution"]
    save_json(args.output / "summary.json", report)
    print(f"Saved fresh-split validation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
