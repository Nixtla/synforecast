"""Corroborate the cached native pilot using feature distances and grid sensitivity.

Runs offline; no feature extraction or synthetic generation is repeated. These
are experimental diagnostics, not a released distance API or a DTW evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from _env import environment_metadata
from benchmark_feature_coverage import safe_json, save_json

from synforecast._coverage import embed_features, grid_edges, grid_occupancy
from synforecast._features import FEATURE_NAMES, FEATURE_SCHEMA

QUANTILES = (0.5, 0.9, 0.95)
SIZES = (64, 128)
CORPORA = ("balanced_pool", "synforecast_mar")


def load_cache(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing generated cache {path}. Parquet caches are intentionally gitignored. "
            "Rebuild both pilot seeds using the commands in "
            "benchmarks/data/feature_coverage_pilot/README.md before validation."
        )
    frame = pl.read_parquet(path).sort("unique_id")
    values = frame.select(FEATURE_NAMES).to_numpy()
    ids = np.asarray(frame["unique_id"].to_list())
    valid = np.isfinite(values).all(axis=1)
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate series IDs in {path}")
    return (
        values[valid],
        ids[valid],
        {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_input": len(ids),
            "n_retained": int(valid.sum()),
            "dropped_ids": ids[~valid],
        },
    )


def fit_spaces(reference, matrices):
    """Share the real-only scaler between full-feature and PCA distances."""
    fitted = embed_features([reference, *matrices], "pca", "real", None)
    params = fitted.parameters
    kept = fitted.kept
    mean, scale = np.asarray(params["mean"]), np.asarray(params["scale"])
    full = [(x[:, kept] - mean[kept]) / scale[kept] for x in [reference, *matrices]]
    return {"full": full, "pca2": fitted.coordinates}, params


def nearest(query, candidates):
    """Exact Euclidean distances; inputs are small pilot matrices, already scaled."""
    if not len(query) or not len(candidates):
        raise ValueError("Nearest-neighbour inputs must be non-empty")
    squared = np.sum((query[:, None, :] - candidates[None, :, :]) ** 2, axis=2)
    indices = np.argmin(squared, axis=1)
    return np.sqrt(squared[np.arange(len(query)), indices]), indices


def distance_scores(calibration, evaluation, real_candidates, synthetic):
    """Calibrate only with real data; never use evaluation queries for thresholds."""
    calibration_distances, _ = nearest(calibration, real_candidates)
    real_distances, _ = nearest(evaluation, real_candidates)
    distances, indices = nearest(evaluation, synthetic)
    radii = np.quantile(calibration_distances, QUANTILES)
    return {
        "calibration_radii": radii,
        "calibration_distances": calibration_distances,
        "real_baseline_distances": real_distances,
        "synthetic_distances": distances,
        "synthetic_neighbour_indices": indices,
        "real_baseline_covered": [np.mean(real_distances <= r) for r in radii],
        "synthetic_covered": [np.mean(distances <= r) for r in radii],
        "median_distance": float(np.median(distances)),
        "real_baseline_median_distance": float(np.median(real_distances)),
    }


def split_indices(n_real, synthetic_sizes, seed):
    if n_real != 256 or any(n < max(SIZES) for n in synthetic_sizes.values()):
        raise ValueError("Validation needs 256 complete real rows and >=128 per corpus")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_real)
    return {
        "reference": order[:128],
        "calibration": order[128:192],
        "evaluation": order[192:],
        "synthetic": {
            name: rng.permutation(n)[:128] for name, n in synthetic_sizes.items()
        },
    }


def grid_scores(real, synthetic, edges):
    """All queries remain in the series denominator, even outside the grid."""
    ro, cells, inside = grid_occupancy(real, edges)
    so, _, synth_inside = grid_occupancy(synthetic, edges)
    covered = np.zeros(len(real), dtype=bool)
    covered[inside] = so[cells[inside, 0], cells[inside, 1]]
    return {
        "uncovered_real_cell_fraction": float(np.sum(ro & ~so) / ro.sum())
        if ro.any()
        else None,
        "uncovered_real_series_fraction": float(1 - covered.mean()),
        "real_out_of_range_fraction": float(1 - inside.mean()),
        "synthetic_out_of_range_fraction": float(1 - synth_inside.mean()),
        "real_occupied_cells": int(ro.sum()),
    }


def shifted_edges(edges, phases):
    """Translate the lattice, preserving cell width and covering the original bounds."""
    shifted = []
    for edge, phase in zip(edges, phases, strict=True):
        if not 0 <= phase < 1:
            raise ValueError("Grid phases must lie in [0, 1)")
        width = edge[1] - edge[0]
        shifted.append(
            edge.copy()
            if phase == 0
            else edge[0] + (np.arange(len(edge) + 1) - phase) * width
        )
    # grid_occupancy expects square arrays. A zero-phase axis gets an empty
    # extra cell when the other axis is shifted; occupied-cell scores are unchanged.
    if len(shifted[0]) != len(shifted[1]):
        axis = int(len(shifted[0]) > len(shifted[1]))
        edge = shifted[axis]
        shifted[axis] = np.append(edge, edge[-1] + (edge[1] - edge[0]))
    return tuple(shifted)


def grid_sensitivity(real, synthetic, ids):
    names = list(synthetic)
    spaces, parameters = fit_spaces(real, list(synthetic.values()))
    coordinates = spaces["pca2"]
    reference = coordinates[0]
    removed = int(np.argmax(np.linalg.norm(reference, axis=1)))
    bounds_without_outlier = np.delete(reference, removed, axis=0)
    runs = []
    for bins, setting in product(
        (15, 30, 45), ("real", "pooled", "real_without_one_extreme")
    ):
        population = {
            "real": [reference],
            "pooled": coordinates,
            "real_without_one_extreme": [bounds_without_outlier],
        }[setting]
        edges = grid_edges(
            population, bins, "pooled" if setting == "pooled" else "real"
        )
        for phases in product((0.0, 0.25, 0.5, 0.75), repeat=2):
            shifted = shifted_edges(edges, phases)
            runs.append(
                {
                    "n_bins_before_shift": bins,
                    "bounds": setting,
                    "phase": phases,
                    "edges": shifted,
                    "scores": {
                        name: grid_scores(reference, coords, shifted)
                        for name, coords in zip(names, coordinates[1:], strict=True)
                    },
                }
            )
    return {
        "parameters": parameters,
        "removed_from_bounds_only_id": ids[removed],
        "evaluation_population_unchanged": True,
        "runs": runs,
    }


def repeated_distances(real, synthetic, ids, synthetic_ids, seed, repeats):
    runs, splits, fits = [], [], []
    for repeat in range(repeats):
        split = split_indices(
            len(real), {k: len(v) for k, v in synthetic.items()}, seed + repeat
        )
        reference_pool = real[split["reference"]]
        calibration, evaluation = real[split["calibration"]], real[split["evaluation"]]
        selected = {name: x[split["synthetic"][name]] for name, x in synthetic.items()}
        splits.append(
            {
                "repeat": repeat,
                "seed": seed + repeat,
                **{
                    name: ids[split[name]]
                    for name in ("reference", "calibration", "evaluation")
                },
                "synthetic": {
                    name: synthetic_ids[name][indices]
                    for name, indices in split["synthetic"].items()
                },
            }
        )
        for n_reference in SIZES:
            spaces, params = fit_spaces(
                reference_pool[:n_reference],
                [calibration, evaluation, reference_pool, *selected.values()],
            )
            fits.append(
                {"repeat": repeat, "n_reference_fit": n_reference, "parameters": params}
            )
            for space, matrices in spaces.items():
                _, cal, query, candidates, *corpora = matrices
                for n_synthetic in SIZES:
                    for (name, raw), coords in zip(
                        selected.items(), corpora, strict=True
                    ):
                        scores = distance_scores(
                            cal, query, candidates[:n_synthetic], coords[:n_synthetic]
                        )
                        omitted = np.setdiff1d(
                            np.arange(real.shape[1]), params["kept_columns"]
                        )
                        departures = (
                            np.abs(
                                raw[:n_synthetic, omitted]
                                - np.array(params["mean"])[omitted]
                            )
                            > params["constant_atol"]
                        )
                        runs.append(
                            {
                                "repeat": repeat,
                                "n_reference_fit": n_reference,
                                "n_candidates_per_corpus": n_synthetic,
                                "space": space,
                                "corpus": name,
                                "synthetic_reference_constant_departures": int(
                                    departures.any(axis=1).sum()
                                ),
                                **scores,
                            }
                        )
    return {"splits": splits, "fits": fits, "runs": runs}


def summarize_distances(runs):
    summary = []
    for space, name, n_ref, n_cand in product(("full", "pca2"), CORPORA, SIZES, SIZES):
        selected = [
            r
            for r in runs
            if (
                r["space"],
                r["corpus"],
                r["n_reference_fit"],
                r["n_candidates_per_corpus"],
            )
            == (space, name, n_ref, n_cand)
        ]
        covered = np.array([r["synthetic_covered"] for r in selected])
        baseline = np.array([r["real_baseline_covered"] for r in selected])
        summary.append(
            {
                "space": space,
                "corpus": name,
                "n_reference_fit": n_ref,
                "n_candidates_per_corpus": n_cand,
                "median_covered": np.median(covered, axis=0),
                "covered_repeat_p10_p90": np.quantile(covered, [0.1, 0.9], axis=0),
                "median_real_baseline_covered": np.median(baseline, axis=0),
                "median_paired_coverage_gap": np.median(covered - baseline, axis=0),
                "median_distance": np.median([r["median_distance"] for r in selected]),
                "median_real_baseline_distance": np.median(
                    [r["real_baseline_median_distance"] for r in selected]
                ),
            }
        )
    return summary


def summarize_sensitivity(runs):
    index = {
        (
            r["repeat"],
            r["space"],
            r["corpus"],
            r["n_reference_fit"],
            r["n_candidates_per_corpus"],
        ): r
        for r in runs
    }
    repeats = sorted({r["repeat"] for r in runs})
    result = {}
    for name in CORPORA:
        changes = {
            key: []
            for key in (
                "reference_64_to_128_full_q90_coverage_change",
                "candidates_64_to_128_full_q90_coverage_change",
                "candidates_64_to_128_full_median_distance_change",
                "candidates_64_to_128_full_coverage_change_fixed_128_q90_radius",
                "full_vs_pca2_q90_query_disagreement",
                "full_vs_pca2_neighbour_agreement",
            )
        }
        for repeat in repeats:
            full = index[repeat, "full", name, 128, 128]
            pca = index[repeat, "pca2", name, 128, 128]
            small_ref = index[repeat, "full", name, 64, 128]
            small_corpus = index[repeat, "full", name, 128, 64]
            changes["reference_64_to_128_full_q90_coverage_change"].append(
                full["synthetic_covered"][1] - small_ref["synthetic_covered"][1]
            )
            changes["candidates_64_to_128_full_q90_coverage_change"].append(
                full["synthetic_covered"][1] - small_corpus["synthetic_covered"][1]
            )
            changes["candidates_64_to_128_full_median_distance_change"].append(
                full["median_distance"] - small_corpus["median_distance"]
            )
            changes[
                "candidates_64_to_128_full_coverage_change_fixed_128_q90_radius"
            ].append(
                full["synthetic_covered"][1]
                - np.mean(
                    small_corpus["synthetic_distances"] <= full["calibration_radii"][1]
                )
            )
            changes["full_vs_pca2_q90_query_disagreement"].append(
                np.mean(
                    (full["synthetic_distances"] <= full["calibration_radii"][1])
                    != (pca["synthetic_distances"] <= pca["calibration_radii"][1])
                )
            )
            changes["full_vs_pca2_neighbour_agreement"].append(
                np.mean(
                    full["synthetic_neighbour_indices"]
                    == pca["synthetic_neighbour_indices"]
                )
            )
        result[name] = {
            key: {
                "median": np.median(values),
                "repeat_p10_p90": np.quantile(values, [0.1, 0.9]),
            }
            for key, values in changes.items()
        }
    result["balanced_strictly_higher_coverage_fraction"] = [
        {
            "space": space,
            "n_reference_fit": n_ref,
            "n_candidates_per_corpus": n_cand,
            "by_radius_quantile": np.mean(
                [
                    np.array(
                        index[i, space, CORPORA[0], n_ref, n_cand]["synthetic_covered"]
                    )
                    > np.array(
                        index[i, space, CORPORA[1], n_ref, n_cand]["synthetic_covered"]
                    )
                    for i in repeats
                ],
                axis=0,
            ),
        }
        for space, n_ref, n_cand in product(("full", "pca2"), SIZES, SIZES)
    ]
    return result


def summarize_grid(runs):
    result = []
    for bins, bounds, name in product(
        (15, 30, 45), ("real", "pooled", "real_without_one_extreme"), CORPORA
    ):
        selected = [
            r
            for r in runs
            if r["n_bins_before_shift"] == bins and r["bounds"] == bounds
        ]
        result.append(
            {
                "n_bins": bins,
                "bounds": bounds,
                "corpus": name,
                "zero_phase": selected[0]["scores"][name],
                "phase_min_max": {
                    metric: [
                        min(r["scores"][name][metric] for r in selected),
                        max(r["scores"][name][metric] for r in selected),
                    ]
                    for metric in (
                        "uncovered_real_cell_fraction",
                        "uncovered_real_series_fraction",
                        "real_out_of_range_fraction",
                        "synthetic_out_of_range_fraction",
                    )
                },
            }
        )
    return result


def save_details(path, details):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        safe_json(details), allow_nan=False, separators=(",", ":")
    ).encode()
    path.write_bytes(gzip.compress(encoded, mtime=0))
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def controls(directory, real):
    names = ("dependence", "seasonality", "spike", "level_shift", "variance_shift")
    frames = {
        name: load_cache(directory / f"control_{name}.parquet")
        for name in ("base", *names)
    }
    base_ids = frames["base"][1]
    if any(not np.array_equal(ids, base_ids) for _, ids, _ in frames.values()):
        raise ValueError("Controlled feature pairs are not aligned")
    spaces, params = fit_spaces(real, [f[0] for f in frames.values()])
    report = {}
    for space, matrices in spaces.items():
        _, base, *changes = matrices
        baseline_loo = np.linalg.norm(base[:, None] - base[None, :], axis=2)
        np.fill_diagonal(baseline_loo, np.inf)
        radius = np.quantile(baseline_loo.min(axis=1), 0.9)
        report[space] = {
            "baseline_loo_q90_radius": radius,
            "mechanisms": {
                name: {
                    "median_distance_to_base": np.median(nearest(changed, base)[0]),
                    "fraction_outside_base_radius": np.mean(
                        nearest(changed, base)[0] > radius
                    ),
                }
                for name, changed in zip(names, changes, strict=True)
            },
        }
    return {
        "spaces": report,
        "parameters": params,
        "inputs": {name: row[2] for name, row in frames.items()},
    }


def validate_native_schema(source):
    """Accept the unchanged pilot definitions without relabelling old artifacts."""
    if source["schema"] not in (FEATURE_SCHEMA, "native_candidate_v1") or tuple(
        source["features"]
    ) != tuple(FEATURE_NAMES):
        raise ValueError("Pilot cache feature schema does not match the implementation")


def run_input(directory, seed, repeats, output):
    source = json.loads((directory / "summary.json").read_text())
    validate_native_schema(source)
    report = {
        "source_schema": source["schema"],
        "pilot_seed": source["seed"],
        "source_summary_sha256": hashlib.sha256(
            (directory / "summary.json").read_bytes()
        ).hexdigest(),
        "groups": {},
    }
    for group in ("Monthly", "Yearly"):
        loaded = {
            name: load_cache(directory / group / f"{name}.parquet")
            for name in ("real", *CORPORA)
        }
        real, ids, _ = loaded["real"]
        synth = {name: loaded[name][0] for name in CORPORA}
        distances = repeated_distances(
            real, synth, ids, {name: loaded[name][1] for name in CORPORA}, seed, repeats
        )
        grid = grid_sensitivity(real, synth, ids)
        details = save_details(
            output / f"{source['seed']}_{group}_details.json.gz",
            {"distances": distances, "grid": grid},
        )
        report["groups"][group] = {
            "extraction": {
                k: source["groups"][group][k]
                for k in ("period", "effective_window_size")
            },
            "inputs": {name: data[2] for name, data in loaded.items()},
            "details": details,
            "distance_summary": summarize_distances(distances["runs"]),
            "sensitivity_summary": summarize_sensitivity(distances["runs"]),
            "grid_summary": summarize_grid(grid["runs"]),
            "removed_from_bounds_only_id": grid["removed_from_bounds_only_id"],
            "controls": controls(directory / group, real)
            if group == "Monthly"
            else None,
        }
        print(
            f"Validated {directory.name}/{group}: {repeats} paired splits", flush=True
        )
    return report


def plot_report(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for row, sample in enumerate(report["samples"]):
        for col, (group, data) in enumerate(sample["groups"].items()):
            ax = axes[row, col]
            for offset, space in ((-0.18, "full"), (0.18, "pca2")):
                entries = [
                    s
                    for s in data["distance_summary"]
                    if s["space"] == space
                    and s["n_reference_fit"] == 128
                    and s["n_candidates_per_corpus"] == 128
                ]
                median = np.array([s["median_covered"][1] for s in entries])
                intervals = np.array(
                    [s["covered_repeat_p10_p90"][:, 1] for s in entries]
                ).T
                ax.errorbar(
                    np.arange(2) + offset,
                    median,
                    yerr=[median - intervals[0], intervals[1] - median],
                    fmt="o",
                    capsize=4,
                    label=space,
                )
            baseline = [
                s
                for s in data["distance_summary"]
                if s["n_reference_fit"] == 128 and s["n_candidates_per_corpus"] == 128
            ]
            for y in [s["median_real_baseline_covered"][1] for s in baseline]:
                ax.axhline(y, color="grey", alpha=0.25, ls="--")
            ax.set(
                title=f"{group}, pilot seed {sample['pilot_seed']}",
                xticks=range(2),
                xticklabels=CORPORA,
                ylim=(0, 1.03),
                ylabel="Held-out real fraction within real-calibrated radius",
            )
            ax.legend()
    fig.suptitle(
        "Full features vs PCA(2) · q90 radius · 128 candidates per corpus\nBars: 10–90% split variability, not confidence intervals; grey: real baselines"
    )
    fig.tight_layout()
    fig.savefig(output / "distance_corroboration.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for row, sample in enumerate(report["samples"]):
        for col, (group, data) in enumerate(sample["groups"].items()):
            ax = axes[row, col]
            for bounds, label in (
                ("real", "real bounds"),
                ("pooled", "pooled bounds"),
                ("real_without_one_extreme", "one extreme omitted from bounds"),
            ):
                rows = [
                    s
                    for s in data["grid_summary"]
                    if s["bounds"] == bounds and s["corpus"] == "balanced_pool"
                ]
                bins = [s["n_bins"] for s in rows]
                score = [s["zero_phase"]["uncovered_real_cell_fraction"] for s in rows]
                ranges = np.array(
                    [s["phase_min_max"]["uncovered_real_cell_fraction"] for s in rows]
                ).T
                ax.plot(bins, score, "o-", label=label)
                ax.fill_between(bins, ranges[0], ranges[1], alpha=0.15)
            ax.set(
                title=f"{group}, pilot seed {sample['pilot_seed']}",
                xticks=[15, 30, 45],
                xlabel="Bins per axis before lattice shift",
                ylabel="Uncovered real occupied-cell fraction",
                ylim=(0, 1),
            )
            ax.legend(fontsize=8)
    fig.suptitle(
        "Balanced pool · fixed real-fitted PCA and evaluation populations\nLines: zero phase; bands: min–max over 16 lattice shifts"
    )
    fig.tight_layout()
    fig.savefig(output / "grid_sensitivity.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot", type=Path, default=Path("benchmarks/data/feature_coverage_pilot")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_validation"),
    )
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    report = {
        "environment": environment_metadata(),
        "schema": FEATURE_SCHEMA,
        "schema_frozen": True,
        "protocol": {
            "distance": "Euclidean on reference-standardized nonconstant features; no whitening",
            "quantiles": QUANTILES,
            "reference_fit_sizes": SIZES,
            "candidate_sizes": SIZES,
            "n_calibration": 64,
            "n_evaluation": 64,
            "split_seed": args.seed,
            "repeats": args.repeats,
            "resampling": "nested samples without replacement; repeat ranges are not confidence intervals",
            "synthetic_selection": "complete cases, equal retained counts; input failures separately reported",
            "length_matching": "source corpora matched in the original pilot; random subsets not individually rematched",
            "grid": "fixed all-real PCA, bins 15/30/45, 16 lattice phases, three bound populations",
            "controls": "illustrative base-LOO-q90 radius with 32 pairs, not held-out population calibration",
        },
        "samples": [
            run_input(p, args.seed, args.repeats, args.output)
            for p in (args.pilot, args.pilot / "replication")
        ],
    }
    save_json(args.output / "summary.json", report)
    plot_report(report, args.output)
    print(f"Saved validation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
