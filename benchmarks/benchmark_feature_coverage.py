"""Native feature pilot on sampled M4 training data and controlled mechanisms.

Run: python benchmarks/benchmark_feature_coverage.py --quick --download
Replays cached training files offline without --download. This pilot does not
freeze the schema or claim a full M4 benchmark. Outputs include feature caches,
reference-space parameters, coordinates, diagnostics, and a summary figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import numpy as np
import polars as pl
from _env import environment_metadata
from datasetsforecast.m4 import M4, M4Info

from synforecast import (
    compare_feature_coverage,
    compute_features,
    interpretable_pool,
    pretraining_pool,
)
from synforecast._features import FEATURE_NAMES, FEATURE_SCHEMA, compute_feature_set
from synforecast.generators import MARGenerator

# Declared before inspecting results. Failure keeps the schema provisional.
CRITERIA = {
    "minimum_overall_retention": 0.95,
    "minimum_shortest_quartile_retention": 0.90,
    "minimum_control_direction_fraction": 0.90,
    # Spikiness is variance of leave-one-out variances and has a much smaller
    # numerical scale than ACF/strength/shift summaries at length 240.
    "minimum_control_median_feature_delta": {
        "dependence": 0.05,
        "seasonality": 0.05,
        "spike": 0.0001,
        "level_shift": 0.05,
        "variance_shift": 0.05,
    },
    "near_redundancy_absolute_correlation": 0.95,
    "minimum_displayed_real_variance_for_standalone_2d_claim": 0.80,
}
GROUPS = {"Monthly": (12, "MS"), "Yearly": (None, "YS")}
# Declared physical cycles, consistent with the pilot and generator settings.
# M4's nonseasonal Weekly/Daily convention is a separately reported sensitivity.
AVAILABILITY_RECIPES = {
    "Yearly": {"primary": (None, 5), "default_window": (None, 10)},
    "Quarterly": {"primary": (4, 4)},
    "Monthly": {"primary": (12, 12)},
    "Weekly": {"primary": (52, 52), "m4_nonseasonal": (None, 10)},
    "Daily": {"primary": (7, 7), "m4_nonseasonal": (None, 10)},
    "Hourly": {"primary": (24, 24)},
}
# Selected after the all-row audit: annual Weekly excludes 65/359 series.
# This is an explicit panel recipe, never a per-series fallback.
RECOMMENDED_RECIPES = {
    group: "m4_nonseasonal" if group == "Weekly" else "primary"
    for group in AVAILABILITY_RECIPES
}
CONTROLS = {
    "dependence": ("acf1", 1),
    "seasonality": ("seasonal_strength", 1),
    "spike": ("spike", 1),
    "level_shift": ("max_level_shift", 1),
    "variance_shift": ("max_var_shift", 1),
}


def safe_json(value):
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + "\n")


def training_file(cache, group, download):
    path = cache / f"{group}-train.csv"
    url = M4.source_url.rstrip("/") + f"/Train/{group}-train.csv"
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"Missing {path}; use --download to fetch M4 training data"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        print(f"Downloading {url}", flush=True)
        with urlopen(url, timeout=60) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        partial.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return path, {"url": url, "sha256": digest.hexdigest(), "split": "train"}


def sample_training(path, group, n_series, seed, max_length):
    """Sample row indices uniformly before reading values; never inspect test data."""
    count = M4Info[group].n_ts
    if not 2 <= n_series <= count:
        raise ValueError(f"n_series must be between 2 and {count}")
    selected = set(
        np.random.default_rng(seed).choice(count, n_series, replace=False).tolist()
    )
    series, lengths = {}, {}
    with path.open(newline="") as source:
        reader = csv.reader(source)
        next(reader)
        for i, row in enumerate(reader):
            if i not in selected:
                continue
            observations = row[1:]
            while observations and not observations[-1].strip():
                observations.pop()
            if any(not x.strip() for x in observations):
                raise ValueError(
                    f"Internal missing observation in training series {row[0]}"
                )
            values = np.array([float(x) for x in observations], dtype=float)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f"Invalid raw training values for {row[0]}")
            lengths[row[0]] = len(values)
            series[row[0]] = values[-max_length:]
    if len(series) != n_series:
        raise ValueError("M4 training file did not contain the expected sampled rows")
    return series, lengths


def panel(series):
    return pl.DataFrame(
        {
            "unique_id": [uid for uid, values in series.items() for _ in values],
            "ds": np.concatenate([np.arange(len(x)) for x in series.values()]),
            "y": np.concatenate(list(series.values())),
        }
    )


def generate_matched(real, period, frequency, seed, *, include_pretraining=False):
    """Exact one-to-one length matching; balanced allocation independent of lengths."""
    prototypes = interpretable_pool(
        freq=frequency, seasonal_period=period or 1, seed=seed, engine="polars"
    )
    mar = MARGenerator(
        min_length=200,
        max_length=200,
        freq=frequency,
        seasonal_period=period,
        seed=seed,
        engine="polars",
    )
    assignment = np.arange(len(real)) % len(prototypes)
    np.random.default_rng(seed).shuffle(assignment)
    pools = {"balanced_pool": prototypes}
    if include_pretraining:
        pools["pretraining_pool"] = pretraining_pool(
            freq=frequency, seasonal_period=period or 1, seed=seed, engine="polars"
        )
    pools["synforecast_mar"] = [mar]
    corpora, provenance, failures = {}, {}, {}
    for corpus_index, (corpus, pool) in enumerate(pools.items()):
        corpus_seed = seed + 100000 * corpus_index if include_pretraining else seed
        if include_pretraining:
            rng = np.random.default_rng(corpus_seed)
            # Cycle across every prototype before reusing one; remainder is
            # random so small runs do not systematically omit trailing variants.
            assigned = np.concatenate(
                [
                    rng.permutation(len(pool))
                    for _ in range((len(real) + len(pool) - 1) // len(pool))
                ]
            )[: len(real)]
            rng.shuffle(assigned)
        else:
            assigned = (
                assignment
                if corpus == "balanced_pool"
                else np.zeros(len(real), dtype=int)
            )
        values, sources, errors = {}, {}, {}
        for i, (uid, reference) in enumerate(real.items()):
            prototype = pool[int(assigned[i])]
            config = prototype.model_dump(
                by_alias=True, include=set(type(prototype).model_fields)
            )
            config.update(
                min_length=len(reference),
                max_length=len(reference),
                seed=corpus_seed + i,
            )
            label = (
                f"{int(assigned[i])}:{type(prototype).__name__}"
                if corpus != "synforecast_mar"
                else "MARGenerator"
            )
            sources[uid] = label
            try:
                generated = (
                    type(prototype)(**config).generate(1, n_jobs=1)["y"].to_numpy()
                )
                if len(generated) != len(reference) or not np.isfinite(generated).all():
                    raise ValueError(
                        f"Generated length={len(generated)}, expected={len(reference)}; "
                        f"non-finite observations={int((~np.isfinite(generated)).sum())}"
                    )
                values[uid] = generated
            except (ValueError, RuntimeError) as exc:
                errors[uid] = str(exc)
        if not values:
            raise ValueError(f"Every draw failed for {corpus}")
        corpora[corpus], provenance[corpus], failures[corpus] = values, sources, errors
    configs = {
        "balanced_pool": [
            p.model_dump(mode="json", by_alias=True, include=set(type(p).model_fields))
            for p in prototypes
        ],
        "synforecast_mar": mar.model_dump(
            mode="json", by_alias=True, include=set(type(mar).model_fields)
        ),
        "generation": "public generate(1, n_jobs=1), seed=corpus_seed+sample_index",
    }
    if include_pretraining:
        configs["pretraining_pool"] = [
            p.model_dump(mode="json", by_alias=True, include=set(type(p).model_fields))
            for p in pools["pretraining_pool"]
        ]
        configs["generation"] = (
            "public generate(1, n_jobs=1); seed=base+100000*corpus_index+sample_index; shuffled balanced cycles across prototypes"
        )
    return corpora, provenance, failures, configs


def extract(series, period, failed_ids=(), window_size=None):
    features = compute_features(
        panel(series), seasonal_period=period, window_size=window_size
    )
    if failed_ids:
        failed = pl.DataFrame(
            {
                "unique_id": list(failed_ids),
                **{name: [float("nan")] * len(failed_ids) for name in FEATURE_NAMES},
            }
        )
        features = pl.concat([features, failed])
    return features.sort("unique_id")


def retention(features, lengths):
    matrix = features.select(FEATURE_NAMES).to_numpy()
    finite = np.isfinite(matrix)
    ids = features["unique_id"].to_list()
    size = np.array([lengths[uid] for uid in ids])
    cutoff = float(np.quantile(size, 0.25))
    shortest = size <= cutoff
    valid = finite.all(axis=1)
    return {
        "n_input": len(ids),
        "n_retained": int(valid.sum()),
        "fraction": float(valid.mean()),
        "shortest_quartile_max_length": cutoff,
        "shortest_quartile_n": int(shortest.sum()),
        "shortest_quartile_retention": float(valid[shortest].mean()),
        "undefined_per_feature": dict(
            zip(FEATURE_NAMES, (~finite).sum(axis=0).tolist(), strict=True)
        ),
        "dropped_ids": [uid for uid, keep in zip(ids, valid, strict=True) if not keep],
    }


def projection_report(features, result, coordinates):
    ids = result.real_diagnostics.retained_ids
    if len(coordinates) != len(ids):
        raise ValueError("Reference coordinates are not aligned")
    fitted = features.filter(pl.col("unique_id").is_in(ids))
    x = fitted.select(FEATURE_NAMES).to_numpy()
    params = result.metadata["parameters"]
    kept = params["kept_columns"]
    z = (x[:, kept] - np.array(params["mean"])[kept]) / np.array(params["scale"])[kept]
    residual = np.sqrt(
        np.mean((z - coordinates @ np.array(params["components"])) ** 2, axis=1)
    )
    correlation = np.atleast_2d(np.corrcoef(z, rowvar=False))
    names = [FEATURE_NAMES[i] for i in kept]
    pairs = [
        {"a": names[i], "b": names[j], "correlation": correlation[i, j]}
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if abs(correlation[i, j]) >= CRITERIA["near_redundancy_absolute_correlation"]
    ]
    return {
        "features": names,
        "correlation": correlation,
        "near_redundant_pairs": pairs,
        "explained_variance": result.explained_variance,
        "reconstruction_rmse_quantiles": np.quantile(residual, [0, 0.5, 0.9, 1]),
        "retained_reference_ids": list(ids),
    }


def controlled_series(seed, draws=32, length=240, period=12):
    rng = np.random.default_rng(seed)
    base, changed = {}, {name: {} for name in CONTROLS}
    for i in range(draws):
        uid = f"control{i:03d}"
        noise = rng.normal(size=length)
        base[uid] = noise
        ar = np.empty(length)
        ar[0] = noise[0]
        for j in range(1, length):
            ar[j] = 0.9 * ar[j - 1] + noise[j]
        changed["dependence"][uid] = ar
        changed["seasonality"][uid] = noise + 4 * np.sin(
            2 * np.pi * np.arange(length) / period
        )
        spike = noise.copy()
        spike[length // 2] += 30
        changed["spike"][uid] = spike
        step = noise.copy()
        step[length // 2 :] += 5
        changed["level_shift"][uid] = step
        variance = noise.copy()
        variance[length // 2 :] *= 5
        changed["variance_shift"][uid] = variance
    return base, changed


def control_report(reference, seed, directory):
    base, variations = controlled_series(seed)
    frames = {
        "base": extract(base, 12),
        **{name: extract(series, 12) for name, series in variations.items()},
    }
    results = compare_feature_coverage(
        reference, frames, precomputed=True, missing="drop"
    )
    params = results["base"].metadata["parameters"]
    kept = params["kept_columns"]
    scale = np.array(params["scale"])[kept]
    components = np.array(params["components"])
    report = {}
    for name, (feature, direction) in CONTROLS.items():
        delta = frames[name][feature].to_numpy() - frames["base"][feature].to_numpy()
        standardized_delta = (
            frames[name].select(FEATURE_NAMES).to_numpy()[:, kept]
            - frames["base"].select(FEATURE_NAMES).to_numpy()[:, kept]
        ) / scale
        projected = standardized_delta @ components.T
        original_energy = np.sum(standardized_delta**2, axis=1)
        fraction = np.sum(projected**2, axis=1) / original_energy
        report[name] = {
            "feature": feature,
            "direction_fraction": float(np.mean(direction * delta > 0)),
            "median_feature_delta": float(np.median(delta)),
            "median_projected_change_fraction": float(np.median(fraction)),
            "passed": bool(
                np.mean(direction * delta > 0)
                >= CRITERIA["minimum_control_direction_fraction"]
                and direction * np.median(delta)
                >= CRITERIA["minimum_control_median_feature_delta"][name]
            ),
        }
    for name, features in frames.items():
        features.write_parquet(directory / f"control_{name}.parquet")
    return report, frames, results


def summarize_result(result):
    return {
        name: getattr(result, name)
        for name in (
            "miscoverage",
            "reverse_miscoverage",
            "uncovered_real_cell_fraction",
            "uncovered_real_series_fraction",
            "n_synthetic_out_of_range",
            "synthetic_out_of_range_fraction",
            "space_id",
            "fit",
            "grid_range",
        )
    } | {
        "real_occupied_cells": int(result.real_occupancy.sum()),
        "synthetic_occupied_cells": int(result.synthetic_occupancy.sum()),
    }


def availability_summary(ids, lengths, matrix):
    """Report complete-case and per-feature availability, including the short tail."""
    ids = np.asarray(ids)
    lengths = np.asarray(lengths)
    finite = np.isfinite(matrix)
    valid = finite.all(axis=1)
    cutoff = np.quantile(lengths, 0.25)
    shortest = lengths <= cutoff
    features = {}
    for column, name in enumerate(FEATURE_NAMES):
        values = matrix[finite[:, column], column]
        features[name] = {
            "n_undefined": int((~finite[:, column]).sum()),
            "finite_quantiles_min_p01_p50_p99_max": np.quantile(
                values, [0, 0.01, 0.5, 0.99, 1]
            )
            if len(values)
            else None,
            "constant_on_finite_rows": bool(np.ptp(values) <= 1e-12)
            if len(values)
            else None,
        }
    return {
        "n_input": len(ids),
        "n_retained": int(valid.sum()),
        "n_dropped": int((~valid).sum()),
        "dropped_ids_sha256": hashlib.sha256(
            json.dumps(sorted(ids[~valid].tolist()), separators=(",", ":")).encode()
        ).hexdigest(),
        "retained_fraction": float(valid.mean()),
        "shortest_quartile_max_length": cutoff,
        "shortest_quartile_n": int(shortest.sum()),
        "shortest_quartile_retention": float(valid[shortest].mean()),
        "dropped_examples_first_10": [
            {
                "id": uid,
                "length": int(n),
                "undefined_features": [
                    name for name, ok in zip(FEATURE_NAMES, row, strict=True) if not ok
                ],
            }
            for uid, n, row in zip(
                ids[~valid][:10], lengths[~valid][:10], finite[~valid][:10], strict=True
            )
        ],
        "features": features,
        "passes_pilot_retention_criteria": bool(
            valid.mean() >= CRITERIA["minimum_overall_retention"]
            and valid[shortest].mean()
            >= CRITERIA["minimum_shortest_quartile_retention"]
        ),
    }


def audit_availability(args, group):
    """Extract every real training series, streaming raw data; save no feature cache."""
    path, source = training_file(args.cache, group, args.download)
    recipes = AVAILABILITY_RECIPES[group]
    ids, original_lengths, lengths = [], [], []
    matrices = {name: [] for name in recipes}
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            uid, observations = row[0], row[1:]
            while observations and not observations[-1].strip():
                observations.pop()
            if not observations or any(not x.strip() for x in observations):
                raise ValueError(f"Empty or internally missing training series {uid}")
            values = np.array(observations, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite raw training series {uid}")
            ids.append(uid)
            original_lengths.append(len(values))
            values = values[-args.max_length :]
            lengths.append(len(values))
            for name, (period, window) in recipes.items():
                extracted = compute_feature_set(values, period, window)
                matrices[name].append([extracted[feature] for feature in FEATURE_NAMES])
    if len(ids) != M4Info[group].n_ts or len(set(ids)) != len(ids):
        raise ValueError(f"Unexpected count or duplicate IDs in {path}")
    results = {}
    for name, (period, window) in recipes.items():
        results[name] = availability_summary(
            ids, lengths, np.asarray(matrices[name])
        ) | {
            "seasonal_period": period,
            "window_size": window,
            "minimum_length_for_all_features": max(11, 2 * window, 2 * (period or 0)),
        }
    primary = results["primary"]
    print(
        f"{group}: {primary['n_retained']}/{len(ids)} real series retained under primary recipe",
        flush=True,
    )
    return {
        "source": source,
        "recommended_recipe": RECOMMENDED_RECIPES[group],
        "m4_declared_seasonality": M4Info[group].seasonality,
        "n_series": len(ids),
        "original_length_quantiles_min_p25_p50_p75_max": np.quantile(
            original_lengths, [0, 0.25, 0.5, 0.75, 1]
        ),
        "effective_length_quantiles_min_p25_p50_p75_max": np.quantile(
            lengths, [0, 0.25, 0.5, 0.75, 1]
        ),
        "n_capped": int(np.sum(np.array(original_lengths) > args.max_length)),
        "recipes": results,
    }


def matched_complete_ids(frames):
    """One common complete cohort preserves exact lengths across all corpora."""
    valid = []
    for frame in frames.values():
        matrix = frame.select(FEATURE_NAMES).to_numpy()
        ids = frame["unique_id"].to_list()
        valid.append(
            {
                uid
                for uid, keep in zip(ids, np.isfinite(matrix).all(axis=1), strict=True)
                if keep
            }
        )
    common = sorted(set.intersection(*valid))
    if len(common) < 2:
        raise ValueError(
            "Fewer than two shared complete series remain for matched comparison"
        )
    return common


def smoke_group(args, group):
    from validate_feature_coverage import distance_scores, fit_spaces, save_details

    started = perf_counter()
    source, metadata = training_file(args.cache, group, args.download)
    real, original_lengths = sample_training(
        source, group, 128, args.seed, args.max_length
    )
    ids = np.asarray(list(real))
    order = np.random.default_rng(args.seed).permutation(len(ids))
    split = {
        "reference": ids[order[:64]].tolist(),
        "calibration": ids[order[64:96]].tolist(),
        "evaluation": ids[order[96:]].tolist(),
    }
    period, window = AVAILABILITY_RECIPES[group][RECOMMENDED_RECIPES[group]]
    frequency = {
        "Yearly": "YS",
        "Quarterly": "QS",
        "Monthly": "MS",
        "Weekly": "W",
        "Daily": "D",
        "Hourly": "h",
    }[group]
    reference = {uid: real[uid] for uid in split["reference"]}
    lengths = {uid: len(x) for uid, x in real.items()}
    corpora, provenance, failures, configs = generate_matched(
        reference, period, frequency, args.seed + 1000, include_pretraining=True
    )
    frames = {"real": extract(reference, period, window_size=window)}
    frames.update(
        {
            name: extract(values, period, failures[name], window_size=window)
            for name, values in corpora.items()
        }
    )
    common = matched_complete_ids(frames)
    matched = {
        name: frame.filter(pl.col("unique_id").is_in(common)).sort("unique_id")
        for name, frame in frames.items()
    }
    calibration = extract(
        {uid: real[uid] for uid in split["calibration"]}, period, window_size=window
    )
    evaluation = extract(
        {uid: real[uid] for uid in split["evaluation"]}, period, window_size=window
    )
    # Save the actual matrix order so per-query distances are inspectable.
    split["reference"] = frames["real"]["unique_id"].to_list()
    split["calibration"] = calibration["unique_id"].to_list()
    split["evaluation"] = evaluation["unique_id"].to_list()
    names = list(corpora)
    matrices = [
        f.select(FEATURE_NAMES).to_numpy()
        for f in (
            calibration,
            evaluation,
            matched["real"],
            *(matched[name] for name in names),
        )
    ]
    if not all(np.isfinite(x).all() for x in matrices):
        raise ValueError(
            "Real calibration/evaluation features are incomplete; inspect extraction recipe"
        )
    spaces, parameters = fit_spaces(
        frames["real"].select(FEATURE_NAMES).to_numpy(), matrices
    )
    distance_summary, distance_details = {}, {}
    for space, coordinates in spaces.items():
        _, cal, query, real_candidates, *synthetics = coordinates
        distance_summary[space], distance_details[space] = {}, {}
        for name, synthetic in zip(names, synthetics, strict=True):
            scores = distance_scores(cal, query, real_candidates, synthetic)
            distance_details[space][name] = scores
            distance_summary[space][name] = {
                k: scores[k]
                for k in (
                    "calibration_radii",
                    "real_baseline_covered",
                    "synthetic_covered",
                    "median_distance",
                    "real_baseline_median_distance",
                )
            } | {
                "distance_p10_p50_p90": np.quantile(
                    scores["synthetic_distances"], [0.1, 0.5, 0.9]
                ),
                "real_baseline_distance_p10_p50_p90": np.quantile(
                    scores["real_baseline_distances"], [0.1, 0.5, 0.9]
                ),
            }
    grids, grid_details = {}, {}
    for bins in (15, 30, 45):
        results = compare_feature_coverage(
            frames["real"],
            {name: matched[name] for name in names},
            precomputed=True,
            n_bins=bins,
        )
        grids[bins] = {
            name: summarize_result(result) for name, result in results.items()
        }
        grid_details[bins] = {name: asdict(result) for name, result in results.items()}
    primary = next(iter(results.values()))
    details = save_details(
        args.output / f"{group}_details.json.gz",
        {
            "split_ids": split,
            "generation_order_ids": list(reference),
            "matched_candidate_ids": common,
            "generator_configs": configs,
            "generator_provenance": provenance,
            "generation_failures": failures,
            "parameters": parameters,
            "distances": distance_details,
            "grids": grid_details,
        },
    )
    report = {
        "source": metadata,
        "extraction": {
            "schema": FEATURE_SCHEMA,
            "features": FEATURE_NAMES,
            "seasonal_period": period,
            "window_size": window,
            "frequency": frequency,
        },
        "original_lengths": original_lengths,
        "effective_lengths": lengths,
        "retention": {
            name: retention(frame, lengths) for name, frame in frames.items()
        },
        "generation_failures": failures,
        "n_requested_per_corpus": 64,
        "n_matched_candidates_per_corpus": len(common),
        "excluded_from_matched_comparison": {
            name: sorted(set(frame["unique_id"].to_list()) - set(common))
            for name, frame in frames.items()
        },
        "n_reference_fit": 64,
        "n_calibration": 32,
        "n_evaluation": 32,
        "pca_explained_variance": primary.explained_variance,
        "synthetic_reference_constant_departures": {
            name: result.synthetic_constant_feature_deviations
            for name, result in results.items()
        },
        "distances": distance_summary,
        "grids": grids,
        "details": details,
        "elapsed_seconds": perf_counter() - started,
    }
    print(
        f"{group}: matched {len(common)}/64 candidates per corpus; full-feature q90 coverage "
        + ", ".join(
            f"{name}={distance_summary['full'][name]['synthetic_covered'][1]:.1%}"
            for name in names
        ),
        flush=True,
    )
    return report


def run_group(args, group, output):
    started = perf_counter()
    period, frequency = GROUPS[group]
    source, source_metadata = training_file(args.cache, group, args.download)
    real, original_lengths = sample_training(
        source, group, args.n_series, args.seed, args.max_length
    )
    lengths = {uid: len(x) for uid, x in real.items()}
    print(
        f"{group}: {len(real)} training series, lengths {min(lengths.values())}..{max(lengths.values())}",
        flush=True,
    )
    corpora, provenance, failures, configs = generate_matched(
        real, period, frequency, args.seed + 1000
    )
    window_size = args.yearly_window if group == "Yearly" else None
    all_features = {"real": extract(real, period, window_size=window_size)}
    all_features.update(
        {
            name: extract(series, period, failures[name], window_size=window_size)
            for name, series in corpora.items()
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, features in all_features.items():
        features.write_parquet(output / f"{name}.parquet")
    real_features = all_features["real"]
    synthetic_features = {name: f for name, f in all_features.items() if name != "real"}
    retention_report = {name: retention(f, lengths) for name, f in all_features.items()}
    runs = {}
    primary = None
    for fit, bins in [("real", 15), ("real", 30), ("real", 45), ("pooled", 30)]:
        results = compare_feature_coverage(
            real_features,
            synthetic_features,
            precomputed=True,
            fit=fit,
            n_bins=bins,
            missing="drop",
        )
        key = f"{fit}_{bins}"
        runs[key] = {name: summarize_result(result) for name, result in results.items()}
        save_json(
            output / f"{key}_space.json",
            {name: asdict(result) for name, result in results.items()},
        )
        if fit == "real" and bins == 30:
            primary = results
    reference = next(iter(primary.values()))
    # The direct API sees only finite generator outputs; failed draws remain
    # explicitly represented in the cached evaluation's retention denominator.
    direct = compare_feature_coverage(
        panel(real),
        {name: panel(series) for name, series in corpora.items()},
        seasonal_period=period,
        window_size=window_size,
        missing="drop",
    )
    cache_agreement = {}
    for name, result in primary.items():
        np.testing.assert_allclose(result.real_embedding, direct[name].real_embedding)
        np.testing.assert_allclose(
            result.synthetic_embedding, direct[name].synthetic_embedding
        )
        if result.miscoverage != direct[name].miscoverage:
            raise AssertionError("Native and cached coverage disagree")
        cache_agreement[name] = True
    projection = projection_report(real_features, reference, reference.real_embedding)
    four = compare_feature_coverage(
        real_features,
        synthetic_features,
        precomputed=True,
        features=FEATURE_NAMES[:4],
        missing="drop",
    )
    four_summary = {
        "explained_variance": next(iter(four.values())).explained_variance,
        "scores": {name: summarize_result(result) for name, result in four.items()},
    }
    sensitivity = {}
    for name, features in synthetic_features.items():
        half = features.sample(
            n=max(2, len(features) // 2), seed=args.seed, shuffle=True
        ).sort("unique_id")
        result = compare_feature_coverage(
            real_features, {name: half}, precomputed=True, missing="drop"
        )[name]
        sensitivity[name] = summarize_result(result) | {
            "n_input": len(half),
            "selection": "seeded uniform half-sample",
            "selected_ids": half["unique_id"].to_list(),
        }
    controls = None
    if group == "Monthly":
        controls, _, _ = control_report(real_features, args.seed + 2000, output)
    # Explicit reference sensitivity, not a silently trimmed primary fit.
    retained = real_features.filter(
        pl.col("unique_id").is_in(reference.real_diagnostics.retained_ids)
    )
    magnitude = np.linalg.norm(reference.real_embedding, axis=1)
    removed_id = reference.real_diagnostics.retained_ids[int(np.argmax(magnitude))]
    reduced = retained.filter(pl.col("unique_id") != removed_id)
    reduced_results = compare_feature_coverage(
        reduced, synthetic_features, precomputed=True, missing="drop"
    )
    report = {
        "source": source_metadata,
        "original_lengths": original_lengths,
        "effective_lengths": lengths,
        "period": period,
        "window_size": window_size,
        "effective_window_size": window_size or period or 10,
        "frequency": frequency,
        "n_capped_series": sum(n > args.max_length for n in original_lengths.values()),
        "generator_configs": configs,
        "generator_provenance": provenance,
        "generation_failures": failures,
        "retention": retention_report,
        "native_precomputed_agreement": cache_agreement,
        "projection": projection,
        "runs": runs,
        "four_feature_sensitivity": four_summary,
        "half_sample_sensitivity": sensitivity,
        "remove_one_reference_outlier_sensitivity": {
            "removed_id": removed_id,
            "scores": {
                name: summarize_result(result)
                for name, result in reduced_results.items()
            },
        },
        "controls": controls,
        "acceptance": {
            "retention": all(
                r["fraction"] >= CRITERIA["minimum_overall_retention"]
                and r["shortest_quartile_retention"]
                >= CRITERIA["minimum_shortest_quartile_retention"]
                for r in retention_report.values()
            ),
            "standalone_2d_variance": sum(reference.explained_variance)
            >= CRITERIA["minimum_displayed_real_variance_for_standalone_2d_claim"],
            "controlled_responses": all(r["passed"] for r in controls.values())
            if controls
            else None,
        },
        "elapsed_seconds": perf_counter() - started,
    }
    print(
        f"{group}: retention {retention_report['real']['fraction']:.1%}, PCA variance {sum(reference.explained_variance):.1%}",
        flush=True,
    )
    return report


def plot_summary(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(report["groups"])
    fig, axes = plt.subplots(
        len(groups), 2, figsize=(12, 4 * len(groups)), squeeze=False
    )
    for row, group in enumerate(groups):
        data = report["groups"][group]
        names = list(data["retention"])
        axes[row, 0].bar(names, [data["retention"][name]["fraction"] for name in names])
        axes[row, 0].axhline(
            CRITERIA["minimum_overall_retention"], ls="--", color="black"
        )
        axes[row, 0].set(ylim=(0, 1.05), title=f"{group}: complete-case retention")
        corr = np.asarray(data["projection"]["correlation"])
        ax = axes[row, 1]
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
        labels = data["projection"]["features"]
        ax.set(
            xticks=range(len(labels)),
            yticks=range(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            title=f"{group}: real feature correlation (window={data['effective_window_size']})",
        )
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Native feature pilot — sampled training panels, not full M4 coverage")
    fig.tight_layout()
    fig.savefig(output / "pilot_diagnostics.png", dpi=150)
    plt.close(fig)

    if "Monthly" in report["groups"]:
        controls = report["groups"]["Monthly"]["controls"]
        names = list(controls)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            names,
            [controls[name]["median_projected_change_fraction"] for name in names],
        )
        ax.set(
            ylim=(0, 1),
            ylabel="Median fraction of squared feature change retained",
            title="Controlled changes projected onto Monthly real-fitted PCA (2D)",
        )
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        fig.savefig(output / "controlled_projection.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Compare three sources across all frequencies on 128 real series per group",
    )
    parser.add_argument(
        "--availability-only",
        action="store_true",
        help="Audit all real training rows; no generation or Parquet output",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use 64 series per group instead of 256"
    )
    parser.add_argument("--n-series", type=int)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--yearly-window",
        type=int,
        default=None,
        help="Explicit nonseasonal window width; None retains the default 10",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Keep at most this many trailing training observations for all corpora",
    )
    parser.add_argument("--groups", nargs="+", choices=AVAILABILITY_RECIPES)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/feature_coverage/m4")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.groups = args.groups or list(
        AVAILABILITY_RECIPES if args.availability_only or args.smoke else GROUPS
    )
    args.output = args.output or (
        Path("benchmarks/data/feature_coverage_smoke")
        if args.smoke
        else Path(
            "benchmarks/data/feature_coverage_availability"
            if args.availability_only
            else "benchmarks/data/feature_coverage_pilot"
        )
    )
    if (
        not args.availability_only
        and not args.smoke
        and any(group not in GROUPS for group in args.groups)
    ):
        parser.error(
            "Synthetic pilot currently supports Monthly/Yearly; use --availability-only for other groups"
        )
    if args.yearly_window is not None and args.yearly_window < 2:
        parser.error("--yearly-window must be at least 2")
    if args.max_length < 24:
        parser.error("--max-length must be at least 24 for the monthly pilot")
    logging.basicConfig(level=logging.WARNING)
    if args.smoke:
        if (
            args.availability_only
            or args.quick
            or args.n_series is not None
            or args.yearly_window is not None
        ):
            parser.error(
                "--smoke uses fixed counts and recommended recipes; omit --availability-only/--quick/--n-series/--yearly-window"
            )
        report = {
            "environment": environment_metadata(),
            "schema": FEATURE_SCHEMA,
            "schema_frozen": True,
            "seed": args.seed,
            "max_length": args.max_length,
            "protocol": "one generation seed; 64 real fitting/generation lengths, 32 calibration and 32 evaluation queries; exact length/count matching on shared complete IDs; q50/q90/q95 real-calibrated Euclidean coverage; exploratory real-fit PCA grids",
            "scope": "small benchmark, not full M4 or repeated-generator uncertainty",
            "groups": {},
        }
        for group in args.groups:
            report["groups"][group] = smoke_group(args, group)
            save_json(args.output / "summary.json", report)
        print(f"Saved small benchmark to {args.output}", flush=True)
        return
    if args.availability_only:
        if args.quick or args.n_series is not None or args.yearly_window is not None:
            # n_series is assigned below for the synthetic pilot only.
            parser.error(
                "Availability audit uses all training rows and fixed recipes; omit --quick/--n-series/--yearly-window"
            )
        report = {
            "environment": environment_metadata(),
            "schema": FEATURE_SCHEMA,
            "features": FEATURE_NAMES,
            "schema_frozen": True,
            "scope": "all real training rows; availability only, not synthetic coverage",
            "max_length": args.max_length,
            "length_policy": "last max_length training observations, matching the pilot",
            "criteria": {
                k: CRITERIA[k]
                for k in (
                    "minimum_overall_retention",
                    "minimum_shortest_quartile_retention",
                )
            },
            "groups": {group: audit_availability(args, group) for group in args.groups},
        }
        save_json(args.output / "summary.json", report)
        print(f"Saved availability audit to {args.output}", flush=True)
        return
    args.n_series = (
        args.n_series if args.n_series is not None else (64 if args.quick else 256)
    )
    report = {
        "environment": environment_metadata(),
        "schema": FEATURE_SCHEMA,
        "features": FEATURE_NAMES,
        "criteria": CRITERIA,
        "seed": args.seed,
        "n_series": args.n_series,
        "max_length": args.max_length,
        "length_policy": "trailing training observations; exact synthetic length matching",
        "groups": {},
    }
    for group in args.groups:
        report["groups"][group] = run_group(args, group, args.output / group)
    report["schema_frozen"] = True
    save_json(args.output / "summary.json", report)
    plot_summary(report, args.output)
    print(f"Saved pilot to {args.output}", flush=True)


if __name__ == "__main__":
    main()
