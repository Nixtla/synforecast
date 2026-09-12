"""Export real queries and their nearest synthetic neighbours for the docs notebook.

Selection is by rank of the balanced pool's full-feature nearest distance at
fixed percentiles, not by visual appeal. Values come from the same real samples,
splits, seeds, and first folds as the committed benchmarks. Only a few raw series
are written, so the notebook can stay offline.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import benchmark_cross_frequency as cross
import benchmark_third_panels as third
import numpy as np
import validate_seasonal_composition as seasonal
from _env import environment_metadata
from benchmark_feature_coverage import (
    generate_matched,
    safe_json,
    sample_training,
    training_file,
)
from investigate_hourly_coverage import CORPORA, draw_candidates, feature_rows
from validate_hourly_composition import (
    COMPOSITION,
    generate_composition,
    seeded_windows,
    transfer_panel,
)
from validate_hourly_splits import (
    GENERATION_SEEDS,
    SPLIT_SEED,
    fresh_splits,
    scores_for_split,
)
from validate_traffic_composition import TARGETED, TRAFFIC_SPLIT_SEED, generate_targeted

PERCENTILES = (0.1, 0.5, 0.9)
GROUPS = ("Monthly", "Daily")
THIRD_GROUPS = ("Quarterly", "Daily")


def select_ranks(distances, percentiles=PERCENTILES):
    """Indices at fixed rank percentiles of a distance vector (stable order)."""
    order = np.argsort(np.asarray(distances), kind="stable")
    return [int(order[int(round(q * (len(order) - 1)))]) for q in percentiles]


def rounded(values):
    return [float(f"{v:.6g}") for v in np.asarray(values, dtype=float)]


def neighbour(scores, name, q, values, provenance, selected):
    index = int(scores[name]["full"]["neighbour_indices"][q])
    slot = selected[index]
    return {
        "slot": slot,
        "prototype": provenance[slot],
        "distance": float(scores[name]["full"]["distances"][q]),
        "values": rounded(values[slot]),
    }


def hourly_examples(cache, transfer_cache, download):
    """M4 Hourly fold 0 and fresh traffic fold 0, first generation seed."""
    source, _ = training_file(cache, "Hourly", download)
    real, _ = sample_training(source, "Hourly", 414, SPLIT_SEED, 512)
    prior = json.loads(
        Path("benchmarks/data/feature_coverage_hourly_splits/summary.json").read_text()
    )
    splits = fresh_splits(list(real), prior["protocol"]["excluded_old_evaluation_ids"])
    real_rows = feature_rows(real)
    seed = GENERATION_SEEDS[0]
    slots = [f"slot{i:03d}" for i in range(256)]
    template = {uid: np.zeros(512) for uid in slots}
    generated, rows, common, generation = draw_candidates(template, slots, seed)
    selected = common[:240]
    provenance = dict(generation["provenance"])
    for name, factory in (
        (COMPOSITION, generate_composition),
        (TARGETED, generate_targeted),
    ):
        values, prov, failures, _ = factory(dict.fromkeys(slots, 512), seed)
        if failures:
            raise ValueError(f"{name} failed on {sorted(failures)[:3]}")
        generated[name] = values
        rows[name] = feature_rows(values)
        provenance[name] = prov
    panels = {}
    split = splits[0]
    scores, _, _ = scores_for_split(real, real_rows, split, selected, generated, rows)
    panels["M4 Hourly"] = examples_from(
        real,
        split,
        scores,
        selected,
        generated,
        provenance,
        ("balanced_pool", COMPOSITION, TARGETED),
    )
    traffic, _ = transfer_panel(transfer_cache, download)
    windows, _ = seeded_windows(traffic, 512, SPLIT_SEED)
    first_round = json.loads(
        Path(
            "benchmarks/data/feature_coverage_hourly_composition/summary.json"
        ).read_text()
    )
    excluded = [uid for s in first_round["transfer_splits"] for uid in s["evaluation"]]
    traffic_splits = fresh_splits(list(windows), excluded, seed=TRAFFIC_SPLIT_SEED)
    traffic_rows = feature_rows(windows)
    split = traffic_splits[0]
    scores, _, _ = scores_for_split(
        windows, traffic_rows, split, selected, generated, rows
    )
    panels["Traffic"] = examples_from(
        windows,
        split,
        scores,
        selected,
        generated,
        provenance,
        ("balanced_pool", COMPOSITION, TARGETED),
    )
    return panels


def examples_from(real, split, scores, selected, generated, provenance, names):
    picks = select_ranks(scores["balanced_pool"]["full"]["distances"])
    examples = []
    for q, percentile in zip(picks, PERCENTILES, strict=True):
        uid = split["evaluation"][q]
        examples.append(
            {
                "real_id": uid,
                "rank_percentile_of_balanced_distance": percentile,
                "real_values": rounded(real[uid]),
                "neighbours": {
                    name: neighbour(
                        scores, name, q, generated[name], provenance[name], selected
                    )
                    for name in names
                },
            }
        )
    return {"fold": split["fold"], "examples": examples}


def cross_frequency_examples(cache, download):
    """First fold and first generation seed of the cross-frequency benchmark."""
    panels = {}
    for group in GROUPS:
        source, _ = training_file(cache, group, download)
        real, _ = sample_training(source, group, cross.N_SAMPLE, cross.SPLIT_SEED, 512)
        panels[group] = fold_examples(
            group, real, cross.SPLIT_SEED, cross.GENERATION_SEEDS[0]
        )
    return panels


def third_panel_examples(cache, download):
    """First fold and first generation seed of the third-panel validation."""
    panels = {}
    for group in THIRD_GROUPS:
        series = third.fetch_panel(cache, third.PANELS[group], download)
        real, _ = third.prepare_real(series, third.SPLIT_SEED)
        panels[f"{group} ({third.PANELS[group]['dataset']})"] = fold_examples(
            group, real, third.SPLIT_SEED, third.GENERATION_SEEDS[0]
        )
    return panels


def seasonal_examples(cache, download):
    """Tourism monthly, first validation fold and seed, with the moderated corpus."""
    spec = seasonal.VALIDATION_PANELS["tourism_monthly"]
    series = third.fetch_panel(cache, spec, download)
    real, _ = third.prepare_real(series, seasonal.VALIDATION_SPLIT_SEED)
    panel = fold_examples(
        spec["group"],
        real,
        seasonal.VALIDATION_SPLIT_SEED,
        seasonal.VALIDATION_GENERATION_SEEDS[0],
        extra=seasonal.MODERATED,
    )
    return {"Monthly (tourism_monthly)": panel}


def fold_examples(group, real, split_seed, generation_seed, extra=None):
    frequency, period, window = cross.panel_recipes()[group]
    rows, _ = cross.complete_real(cross.feature_rows(real, period, window))
    split = cross.fresh_splits(list(rows), [], seed=split_seed)[0]
    seed = generation_seed + 1000 * split["fold"]
    if extra is not None:
        return seasonal_fold_examples(
            real, rows, split, period, frequency, window, seed, extra
        )
    synthetic_rows, common, generation = cross.draw_fold(
        real, split, period, frequency, window, seed
    )
    selected = common[: cross.N_MATCHED]
    scores, _, _ = cross.score_fold(rows, split, selected, synthetic_rows)
    # Regenerate the exact synthetic values for the selected neighbours.
    slots = [f"slot{i:03d}" for i in range(len(split["candidates"]))]
    template = {
        slot: real[uid] for slot, uid in zip(slots, split["candidates"], strict=True)
    }
    corpora, provenance, _, _ = generate_matched(
        template, period, frequency, seed, include_pretraining=True
    )
    picks = select_ranks(scores["balanced_pool"]["full"]["distances"])
    examples = []
    for q, percentile in zip(picks, PERCENTILES, strict=True):
        uid = split["evaluation"][q]
        examples.append(
            {
                "real_id": uid,
                "rank_percentile_of_balanced_distance": percentile,
                "real_values": rounded(real[uid]),
                "neighbours": {
                    name: neighbour(
                        scores, name, q, corpora[name], provenance[name], selected
                    )
                    for name in CORPORA[:2]
                },
            }
        )
    return {"fold": split["fold"], "seed": seed, "examples": examples}


def seasonal_fold_examples(real, rows, split, period, frequency, window, seed, extra):
    synthetic_rows, selected, generation = seasonal.draw_with_composition(
        real, split, period, frequency, window, seed
    )
    scores, _, _ = cross.score_fold(
        rows, split, selected, synthetic_rows, seasonal.SOURCES
    )
    slots = [f"slot{i:03d}" for i in range(len(split["candidates"]))]
    template = {
        slot: real[uid] for slot, uid in zip(slots, split["candidates"], strict=True)
    }
    corpora, provenance, _, _ = generate_matched(
        template, period, frequency, seed, include_pretraining=True
    )
    factory, recipe = seasonal.POOLS[extra]
    values, prov, _, _ = generate_composition(
        {slot: len(template[slot]) for slot in slots},
        seed,
        pool_factory=partial(factory, period, frequency),
        recipe=recipe,
        seed_offset=seasonal.SEED_OFFSETS[extra],
    )
    corpora[extra], provenance[extra] = values, prov
    picks = select_ranks(scores["balanced_pool"]["full"]["distances"])
    examples = []
    for q, percentile in zip(picks, PERCENTILES, strict=True):
        uid = split["evaluation"][q]
        examples.append(
            {
                "real_id": uid,
                "rank_percentile_of_balanced_distance": percentile,
                "real_values": rounded(real[uid]),
                "neighbours": {
                    name: neighbour(
                        scores, name, q, corpora[name], provenance[name], selected
                    )
                    for name in (*CORPORA[:2], extra)
                },
            }
        )
    return {"fold": split["fold"], "seed": seed, "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/feature_coverage/m4")
    )
    parser.add_argument(
        "--transfer-cache", type=Path, default=Path(".cache/feature_coverage/monash")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_examples/examples.json"),
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    report = {
        "environment": environment_metadata(),
        "selection": "evaluation queries at rank percentiles 0.1/0.5/0.9 of the balanced pool's full-feature nearest distance in the first fold of the first generation seed",
        "values": "raw series (trailing 512 observations at most), rounded to six significant digits; plots standardize per series",
        "panels": {},
    }
    report["panels"].update(cross_frequency_examples(args.cache, args.download))
    report["panels"].update(third_panel_examples(args.transfer_cache, args.download))
    report["panels"].update(seasonal_examples(args.transfer_cache, args.download))
    report["panels"].update(
        hourly_examples(args.cache, args.transfer_cache, args.download)
    )
    # Compact encoding: a few raw series should stay small enough to commit.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe_json(report), allow_nan=False, separators=(",", ":")) + "\n"
    )
    print(f"Saved examples to {args.output}", flush=True)


if __name__ == "__main__":
    main()
