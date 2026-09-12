"""Pre-declared augmented-pool test of the shipped ``seasonal_pool`` preset.

The seasonal composition round showed, post hoc, that adding a moderated
seasonal corpus to the balanced pool raised coverage on five quarterly and
monthly panels. This script declares that comparison as the criterion up
front and applies it to four panels never used in any earlier step: fresh M4
Quarterly and Monthly samples, M1 monthly, and the intermittent car-parts
panel, which is included to check that the addition does no harm where strong
seasonality is absent. The preset is the public ``synforecast.seasonal_pool``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import benchmark_cross_frequency as cross
import benchmark_third_panels as third
import numpy as np
from _env import environment_metadata
from benchmark_feature_coverage import sample_training, save_json, training_file
from investigate_hourly_coverage import CORPORA
from validate_feature_coverage import save_details
from validate_hourly_composition import generate_composition

from synforecast import interpretable_pool, seasonal_pool
from synforecast._features import FEATURE_NAMES, FEATURE_SCHEMA

PRESET = "seasonal_pool"
SOURCES = (*CORPORA, PRESET)
SEED_OFFSET = 700000
SPLIT_SEED = 20260919
GENERATION_SEEDS = (81261913, 82261913, 83261913)
N_CANDIDATES = cross.N_MATCHED
# Declared before any result. Union coverage uses each fold's real-calibrated
# q90 radius and the minimum of the stored per-source nearest distances, so
# both unions have 480 candidates and the control is more of the existing
# corpus. "Gain" needs a median paired improvement of at least 5 points in
# three of four folds; "no harm" needs no fold's median paired change below
# -2 points. Car parts is expected to test "no harm" rather than "gain".
CRITERION = {
    "gain_minimum_median_paired_change": 0.05,
    "gain_minimum_splits": 3,
    "no_harm_minimum_median_paired_change": -0.02,
}
PANELS = {
    "m4_quarterly_fresh": {
        "source": "m4",
        "group": "Quarterly",
        "description": "640 M4 Quarterly training series sampled with a new seed",
    },
    "m4_monthly_fresh": {
        "source": "m4",
        "group": "Monthly",
        "description": "640 M4 Monthly training series sampled with a new seed",
    },
    "m1_monthly": {
        "source": "monash",
        "group": "Monthly",
        "record": 4656159,
        "file": "m1_monthly_dataset.zip",
        "sha256": "2780e591d1a0e3fb105575b413fcd218d59eac264af045356dbe5f85c9ff470f",
        "description": "617 monthly M1 competition series (Makridakis et al., 1982)",
    },
    "car_parts": {
        "source": "monash",
        "group": "Monthly",
        "record": 4656021,
        "file": "car_parts_dataset_without_missing_values.zip",
        "sha256": "a0b1e87e2329098837e5cfd020e16a36e30efa9a29a1fae8d1bd3603dfefcfea",
        "description": "2674 monthly intermittent car-part sales series, 51 observations each (Hyndman, expsmooth)",
    },
}


def preset_factory(period, freq, length=120, seed=None, engine="polars"):
    return seasonal_pool(
        min_length=length,
        max_length=length,
        freq=freq,
        seed=seed,
        seasonal_period=period,
        engine=engine,
    )


def load_panel(spec, m4_cache, monash_cache, download):
    if spec["source"] == "m4":
        source, _ = training_file(m4_cache, spec["group"], download)
        real, _ = sample_training(source, spec["group"], 640, SPLIT_SEED, 512)
        return real
    series = third.fetch_panel(monash_cache, spec, download)
    real, _ = third.prepare_real(series, SPLIT_SEED)
    return real


def union_coverage(scores):
    """Coverage of 480-candidate unions at the fold's real-calibrated radius."""
    radius = scores["balanced_pool"]["full"]["radii"][1]
    base = np.asarray(scores["balanced_pool"]["full"]["distances"])
    result = {}
    for name in ("pretraining_pool", PRESET):
        union = np.minimum(base, np.asarray(scores[name]["full"]["distances"]))
        result[f"balanced_plus_{name}"] = float(np.mean(union <= radius))
    result["paired_change"] = (
        result[f"balanced_plus_{PRESET}"] - result["balanced_plus_pretraining_pool"]
    )
    return result


def criterion(runs):
    per_fold = []
    for fold in range(4):
        per_fold.append(
            float(
                np.median(
                    [run["folds"][fold]["union"]["paired_change"] for run in runs]
                )
            )
        )
    return {
        "per_split_median_paired_change": per_fold,
        "gain": sum(
            change >= CRITERION["gain_minimum_median_paired_change"]
            for change in per_fold
        )
        >= CRITERION["gain_minimum_splits"],
        "no_harm": all(
            change >= CRITERION["no_harm_minimum_median_paired_change"]
            for change in per_fold
        ),
    }


def run_panel(name, spec, args, output):
    from functools import partial

    frequency, period, window = cross.panel_recipes()[spec["group"]]
    real = load_panel(spec, args.cache, args.monash_cache, args.download)
    rows, dropped = cross.complete_real(cross.feature_rows(real, period, window))
    if len(rows) < 352:
        raise ValueError(f"{name}: only {len(rows)} complete series; need 352")
    splits = cross.fresh_splits(list(rows), [], seed=SPLIT_SEED)
    runs = []
    for seed in GENERATION_SEEDS:
        folds, details = [], []
        for split in splits:
            fold_seed = seed + 1000 * split["fold"]
            synthetic_rows, common, generation = cross.draw_fold(
                real, split, period, frequency, window, fold_seed
            )
            slots = [f"slot{i:03d}" for i in range(len(split["candidates"]))]
            lengths = {
                slot: len(real[uid])
                for slot, uid in zip(slots, split["candidates"], strict=True)
            }
            values, provenance, failures, configs = generate_composition(
                lengths,
                fold_seed,
                pool_factory=partial(preset_factory, period, frequency),
                recipe=dict.fromkeys(
                    g.alias for g in preset_factory(period, frequency)
                ),
                seed_offset=SEED_OFFSET,
            )
            preset_rows = cross.feature_rows(values, period, window)
            selected = common[:N_CANDIDATES]
            incomplete = [
                s
                for s in selected
                if s not in preset_rows or not np.isfinite(preset_rows[s]).all()
            ]
            if incomplete:
                raise ValueError(f"{name}: preset incomplete on {incomplete[:5]}")
            synthetic_rows[PRESET] = preset_rows
            generation["provenance"][PRESET] = provenance
            generation["failures"][PRESET] = failures
            generation["configs"][PRESET] = configs
            scores, parameters, real_ids = cross.score_fold(
                rows, split, selected, synthetic_rows, SOURCES
            )
            folds.append(
                {
                    "fold": split["fold"],
                    "seed": fold_seed,
                    "n_shared_complete": len(common),
                    "kept_features": [
                        FEATURE_NAMES[i] for i in parameters["kept_columns"]
                    ],
                    "nearest_prototype_counts": {
                        src: dict(
                            Counter(
                                generation["provenance"][src][selected[i]]
                                for i in scores[src]["full"]["neighbour_indices"]
                            )
                        )
                        for src in SOURCES
                    },
                    "union": union_coverage(scores),
                    "scores": cross.compact(scores),
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
        detail_file = save_details(output / f"{name}_{seed}_details.json.gz", details)
        runs.append({"seed": seed, "details": detail_file, "folds": folds})
        print(
            f"{name} seed {seed}: union paired change by fold "
            + str([round(f["union"]["paired_change"], 4) for f in folds])
            + "; standalone q90 "
            + str(
                {
                    src: [
                        round(float(f["scores"][src]["full"]["coverage"][1]), 3)
                        for f in folds
                    ]
                    for src in SOURCES
                }
            ),
            flush=True,
        )
    panel = {
        "description": spec["description"],
        "group": spec["group"],
        "frequency": frequency,
        "seasonal_period": period,
        "window_size": window,
        "n_sampled": len(real),
        "n_complete": len(rows),
        "dropped_incomplete_ids": dropped,
        "effective_length_quantiles_min_p25_p50_p75_max": np.quantile(
            [len(real[uid]) for uid in rows], [0, 0.25, 0.5, 0.75, 1]
        ),
        "splits": splits,
        "runs": runs,
        "summary": cross.aggregate(runs, SOURCES),
        "existing_gap_flags": cross.gap_flags(runs),
        "union_summary": {
            key: {
                "median": float(
                    np.median([f["union"][key] for r in runs for f in r["folds"]])
                ),
                "min_max": [
                    float(min(f["union"][key] for r in runs for f in r["folds"])),
                    float(max(f["union"][key] for r in runs for f in r["folds"])),
                ],
            }
            for key in (
                "balanced_plus_pretraining_pool",
                f"balanced_plus_{PRESET}",
                "paired_change",
            )
        }
        | {
            "fraction_pairs_higher": float(
                np.mean(
                    [f["union"]["paired_change"] > 0 for r in runs for f in r["folds"]]
                )
            )
        },
        "criterion": criterion(runs),
    }
    if spec["source"] == "monash":
        panel["source"] = {k: spec[k] for k in ("record", "file", "sha256")}
    for run in runs:
        for fold in run["folds"]:
            for scores in fold["scores"].values():
                del scores["attribution"]
    print(f"  {name}: {panel['criterion']}", flush=True)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/feature_coverage/m4")
    )
    parser.add_argument(
        "--monash-cache", type=Path, default=Path(".cache/feature_coverage/monash")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_seasonal_preset"),
    )
    parser.add_argument("--panels", nargs="+", choices=list(PANELS))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if len(interpretable_pool(freq="MS")) != 42:
        raise ValueError(
            "interpretable_pool must remain unchanged; seasonal_pool is additive"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "completed": False,
        "schema": FEATURE_SCHEMA,
        "environment": environment_metadata(),
        "protocol": {
            "preset": "synforecast.seasonal_pool, period derived from the panel recipe",
            "split_seed": SPLIT_SEED,
            "generation_seeds": GENERATION_SEEDS,
            "preset_seed_offset": SEED_OFFSET,
            "panels": PANELS,
            "criterion": CRITERION,
            "union": "480-candidate unions of the balanced pool with the preset or with the pretraining pool; nearest distance is the per-query minimum; radius is the fold's real-calibrated q90",
            "folds": "same four-fold fresh-query protocol, matched candidates, and real-only calibration as the earlier benchmarks",
            "limits": "one declared preset; real roles overlap across folds; twelve fold-draw measurements per panel are not independent; no confidence intervals; forecasting utility not measured",
        },
        "panels": {},
    }
    for name in args.panels or list(PANELS):
        report["panels"][name] = run_panel(name, PANELS[name], args, args.output)
        save_json(args.output / "summary.json", report)
    report["completed"] = True
    save_json(args.output / "summary.json", report)
    print(f"Saved seasonal preset validation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
