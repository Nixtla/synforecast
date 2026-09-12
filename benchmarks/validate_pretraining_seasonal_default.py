"""Pre-declared test for making ``include_seasonal`` the pretraining_pool default.

Compares two versions of the public pool as equal-count corpora on every real
panel used so far, plus fresh M4 samples: ``pretraining_pool()`` as shipped and
``pretraining_pool(include_seasonal=True)``. Because both are drawn the same
way into 256 requested and 240 matched slots, the comparison includes the
share displacement a default change would cause. Yearly panels are excluded
because the flag leaves the pool unchanged there.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import benchmark_cross_frequency as cross
import benchmark_third_panels as third
import numpy as np
from _env import environment_metadata
from benchmark_feature_coverage import (
    extract,
    matched_complete_ids,
    sample_training,
    save_json,
    training_file,
)
from validate_feature_coverage import save_details
from validate_hourly_composition import seeded_windows
from validate_seasonal_composition import VALIDATION_PANELS
from validate_seasonal_preset import PANELS as PRESET_PANELS

from synforecast import pretraining_pool
from synforecast._features import FEATURE_NAMES, FEATURE_SCHEMA

CURRENT = "pretraining_pool"
EXTENDED = "pretraining_pool_with_seasonal"
SOURCES = (CURRENT, EXTENDED)
SPLIT_SEED = 20260920
GENERATION_SEEDS = (91261913, 92261913, 93261913)
CORPUS_OFFSETS = {CURRENT: 100000, EXTENDED: 800000}
N_SAMPLE = 640
# Declared before any result. Paired change is EXTENDED minus CURRENT coverage
# at the fold's real-calibrated q90 radius, median over the three draws.
# Decision rule for flipping the default: no-harm on every panel and gain on
# at least three of the four strongly seasonal target panels.
CRITERION = {
    "no_harm_minimum_median_paired_change": -0.02,
    "gain_minimum_median_paired_change": 0.05,
    "gain_minimum_splits": 3,
    "gain_target_panels": [
        "tourism_quarterly",
        "tourism_monthly",
        "hospital",
        "m1_monthly",
    ],
    "gain_target_minimum_panels": 3,
}
RECIPES = {
    "Hourly": ("h", 24, 24),
    **cross.panel_recipes(),
}
PANELS = {
    "m4_quarterly": {"source": "m4", "group": "Quarterly"},
    "m4_monthly": {"source": "m4", "group": "Monthly"},
    "m4_weekly": {"source": "m4", "group": "Weekly"},
    "m4_daily": {"source": "m4", "group": "Daily"},
    "m4_hourly": {"source": "m4", "group": "Hourly"},
    "tourism_quarterly": {
        "source": "monash",
        "group": "Quarterly",
        **third.PANELS["Quarterly"],
    },
    "hospital": {"source": "monash", "group": "Monthly", **third.PANELS["Monthly"]},
    "traffic_weekly": {"source": "monash", "group": "Weekly", **third.PANELS["Weekly"]},
    "weather": {"source": "monash", "group": "Daily", **third.PANELS["Daily"]},
    "traffic_hourly": {
        "source": "monash",
        "group": "Hourly",
        "record": 4656132,
        "file": "traffic_hourly_dataset.zip",
        "sha256": "3db12ba866a9c9d3c8109b7b6d189a990c38d0e5002fa2617022157358d08299",
        "description": "862 hourly freeway occupancy series, seeded 512-observation windows",
    },
    "m3_quarterly": {"source": "monash", **VALIDATION_PANELS["m3_quarterly"]},
    "m3_monthly": {"source": "monash", **VALIDATION_PANELS["m3_monthly"]},
    "tourism_monthly": {"source": "monash", **VALIDATION_PANELS["tourism_monthly"]},
    "m1_monthly": {"source": "monash", **PRESET_PANELS["m1_monthly"]},
    "car_parts": {"source": "monash", **PRESET_PANELS["car_parts"]},
}
M4_COUNTS = {
    "Quarterly": 640,
    "Monthly": 640,
    "Weekly": 359,
    "Daily": 640,
    "Hourly": 414,
}


def load_panel(spec, m4_cache, monash_cache, download):
    group = spec["group"]
    if spec["source"] == "m4":
        source, _ = training_file(m4_cache, group, download)
        real, _ = sample_training(source, group, M4_COUNTS[group], SPLIT_SEED, 512)
        return real
    series = third.fetch_panel(monash_cache, spec, download)
    if group == "Hourly":
        real, _ = seeded_windows(series, 512, SPLIT_SEED)
        return real
    real, _ = third.prepare_real(series, SPLIT_SEED)
    return real


def draw_pool(pool, template, seed):
    """Same cycling allocation and per-slot seeding as the pretraining benchmark draws."""
    rng = np.random.default_rng(seed)
    assigned = np.concatenate(
        [
            rng.permutation(len(pool))
            for _ in range((len(template) + len(pool) - 1) // len(pool))
        ]
    )[: len(template)]
    rng.shuffle(assigned)
    values, provenance, failures = {}, {}, {}
    for i, (uid, reference) in enumerate(template.items()):
        prototype = pool[int(assigned[i])]
        config = prototype.model_dump(
            by_alias=True, include=set(type(prototype).model_fields)
        )
        config.update(
            min_length=len(reference), max_length=len(reference), seed=seed + i
        )
        provenance[uid] = (
            f"{int(assigned[i])}:{prototype.alias or type(prototype).__name__}"
        )
        try:
            generated = type(prototype)(**config).generate(1, n_jobs=1)["y"].to_numpy()
            if len(generated) != len(reference) or not np.isfinite(generated).all():
                raise ValueError("length mismatch or non-finite output")
            values[uid] = generated
        except (ValueError, RuntimeError) as exc:
            failures[uid] = str(exc)
    return values, provenance, failures


def draw_fold(real, split, period, frequency, window, seed):
    slots = [f"slot{i:03d}" for i in range(len(split["candidates"]))]
    template = {
        slot: real[uid] for slot, uid in zip(slots, split["candidates"], strict=True)
    }
    pools = {
        CURRENT: pretraining_pool(
            freq=frequency, seasonal_period=period, engine="polars"
        ),
        EXTENDED: pretraining_pool(
            freq=frequency,
            seasonal_period=period,
            engine="polars",
            include_seasonal=True,
        ),
    }
    rows, generation = (
        {},
        {"provenance": {}, "failures": {}, "n_retained_by_source": {}},
    )
    frames = {}
    for name, pool in pools.items():
        values, provenance, failures = draw_pool(
            pool, template, seed + CORPUS_OFFSETS[name]
        )
        frames[name] = extract(values, period, failures, window_size=window)
        rows[name] = dict(
            zip(
                frames[name]["unique_id"].to_list(),
                frames[name].select(FEATURE_NAMES).to_numpy(),
                strict=True,
            )
        )
        generation["provenance"][name] = provenance
        generation["failures"][name] = failures
        generation["n_retained_by_source"][name] = int(
            sum(np.isfinite(row).all() for row in rows[name].values())
        )
        generation[f"{name}_size"] = len(pool)
    valid = set(matched_complete_ids(frames))
    common = [slot for slot in slots if slot in valid]
    return rows, common, generation


def paired_criterion(runs):
    per_fold = [
        float(np.median([run["folds"][fold]["paired_change"] for run in runs]))
        for fold in range(4)
    ]
    return {
        "per_split_median_paired_change": per_fold,
        "no_harm": all(
            c >= CRITERION["no_harm_minimum_median_paired_change"] for c in per_fold
        ),
        "gain": sum(
            c >= CRITERION["gain_minimum_median_paired_change"] for c in per_fold
        )
        >= CRITERION["gain_minimum_splits"],
    }


def run_panel(name, spec, args, output):
    frequency, period, window = RECIPES[spec["group"]]
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
            synthetic_rows, common, generation = draw_fold(
                real, split, period, frequency, window, fold_seed
            )
            if len(common) < cross.MINIMUM_MATCHED:
                raise ValueError(
                    f"{name} seed {fold_seed}: only {len(common)} shared slots"
                )
            selected = common[: cross.N_MATCHED]
            scores, parameters, real_ids = cross.score_fold(
                rows, split, selected, synthetic_rows, SOURCES
            )
            change = (
                scores[EXTENDED]["full"]["coverage"][1]
                - scores[CURRENT]["full"]["coverage"][1]
            )
            folds.append(
                {
                    "fold": split["fold"],
                    "seed": fold_seed,
                    "n_shared_complete": len(common),
                    "pool_sizes": {n: generation[f"{n}_size"] for n in SOURCES},
                    "paired_change": float(change),
                    "nearest_prototype_counts": {
                        src: dict(
                            Counter(
                                generation["provenance"][src][selected[i]]
                                for i in scores[src]["full"]["neighbour_indices"]
                            )
                        )
                        for src in SOURCES
                    },
                    "kept_features": [
                        FEATURE_NAMES[i] for i in parameters["kept_columns"]
                    ],
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
            f"{name} seed {seed}: paired change by fold "
            + str([round(f["paired_change"], 4) for f in folds])
            + "; q90 "
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
        "description": spec.get("description", f"M4 {spec['group']} training sample"),
        "group": spec["group"],
        "frequency": frequency,
        "seasonal_period": period,
        "window_size": window,
        "n_sampled": len(real),
        "n_complete": len(rows),
        "dropped_incomplete_ids": dropped,
        "splits": splits,
        "runs": runs,
        "summary": cross.aggregate(runs, SOURCES),
        "paired_change_summary": {
            "median": float(
                np.median([f["paired_change"] for r in runs for f in r["folds"]])
            ),
            "min_max": [
                float(min(f["paired_change"] for r in runs for f in r["folds"])),
                float(max(f["paired_change"] for r in runs for f in r["folds"])),
            ],
            "fraction_pairs_higher": float(
                np.mean([f["paired_change"] > 0 for r in runs for f in r["folds"]])
            ),
        },
        "criterion": paired_criterion(runs),
    }
    if spec["source"] == "monash":
        panel["source"] = {k: spec[k] for k in ("record", "file", "sha256")}
    for run in runs:
        for fold in run["folds"]:
            for scores in fold["scores"].values():
                del scores["attribution"]
    print(f"  {name}: {panel['criterion']}", flush=True)
    return panel


def decision(panels):
    no_harm_everywhere = all(p["criterion"]["no_harm"] for p in panels.values())
    gains = [
        name
        for name in CRITERION["gain_target_panels"]
        if name in panels and panels[name]["criterion"]["gain"]
    ]
    return {
        "no_harm_on_every_panel": no_harm_everywhere,
        "failing_no_harm": [
            n for n, p in panels.items() if not p["criterion"]["no_harm"]
        ],
        "gain_target_panels_passed": gains,
        "flip_default": bool(
            no_harm_everywhere and len(gains) >= CRITERION["gain_target_minimum_panels"]
        ),
    }


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
        default=Path("benchmarks/data/feature_coverage_pretraining_seasonal_default"),
    )
    parser.add_argument("--panels", nargs="+", choices=list(PANELS))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "completed": False,
        "schema": FEATURE_SCHEMA,
        "environment": environment_metadata(),
        "protocol": {
            "sources": {
                CURRENT: "pretraining_pool() as shipped: interpretable_pool + 12 meta-generator instances",
                EXTENDED: "pretraining_pool(include_seasonal=True): the same plus the 8 seasonal_pool instances",
            },
            "split_seed": SPLIT_SEED,
            "generation_seeds": GENERATION_SEEDS,
            "corpus_seed_offsets": CORPUS_OFFSETS,
            "allocation": "shuffled cycling across all prototypes per corpus, 256 requested, first 240 shared complete slots matched to real candidates",
            "criterion": CRITERION,
            "panels": {
                k: {kk: vv for kk, vv in v.items() if kk != "sha256"}
                for k, v in PANELS.items()
            },
            "yearly": "excluded; include_seasonal leaves the pool unchanged at period 1",
            "limits": "coverage only; real roles overlap across folds; twelve fold-draw measurements per panel are not independent; no confidence intervals; forecasting utility not measured",
        },
        "panels": {},
    }
    for name in args.panels or list(PANELS):
        report["panels"][name] = run_panel(name, PANELS[name], args, args.output)
        save_json(args.output / "summary.json", report)
    report["decision"] = decision(report["panels"])
    report["completed"] = args.panels is None
    save_json(args.output / "summary.json", report)
    print(f"Decision: {report['decision']}", flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
