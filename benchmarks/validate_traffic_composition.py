"""Second composition round, targeted at the traffic_hourly differences.

The first composition round transferred weakly to Monash traffic_hourly, and
its nearest neighbours differed from real traffic queries in spectral entropy,
autocorrelation, crossing points, and maximum level shift. This round keeps
the public generator classes, adds noisier and level-shifting configurations,
and is evaluated on traffic queries never used before, with an M4 retention
check on the earlier splits. No generator, preset, or feature is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from _env import environment_metadata
from benchmark_feature_coverage import sample_training, save_json, training_file
from investigate_hourly_coverage import CORPORA, draw_candidates, feature_rows
from validate_feature_coverage import save_details
from validate_hourly_composition import (
    COMPOSITION,
    best_existing,
    check_replay,
    composition_criterion,
    generate_composition,
    paired_effects,
    prototype_neighbour_counts,
    seeded_windows,
    transfer_panel,
)
from validate_hourly_splits import (
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
    SeasonalGenerator,
    TSIGenerator,
)

TARGETED = "targeted_composition"
SOURCES = (*CORPORA, COMPOSITION, TARGETED)
COMPOSITIONS = (COMPOSITION, TARGETED)
# New split seed; the first round's 256 traffic evaluation IDs are excluded.
TRAFFIC_SPLIT_SEED = 20260915
TARGETED_SEED_OFFSET = 400000
# Declared before inspecting any result. Primary: same 25-point, three-of-four
# fold criterion against the best existing source on fresh traffic queries.
# Secondary: the targeted round should beat the first round in a majority of
# paired fresh-traffic comparisons and lose at most 10 points of median paired
# full-feature coverage on the earlier (non-fresh) M4 splits.
SECONDARY = {
    "minimum_fraction_fresh_traffic_pairs_above_first_round": 0.5,
    "minimum_median_m4_paired_change_vs_first_round": -0.10,
}
TARGETED_RECIPE = {
    "tsi_daily_shaped": "unchanged from the first round",
    "tsi_daily_weekly": "unchanged from the first round",
    "tsi_daily_noisy": "first-round daily TSI with noise 30–150% of signal and all five irregular processes (heavy tails, GARCH-like)",
    "tsi_daily_weekly_noisy": "first-round daily/weekly TSI with the same noisier irregular component",
    "seasonal_trend_breaks": "unchanged from the first round",
    "seasonal_level_outages": "sine period 24 with two level shifts plus injected temporary drops and dips (outage-like)",
    "energy_residential_weekend": "residential profile with sharper commuting peaks, stronger weekend reduction, more noise, slope change, rare deep dips",
    "energy_commercial_weekend": "commercial profile with stronger weekend reduction, more noise, extreme-weather multipliers, slope change, rare temporary drops",
}


def targeted_composition_pool(length=512, freq="h", seed=None, engine="polars"):
    """Public classes only; noisier, level-shifting variants replace the ETS pair."""
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
    weekly = {
        **tsi,
        "seasonal_periods": [24.0, 168.0],
        "n_seasonal_range": (2, 2),
        "harmonics_prob": 0.5,
        "amplitude_modulation_prob": 0.5,
    }
    noisy = {
        "noise_scale_range": (0.3, 1.5),
        "irregular_types": ["gaussian", "ar1", "garch_like", "student_t", "laplace"],
        "tail_df_range": (2.5, 6.0),
    }
    return [
        TSIGenerator(**base, seed=_seed(0), alias="tsi_daily_shaped", **tsi),
        TSIGenerator(**base, seed=_seed(1), alias="tsi_daily_weekly", **weekly),
        TSIGenerator(
            **base, seed=_seed(2), alias="tsi_daily_noisy", **{**tsi, **noisy}
        ),
        TSIGenerator(
            **base, seed=_seed(3), alias="tsi_daily_weekly_noisy", **{**weekly, **noisy}
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
            alias="seasonal_level_outages",
            seasonality_period=24,
            seasonality_amplitude=1.0,
            noise_level=0.2,
            base_level=10.0,
            changepoints=True,
            num_changepoints=2,
            changepoint_type="level",
            changepoint_level_changes=[1.0, -1.2],
            anomalies=True,
            anomaly_types=["level_shift", "dip"],
            anomaly_fraction=0.02,
            level_shift_magnitude=-1.5,
            level_shift_duration=12,
            dip_magnitude=-2.0,
        ),
        EnergyLoadGenerator(
            **base,
            seed=_seed(6),
            alias="energy_residential_weekend",
            load_type="residential",
            daily_amplitude=20.0,
            peak_amplitude=60.0,
            weekly_amplitude=40.0,
            noise_std=15.0,
            changepoints=True,
            num_changepoints=1,
            changepoint_type="trend",
            changepoint_trend_changes=[0.1],
            anomalies=True,
            anomaly_types=["dip"],
            anomaly_fraction=0.01,
            dip_magnitude=-90.0,
        ),
        EnergyLoadGenerator(
            **base,
            seed=_seed(7),
            alias="energy_commercial_weekend",
            load_type="commercial",
            weekly_amplitude=40.0,
            noise_std=15.0,
            extreme_weather_prob=0.01,
            changepoints=True,
            num_changepoints=1,
            changepoint_type="trend",
            changepoint_trend_changes=[-0.1],
            anomalies=True,
            anomaly_types=["level_shift"],
            anomaly_fraction=0.01,
            level_shift_magnitude=-60.0,
            level_shift_duration=8,
        ),
    ]


def generate_targeted(lengths, seed):
    return generate_composition(
        lengths,
        seed,
        pool_factory=targeted_composition_pool,
        recipe=TARGETED_RECIPE,
        seed_offset=TARGETED_SEED_OFFSET,
    )


def add_composition(
    generated, rows, generation, name, values, provenance, failures, configs, selected
):
    frame_rows = feature_rows(values)
    incomplete = [
        uid
        for uid in selected
        if uid not in frame_rows or not np.isfinite(frame_rows[uid]).all()
    ]
    if incomplete:
        raise ValueError(
            f"{name} incomplete on paired slots {incomplete[:5]}; no replacement allowed"
        )
    generated[name] = values
    rows[name] = frame_rows
    generation[f"{name}_provenance"] = provenance
    generation[f"{name}_failures"] = failures
    generation[f"{name}_configs"] = configs
    generation["n_retained_by_source"][name] = int(
        sum(np.isfinite(row).all() for row in frame_rows.values())
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
                    "nearest_prototype_counts": {
                        name: prototype_neighbour_counts(
                            scores, generation[f"{name}_provenance"], selected, name
                        )
                        for name in COMPOSITIONS
                    },
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
                "n_retained_by_source": generation["n_retained_by_source"],
                "details": detail_file,
                "folds": folds,
            }
        )
        print(
            f"{prefix} seed {seed}: full-feature q90 by split "
            + str(
                {
                    name: [
                        round(float(f["scores"][name]["full"]["coverage"][1]), 4)
                        for f in folds
                    ]
                    for name in SOURCES
                }
            ),
            flush=True,
        )
    return runs


def first_round(_run, fold, metric):
    return fold["scores"][COMPOSITION][metric]


def best_existing_scores(_run, fold, metric):
    return fold["scores"][best_existing(fold["scores"], metric)][metric]


def panel_summary(runs):
    return {
        "aggregate": aggregate_runs(runs, SOURCES),
        "existing_gate": persistence_gate(runs),
        "criterion": {name: composition_criterion(runs, name) for name in COMPOSITIONS},
        "paired_targeted_vs_best_existing": paired_effects(
            runs, best_existing_scores, TARGETED
        ),
        "paired_first_round_vs_best_existing": paired_effects(
            runs, best_existing_scores, COMPOSITION
        ),
        "paired_targeted_vs_first_round": paired_effects(runs, first_round, TARGETED),
        "nearest_prototype_counts": {
            name: {
                metric: dict(
                    sum(
                        (
                            Counter(fold["nearest_prototype_counts"][name][metric])
                            for run in runs
                            for fold in run["folds"]
                        ),
                        Counter(),
                    )
                )
                for metric in ("full", "profile")
            }
            for name in COMPOSITIONS
        },
        "attribution_summary": {
            name: {
                feature: {
                    key: np.median(
                        [
                            fold["scores"][name]["attribution"][feature][key]
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
            }
            for name in COMPOSITIONS
        },
    }


def secondary_checks(traffic_summary, m4_summary):
    fraction = traffic_summary["paired_targeted_vs_first_round"]["full"][
        "fraction_runs_with_higher_q90_coverage"
    ]
    retention = m4_summary["paired_targeted_vs_first_round"]["full"][
        "median_q90_coverage_change"
    ]
    return {
        "fresh_traffic_fraction_above_first_round": fraction,
        "fresh_traffic_passed": bool(
            fraction
            > SECONDARY["minimum_fraction_fresh_traffic_pairs_above_first_round"]
        ),
        "m4_median_paired_change_vs_first_round": retention,
        "m4_retention_passed": bool(
            retention >= SECONDARY["minimum_median_m4_paired_change_vs_first_round"]
        ),
    }


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
        default=Path(
            "benchmarks/data/feature_coverage_hourly_composition/summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_traffic_composition"),
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prior = json.loads(args.prior.read_text())
    if prior["schema"] != FEATURE_SCHEMA or not prior["completed"]:
        raise ValueError("A completed first composition round is required")
    if len(interpretable_pool(freq="h")) != 42:
        raise ValueError("Production interpretable_pool must remain unchanged")
    source, source_metadata = training_file(args.cache, "Hourly", args.download)
    if source_metadata["sha256"] != prior["source"]["sha256"]:
        raise ValueError("Hourly source hash differs from the first round")
    traffic, transfer_metadata = transfer_panel(args.transfer_cache, args.download)
    if transfer_metadata["zip_sha256"] != prior["transfer_source"]["zip_sha256"]:
        raise ValueError("Traffic archive hash differs from the first round")
    real, _ = sample_training(source, "Hourly", 414, SPLIT_SEED, 512)
    m4_splits = prior["splits"]
    real_rows = feature_rows(real)
    windows, offsets = seeded_windows(traffic, 512, SPLIT_SEED)
    if offsets != {k: int(v) for k, v in prior["transfer_window_offsets"].items()}:
        raise ValueError("Traffic windows differ from the first round")
    traffic_rows = feature_rows(windows)
    excluded = [
        uid for split in prior["transfer_splits"] for uid in split["evaluation"]
    ]
    if len(set(excluded)) != 256:
        raise ValueError("Expected 256 previously used traffic evaluation IDs")
    traffic_splits = fresh_splits(list(windows), excluded, seed=TRAFFIC_SPLIT_SEED)
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
            "m4_split_seed": SPLIT_SEED,
            "traffic_split_seed": TRAFFIC_SPLIT_SEED,
            "generation_seeds": GENERATION_SEEDS,
            "targeted_seed_offset": TARGETED_SEED_OFFSET,
            "excluded_first_round_traffic_evaluation_ids": excluded,
            "n_requested_per_source": 256,
            "n_matched_candidates": 240,
            "period": 24,
            "window": 24,
            "length": 512,
            "targeted_recipe": TARGETED_RECIPE,
            "criterion": prior["protocol"]["criterion"],
            "secondary": SECONDARY,
            "m4_role": "retention check on the first round's splits; these queries informed both rounds and are not fresh evidence",
            "traffic_role": "fresh evaluation groups disjoint from every first-round traffic query; the first-round traffic attribution motivated the targeted variants",
            "limits": "single targeted configuration; real roles overlap across folds; three draws reused across panels and rounds; no confidence intervals; forecasting utility not measured",
        },
        "m4_splits": m4_splits,
        "traffic_splits": traffic_splits,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    prior_by_seed = {run["seed"]: run for run in prior["m4_runs"]}
    draws = {}
    for seed in GENERATION_SEEDS:
        generated, rows, common, generation = draw_candidates(template, slots, seed)
        selected = common[:240]
        add_composition(
            generated,
            rows,
            generation,
            COMPOSITION,
            *generate_composition(dict.fromkeys(slots, 512), seed),
            selected,
        )
        add_composition(
            generated,
            rows,
            generation,
            TARGETED,
            *generate_targeted(dict.fromkeys(slots, 512), seed),
            selected,
        )
        draws[seed] = (generated, rows, selected, generation)
        print(
            f"Seed {seed}: retained by source {generation['n_retained_by_source']}; "
            f"targeted failures {len(generation[f'{TARGETED}_failures'])}",
            flush=True,
        )
    report["traffic_runs"] = evaluate_panel(
        windows, traffic_rows, traffic_splits, draws, args.output, "traffic"
    )
    report["traffic_summary"] = panel_summary(report["traffic_runs"])
    save_json(args.output / "summary.json", report)
    print(
        f"Fresh traffic gate: {report['traffic_summary']['existing_gate']}; "
        f"criterion: {report['traffic_summary']['criterion']}",
        flush=True,
    )
    report["m4_runs"] = evaluate_panel(
        real, real_rows, m4_splits, draws, args.output, "m4"
    )
    for run in report["m4_runs"]:
        check_replay(run["folds"], prior_by_seed[run["seed"]], (*CORPORA, COMPOSITION))
    report["m4_summary"] = panel_summary(report["m4_runs"])
    report["secondary_checks"] = secondary_checks(
        report["traffic_summary"], report["m4_summary"]
    )
    print(f"Secondary checks: {report['secondary_checks']}", flush=True)
    report["completed"] = True
    for run in report["traffic_runs"] + report["m4_runs"]:
        for fold in run["folds"]:
            for scores in fold["scores"].values():
                del scores["attribution"]
    save_json(args.output / "summary.json", report)
    print(f"Saved targeted composition validation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
