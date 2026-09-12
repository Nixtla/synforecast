"""Seasonal-strength composition round with held-out validation panels.

The third-panel validation found that tourism quarterly and hospital monthly
are under-covered because real series are more strongly seasonal, on a moving
level, than any pool neighbour. This round configures public generator classes
for strong period-4 and period-12 seasonality with level drift, scores the
corpus on those two design panels (same splits and draws, existing scores
replayed), and then on three panels that did not inform it: M3 quarterly, M3
monthly, and tourism monthly. No generator, preset, or feature is changed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import partial
from pathlib import Path

import benchmark_cross_frequency as cross
import benchmark_third_panels as third
import numpy as np
from _env import environment_metadata
from benchmark_feature_coverage import save_json
from investigate_hourly_coverage import CORPORA
from validate_feature_coverage import save_details
from validate_hourly_composition import composition_criterion, generate_composition
from validate_hourly_splits import GATE

from synforecast import interpretable_pool, seasonal_pool
from synforecast._features import FEATURE_NAMES, FEATURE_SCHEMA
from synforecast.generators import ETSGenerator, SeasonalGenerator, TSIGenerator

SEASONAL = "seasonal_composition"
MODERATED = "moderated_seasonal_composition"
COMPOSITIONS = (SEASONAL, MODERATED)
SOURCES = (*CORPORA, *COMPOSITIONS)
METRICS = ("full", "manhattan", "omit_trend_strength")
SEED_OFFSETS = {SEASONAL: 500000, MODERATED: 600000}
# Design panels: the third-panel splits, seeds, and slots are reused exactly.
DESIGN_PANELS = ("Quarterly", "Monthly")
# Validation panels: declared before any result; none informed the recipe.
VALIDATION_PANELS = {
    "m3_quarterly": {
        "group": "Quarterly",
        "record": 4656262,
        "file": "m3_quarterly_dataset.zip",
        "sha256": "c59f208d1ad3d38ff039e0bdc9e356fe7c83019ecd2dcae0a930b607c22b50c2",
        "description": "756 quarterly M3 competition series (Makridakis and Hibon, 2000)",
    },
    "m3_monthly": {
        "group": "Monthly",
        "record": 4656298,
        "file": "m3_monthly_dataset.zip",
        "sha256": "80b0c1c92b6273f12d346256d3ee29489a7b081823430ad8b3a1cc01e06ccafd",
        "description": "1428 monthly M3 competition series (Makridakis and Hibon, 2000)",
    },
    "tourism_monthly": {
        "group": "Monthly",
        "record": 4656096,
        "file": "tourism_monthly_dataset.zip",
        "sha256": "6e434ad8a0ef6acb55ebee0e1d1ffe22707ea4e62c05e091326267f3a5b4a93c",
        "description": "366 monthly tourism demand series (Athanasopoulos et al., 2011)",
    },
}
VALIDATION_SPLIT_SEED = 20260918
VALIDATION_GENERATION_SEEDS = (71261913, 72261913, 73261913)
CRITERION = {
    "minimum_median_composition_minus_best_existing_q90_gain": GATE[
        "minimum_median_real_minus_best_source_q90_gap"
    ],
    "minimum_splits": GATE["minimum_splits"],
    "scope": "primary for the moderated recipe on each validation panel; design panels are reported for reference",
}
# Round 1, declared before any result. On the design panels it produced near-
# deterministic seasonality (seasonal strength 0.9-1.0, low entropy, lag-1 ACF
# near zero at period 4) and covered fewer queries than the existing pools.
RECIPE = {
    "tsi_seasonal_multiplicative": "TSI: one harmonic at the panel period, amplitude 1–4 with overtones and envelopes half the time, trend movement −3..6 composed multiplicatively so the swing grows with the level, noise 5–50% of signal",
    "tsi_seasonal_additive": "TSI: as above with additive composition and trend movement −3..3",
    "ets_MAdM_peaked": "ETS(M,Ad,M): multiplicative factors 1 + 0.5 cos, level 100, damped additive trend, alpha 0.2, gamma 0.05",
    "ets_MAdM_sharp": "ETS(M,Ad,M): multiplicative factors exp(cos) normalized to unit mean, one dominant season per cycle",
    "ets_MNM_walk": "ETS(M,N,M): peaked multiplicative factors on a random-walk level (alpha 0.3), no trend",
    "ets_AAdA_strong": "ETS(A,Ad,A): additive seasonal amplitude 30 on level 100, damped trend, noise 3",
    "seasonal_drift_level_breaks": "Sine at the panel period plus linear drift and two random-location level shifts",
    "seasonal_trend_breaks": "Sine at the panel period plus two random-location slope changes",
}
# Round 2, declared after inspecting round 1 on the design panels only. Real
# under-covered series have moderate seasonal strength (medians 0.56 quarterly,
# 0.22 monthly) under substantial noise, persistent lag-1 dependence, and
# level shifts; the moderated recipe widens noise and dependence ranges so the
# corpus spans that region instead of saturating it.
RECIPE_MODERATED = {
    "tsi_moderate_multiplicative": "TSI: one harmonic at the panel period, amplitude 0.5-3, overtones and envelopes half the time, multiplicative trend movement -3..6, noise 30-300% of signal from Gaussian, AR(1) (phi 0.5-0.95), or Student-t",
    "tsi_moderate_additive": "TSI: as above with additive composition and trend movement -3..3",
    "tsi_persistent": "TSI: additive, AR(1) noise with phi 0.8-0.97 at 100-400% of signal, so lag-1 persistence dominates a moderate seasonal cycle",
    "tsi_weak_seasonal": "TSI: amplitude 0.2-1 under noise 100-600% of signal, covering weakly seasonal series",
    "ets_MAdM_noisy": "ETS(M,Ad,M): peaked multiplicative factors, alpha 0.3, gamma 0.1, multiplicative noise 15%",
    "ets_AAdA_noisy": "ETS(A,Ad,A): additive seasonal amplitude 20 on level 100, alpha 0.3, noise 10",
    "seasonal_noisy_level_breaks": "Sine at the panel period, noise 0.6, linear drift, two random-location level shifts",
    "seasonal_noisy_trend_breaks": "Sine at the panel period, noise 0.6, two random-location slope changes",
}


def seasonal_factors(period, sharpness):
    phase = 2 * np.pi * np.arange(period) / period
    factors = np.exp(sharpness * np.cos(phase))
    return (factors / factors.mean()).tolist()


def seasonal_composition_pool(period, freq, length=120, seed=None, engine="polars"):
    """Public generator classes configured for strong seasonality on a moving level."""
    base = {"min_length": length, "max_length": length, "freq": freq, "engine": engine}

    def _seed(i):
        return None if seed is None else seed + i

    tsi = {
        "seasonal_periods": [float(period)],
        "n_seasonal_range": (1, 1),
        "seasonal_amplitude_range": (1.0, 4.0),
        "harmonics_prob": 0.5,
        "amplitude_modulation_prob": 0.5,
        "trend_types": [
            "linear",
            "piecewise_linear",
            "damped",
            "logistic",
            "exponential",
        ],
        "trend_slope_range": (-3.0, 6.0),
        "multiplicative_prob": 1.0,
        "noise_scale_range": (0.05, 0.5),
        "irregular_types": ["gaussian", "ar1"],
        "ar1_phi_range": (0.3, 0.9),
        "level_range": (5.0, 10.0),
        "scale_range": (1.0, 1.0),
    }
    ets = {
        "seasonal_period": period,
        "level": 100.0,
        "alpha": 0.2,
        "beta": 0.02,
        "gamma": 0.05,
        "phi": 0.9,
    }
    peaked = seasonal_factors(period, 0.5)
    return [
        TSIGenerator(**base, seed=_seed(0), alias="tsi_seasonal_multiplicative", **tsi),
        TSIGenerator(
            **base,
            seed=_seed(1),
            alias="tsi_seasonal_additive",
            **{**tsi, "multiplicative_prob": 0.0, "trend_slope_range": (-3.0, 3.0)},
        ),
        ETSGenerator(
            **base,
            seed=_seed(2),
            alias="ets_MAdM_peaked",
            error_type="mul",
            trend_type="add",
            seasonal_type="mul",
            trend=0.3,
            damped=True,
            noise_std=0.03,
            seasonal=peaked,
            **ets,
        ),
        ETSGenerator(
            **base,
            seed=_seed(3),
            alias="ets_MAdM_sharp",
            error_type="mul",
            trend_type="add",
            seasonal_type="mul",
            trend=0.3,
            damped=True,
            noise_std=0.03,
            seasonal=seasonal_factors(period, 1.0),
            **ets,
        ),
        ETSGenerator(
            **base,
            seed=_seed(4),
            alias="ets_MNM_walk",
            error_type="mul",
            trend_type=None,
            seasonal_type="mul",
            noise_std=0.05,
            seasonal=peaked,
            **{**ets, "alpha": 0.3},
        ),
        ETSGenerator(
            **base,
            seed=_seed(5),
            alias="ets_AAdA_strong",
            error_type="add",
            trend_type="add",
            seasonal_type="add",
            trend=0.2,
            damped=True,
            noise_std=3.0,
            seasonal=(30 * np.cos(2 * np.pi * np.arange(period) / period)).tolist(),
            **ets,
        ),
        SeasonalGenerator(
            **base,
            seed=_seed(6),
            alias="seasonal_drift_level_breaks",
            seasonality_period=period,
            seasonality_amplitude=1.0,
            trend=0.008,
            noise_level=0.1,
            base_level=10.0,
            changepoints=True,
            num_changepoints=2,
            changepoint_type="level",
            changepoint_level_changes=[0.6, -0.5],
        ),
        SeasonalGenerator(
            **base,
            seed=_seed(7),
            alias="seasonal_trend_breaks",
            seasonality_period=period,
            seasonality_amplitude=1.0,
            noise_level=0.1,
            base_level=10.0,
            changepoints=True,
            num_changepoints=2,
            changepoint_type="trend",
            changepoint_trend_changes=[0.01, -0.015],
        ),
    ]


def moderated_seasonal_pool(period, freq, length=120, seed=None, engine="polars"):
    """Round 2 is the shipped opt-in ``seasonal_pool``; one definition only."""
    return seasonal_pool(
        min_length=length,
        max_length=length,
        freq=freq,
        seed=seed,
        seasonal_period=period,
        engine=engine,
    )


POOLS = {
    SEASONAL: (seasonal_composition_pool, RECIPE),
    MODERATED: (moderated_seasonal_pool, RECIPE_MODERATED),
}


def draw_with_composition(real, split, period, frequency, window, seed):
    """Existing draws as in the cross-frequency benchmark, plus both compositions."""
    rows, common, generation = cross.draw_fold(
        real, split, period, frequency, window, seed
    )
    slots = [f"slot{i:03d}" for i in range(len(split["candidates"]))]
    lengths = {
        slot: len(real[uid])
        for slot, uid in zip(slots, split["candidates"], strict=True)
    }
    selected = common[: cross.N_MATCHED]
    for name, (factory, recipe) in POOLS.items():
        values, provenance, failures, configs = generate_composition(
            lengths,
            seed,
            pool_factory=partial(factory, period, frequency),
            recipe=recipe,
            seed_offset=SEED_OFFSETS[name],
        )
        composition_rows = cross.feature_rows(values, period, window)
        incomplete = [
            slot
            for slot in selected
            if slot not in composition_rows
            or not np.isfinite(composition_rows[slot]).all()
        ]
        if incomplete:
            raise ValueError(
                f"{name} incomplete on paired slots {incomplete[:5]}; no replacement allowed"
            )
        rows[name] = composition_rows
        generation["provenance"][name] = provenance
        generation["failures"][name] = failures
        generation["configs"][name] = configs
        generation["n_retained_by_source"][name] = int(
            sum(np.isfinite(row).all() for row in composition_rows.values())
        )
    return rows, selected, generation


def evaluate_splits(
    real, rows, splits, period, frequency, window, seeds, output, label
):
    runs = []
    for seed in seeds:
        folds, details = [], []
        for split in splits:
            fold_seed = seed + 1000 * split["fold"]
            synthetic_rows, selected, generation = draw_with_composition(
                real, split, period, frequency, window, fold_seed
            )
            scores, parameters, real_ids = cross.score_fold(
                rows, split, selected, synthetic_rows, SOURCES
            )
            folds.append(
                {
                    "fold": split["fold"],
                    "seed": fold_seed,
                    "n_shared_complete": generation["n_shared_complete"],
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
                        for name in SOURCES
                    },
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
        detail_file = save_details(output / f"{label}_{seed}_details.json.gz", details)
        runs.append({"seed": seed, "details": detail_file, "folds": folds})
        print(
            f"{label} seed {seed}: full-feature q90 by fold "
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


def paired_effects(runs, name=SEASONAL):
    """Median paired q90 change of the composition against the best existing source."""
    results = {}
    for metric in METRICS:
        differences, distance_changes = [], []
        for run in runs:
            for fold in run["folds"]:
                scores = fold["scores"]
                best = max(CORPORA, key=lambda n: scores[n][metric]["coverage"][1])
                differences.append(
                    scores[name][metric]["coverage"][1]
                    - scores[best][metric]["coverage"][1]
                )
                distance_changes.append(
                    scores[name][metric]["distance_quantiles_p10_p50_p90"][1]
                    - scores[best][metric]["distance_quantiles_p10_p50_p90"][1]
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


def check_replay(runs, prior_panel):
    """Existing sources must reproduce the third-panel scores exactly."""
    prior_by_seed = {run["seed"]: run for run in prior_panel["runs"]}
    for run in runs:
        for fold, recorded in zip(
            run["folds"], prior_by_seed[run["seed"]]["folds"], strict=True
        ):
            for name in CORPORA:
                for metric in METRICS:
                    for key in ("coverage", "radii", "distance_quantiles_p10_p50_p90"):
                        np.testing.assert_allclose(
                            fold["scores"][name][metric][key],
                            recorded["scores"][name][metric][key],
                            rtol=1e-12,
                            atol=0,
                        )


def summarize(panel):
    runs = panel["runs"]
    panel["summary"] = cross.aggregate(runs, SOURCES)
    panel["existing_gap_flags"] = cross.gap_flags(runs)
    panel["criterion"] = {
        name: composition_criterion(runs, name) for name in COMPOSITIONS
    }
    panel["paired_vs_best_existing"] = {
        name: paired_effects(runs, name) for name in COMPOSITIONS
    }
    for run in runs:
        for fold in run["folds"]:
            for scores in fold["scores"].values():
                del scores["attribution"]
    print(
        f"  existing gap {panel['existing_gap_flags']['per_split_median_gap']}; "
        f"criterion {panel['criterion']}",
        flush=True,
    )
    return panel


def augmented_pool_analysis(output, panel_names):
    """Equal-count union comparison from saved per-query distances.

    A composition can only be added to a pool, not replace it, so compare the
    balanced pool plus each composition (480 candidates) against the balanced
    pool plus the pretraining pool (480 candidates of more of the same). Union
    nearest distances are minima of the stored per-source distances; radii are
    the fold's real-calibrated ones, so unions are never advantaged by density
    over the 480-candidate control.
    """
    import gzip

    results = {}
    for panel in panel_names:
        per_union = {name: [] for name in (*COMPOSITIONS, "pretraining_pool")}
        for path in sorted(output.glob(f"{panel}_*_details.json.gz")):
            for fold in json.loads(gzip.decompress(path.read_bytes())):
                scores = fold["scores"]
                radius = scores["balanced_pool"]["full"]["radii"][1]
                base = np.asarray(scores["balanced_pool"]["full"]["distances"])
                for name in per_union:
                    union = np.minimum(
                        base, np.asarray(scores[name]["full"]["distances"])
                    )
                    per_union[name].append(float(np.mean(union <= radius)))
        control = np.asarray(per_union["pretraining_pool"])
        results[panel] = {
            "n_fold_draw_pairs": len(control),
            "balanced_plus_pretraining_median_q90": float(np.median(control)),
        }
        for name in COMPOSITIONS:
            values = np.asarray(per_union[name])
            results[panel][f"balanced_plus_{name}"] = {
                "median_q90": float(np.median(values)),
                "paired_median_change_vs_control": float(np.median(values - control)),
                "min_max_change": [
                    float((values - control).min()),
                    float((values - control).max()),
                ],
                "fraction_pairs_higher": float(np.mean(values > control)),
            }
    return results


def run_design_panel(group, cache, download, prior, output):
    spec = third.PANELS[group]
    frequency, period, window = cross.panel_recipes()[group]
    series = third.fetch_panel(cache, spec, download)
    real, offsets = third.prepare_real(series, third.SPLIT_SEED)
    rows, _ = cross.complete_real(cross.feature_rows(real, period, window))
    prior_panel = prior["panels"][group]
    splits = cross.fresh_splits(list(rows), [], seed=third.SPLIT_SEED)
    if splits != prior_panel["splits"]:
        raise ValueError(f"{group}: splits differ from the third-panel validation")
    panel = {
        "role": "design",
        "source": prior_panel["source"],
        "frequency": frequency,
        "seasonal_period": period,
        "window_size": window,
        "n_complete": len(rows),
        "splits": splits,
        "runs": evaluate_splits(
            real,
            rows,
            splits,
            period,
            frequency,
            window,
            third.GENERATION_SEEDS,
            output,
            f"design_{group}",
        ),
    }
    check_replay(panel["runs"], prior_panel)
    return summarize(panel)


def run_validation_panel(name, cache, download, output):
    spec = VALIDATION_PANELS[name]
    group = spec["group"]
    frequency, period, window = cross.panel_recipes()[group]
    series = third.fetch_panel(cache, spec, download)
    real, offsets = third.prepare_real(series, VALIDATION_SPLIT_SEED)
    rows, dropped = cross.complete_real(cross.feature_rows(real, period, window))
    if len(rows) < 352:
        raise ValueError(f"{name}: only {len(rows)} complete series; need 352")
    splits = cross.fresh_splits(list(rows), [], seed=VALIDATION_SPLIT_SEED)
    lengths = [len(real[uid]) for uid in rows]
    panel = {
        "role": "validation",
        "source": {k: spec[k] for k in ("record", "file", "sha256", "description")},
        "group": group,
        "frequency": frequency,
        "seasonal_period": period,
        "window_size": window,
        "n_series_in_panel": len(series),
        "n_sampled": len(real),
        "n_complete": len(rows),
        "dropped_incomplete_ids": dropped,
        "effective_length_quantiles_min_p25_p50_p75_max": np.quantile(
            lengths, [0, 0.25, 0.5, 0.75, 1]
        ),
        "splits": splits,
        "runs": evaluate_splits(
            real,
            rows,
            splits,
            period,
            frequency,
            window,
            VALIDATION_GENERATION_SEEDS,
            output,
            name,
        ),
    }
    return summarize(panel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/feature_coverage/monash")
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_third_panels/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/data/feature_coverage_seasonal_composition"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--augment-only",
        action="store_true",
        help="recompute the equal-count union analysis from saved details",
    )
    args = parser.parse_args()
    if args.augment_only:
        report = json.loads((args.output / "summary.json").read_text())
        report["augmented_pool_480"] = augmented_pool_analysis(
            args.output, list(report["panels"])
        )
        save_json(args.output / "summary.json", report)
        print(json.dumps(report["augmented_pool_480"], indent=1), flush=True)
        return
    prior = json.loads(args.prior.read_text())
    if prior["schema"] != FEATURE_SCHEMA or not prior["completed"]:
        raise ValueError("A completed third-panel validation is required")
    if len(interpretable_pool(freq="QS")) != 42:
        raise ValueError("Production interpretable_pool must remain unchanged")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "completed": False,
        "schema": FEATURE_SCHEMA,
        "environment": environment_metadata(),
        "protocol": {
            "design_panels": DESIGN_PANELS,
            "design_split_seed": third.SPLIT_SEED,
            "design_generation_seeds": third.GENERATION_SEEDS,
            "validation_panels": VALIDATION_PANELS,
            "validation_split_seed": VALIDATION_SPLIT_SEED,
            "validation_generation_seeds": VALIDATION_GENERATION_SEEDS,
            "composition_seed_offsets": SEED_OFFSETS,
            "recipes": {SEASONAL: RECIPE, MODERATED: RECIPE_MODERATED},
            "rounds": "round 1 was declared before any result and evaluated on the design panels; round 2 was declared after inspecting round 1 on the design panels only; both are scored everywhere, and no recipe was changed after the validation panels were run",
            "criterion": CRITERION,
            "n_matched_candidates": cross.N_MATCHED,
            "folds": "same four-fold fresh-query protocol, matched candidates, and real-only calibration as the third-panel validation; composition draws are fold-specific and matched to the same 240 slots",
            "limits": "single declared configuration; real roles overlap across folds; twelve fold-draw measurements per panel are not independent; no confidence intervals; forecasting utility not measured",
        },
        "panels": {},
    }
    for group in DESIGN_PANELS:
        report["panels"][f"design_{group}"] = run_design_panel(
            group, args.cache, args.download, prior, args.output
        )
        save_json(args.output / "summary.json", report)
    if not args.skip_validation:
        for name in VALIDATION_PANELS:
            report["panels"][name] = run_validation_panel(
                name, args.cache, args.download, args.output
            )
            save_json(args.output / "summary.json", report)
    report["augmented_pool_480"] = augmented_pool_analysis(
        args.output, list(report["panels"])
    )
    report["completed"] = not args.skip_validation
    save_json(args.output / "summary.json", report)
    print(f"Saved seasonal composition validation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
