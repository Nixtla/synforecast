"""Check experimental distance/grid diagnostics against small independent oracles."""

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def validation(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "coverage_validation", directory / "validate_feature_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hourly(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "hourly_investigation", directory / "investigate_hourly_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fresh_hourly(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "fresh_hourly_validation", directory / "validate_hourly_splits.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def composition(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "hourly_composition_validation", directory / "validate_hourly_composition.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def targeted(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "traffic_composition_validation", directory / "validate_traffic_composition.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cross(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "cross_frequency_benchmark", directory / "benchmark_cross_frequency.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def third(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "third_panel_benchmark", directory / "benchmark_third_panels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def seasonal(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "seasonal_composition_validation",
        directory / "validate_seasonal_composition.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def preset_validation(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "seasonal_preset_validation", directory / "validate_seasonal_preset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_hourly_queries_exclude_old_queries_and_are_disjoint(fresh_hourly):
    ids = [f"H{i}" for i in range(414)]
    splits = fresh_hourly.fresh_splits(ids, ids[:32])
    queries = []
    for split in splits:
        query, cal, candidates = [
            set(split[k]) for k in ("evaluation", "calibration", "candidates")
        ]
        assert (len(query), len(cal), len(candidates)) == (64, 32, 256)
        assert not query & cal and not query & candidates and not cal & candidates
        assert set(split["reference"]) <= candidates
        assert len(split["reference"]) == 64
        assert not (query | cal | candidates) & set(ids[:32])
        queries.extend(query)
    assert len(set(queries)) == len(queries) == 256
    assert splits == fresh_hourly.fresh_splits(ids, ids[:32])


def test_daily_profile_shape_is_phase_scale_and_offset_invariant(fresh_hourly):
    cycle = np.sin(2 * np.pi * np.arange(24) / 24)
    x = np.tile(cycle, 10)
    a = fresh_hourly.daily_profile(x)
    b = fresh_hourly.daily_profile(np.roll(x * 3 + 9, 5))
    d, _ = fresh_hourly.profile_nearest(a[None], b[None])
    np.testing.assert_allclose(d, 0, atol=1e-14)
    second = fresh_hourly.daily_profile(
        np.tile(np.sin(4 * np.pi * np.arange(24) / 24), 10)
    )
    d, _ = fresh_hourly.profile_nearest(a[None], second[None])
    np.testing.assert_allclose(d, np.sqrt(2))
    np.testing.assert_array_equal(fresh_hourly.daily_profile(np.ones(48)), np.zeros(24))


def test_hourly_control_variants_share_components_and_are_reproducible(fresh_hourly):
    variants = fresh_hourly.controlled_variants(42, count=3)
    again = fresh_hourly.controlled_variants(42, count=3)
    for uid in variants["cycle"]:
        np.testing.assert_allclose(
            variants["cycle_drift"][uid] - variants["cycle"][uid],
            variants["cycle_drift_amplitude"][uid] - variants["cycle_amplitude"][uid],
            atol=1e-14,
        )
        for name in variants:
            np.testing.assert_array_equal(variants[name][uid], again[name][uid])
            assert len(variants[name][uid]) == 512
            assert np.isfinite(variants[name][uid]).all()


def test_hourly_persistence_gate_requires_three_splits(fresh_hourly):
    def run(coverage):
        return {
            "folds": [
                {
                    "scores": {
                        name: {
                            "full": {
                                "real_baseline_coverage": [0.5, 0.9, 0.95],
                                "coverage": [0.2, value, 0.8],
                            }
                        }
                        for name in fresh_hourly.CORPORA
                    }
                }
                for value in coverage
            ]
        }

    assert fresh_hourly.persistence_gate([run([0.1, 0.1, 0.1, 0.9])])[
        "trigger_controls"
    ]
    assert not fresh_hourly.persistence_gate([run([0.1, 0.1, 0.9, 0.9])])[
        "trigger_controls"
    ]


def test_hourly_metric_geometry_changes_nearest_candidate(hourly):
    query = np.array([[0.0, 0.0]])
    candidates = np.array([[3.0, 0.0], [2.0, 2.0]])
    distance, index = hourly.nearest_metric(query, candidates)
    np.testing.assert_allclose(distance, [np.sqrt(8)])
    assert index.tolist() == [1]
    distance, index = hourly.nearest_metric(query, candidates, "manhattan")
    np.testing.assert_allclose(distance, [3])
    assert index.tolist() == [0]


def test_hourly_attribution_is_actual_squared_distance_decomposition(hourly):
    result = hourly.distance_attribution(
        np.array([[3.0, 4.0]]), np.array([[0.0, 0.0]]), np.array([0]), ["a", "b"]
    )
    assert result["a"]["mean_squared_distance_fraction"] == pytest.approx(9 / 25)
    assert result["b"]["mean_squared_distance_fraction"] == pytest.approx(16 / 25)
    assert result["b"]["median_real_minus_neighbour_standardized"] == 4
    identical = hourly.distance_attribution(
        np.zeros((1, 2)), np.zeros((1, 2)), np.array([0]), ["a", "b"]
    )
    assert identical["a"]["mean_squared_distance_fraction"] == 0


def test_hourly_component_variance_context_is_scale_and_offset_invariant(hourly):
    values = np.sin(np.arange(96) * 2 * np.pi / 24) + np.arange(96) / 100
    original = hourly.component_variance_ratios(values)
    transformed = hourly.component_variance_ratios(values * 3 + 7)
    assert transformed == pytest.approx(original)
    assert original["trend"] > 0
    assert hourly.component_variance_ratios(np.ones(96)) == {
        "trend": 0,
        "seasonal": 0,
        "remainder": 0,
    }


def test_hourly_fixed_anchor_is_independent_of_candidate_size_and_synthetic(hourly):
    cal, query, anchor = np.array([[1.0], [3.0]]), np.array([[2.0]]), np.array([[0.0]])
    small = hourly.metric_scores(
        cal, query, anchor, np.array([[5.0]]), anchor, "euclidean"
    )
    large = hourly.metric_scores(
        cal,
        query,
        np.array([[0.0], [2.0]]),
        np.array([[5.0], [2.5]]),
        anchor,
        "euclidean",
    )
    np.testing.assert_allclose(small["anchor_radii"], large["anchor_radii"])
    assert large["distances"][0] <= small["distances"][0]
    assert not np.array_equal(small["radii"], large["radii"])
    assert np.all(
        np.array(large["fixed_anchor_coverage"]) >= small["fixed_anchor_coverage"]
    )


def test_hourly_feature_omissions_rerank_without_changing_frozen_parameters(hourly):
    n = len(hourly.FEATURE_NAMES)
    rows = {
        "ref": np.zeros(n),
        "cal": np.ones(n),
        "query": np.zeros(n),
        "a": np.zeros(n),
        "b": np.zeros(n),
    }
    synthetic = {"a": np.zeros(n), "b": np.zeros(n)}
    synthetic["a"][0] = 4
    synthetic["b"][1] = 3
    parameters = {
        "mean": np.zeros(n).tolist(),
        "scale": np.ones(n).tolist(),
        "components": np.eye(n)[:2].tolist(),
    }
    split = {"reference": ["ref"], "calibration": ["cal"], "evaluation": ["query"]}
    result = hourly.evaluate_candidates(
        rows, split, ["a", "b"], {"source": synthetic}, parameters
    )["source"]
    assert result["full"]["neighbour_indices"].tolist() == [1]
    assert result["omit_spectral_entropy"]["neighbour_indices"].tolist() == [0]
    assert parameters["scale"] == [1.0] * n


def test_distance_oracle_and_real_only_calibration(validation):
    calibration = np.array([[1.0], [3.0]])
    evaluation = np.array([[0.0], [10.0]])
    real = np.array([[0.0], [4.0]])
    synth = np.array([[0.0], [9.0]])
    result = validation.distance_scores(calibration, evaluation, real, synth)
    np.testing.assert_allclose(result["calibration_radii"], [1, 1, 1])
    np.testing.assert_allclose(result["synthetic_distances"], [0, 1])
    np.testing.assert_allclose(result["real_baseline_distances"], [0, 6])
    np.testing.assert_allclose(result["synthetic_covered"], [1, 1, 1])
    np.testing.assert_allclose(result["real_baseline_covered"], [0.5, 0.5, 0.5])
    changed = validation.distance_scores(
        calibration, evaluation * 100, real, synth + 100
    )
    np.testing.assert_array_equal(
        result["calibration_radii"], changed["calibration_radii"]
    )
    with pytest.raises(ValueError, match="non-empty"):
        validation.nearest(evaluation, synth[:0])


def test_missing_local_cache_explains_rebuild(validation, tmp_path):
    with pytest.raises(FileNotFoundError, match="Rebuild both pilot seeds"):
        validation.load_cache(tmp_path / "real.parquet")


def test_final_schema_accepts_unchanged_pilot_but_rejects_other_definitions(validation):
    for label in ("native_candidate_v1", "native_v1"):
        validation.validate_native_schema(
            {"schema": label, "features": validation.FEATURE_NAMES}
        )
    with pytest.raises(ValueError, match="schema"):
        validation.validate_native_schema(
            {"schema": "native_v2", "features": validation.FEATURE_NAMES}
        )
    with pytest.raises(ValueError, match="schema"):
        validation.validate_native_schema(
            {"schema": "native_v1", "features": validation.FEATURE_NAMES[::-1]}
        )


def test_full_space_detects_a_change_hidden_by_projection(validation):
    rng = np.random.default_rng(10)
    reference = rng.normal(size=(80, 4))
    _, params = validation.fit_spaces(reference, [])
    components = np.array(params["components"])
    # Construct a displacement orthogonal to both retained components.
    _, _, vt = np.linalg.svd(components, full_matrices=True)
    shift = vt[-1] * np.array(params["scale"]) * 10
    spaces, unchanged = validation.fit_spaces(reference, [reference + shift])
    assert unchanged == params
    assert validation.nearest(spaces["full"][1], spaces["full"][0])[0].min() > 3
    np.testing.assert_allclose(
        validation.nearest(spaces["pca2"][1], spaces["pca2"][0])[0], 0, atol=1e-13
    )


def test_splits_are_disjoint_nested_and_repeatable(validation):
    split = validation.split_indices(256, {"a": 250, "b": 256}, 42)
    again = validation.split_indices(256, {"a": 250, "b": 256}, 42)
    all_real = np.concatenate(
        [split[k] for k in ("reference", "calibration", "evaluation")]
    )
    assert len(np.unique(all_real)) == 256
    for key in ("reference", "calibration", "evaluation"):
        np.testing.assert_array_equal(split[key], again[key])
    for values in split["synthetic"].values():
        assert len(values) == len(np.unique(values)) == 128
        assert set(values[:64]) < set(values[:128])
    with pytest.raises(ValueError, match="256 complete"):
        validation.split_indices(255, {"a": 250}, 42)


def test_grid_keeps_outliers_in_the_series_denominator(validation):
    edges = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    real = np.array([[0.5, 0.5], [1.5, 1.5], [3, 3]])
    synthetic = np.array([[0.5, 0.5], [9, 9]])
    result = validation.grid_scores(real, synthetic, edges)
    assert result["uncovered_real_cell_fraction"] == 0.5
    assert result["uncovered_real_series_fraction"] == pytest.approx(2 / 3)
    assert result["real_out_of_range_fraction"] == pytest.approx(1 / 3)
    assert result["synthetic_out_of_range_fraction"] == 0.5


@pytest.mark.parametrize("phases", [(0, 0), (0, 0.5), (0.25, 0), (0.75, 0.25)])
def test_grid_phase_preserves_width_and_encloses_original_range(validation, phases):
    edges = (np.linspace(-1, 1, 4), np.linspace(-3, 3, 4))
    shifted = validation.shifted_edges(edges, phases)
    assert len(shifted[0]) == len(shifted[1])
    for original, changed in zip(edges, shifted, strict=True):
        np.testing.assert_allclose(np.diff(changed), original[1] - original[0])
        assert changed[0] <= original[0]
        assert changed[-1] >= original[-1]
    if phases == (0, 0):
        np.testing.assert_array_equal(shifted, edges)


def test_more_candidates_cannot_increase_distance(validation):
    rng = np.random.default_rng(2)
    query, candidates = rng.normal(size=(24, 5)), rng.normal(size=(30, 5))
    small, _ = validation.nearest(query, candidates[:15])
    large, _ = validation.nearest(query, candidates)
    assert np.all(large <= small)


def test_fixed_grid_baseline_agrees_with_public_api(validation):
    from synforecast import compare_feature_coverage

    rng = np.random.default_rng(7)
    real, synthetic = rng.normal(size=(48, 3)), rng.normal(size=(50, 3)) * 2

    def frame(values):
        return pl.DataFrame(
            {
                "unique_id": [f"id{i:03d}" for i in range(len(values))],
                **{f"f{i}": values[:, i] for i in range(3)},
            }
        )

    reference = compare_feature_coverage(
        frame(real), {"synthetic": frame(synthetic)}, precomputed=True
    )["synthetic"]
    experiments = validation.grid_sensitivity(
        real, {"synthetic": synthetic}, np.arange(len(real))
    )
    baseline = next(
        r
        for r in experiments["runs"]
        if r["bounds"] == "real"
        and r["n_bins_before_shift"] == 30
        and r["phase"] == (0, 0)
    )
    np.testing.assert_allclose(baseline["edges"], reference.grid_edges)
    for name in (
        "uncovered_real_cell_fraction",
        "uncovered_real_series_fraction",
        "synthetic_out_of_range_fraction",
    ):
        assert baseline["scores"]["synthetic"][name] == pytest.approx(
            getattr(reference, name)
        )


def test_distance_details_replay_and_fixed_radius_sensitivity(validation, tmp_path):
    rng = np.random.default_rng(2)
    real = rng.normal(size=(256, 4))
    synthetic = {name: rng.normal(size=(140, 4)) for name in validation.CORPORA}
    ids = np.arange(len(real))
    synthetic_ids = {name: np.arange(len(x)) for name, x in synthetic.items()}
    details = validation.repeated_distances(real, synthetic, ids, synthetic_ids, 9, 2)
    summary = validation.summarize_sensitivity(details["runs"])
    for name in validation.CORPORA:
        fixed = summary[name][
            "candidates_64_to_128_full_coverage_change_fixed_128_q90_radius"
        ]
        assert fixed["median"] >= 0
        assert fixed["repeat_p10_p90"][0] >= 0
    path = tmp_path / "details.json.gz"
    first = validation.save_details(path, details)
    restored = json.loads(gzip.decompress(path.read_bytes()))
    assert (
        restored["splits"][0]["reference"] == details["splits"][0]["reference"].tolist()
    )
    assert first == validation.save_details(path, details)
    split = details["splits"][0]
    fit = details["fits"][0]
    params = fit["parameters"]
    kept = params["kept_columns"]
    query = (
        real[split["evaluation"]][:, kept] - np.array(params["mean"])[kept]
    ) / np.array(params["scale"])[kept]
    run = details["runs"][0]
    candidates = synthetic[run["corpus"]][split["synthetic"][run["corpus"]]][:64]
    candidates = (candidates[:, kept] - np.array(params["mean"])[kept]) / np.array(
        params["scale"]
    )[kept]
    distances, indices = validation.nearest(query, candidates)
    np.testing.assert_allclose(distances, run["synthetic_distances"])
    np.testing.assert_array_equal(indices, run["synthetic_neighbour_indices"])


def test_composition_pool_uses_public_generators_and_leaves_presets_unchanged(
    composition,
):
    from synforecast import interpretable_pool, pretraining_pool
    from synforecast.generators import (
        EnergyLoadGenerator,
        ETSGenerator,
        SeasonalGenerator,
        TSIGenerator,
    )

    pool = composition.native_composition_pool(length=96, seed=7)
    assert [p.alias for p in pool] == list(composition.COMPOSITION_RECIPE)
    assert len(set(composition.COMPOSITION_RECIPE)) == 8
    for prototype in pool:
        assert isinstance(
            prototype,
            (TSIGenerator, ETSGenerator, SeasonalGenerator, EnergyLoadGenerator),
        )
        for name in ("seasonal_period", "seasonality_period"):
            if hasattr(prototype, name):
                assert getattr(prototype, name) == 24
        if isinstance(prototype, TSIGenerator):
            assert set(prototype.seasonal_periods) <= {24.0, 168.0}
            assert prototype.n_seasonal_range[0] >= 1
        values = prototype.generate(1, n_jobs=1)["y"].to_numpy()
        assert len(values) == 96 and np.isfinite(values).all()
    # Production presets are untouched by the benchmark-only composition.
    assert len(interpretable_pool(freq="h")) == 42
    assert len(pretraining_pool(freq="h")) == 54
    assert composition.COMPOSITION_SEED_OFFSET not in (0, 100000, 200000)


def test_composition_generation_matches_lengths_and_is_reproducible(composition):
    lengths = {f"slot{i:03d}": 64 + 8 * i for i in range(8)}
    values, provenance, failures, configs = composition.generate_composition(lengths, 5)
    again = composition.generate_composition(lengths, 5)[0]
    assert failures == {}
    assert {uid: len(v) for uid, v in values.items()} == lengths
    assert sorted(provenance.values()) == sorted(composition.COMPOSITION_RECIPE)
    assert [c["alias"] for c in configs] == list(composition.COMPOSITION_RECIPE)
    for uid in lengths:
        np.testing.assert_array_equal(values[uid], again[uid])
    other = composition.generate_composition(lengths, 6)[0]
    assert any(not np.array_equal(values[uid], other[uid]) for uid in lengths)


def test_tsf_reader_and_seeded_windows(composition, tmp_path):
    path = tmp_path / "panel.tsf"
    path.write_text(
        "# comment\n@relation x\n@frequency hourly\n@missing false\n@data\n"
        "T1:2015-01-01 00-00-01:1,2,3,4,5,6\nT2:2015-01-01 00-00-01:6,5,4,3,2,1\n"
    )
    series = composition.read_tsf_hourly(path)
    assert list(series) == ["T1", "T2"]
    np.testing.assert_array_equal(series["T2"], [6, 5, 4, 3, 2, 1])
    windows, offsets = composition.seeded_windows(series, 4, 3)
    again, _ = composition.seeded_windows(series, 4, 3)
    for uid in series:
        assert len(windows[uid]) == 4 and 0 <= offsets[uid] <= 2
        np.testing.assert_array_equal(windows[uid], series[uid][offsets[uid] :][:4])
        np.testing.assert_array_equal(windows[uid], again[uid])
    with pytest.raises(ValueError, match="shorter"):
        composition.seeded_windows(series, 7, 3)
    path.write_text("@missing false\n@data\nT1:s:1,2\nT2:s:1,2,3\n")
    with pytest.raises(ValueError, match="equal lengths"):
        composition.read_tsf_hourly(path)
    path.write_text("@missing true\n@data\nT1:s:1,2\n")
    with pytest.raises(ValueError, match="complete"):
        composition.read_tsf_hourly(path)


def test_composition_criterion_and_paired_effects(composition):
    def run(existing, new):
        return {
            "seed": 1,
            "folds": [
                {
                    "fold": fold,
                    "scores": {
                        **{
                            name: {
                                metric: {
                                    "coverage": [0.1, existing[fold], 0.5],
                                    "radii": [1, 2, 3],
                                    "distance_quantiles_p10_p50_p90": [1, 2, 3],
                                }
                                for metric in composition.METRICS
                            }
                            for name in composition.CORPORA
                        },
                        composition.COMPOSITION: {
                            metric: {
                                "coverage": [0.1, new[fold], 0.5],
                                "radii": [1, 2, 3],
                                "distance_quantiles_p10_p50_p90": [1, 1.5, 3],
                            }
                            for metric in composition.METRICS
                        },
                    },
                }
                for fold in range(4)
            ],
        }

    passing = [run([0.0, 0.0, 0.0, 0.0], [0.3, 0.3, 0.3, 0.1])]
    assert composition.composition_criterion(passing)["passed"]
    assert not composition.composition_criterion(
        [run([0.0, 0.0, 0.0, 0.0], [0.3, 0.3, 0.1, 0.1])]
    )["passed"]
    effects = composition.paired_effects(
        passing,
        lambda _run, fold, metric: fold["scores"][
            composition.best_existing(fold["scores"], metric)
        ][metric],
    )["full"]
    assert effects["median_q90_coverage_change"] == pytest.approx(0.3)
    assert effects["fraction_runs_with_higher_q90_coverage"] == 1
    assert effects["median_nearest_distance_change"] == pytest.approx(-0.5)
    with pytest.raises(AssertionError):
        composition.check_replay(
            passing[0]["folds"], run([0.0, 0.0, 0.0, 0.1], [0.0] * 4)
        )


def test_targeted_pool_keeps_first_round_prototypes_and_adds_level_noise(
    targeted, composition
):
    from synforecast import interpretable_pool
    from synforecast.generators import (
        EnergyLoadGenerator,
        SeasonalGenerator,
        TSIGenerator,
    )

    pool = targeted.targeted_composition_pool(length=96, seed=3)
    first = {p.alias: p for p in composition.native_composition_pool(length=96, seed=3)}
    assert [p.alias for p in pool] == list(targeted.TARGETED_RECIPE)
    kept = [
        alias
        for alias, note in targeted.TARGETED_RECIPE.items()
        if note.startswith("unchanged")
    ]
    assert kept == ["tsi_daily_shaped", "tsi_daily_weekly", "seasonal_trend_breaks"]
    for prototype in pool:
        assert isinstance(
            prototype, (TSIGenerator, SeasonalGenerator, EnergyLoadGenerator)
        )
        if prototype.alias in kept:
            assert prototype.model_dump(exclude={"seed"}) == first[
                prototype.alias
            ].model_dump(exclude={"seed"})
        values = prototype.generate(1, n_jobs=1)["y"].to_numpy()
        assert len(values) == 96 and np.isfinite(values).all()
    noisy = [p for p in pool if p.alias.endswith("_noisy")]
    assert len(noisy) == 2 and all(p.noise_scale_range[0] >= 0.3 for p in noisy)
    assert sum(p.anomalies for p in pool) == 3
    assert len(interpretable_pool(freq="h")) == 42
    assert targeted.TARGETED_SEED_OFFSET not in (
        0,
        100000,
        200000,
        composition.COMPOSITION_SEED_OFFSET,
    )
    lengths = {f"slot{i:03d}": 64 for i in range(8)}
    values, provenance, failures, _ = targeted.generate_targeted(lengths, 4)
    assert failures == {} and sorted(provenance.values()) == sorted(
        targeted.TARGETED_RECIPE
    )
    for uid in lengths:
        assert not np.array_equal(
            values[uid], composition.generate_composition(lengths, 4)[0][uid]
        )


def test_targeted_secondary_checks_use_declared_thresholds(targeted):
    def summary(fraction, change):
        return {
            "paired_targeted_vs_first_round": {
                "full": {
                    "fraction_runs_with_higher_q90_coverage": fraction,
                    "median_q90_coverage_change": change,
                }
            }
        }

    result = targeted.secondary_checks(summary(0.75, 0.0), summary(0.0, -0.05))
    assert result["fresh_traffic_passed"] and result["m4_retention_passed"]
    result = targeted.secondary_checks(summary(0.5, 0.0), summary(0.0, -0.11))
    assert not result["fresh_traffic_passed"] and not result["m4_retention_passed"]


def test_cross_frequency_recipes_follow_the_audited_recommendations(cross):
    recipes = cross.panel_recipes()
    assert set(recipes) == {"Yearly", "Quarterly", "Monthly", "Weekly", "Daily"}
    assert recipes["Weekly"] == ("W", None, 10)
    assert recipes["Monthly"] == ("MS", 12, 12)
    assert recipes["Yearly"][1] is None and recipes["Daily"][1:] == (7, 7)
    assert cross.FLAGS["material"] > cross.FLAGS["minor"] > 0
    assert cross.GENERATION_SEEDS[0] != cross.SPLIT_SEED


def test_cross_frequency_drops_incomplete_real_rows_and_constant_features(cross):
    n = len(cross.FEATURE_NAMES)
    rows = {"a": np.ones(n), "b": np.full(n, np.nan), "c": np.arange(n, dtype=float)}
    complete, dropped = cross.complete_real(rows)
    assert list(complete) == ["a", "c"] and dropped == ["b"]
    parameters = {
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "kept_columns": [i for i in range(n) if cross.FEATURE_NAMES[i] != "seas_acf1"],
    }
    kept, schemes = cross.projections(parameters)
    assert "seas_acf1" not in kept and len(kept) == n - 1
    assert schemes["omit_trend_strength"][0].shape == (n - 1, n - 2)
    projected = cross.project(np.arange(n, dtype=float)[None], parameters)
    assert projected.shape == (1, n - 1)
    parameters["kept_columns"] = [cross.FEATURE_NAMES.index("acf1")]
    assert "omit_trend_strength" not in cross.projections(parameters)[1]


def test_cross_frequency_gap_flags_use_median_over_seeds(cross):
    def run(best, real=0.9):
        return {
            "folds": [
                {
                    "scores": {
                        name: {
                            "full": {
                                "coverage": [
                                    0.1,
                                    value if name == "balanced_pool" else 0.0,
                                    0.5,
                                ],
                                "real_baseline_coverage": [0.5, real, 0.95],
                            }
                        }
                        for name in cross.CORPORA
                    }
                }
                for value in best
            ]
        }

    flags = cross.gap_flags(
        [
            run([0.8, 0.8, 0.8, 0.8]),
            run([0.6, 0.6, 0.6, 0.0]),
            run([0.7, 0.7, 0.7, 0.0]),
        ]
    )
    assert flags["per_split_median_gap"] == pytest.approx([0.2, 0.2, 0.2, 0.9])
    assert flags["minor_gap"] and not flags["material_gap"]
    assert not cross.gap_flags([run([0.85] * 4)])["minor_gap"]


def test_third_panels_declare_one_public_panel_per_non_hourly_frequency(third, cross):
    assert set(third.PANELS) == set(cross.panel_recipes())
    assert "Hourly" not in third.PANELS
    for spec in third.PANELS.values():
        assert len(spec["sha256"]) == 64 and spec["file"].endswith(".zip")
    assert third.SPLIT_SEED not in (cross.SPLIT_SEED, 20260914, 20260915)
    assert not set(third.GENERATION_SEEDS) & set(cross.GENERATION_SEEDS)


def test_third_panel_tsf_reader_accepts_unequal_lengths_and_latin1(third, tmp_path):
    path = tmp_path / "panel.tsf"
    path.write_bytes(
        b"# Athanasopoulos \x96 tourism\n@relation x\n@missing false\n@equallength false\n"
        b"@data\nT1:1979-01-01 00-00-00:1,2,3\nT2:rain:4,5\n"
    )
    series = third.read_tsf(path)
    assert [len(v) for v in series.values()] == [3, 2]
    np.testing.assert_array_equal(series["T2"], [4, 5])
    path.write_text("@missing false\n@data\nT1:s:1,?,3\n")
    with pytest.raises(ValueError, match="Missing values"):
        third.read_tsf(path)


def test_third_panel_sampling_and_windows_are_seeded(third):
    series = {f"S{i}": np.arange(700 if i % 2 else 100, dtype=float) for i in range(10)}
    real, offsets = third.prepare_real(series, 3, n_sample=6, length=512)
    again, _ = third.prepare_real(series, 3, n_sample=6, length=512)
    assert len(real) == 6 and list(real) == list(again)
    for uid, values in real.items():
        assert len(values) == (512 if len(series[uid]) > 512 else len(series[uid]))
        assert values[0] == offsets[uid]
        np.testing.assert_array_equal(values, again[uid])
    assert all(offsets[uid] == 0 for uid in real if len(series[uid]) <= 512)


@pytest.mark.parametrize(("period", "freq"), [(4, "QS"), (12, "MS")])
def test_seasonal_pool_is_period_specific_and_uses_public_classes(
    seasonal, period, freq
):
    from synforecast import interpretable_pool
    from synforecast.generators import ETSGenerator, SeasonalGenerator, TSIGenerator

    pool = seasonal.seasonal_composition_pool(period, freq, length=60, seed=1)
    assert [p.alias for p in pool] == list(seasonal.RECIPE)
    for prototype in pool:
        assert isinstance(prototype, (TSIGenerator, ETSGenerator, SeasonalGenerator))
        for name in ("seasonal_period", "seasonality_period"):
            if hasattr(prototype, name):
                assert getattr(prototype, name) == period
        if isinstance(prototype, TSIGenerator):
            assert prototype.seasonal_periods == [float(period)]
        if isinstance(prototype, ETSGenerator) and prototype.seasonal_type == "mul":
            assert len(prototype.seasonal) == period
            assert np.mean(prototype.seasonal) == pytest.approx(1.0)
        values = prototype.generate(1, n_jobs=1)["y"].to_numpy()
        assert len(values) == 60 and np.isfinite(values).all()
    assert len(interpretable_pool(freq=freq)) == 42


def test_seasonal_factors_and_declarations(seasonal, third, cross):
    factors = seasonal.seasonal_factors(12, 1.0)
    assert len(factors) == 12 and max(factors) > 2 * min(factors)
    assert np.mean(factors) == pytest.approx(1.0)
    assert set(seasonal.DESIGN_PANELS) <= set(third.PANELS)
    for spec in seasonal.VALIDATION_PANELS.values():
        assert spec["group"] in seasonal.DESIGN_PANELS and len(spec["sha256"]) == 64
    assert seasonal.VALIDATION_SPLIT_SEED not in (third.SPLIT_SEED, cross.SPLIT_SEED)
    assert not set(seasonal.VALIDATION_GENERATION_SEEDS) & set(third.GENERATION_SEEDS)
    offsets = set(seasonal.SEED_OFFSETS.values())
    assert len(offsets) == 2 and not offsets & {0, 100000, 200000, 300000, 400000}
    assert set(seasonal.POOLS) == set(seasonal.COMPOSITIONS)


def test_seasonal_augmented_pool_analysis_uses_union_minimum(seasonal, tmp_path):
    import gzip

    def fold(balanced, pretraining, moderated, strong, radius=1.0):
        return {
            "scores": {
                "balanced_pool": {
                    "full": {"radii": [0.5, radius, 2.0], "distances": balanced}
                },
                "pretraining_pool": {"full": {"distances": pretraining}},
                "moderated_seasonal_composition": {"full": {"distances": moderated}},
                "seasonal_composition": {"full": {"distances": strong}},
            }
        }

    details = [
        fold(
            [2.0, 2.0, 0.5, 2.0], [0.5, 2.0, 2.0, 2.0], [0.5, 0.5, 2.0, 0.5], [2.0] * 4
        ),
        fold(
            [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [0.5, 2.0, 2.0, 2.0], [2.0] * 4
        ),
    ]
    (tmp_path / "panel_1_details.json.gz").write_bytes(
        gzip.compress(json.dumps(details).encode())
    )
    result = seasonal.augmented_pool_analysis(tmp_path, ["panel"])["panel"]
    assert result["n_fold_draw_pairs"] == 2
    assert result["balanced_plus_pretraining_median_q90"] == pytest.approx(0.25)
    moderated = result["balanced_plus_moderated_seasonal_composition"]
    assert moderated["median_q90"] == pytest.approx(0.625)
    assert moderated["paired_median_change_vs_control"] == pytest.approx(0.375)
    assert moderated["fraction_pairs_higher"] == 1.0
    strong = result["balanced_plus_seasonal_composition"]
    assert strong["fraction_pairs_higher"] == 0.0


def test_seasonal_preset_validation_declares_new_panels_and_union_criterion(
    preset_validation, seasonal, third, cross
):
    module = preset_validation
    assert module.SPLIT_SEED not in (
        cross.SPLIT_SEED,
        third.SPLIT_SEED,
        seasonal.VALIDATION_SPLIT_SEED,
    )
    used = set(cross.GENERATION_SEEDS) | set(third.GENERATION_SEEDS)
    used |= set(seasonal.VALIDATION_GENERATION_SEEDS)
    assert not set(module.GENERATION_SEEDS) & used
    assert module.SEED_OFFSET not in set(seasonal.SEED_OFFSETS.values())
    monash = {v["file"] for v in module.PANELS.values() if v["source"] == "monash"}
    earlier = {v["file"] for v in third.PANELS.values()}
    earlier |= {v["file"] for v in seasonal.VALIDATION_PANELS.values()}
    assert not monash & earlier
    scores = {
        "balanced_pool": {
            "full": {"radii": [0.5, 1.0, 2.0], "distances": [2.0, 0.5, 2.0, 2.0]}
        },
        "pretraining_pool": {"full": {"distances": [0.5, 2.0, 2.0, 2.0]}},
        "seasonal_pool": {"full": {"distances": [0.5, 2.0, 0.5, 2.0]}},
    }
    union = module.union_coverage(scores)
    assert union["balanced_plus_pretraining_pool"] == pytest.approx(0.5)
    assert union["balanced_plus_seasonal_pool"] == pytest.approx(0.75)
    assert union["paired_change"] == pytest.approx(0.25)
    runs = [
        {"folds": [{"union": {"paired_change": c}} for c in changes]}
        for changes in ([0.1, 0.1, 0.1, 0.0], [0.2, 0.0, 0.2, -0.01])
    ]
    result = module.criterion(runs)
    assert result["per_split_median_paired_change"] == pytest.approx(
        [0.15, 0.05, 0.15, -0.005]
    )
    assert result["gain"] and result["no_harm"]
    assert not module.criterion(
        [{"folds": [{"union": {"paired_change": c}} for c in [0.1, 0.1, -0.05, 0.1]]}]
    )["no_harm"]
    pool = module.preset_factory(4, "QS", length=40, seed=1)
    assert [g.alias for g in pool] == [
        g.alias for g in seasonal.moderated_seasonal_pool(4, "QS", length=40, seed=1)
    ]
