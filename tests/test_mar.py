"""Tests for MARGenerator and feature-targeted MAR generation."""

import numpy as np
import pytest

from synforecast._features import acf1, compute_features
from synforecast.base import _GEN_TYPE_MAP, BaseGenerator
from synforecast.generators import MARGenerator
from tests.helpers import assert_acf, assert_long_format, sample_acf, series_values

BASE = {"min_length": 64, "max_length": 128, "freq": "h", "seed": 42}


class TestMarApi:
    """Public API and structural behavior."""

    def test_is_base_generator(self) -> None:
        assert isinstance(MARGenerator(**BASE), BaseGenerator)

    def test_long_format(self, engine: str) -> None:
        frame = MARGenerator(**BASE, engine=engine).generate(n_series=4)
        assert_long_format(frame, n_series=4, min_length=64, max_length=128)

    def test_seed_determinism(self, engine: str) -> None:
        first = series_values(MARGenerator(**BASE, engine=engine).generate(n_series=4))
        second = series_values(MARGenerator(**BASE, engine=engine).generate(n_series=4))
        assert first.keys() == second.keys()
        for series_id in first:
            np.testing.assert_array_equal(first[series_id], second[series_id])

    def test_different_seeds_differ(self) -> None:
        first = MARGenerator(**{**BASE, "seed": 1}).generate_single_series(96)
        second = MARGenerator(**{**BASE, "seed": 2}).generate_single_series(96)
        assert not np.array_equal(first, second)

    def test_requested_lengths_are_finite(self) -> None:
        generator = MARGenerator(**BASE)
        for length in (1, 2, 10, 200):
            values = generator.generate_single_series(length)
            assert values.shape == (length,)
            assert np.all(np.isfinite(values))

    def test_native_batch_parameter_contract(self) -> None:
        generator = MARGenerator(**BASE)
        assert generator._batch_gen_type == _GEN_TYPE_MAP["MARGenerator"] == 30
        scalars, arrays = generator._get_batch_params()
        assert scalars.shape == (12,)
        assert arrays == []

    def test_native_batch_is_independent_of_worker_count(self) -> None:
        first = MARGenerator(**BASE).generate(n_series=8, n_jobs=1)
        second = MARGenerator(**BASE).generate(n_series=8, n_jobs=2)
        first_values = series_values(first)
        second_values = series_values(second)
        assert first_values.keys() == second_values.keys()
        for series_id in first_values:
            np.testing.assert_array_equal(
                first_values[series_id], second_values[series_id]
            )

    def test_fixed_native_batch_parameter_layout(self) -> None:
        generator = MARGenerator(
            **BASE,
            weights=[0.25, 0.75],
            ar_coefficients=[[0.5], [0.2, -0.1]],
            intercepts=[1.0, -1.0],
            noise_scales=[0.5, 1.5],
        )
        scalars, arrays = generator._get_batch_params()
        assert scalars[9] == 1.0
        np.testing.assert_array_equal(arrays[1], [1.0, 2.0])
        np.testing.assert_array_equal(arrays[2], [0.5, 0.2, -0.1])


class TestMarBehavior:
    """Statistical and fixed-mode behavior."""

    def test_finite_and_bounded_across_pool(self) -> None:
        generator = MARGenerator(**{**BASE, "seed": 0})
        for _ in range(64):
            values = generator.generate_single_series(128)
            assert np.all(np.isfinite(values))
            assert np.abs(values).max() < 1e8

    def test_standardized_by_default(self) -> None:
        generator = MARGenerator(**{**BASE, "seed": 0})
        for _ in range(12):
            values = generator.generate_single_series(256)
            assert values.mean() == pytest.approx(0.0, abs=1e-7)
            assert values.std() == pytest.approx(1.0, abs=1e-7)

    def test_standardize_false_keeps_raw_scale(self) -> None:
        generator = MARGenerator(**BASE, standardize=False)
        scales = [generator.generate_single_series(256).std() for _ in range(12)]
        assert any(abs(scale - 1.0) > 1e-3 for scale in scales)

    @pytest.mark.parametrize("native", [False, True])
    @pytest.mark.parametrize("length", [1, 2, 128])
    def test_random_fallback_honors_standardization(
        self, native: bool, length: int
    ) -> None:
        # These intercepts force all eight attempted paths beyond the bound.
        args = {
            **BASE,
            "min_length": length,
            "max_length": length,
            "intercept_scale": 1e20,
        }
        raw_generator = MARGenerator(**args, standardize=False)
        normalized_generator = MARGenerator(**args, standardize=True)

        def draw(generator: MARGenerator) -> np.ndarray:
            if native:
                return next(iter(series_values(generator.generate(1)).values()))
            return generator.generate_single_series(length)

        raw = draw(raw_generator)
        normalized = draw(normalized_generator)
        assert np.abs(raw).max() < 10.0  # Confirms the Gaussian fallback was used.
        expected = (raw - raw.mean()) / raw.std() if length > 1 else raw
        np.testing.assert_allclose(normalized, expected, atol=1e-12)

    def test_seasonal_period_is_accepted(self) -> None:
        values = MARGenerator(**BASE, seasonal_period=24).generate_single_series(200)
        assert np.all(np.isfinite(values))

    def test_fixed_parameters_vary_with_seed(self) -> None:
        fixed = {
            "weights": [1.0],
            "ar_coefficients": [[0.7]],
            "intercepts": [0.0],
            "noise_scales": [1.0],
        }
        first = MARGenerator(**{**BASE, "seed": 1}, **fixed).generate_single_series(512)
        second = MARGenerator(**{**BASE, "seed": 2}, **fixed).generate_single_series(
            512
        )
        assert not np.array_equal(first, second)
        assert abs(acf1(first)) < 1.0
        assert abs(acf1(second)) < 1.0

    @pytest.mark.stats
    def test_random_pool_has_acf_diversity(self) -> None:
        generator = MARGenerator(**{**BASE, "seed": 7})
        autocorrelations = np.array(
            [acf1(generator.generate_single_series(256)) for _ in range(48)]
        )
        assert np.ptp(autocorrelations) > 0.5

    @pytest.mark.stats
    def test_native_random_pool_has_acf_diversity(self) -> None:
        generator = MARGenerator(
            **{**BASE, "seed": 7, "min_length": 256, "max_length": 256}
        )
        values = series_values(generator.generate(n_series=48))
        autocorrelations = np.array([acf1(series) for series in values.values()])
        assert np.ptp(autocorrelations) > 0.5
        assert np.mean(np.abs(autocorrelations) > 0.3) > 0.25

    @pytest.mark.stats
    def test_native_random_seasonal_pool_shows_seasonal_lag(self) -> None:
        generator = MARGenerator(
            **{**BASE, "seed": 11, "min_length": 512, "max_length": 512},
            seasonal_period=12,
        )
        values = series_values(generator.generate(n_series=48))
        lag12 = np.array([sample_acf(series, 12) for series in values.values()])
        assert np.mean(np.abs(lag12) > 0.2) > 0.2

    @pytest.mark.stats
    @pytest.mark.parametrize("native", [True, False])
    def test_two_component_mixture_is_bimodal_in_both_paths(self, native: bool) -> None:
        generator = MARGenerator(
            min_length=20_000,
            max_length=20_000,
            freq="D",
            seed=5,
            weights=[0.5, 0.5],
            ar_coefficients=[[0.2], [0.1, 0.05, -0.05]],
            intercepts=[-6.0, 6.0],
            noise_scales=[0.3, 0.3],
            standardize=False,
        )
        if native:
            values = next(iter(series_values(generator.generate(n_series=1)).values()))
        else:
            values = generator.generate_single_series(20_000)
        assert values.mean() == pytest.approx(0.0, abs=0.3)
        negative = np.mean(values < -3.0)
        positive = np.mean(values > 3.0)
        assert 0.35 < negative < 0.65
        assert 0.35 < positive < 0.65
        assert np.mean(np.abs(values) < 3.0) < 0.1

    @pytest.mark.parametrize("native", [True, False])
    def test_fixed_mode_raises_instead_of_falling_back(self, native: bool) -> None:
        generator = MARGenerator(
            **BASE,
            weights=[1.0],
            ar_coefficients=[[0.5]],
            intercepts=[1e9],
            noise_scales=[1.0],
            standardize=False,
        )
        with pytest.raises(ValueError, match="fixed"):
            if native:
                generator.generate(n_series=1)
            else:
                generator.generate_single_series(64)

    @pytest.mark.stats
    def test_fixed_ar1_reproduces_acf(self) -> None:
        generator = MARGenerator(
            min_length=8000,
            max_length=8000,
            freq="D",
            seed=8,
            weights=[1.0],
            ar_coefficients=[[0.8]],
            intercepts=[0.0],
            noise_scales=[1.0],
        )
        assert_acf(generator.generate_single_series(8000), 1, 0.8)

    @pytest.mark.stats
    def test_native_fixed_ar1_reproduces_raw_moments_and_acf(self) -> None:
        generator = MARGenerator(
            min_length=16_000,
            max_length=16_000,
            freq="D",
            seed=8,
            weights=[1.0],
            ar_coefficients=[[0.6]],
            intercepts=[2.0],
            noise_scales=[0.8],
            standardize=False,
        )
        values = next(iter(series_values(generator.generate(n_series=1)).values()))
        # AR(1) has mean c / (1 - phi) and standard deviation
        # sigma / sqrt(1 - phi**2), both equal to 5 and 1 here.
        assert values.mean() == pytest.approx(5.0, abs=0.1)
        assert values.std() == pytest.approx(1.0, abs=0.05)
        assert_acf(values, 1, 0.6)

    @pytest.mark.stats
    def test_native_fixed_seasonal_ar_reproduces_lag_acf(self) -> None:
        generator = MARGenerator(
            min_length=16_000,
            max_length=16_000,
            freq="D",
            seed=9,
            weights=[1.0],
            ar_coefficients=[[0.0] * 11 + [0.75]],
            intercepts=[0.0],
            noise_scales=[1.0],
        )
        values = next(iter(series_values(generator.generate(n_series=1)).values()))
        assert_acf(values, 12, 0.75)


class TestMarValidation:
    """Validation for random and fixed modes."""

    def test_fixed_fields_are_all_or_nothing(self) -> None:
        with pytest.raises(ValueError, match="provided together"):
            MARGenerator(**BASE, weights=[1.0])

    def test_fixed_component_counts_must_match(self) -> None:
        with pytest.raises(ValueError, match="component counts"):
            MARGenerator(
                **BASE,
                weights=[0.5, 0.5],
                ar_coefficients=[[0.5]],
                intercepts=[0.0, 0.0],
                noise_scales=[1.0, 1.0],
            )

    def test_nonstationary_component_rejected(self) -> None:
        with pytest.raises(ValueError, match="stationary"):
            MARGenerator(
                **BASE,
                weights=[1.0],
                ar_coefficients=[[1.2]],
                intercepts=[0.0],
                noise_scales=[1.0],
            )

    @pytest.mark.parametrize("period", [0, 1])
    def test_short_seasonal_period_rejected(self, period: int) -> None:
        with pytest.raises(ValueError, match="seasonal_period"):
            MARGenerator(**BASE, seasonal_period=period)

    @pytest.mark.parametrize(
        "noise_range", [(0.0, 1.0), (2.0, 1.0), (0.1, float("inf"))]
    )
    def test_invalid_noise_range_rejected(
        self, noise_range: tuple[float, float]
    ) -> None:
        with pytest.raises(ValueError, match="noise_scale_range"):
            MARGenerator(**BASE, noise_scale_range=noise_range)

    @pytest.mark.parametrize("field", ["intercept_scale", "weights_concentration"])
    def test_non_finite_scalars_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            MARGenerator(**BASE, **{field: float("inf")})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_components", 65),
            ("max_ar_order", 101),
            ("burn_in", 1_000_001),
            ("seasonal_period", 10_001),
        ],
    )
    def test_oversized_parameters_rejected(self, field: str, value: int) -> None:
        with pytest.raises(ValueError, match=field):
            MARGenerator(**BASE, **{field: value})

    def test_non_stationary_mixture_rejected(self) -> None:
        with pytest.raises(ValueError, match="second-order stationary"):
            MARGenerator(
                **BASE,
                weights=[0.5, 0.5],
                ar_coefficients=[[-1.9, -0.95], [1.9, -0.95]],
                intercepts=[0.0, 0.0],
                noise_scales=[1.0, 1.0],
            )

    def test_stationary_mixture_of_stationary_components_accepted(self) -> None:
        generator = MARGenerator(
            **BASE,
            weights=[0.3, 0.7],
            ar_coefficients=[[0.9], [0.5, -0.3]],
            intercepts=[0.0, 1.0],
            noise_scales=[1.0, 0.5],
        )
        assert generator.weights == pytest.approx([0.3, 0.7])

    def test_overflowing_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite sum"):
            MARGenerator(
                **BASE,
                weights=[1e308, 1e308],
                ar_coefficients=[[0.5], [0.5]],
                intercepts=[0.0, 0.0],
                noise_scales=[1.0, 1.0],
            )


class TestMarFeatureTargeting:
    """Feature-targeted evolutionary search behavior."""

    @pytest.mark.parametrize("tolerance", [1e-15, 2.0])
    def test_search_reports_convergence_and_budget(self, tolerance: float) -> None:
        generator = MARGenerator.tune_to_features(
            {"acf1": 0.7},
            64,
            96,
            "D",
            n_generations=2,
            population_size=5,
            n_draws_per_candidate=2,
            seed=9,
            tolerance=tolerance,
        )
        diagnostics = generator.tuning_diagnostics
        assert diagnostics is not None
        assert np.isfinite(diagnostics["best_distance"])
        assert diagnostics["converged"] is (tolerance == 2.0)
        assert diagnostics["generations_run"] == (1 if tolerance == 2.0 else 2)
        assert diagnostics["candidates_evaluated"] == 5 * diagnostics["generations_run"]
        diagnostics["best_distance"] = -1
        assert generator.tuning_diagnostics["best_distance"] >= 0
        assert MARGenerator(**BASE).tuning_diagnostics is None

    def test_unknown_feature_rejected(self) -> None:
        with pytest.raises(ValueError, match="supported names"):
            MARGenerator.tune_to_features(
                {"unknown": 0.5}, 64, 64, "D", n_generations=1, population_size=1
            )

    def test_seasonal_target_requires_period(self) -> None:
        with pytest.raises(ValueError, match="requires seasonal_period"):
            MARGenerator.tune_to_features(
                {"seasonal_strength": 0.8},
                64,
                64,
                "D",
                n_generations=1,
                population_size=1,
            )

    @pytest.mark.parametrize(
        "target", [{"acf1": -1.1}, {"acf1": 1.1}, {"spectral_entropy": 1.1}]
    )
    def test_out_of_range_target_rejected(self, target: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="target"):
            MARGenerator.tune_to_features(
                target, 64, 64, "D", n_generations=1, population_size=1
            )

    @pytest.mark.parametrize(
        ("min_length", "max_length", "message"),
        [(2, 64, "min_length"), (65, 64, "max_length")],
    )
    def test_invalid_target_length_range_rejected(
        self, min_length: int, max_length: int, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            MARGenerator.tune_to_features(
                {"acf1": 0.5},
                min_length,
                max_length,
                "D",
                n_generations=1,
                population_size=1,
            )

    def test_seasonal_target_requires_two_periods_at_min_length(self) -> None:
        with pytest.raises(ValueError, match="2 \\* seasonal_period"):
            MARGenerator.tune_to_features(
                {"seasonal_strength": 0.5},
                23,
                48,
                "D",
                seasonal_period=12,
                n_generations=1,
                population_size=1,
            )

    def test_targeting_rejects_oversized_seasonal_period_before_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def search_started(*_args: object, **_kwargs: object) -> None:
            pytest.fail("candidate search started before validation completed")

        monkeypatch.setattr(MARGenerator, "_random_candidate", search_started)
        with pytest.raises(ValueError, match="seasonal_period must be in"):
            MARGenerator.tune_to_features(
                {"acf1": 0.5},
                64,
                64,
                "D",
                seasonal_period=10_001,
                n_generations=1,
                population_size=1,
            )

    def test_all_invalid_candidates_report_search_exhaustion(self) -> None:
        with pytest.raises(ValueError, match="no valid MAR candidate"):
            MARGenerator.tune_to_features(
                {"acf1": 0.5},
                64,
                64,
                "D",
                n_generations=1,
                population_size=1,
                n_draws_per_candidate=1,
                seed=5,
            )

    def test_candidate_draw_lengths_span_configured_range(self) -> None:
        np.testing.assert_array_equal(
            MARGenerator._evaluation_lengths(64, 96, 3), [64, 80, 96]
        )
        np.testing.assert_array_equal(MARGenerator._evaluation_lengths(64, 96, 1), [80])

    def test_search_is_deterministic(self) -> None:
        arguments = {
            "target_features": {"acf1": 0.7},
            "min_length": 64,
            "max_length": 96,
            "freq": "D",
            "n_generations": 2,
            "population_size": 5,
            "n_draws_per_candidate": 2,
            "seed": 9,
        }
        first = MARGenerator.tune_to_features(**arguments)
        second = MARGenerator.tune_to_features(**arguments)
        assert first.weights == second.weights
        assert first.ar_coefficients == second.ar_coefficients
        assert first.intercepts == second.intercepts
        assert first.noise_scales == second.noise_scales

    @pytest.mark.parametrize(
        ("feature", "target", "seasonal_period", "seed"),
        [
            ("spectral_entropy", 0.8, None, 31),
            ("trend_strength", 0.6, None, 32),
            ("seasonal_strength", 0.7, 12, 33),
            ("acf1", 0.7, None, 34),
        ],
    )
    @pytest.mark.stats
    @pytest.mark.slow
    def test_search_targets_each_feature_across_length_range(
        self,
        feature: str,
        target: float,
        seasonal_period: int | None,
        seed: int,
    ) -> None:
        generator = MARGenerator.tune_to_features(
            {feature: target},
            min_length=64,
            max_length=128,
            freq="D",
            seasonal_period=seasonal_period,
            n_generations=5,
            population_size=10,
            n_draws_per_candidate=3,
            seed=seed,
        )
        realized = np.mean(
            [
                compute_features(
                    generator.generate_single_series(length), seasonal_period
                )[feature]
                for length in (64, 96, 128)
                for _ in range(8)
            ]
        )
        assert abs(realized - target) < 0.2
