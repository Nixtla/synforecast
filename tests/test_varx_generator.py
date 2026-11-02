"""Tests for VARXGenerator."""

import numpy as np
import pytest

from synforecast.generators import VARXGenerator
from synforecast.generators.varx import VARXTruth
from tests.helpers import assert_long_format, assert_std, to_pandas

EXOG = ["exog_shared_0", "exog_future_0", "exog_future_1", "exog_hist_0", "group"]


def make_gen(engine: str = "pandas", **kwargs) -> VARXGenerator:
    params = {
        "min_length": 200,
        "max_length": 200,
        "freq": "D",
        "engine": engine,
        "seed": 7,
        "n_shared_exog": 1,
        "n_future_exog": 2,
        "n_hist_exog": 1,
        "exog_cross_scale": 0.5,
        "n_groups": 2,
    }
    params.update(kwargs)
    return VARXGenerator(**params)


def wide(df, n_series: int) -> dict[str, np.ndarray]:
    """{column: (T, N)} in series and time order."""
    pdf = to_pandas(df).sort_values(["unique_id", "ds"])
    return {
        col: pdf[col].to_numpy().reshape(n_series, -1).T
        for col in ["y", *(c for c in EXOG if c in pdf)]
    }


def residuals(truth: VARXTruth, cols: dict[str, np.ndarray]) -> np.ndarray:
    """``y`` minus everything the truth explains, from the second period on."""
    y = cols["y"]
    future = np.stack([cols["exog_future_0"], cols["exog_future_1"]], axis=2)
    fitted = (
        truth.intercept
        + y[:-1] @ truth.lag_matrices[0].T
        + cols["exog_shared_0"][1:, :1] @ truth.shared_effects.T
        + np.einsum("kij,tjk->ti", truth.future_effects, future[1:])
        + cols["exog_hist_0"][:-1] @ truth.hist_effects[0].T
    )
    return y[1:] - fitted


class TestVARXAPI:
    def test_long_format_with_regressors(self, engine: str) -> None:
        df = make_gen(engine).generate(n_series=4)
        assert_long_format(
            df, n_series=4, min_length=200, max_length=200, extra_cols=set(EXOG)
        )

    def test_seed_determinism(self) -> None:
        a = to_pandas(make_gen().generate(n_series=3))
        b = to_pandas(make_gen().generate(n_series=3))
        assert a.equals(b)

    def test_truth_shapes_and_groups(self) -> None:
        gen = make_gen()
        cols = wide(gen.generate(n_series=6), 6)
        truth = gen.truth
        assert truth.lag_matrices.shape == (1, 6, 6)
        assert truth.shared_effects.shape == (6, 1)
        assert truth.future_effects.shape == (2, 6, 6)
        assert truth.hist_effects.shape == (1, 6, 6)
        np.testing.assert_array_equal(truth.groups, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(cols["group"][0], truth.groups)
        shared = cols["exog_shared_0"]
        np.testing.assert_array_equal(shared, shared[:, :1].repeat(6, axis=1))

    def test_truth_needs_a_draw(self) -> None:
        with pytest.raises(ValueError, match="generate first"):
            _ = make_gen().truth

    def test_process_follows_the_truth_exactly_without_noise(self) -> None:
        gen = make_gen(innovation_covariance=(1e-24 * np.eye(5)).tolist())
        cols = wide(gen.generate(n_series=5), 5)
        np.testing.assert_allclose(residuals(gen.truth, cols), 0.0, atol=1e-9)

    def test_user_coefficients_are_kept(self) -> None:
        a = [[[0.5, 0.2], [0.0, 0.4]]]
        gen = make_gen(coef_matrices=a, cross_structure="sparse", n_groups=None)
        gen.generate(n_series=2)
        np.testing.assert_array_equal(gen.truth.lag_matrices, a)
        assert gen.truth.groups is None

    def test_own_effects_only_by_default(self) -> None:
        gen = make_gen(exog_cross_scale=0.0)
        gen.generate(n_series=4)
        effects = gen.truth.future_effects[0]
        np.testing.assert_array_equal(effects, np.diag(np.diag(effects)))

    def test_group_structure(self) -> None:
        gen = make_gen(cross_structure="groups", n_groups=3)
        gen.generate(n_series=9)
        a, groups = gen.truth.lag_matrices[0], gen.truth.groups
        for i in range(9):
            for g in range(3):
                others = [j for j in np.flatnonzero(groups == g) if j != i]
                np.testing.assert_allclose(a[i, others], a[i, others[0]])

    def test_sparse_structure(self) -> None:
        gen = make_gen(cross_structure="sparse", sparsity=0.2, n_groups=None)
        gen.generate(n_series=40)
        a = gen.truth.lag_matrices[0]
        off = a[~np.eye(40, dtype=bool)]
        assert 0.1 < np.mean(off != 0) < 0.3
        assert np.all(np.diag(a) > 0)

    @pytest.mark.parametrize("structure", ["dense", "sparse", "groups"])
    def test_lag_two_processes_stay_stable(self, structure: str) -> None:
        gen = make_gen(lag_order=2, cross_structure=structure, n_groups=4)
        values = wide(gen.generate(n_series=40), 40)["y"]
        assert np.isfinite(values).all() and np.abs(values).max() < 1e3

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="requires n_groups"):
            make_gen(cross_structure="groups", n_groups=None)
        with pytest.raises(ValueError, match="exceeds n_series"):
            make_gen(n_groups=5).generate(n_series=3)

    def test_generate_single_series(self) -> None:
        values = make_gen(n_groups=None).generate_single_series(50)
        assert values.shape == (50,) and np.isfinite(values).all()


@pytest.mark.stats
class TestVARXStats:
    def test_residuals_are_the_innovations(self) -> None:
        gen = make_gen(min_length=3000, max_length=3000)
        cols = wide(gen.generate(n_series=4), 4)
        for e in residuals(gen.truth, cols).T:
            assert_std(e, expected=1.0)

    def test_regressors_are_unit_variance_ar1(self) -> None:
        gen = make_gen(min_length=5000, max_length=5000, exog_persistence=0.6)
        x = wide(gen.generate(n_series=2), 2)["exog_future_0"][:, 0]
        assert_std(x, expected=1.0)
        assert abs(np.corrcoef(x[1:], x[:-1])[0, 1] - 0.6) < 0.05

    def test_oracle_one_step_error_is_the_innovation(self) -> None:
        gen = make_gen(min_length=600, max_length=600)
        cols = wide(gen.generate(n_series=3), 3)
        truth = gen.truth
        errors = []
        for t in range(100, 600):
            future = np.stack([cols["exog_future_0"], cols["exog_future_1"]], axis=2)
            forecast = truth.forecast(
                cols["y"][:t],
                cols["exog_shared_0"][t : t + 1, :1],
                future[t : t + 1],
                cols["exog_hist_0"][t - 1][:, None],
                1,
            )
            errors.append(cols["y"][t] - forecast[0])
        assert_std(np.concatenate(errors), expected=1.0)
