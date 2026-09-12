"""Independent numerical oracles for the native coverage candidates."""

import numpy as np
import pytest

from synforecast._features import FEATURE_NAMES, compute_feature_set, compute_features
from tests.test_features import _moving_average


def oracle(values, period):
    """Direct calculations favor clarity over the Rust kernel's rolling updates."""
    x = (
        (values - values.mean()) / values.std()
        if values.std()
        else np.zeros_like(values)
    )

    def acf(a, lag):
        centered = a - a.mean()
        return (
            float(centered[:-lag] @ centered[lag:] / (centered @ centered))
            if np.var(a)
            else 0.0
        )

    trend = _moving_average(x, period)
    detrended = x - trend
    phases = np.array([detrended[i::period].mean() for i in range(period)])
    remainder = detrended - np.resize(phases - phases.mean(), len(x))
    windows = np.lib.stride_tricks.sliding_window_view(x, period)
    return {
        "x_acf10": sum(acf(x, lag) ** 2 for lag in range(1, 11)),
        "diff1_acf1": acf(np.diff(x), 1),
        "seas_acf1": acf(x, period),
        "spike": np.var([np.var(np.delete(remainder, i)) for i in range(len(x))]),
        "lumpiness": np.var(
            [np.var(x[i : i + period]) for i in range(0, len(x) - period + 1, period)]
        ),
        "max_level_shift": np.max(
            np.abs(windows.mean(axis=1)[period:] - windows.mean(axis=1)[:-period])
        ),
        "max_var_shift": np.max(
            np.abs(windows.var(axis=1)[period:] - windows.var(axis=1)[:-period])
        ),
        "crossing_points": np.count_nonzero(np.diff(x <= np.median(x))),
    }


@pytest.mark.parametrize("period", [2, 3, 4, 12])
@pytest.mark.parametrize("length", [30, 49, 103])
def test_against_direct_oracles(period, length):
    values = np.random.default_rng(length).normal(size=length)
    actual = compute_feature_set(values, period)
    assert tuple(actual) == FEATURE_NAMES
    for name, expected in oracle(values, period).items():
        assert actual[name] == pytest.approx(expected, abs=1e-12), name
    for name, expected in compute_features(values, period).items():
        assert actual[name] == expected


@pytest.mark.parametrize("scale,offset", [(3.0, 0.0), (1.0, 100.0), (0.01, -10.0)])
def test_new_features_affine_invariance(scale, offset):
    values = np.random.default_rng(54).normal(size=100)
    expected = compute_feature_set(values, 4)
    actual = compute_feature_set(scale * values + offset, 4)
    for name in FEATURE_NAMES[4:]:
        assert actual[name] == pytest.approx(expected[name], rel=1e-8, abs=1e-10)


def test_constant_and_short_contracts():
    actual = compute_feature_set(np.ones(100), 12)
    assert all(value == 0.0 for value in actual.values())
    assert all(np.isnan(x) for x in compute_feature_set(np.ones(2), None).values())
    short = compute_feature_set(np.arange(10.0), 12)
    assert np.isnan(short["seasonal_strength"])
    assert np.isnan(short["x_acf10"])
    assert np.isnan(short["seas_acf1"])
    assert np.isnan(short["lumpiness"])
    assert np.isnan(short["max_level_shift"])
    # Evaluation marks unavailable cycles; MAR's existing fallback is unchanged.
    assert compute_features(np.arange(10.0), 12)["seasonal_strength"] == 0
    without_period = compute_feature_set(np.arange(20.0), None)
    assert without_period["seas_acf1"] == 0
    assert np.isfinite(without_period["lumpiness"])
    assert np.isnan(compute_feature_set(np.arange(19.0), None)["lumpiness"])


@pytest.mark.parametrize(
    "values", [np.array([]), np.array([1.0, np.inf]), np.ones((3, 3))]
)
def test_invalid_values(values):
    with pytest.raises(ValueError):
        compute_feature_set(values, None)


def test_step_and_variance_change():
    step = np.r_[np.zeros(20), np.ones(20)]
    result = compute_feature_set(step, None)
    assert result["max_level_shift"] == pytest.approx(2.0)
    assert result["crossing_points"] == 1
    variance_change = np.r_[np.tile([-1.0, 1.0], 10), np.tile([-3.0, 3.0], 10)]
    result = compute_feature_set(variance_change, None)
    assert result["max_var_shift"] == pytest.approx(1.6)
    assert result["lumpiness"] == pytest.approx(0.64)


def test_explicit_window_on_short_nonseasonal_series():
    values = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 4.0, 6.0, 4.0, 2.0, 0.0, 1.0, 2.0])
    default = compute_feature_set(values, None)
    actual = compute_feature_set(values, None, window_size=5)
    x = (values - values.mean()) / values.std()
    windows = np.lib.stride_tricks.sliding_window_view(x, 5)
    assert np.isnan(default["lumpiness"])
    assert actual["lumpiness"] == pytest.approx(
        np.var([np.var(x[:5]), np.var(x[5:10])])
    )
    assert actual["max_level_shift"] == pytest.approx(
        np.max(np.abs(windows.mean(axis=1)[5:] - windows.mean(axis=1)[:-5]))
    )
    assert actual["max_var_shift"] == pytest.approx(
        np.max(np.abs(windows.var(axis=1)[5:] - windows.var(axis=1)[:-5]))
    )
    for name in set(FEATURE_NAMES) - {"lumpiness", "max_level_shift", "max_var_shift"}:
        assert actual[name] == default[name]
