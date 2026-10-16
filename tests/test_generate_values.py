"""generate_values returns exactly the series generate() draws, on native and threaded generators alike."""

import numpy as np
import pytest

from synforecast.generators.sarima import SARIMAGenerator
from synforecast.presets import pretraining_pool

POOL = pretraining_pool(
    min_length=40, max_length=90, freq="D", seed=3, include_seasonal=True
)


@pytest.mark.parametrize("member", range(len(POOL)))
def test_values_match_generate(member: int) -> None:
    kwargs = {
        "min_length": 40,
        "max_length": 90,
        "freq": "D",
        "seed": 3,
        "include_seasonal": True,
        "members": [member],
    }
    df = pretraining_pool(**kwargs)[0].generate(n_series=3, n_jobs=1)
    values = pretraining_pool(**kwargs)[0].generate_values(n_series=3, n_jobs=1)
    assert len(values) == 3
    for i, v in enumerate(values):
        expected = df.loc[
            df["unique_id"] == df["unique_id"].unique()[i], "y"
        ].to_numpy()
        np.testing.assert_array_equal(v, expected)


def test_values_do_not_depend_on_workers() -> None:
    one = pretraining_pool(
        min_length=64, max_length=64, freq="h", seed=8, members=[45]
    )[0].generate_values(6, n_jobs=1)
    many = pretraining_pool(
        min_length=64, max_length=64, freq="h", seed=8, members=[45]
    )[0].generate_values(6, n_jobs=4)
    for a, b in zip(one, many, strict=True):
        np.testing.assert_array_equal(a, b)


def test_overlays_and_missingness_are_kept() -> None:
    kwargs = {
        "min_length": 60,
        "max_length": 60,
        "freq": "D",
        "seed": 5,
        "missing_data": True,
        "missing_rate": 0.3,
    }
    values = SARIMAGenerator(**kwargs).generate_values(4, n_jobs=1)
    df = SARIMAGenerator(**kwargs).generate(4, n_jobs=1)
    assert all(np.isnan(v).any() for v in values)
    np.testing.assert_array_equal(np.concatenate(values), df["y"].to_numpy())


@pytest.mark.parametrize("n_series", [0, -1])
def test_rejects_nonpositive_n_series(n_series: int) -> None:
    with pytest.raises(ValueError):
        POOL[0].generate_values(n_series)
