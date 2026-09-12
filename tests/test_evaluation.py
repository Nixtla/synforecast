"""Coverage arithmetic, reference stability, retention, and panel contracts."""

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest

from synforecast._coverage import grid_occupancy, occupancy_scores
from synforecast._features import FEATURE_NAMES
from synforecast._features import compute_features as targeting_features
from synforecast.evaluation import (
    compare_feature_coverage,
    compute_features,
    feature_coverage,
)


def frame(engine, data):
    return pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)


def feature_frame(engine, x=(-2.0, 0.0, 2.0), **extra):
    return frame(
        engine, {"unique_id": [f"id{i}" for i in range(len(x))], "x": list(x), **extra}
    )


def panel(engine):
    rng = np.random.default_rng(12)
    return frame(
        engine,
        {
            "unique_id": np.repeat(["b", "a", "c"], 48),
            "ds": np.tile(np.arange(48)[::-1], 3),
            "y": rng.normal(size=144),
        },
    )


def test_grid_denominators_and_boundaries():
    edges = (np.linspace(0, 3, 4), np.linspace(0, 3, 4))
    real, cells, inside = grid_occupancy(
        np.array([[0, 0], [3, 3], [1, 2], [4, 1], [-1, 0]]), edges
    )
    np.testing.assert_array_equal(inside, [True, True, True, False, False])
    np.testing.assert_array_equal(cells[:3], [[0, 0], [2, 2], [1, 2]])
    syn, _, _ = grid_occupancy(np.array([[0.1, 0.1], [1.2, 0.2]]), edges)
    assert occupancy_scores(real, syn) == pytest.approx((2 / 9, 1 / 9, 2 / 3))
    assert occupancy_scores(real, np.zeros_like(real)) == pytest.approx((3 / 9, 0, 1))
    assert occupancy_scores(real, real) == (0, 0, 0)


def test_identity_rank_one_and_id_mapping(engine):
    real = feature_frame(engine)
    result = feature_coverage(real, real, precomputed=True)
    assert result.miscoverage == result.reverse_miscoverage == 0
    assert result.uncovered_real_series_fraction == 0
    assert result.real_diagnostics.retained_ids == ("id0", "id1", "id2")
    assert result.explained_variance == (1, 0)
    assert result.n_synthetic_out_of_range == 0
    np.testing.assert_array_equal(result.real_embedding[:, 1], 0)
    assert np.all(np.diff(result.grid_edges[1]) > 0)


def test_reference_space_stability_and_constant_deviations(engine):
    real = feature_frame(engine, constant=[1.0] * 3)
    synthetic = feature_frame(engine, (-1.0, 0.0, 1.0), constant=[1.0, 2.0, 1.0])
    first = compare_feature_coverage(real, {"first": synthetic}, precomputed=True)[
        "first"
    ]
    later = compare_feature_coverage(
        real,
        {"first": synthetic, "extreme": feature_frame(engine, (1e6,), constant=[50.0])},
        precomputed=True,
    )["first"]
    assert first.space_id == later.space_id
    assert first.metadata["reference_id"] == later.metadata["reference_id"]
    np.testing.assert_array_equal(first.real_embedding, later.real_embedding)
    np.testing.assert_array_equal(first.synthetic_embedding, later.synthetic_embedding)
    np.testing.assert_array_equal(first.real_occupancy, later.real_occupancy)
    assert first.miscoverage == later.miscoverage
    assert first.features_used == ("x",)
    assert first.constant_features_dropped == ("constant",)
    assert first.synthetic_constant_feature_deviations == {"constant": 1}


def test_out_of_range_and_pooled_grid(engine):
    real = feature_frame(engine)
    matching = feature_coverage(real, real, precomputed=True, n_bins=4)
    extra = feature_frame(engine, (-2.0, 0.0, 2.0, 1000.0))
    result = feature_coverage(real, extra, precomputed=True, n_bins=4)
    assert result.space_id == matching.space_id
    assert result.miscoverage == result.reverse_miscoverage == 0
    assert result.n_synthetic_out_of_range == 1
    assert result.synthetic_out_of_range_fraction == 0.25
    pooled = feature_coverage(
        real, extra, precomputed=True, grid_range="pooled", n_bins=4
    )
    assert pooled.space_id != result.space_id
    assert pooled.grid_edges[0][-1] > result.grid_edges[0][-1]
    assert pooled.n_synthetic_out_of_range == 0
    all_out = feature_coverage(
        real, feature_frame(engine, (100.0, 200.0)), precomputed=True
    )
    assert (
        all_out.uncovered_real_cell_fraction
        == all_out.uncovered_real_series_fraction
        == 1
    )
    assert all_out.reverse_miscoverage == 0
    assert all_out.synthetic_out_of_range_fraction == 1


def test_real_series_fraction_weights_series_not_cells(engine):
    real = feature_frame(engine, (-1.0, -1.0, -1.0, 1.0))
    synthetic = feature_frame(engine, (1.0,))
    result = feature_coverage(real, synthetic, precomputed=True, n_bins=2)
    assert result.uncovered_real_cell_fraction == 0.5
    assert result.uncovered_real_series_fraction == 0.75
    assert result.uncovered_real_ids == ("id0", "id1", "id2")


def test_precomputed_column_alignment_and_selection(engine):
    real = feature_frame(engine, y=[0.0, 2.0, 1.0])
    swapped = frame(
        engine,
        {
            "y": [0.0, 2.0, 1.0],
            "unique_id": ["id0", "id1", "id2"],
            "x": [-2.0, 0.0, 2.0],
        },
    )
    assert feature_coverage(real, swapped, precomputed=True).miscoverage == 0
    with pytest.raises(ValueError, match="columns must match"):
        feature_coverage(real, feature_frame(engine), precomputed=True)
    assert (
        feature_coverage(
            real, feature_frame(engine), precomputed=True, features=["x"]
        ).miscoverage
        == 0
    )


def test_missing_diagnostics_and_reference_retention(engine, caplog):
    real = feature_frame(engine, (-2.0, np.nan, 2.0))
    synthetic = feature_frame(engine, (-2.0, np.inf, 2.0))
    with pytest.raises(ValueError, match="non-finite"):
        feature_coverage(real, synthetic, precomputed=True)
    with caplog.at_level(logging.WARNING, logger="synforecast.evaluation"):
        result = feature_coverage(real, synthetic, precomputed=True, missing="drop")
    assert len(caplog.records) == 2
    assert result.real_diagnostics.n_input == 3
    assert result.real_diagnostics.retained_ids == ("id0", "id2")
    assert result.real_diagnostics.dropped_ids == ("id1",)
    assert result.synthetic_diagnostics.drop_reasons == {"id1": ("x",)}
    assert result.miscoverage == 0
    with pytest.raises(ValueError, match="no retained"):
        feature_coverage(
            real, feature_frame(engine, (np.nan,)), precomputed=True, missing="drop"
        )


@pytest.mark.parametrize("x", [(1.0,), (1.0, 1.0)])
def test_fitting_population_requirements(engine, x):
    with pytest.raises(ValueError, match="at least two|No variable"):
        feature_coverage(
            feature_frame(engine, x), feature_frame(engine), precomputed=True
        )
    result = feature_coverage(
        feature_frame(engine, x), feature_frame(engine), precomputed=True, fit="pooled"
    )
    assert result.fit == "pooled"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bins": 0},
        {"n_bins": True},
        {"features": []},
        {"features": "x"},
        {"features": ["x", "x"]},
        {"features": ["missing"]},
        {"fit": "invalid"},
        {"embedding": "invalid"},
        {"grid_range": "invalid"},
        {"missing": "impute"},
        {"n_jobs": 0},
        {"n_jobs": True},
        {"embedding": "tsne"},
        {"embedding": "tsne", "fit": "pooled"},
    ],
)
def test_invalid_options(kwargs):
    real = feature_frame("pandas")
    with pytest.raises(ValueError):
        feature_coverage(real, real, precomputed=True, **kwargs)


def test_precomputed_invalid_ids_and_dtypes(engine):
    real = feature_frame(engine)
    for bad in (
        frame(engine, {"unique_id": ["x", "x"], "x": [1.0, 2.0]}),
        frame(engine, {"unique_id": ["x", None], "x": [1.0, 2.0]}),
        frame(engine, {"unique_id": ["x"], "x": ["text"]}),
    ):
        with pytest.raises(ValueError):
            feature_coverage(real, bad, precomputed=True)


def test_native_panel_precomputed_and_workers(engine):
    real = panel(engine)
    extracted = compute_features(real, seasonal_period=4)
    assert type(extracted) is type(real)
    assert list(extracted.columns) == ["unique_id", *FEATURE_NAMES]
    expected = feature_coverage(extracted, extracted, precomputed=True)
    actual = feature_coverage(real, real, seasonal_period=4, n_jobs=2)
    np.testing.assert_allclose(actual.real_embedding, expected.real_embedding)
    assert actual.miscoverage == 0
    assert actual.metadata["parameters"]["schema"] == "native_v1"
    assert actual.metadata["parameters"]["effective_window_size"] == 4
    native = (
        extracted.to_dict(as_series=False)
        if engine == "polars"
        else extracted.to_dict("list")
    )
    # Independent call to the existing MAR helper verifies its four-value contract.
    if engine == "polars":
        values = real.filter(pl.col("unique_id") == "a").sort("ds")["y"].to_numpy()
    else:
        values = real[real.unique_id == "a"].sort_values("ds").y.to_numpy()
    for name, value in targeting_features(values, 4).items():
        assert native[name][0] == pytest.approx(value)


def test_custom_columns_and_short_series(engine, caplog):
    df = frame(
        engine, {"series": ["b", "a", "a"], "time": [0, 1, 0], "value": [0.0, 1.0, 2.0]}
    )
    with caplog.at_level(logging.WARNING, logger="synforecast.evaluation"):
        result = compute_features(
            df, id_col="series", time_col="time", target_col="value", features=["acf1"]
        )
    assert len(result) == 2
    assert list(result.columns) == ["series", "acf1"]
    assert len(caplog.records) == 1
    assert "2 series" in caplog.records[0].message


@pytest.mark.parametrize(
    "data",
    [
        {"unique_id": ["a", "a"], "ds": [1, 1], "y": [1.0, 2.0]},
        {"unique_id": [None], "ds": [1], "y": [1.0]},
        {"unique_id": ["a"], "ds": [None], "y": [1.0]},
        {"unique_id": ["a"], "ds": [1], "y": [np.inf]},
        {"unique_id": ["a"], "ds": [1], "y": ["invalid"]},
    ],
)
def test_invalid_raw_panels(engine, data):
    with pytest.raises(ValueError):
        compute_features(frame(engine, data))


def test_pandas_polars_agreement():
    pandas = feature_coverage(panel("pandas"), panel("pandas"), seasonal_period=4)
    polars = feature_coverage(panel("polars"), panel("polars"), seasonal_period=4)
    np.testing.assert_array_equal(pandas.real_embedding, polars.real_embedding)
    assert pandas.space_id == polars.space_id


@pytest.mark.parametrize("column", ["unique_id", "ds"])
def test_nonfinite_numeric_panel_identifiers(engine, column):
    data = {"unique_id": [0.0, 0.0, 0.0], "ds": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]}
    data[column][0] = np.nan
    with pytest.raises(ValueError):
        compute_features(frame(engine, data))


def test_nonfinite_numeric_feature_identifiers(engine):
    real = feature_frame(engine)
    invalid = frame(engine, {"unique_id": [np.nan], "x": [1.0]})
    with pytest.raises(ValueError):
        feature_coverage(real, invalid, precomputed=True)


def test_tsne_seed_and_small_population():
    pytest.importorskip("sklearn")
    real = feature_frame("pandas")
    with pytest.raises(ValueError, match="more than 30"):
        feature_coverage(
            real, real, precomputed=True, embedding="tsne", fit="pooled", seed=4
        )
    rng = np.random.default_rng(4)
    real = feature_frame("pandas", rng.normal(size=20), y=rng.normal(size=20).tolist())
    kwargs = {"precomputed": True, "embedding": "tsne", "fit": "pooled", "seed": 4}
    first = feature_coverage(real, real, **kwargs)
    second = feature_coverage(real, real, **kwargs)
    np.testing.assert_array_equal(first.real_embedding, second.real_embedding)
    assert first.space_id == second.space_id


def test_plot_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = feature_frame("pandas")
    result = feature_coverage(real, feature_frame("pandas", (100.0,)), precomputed=True)
    ax = result.plot()
    assert "outside grid: 1" in ax.get_title()
    plt.close(ax.figure)


def test_window_configuration_is_validated_recorded_and_forwarded(engine):
    real = panel(engine)
    result = feature_coverage(real, real, window_size=5)
    assert result.metadata["parameters"]["window_size"] == 5
    assert result.miscoverage == 0
    default = feature_coverage(real, real)
    assert result.space_id != default.space_id
    raw = compute_features(real, window_size=5)
    cached = feature_coverage(raw, raw, precomputed=True)
    np.testing.assert_allclose(result.real_embedding, cached.real_embedding)
    for invalid in (0, 1, True, 2.5):
        with pytest.raises(ValueError, match="window_size"):
            compute_features(real, window_size=invalid)
