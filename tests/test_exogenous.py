"""Tests for calendar feature and diagnostic label columns."""

import numpy as np
import polars as pl
import pytest

from synforecast.dataset import SynSet
from synforecast.exogenous import (
    CalendarFeatures,
    DiagnosticsConfig,
    SeriesMetadata,
    build_flag_column,
    extract_datetime_features,
)
from synforecast.generators import (
    CopulaGenerator,
    DailyActiveUsersGenerator,
    IoTSensorGenerator,
    RandomWalkGenerator,
    VARGenerator,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestColumnConfigs:
    """Tests for CalendarFeatures and DiagnosticsConfig validation."""

    def test_calendar_defaults(self) -> None:
        cfg = CalendarFeatures()
        assert cfg.basic is True
        assert cfg.cyclical is False

    def test_diagnostics_columns(self) -> None:
        cfg = DiagnosticsConfig(anomalies=True, changepoints=True, missing=True)
        assert cfg.columns == ["anomaly_flag", "changepoint_flag", "missing_flag"]
        assert DiagnosticsConfig(missing=True).columns == ["missing_flag"]

    def test_empty_configs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="pass None"):
            CalendarFeatures(basic=False, cyclical=False)
        with pytest.raises(ValueError, match="pass None"):
            DiagnosticsConfig()


# ---------------------------------------------------------------------------
# Core columns only
# ---------------------------------------------------------------------------
class TestNoExtraColumns:
    """Without column configs the output has the three core columns."""

    def test_no_configs_give_three_columns(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "seed": 42,
                "engine": "polars",
            }
        )
        df = gen.generate(n_series=2)
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"unique_id", "ds", "y"}


# ---------------------------------------------------------------------------
# Datetime features
# ---------------------------------------------------------------------------
class TestDatetimeFeatures:
    """Tests for datetime feature extraction."""

    def test_basic_daily_features(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 30,
                "max_length": 30,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "calendar_features": CalendarFeatures(),
            }
        )
        df = gen.generate(n_series=1)
        expected_cols = {
            "unique_id",
            "ds",
            "y",
            "year",
            "quarter",
            "month",
            "day_of_year",
            "day_of_week",
            "day_of_month",
            "is_weekend",
        }
        assert expected_cols.issubset(set(df.columns))
        # daily data should NOT have hour
        assert "hour" not in df.columns

    def test_hourly_features_include_hour(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "h",
                "engine": "polars",
                "seed": 42,
                "calendar_features": CalendarFeatures(),
            }
        )
        df = gen.generate(n_series=1)
        assert "hour" in df.columns

    def test_cyclical_features(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "h",
                "engine": "polars",
                "seed": 42,
                "calendar_features": CalendarFeatures(cyclical=True, basic=False),
            }
        )
        df = gen.generate(n_series=1)
        for col in [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "doy_sin",
            "doy_cos",
        ]:
            assert col in df.columns, f"Missing cyclical column: {col}"

        # Cyclical values should be in [-1, 1]
        for col in ["hour_sin", "hour_cos"]:
            vals = df[col].to_numpy()
            assert np.all(vals >= -1.0 - 1e-6)
            assert np.all(vals <= 1.0 + 1e-6)

    def test_extract_datetime_features_function(self) -> None:
        timestamps = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-02-01"),
            np.timedelta64(1, "D"),
        ).astype("datetime64[ns]")

        features = extract_datetime_features(
            timestamps, freq="D", include_basic=True, include_cyclical=False
        )
        assert "year" in features
        assert np.all(features["year"] == 2020)
        assert features["month"][0] == 1

    def test_cyclical_encoding_range(self) -> None:
        timestamps = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-02"),
            np.timedelta64(1, "h"),
        ).astype("datetime64[ns]")

        features = extract_datetime_features(
            timestamps, freq="h", include_basic=False, include_cyclical=True
        )
        for col in ["hour_sin", "hour_cos"]:
            assert col in features
            assert np.all(features[col] >= -1.0 - 1e-6)
            assert np.all(features[col] <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Flag columns
# ---------------------------------------------------------------------------
class TestFlagColumns:
    """Tests for anomaly, changepoint, and missing flags."""

    def test_anomaly_flags_present(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 200,
                "max_length": 200,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "anomalies": True,
                "anomaly_fraction": 0.1,
                "diagnostics": DiagnosticsConfig(anomalies=True),
            }
        )
        df = gen.generate(n_series=2)
        assert "anomaly_flag" in df.columns
        flags = df["anomaly_flag"].to_numpy()
        # Should have some anomalies flagged
        assert np.sum(flags) > 0
        # Should be binary
        assert set(np.unique(flags)).issubset({0, 1})

    def test_changepoint_flags_present(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 200,
                "max_length": 200,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "changepoints": True,
                "num_changepoints": 3,
                "diagnostics": DiagnosticsConfig(changepoints=True),
            }
        )
        df = gen.generate(n_series=2)
        assert "changepoint_flag" in df.columns
        flags = df["changepoint_flag"].to_numpy()
        assert np.sum(flags) > 0
        assert set(np.unique(flags)).issubset({0, 1})

    def test_missing_flags_present(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 200,
                "max_length": 200,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "missing_data": True,
                "missing_rate": 0.1,
                "diagnostics": DiagnosticsConfig(missing=True),
            }
        )
        df = gen.generate(n_series=2)
        assert "missing_flag" in df.columns
        flags = df["missing_flag"].to_numpy()
        assert np.sum(flags) > 0
        assert set(np.unique(flags)).issubset({0, 1})

    def test_flags_without_injection_are_zero(self) -> None:
        """If anomalies are not enabled, anomaly_flag should be all zeros."""
        gen = RandomWalkGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "anomalies": False,
                "diagnostics": DiagnosticsConfig(anomalies=True),
            }
        )
        df = gen.generate(n_series=1)
        assert "anomaly_flag" in df.columns
        assert df["anomaly_flag"].sum() == 0

    def test_build_flag_column_function(self) -> None:
        meta = SeriesMetadata(
            values=np.zeros(10),
            timestamps=np.zeros(10),
            series_id=0,
            length=10,
            anomaly_indices=np.array([2, 5, 7]),
        )
        flags = build_flag_column([meta], "anomaly")
        assert len(flags) == 10
        assert flags[2] == 1
        assert flags[5] == 1
        assert flags[7] == 1
        assert flags[0] == 0

    def test_build_flag_column_out_of_bounds(self) -> None:
        """Out-of-bounds indices should be safely ignored."""
        meta = SeriesMetadata(
            values=np.zeros(5),
            timestamps=np.zeros(5),
            series_id=0,
            length=5,
            anomaly_indices=np.array([1, 10, -1]),
        )
        flags = build_flag_column([meta], "anomaly")
        assert len(flags) == 5
        assert flags[1] == 1
        assert np.sum(flags) == 1  # 10 and -1 are out of bounds


# ---------------------------------------------------------------------------
# Integration with overriding generators
# ---------------------------------------------------------------------------
class TestOverridingGenerators:
    """Test extra columns in generators that override generate()."""

    def test_copula_with_extra_columns(self) -> None:
        gen = CopulaGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "anomalies": True,
                "calendar_features": CalendarFeatures(),
                "diagnostics": DiagnosticsConfig(anomalies=True),
            }
        )
        df = gen.generate(n_series=3)
        assert "anomaly_flag" in df.columns
        assert "year" in df.columns
        assert df["unique_id"].n_unique() == 3

    def test_var_with_extra_columns(self) -> None:
        gen = VARGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "changepoints": True,
                "num_changepoints": 2,
                "calendar_features": CalendarFeatures(cyclical=True, basic=False),
                "diagnostics": DiagnosticsConfig(changepoints=True),
            }
        )
        df = gen.generate(n_series=3)
        assert "changepoint_flag" in df.columns
        assert "month_sin" in df.columns
        assert df["unique_id"].n_unique() == 3

    def test_iot_sensor_with_extra_columns(self) -> None:
        gen = IoTSensorGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "h",
                "engine": "polars",
                "seed": 42,
                "n_sensors": 1,
                "missing_data": True,
                "calendar_features": CalendarFeatures(),
                "diagnostics": DiagnosticsConfig(missing=True),
            }
        )
        df = gen.generate(n_series=2)
        assert "missing_flag" in df.columns
        assert "hour" in df.columns

    def test_dau_with_extra_columns_and_event_col(self) -> None:
        gen = DailyActiveUsersGenerator(
            **{
                "min_length": 100,
                "max_length": 100,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "anomalies": True,
                "calendar_features": CalendarFeatures(),
                "diagnostics": DiagnosticsConfig(anomalies=True),
            }
        )
        df = gen.generate(n_series=2)
        # DAU should still have its event column
        assert "event" in df.columns
        assert "anomaly_flag" in df.columns
        assert "year" in df.columns

    def test_iot_multivariate_with_extra_columns(self) -> None:
        gen = IoTSensorGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "h",
                "engine": "polars",
                "seed": 42,
                "n_sensors": 3,
                "calendar_features": CalendarFeatures(),
            }
        )
        df = gen.generate(n_series=2)
        assert "hour" in df.columns
        # 2 networks * 3 sensors = 6 series
        assert df["unique_id"].n_unique() == 6


# ---------------------------------------------------------------------------
# All-in-one integration
# ---------------------------------------------------------------------------
class TestFullIntegration:
    """Test combining all extra column types at once."""

    def test_all_column_types_together(self) -> None:
        gen = RandomWalkGenerator(
            **{
                "min_length": 200,
                "max_length": 200,
                "freq": "h",
                "engine": "polars",
                "seed": 42,
                "anomalies": True,
                "changepoints": True,
                "missing_data": True,
                "calendar_features": CalendarFeatures(cyclical=True),
                "diagnostics": DiagnosticsConfig(
                    anomalies=True, changepoints=True, missing=True
                ),
            }
        )
        df = gen.generate(n_series=3)

        # Core columns
        assert "unique_id" in df.columns
        assert "ds" in df.columns
        assert "y" in df.columns

        # Flag columns
        assert "anomaly_flag" in df.columns
        assert "changepoint_flag" in df.columns
        assert "missing_flag" in df.columns

        # Datetime columns (hourly)
        assert "year" in df.columns
        assert "hour" in df.columns
        assert "hour_sin" in df.columns

        # Series count
        assert df["unique_id"].n_unique() == 3

    def test_pandas_backend(self) -> None:
        """Extra columns should work with the pandas backend too."""
        import pandas as pd

        gen = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "seed": 42,
                "engine": "pandas",
                "anomalies": True,
                "calendar_features": CalendarFeatures(),
                "diagnostics": DiagnosticsConfig(anomalies=True),
            }
        )
        df = gen.generate(n_series=1)
        assert isinstance(df, pd.DataFrame)
        assert "anomaly_flag" in df.columns
        assert "year" in df.columns


# ---------------------------------------------------------------------------
# SynSet integration
# ---------------------------------------------------------------------------
class TestSynSetIntegration:
    """Test column configs with SynSet."""

    def test_synset_with_extra_columns(self) -> None:
        columns = {
            "calendar_features": CalendarFeatures(),
            "diagnostics": DiagnosticsConfig(anomalies=True),
        }
        gen1 = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "anomalies": True,
                **columns,
            }
        )
        gen2 = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "engine": "polars",
                "seed": 43,
                "anomalies": True,
                **columns,
            }
        )
        synset = SynSet(generators=[gen1, gen2])
        df = synset.generate(n_series_per_generator=2)
        assert "anomaly_flag" in df.columns
        assert "year" in df.columns

    def test_synset_mismatched_columns_no_crash(self) -> None:
        """SynSet with mismatched column configs should not crash."""
        gen1 = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "engine": "polars",
                "seed": 42,
                "calendar_features": CalendarFeatures(),
            }
        )
        gen2 = RandomWalkGenerator(
            **{
                "min_length": 50,
                "max_length": 50,
                "freq": "D",
                "engine": "polars",
                "seed": 43,
                "diagnostics": DiagnosticsConfig(anomalies=True),
            }
        )
        # Construction should not crash even with mismatched configs
        synset = SynSet(generators=[gen1, gen2])
        assert synset is not None
