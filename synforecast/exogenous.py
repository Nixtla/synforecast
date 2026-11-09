"""Calendar features and diagnostic labels added to generator output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CalendarFeatures(BaseModel):
    """Calendar columns derived from the timestamps.

    These are deterministic functions of time, so they are known for every
    future period and can be used as future-known model inputs. Features a
    frequency cannot vary (for example hour of day at daily frequency) are
    omitted.

    Args:
        basic (bool): Add year, quarter, month, day of year and, where the
            frequency resolves them, day of week, day of month, weekend flag,
            hour and minute (default: True).
        cyclical (bool): Add sin/cos encodings of hour, day of week, month and
            day of year (default: False).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    basic: bool = Field(default=True, description="Add calendar columns")
    cyclical: bool = Field(default=False, description="Add sin/cos encodings")

    @model_validator(mode="after")
    def _check(self) -> CalendarFeatures:
        if not (self.basic or self.cyclical):
            raise ValueError("enable basic and/or cyclical, or pass None")
        return self


class DiagnosticsConfig(BaseModel):
    """Ground-truth labels of what the overlays injected.

    Adds binary columns marking the steps where anomalies, changepoints or
    missing values were injected. These are labels for evaluation (for
    example anomaly or changepoint detection), not model inputs: realized
    anomalies and breaks are not known at a forecast origin, so do not feed
    these columns to a forecaster as known-future covariates. Events that are
    announced in advance belong in a grouped task with explicit roles (see
    :mod:`synforecast.groups`).

    Args:
        anomalies (bool): Add an ``anomaly_flag`` column (default: False).
        changepoints (bool): Add a ``changepoint_flag`` column
            (default: False).
        missing (bool): Add a ``missing_flag`` column (default: False).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anomalies: bool = Field(default=False, description="Add anomaly_flag")
    changepoints: bool = Field(default=False, description="Add changepoint_flag")
    missing: bool = Field(default=False, description="Add missing_flag")

    @model_validator(mode="after")
    def _check(self) -> DiagnosticsConfig:
        if not (self.anomalies or self.changepoints or self.missing):
            raise ValueError("enable at least one label, or pass None")
        return self

    @property
    def columns(self) -> list[str]:
        """Names of the label columns this config adds."""
        names = []
        if self.anomalies:
            names.append("anomaly_flag")
        if self.changepoints:
            names.append("changepoint_flag")
        if self.missing:
            names.append("missing_flag")
        return names


@dataclass
class SeriesMetadata:
    """Accumulated metadata for a single generated series.

    Carries per-series data through the generation pipeline to
    the DataFrame construction step.
    """

    values: np.ndarray
    timestamps: np.ndarray | None
    series_id: int
    length: int
    anomaly_indices: np.ndarray | None = None
    changepoint_indices: np.ndarray | None = None
    missing_indices: np.ndarray | None = None
    extra_columns: dict[str, np.ndarray] = field(default_factory=dict)


def extract_datetime_features(
    timestamps: np.ndarray,
    freq: str,
    include_basic: bool = True,
    include_cyclical: bool = False,
) -> dict[str, np.ndarray]:
    """Extract datetime features from a numpy datetime64 array.

    Features are filtered based on the frequency to avoid constant columns.
    For example, hourly features are omitted for daily frequency data.

    Args:
        timestamps: Array of numpy datetime64 values
        freq: Frequency as a pandas offset alias (e.g., 'h', 'D', 'MS')
        include_basic: Include calendar features (year, month, etc.)
        include_cyclical: Include sin/cos cyclical encodings

    Returns:
        Dict mapping column names to numpy arrays
    """
    ts = pd.DatetimeIndex(timestamps)

    features: dict[str, np.ndarray] = {}

    # Classify the step size to skip features that would be constant
    # (e.g. hour-of-day for daily data). Non-fixed offsets (monthly and
    # coarser) have no fixed Timedelta and only keep the coarse features.
    try:
        step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    except ValueError:
        step = None
    sub_monthly = step is not None and step <= pd.Timedelta(days=1)
    sub_daily = step is not None and step < pd.Timedelta(days=1)
    sub_hourly = step is not None and step < pd.Timedelta(hours=1)

    if include_basic:
        features["year"] = np.asarray(ts.year, dtype=np.int32)
        features["quarter"] = np.asarray(ts.quarter, dtype=np.int8)
        features["month"] = np.asarray(ts.month, dtype=np.int8)
        features["day_of_year"] = np.asarray(ts.dayofyear, dtype=np.int16)

        if sub_monthly:
            features["day_of_week"] = np.asarray(ts.dayofweek, dtype=np.int8)
            features["day_of_month"] = np.asarray(ts.day, dtype=np.int8)
            features["is_weekend"] = np.asarray(ts.dayofweek >= 5, dtype=np.int8)

        if sub_daily:
            features["hour"] = np.asarray(ts.hour, dtype=np.int8)

        if sub_hourly:
            features["minute"] = np.asarray(ts.minute, dtype=np.int8)

    if include_cyclical:
        hour = np.asarray(ts.hour)
        dow = np.asarray(ts.dayofweek)
        month = np.asarray(ts.month)
        doy = np.asarray(ts.dayofyear)

        if sub_daily:
            features["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
            features["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

        if sub_monthly:
            features["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
            features["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

        features["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12).astype(np.float32)
        features["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12).astype(np.float32)

        features["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365.25).astype(np.float32)
        features["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365.25).astype(np.float32)

    return features


def build_flag_column(
    series_metadata: list[SeriesMetadata],
    flag_type: str,
) -> np.ndarray:
    """Build a binary flag array from series metadata.

    Args:
        series_metadata: List of SeriesMetadata with index arrays
        flag_type: One of 'anomaly', 'changepoint', 'missing'

    Returns:
        Concatenated binary int8 array (1 = flagged, 0 = not)
    """
    attr_name = f"{flag_type}_indices"
    all_flags = []
    for meta in series_metadata:
        flags = np.zeros(meta.length, dtype=np.int8)
        indices = getattr(meta, attr_name, None)
        if indices is not None and len(indices) > 0:
            valid = indices[(indices >= 0) & (indices < meta.length)]
            flags[valid] = 1
        all_flags.append(flags)
    return np.concatenate(all_flags)
