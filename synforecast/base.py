"""Base classes for time series generators."""

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from narwhals.stable.v2.typing import IntoDataFrameT
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from ._core import _add_anomalies, _add_changepoints, _add_missingness, _overlay_scale
from ._lib import batch as _rs_batch
from .exogenous import (
    ExogenousConfig,
    SeriesMetadata,
    build_flag_column,
    extract_datetime_features,
    generate_correlated_exog,
)

# Match Rayon's own default: all logical cores.
# Override with RAYON_NUM_THREADS (e.g. when running inside multiprocessing.Pool).
_default_n_workers: int = int(os.environ.get("RAYON_NUM_THREADS", os.cpu_count() or 1))

# Mapping from generator class names to Rust batch type IDs.
# Must match the constants in rust/src/batch.rs.
#
# IDs 17, 18, 23, 26 are intentionally absent — the four generators that are
# not yet in the batch path:
#   17 — CopulaGenerator
#   18 — VARGenerator
#   23 — DailyActiveUsersGenerator
#   26 — IoTSensorGenerator
# Add _get_batch_params() to any of these generators to enable batch generation.
_GEN_TYPE_MAP: dict[str, int] = {
    "RandomWalkGenerator": 0,
    "SeasonalGenerator": 1,
    "SARIMAGenerator": 2,
    "ETSGenerator": 3,
    "INARGenerator": 4,
    "OrnsteinUhlenbeckGenerator": 5,
    "GeometricBrownianMotionGenerator": 6,
    "JumpDiffusionGenerator": 7,
    "PoissonProcessGenerator": 8,
    "CyclicGenerator": 9,
    "GARCHGenerator": 10,
    "HawkesProcessGenerator": 11,
    "ChaoticSystemGenerator": 12,
    "BoundedProcessGenerator": 13,
    "LevyProcessGenerator": 14,
    "StochasticVolatilityGenerator": 15,
    "RegimeSwitchingGenerator": 16,
    "StateSpaceGenerator": 19,
    "FractionalBrownianMotionGenerator": 20,
    "GaussianProcessGenerator": 21,
    "IntermittentDemandGenerator": 22,
    "EnergyLoadGenerator": 24,
    "VitalSignsGenerator": 25,
    "ClickstreamGenerator": 27,
    "TSIGenerator": 28,
    "TCMGenerator": 29,
    "MARGenerator": 30,
    "KernelSynthGenerator": 31,
}


def _categorize_ids(df: nw.DataFrame, id_col: str) -> nw.DataFrame:
    """Cast the id column to a categorical dtype.

    pandas-like backends keep integer categories (matching utilsforecast);
    other backends only support string categoricals, so ids become string
    categories there. Backends without categorical support keep plain ids.
    """
    if df.implementation.is_pandas_like():
        return df.with_columns(nw.col(id_col).cast(nw.Categorical()))
    try:
        return df.with_columns(nw.col(id_col).cast(nw.String()).cast(nw.Categorical()))
    except Exception:
        return df


def _batch_results_to_metadata(
    gen: "BaseGenerator",
    results: list[dict],
    lengths: np.ndarray,
    start_id: int,
    pi_seeds: np.ndarray,
) -> list["SeriesMetadata"]:
    """Convert raw Rust batch results into a list of SeriesMetadata.

    Shared by BaseGenerator._generate_parallel and SynSet._generate_parallel
    to avoid duplicating this logic. With waveform or observation on, the
    kernel drew only the base series and every overlay is added here from
    the flat ``pi_seeds`` (three per series) passed to it.
    """
    has_exog = gen.exogenous is not None
    all_metadata: list[SeriesMetadata] = []
    for i, r in enumerate(results):
        length = int(lengths[i])

        if gen._ordered_overlays:
            values, cp_idx, anom_idx, miss_idx = gen._apply_pattern_injection(
                np.asarray(r["values"]), pi_seeds[3 * i : 3 * i + 3]
            )
            all_metadata.append(
                SeriesMetadata(
                    values=values,
                    timestamps=None,
                    series_id=start_id + i,
                    length=length,
                    anomaly_indices=anom_idx,
                    changepoint_indices=cp_idx,
                    missing_indices=miss_idx,
                )
            )
            continue

        cp_idx = (
            np.asarray(r["cp_indices"])
            if has_exog and gen.exogenous.changepoint_flags
            else None
        )
        anom_idx = (
            np.asarray(r["anom_indices"])
            if has_exog and gen.exogenous.anomaly_flags
            else None
        )
        miss_idx = (
            np.asarray(r["miss_indices"])
            if has_exog and gen.exogenous.missing_flags
            else None
        )

        all_metadata.append(
            SeriesMetadata(
                values=np.asarray(r["values"]),
                timestamps=None,
                series_id=start_id + i,
                length=length,
                anomaly_indices=anom_idx,
                changepoint_indices=cp_idx,
                missing_indices=miss_idx,
            )
        )
    return all_metadata


def _waveform_on_steps(shape: str, period: int, duty: float, width: int) -> int:
    """Nominal on-time of one waveform cycle in steps; the whole cycle for sawtooth and level shapes."""
    if shape == "spike":
        return min(width, period - 1)
    if shape in ("pulse", "trapezoid"):
        return int(np.clip(round(duty * period), 1, period - 1))
    return period


class BaseGenerator(BaseModel, ABC):
    """Base class for all time series generators.

    Args:
        min_length (int): Minimum length of each series
        max_length (int): Maximum length of each series
        freq (str | int): Frequency of the data. Either a pandas offset alias
            (e.g. 'D', 'h', '5min', 'MS', 'W-MON') or an integer for an
            integer time index
        engine (str): Output dataframe library (default: 'pandas').
            Options are 'pandas', 'polars', 'cudf', 'modin', 'pyarrow'
        alias (str | None): Name of the generator (default: class name)
        id_col (str): Name of the ID column (default: 'unique_id')
        time_col (str): Name of the timestamp column (default: 'ds')
        target_col (str): Name of the value column (default: 'y')
        start_datetime (str): First timestamp of every series, in any format
            accepted by pandas.Timestamp (default: '2000-01-01'). Ignored
            when freq is an integer
        seed (int | None): Random seed for reproducibility (default: None)

        Exogenous parameters:
        exogenous (ExogenousConfig | None): Configuration for exogenous variable
            generation. None = no exogenous columns (default: None)

        Missingness parameters:
        missing_data (bool): Enable missing data patterns (default: False)
        missing_pattern (str): Pattern: 'random', 'block', 'seasonal' (default: 'random')
        missing_rate (float): Proportion of missing values 0-1 (default: 0.1)
        missing_block_size (int): Size of missing blocks for 'block' pattern (default: 3)
        missing_seasonal_period (int): Period for 'seasonal' pattern (default: 7)

        Anomaly parameters:
        anomalies (bool): Enable anomaly injection (default: False)
        anomaly_fraction (float): Fraction of points that are anomalies (default: 0.05)
        anomaly_types (list[str]): Types: 'spike', 'dip', 'level_shift'
            (default: ['spike', 'dip'])
        spike_magnitude (float): Magnitude of spikes (default: 10.0)
        dip_magnitude (float): Magnitude of dips (default: -10.0)
        level_shift_magnitude (float): Magnitude of level shifts (default: 20.0)
        level_shift_duration (int): Duration of level shifts in time steps (default: 10)

        Changepoint parameters:
        changepoints (bool): Enable changepoint injection (default: False)
        num_changepoints (int): Number of changepoints (default: 2)
        changepoint_type (str): Type: 'level', 'trend', 'variance', 'mixed' (default: 'level')
        changepoint_level_changes (list[float] | None): Size of level changes (default: random)
        changepoint_trend_changes (list[float] | None): Size of trend changes (default: random)
        changepoint_variance_changes (list[float] | None): Size of variance changes
            (default: random)
        changepoint_locations (list[float] | None): Relative positions 0-1 (default: random)

        Calendar-cycle parameters:
        calendar_cycles (bool): Add a time-of-day and day-of-week cycle aligned to
            the timestamps, on fixed sub-weekly frequencies (default: False)
        calendar_mode (str): 'additive' or 'multiplicative' (default: 'additive')
        calendar_amplitude (float): Additive profile peak, in standard deviations
            of the series (default: 1.0)
        calendar_swing (float): Multiplicative profile peak, as a fraction of
            the level, below 1 (default: 0.3)
        calendar_harmonics (int): Harmonics of the time-of-day profile (default: 3)
        calendar_weekly (float): Day-of-week strength relative to time-of-day
            (default: 0.0)

        Waveform parameters:
        waveform (bool): Add a sharp-edged repeating template whose period is
            in time steps, independent of freq (default: False)
        waveform_shape (str): 'pulse', 'spike', 'sawtooth', 'trapezoid',
            'levels' or 'staircase' (default: 'pulse')
        waveform_period (int): Cycle length in steps; mean dwell for 'levels'
            and 'staircase' (default: 96)
        waveform_duty (float): On-fraction of a cycle for 'pulse' and
            'trapezoid' (default: 0.1)
        waveform_width (int): Spike width in steps (default: 1)
        waveform_ramp (float): Trapezoid ramp per side, as a fraction of the
            on-time (default: 0.25)
        waveform_levels (int): Number of levels for 'levels' and 'staircase'
            (default: 4)
        waveform_timing (str): 'periodic', 'jitter' or 'renewal' (default:
            'periodic')
        waveform_jitter (float): Per-cycle start jitter, as a fraction of the
            period (default: 0.05)
        waveform_renewal (str): 'gamma' or 'lognormal' renewal durations
            (default: 'gamma')
        waveform_cv (float): Coefficient of variation of renewal durations
            (default: 0.5)
        waveform_direction (str): 'peak' or 'dip' (default: 'peak')
        waveform_mode (str): 'additive' or 'multiplicative' (default: 'additive')
        waveform_amplitude (float): Additive template peak, in standard
            deviations of the series (default: 3.0)
        waveform_swing (float): Multiplicative template peak, as a fraction of
            the level (default: 0.5)
        waveform_base (float): Weight of the base series around its mean;
            0 leaves the exact template (default: 1.0)

        Observation parameters:
        observation (bool): Pass the series through an observation process,
            after anomalies and before missingness (default: False)
        observation_floor (float | None): Saturation floor, as a fraction of
            the way from the series' minimum to its maximum (default: None)
        observation_cap (float | None): Saturation cap, as a fraction of the
            way from the series' minimum to its maximum (default: None)
        observation_quantum (float): Value-grid step, in standard deviations of
            the series; 0 disables quantization (default: 0.0)
        observation_hold (int): Sample-and-hold block in steps; 1 disables it
            (default: 1)
        observation_stuck_fraction (float): Share of steps inside windows
            where the reading freezes (default: 0.0)
        observation_stuck_length (int): Mean length of a frozen window in steps
            (default: 20)

        With waveform or observation on, overlays run in the order calendar,
        waveform, changepoints, anomalies, observation, missingness; otherwise
        the calendar cycle is added last, as before.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Required fields
    min_length: int = Field(..., gt=0, description="Minimum length of each series")
    max_length: int = Field(..., description="Maximum length of each series")
    freq: str | int = Field(
        ..., description="Frequency of the data (pandas offset alias or integer)"
    )

    # Optional fields with defaults
    engine: Literal["pandas", "polars", "cudf", "modin", "pyarrow"] = Field(
        default="pandas", description="Output dataframe library"
    )
    alias: str | None = Field(default=None, description="Name of the generator")
    id_col: str = Field(default="unique_id", description="Name of the ID column")
    time_col: str = Field(default="ds", description="Name of the timestamp column")
    target_col: str = Field(default="y", description="Name of the value column")
    start_datetime_str: str = Field(
        default="2000-01-01",
        alias="start_datetime",
        description="First timestamp of every series",
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )

    # Innovation distribution
    innovation_distribution: Literal[
        "normal", "t", "laplace", "uniform", "skew_normal"
    ] = Field(
        default="normal",
        description=(
            "Distribution for innovation/noise terms. Options: "
            "'normal' (Gaussian), 't' (Student's t, heavy tails), "
            "'laplace' (double exponential, sharp peak), "
            "'uniform' (bounded), 'skew_normal' (asymmetric)"
        ),
    )
    innovation_params: dict[str, float] | None = Field(
        default=None,
        description=(
            "Distribution-specific parameters. "
            "For 't': {'df': 5} (degrees of freedom, must be > 2). "
            "For 'skew_normal': {'alpha': 5} (skewness parameter)."
        ),
    )

    # Exogenous configuration
    exogenous: ExogenousConfig | None = Field(
        default=None,
        description="Configuration for exogenous variable generation",
    )

    # Missingness parameters
    missing_data: bool = Field(
        default=False, description="Enable missing data patterns"
    )
    missing_pattern: Literal["random", "block", "seasonal"] = Field(
        default="random", description="Missing data pattern"
    )
    missing_rate: float = Field(
        default=0.1, ge=0, le=1, description="Proportion of missing values"
    )
    missing_block_size: int = Field(
        default=3, ge=1, description="Size of missing blocks for 'block' pattern"
    )
    missing_seasonal_period: int = Field(
        default=7, ge=1, description="Period for 'seasonal' pattern"
    )

    # Anomaly parameters
    anomalies: bool = Field(default=False, description="Enable anomaly injection")
    anomaly_fraction: float = Field(
        default=0.05, ge=0, le=1, description="Fraction of points that are anomalies"
    )
    anomaly_types: list[str] = Field(
        default=["spike", "dip"], description="Types of anomalies to inject"
    )
    spike_magnitude: float = Field(default=10.0, description="Magnitude of spikes")
    dip_magnitude: float = Field(default=-10.0, description="Magnitude of dips")
    level_shift_magnitude: float = Field(
        default=20.0, description="Magnitude of level shifts"
    )
    level_shift_duration: int = Field(
        default=10, ge=1, description="Duration of level shifts in time steps"
    )

    # Changepoint parameters
    changepoints: bool = Field(
        default=False, description="Enable changepoint injection"
    )
    num_changepoints: int = Field(default=2, ge=0, description="Number of changepoints")
    changepoint_type: Literal["level", "trend", "variance", "mixed"] = Field(
        default="level", description="Type of changepoint"
    )
    changepoint_level_changes: list[float] | None = Field(
        default=None, description="Size of level changes"
    )
    changepoint_trend_changes: list[float] | None = Field(
        default=None, description="Size of trend changes"
    )
    changepoint_variance_changes: list[float] | None = Field(
        default=None, description="Size of variance changes"
    )
    changepoint_locations: list[float] | None = Field(
        default=None, description="Relative positions 0-1"
    )
    overlay_scale: Literal["absolute", "relative"] = Field(
        default="absolute",
        description=(
            "Units of anomaly and changepoint magnitudes. 'absolute' adds them as "
            "given. 'relative' measures them in standard deviations of the series "
            "before any overlay: spike, dip and level-shift magnitudes and level "
            "changes are multiplied by it, trend changes are the total drift over "
            "the rest of the series, and random level and trend changes are drawn "
            "from U(-3, 3)."
        ),
    )

    # Calendar-cycle parameters
    calendar_cycles: bool = Field(
        default=False,
        description=(
            "Add a time-of-day and day-of-week cycle aligned to the timestamps. "
            "Only applies to fixed sub-weekly frequencies: the daily cycle needs a "
            "step under one day, the weekly cycle a step of at most one day."
        ),
    )
    calendar_mode: Literal["additive", "multiplicative"] = Field(
        default="additive",
        description=(
            "'additive' adds calendar_amplitude * std * profile. 'multiplicative' "
            "scales a strictly positive series by (1 + calendar_swing * profile); "
            "series with non-positive values get the additive cycle instead."
        ),
    )
    calendar_amplitude: float = Field(
        default=1.0,
        ge=0,
        description="Additive profile peak, in standard deviations of the series",
    )
    calendar_swing: float = Field(
        default=0.3,
        ge=0,
        lt=1,
        description="Multiplicative profile peak, as a fraction of the level",
    )
    calendar_harmonics: int = Field(
        default=3, ge=1, description="Fourier harmonics of the time-of-day profile"
    )
    calendar_weekly: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Strength of the day-of-week profile relative to the time-of-day one; "
            "on daily data it is the only cycle, so 0 adds nothing there."
        ),
    )

    # Waveform parameters
    waveform: bool = Field(
        default=False,
        description=(
            "Add a sharp-edged repeating template (pulse, spike, sawtooth, "
            "trapezoid or level switching) whose period is in time steps."
        ),
    )
    waveform_shape: Literal[
        "pulse", "spike", "sawtooth", "trapezoid", "levels", "staircase"
    ] = Field(
        default="pulse",
        description=(
            "'pulse' is a rectangle of duty waveform_duty, 'spike' a pulse "
            "waveform_width steps wide, 'sawtooth' a ramp-and-reset over each "
            "cycle, 'trapezoid' a pulse with linear ramps, 'levels' a jump to a "
            "random other level each cycle, 'staircase' a one-level step in a "
            "persistent direction each cycle."
        ),
    )
    waveform_period: int = Field(
        default=96,
        ge=2,
        description="Cycle length in steps; mean dwell for 'levels' and 'staircase'",
    )
    waveform_duty: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="On-fraction of a cycle for 'pulse' and 'trapezoid'",
    )
    waveform_width: int = Field(default=1, ge=1, description="Spike width in steps")
    waveform_ramp: float = Field(
        default=0.25,
        gt=0,
        le=0.5,
        description="Trapezoid ramp per side, as a fraction of the on-time",
    )
    waveform_levels: int = Field(
        default=4, ge=2, description="Number of levels for 'levels' and 'staircase'"
    )
    waveform_timing: Literal["periodic", "jitter", "renewal"] = Field(
        default="periodic",
        description=(
            "'periodic' repeats every waveform_period steps from a random phase; "
            "'jitter' moves each cycle start by a normal offset of std "
            "waveform_jitter * period; 'renewal' draws on and off durations "
            "(whole cycles for sawtooth and level shapes) from waveform_renewal "
            "with the nominal means."
        ),
    )
    waveform_jitter: float = Field(
        default=0.05,
        ge=0,
        le=0.5,
        description="Per-cycle start jitter std, as a fraction of the period",
    )
    waveform_renewal: Literal["gamma", "lognormal"] = Field(
        default="gamma", description="Distribution of renewal durations"
    )
    waveform_cv: float = Field(
        default=0.5,
        gt=0,
        description="Coefficient of variation of renewal durations",
    )
    waveform_direction: Literal["peak", "dip"] = Field(
        default="peak", description="Whether the template rises or falls"
    )
    waveform_mode: Literal["additive", "multiplicative"] = Field(
        default="additive",
        description=(
            "'additive' adds waveform_amplitude * std * template. "
            "'multiplicative' scales a strictly positive series by "
            "(1 +/- waveform_swing * template); other series get the additive "
            "template instead."
        ),
    )
    waveform_amplitude: float = Field(
        default=3.0,
        ge=0,
        description="Additive template peak, in standard deviations of the series",
    )
    waveform_swing: float = Field(
        default=0.5,
        ge=0,
        description="Multiplicative template peak, as a fraction of the level",
    )
    waveform_base: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description=(
            "Weight of the base series around its mean before the template is "
            "added; small values let the template dominate, 0 leaves it exact."
        ),
    )

    # Observation parameters
    observation: bool = Field(
        default=False,
        description=(
            "Pass the series through an observation process (saturation, "
            "quantization, sample-and-hold, frozen windows), after anomalies "
            "and before missingness."
        ),
    )
    observation_floor: float | None = Field(
        default=None,
        ge=0,
        lt=1,
        description=(
            "Saturation floor, as a fraction of the way from the series' "
            "minimum (0) to its maximum (1)"
        ),
    )
    observation_cap: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description=(
            "Saturation cap, as a fraction of the way from the series' "
            "minimum (0) to its maximum (1)"
        ),
    )
    observation_quantum: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Value-grid step in standard deviations of the series; readings are "
            "rounded to multiples of it. 0 disables quantization."
        ),
    )
    observation_hold: int = Field(
        default=1,
        ge=1,
        description=(
            "Sample-and-hold block in steps: each block repeats its first "
            "reading. 1 disables it."
        ),
    )
    observation_stuck_fraction: float = Field(
        default=0.0,
        ge=0,
        lt=1,
        description="Share of steps inside windows where the reading freezes",
    )
    observation_stuck_length: int = Field(
        default=20, ge=2, description="Mean length of a frozen window in steps"
    )

    # Runtime state is private: it is not configuration and must not appear in
    # the constructor or serialized model.
    _start_datetime: pd.Timestamp = PrivateAttr()
    _rng: np.random.Generator = PrivateAttr()
    _ts_cache: np.ndarray | None = PrivateAttr(default=None)

    @property
    def start_datetime(self) -> pd.Timestamp:
        """Parsed first timestamp for datetime-indexed series."""
        return self._start_datetime

    @property
    def rng(self) -> np.random.Generator:
        """Generator-local random number stream."""
        return self._rng

    @field_validator("freq")
    @classmethod
    def validate_freq(cls, v: str | int) -> str | int:
        """Validate that freq is a pandas offset alias or a positive integer."""
        if isinstance(v, int):
            if v <= 0:
                raise ValueError("Integer freq must be positive")
            return v
        try:
            pd.tseries.frequencies.to_offset(v)
        except ValueError as e:
            raise ValueError(
                f"Invalid frequency '{v}'. Use a pandas offset alias "
                "(e.g. 'D', 'h', '5min', 'MS') or an integer."
            ) from e
        return v

    @field_validator("anomaly_types")
    @classmethod
    def validate_anomaly_types(cls, v: list[str]) -> list[str]:
        """Validate anomaly types."""
        valid_anomaly_types = {"spike", "dip", "level_shift"}
        for anomaly_type in v:
            if anomaly_type not in valid_anomaly_types:
                raise ValueError(
                    f"anomaly_type must be one of {valid_anomaly_types}, got '{anomaly_type}'"
                )
        return v

    @model_validator(mode="after")
    def validate_and_compute_fields(self) -> "BaseGenerator":
        """Validate cross-field constraints and compute derived fields."""
        if self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")

        columns = (self.id_col, self.time_col, self.target_col)
        if any(not column.strip() for column in columns):
            raise ValueError("output column names must not be empty")
        if len(set(columns)) != len(columns):
            raise ValueError("id_col, time_col, and target_col must be distinct")

        params = self.innovation_params or {}
        allowed_params = {
            "normal": set(),
            "t": {"df"},
            "laplace": set(),
            "uniform": set(),
            "skew_normal": {"alpha"},
        }[self.innovation_distribution]
        unknown_params = set(params) - allowed_params
        if unknown_params:
            names = ", ".join(sorted(unknown_params))
            raise ValueError(
                f"unsupported innovation_params for "
                f"{self.innovation_distribution!r}: {names}"
            )
        if self.innovation_distribution == "t":
            df = params.get("df", 5.0)
            if not df > 2:
                raise ValueError("innovation_params['df'] must be > 2")

        if self.changepoint_locations is not None and any(
            not np.isfinite(location) or not 0.0 <= location <= 1.0
            for location in self.changepoint_locations
        ):
            raise ValueError("changepoint_locations must contain values in [0, 1]")
        if self.changepoint_variance_changes is not None and any(
            not np.isfinite(change) or change <= 0
            for change in self.changepoint_variance_changes
        ):
            raise ValueError("changepoint_variance_changes must be positive")
        if (
            self.waveform_mode == "multiplicative"
            and self.waveform_direction == "dip"
            and self.waveform_swing > 1
        ):
            raise ValueError("waveform_swing must be <= 1 for multiplicative dips")
        if (
            self.observation_floor is not None
            and self.observation_cap is not None
            and self.observation_floor >= self.observation_cap
        ):
            raise ValueError("observation_floor must be below observation_cap")

        if self.exogenous is not None:
            reserved = set(columns)
            if self.exogenous.anomaly_flags:
                reserved.add("anomaly_flag")
            if self.exogenous.changepoint_flags:
                reserved.add("changepoint_flag")
            if self.exogenous.missing_flags:
                reserved.add("missing_flag")
            collisions = reserved.intersection(
                config.name for config in self.exogenous.correlated
            )
            if collisions:
                names = ", ".join(sorted(collisions))
                raise ValueError(
                    f"correlated exogenous names collide with output columns: {names}"
                )

        self._start_datetime = pd.Timestamp(self.start_datetime_str)
        self._rng = np.random.default_rng(self.seed)

        if self.alias is None:
            object.__setattr__(self, "alias", self.__class__.__name__)

        return self

    def __repr__(self) -> str:
        return self.alias

    def _timestamps(self, length: int) -> np.ndarray:
        """Timestamps (or integer time indices) for a series of `length` steps.

        All series share the same start, so a single range of max_length is
        cached and sliced per series.
        """
        cache = self._ts_cache
        if cache is None or len(cache) < length:
            periods = max(length, self.max_length)
            if isinstance(self.freq, int):
                cache = np.arange(periods, dtype=np.int64) * self.freq
            else:
                cache = pd.date_range(
                    self.start_datetime, periods=periods, freq=self.freq
                ).values
            self._ts_cache = cache
        return cache[:length]

    def _freq_timedelta(self) -> pd.Timedelta | None:
        """Duration of one time step, or None when freq is an integer or a
        non-fixed offset (e.g. 'MS', 'ME', 'YS')."""
        if isinstance(self.freq, int):
            return None
        try:
            return pd.Timedelta(pd.tseries.frequencies.to_offset(self.freq))
        except ValueError:
            return None

    def _step_hours(self, default: float = 24.0) -> float:
        """Hours per time step; `default` when freq is an integer or a
        non-fixed offset."""
        td = self._freq_timedelta()
        if td is None:
            return default
        return td / pd.Timedelta(hours=1)

    def _sample_innovations(
        self,
        size: int | tuple[int, ...],
        scale: float = 1.0,
    ) -> np.ndarray:
        """Sample innovation terms from the configured distribution.

        All distributions are standardized to have approximately mean 0
        and standard deviation equal to ``scale``.

        Args:
            size: Shape of the output array.
            scale: Desired standard deviation of the innovations.

        Returns:
            Innovation samples with the requested shape.
        """
        dist = self.innovation_distribution
        params = self.innovation_params or {}

        if dist == "normal":
            return self.rng.normal(0.0, scale, size)

        if dist == "t":
            df = params.get("df", 5.0)
            if df <= 2:
                raise ValueError(
                    "Degrees of freedom (df) must be > 2 for finite variance."
                )
            # standard_t has variance df/(df-2); rescale to desired sigma
            t_scale = scale * np.sqrt((df - 2) / df)
            return self.rng.standard_t(df, size) * t_scale

        if dist == "laplace":
            # Laplace with scale b has variance 2*b^2, so b = sigma/sqrt(2)
            laplace_scale = scale / np.sqrt(2.0)
            return self.rng.laplace(0.0, laplace_scale, size)

        if dist == "uniform":
            # Uniform[-a, a] has variance a^2/3, so a = sigma*sqrt(3)
            a = scale * np.sqrt(3.0)
            return self.rng.uniform(-a, a, size)

        if dist == "skew_normal":
            alpha = params.get("alpha", 5.0)
            delta = alpha / np.sqrt(1.0 + alpha**2)
            u0 = self.rng.normal(0.0, 1.0, size)
            v = self.rng.normal(0.0, 1.0, size)
            raw = delta * np.abs(u0) + np.sqrt(1.0 - delta**2) * v
            # Centre and rescale: skew-normal has mean delta*sqrt(2/pi)
            # and variance 1 - 2*delta^2/pi
            mean = delta * np.sqrt(2.0 / np.pi)
            std = np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
            return (raw - mean) * (scale / std) if std > 0 else raw * scale

        raise ValueError(f"Unknown innovation_distribution: {dist!r}")

    @property
    def _rs_innov_dist(self) -> int:
        """Map innovation_distribution name to Rust enum int."""
        return {"normal": 0, "t": 1, "laplace": 2, "uniform": 3, "skew_normal": 4}[
            self.innovation_distribution
        ]

    @property
    def _rs_innov_param(self) -> float:
        """Extract the single extra parameter needed by the Rust sampler."""
        params = self.innovation_params or {}
        if self.innovation_distribution == "t":
            return float(params.get("df", 5.0))
        if self.innovation_distribution == "skew_normal":
            return float(params.get("alpha", 5.0))
        return 0.0

    @property
    def _batch_gen_type(self) -> int | None:
        """Return Rust batch generator type ID, or None if not supported."""
        return _GEN_TYPE_MAP.get(self.__class__.__name__)

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """Return (scalar_params, array_params) for Rust batch generation.

        Override in subclasses to enable the batch code path. The parameter
        order must match the ``dispatch_generator`` match arm in
        ``rust/src/batch.rs``.

        Returns None by default (batch path disabled).
        """
        return None

    def _build_pi_config_dict(self) -> dict:
        """Build pattern injection config dict for Rust batch functions."""
        cfg: dict = {
            "changepoints": self.changepoints,
            "num_changepoints": self.num_changepoints,
            "changepoint_type": self.changepoint_type,
            "anomalies": self.anomalies,
            "anomaly_types": self.anomaly_types,
            "anomaly_fraction": self.anomaly_fraction,
            "spike_magnitude": self.spike_magnitude,
            "dip_magnitude": self.dip_magnitude,
            "level_shift_magnitude": self.level_shift_magnitude,
            "level_shift_duration": self.level_shift_duration,
            "missing_data": self.missing_data,
            "missing_pattern": self.missing_pattern,
            "missing_rate": self.missing_rate,
            "missing_block_size": self.missing_block_size,
            "missing_seasonal_period": self.missing_seasonal_period,
            "overlay_scale": self.overlay_scale,
        }
        if self.changepoint_locations is not None:
            cfg["changepoint_locations"] = self.changepoint_locations
        if self.changepoint_level_changes is not None:
            cfg["changepoint_level_changes"] = self.changepoint_level_changes
        if self.changepoint_trend_changes is not None:
            cfg["changepoint_trend_changes"] = self.changepoint_trend_changes
        if self.changepoint_variance_changes is not None:
            cfg["changepoint_variance_changes"] = self.changepoint_variance_changes
        if self._ordered_overlays:
            # The native kernel then draws the base only; _apply_pattern_injection adds every overlay in order.
            cfg.update(changepoints=False, anomalies=False, missing_data=False)
        return cfg

    @property
    def _ordered_overlays(self) -> bool:
        """Whether overlays run as calendar, waveform, changepoints, anomalies, observation, missingness."""
        return self.waveform or self.observation

    def _add_missingness(
        self, values: np.ndarray, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Add missing data patterns to a time series.

        Args:
            values: Array of time series values
            rng: Stream for the missingness draws. Defaults to ``self.rng``.

        Returns:
            Tuple of (modified values, missing indices or None)
        """
        if not self.missing_data:
            return values, None

        values, metadata = _add_missingness(
            values=values,
            rng=self.rng if rng is None else rng,
            missing_pattern=self.missing_pattern,
            missing_rate=self.missing_rate,
            missing_block_size=self.missing_block_size,
            missing_seasonal_period=self.missing_seasonal_period,
        )

        if self.exogenous and self.exogenous.missing_flags:
            return values, metadata["missing_indices"]
        return values, None

    def _add_anomalies(
        self,
        values: np.ndarray,
        rng: np.random.Generator | None = None,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Add anomalies to a time series.

        Args:
            values: Array of time series values
            rng: Stream for the anomaly draws. Defaults to ``self.rng``.

        Returns:
            Tuple of (modified values, anomaly indices or None)
        """
        if not self.anomalies:
            return values, None

        values, metadata = _add_anomalies(
            values=values,
            rng=self.rng if rng is None else rng,
            anomaly_types=self.anomaly_types,
            anomaly_fraction=self.anomaly_fraction,
            spike_magnitude=self.spike_magnitude,
            dip_magnitude=self.dip_magnitude,
            level_shift_magnitude=self.level_shift_magnitude,
            level_shift_duration=self.level_shift_duration,
            scale=scale,
        )

        if self.exogenous and self.exogenous.anomaly_flags:
            return values, metadata["anomaly_indices"]
        return values, None

    def _add_changepoints(
        self,
        values: np.ndarray,
        rng: np.random.Generator | None = None,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Add changepoints (structural breaks) to a time series.

        Args:
            values: Array of time series values
            rng: Stream for the changepoint draws. Defaults to ``self.rng``.

        Returns:
            Tuple of (modified values, changepoint indices or None)
        """
        if not self.changepoints or self.num_changepoints == 0:
            return values, None

        changepoint_locations = (
            np.array(self.changepoint_locations)
            if self.changepoint_locations is not None
            else np.array([])
        )
        changepoint_level_changes = (
            np.array(self.changepoint_level_changes)
            if self.changepoint_level_changes is not None
            else np.array([])
        )
        changepoint_trend_changes = (
            np.array(self.changepoint_trend_changes)
            if self.changepoint_trend_changes is not None
            else np.array([])
        )
        changepoint_variance_changes = (
            np.array(self.changepoint_variance_changes)
            if self.changepoint_variance_changes is not None
            else np.array([])
        )

        values, metadata = _add_changepoints(
            values=values,
            rng=self.rng if rng is None else rng,
            num_changepoints=self.num_changepoints,
            changepoint_locations=changepoint_locations,
            changepoint_type=self.changepoint_type,
            changepoint_level_changes=changepoint_level_changes,
            changepoint_trend_changes=changepoint_trend_changes,
            changepoint_variance_changes=changepoint_variance_changes,
            relative=self.overlay_scale == "relative",
            scale=scale,
        )

        if self.exogenous and self.exogenous.changepoint_flags:
            return values, metadata["changepoint_indices"]
        return values, None

    def _draw_pi_seeds(self, n_series: int) -> np.ndarray:
        """Draw (changepoint, anomaly, missingness) seeds per series, as in the Rust batch path."""
        return self.rng.integers(0, 2**63, size=(n_series, 3), dtype=np.uint64)

    def _apply_pattern_injection(
        self, values: np.ndarray, pi_seeds: Sequence[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Apply the full pattern injection pipeline.

        Runs changepoints → anomalies → missingness in order, each on its own
        stream. With waveform or observation on, it runs calendar → waveform →
        changepoints → anomalies → observation → missingness instead, the new
        overlays on streams derived from the same seeds.

        Args:
            values: Array of time series values
            pi_seeds: One row of :meth:`_draw_pi_seeds`.

        Returns:
            Tuple of (values, changepoint_indices, anomaly_indices, missing_indices). Index arrays are None when the corresponding flag is not requested.
        """
        cp_seed, anom_seed, miss_seed = (int(s) for s in pi_seeds)
        if self._ordered_overlays:
            return self._apply_ordered_overlays(values, cp_seed, anom_seed, miss_seed)
        # Relative magnitudes are sized on the base series, before any overlay, as in the Rust batch path.
        scale = _overlay_scale(values) if self.overlay_scale == "relative" else 1.0
        values, cp_indices = self._add_changepoints(
            values, np.random.default_rng(cp_seed), scale
        )
        values, anom_indices = self._add_anomalies(
            values, np.random.default_rng(anom_seed), scale
        )
        values, miss_indices = self._add_missingness(
            values, np.random.default_rng(miss_seed)
        )
        return values, cp_indices, anom_indices, miss_indices

    def _apply_ordered_overlays(
        self, values: np.ndarray, cp_seed: int, anom_seed: int, miss_seed: int
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Calendar → waveform → changepoints → anomalies → observation → missingness, each on its own stream."""
        values = np.array(values, dtype=np.float64)
        # Streams for the new overlays derive from the pattern-injection seeds, so self.rng is consumed as before.
        calendar_rng, waveform_rng, observation_rng = (
            np.random.default_rng([cp_seed, anom_seed, miss_seed, k]) for k in range(3)
        )
        if self.calendar_cycles:
            values = self._add_calendar_cycles(values, calendar_rng)
        if self.waveform:
            values = self._add_waveform(values, waveform_rng)
        # Relative magnitudes are sized on the series with its periodic structure.
        scale = _overlay_scale(values) if self.overlay_scale == "relative" else 1.0
        values, cp_indices = self._add_changepoints(
            values, np.random.default_rng(cp_seed), scale
        )
        values, anom_indices = self._add_anomalies(
            values, np.random.default_rng(anom_seed), scale
        )
        if self.observation:
            values = self._add_observation(values, observation_rng)
        values, miss_indices = self._add_missingness(
            values, np.random.default_rng(miss_seed)
        )
        return values, cp_indices, anom_indices, miss_indices

    def _waveform_cycles(
        self, length: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cycle starts (the first at or before step 0) and on-durations covering `length` steps."""
        period = self.waveform_period
        shape = self.waveform_shape
        on = _waveform_on_steps(shape, period, self.waveform_duty, self.waveform_width)
        whole = shape in ("sawtooth", "levels", "staircase")
        if self.waveform_timing == "renewal":

            def durations(mean: float, n: int) -> np.ndarray:
                cv = self.waveform_cv
                if self.waveform_renewal == "gamma":
                    draws = rng.gamma(1.0 / cv**2, mean * cv**2, n)
                else:
                    sigma2 = np.log1p(cv**2)
                    draws = rng.lognormal(np.log(mean) - sigma2 / 2, np.sqrt(sigma2), n)
                return np.maximum(1, np.rint(np.minimum(draws, 1e9))).astype(np.int64)

            n = length // period + 2
            ons: list[np.ndarray] = []
            cycles: list[np.ndarray] = []
            total = 0
            while total <= length + period:
                if whole:
                    cyc = durations(period, n)
                    on_k = cyc
                else:
                    on_k = np.full(n, on) if shape == "spike" else durations(on, n)
                    cyc = on_k + durations(period - on, n)
                ons.append(on_k)
                cycles.append(cyc)
                total += int(cyc.sum())
            on_len = np.concatenate(ons)
            cycle_len = np.concatenate(cycles)
            # Start inside the first cycle so the origin falls at a random point of the process.
            offset = int(rng.integers(cycle_len[0]))
            starts = np.concatenate(([0], np.cumsum(cycle_len)[:-1])) - offset
            return starts, on_len
        # Cycles -1 .. length // period + 3 cover every step even after a half-period jitter.
        n = length // period + 5
        phase = int(rng.integers(period))
        starts = np.arange(-1, n - 1, dtype=np.int64) * period - phase
        if self.waveform_timing == "jitter" and self.waveform_jitter > 0:
            half = period // 2
            shift = np.rint(rng.normal(0.0, self.waveform_jitter * period, n))
            starts = np.sort(starts + np.clip(shift, -half, half).astype(np.int64))
        gaps = np.diff(starts, append=starts[-1] + period)
        return starts, np.minimum(on, gaps) if not whole else gaps

    def _waveform_template(self, length: int, rng: np.random.Generator) -> np.ndarray:
        """Template in [0, 1] for one series: sharp-edged shape per cycle, cycles timed by waveform_timing."""
        starts, on_len = self._waveform_cycles(length, rng)
        t = np.arange(length, dtype=np.int64)
        cycle = np.searchsorted(starts, t, side="right") - 1
        pos = t - starts[cycle]
        on = on_len[cycle]
        shape = self.waveform_shape
        if shape in ("pulse", "spike"):
            return (pos < on).astype(np.float64)
        if shape == "trapezoid":
            ramp = np.clip(np.rint(self.waveform_ramp * on), 1, np.maximum(on // 2, 1))
            rise = np.minimum((pos + 1) / (ramp + 1), (on - pos) / (ramp + 1))
            return np.where(pos < on, np.minimum(rise, 1.0), 0.0)
        if shape == "sawtooth":
            return (pos + 1) / on
        top = self.waveform_levels - 1
        k = starts.size
        if shape == "levels":
            # Each cycle jumps to a uniformly drawn other level.
            level = (rng.integers(top + 1) + np.cumsum(rng.integers(1, top + 1, k))) % (
                top + 1
            )
        else:
            # A one-level step per cycle in a direction that persists with probability 0.75, folded at both ends.
            flips = np.where(rng.random(k) < 0.25, -1, 1)
            steps = rng.choice((-1, 1)) * np.cumprod(flips)
            walk = rng.integers(top + 1) + np.cumsum(steps)
            folded = np.mod(walk, 2 * top)
            level = np.where(folded <= top, folded, 2 * top - folded)
        return level[cycle] / top

    def _add_waveform(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Shrink the base around its mean by waveform_base and add the template; missing values stay missing."""
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return values
        scale = _overlay_scale(values)
        template = self._waveform_template(values.size, rng)
        mean = finite.mean()
        weight = self.waveform_base
        base = values if weight == 1 else mean + weight * (values - mean)
        sign = 1.0 if self.waveform_direction == "peak" else -1.0
        if self.waveform_mode == "multiplicative" and finite.min() > 0:
            return base * (1.0 + sign * self.waveform_swing * template)
        return base + sign * self.waveform_amplitude * scale * template

    def _add_observation(
        self, values: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Saturate, quantize, sample-and-hold and freeze readings; missing values stay missing and never spread."""
        finite_mask = np.isfinite(values)
        if finite_mask.sum() < 2:
            return values
        values = values.copy()
        finite = values[finite_mask]
        # Bounds sit at fractions of the range, so a rare pulse is truncated rather than removed.
        low, span = finite.min(), finite.max() - finite.min()
        lower = (
            -np.inf
            if self.observation_floor is None
            else low + self.observation_floor * span
        )
        upper = (
            np.inf
            if self.observation_cap is None
            else low + self.observation_cap * span
        )
        values[finite_mask] = np.clip(finite, lower, upper)
        if self.observation_quantum > 0:
            # Clipping first keeps every reading on the grid.
            quantum = self.observation_quantum * _overlay_scale(values)
            values = np.round(values / quantum) * quantum
        n = values.size
        steps = np.arange(n)
        if self.observation_hold > 1:
            hold = self.observation_hold
            block = (steps + int(rng.integers(hold))) // hold
            edges = np.flatnonzero(np.diff(block, prepend=-1))
            # Each block repeats its first finite reading.
            first = np.minimum.reduceat(np.where(finite_mask, steps, n), edges)
            source = np.append(values, np.nan)[first]
            values = np.where(finite_mask, source[block - block[0]], np.nan)
        if self.observation_stuck_fraction > 0:
            mean_length = self.observation_stuck_length
            count = max(1, round(self.observation_stuck_fraction * n / mean_length))
            lengths = rng.integers(
                max(2, mean_length // 2), mean_length + mean_length // 2 + 1, count
            )
            starts = rng.integers(n, size=count)
            # A frozen window repeats the last finite reading at or before its start.
            last = np.maximum.accumulate(np.where(finite_mask, steps, -1))
            for start, length in zip(starts, lengths, strict=True):
                if last[start] < 0:
                    continue
                window = slice(start, start + length)
                values[window] = np.where(
                    finite_mask[window], values[last[start]], np.nan
                )
        return values

    def _build_dataframe(
        self,
        series_metadata: list[SeriesMetadata],
    ) -> IntoDataFrameT:
        """Build the output DataFrame from accumulated series metadata.

        Handles flattening values/timestamps/IDs, adding exogenous columns
        (pattern flags, datetime features, correlated variables), and merging
        any generator-specific extra columns.

        Args:
            series_metadata: List of SeriesMetadata for all generated series;
                series without timestamps get the generator's calendar.
        Returns:
            DataFrame in the configured engine.
        """
        # Flatten core columns
        flat_values = np.concatenate([m.values for m in series_metadata])
        flat_timestamps = np.concatenate(
            [
                self._timestamps(m.length) if m.timestamps is None else m.timestamps
                for m in series_metadata
            ]
        )
        flat_ids = np.repeat(
            np.array([m.series_id for m in series_metadata], dtype=np.int64),
            [m.length for m in series_metadata],
        )

        result: dict[str, Any] = {
            self.id_col: flat_ids,
            self.time_col: flat_timestamps,
            self.target_col: flat_values,
        }

        # Merge generator-specific extra columns (e.g., DAU event_col)
        extra_col_names: set[str] = set()
        for meta in series_metadata:
            if meta.extra_columns:
                extra_col_names.update(meta.extra_columns.keys())
        for col_name in sorted(extra_col_names):
            result[col_name] = np.concatenate(
                [
                    meta.extra_columns.get(col_name, np.full(meta.length, np.nan))
                    for meta in series_metadata
                ]
            )

        # Add exogenous columns
        if self.exogenous is not None:
            # Pattern injection flag columns
            if self.exogenous.anomaly_flags:
                result["anomaly_flag"] = build_flag_column(series_metadata, "anomaly")
            if self.exogenous.changepoint_flags:
                result["changepoint_flag"] = build_flag_column(
                    series_metadata, "changepoint"
                )
            if self.exogenous.missing_flags:
                result["missing_flag"] = build_flag_column(series_metadata, "missing")

            # Datetime features
            if self.exogenous.datetime_features or self.exogenous.datetime_cyclical:
                if isinstance(self.freq, int):
                    raise ValueError(
                        "Datetime features require a datetime frequency; "
                        "got an integer freq."
                    )
                dt_cols = extract_datetime_features(
                    flat_timestamps,
                    self.freq,
                    include_basic=self.exogenous.datetime_features,
                    include_cyclical=self.exogenous.datetime_cyclical,
                )
                result.update(dt_cols)

            # Correlated exogenous
            for exog_config in self.exogenous.correlated:
                result[exog_config.name] = generate_correlated_exog(
                    series_metadata, flat_values, exog_config, self.rng
                )

        df_nw = nw.DataFrame.from_dict(result, backend=self.engine)
        df = _categorize_ids(df_nw, self.id_col).to_native()

        return df

    def _generate_one_series(
        self,
        length: int,
        series_id: int,
        gen_seed: int,
        pi_seeds: Sequence[int] | np.ndarray,
    ) -> SeriesMetadata:
        """Generate a single series with timestamps and pattern injection.

        Args:
            length: Length of the series to generate
            series_id: ID for this series
            gen_seed: Seed for the series' generation stream.
            pi_seeds: One row of :meth:`_draw_pi_seeds`.

        Returns:
            SeriesMetadata for the generated series, without timestamps.
        """
        local_copy = self.model_copy()
        local_copy._rng = np.random.default_rng(gen_seed)
        values = local_copy.generate_single_series(length=length)

        values, cp_indices, anom_indices, miss_indices = self._apply_pattern_injection(
            values, pi_seeds
        )

        return SeriesMetadata(
            values=values,
            timestamps=None,
            series_id=series_id,
            length=length,
            anomaly_indices=anom_indices,
            changepoint_indices=cp_indices,
            missing_indices=miss_indices,
        )

    def generate(
        self,
        n_series: int,
        start_id: int = 0,
        n_jobs: int = -1,
    ) -> IntoDataFrameT:
        """Generate synthetic time series data.

        Args:
            n_series (int): Number of time series to generate
            start_id (int, optional): Starting ID for the series numbering (default: 0)
                Series will be numbered from start_id to start_id + n_series - 1
            n_jobs (int): Number of parallel workers. -1 (default) uses
                ``RAYON_NUM_THREADS`` if set, otherwise all logical cores.
                Results are seed-deterministic and do not depend on n_jobs.

        Returns:
            DataFrame in long format with columns [id_col, time_col, target_col] (default ['unique_id', 'ds', 'y']), plus any exogenous or flag columns.
        """
        if n_series < 1:
            raise ValueError(f"n_series must be a positive integer, got {n_series}.")
        return self._build_dataframe(
            self._metadata_with_calendar(n_series, start_id, n_jobs)
        )

    def generate_values(self, n_series: int, n_jobs: int = -1) -> list[np.ndarray]:
        """Generate series as value arrays, without building a DataFrame.

        Draws exactly the series :meth:`generate` would, with the same overlays
        (missing points as NaN), and leaves the generator in the same state.
        No DataFrame, exogenous column, or calendar is built, so lengths beyond
        pandas' datetime range work. Generators with their own generation
        scheme override :meth:`_generate_metadata`, which both methods share.

        Args:
            n_series (int): Number of time series to generate.
            n_jobs (int): Number of parallel workers. -1 (default) uses
                ``RAYON_NUM_THREADS`` if set, otherwise all logical cores.
                Results are seed-deterministic and do not depend on n_jobs.

        Returns:
            One 1-D float array per series in id order (``n_series * n_sensors``
            for ``IoTSensorGenerator``).
        """
        if n_series < 1:
            raise ValueError(f"n_series must be a positive integer, got {n_series}.")
        metadata = self._metadata_with_calendar(n_series, 0, n_jobs)
        # Correlated exogenous draws still advance self.rng, as in generate().
        if self.exogenous is not None and self.exogenous.correlated:
            flat_values = np.concatenate([m.values for m in metadata])
            for exog_config in self.exogenous.correlated:
                generate_correlated_exog(metadata, flat_values, exog_config, self.rng)
        return [np.asarray(m.values, dtype=np.float64) for m in metadata]

    def _metadata_with_calendar(
        self,
        n_series: int,
        start_id: int,
        n_jobs: int,
    ) -> list[SeriesMetadata]:
        """:meth:`_generate_metadata` plus the calendar cycle, which draws its seeds only when enabled."""
        return self._apply_calendar_cycles(
            self._generate_metadata(n_series, start_id, n_jobs)
        )

    def _apply_calendar_cycles(
        self, metadata: list[SeriesMetadata]
    ) -> list[SeriesMetadata]:
        """Add the calendar cycle to drawn series in place, drawing its seeds only when enabled.

        Every path that draws series for a DataFrame must call this after its
        metadata, including ``SynSet``'s multi-generator batch. With waveform or
        observation on, the cycle is already added in order by
        :meth:`_apply_pattern_injection`, so this is a no-op.
        """
        if not self.calendar_cycles or self._ordered_overlays:
            return metadata
        seeds = self.rng.integers(0, 2**63, size=len(metadata), dtype=np.uint64)
        for m, seed in zip(metadata, seeds, strict=True):
            m.values = self._add_calendar_cycles(
                np.asarray(m.values, dtype=np.float64), np.random.default_rng(seed)
            )
        return metadata

    def _calendar_profile(
        self, length: int, rng: np.random.Generator
    ) -> np.ndarray | None:
        """Time-of-day plus day-of-week profile with peak 1, or None when the frequency has no calendar cycle."""
        step = self._freq_timedelta()
        if step is None or step <= pd.Timedelta(0):
            return None
        step_hours = step / pd.Timedelta(hours=1)
        has_daily = step_hours < 24
        has_weekly = step_hours <= 24 and self.calendar_weekly > 0
        if not (has_daily or has_weekly):
            return None
        start = self.start_datetime
        hours = (start - start.normalize()) / pd.Timedelta(hours=1)
        hours = hours + step_hours * np.arange(length, dtype=np.float64)
        dow = (start.dayofweek + np.floor_divide(hours, 24.0).astype(np.int64)) % 7
        profile = np.zeros(length)
        if has_daily:
            k = np.arange(1, self.calendar_harmonics + 1)
            coef = rng.normal(size=(2, k.size)) / k
            grid = np.linspace(0.0, 1.0, 1440, endpoint=False)

            def daily(theta: np.ndarray) -> np.ndarray:
                angle = 2 * np.pi * np.outer(theta, k)
                return np.cos(angle) @ coef[0] + np.sin(angle) @ coef[1]

            peak = np.abs(daily(grid)).max()
            if peak > 0:
                profile += daily(np.mod(hours, 24.0) / 24.0) / peak
        if has_weekly:
            # Weekend contrast of random sign plus small day-to-day variation.
            days = 0.25 * rng.normal(size=7)
            days[5:] += rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.5)
            days -= days.mean()
            days /= np.abs(days).max()
            weight = self.calendar_weekly if has_daily else 1.0
            profile = (profile + weight * days[dow]) / (1.0 + weight * has_daily)
        return profile

    def _add_calendar_cycles(
        self, values: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Apply the calendar profile to one series; missing values stay missing."""
        profile = self._calendar_profile(values.size, rng)
        if profile is None:
            return values
        finite = values[np.isfinite(values)]
        if self.calendar_mode == "multiplicative" and finite.size and finite.min() > 0:
            return values * (1.0 + self.calendar_swing * profile)
        return values + self.calendar_amplitude * _overlay_scale(values) * profile

    def _generate_metadata(
        self,
        n_series: int,
        start_id: int,
        n_jobs: int,
    ) -> list[SeriesMetadata]:
        """Draw the series of :meth:`generate` as metadata, without timestamps.

        Both :meth:`generate` and :meth:`generate_values` go through this hook,
        so a generator with its own generation scheme overrides this method
        rather than :meth:`generate`. Timestamps are left as None and filled in
        by :meth:`_build_dataframe`.
        """
        if n_jobs == -1:
            n_jobs = _default_n_workers
        return self._generate_parallel(n_series, start_id, n_jobs)

    def _generate_parallel(
        self,
        n_series: int,
        start_id: int,
        n_jobs: int,
    ) -> list[SeriesMetadata]:
        """Generate series in parallel via Rust rayon batch path.

        Generators without a Rust batch implementation use the threaded path.
        """
        gen_type = self._batch_gen_type
        batch_params = self._get_batch_params() if gen_type is not None else None
        if batch_params is None:
            return self._generate_parallel_threaded(n_series, start_id, n_jobs)

        scalar_params, array_params = batch_params

        # Pre-sample lengths and seeds deterministically from master RNG
        lengths = np.array(
            [
                int(self.rng.integers(self.min_length, self.max_length + 1))
                for _ in range(n_series)
            ],
            dtype=np.int32,
        )
        gen_seeds = self.rng.integers(0, 2**63, size=n_series, dtype=np.uint64)
        # 3 PI seeds per series: changepoint, anomaly, missingness
        pi_seeds = self._draw_pi_seeds(n_series).ravel()

        pi_config = self._build_pi_config_dict()

        results = _rs_batch.generate_batch(
            gen_type,
            scalar_params,
            array_params,
            lengths,
            gen_seeds,
            pi_seeds,
            pi_config,
            n_workers=n_jobs,
        )

        return _batch_results_to_metadata(self, results, lengths, start_id, pi_seeds)

    def _generate_parallel_threaded(
        self,
        n_series: int,
        start_id: int,
        n_jobs: int,
    ) -> list[SeriesMetadata]:
        """Generate series in parallel using ThreadPoolExecutor.

        Pre-samples all random state from self.rng sequentially for
        reproducibility, then dispatches generation to threads.
        """
        # Pre-sample lengths and seeds deterministically from master RNG
        lengths = [
            int(self.rng.integers(self.min_length, self.max_length + 1))
            for _ in range(n_series)
        ]
        gen_seeds = [int(self.rng.integers(0, 2**63)) for _ in range(n_series)]
        pi_seeds = self._draw_pi_seeds(n_series)

        workers = max(1, min(n_series, n_jobs))

        def gen_one(i: int) -> SeriesMetadata:
            return self._generate_one_series(
                length=lengths[i],
                series_id=start_id + i,
                gen_seed=gen_seeds[i],
                pi_seeds=pi_seeds[i],
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            all_metadata = list(pool.map(gen_one, range(n_series)))

        return all_metadata

    @abstractmethod
    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single time series.

        Args:
            length (int): The length of the series to generate

        Returns:
            Array of time series values
        """
        pass
