"""Grouped tasks: aligned target and covariate channels with known structure.

A :class:`GroupPrior` samples groups of aligned channels. Every channel is a
target or a covariate, and every covariate is either known in the future
(``"future"``) or observed only up to the forecast origin (``"past"``). The
mechanism decides how the channels relate:

- :class:`RegressionSpec`: one target caused by sampled covariates through a
  random structural regression, plus distractors from the same pool;
- :class:`FixedRegressionSpec`: an explicit regression (probes);
- :class:`CoupledSpec`: related channels from one base generator, coupled by
  mixing and lead-lag;
- :class:`VARXSpec`: a linear VAR panel of targets driven by regressors, with
  an oracle forecast (:class:`VARXTruth`).

The ground truth of every group is returned with it as a versioned recipe.

The idea of teaching a forecasting model to use covariates from synthetic
regression problems follows TabPFN (Hollmann et al., 2025,
https://doi.org/10.1038/s41586-024-08328-6) and the synthetic covariate data
of Chronos-2 (Ansari et al., 2025, https://arxiv.org/abs/2510.15821). The
term sampler, the edge functions, the calibration of the effect strength and
the role assignment are SynForecast's own design rather than a reproduction of
either reference.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from synforecast.base import BaseGenerator

RECIPE_VERSION = 1

EdgeKind = Literal[
    "linear",
    "saturating",
    "quadratic",
    "signed_sqrt",
    "abs_sqrt",
    "threshold",
    "hinge",
]
InteractionKind = Literal["product", "gate"]
Role = Literal["future", "past"]

_EDGE_KINDS: tuple[str, ...] = (
    "linear",
    "saturating",
    "quadratic",
    "signed_sqrt",
    "abs_sqrt",
    "threshold",
    "hinge",
)
_INTERACTION_KINDS: tuple[str, ...] = ("product", "gate")
_RESERVED_PARAMS = frozenset(
    {"min_length", "max_length", "freq", "seed", "engine", "exogenous", "overlays"}
)
# Emitted values must stay finite and below this magnitude
_MAX_ABS = 1e8
# Variances below this on the calibration window count as degenerate
_MIN_STD = 1e-8
# Softness range of saturating edges, log-uniform, on the standardized scale
_SOFTNESS_RANGE = (0.5, 4.0)
# Threshold range of threshold/hinge/gate edges, on the standardized scale
_THRESHOLD_RANGE = (-1.0, 1.0)
# Relative magnitude of a sampled term coefficient before strength calibration
_COEF_RANGE = (0.5, 1.0)
# Lag sampling: P(lag = 0), then P(1 <= lag <= 8), the rest up to max_lag
_P_LAG_ZERO = 0.5
_P_LAG_SHORT = 0.35
_SHORT_LAG = 8
# Companion spectral radius random VAR lag matrices are rescaled to
_VAR_STABILITY = 0.95


class GeneratorSpec(BaseModel):
    """A generator class plus explicit parameters, built fresh for every draw.

    The prior instantiates the generator with its own child seed each time it
    needs a series, so generators that randomize parameters in their
    constructor (for example ``SARIMAGenerator`` without ``ar_params``) draw
    fresh parameters per series, and the wrapped generator's own seed never
    affects the result. Overlays and exogenous settings are not allowed: the
    spec describes latent dynamics only.

    Args:
        generator (type[BaseGenerator]): Generator class.
        params (dict[str, Any]): Constructor keyword arguments other than
            ``min_length``, ``max_length``, ``freq`` and ``seed`` (default:
            none).
        positive (bool): The generator's output is non-negative, which allows
            multiplicative composition when the spec is a target (default:
            False).

    Example:
        >>> from synforecast.generators import SARIMAGenerator
        >>> spec = GeneratorSpec(generator=SARIMAGenerator, params={"p": 1, "q": 0})
        >>> spec.draw(length=64, freq="D", seed=0).shape
        (64,)
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    generator: type[BaseGenerator]
    params: dict[str, Any] = Field(default_factory=dict)
    positive: bool = False

    @field_validator("params")
    @classmethod
    def _check_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        reserved = sorted(_RESERVED_PARAMS & set(params))
        if reserved:
            raise ValueError(f"params must not set {reserved}; the prior owns them")
        # NumPy arrays and scalars become lists and Python scalars so specs
        # serialize into recipes and fingerprints.
        params = _jsonable(params)
        try:
            json.dumps(params)
        except TypeError as exc:
            raise ValueError(f"params must be JSON-serializable: {exc}") from exc
        return params

    @field_serializer("generator")
    def _serialize_generator(self, generator: type[BaseGenerator]) -> str:
        return f"{generator.__module__}.{generator.__qualname__}"

    @property
    def name(self) -> str:
        """Name of the generator class."""
        return self.generator.__name__

    def draw(self, length: int, freq: str | int, seed: int) -> np.ndarray:
        """Draw one series of ``length`` values with a freshly built generator."""
        gen = self.generator(
            min_length=length, max_length=length, freq=freq, seed=seed, **self.params
        )
        return np.asarray(gen.generate_single_series(length), dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        """JSON-serializable description used in recipes."""
        return self.model_dump(mode="json")


class Term(BaseModel):
    """One additive term of the covariate effect.

    Main edges apply ``coef * f(x[t - lag])`` to one driver ``x``; ``param`` is
    the softness ``s`` of ``saturating`` (``s * tanh(x / s)``) or the threshold
    ``tau`` of ``threshold`` (``1[x > tau]``) and ``hinge``
    (``max(x - tau, 0)``). Interactions combine two drivers: ``product`` is
    ``coef * x[t - lag] * w[t - partner_lag]`` and ``gate`` is
    ``coef * x[t - lag] * 1[w[t - partner_lag] > tau]``.

    Args:
        kind (str): Edge or interaction kind.
        driver (int): Index of the driver, in the mechanism's driver order.
        coef (float): Coefficient.
        lag (int): Lag of the driver in time steps (default: 0).
        param (float | None): Softness or threshold, where the kind needs one.
        partner (int | None): Second driver of an interaction.
        partner_lag (int): Lag of the second driver (default: 0).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[
        "linear",
        "saturating",
        "quadratic",
        "signed_sqrt",
        "abs_sqrt",
        "threshold",
        "hinge",
        "product",
        "gate",
    ]
    driver: int = Field(ge=0)
    coef: float
    lag: int = Field(default=0, ge=0)
    param: float | None = None
    partner: int | None = Field(default=None, ge=0)
    partner_lag: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Term:
        interaction = self.kind in _INTERACTION_KINDS
        if interaction != (self.partner is not None):
            raise ValueError("partner is required for, and only for, interactions")
        if self.kind == "saturating" and self.param is not None and self.param <= 0:
            raise ValueError("saturating softness must be positive")
        return self

    @property
    def max_lag(self) -> int:
        """Largest lag the term reads."""
        return max(self.lag, self.partner_lag)

    def drivers(self) -> tuple[int, ...]:
        """Indices of the drivers the term reads."""
        return (self.driver,) if self.partner is None else (self.driver, self.partner)


class RoleSpec(BaseModel):
    """Input availability of covariates.

    Each covariate is independently known in the future with probability
    ``p_known_future`` and observed only in the past otherwise. Roles are drawn
    the same way for informative covariates and distractors, so a role says
    nothing about informativeness.

    Args:
        p_known_future (float): Probability that a covariate is known in the
            future (default: 0.5).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    p_known_future: float = Field(default=0.5, ge=0.0, le=1.0)


def default_driver_pool() -> list[GeneratorSpec]:
    """Generator pool for covariates: smooth, persistent, seasonal, regime and
    count-like drivers. Informative covariates and distractors share it."""
    from synforecast import generators as g

    return [
        GeneratorSpec(generator=g.SARIMAGenerator),
        GeneratorSpec(generator=g.OrnsteinUhlenbeckGenerator),
        GeneratorSpec(generator=g.RandomWalkGenerator),
        GeneratorSpec(generator=g.SeasonalGenerator),
        GeneratorSpec(generator=g.GaussianProcessGenerator),
        GeneratorSpec(generator=g.KernelSynthGenerator),
        GeneratorSpec(generator=g.RegimeSwitchingGenerator),
        GeneratorSpec(generator=g.TSIGenerator),
        GeneratorSpec(generator=g.IntermittentDemandGenerator, positive=True),
        GeneratorSpec(generator=g.PoissonProcessGenerator, positive=True),
        GeneratorSpec(generator=g.EnergyLoadGenerator, positive=True),
    ]


def default_target_pool() -> list[GeneratorSpec]:
    """Generator pool for the target's own dynamics."""
    from synforecast import generators as g

    return [
        GeneratorSpec(generator=g.SARIMAGenerator),
        GeneratorSpec(generator=g.ETSGenerator),
        GeneratorSpec(generator=g.SeasonalGenerator),
        GeneratorSpec(generator=g.RandomWalkGenerator),
        GeneratorSpec(generator=g.OrnsteinUhlenbeckGenerator),
        GeneratorSpec(generator=g.KernelSynthGenerator),
        GeneratorSpec(generator=g.TSIGenerator),
        GeneratorSpec(generator=g.IntermittentDemandGenerator, positive=True),
        GeneratorSpec(generator=g.PoissonProcessGenerator, positive=True),
        GeneratorSpec(generator=g.DailyActiveUsersGenerator, positive=True),
        GeneratorSpec(generator=g.EnergyLoadGenerator, positive=True),
    ]


def _check_range(
    name: str, value: tuple[float, float], low: float, high: float
) -> None:
    lo, hi = value
    if not (low <= lo <= hi <= high):
        raise ValueError(f"{name} must satisfy {low} <= low <= high <= {high}")


class RegressionSpec(BaseModel):
    """Random structural regression of one target on sampled covariates.

    Per group: draw the covariate count, decide whether the group is null (no
    covariate carries information), draw informative covariates and
    distractors from the same ``driver_pool``, sample one main edge per
    informative covariate plus an optional interaction, optional causal
    distributed lags (adstock) and effect persistence, and compose the effect
    with a target base from ``target_pool``. See :class:`GroupPrior` for the
    calibration of the effect strength.

    Args:
        target_pool (list[GeneratorSpec]): Target base dynamics.
        driver_pool (list[GeneratorSpec]): Covariate dynamics.
        n_covariates (tuple[int, int]): Inclusive range of emitted covariates
            (default: (1, 6)).
        n_informative (tuple[int, int]): Inclusive range of informative
            covariates in a non-null group, capped at the covariate count
            (default: (1, 3)).
        p_null (float): Probability that every covariate is a distractor
            (default: 0.15).
        edge_kinds (list[str]): Main edge kinds (default: all).
        max_lag (int): Largest edge lag (default: 64).
        p_interaction (float): Probability of one product or gate interaction
            when at least two covariates are informative (default: 0.25).
        p_adstock (float): Per-term probability of a geometric distributed lag
            (default: 0.25).
        adstock_decay (tuple[float, float]): Decay range of the distributed
            lag (default: (0.3, 0.9)).
        p_persistence (float): Probability of AR(1) persistence of the summed
            effect (default: 0.2).
        persistence_phi (tuple[float, float]): AR coefficient range of the
            persistence (default: (0.1, 0.9)).
        p_target_ar (float): Probability of AR(1) feedback of the target
            (default: 0.1).
        target_ar_phi (tuple[float, float]): AR coefficient range of the
            target feedback (default: (0.1, 0.9)).
        effect_share (tuple[float, float]): Range of the additive effect's
            variance share ``Var(effect) / (Var(effect) + Var(base))`` on the
            calibration window (default: (0.1, 0.9)).
        p_multiplicative (float): Probability of multiplicative composition
            ``base * exp(effect)`` for positive target specs (default: 0.3).
        log_effect_std (tuple[float, float]): Range of the effect's standard
            deviation in log space under multiplicative composition, on the
            calibration window (default: (0.05, 0.7)).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["regression"] = "regression"
    target_pool: list[GeneratorSpec] = Field(default_factory=default_target_pool)
    driver_pool: list[GeneratorSpec] = Field(default_factory=default_driver_pool)
    n_covariates: tuple[int, int] = (1, 6)
    n_informative: tuple[int, int] = (1, 3)
    p_null: float = Field(default=0.15, ge=0.0, le=1.0)
    edge_kinds: list[EdgeKind] = Field(default_factory=lambda: list(_EDGE_KINDS))
    max_lag: int = Field(default=64, ge=0)
    p_interaction: float = Field(default=0.25, ge=0.0, le=1.0)
    p_adstock: float = Field(default=0.25, ge=0.0, le=1.0)
    adstock_decay: tuple[float, float] = (0.3, 0.9)
    p_persistence: float = Field(default=0.2, ge=0.0, le=1.0)
    persistence_phi: tuple[float, float] = (0.1, 0.9)
    p_target_ar: float = Field(default=0.1, ge=0.0, le=1.0)
    target_ar_phi: tuple[float, float] = (0.1, 0.9)
    effect_share: tuple[float, float] = (0.1, 0.9)
    p_multiplicative: float = Field(default=0.3, ge=0.0, le=1.0)
    log_effect_std: tuple[float, float] = (0.05, 0.7)

    @model_validator(mode="after")
    def _check(self) -> RegressionSpec:
        if not self.target_pool or not self.driver_pool:
            raise ValueError("target_pool and driver_pool must not be empty")
        if not self.edge_kinds:
            raise ValueError("edge_kinds must not be empty")
        c_lo, c_hi = self.n_covariates
        if not 0 <= c_lo <= c_hi:
            raise ValueError("n_covariates must satisfy 0 <= low <= high")
        i_lo, i_hi = self.n_informative
        if not 1 <= i_lo <= i_hi:
            raise ValueError("n_informative must satisfy 1 <= low <= high")
        _check_range("adstock_decay", self.adstock_decay, 0.0, 0.99)
        _check_range("persistence_phi", self.persistence_phi, 0.0, 0.99)
        _check_range("target_ar_phi", self.target_ar_phi, 0.0, 0.99)
        _check_range("effect_share", self.effect_share, 0.0, 0.99)
        lo, hi = self.log_effect_std
        if not 0 < lo <= hi:
            raise ValueError("log_effect_std must satisfy 0 < low <= high")
        return self

    @property
    def largest_lag(self) -> int:
        """Largest lag any term can read."""
        return self.max_lag


class FixedRegressionSpec(BaseModel):
    """A fully specified regression: explicit drivers, terms and noise.

    The target is

        y[t] = target_ar * y[t-1] + intercept + base[t] + sum(terms)[t] + noise[t]

    on the raw (unstandardized) driver values, with ``base = 0`` when
    ``target`` is None and Gaussian ``noise`` of std ``noise_std``. Used for
    probes and controlled experiments. ``n_distractors`` extra covariates are
    drawn from the driver specs in turn and enter no term. All covariates get
    the same ``role``.

    Args:
        drivers (list[GeneratorSpec]): Informative drivers; terms index them.
        terms (list[Term]): Effect terms.
        intercept (float): Constant (default: 0).
        noise_std (float): Std of additive Gaussian noise (default: 0).
        target_ar (float): Target AR(1) coefficient in [0, 1) (default: 0).
        target (GeneratorSpec | None): Target base dynamics, or None for a
            zero base (default: None).
        n_distractors (int): Uninformative covariates (default: 0).
        role (str): Role of every covariate, 'future' or 'past'
            (default: 'future').
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["fixed"] = "fixed"
    drivers: list[GeneratorSpec]
    terms: list[Term]
    intercept: float = 0.0
    noise_std: float = Field(default=0.0, ge=0.0)
    target_ar: float = Field(default=0.0, ge=0.0, lt=1.0)
    target: GeneratorSpec | None = None
    n_distractors: int = Field(default=0, ge=0)
    role: Role = "future"

    @model_validator(mode="after")
    def _check(self) -> FixedRegressionSpec:
        if not self.drivers:
            raise ValueError("drivers must not be empty")
        for term in self.terms:
            if max(term.drivers()) >= len(self.drivers):
                raise ValueError("a term references a driver that does not exist")
        return self

    @property
    def largest_lag(self) -> int:
        """Largest lag any term reads."""
        return max((t.max_lag for t in self.terms), default=0)


class CoupledSpec(BaseModel):
    """Related channels from one base generator, coupled cotemporaneously
    and by lead-lag (a Chronos-2-style multivariatizer).

    Per group, one spec from ``base_pool`` draws every channel. Channels are
    standardized with calibration-window statistics, coupled, then restored
    to each channel's calibration-window level and scale:

    - ``"mixing"``: channels become linear combinations ``L @ z`` where ``L``
      is the Cholesky factor of a random correlation target
      ``C = (1 - s) I + s Q`` (``Q`` the correlation matrix of a random
      Gaussian Gram matrix, ``s`` the mixing strength), so cross-correlations
      are of size ``s``.
    - ``"leadlag"``: with probability ``p_leadlag`` (at least once), a
      channel becomes ``sign * z_src[t - lag] + sigma * eps`` of an earlier
      channel. Lags read the burn-in, so nothing wraps around; the prior's
      ``burn_in`` must exceed the longest possible chain of lags,
      ``(n_channels[1] - 1) * lag_range[1]``.

    With ``targets='all'`` every channel is a target (multivariate
    forecasting). With ``targets='one'`` one random channel is the target
    and the others are covariates whose roles follow the prior's
    :class:`RoleSpec`. Coupling makes channels related, not causally
    informative: a recipe makes no claim that a covariate helps.

    The recipe follows Chronos-2 (Ansari et al., 2025,
    https://arxiv.org/abs/2510.15821); the mixing construction, lag ranges
    and calibration are SynForecast's own design rather than a reproduction
    of the reference code.

    Args:
        base_pool (list[GeneratorSpec]): Base dynamics, one spec per group
            (default: the default target pool).
        n_channels (tuple[int, int]): Inclusive range of channels
            (default: (2, 6)).
        couplings (list[str]): Subset of ['mixing', 'leadlag'], applied in
            that order (default: both).
        mixing_strength (tuple[float, float]): Range of the mixing strength in
            [0, 1) (default: (0.2, 0.9)).
        lag_range (tuple[int, int]): Inclusive range of lead-lag offsets
            (default: (1, 24)).
        noise_scale (tuple[float, float]): Range of the lead-lag noise std,
            relative to unit-variance channels (default: (0.02, 0.2)).
        p_leadlag (float): Probability that a non-first channel becomes a
            lead-lag copy (default: 0.5).
        targets (str): 'all' or 'one' (default: 'all').
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["coupled"] = "coupled"
    base_pool: list[GeneratorSpec] = Field(default_factory=default_target_pool)
    n_channels: tuple[int, int] = (2, 6)
    couplings: list[Literal["mixing", "leadlag"]] = Field(
        default_factory=lambda: ["mixing", "leadlag"]
    )
    mixing_strength: tuple[float, float] = (0.2, 0.9)
    lag_range: tuple[int, int] = (1, 24)
    noise_scale: tuple[float, float] = (0.02, 0.2)
    p_leadlag: float = Field(default=0.5, ge=0.0, le=1.0)
    targets: Literal["all", "one"] = "all"

    @model_validator(mode="after")
    def _check(self) -> CoupledSpec:
        if not self.base_pool:
            raise ValueError("base_pool must not be empty")
        if not self.couplings:
            raise ValueError("couplings must not be empty")
        lo, hi = self.n_channels
        if not 1 <= lo <= hi:
            raise ValueError("n_channels must satisfy 1 <= low <= high")
        _check_range("mixing_strength", self.mixing_strength, 0.0, 0.99)
        lag_lo, lag_hi = self.lag_range
        if not 1 <= lag_lo <= lag_hi:
            raise ValueError("lag_range must satisfy 1 <= low <= high")
        n_lo, n_hi = self.noise_scale
        if not 0 < n_lo <= n_hi:
            raise ValueError("noise_scale must satisfy 0 < low <= high")
        return self

    @property
    def largest_lag(self) -> int:
        """Largest accumulated lead-lag offset: a lead-lag source can itself be
        a lead-lag copy, so chains reach ``(n_channels - 1) * max lag``."""
        if "leadlag" not in self.couplings:
            return 0
        return (self.n_channels[1] - 1) * self.lag_range[1]


class VARXSpec(BaseModel):
    """A linear VAR panel of several targets driven by regressors with known
    effects, for benchmarking forecasters that use regressors.

    With ``N`` targets, ``p`` lags and ``S`` shared, ``F`` future-known and
    ``H`` past-only regressors::

        y[t] = intercept + sum_l A_l @ y[t - 1 - l]
               + B_shared @ shared[t]
               + sum_k B_future[k] @ future[t, :, k]
               + sum_k B_past[k] @ past[t - 1, :, k] + e[t]

    Every regressor is a unit-variance AR(1) with coefficient
    ``exog_persistence``. Shared regressors (one channel each, role
    'future') move every target, such as a promotion calendar; future-known
    regressors (one channel per target, role 'future') move their own target
    and, with ``exog_cross_scale > 0``, the others, such as a plan or price;
    past-only regressors (one channel per target, role 'past') move it one
    period later, such as an order book. ``e`` has covariance
    ``innovation_covariance`` (default: identity).

    Lag matrices are random with ``cross_structure`` ``'dense'``,
    ``'sparse'`` (each off-diagonal entry nonzero with probability
    ``sparsity``) or ``'clusters'`` (each target responds to its own lag and
    to the lagged mean of each of ``n_clusters`` contiguous clusters), and
    are rescaled to a stable process; ``coef_matrices`` overrides them.
    :meth:`VARXTruth.from_recipe` rebuilds the process of a group and its
    oracle forecast.

    Args:
        n_targets (tuple[int, int]): Inclusive range of targets
            (default: (2, 6)).
        lag_order (int): VAR lag order (default: 1).
        n_shared (int): Shared regressors (default: 0).
        n_future (int): Future-known regressors per target (default: 1).
        n_past (int): Past-only regressors per target (default: 0).
        exog_persistence (float): AR(1) coefficient of every regressor in
            [0, 1) (default: 0.5).
        exog_scale (float): Std of the shared effects and of a regressor's
            effect on its own target (default: 1.0).
        exog_cross_scale (float): Std of the total effect of the other
            targets' regressors on one target (default: 0.0).
        cross_structure (str): 'dense', 'sparse' or 'clusters'
            (default: 'dense').
        sparsity (float): Nonzero share of off-diagonal lags under 'sparse'
            (default: 0.1).
        n_clusters (int | None): Contiguous target clusters; required for
            'clusters' (default: None).
        coef_matrices (list | None): Fixed lag matrices, one per lag; fixes
            the target count to their size (default: None).
        intercept (list[float] | None): Intercept (default: zeros).
        innovation_covariance (list | None): Innovation covariance
            (default: identity).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["varx"] = "varx"
    n_targets: tuple[int, int] = (2, 6)
    lag_order: int = Field(default=1, ge=1)
    n_shared: int = Field(default=0, ge=0)
    n_future: int = Field(default=1, ge=0)
    n_past: int = Field(default=0, ge=0)
    exog_persistence: float = Field(default=0.5, ge=0.0, lt=1.0)
    exog_scale: float = Field(default=1.0, ge=0.0)
    exog_cross_scale: float = Field(default=0.0, ge=0.0)
    cross_structure: Literal["dense", "sparse", "clusters"] = "dense"
    sparsity: float = Field(default=0.1, gt=0.0, le=1.0)
    n_clusters: int | None = Field(default=None, ge=1)
    coef_matrices: list[list[list[float]]] | None = None
    intercept: list[float] | None = None
    innovation_covariance: list[list[float]] | None = None

    @model_validator(mode="after")
    def _check(self) -> VARXSpec:
        lo, hi = self.n_targets
        if not 1 <= lo <= hi:
            raise ValueError("n_targets must satisfy 1 <= low <= high")
        if self.cross_structure == "clusters" and self.n_clusters is None:
            raise ValueError("cross_structure='clusters' requires n_clusters")
        if self.n_clusters is not None and self.n_clusters > lo:
            raise ValueError("n_clusters exceeds the smallest target count")
        fixed = {
            "coef_matrices": None
            if not self.coef_matrices
            else len(self.coef_matrices[0]),
            "intercept": None if self.intercept is None else len(self.intercept),
            "innovation_covariance": None
            if self.innovation_covariance is None
            else len(self.innovation_covariance),
        }
        for name, size in fixed.items():
            if size is not None and (lo, hi) != (size, size):
                raise ValueError(f"{name} fixes n_targets to ({size}, {size})")
        if self.coef_matrices is not None:
            mats = np.asarray(self.coef_matrices, dtype=np.float64)
            n = lo
            if mats.shape != (self.lag_order, n, n):
                raise ValueError(
                    f"coef_matrices must have shape (lag_order, N, N) = "
                    f"({self.lag_order}, {n}, {n}), got {mats.shape}"
                )
            if not np.isfinite(mats).all():
                raise ValueError("coef_matrices must be finite")
            if _spectral_radius(list(mats)) >= 1.0:
                raise ValueError("coef_matrices define an unstable VAR")
        if self.intercept is not None and not np.isfinite(self.intercept).all():
            raise ValueError("intercept must be finite")
        if self.innovation_covariance is not None:
            cov = np.asarray(self.innovation_covariance, dtype=np.float64)
            n = lo
            if cov.shape != (n, n) or not np.isfinite(cov).all():
                raise ValueError(
                    f"innovation_covariance must be a finite ({n}, {n}) matrix"
                )
            if not np.allclose(cov, cov.T):
                raise ValueError("innovation_covariance must be symmetric")
            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "innovation_covariance must be positive definite"
                ) from exc
        return self

    @property
    def largest_lag(self) -> int:
        """Largest lag of the process."""
        return self.lag_order


# ---------------------------------------------------------------------------
# Causal kernels. Every output value at time t depends only on inputs <= t.
# ---------------------------------------------------------------------------


def _lagged(x: np.ndarray, lag: int) -> np.ndarray:
    """``x[t - lag]``, holding the first value before the start (no wrap)."""
    if lag == 0:
        return x
    out = np.empty_like(x)
    out[:lag] = x[0]
    out[lag:] = x[:-lag]
    return out


def _ar_filter(x: np.ndarray, phi: float) -> np.ndarray:
    """``y[t] = phi * y[t-1] + x[t]`` started at the steady state of ``x[0]``."""
    if phi == 0.0:
        return x.copy()
    out = np.empty_like(x)
    prev = x[0] / (1.0 - phi)
    for t, value in enumerate(x):
        prev = phi * prev + value
        out[t] = prev
    return out


def _edge(kind: str, x: np.ndarray, param: float | None) -> np.ndarray:
    if kind == "linear":
        return x
    if kind == "saturating":
        s = 1.0 if param is None else param
        return s * np.tanh(x / s)
    if kind == "quadratic":
        return x * x
    if kind == "signed_sqrt":
        return np.sign(x) * np.sqrt(np.abs(x))
    if kind == "abs_sqrt":
        return np.sqrt(np.abs(x))
    tau = 0.0 if param is None else param
    if kind == "threshold":
        return (x > tau).astype(np.float64)
    if kind == "hinge":
        return np.maximum(x - tau, 0.0)
    raise ValueError(f"unknown edge kind {kind!r}")


def _term_values(term: Term, z: np.ndarray) -> np.ndarray:
    x = _lagged(z[term.driver], term.lag)
    if term.kind == "product":
        return term.coef * x * _lagged(z[term.partner], term.partner_lag)
    if term.kind == "gate":
        tau = 0.0 if term.param is None else term.param
        gate = _lagged(z[term.partner], term.partner_lag) > tau
        return term.coef * x * gate
    return term.coef * _edge(term.kind, x, term.param)


def _raw_effect(
    z: np.ndarray,
    terms: list[Term],
    adstock: list[float | None],
    persistence: float,
    length: int,
) -> np.ndarray:
    """Sum of terms, each optionally through a normalized geometric lag, then
    optional AR(1) persistence. Causal by construction."""
    effect = np.zeros(length)
    for term, decay in zip(terms, adstock, strict=True):
        values = _term_values(term, z)
        if decay is not None:
            values = (1.0 - decay) * _ar_filter(values, decay)
        effect += values
    if persistence:
        effect = _ar_filter(effect, persistence)
    return effect


def _compose(
    base: np.ndarray,
    z: np.ndarray,
    terms: list[Term],
    adstock: list[float | None],
    persistence: float,
    target_ar: float,
    burn_in: int,
    composition: str,
    share: float,
    log_std: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compose a target from its base and standardized drivers.

    The effect is centered and scaled with statistics of the first
    ``burn_in`` steps only, so the output at time ``t`` depends on inputs at
    or before ``max(t, burn_in - 1)``; the emitted window starts at
    ``burn_in``. Returns the target, the centered unscaled effect and the
    calibration record.
    """
    effect = _raw_effect(z, terms, adstock, persistence, base.shape[0])
    center = float(effect[:burn_in].mean())
    effect = effect - center
    effect_std = float(effect[:burn_in].std())
    calibration = "calibrated"
    if not terms:
        calibration, effect_std = "null", 1.0
    elif effect_std < _MIN_STD:
        calibration, effect_std = "nominal_effect", 1.0
    if composition == "multiplicative":
        scale = log_std / effect_std
        with np.errstate(over="ignore", invalid="ignore"):
            latent = base * np.exp(scale * effect)
    else:
        base_std = float(base[:burn_in].std())
        if base_std < _MIN_STD:
            if calibration == "calibrated":
                calibration = "nominal_base"
            base_std = 1.0
        scale = float(np.sqrt(share / (1.0 - share)) * base_std / effect_std)
        latent = base + scale * effect
    with np.errstate(over="ignore", invalid="ignore"):
        y = _ar_filter(latent, target_ar)
    info = {"center": center, "scale": float(scale), "calibration": calibration}
    return y, effect, info


def _calibrated(x: np.ndarray, burn_in: int) -> np.ndarray:
    """Standardize rows with mean and std of the first ``burn_in`` steps."""
    mean = x[:, :burn_in].mean(axis=1, keepdims=True)
    std = x[:, :burn_in].std(axis=1, keepdims=True)
    return (x - mean) / np.where(std < _MIN_STD, 1.0, std)


def _unit_ar1(
    shape: tuple[int, ...], phi: float, rng: np.random.Generator
) -> np.ndarray:
    """Unit-variance AR(1) along axis 0, started from its stationary law."""
    shocks = rng.standard_normal(shape)
    out = np.empty(shape)
    if shape[0] == 0:
        return out
    out[0] = shocks[0]
    root = np.sqrt(1.0 - phi**2)
    for t in range(1, shape[0]):
        out[t] = phi * out[t - 1] + root * shocks[t]
    return out


def _varx_lag_matrices(
    spec: VARXSpec,
    n: int,
    clusters: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random stable lag matrices of the configured structure."""
    mats = []
    for lag in range(spec.lag_order):
        if spec.cross_structure == "dense":
            mats.append(rng.uniform(-0.3, 0.3, (n, n)) * 0.7**lag)
            continue
        own = np.diag(rng.uniform(0.2, 0.6, n))
        if spec.cross_structure == "sparse":
            mask = rng.random((n, n)) < spec.sparsity
            np.fill_diagonal(mask, False)
            cross = mask * rng.uniform(-0.5, 0.5, (n, n))
        else:
            assert clusters is not None
            sizes = np.bincount(clusters)
            totals = rng.uniform(-0.5, 0.5, (n, len(sizes)))
            cross = (totals / sizes)[:, clusters]
        mats.append((own + cross) * 0.7**lag)
    radius = _spectral_radius(mats)
    if radius >= _VAR_STABILITY:
        # Scaling A_l by s^l maps every companion eigenvalue z to s * z
        s = _VAR_STABILITY / radius
        mats = [mat * s ** (lag + 1) for lag, mat in enumerate(mats)]
    return np.stack(mats)


def _varx_effects(
    spec: VARXSpec, n_regressors: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_regressors, N, N)`` regressor effects: own effects on the
    diagonal, cross effects off it."""
    out = np.zeros((n_regressors, n, n))
    for k in range(n_regressors):
        if spec.exog_cross_scale > 0 and n > 1:
            scale = spec.exog_cross_scale / np.sqrt(n - 1)
            out[k] = rng.normal(0.0, scale, (n, n))
        np.fill_diagonal(out[k], rng.normal(0.0, spec.exog_scale, n))
    return out


def _ok(values: np.ndarray) -> bool:
    return bool(
        np.isfinite(values).all() and np.abs(values).max(initial=0.0) < _MAX_ABS
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Group:
    """One sampled group of aligned channels.

    Attributes:
        group_id (int): Group identifier, unique within a sampling stream.
        timestamps (np.ndarray): Shared timestamps or integer time indices.
        values (np.ndarray): Channel values, shape ``(n_channels, length)``.
        roles (tuple[str, ...]): Role of every channel: 'target', 'future'
            or 'past'.
        recipe (dict): Versioned ground truth of the group, JSON-serializable.
    """

    group_id: int
    timestamps: np.ndarray
    values: np.ndarray
    roles: tuple[str, ...]
    recipe: dict = field(repr=False)

    @property
    def target_indices(self) -> list[int]:
        """Channel indices of the targets."""
        return [i for i, role in enumerate(self.roles) if role == "target"]

    @property
    def target_index(self) -> int:
        """Channel index of the target, for groups with exactly one target."""
        targets = self.target_indices
        if len(targets) != 1:
            raise ValueError(
                f"group has {len(targets)} targets; use target_indices instead"
            )
        return targets[0]

    @property
    def covariate_indices(self) -> list[int]:
        """Channel indices of the covariates."""
        return [i for i, role in enumerate(self.roles) if role != "target"]

    @property
    def length(self) -> int:
        """Number of time steps."""
        return self.values.shape[1]


@dataclass(frozen=True)
class GroupBatch:
    """Groups sampled by one :meth:`GroupPrior.sample` call."""

    groups: tuple[Group, ...]

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[Group]:
        return iter(self.groups)

    def __getitem__(self, index: int) -> Group:
        return self.groups[index]

    @property
    def diagnostics(self) -> dict[str, int]:
        """Counts of null groups, redraws, composition fallbacks and shuffled
        controls (counted on their source groups)."""
        recipes = [g.recipe.get("source", g.recipe) for g in self.groups]
        return {
            "n_groups": len(self.groups),
            "n_null": sum(bool(r.get("null")) for r in recipes),
            "n_redrawn": sum(r["attempt"] > 0 for r in recipes),
            "n_fallback": sum(r.get("fallback") is not None for r in recipes),
            "n_controls": sum("control" in g.recipe for g in self.groups),
        }

    def to_long(
        self,
        engine: Literal["pandas", "polars", "cudf", "modin", "pyarrow"] = "pandas",
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        group_col: str = "group_id",
        role_col: str = "role",
    ) -> Any:
        """Long format: one row per channel and time step.

        Columns are ``group_col``, ``id_col`` (``"{group}_{channel}"``),
        ``time_col``, ``target_col`` and ``role_col``. Ground truth stays in the
        recipes. Serialization consumes no random draws.
        """
        names = [group_col, id_col, time_col, target_col, role_col]
        if len(set(names)) != len(names):
            raise ValueError("column names must be distinct")
        group_ids, ids, times, values, roles = [], [], [], [], []
        for group in self.groups:
            n_channels, length = group.values.shape
            group_ids.append(np.full(n_channels * length, group.group_id, np.int64))
            ids.append(
                np.repeat([f"{group.group_id}_{c}" for c in range(n_channels)], length)
            )
            times.append(np.tile(group.timestamps, n_channels))
            values.append(group.values.reshape(-1))
            roles.append(np.repeat(np.asarray(group.roles), length))
        result = {
            group_col: np.concatenate(group_ids)
            if group_ids
            else np.empty(0, np.int64),
            id_col: np.concatenate(ids) if ids else np.empty(0, str),
            time_col: np.concatenate(times) if times else np.empty(0, "datetime64[ns]"),
            target_col: np.concatenate(values) if values else np.empty(0),
            role_col: np.concatenate(roles) if roles else np.empty(0, str),
        }
        return nw.DataFrame.from_dict(result, backend=engine).to_native()

    def shuffle_covariates(self, seed: int) -> GroupBatch:
        """Control batch: each target keeps its own values but gets the whole
        covariate set (values and roles) of another group.

        Groups are matched within buckets of equal length, target count and
        covariate count; every bucket needs at least two groups. Whole
        covariate sets move together, so relations between covariates are
        preserved while their relation to the targets is broken. Targets keep
        their channel positions; the donor's covariates fill the covariate
        positions in order, with their roles.

        A control group's recipe is a control record: ``source`` holds the
        receiving group's recipe, ``covariates_from`` the donor's group ID and
        ``covariate_sources`` which donor channel fills each covariate
        position. Control groups have no single generating process, so
        :meth:`GroupPrior.replay` and :meth:`VARXTruth.from_recipe` reject
        them; replay the source groups instead.
        """
        rng = np.random.default_rng(seed)
        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i, group in enumerate(self.groups):
            key = (
                group.length,
                len(group.target_indices),
                len(group.covariate_indices),
            )
            buckets[key].append(i)
        donors = [0] * len(self.groups)
        for members in buckets.values():
            if len(members) < 2:
                raise ValueError(
                    "every (length, target count, covariate count) bucket needs "
                    "at least two groups"
                )
            order = rng.permutation(members)
            shift = int(rng.integers(1, len(members)))
            for receiver, donor in zip(order, np.roll(order, shift), strict=True):
                donors[receiver] = int(donor)
        shuffled = []
        for i, group in enumerate(self.groups):
            donor = self.groups[donors[i]]
            values = group.values.copy()
            roles = list(group.roles)
            sources = []
            for dst, src in zip(
                group.covariate_indices, donor.covariate_indices, strict=True
            ):
                values[dst] = donor.values[src]
                roles[dst] = donor.roles[src]
                sources.append({"channel": dst, "donor_channel": src})
            recipe = {
                "version": RECIPE_VERSION,
                "control": "shuffled_covariates",
                "source": group.recipe,
                "covariates_from": donor.group_id,
                "covariate_sources": sources,
            }
            shuffled.append(
                replace(group, values=values, roles=tuple(roles), recipe=recipe)
            )
        return GroupBatch(tuple(shuffled))


def _spectral_radius(mats: list[np.ndarray]) -> float:
    """Spectral radius of the VAR companion matrix."""
    p, k = len(mats), mats[0].shape[0]
    companion = np.zeros((k * p, k * p))
    for i, mat in enumerate(mats):
        companion[:k, i * k : (i + 1) * k] = mat
    if p > 1:
        companion[k:, :-k] = np.eye(k * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


@dataclass(frozen=True)
class VARXTruth:
    """The process behind one :class:`VARXSpec` group.

    Shapes use ``N`` targets, ``p`` lags and ``S``/``F``/``H`` shared,
    future-known and past-only regressors. ``clusters`` holds every target's
    cluster (None without clusters).
    """

    lag_matrices: np.ndarray  # (p, N, N)
    intercept: np.ndarray  # (N,)
    shared_effects: np.ndarray  # (N, S)
    future_effects: np.ndarray  # (F, N, N)
    past_effects: np.ndarray  # (H, N, N)
    innovation_covariance: np.ndarray  # (N, N)
    exog_persistence: float
    clusters: np.ndarray | None  # (N,)

    @classmethod
    def from_recipe(cls, recipe: dict) -> VARXTruth:
        """Rebuild the truth of a VARX group from its recipe."""
        if "control" in recipe:
            raise ValueError(
                "control groups have no single VARX process; use the source "
                "group's recipe"
            )
        if recipe.get("mechanism") != "varx":
            raise ValueError("recipe is not a VARX group")
        process = recipe["process"]
        clusters = process["clusters"]
        return cls(
            lag_matrices=np.asarray(process["lag_matrices"], dtype=np.float64),
            intercept=np.asarray(process["intercept"], dtype=np.float64),
            shared_effects=np.asarray(process["shared_effects"], dtype=np.float64),
            future_effects=np.asarray(process["future_effects"], dtype=np.float64),
            past_effects=np.asarray(process["past_effects"], dtype=np.float64),
            innovation_covariance=np.asarray(
                process["innovation_covariance"], dtype=np.float64
            ),
            exog_persistence=float(process["exog_persistence"]),
            clusters=None if clusters is None else np.asarray(clusters),
        )

    @staticmethod
    def split(group: Group) -> dict[str, np.ndarray]:
        """Channels of a VARX group by kind, in target and regressor order:
        ``y`` ``(T, N)``, ``shared`` ``(T, S)``, ``future`` ``(T, N, F)`` and
        ``past`` ``(T, N, H)``."""
        if group.recipe.get("mechanism") != "varx":
            raise ValueError("split needs an unshuffled VARX group")
        layout = group.recipe["channels"]
        n = sum(c["kind"] == "target" for c in layout)
        counts = {
            kind: max((c["index"] + 1 for c in layout if c["kind"] == kind), default=0)
            for kind in ("shared", "future", "past")
        }
        length = group.length
        out = {
            "y": np.empty((length, n)),
            "shared": np.empty((length, counts["shared"])),
            "future": np.empty((length, n, counts["future"])),
            "past": np.empty((length, n, counts["past"])),
        }
        for c in layout:
            values = group.values[c["channel"]]
            if c["kind"] == "target":
                out["y"][:, c["series"]] = values
            elif c["kind"] == "shared":
                out["shared"][:, c["index"]] = values
            else:
                out[c["kind"]][:, c["series"], c["index"]] = values
        return out

    def forecast(
        self,
        y: np.ndarray,
        shared: np.ndarray,
        future: np.ndarray,
        past: np.ndarray,
        h: int,
    ) -> np.ndarray:
        """Conditional-mean forecast ``(h, N)`` after the last row of ``y``.

        The optimal squared-error forecast of the process, an oracle for
        benchmarking. ``y`` is the history ``(T, N)`` with ``T >= p``;
        ``shared`` ``(h, S)`` and ``future`` ``(h, N, F)`` hold the
        future-known regressors of the forecast periods; ``past`` ``(N, H)``
        holds the past-only regressors at the last observed period, whose
        later values are replaced by their AR(1) expectation.
        """
        p = self.lag_matrices.shape[0]
        if y.shape[0] < p:
            raise ValueError(f"y needs at least {p} rows, got {y.shape[0]}")
        state = list(y[-p:])
        past_now = np.asarray(past, dtype=np.float64)
        out = np.empty((h, y.shape[1]))
        for s in range(h):
            value = self.intercept + self.shared_effects @ shared[s]
            for lag in range(p):
                value = value + self.lag_matrices[lag] @ state[-1 - lag]
            for k in range(self.future_effects.shape[0]):
                value = value + self.future_effects[k] @ future[s, :, k]
            for k in range(self.past_effects.shape[0]):
                value = value + self.past_effects[k] @ past_now[:, k]
            past_now = past_now * self.exog_persistence
            out[s] = value
            state.append(value)
        return out

    def forecast_group(self, group: Group, origin: int, h: int) -> np.ndarray:
        """Oracle forecast ``(h, N)`` of a VARX group from ``origin``: the
        targets before ``origin`` are history and the future-known regressors
        of ``origin, ..., origin + h - 1`` are known."""
        if not 1 <= origin <= group.length - h:
            raise ValueError("origin must leave h periods inside the group")
        parts = self.split(group)
        return self.forecast(
            parts["y"][:origin],
            parts["shared"][origin : origin + h],
            parts["future"][origin : origin + h],
            parts["past"][origin - 1],
            h,
        )


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------


class GroupPrior(BaseModel):
    """Sample groups of aligned target and covariate channels.

    Every group is generated over ``burn_in + length`` steps. The first
    ``burn_in`` steps are a calibration and warm-up window: lags, distributed
    lags and AR filters read it, the standardization of drivers and the
    strength of the effect are fixed on it, and it is then discarded. No
    statistic of the emitted window feeds back into the generated values, so
    the target at time ``t`` depends only on inputs at or before ``t``.

    Effect strength (``RegressionSpec``): under additive composition the
    centered effect is scaled so that its variance on the calibration window
    is ``share / (1 - share)`` times the base variance there; under
    multiplicative composition (positive target specs only) its log-space std
    on the calibration window is set directly. If the effect or the base is
    constant on the calibration window, a unit nominal scale is used and the
    recipe records it. Realized shares on the emitted window are recorded
    next to the requested ones.

    Channels are shuffled after construction, so channel position carries no
    information. Groups whose emitted values are not finite or exceed 1e8 in
    magnitude are redrawn with a new attempt seed (up to ``max_redraws``); a
    failing multiplicative composition first falls back to additive, which
    the recipe records. Persistent failure raises.

    Seeding: group ``i`` of ``sample(n, seed, start_group)`` depends only on
    ``(seed, start_group + i)`` and the prior's configuration, so groups can
    be replayed individually with :meth:`replay`.

    Args:
        mechanism (RegressionSpec | FixedRegressionSpec | CoupledSpec | VARXSpec):
            How the channels are generated and related
            (default: ``RegressionSpec()``).
        roles (RoleSpec): Role sampling for ``RegressionSpec`` covariates and
            ``CoupledSpec(targets='one')`` covariates (default: ``RoleSpec()``).
        length (tuple[int, int]): Inclusive range of emitted lengths
            (default: (256, 1024)).
        freq (str | int): Pandas offset alias or integer time step
            (default: 'h').
        start_datetime (str): First timestamp (default: '2000-01-01').
        burn_in (int): Calibration and warm-up steps, at least the largest
            lag plus one and at least 16 (default: 256).
        max_redraws (int): Redraws allowed per group (default: 5).

    Example:
        >>> prior = GroupPrior(length=(128, 128), freq="D")
        >>> batch = prior.sample(n_groups=4, seed=0)
        >>> df = batch.to_long()
        >>> sorted(df["role"].unique())[-1]
        'target'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism: RegressionSpec | FixedRegressionSpec | CoupledSpec | VARXSpec = Field(
        default_factory=RegressionSpec, discriminator="kind"
    )
    roles: RoleSpec = Field(default_factory=RoleSpec)
    length: tuple[int, int] = (256, 1024)
    freq: str | int = "h"
    start_datetime: str = "2000-01-01"
    burn_in: int = Field(default=256, ge=16)
    max_redraws: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def _check(self) -> GroupPrior:
        lo, hi = self.length
        if not 2 <= lo <= hi:
            raise ValueError("length must satisfy 2 <= low <= high")
        if self.burn_in <= self.mechanism.largest_lag:
            raise ValueError("burn_in must exceed the largest lag")
        return self

    @property
    def fingerprint(self) -> str:
        """Hash of the configuration, stored in every recipe."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _timestamps(self, length: int) -> np.ndarray:
        if isinstance(self.freq, int):
            return np.arange(length, dtype=np.int64) * self.freq
        return pd.date_range(
            pd.Timestamp(self.start_datetime), periods=length, freq=self.freq
        ).values

    def sample(self, n_groups: int, seed: int, start_group: int = 0) -> GroupBatch:
        """Sample ``n_groups`` groups with IDs ``start_group, start_group + 1, ...``."""
        if n_groups < 0:
            raise ValueError("n_groups must be >= 0")
        fingerprint = self.fingerprint
        return GroupBatch(
            tuple(
                self._sample_group(seed, start_group + i, fingerprint)
                for i in range(n_groups)
            )
        )

    def replay(self, recipe: dict) -> Group:
        """Rebuild the group a recipe describes; the prior must match it."""
        if "control" in recipe:
            raise ValueError(
                "control groups cannot be replayed; replay recipe['source'] and "
                "the donor group instead"
            )
        if recipe.get("version") != RECIPE_VERSION:
            raise ValueError("recipe version does not match this SynForecast")
        if recipe.get("prior") != self.fingerprint:
            raise ValueError("recipe was produced by a different prior configuration")
        return self._sample_group(recipe["seed"], recipe["group_id"], recipe["prior"])

    def _sample_group(self, seed: int, group_id: int, fingerprint: str) -> Group:
        failures: list[str] = []
        for attempt in range(self.max_redraws + 1):
            rng = np.random.default_rng([seed, group_id, attempt])
            result = getattr(self, f"_draw_{self.mechanism.kind}")(rng)
            if isinstance(result, str):
                failures.append(result)
                continue
            channels, roles, recipe = result
            recipe.update(
                version=RECIPE_VERSION,
                prior=fingerprint,
                seed=seed,
                group_id=group_id,
                attempt=attempt,
                failures=failures,
            )
            length = channels.shape[1]
            return Group(
                group_id=group_id,
                timestamps=self._timestamps(length),
                values=channels,
                roles=roles,
                recipe=_jsonable(recipe),
            )
        raise RuntimeError(
            f"group {group_id} failed after {self.max_redraws + 1} attempts: {failures}"
        )

    # -- shared helpers -----------------------------------------------------

    def _draw_series(
        self, specs: list[GeneratorSpec], total: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[int]]:
        seeds = [int(s) for s in rng.integers(0, 2**63, size=len(specs))]
        if not specs:
            return np.empty((0, total)), seeds
        return np.stack(
            [
                spec.draw(total, self.freq, s)
                for spec, s in zip(specs, seeds, strict=True)
            ]
        ), seeds

    def _emit(
        self,
        latent: np.ndarray,
        roles: list[str],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, tuple[str, ...], list[int]]:
        """Drop the burn-in and put the channels in a random order.

        Returns the emitted channels, their roles and, for every latent
        channel, its emitted position.
        """
        order = rng.permutation(len(roles))
        channels = latent[order, self.burn_in :]
        position = np.empty(len(order), dtype=np.int64)
        position[order] = np.arange(len(order))
        return channels, tuple(roles[i] for i in order), position.tolist()

    # -- mechanisms ---------------------------------------------------------

    def _draw_regression(self, rng: np.random.Generator):
        spec: RegressionSpec = self.mechanism  # type: ignore[assignment]
        b = self.burn_in
        length = int(rng.integers(self.length[0], self.length[1] + 1))
        total = b + length

        n_cov = int(rng.integers(spec.n_covariates[0], spec.n_covariates[1] + 1))
        is_null = bool(rng.random() < spec.p_null)
        n_inf = 0
        if not is_null and n_cov > 0:
            n_inf = min(
                n_cov,
                int(rng.integers(spec.n_informative[0], spec.n_informative[1] + 1)),
            )
        target_spec = spec.target_pool[int(rng.integers(len(spec.target_pool)))]
        driver_specs = [
            spec.driver_pool[int(rng.integers(len(spec.driver_pool)))]
            for _ in range(n_cov)
        ]
        roles = [
            "future" if rng.random() < self.roles.p_known_future else "past"
            for _ in range(n_cov)
        ]

        base_seed = int(rng.integers(0, 2**63))
        base = target_spec.draw(total, self.freq, base_seed)
        raw, driver_seeds = self._draw_series(driver_specs, total, rng)
        if not (_ok(base) and _ok(raw)):
            return "latent series not finite or too large"

        cal_mean = raw[:, :b].mean(axis=1) if n_cov else np.empty(0)
        cal_std = raw[:, :b].std(axis=1) if n_cov else np.empty(0)
        cal_std = np.where(cal_std < _MIN_STD, 1.0, cal_std)
        z = (raw - cal_mean[:, None]) / cal_std[:, None]

        terms: list[Term] = []
        for j in range(n_inf):
            terms.append(self._draw_edge(spec, j, rng))
        if n_inf >= 2 and rng.random() < spec.p_interaction:
            a, w = (int(v) for v in rng.choice(n_inf, size=2, replace=False))
            kind = str(rng.choice(_INTERACTION_KINDS))
            terms.append(
                Term(
                    kind=kind,
                    driver=a,
                    partner=w,
                    coef=self._draw_coef(rng),
                    lag=self._draw_lag(spec.max_lag, rng),
                    partner_lag=self._draw_lag(spec.max_lag, rng),
                    param=float(rng.uniform(*_THRESHOLD_RANGE))
                    if kind == "gate"
                    else None,
                )
            )
        adstock = [
            float(rng.uniform(*spec.adstock_decay))
            if rng.random() < spec.p_adstock
            else None
            for _ in terms
        ]
        persistence = (
            float(rng.uniform(*spec.persistence_phi))
            if terms and rng.random() < spec.p_persistence
            else 0.0
        )
        target_ar = (
            float(rng.uniform(*spec.target_ar_phi))
            if rng.random() < spec.p_target_ar
            else 0.0
        )
        multiplicative = bool(
            target_spec.positive and rng.random() < spec.p_multiplicative
        )
        share = float(rng.uniform(*spec.effect_share))
        log_std = float(rng.uniform(*spec.log_effect_std))

        fallback = None
        y = None
        compositions = (
            ["multiplicative", "additive"] if multiplicative else ["additive"]
        )
        for composition in compositions:
            y, effect, info = _compose(
                base,
                z,
                terms,
                adstock,
                persistence,
                target_ar,
                b,
                composition,
                share,
                log_std,
            )
            if _ok(y[b:]):
                break
            y = None
        if y is None:
            return "composed target not finite or too large"
        if multiplicative and composition == "additive":
            fallback = "additive"
        scale = info["scale"]
        center = info["center"]
        calibration = info["calibration"]

        scaled = scale * effect[b:]
        if composition == "additive":
            ve, vb = float(scaled.var()), float(base[b:].var())
            realized = ve / (ve + vb) if ve + vb > 0 else 0.0
        else:
            realized = float(scaled.std())

        channels, out_roles, position = self._emit(
            np.vstack([y[None], raw]), ["target", *roles], rng
        )
        recipe = {
            "mechanism": "regression",
            "null": n_inf == 0,
            "length": length,
            "burn_in": b,
            "target": {
                "channel": position[0],
                "spec": target_spec.describe(),
                "seed": base_seed,
            },
            "covariates": [
                {
                    "channel": position[1 + j],
                    "role": roles[j],
                    "informative": j < n_inf,
                    "spec": driver_specs[j].describe(),
                    "seed": driver_seeds[j],
                    "calibration_mean": float(cal_mean[j]),
                    "calibration_std": float(cal_std[j]),
                }
                for j in range(n_cov)
            ],
            "terms": [t.model_dump() for t in terms],
            "adstock": adstock,
            "persistence": persistence,
            "target_ar": target_ar,
            "composition": composition,
            "fallback": fallback,
            "calibration": calibration,
            "effect_center": center,
            "effect_scale": scale,
            "requested_share": share if composition == "additive" else None,
            "requested_log_std": log_std if composition == "multiplicative" else None,
            "realized": realized,
        }
        return channels, out_roles, recipe

    @staticmethod
    def _draw_coef(rng: np.random.Generator) -> float:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        return sign * float(rng.uniform(*_COEF_RANGE))

    @staticmethod
    def _draw_lag(max_lag: int, rng: np.random.Generator) -> int:
        if max_lag == 0:
            return 0
        u = rng.random()
        if u < _P_LAG_ZERO:
            return 0
        short = min(_SHORT_LAG, max_lag)
        if u < _P_LAG_ZERO + _P_LAG_SHORT or max_lag <= _SHORT_LAG:
            return int(rng.integers(1, short + 1))
        return int(rng.integers(_SHORT_LAG + 1, max_lag + 1))

    def _draw_edge(self, spec: RegressionSpec, driver: int, rng) -> Term:
        kind = str(rng.choice(spec.edge_kinds))
        param = None
        if kind == "saturating":
            lo, hi = np.log(_SOFTNESS_RANGE)
            param = float(np.exp(rng.uniform(lo, hi)))
        elif kind in ("threshold", "hinge"):
            param = float(rng.uniform(*_THRESHOLD_RANGE))
        return Term(
            kind=kind,
            driver=driver,
            coef=self._draw_coef(rng),
            lag=self._draw_lag(spec.max_lag, rng),
            param=param,
        )

    def _draw_fixed(self, rng: np.random.Generator):
        spec: FixedRegressionSpec = self.mechanism  # type: ignore[assignment]
        b = self.burn_in
        length = int(rng.integers(self.length[0], self.length[1] + 1))
        total = b + length
        n_inf = len(spec.drivers)
        specs = list(spec.drivers) + [
            spec.drivers[k % n_inf] for k in range(spec.n_distractors)
        ]
        raw, seeds = self._draw_series(specs, total, rng)
        base_seed = None
        base = np.zeros(total)
        if spec.target is not None:
            base_seed = int(rng.integers(0, 2**63))
            base = spec.target.draw(total, self.freq, base_seed)
        noise = rng.normal(0.0, spec.noise_std, total) if spec.noise_std else 0.0
        if not (_ok(base) and _ok(raw)):
            return "latent series not finite or too large"

        effect = _raw_effect(
            raw, list(spec.terms), [None] * len(spec.terms), 0.0, total
        )
        y = _ar_filter(spec.intercept + base + effect + noise, spec.target_ar)
        if not _ok(y[b:]):
            return "composed target not finite or too large"

        roles = [spec.role] * len(specs)
        channels, out_roles, position = self._emit(
            np.vstack([y[None], raw]), ["target", *roles], rng
        )
        recipe = {
            "mechanism": "fixed",
            "null": not spec.terms,
            "length": length,
            "burn_in": b,
            "target": {
                "channel": position[0],
                "spec": None if spec.target is None else spec.target.describe(),
                "seed": base_seed,
            },
            "covariates": [
                {
                    "channel": position[1 + j],
                    "role": spec.role,
                    "informative": j < n_inf
                    and any(j in t.drivers() for t in spec.terms),
                    "spec": specs[j].describe(),
                    "seed": seeds[j],
                }
                for j in range(len(specs))
            ],
            "terms": [t.model_dump() for t in spec.terms],
            "intercept": spec.intercept,
            "noise_std": spec.noise_std,
            "target_ar": spec.target_ar,
            "fallback": None,
        }
        return channels, out_roles, recipe

    def _draw_coupled(self, rng: np.random.Generator):
        spec: CoupledSpec = self.mechanism  # type: ignore[assignment]
        b = self.burn_in
        length = int(rng.integers(self.length[0], self.length[1] + 1))
        total = b + length
        n = int(rng.integers(spec.n_channels[0], spec.n_channels[1] + 1))
        base_spec = spec.base_pool[int(rng.integers(len(spec.base_pool)))]
        raw, seeds = self._draw_series([base_spec] * n, total, rng)
        if not _ok(raw):
            return "latent series not finite or too large"

        level = raw[:, :b].mean(axis=1)
        scale = raw[:, :b].std(axis=1)
        scale = np.where(scale < _MIN_STD, 1.0, scale)
        z = (raw - level[:, None]) / scale[:, None]
        coupling: dict[str, Any] = {}
        if n >= 2 and "mixing" in spec.couplings:
            strength = float(rng.uniform(*spec.mixing_strength))
            a = rng.normal(0.0, 1.0, (n, n))
            gram = a @ a.T
            d = np.sqrt(np.diag(gram))
            target = (1.0 - strength) * np.eye(n) + strength * gram / np.outer(d, d)
            mix = np.linalg.cholesky(target)
            z = _calibrated(mix @ z, b)
            coupling["mixing"] = {"strength": strength, "matrix": mix}
        if n >= 2 and "leadlag" in spec.couplings:
            chosen = rng.random(n - 1) < spec.p_leadlag
            if not chosen.any():
                chosen[-1] = True
            pairs = []
            for j in range(1, n):
                if not chosen[j - 1]:
                    continue
                src = int(rng.integers(0, j))
                lag = int(rng.integers(spec.lag_range[0], spec.lag_range[1] + 1))
                sign = 1.0 if rng.random() < 0.5 else -1.0
                sigma = float(rng.uniform(*spec.noise_scale))
                lead = sign * _lagged(z[src], lag) + sigma * rng.normal(size=total)
                z[j] = _calibrated(lead[None], b)[0]
                pairs.append(
                    {"src": src, "dst": j, "lag": lag, "sign": sign, "noise": sigma}
                )
            coupling["leadlag"] = pairs
        channels_latent = z * scale[:, None] + level[:, None]

        if spec.targets == "all":
            roles = ["target"] * n
        else:
            target = int(rng.integers(n))
            roles = [
                "target"
                if j == target
                else ("future" if rng.random() < self.roles.p_known_future else "past")
                for j in range(n)
            ]
        if not _ok(channels_latent[:, b:]):
            return "coupled channels not finite or too large"
        channels, out_roles, position = self._emit(channels_latent, roles, rng)
        recipe = {
            "mechanism": "coupled",
            "null": None,
            "length": length,
            "burn_in": b,
            "spec": base_spec.describe(),
            "channels": [
                {
                    "channel": position[j],
                    "role": roles[j],
                    "seed": seeds[j],
                    "calibration_mean": float(level[j]),
                    "calibration_std": float(scale[j]),
                }
                for j in range(n)
            ],
            "coupling": coupling,
            "fallback": None,
        }
        return channels, out_roles, recipe

    def _draw_varx(self, rng: np.random.Generator):
        spec: VARXSpec = self.mechanism  # type: ignore[assignment]
        b = self.burn_in
        length = int(rng.integers(self.length[0], self.length[1] + 1))
        total = b + length
        n = int(rng.integers(spec.n_targets[0], spec.n_targets[1] + 1))
        clusters = (
            None if spec.n_clusters is None else np.arange(n) * spec.n_clusters // n
        )
        if spec.coef_matrices is not None:
            lag_matrices = np.asarray(spec.coef_matrices, dtype=np.float64)
        else:
            lag_matrices = _varx_lag_matrices(spec, n, clusters, rng)
        intercept = (
            np.zeros(n) if spec.intercept is None else np.asarray(spec.intercept)
        )
        covariance = (
            np.eye(n)
            if spec.innovation_covariance is None
            else np.asarray(spec.innovation_covariance, dtype=np.float64)
        )
        shared_effects = rng.normal(0.0, spec.exog_scale, (n, spec.n_shared))
        future_effects = _varx_effects(spec, spec.n_future, n, rng)
        past_effects = _varx_effects(spec, spec.n_past, n, rng)

        phi = spec.exog_persistence
        shared = _unit_ar1((total, spec.n_shared), phi, rng)
        future = _unit_ar1((total, n, spec.n_future), phi, rng)
        past = _unit_ar1((total, n, spec.n_past), phi, rng)
        innovations = rng.standard_normal((total, n)) @ np.linalg.cholesky(covariance).T

        drive = intercept + shared @ shared_effects.T + innovations
        drive += np.einsum("kij,tjk->ti", future_effects, future)
        drive[1:] += np.einsum("kij,tjk->ti", past_effects, past[:-1])
        p = spec.lag_order
        y = np.zeros((total, n))
        y[:p] = drive[:p]
        for t in range(p, total):
            value = drive[t].copy()
            for lag in range(p):
                value += lag_matrices[lag] @ y[t - lag - 1]
            y[t] = value
        if not _ok(y[b:]):
            return "VARX targets not finite or too large"

        latent = [y[:, i] for i in range(n)]
        layout: list[dict[str, Any]] = [
            {"kind": "target", "series": i, "index": 0} for i in range(n)
        ]
        roles = ["target"] * n
        for k in range(spec.n_shared):
            latent.append(shared[:, k])
            layout.append({"kind": "shared", "series": None, "index": k})
            roles.append("future")
        for kind, values, role in (
            ("future", future, "future"),
            ("past", past, "past"),
        ):
            for i in range(n):
                for k in range(values.shape[2]):
                    latent.append(values[:, i, k])
                    layout.append({"kind": kind, "series": i, "index": k})
                    roles.append(role)
        channels, out_roles, position = self._emit(np.stack(latent), roles, rng)
        for j, entry in enumerate(layout):
            entry["channel"] = position[j]
        recipe = {
            "mechanism": "varx",
            "null": spec.n_shared + spec.n_future + spec.n_past == 0,
            "length": length,
            "burn_in": b,
            "channels": layout,
            "process": {
                "lag_matrices": lag_matrices,
                "intercept": intercept,
                "shared_effects": shared_effects,
                "future_effects": future_effects,
                "past_effects": past_effects,
                "innovation_covariance": covariance,
                "exog_persistence": phi,
                "clusters": clusters,
            },
            "fallback": None,
        }
        return channels, out_roles, recipe


# ---------------------------------------------------------------------------
# Probe suite
# ---------------------------------------------------------------------------

_PROBE_RELATIONS: dict[str, tuple[int, list[tuple[str, int, float]]]] = {
    # name: (number of drivers, [(kind, driver, coef), ...])
    "identity": (1, [("linear", 0, 1.0)]),
    "sum": (2, [("linear", 0, 1.0), ("linear", 1, 1.0)]),
    "nonlinear": (
        3,
        [("abs_sqrt", 0, 1.0), ("linear", 1, 1.0), ("quadratic", 2, -1.0)],
    ),
    "quadratic": (1, [("quadratic", 0, 1.0)]),
}


def probe_suite(
    context: int = 24,
    horizon: int = 24,
    target_ar: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9),
    n_distractors: tuple[int, ...] = (0, 2, 4),
    noise_std: float = 0.1,
    freq: str | int = "h",
) -> dict[str, GroupPrior]:
    """Fixed covariate-extraction probes.

    Relations on unit-variance drivers, with a zero base and small Gaussian
    noise: ``identity`` (``y = x``), ``sum`` (``y = x + w``), ``nonlinear``
    (``y = sqrt|x| + w - z**2``) and ``quadratic`` (``y = x**2``), each with
    target AR feedback ``y[t] += phi * y[t-1]``. Covariates are either known
    in the future and act at lag 0, or observed only in the past and act at
    lag ``horizon``, so the information is available for every lead. Drivers
    are IID Gaussian (``iid``) or a unit-variance AR(1) with coefficient 0.9
    (``smooth``). Distractors are drawn from the same driver spec.

    The task design follows the style of the controlled covariate study in
    arXiv:2605.12200 (TabPFN-TS vs Chronos-2); the exact relations, ranges and
    driver processes are SynForecast's own and not a reproduction of it.

    Pair each probe with :meth:`GroupBatch.shuffle_covariates` and a
    target-only forecast as controls.

    Args:
        context (int): Context length, at least ``horizon`` (default: 24).
        horizon (int): Forecast horizon (default: 24).
        target_ar (tuple[float, ...]): Target AR coefficients
            (default: (0.0, 0.3, 0.6, 0.9)).
        n_distractors (tuple[int, ...]): Distractor counts (default: (0, 2, 4)).
        noise_std (float): Std of the additive noise (default: 0.1).
        freq (str | int): Frequency (default: 'h').

    Returns:
        Dict from probe name ``"{relation}/phi{phi}/{role}/d{n}/{driver}"`` to
        its prior; every prior emits ``context + horizon`` steps.
    """
    from synforecast.generators import SARIMAGenerator

    if context < 1 or horizon < 1:
        raise ValueError("context and horizon must be positive")
    if context < horizon:
        raise ValueError(
            "past-only probes act at lag horizon, so context must be at least "
            "horizon for every lead to see its covariate value"
        )
    order = {"d": 0, "q": 0, "P": 0, "D": 0, "Q": 0}
    drivers = {
        "iid": GeneratorSpec(
            generator=SARIMAGenerator, params={**order, "p": 0, "noise_std": 1.0}
        ),
        "smooth": GeneratorSpec(
            generator=SARIMAGenerator,
            params={
                **order,
                "p": 1,
                "ar_params": [0.9],
                "noise_std": float(np.sqrt(1.0 - 0.81)),
            },
        ),
    }
    length = context + horizon
    probes: dict[str, GroupPrior] = {}
    for relation, (n_drivers, spec_terms) in _PROBE_RELATIONS.items():
        for phi in target_ar:
            for role in ("future", "past"):
                lag = 0 if role == "future" else horizon
                terms = [
                    Term(kind=kind, driver=d, coef=coef, lag=lag)
                    for kind, d, coef in spec_terms
                ]
                for n_dist in n_distractors:
                    for driver_name, driver in drivers.items():
                        mechanism = FixedRegressionSpec(
                            drivers=[driver] * n_drivers,
                            terms=terms,
                            noise_std=noise_std,
                            target_ar=phi,
                            n_distractors=n_dist,
                            role=role,
                        )
                        name = f"{relation}/phi{phi}/{role}/d{n_dist}/{driver_name}"
                        probes[name] = GroupPrior(
                            mechanism=mechanism,
                            length=(length, length),
                            freq=freq,
                            burn_in=max(64, lag + 64),
                        )
    return probes
