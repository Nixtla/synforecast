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

Generation is batched: a :meth:`GroupPrior.sample` call plans every group in
Python, draws all channels of all groups in one parallel native call, composes
all regression targets in a second parallel native call, and only then
assembles the groups. Results depend only on ``(seed, group_id)``, never on
the batch size or the number of workers.

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
from functools import cached_property
from typing import Any, Literal, get_args

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)

from synforecast._lib import groups as _rs_groups
from synforecast._numeric import (
    MAX_ABS,
    MIN_STD,
    VAR_STABILITY,
    companion_spectral_radius,
    stabilize,
    validate_covariance,
)
from synforecast.base import _GEN_TYPE_MAP, BaseGenerator

RECIPE_VERSION = 2

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

_EDGE_KINDS: tuple[str, ...] = get_args(EdgeKind)
_INTERACTION_KINDS: tuple[str, ...] = get_args(InteractionKind)
_KIND_CODES = {k: i for i, k in enumerate(_EDGE_KINDS + _INTERACTION_KINDS)}
# Kinds that read ``Term.param``: softness (saturating) or threshold
_PARAM_KINDS = frozenset({"saturating", "threshold", "hinge", "gate"})
_RESERVED_PARAMS = frozenset(
    {"min_length", "max_length", "freq", "seed", "engine", "exogenous", "overlays"}
)
_CALIBRATIONS = ("calibrated", "null", "nominal_effect", "nominal_base", "fixed")
_COMPOSITIONS = ("additive", "multiplicative")
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
# Template length used to resolve a generator's native parameters
_TEMPLATE_LENGTH = 64
# Errors a single draw may raise; they trigger a redraw instead of aborting
_DRAW_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError)


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
# Specs
# ---------------------------------------------------------------------------


class GeneratorSpec(BaseModel):
    """A generator class plus explicit parameters, built fresh for every draw.

    The prior draws every series with its own seed. Generators whose native
    parameters do not depend on the seed are resolved once into a template and
    drawn in parallel batches; generators that randomize parameters in their
    constructor (for example ``SARIMAGenerator`` without ``ar_params``) are
    rebuilt per draw, so each draw gets fresh parameters. The wrapped
    generator's own seed never affects the result. NumPy arrays and scalars
    in ``params`` are converted to lists and Python scalars, and ``params``
    are snapshotted at construction: later changes to the dict affect no draw,
    recipe or fingerprint. Specs with equal content are interchangeable.

    Args:
        generator (type[BaseGenerator]): Generator class.
        params (dict[str, Any]): Constructor keyword arguments, JSON
            serializable, other than ``min_length``, ``max_length``, ``freq``,
            ``seed``, ``engine``, ``exogenous`` and ``overlays`` (default:
            none).
        positive (bool): The generator's output is non-negative. Regression
            targets may then compose multiplicatively, and coupled channels
            are coupled in log space with their own zeros kept
            (default: False).

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

    _params_json: str = PrivateAttr(default="{}")
    _key: str = PrivateAttr(default="")

    @field_validator("params")
    @classmethod
    def _check_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        reserved = sorted(_RESERVED_PARAMS & set(params))
        if reserved:
            raise ValueError(f"params must not set {reserved}; the prior owns them")
        params = _jsonable(params)
        try:
            json.dumps(params, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"params must be JSON-serializable and finite: {exc}"
            ) from exc
        return params

    def model_post_init(self, __context: Any) -> None:
        # Snapshot params so later changes to the dict cannot change draws,
        # recipes or fingerprints; the key identifies the spec by content.
        self._params_json = json.dumps(self.params, sort_keys=True)
        self._key = json.dumps(
            [
                self._serialize_generator(self.generator),
                json.loads(self._params_json),
                self.positive,
            ],
            sort_keys=True,
        )

    @field_serializer("generator")
    def _serialize_generator(self, generator: type[BaseGenerator]) -> str:
        return f"{generator.__module__}.{generator.__qualname__}"

    @field_serializer("params")
    def _serialize_params(self, params: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return json.loads(self._params_json)

    @property
    def name(self) -> str:
        """Name of the generator class."""
        return self.generator.__name__

    def build(self, length: int, freq: str | int, seed: int) -> BaseGenerator:
        """Build the generator for series of ``length`` steps."""
        params = json.loads(self._params_json)
        return self.generator(
            min_length=length, max_length=length, freq=freq, seed=seed, **params
        )

    def draw(self, length: int, freq: str | int, seed: int) -> np.ndarray:
        """Draw one series of ``length`` values with a freshly built generator
        (its Python ``generate_single_series`` path)."""
        gen = self.build(length, freq, seed)
        return np.asarray(gen.generate_single_series(length), dtype=np.float64)

    @cached_property
    def description(self) -> dict[str, Any]:
        """JSON-serializable description used in recipes."""
        return self.model_dump(mode="json")

    def describe(self) -> dict[str, Any]:
        """JSON-serializable description used in recipes."""
        return dict(self.description)


class Term(BaseModel):
    """One additive term of the covariate effect.

    Main edges apply ``coef * f(x[t - lag])`` to one driver ``x``; ``param`` is
    the softness ``s`` of ``saturating`` (``s * tanh(x / s)``, default 1) or
    the threshold ``tau`` of ``threshold`` (``1[x > tau]``) and ``hinge``
    (``max(x - tau, 0)``), default 0. Interactions combine two drivers:
    ``product`` is ``coef * x[t - lag] * w[t - partner_lag]`` and ``gate`` is
    ``coef * x[t - lag] * 1[w[t - partner_lag] > tau]``. Lags read the
    burn-in, holding the first value before the start.

    Scale: under :class:`RegressionSpec`, terms act on drivers standardized
    with calibration-window statistics and ``coef`` is a relative weight
    before strength calibration (the recipe's ``effect_scale`` applies on
    top). Under :class:`FixedRegressionSpec`, terms act on raw driver values
    with their literal coefficients. Recipes record this as
    ``driver_scale``.

    Args:
        kind (str): Edge or interaction kind.
        driver (int): Index of the driver, in the mechanism's driver order.
        coef (float): Coefficient.
        lag (int): Lag of the driver in time steps (default: 0).
        param (float | None): Softness or threshold, only for kinds that use
            one.
        partner (int | None): Second driver, required for interactions.
        partner_lag (int): Lag of the second driver, only for interactions
            (default: 0).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: EdgeKind | InteractionKind
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
        if not interaction and self.partner_lag:
            raise ValueError("partner_lag applies only to interactions")
        if self.param is not None and self.kind not in _PARAM_KINDS:
            raise ValueError(f"{self.kind} takes no param")
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


def default_driver_pool() -> list[GeneratorSpec]:
    """Generator pool for covariates: smooth, persistent, seasonal, regime and
    count-like drivers. Informative covariates and distractors share it.

    KernelSynth is left out because its exact sampler is quadratic in the
    series length; add it explicitly where long smooth compositions matter.
    """
    from synforecast import generators as g

    return [
        GeneratorSpec(generator=g.SARIMAGenerator),
        GeneratorSpec(generator=g.OrnsteinUhlenbeckGenerator),
        GeneratorSpec(generator=g.RandomWalkGenerator),
        GeneratorSpec(generator=g.SeasonalGenerator),
        GeneratorSpec(generator=g.GaussianProcessGenerator),
        GeneratorSpec(generator=g.RegimeSwitchingGenerator),
        GeneratorSpec(generator=g.TSIGenerator),
        GeneratorSpec(generator=g.IntermittentDemandGenerator, positive=True),
        GeneratorSpec(generator=g.PoissonProcessGenerator, positive=True),
        GeneratorSpec(generator=g.EnergyLoadGenerator, positive=True),
    ]


def default_target_pool() -> list[GeneratorSpec]:
    """Generator pool for the target's own dynamics (KernelSynth left out, as
    in :func:`default_driver_pool`)."""
    from synforecast import generators as g

    return [
        GeneratorSpec(generator=g.SARIMAGenerator),
        GeneratorSpec(generator=g.ETSGenerator),
        GeneratorSpec(generator=g.SeasonalGenerator),
        GeneratorSpec(generator=g.RandomWalkGenerator),
        GeneratorSpec(generator=g.OrnsteinUhlenbeckGenerator),
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
    with a target base from ``target_pool``. Every covariate is known in the
    future with probability ``p_known_future`` and past-only otherwise,
    independently of whether it is informative. See :class:`GroupPrior` for
    the calibration of the effect strength.

    Args:
        target_pool (list[GeneratorSpec]): Target base dynamics (default:
            :func:`default_target_pool`).
        driver_pool (list[GeneratorSpec]): Covariate dynamics (default:
            :func:`default_driver_pool`).
        n_covariates (tuple[int, int]): Inclusive range of emitted covariates
            (default: (1, 6)).
        n_informative (tuple[int, int]): Inclusive range of informative
            covariates in a non-null group, capped at the covariate count
            (default: (1, 3)).
        p_null (float): Probability that every covariate is a distractor
            (default: 0.15).
        p_known_future (float): Probability that a covariate is known in the
            future (default: 0.5).
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
            calibration window, before target AR feedback (default:
            (0.1, 0.9)).
        p_multiplicative (float): Probability of multiplicative composition
            ``base * exp(effect)`` for positive target specs (default: 0.3).
        log_effect_std (tuple[float, float]): Range of the effect's standard
            deviation in log space under multiplicative composition, on the
            calibration window (default: (0.05, 0.7)).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["regression"] = "regression"
    target_pool: tuple[GeneratorSpec, ...] = Field(
        default_factory=lambda: tuple(default_target_pool())
    )
    driver_pool: tuple[GeneratorSpec, ...] = Field(
        default_factory=lambda: tuple(default_driver_pool())
    )
    n_covariates: tuple[int, int] = (1, 6)
    n_informative: tuple[int, int] = (1, 3)
    p_null: float = Field(default=0.15, ge=0.0, le=1.0)
    p_known_future: float = Field(default=0.5, ge=0.0, le=1.0)
    edge_kinds: tuple[EdgeKind, ...] = _EDGE_KINDS
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

    def generator_specs(self) -> list[GeneratorSpec]:
        """Every generator spec the mechanism can draw from."""
        return [*self.target_pool, *self.driver_pool]


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

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["fixed"] = "fixed"
    drivers: tuple[GeneratorSpec, ...]
    terms: tuple[Term, ...]
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

    def generator_specs(self) -> list[GeneratorSpec]:
        """Every generator spec the mechanism can draw from."""
        return [*self.drivers, *([self.target] if self.target else [])]


class CoupledSpec(BaseModel):
    """Related channels from one base generator, coupled cotemporaneously
    and by lead-lag (a Chronos-2-style multivariatizer).

    Per group, one spec from ``base_pool`` draws every channel. Channels are
    standardized with calibration-window statistics, coupled, then restored
    to each channel's calibration-window level and scale:

    - ``"mixing"``: channels become linear combinations ``L @ z`` where ``L``
      is the Cholesky factor of a random correlation target
      ``C = (1 - s) I + s Q`` (``Q`` the correlation matrix of a random
      Gaussian Gram matrix, ``s`` the mixing strength). ``C`` is the
      intended correlation for independent unit-variance inputs; shared
      trends or seasonality, finite windows, later lead-lag and log-space
      mapping make the emitted correlations differ.
    - ``"leadlag"``: with probability ``p_leadlag`` (at least once), a
      channel becomes ``sign * z_src[t - lag] + sigma * eps`` of an earlier
      channel. Lags read the burn-in, so nothing wraps around; the prior's
      ``burn_in`` must exceed the longest possible chain of lags,
      ``(n_channels[1] - 1) * lag_range[1]``.

    Positive base specs are coupled in log space and mapped back with
    ``exp``: each channel keeps exactly its own zeros (occurrence stays the
    channel's own) while the sizes of its positive values are coupled. Zeros
    enter the coupling at the channel's calibration-window mean log size.

    With ``targets='all'`` every channel is a target (multivariate
    forecasting). With ``targets='one'`` one random channel is the target and
    the others are covariates, known in the future with probability
    ``p_known_future``. A covariate whose lead-lag chain passes through the
    target is a lagged copy of it; its recipe entry has
    ``derived_from_target``.
    Coupling makes channels related, not causally informative.

    The recipe follows Chronos-2 (Ansari et al., 2025,
    https://arxiv.org/abs/2510.15821); the mixing construction, lag ranges
    and calibration are SynForecast's own design rather than a reproduction
    of the reference code.

    Args:
        base_pool (list[GeneratorSpec]): Base dynamics, one spec per group
            (default: :func:`default_target_pool`).
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
        p_known_future (float): Probability that a covariate is known in the
            future under ``targets='one'`` (default: 0.5).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["coupled"] = "coupled"
    base_pool: tuple[GeneratorSpec, ...] = Field(
        default_factory=lambda: tuple(default_target_pool())
    )
    n_channels: tuple[int, int] = (2, 6)
    couplings: tuple[Literal["mixing", "leadlag"], ...] = ("mixing", "leadlag")
    mixing_strength: tuple[float, float] = (0.2, 0.9)
    lag_range: tuple[int, int] = (1, 24)
    noise_scale: tuple[float, float] = (0.02, 0.2)
    p_leadlag: float = Field(default=0.5, ge=0.0, le=1.0)
    targets: Literal["all", "one"] = "all"
    p_known_future: float = Field(default=0.5, ge=0.0, le=1.0)

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

    def generator_specs(self) -> list[GeneratorSpec]:
        """Every generator spec the mechanism can draw from."""
        return list(self.base_pool)


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
        coef_matrices (list | None): Fixed, stable lag matrices of shape
            ``(lag_order, N, N)``; ``n_targets`` must be ``(N, N)``
            (default: None).
        intercept (list[float] | None): Intercept of length ``N``;
            ``n_targets`` must be ``(N, N)`` (default: zeros).
        innovation_covariance (list | None): Symmetric positive-definite
            ``(N, N)`` innovation covariance; ``n_targets`` must be ``(N, N)``
            (default: identity).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

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
    coef_matrices: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    intercept: tuple[float, ...] | None = None
    innovation_covariance: tuple[tuple[float, ...], ...] | None = None

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
            if mats.shape != (self.lag_order, lo, lo):
                raise ValueError(
                    f"coef_matrices must have shape (lag_order, N, N) = "
                    f"({self.lag_order}, {lo}, {lo}), got {mats.shape}"
                )
            if not np.isfinite(mats).all():
                raise ValueError("coef_matrices must be finite")
            if companion_spectral_radius(list(mats)) >= 1.0:
                raise ValueError("coef_matrices define an unstable VAR")
        if self.intercept is not None and not np.isfinite(self.intercept).all():
            raise ValueError("intercept must be finite")
        if self.innovation_covariance is not None:
            validate_covariance(self.innovation_covariance, lo)
        return self

    @property
    def largest_lag(self) -> int:
        """Largest lag of the process."""
        return self.lag_order

    def generator_specs(self) -> list[GeneratorSpec]:
        """VARX draws no generator series."""
        return []


Mechanism = RegressionSpec | FixedRegressionSpec | CoupledSpec | VARXSpec


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
        components (dict | None): For regression mechanisms sampled with
            ``components=True``: the target's ``base`` and the applied
            ``effect`` on the emitted window (additive, or log-space under
            multiplicative composition), before target AR feedback.
    """

    group_id: int
    timestamps: np.ndarray
    values: np.ndarray
    roles: tuple[str, ...]
    recipe: dict = field(repr=False)
    components: dict | None = field(default=None, repr=False)

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
    time_dtype: str = "datetime64[ns]"

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[Group]:
        return iter(self.groups)

    def __getitem__(self, index: int) -> Group:
        return self.groups[index]

    @property
    def diagnostics(self) -> dict[str, int]:
        """Counts of null groups, redraws, composition fallbacks, generator
        noise fallbacks and shuffled controls (counted on their source
        groups)."""
        recipes = [g.recipe.get("source", g.recipe) for g in self.groups]
        return {
            "n_groups": len(self.groups),
            "n_null": sum(bool(r.get("null")) for r in recipes),
            "n_redrawn": sum(r["attempt"] > 0 for r in recipes),
            "n_fallback": sum(r.get("fallback") is not None for r in recipes),
            "n_generator_fallback": sum(
                bool(r.get("generator_fallbacks")) for r in recipes
            ),
            "n_controls": sum("control" in g.recipe for g in self.groups),
        }

    def to_padded(
        self,
        max_channels: int | None = None,
        max_length: int | None = None,
        align: Literal["end", "start"] = "end",
        dtype: Any = np.float32,
    ) -> dict[str, np.ndarray]:
        """Pad the batch into dense arrays for model training.

        Returns a dict with ``values`` ``(B, C, T)`` (padding is 0),
        ``observed`` ``(B, C, T)`` (False for padding and missing values),
        ``target_mask``, ``known_future_mask`` and ``channel_mask``
        ``(B, C)``, and ``lengths`` ``(B,)``. ``align='end'`` right-aligns
        series (left padding) so the last observed steps line up; longer
        groups keep their last ``max_length`` steps (``'end'``) or first
        steps (``'start'``). Groups with more channels than ``max_channels``
        raise.
        """
        if align not in ("end", "start"):
            raise ValueError("align must be 'end' or 'start'")
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be positive")
        if max_channels is not None and max_channels < 0:
            raise ValueError("max_channels must be non-negative")
        n = len(self.groups)
        c = max((g.values.shape[0] for g in self.groups), default=0)
        t = max((g.length for g in self.groups), default=0)
        c = c if max_channels is None else max_channels
        t = t if max_length is None else max_length
        values = np.zeros((n, c, t), dtype=dtype)
        observed = np.zeros((n, c, t), dtype=bool)
        target_mask = np.zeros((n, c), dtype=bool)
        known_future = np.zeros((n, c), dtype=bool)
        channel_mask = np.zeros((n, c), dtype=bool)
        lengths = np.zeros(n, dtype=np.int64)
        for i, group in enumerate(self.groups):
            k = group.values.shape[0]
            if k > c:
                raise ValueError(f"group {group.group_id} has {k} > {c} channels")
            vals = group.values
            if vals.shape[1] > t:
                vals = vals[:, -t:] if align == "end" else vals[:, :t]
            length = vals.shape[1]
            span = slice(t - length, t) if align == "end" else slice(0, length)
            finite = np.isfinite(vals)
            values[i, :k, span] = np.where(finite, vals, 0.0)
            observed[i, :k, span] = finite
            roles = np.asarray(group.roles)
            target_mask[i, :k] = roles == "target"
            known_future[i, :k] = roles == "future"
            channel_mask[i, :k] = True
            lengths[i] = length
        return {
            "values": values,
            "observed": observed,
            "target_mask": target_mask,
            "known_future_mask": known_future,
            "channel_mask": channel_mask,
            "lengths": lengths,
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
        recipes. Serialization consumes no random draws. For training, prefer
        :meth:`to_padded`, which avoids per-row strings.
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
            time_col: np.concatenate(times) if times else np.empty(0, self.time_dtype),
            target_col: np.concatenate(values) if values else np.empty(0),
            role_col: np.concatenate(roles) if roles else np.empty(0, str),
        }
        return nw.DataFrame.from_dict(result, backend=engine).to_native()

    def shuffle_covariates(self, seed: int) -> GroupBatch:
        """Control batch: each target keeps its own values but gets the whole
        covariate set (values and roles) of another group.

        Donors are drawn among the other groups with the same target and
        covariate counts and at least the receiver's length; their covariates
        are cropped to the receiver's length (their first steps, which keeps
        the crop causal). If no donor is that long, the longest other group is
        used and the control group is cropped to its length (recorded as
        ``cropped_to``). Every (target count, covariate count) bucket needs at
        least two groups. Whole covariate sets move together, so relations
        between covariates are preserved while their relation to the targets
        is broken. Targets keep their channel positions; the donor's
        covariates fill the covariate positions in order, with their roles.

        A control group's recipe is a control record: ``source`` holds the
        receiving group's recipe, ``covariates_from`` the donor's group ID and
        ``covariate_sources`` which donor channel fills each covariate
        position. Control groups have no single generating process, so
        :meth:`GroupPrior.replay` and :meth:`VARXTruth.from_recipe` reject
        them; replay the source groups instead.
        """
        rng = np.random.default_rng(seed)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, group in enumerate(self.groups):
            key = (len(group.target_indices), len(group.covariate_indices))
            buckets[key].append(i)
        for members in buckets.values():
            if len(members) < 2:
                raise ValueError(
                    "every (target count, covariate count) bucket needs at least "
                    "two groups"
                )
        shuffled = []
        for i, group in enumerate(self.groups):
            key = (len(group.target_indices), len(group.covariate_indices))
            others = [j for j in buckets[key] if j != i]
            eligible = [j for j in others if self.groups[j].length >= group.length]
            if eligible:
                donor = self.groups[eligible[int(rng.integers(len(eligible)))]]
                length = group.length
            else:
                donor = max((self.groups[j] for j in others), key=lambda g: g.length)
                length = donor.length
            values = group.values[:, :length].copy()
            roles = list(group.roles)
            sources = []
            for dst, src in zip(
                group.covariate_indices, donor.covariate_indices, strict=True
            ):
                values[dst] = donor.values[src, :length]
                roles[dst] = donor.roles[src]
                sources.append({"channel": dst, "donor_channel": src})
            recipe: dict[str, Any] = {
                "version": RECIPE_VERSION,
                "control": "shuffled_covariates",
                "source": group.recipe,
                "covariates_from": donor.group_id,
                "covariate_sources": sources,
            }
            if length < group.length:
                recipe["cropped_to"] = length
            shuffled.append(
                replace(
                    group,
                    values=values,
                    timestamps=group.timestamps[:length],
                    roles=tuple(roles),
                    recipe=recipe,
                    components=None,
                )
            )
        return GroupBatch(tuple(shuffled), self.time_dtype)


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
        """Rebuild the truth of a VARX group from its full recipe, validating
        the recipe's version and array shapes."""
        if "control" in recipe:
            raise ValueError(
                "control groups have no single VARX process; use the source "
                "group's recipe"
            )
        if recipe.get("mechanism") != "varx":
            raise ValueError("recipe is not a VARX group")
        if recipe.get("version") != RECIPE_VERSION:
            raise ValueError("recipe version does not match this SynForecast")
        if "process" not in recipe:
            raise ValueError("recipe has no process; sample with recipes=True")
        process = recipe["process"]

        def arr(name: str, ndim: int) -> np.ndarray:
            a = np.asarray(process[name], dtype=np.float64)
            if a.ndim != ndim or not np.isfinite(a).all():
                raise ValueError(f"process['{name}'] must be a finite {ndim}-D array")
            return a

        lags = arr("lag_matrices", 3)
        p, n, _ = lags.shape
        intercept = arr("intercept", 1)
        shared = arr("shared_effects", 2)
        cov = arr("innovation_covariance", 2)
        # Effects of absent regressor kinds serialize as empty lists
        future, past = (
            np.zeros((0, n, n)) if not process[name] else arr(name, 3)
            for name in ("future_effects", "past_effects")
        )
        shapes_ok = (
            lags.shape == (p, n, n)
            and intercept.shape == (n,)
            and shared.shape[0] == n
            and future.shape[1:] == (n, n)
            and past.shape[1:] == (n, n)
            and cov.shape == (n, n)
        )
        if not shapes_ok:
            raise ValueError("recipe process arrays have inconsistent shapes")
        clusters = process["clusters"]
        if clusters is not None:
            clusters = np.asarray(clusters, dtype=np.int64)
            if clusters.shape != (n,):
                raise ValueError("process['clusters'] must have one entry per target")
        return cls(
            lag_matrices=lags,
            intercept=intercept,
            shared_effects=shared,
            future_effects=future,
            past_effects=past,
            innovation_covariance=cov,
            exog_persistence=float(process["exog_persistence"]),
            clusters=clusters,
        )

    @staticmethod
    def split(group: Group) -> dict[str, np.ndarray]:
        """Channels of a VARX group by kind, in target and regressor order:
        ``y`` ``(T, N)``, ``shared`` ``(T, S)``, ``future`` ``(T, N, F)`` and
        ``past`` ``(T, N, H)``."""
        recipe = group.recipe
        if recipe.get("mechanism") != "varx" or "channels" not in recipe:
            raise ValueError("split needs an unshuffled VARX group with its recipe")
        layout = recipe["channels"]
        if len(layout) != group.values.shape[0]:
            raise ValueError("recipe channel layout does not match the group")
        n = sum(c["kind"] == "target" for c in layout)
        counts = {
            kind: sum(c["kind"] == kind for c in layout)
            for kind in ("shared", "future", "past")
        }
        if n == 0 or counts["future"] % n or counts["past"] % n:
            raise ValueError("recipe channel layout is inconsistent")
        f, h = counts["future"] // n, counts["past"] // n
        length = group.length
        out = {
            "y": np.empty((length, n)),
            "shared": np.empty((length, counts["shared"])),
            "future": np.empty((length, n, f)),
            "past": np.empty((length, n, h)),
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
# Sampling machinery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Source:
    """How one GeneratorSpec draws: a shared native template ('batch'), a
    native kernel with per-draw parameters ('per_draw'), or Python
    ('python')."""

    mode: Literal["batch", "per_draw", "python"]
    gen_type: int = -1
    scalar: np.ndarray | None = None
    arrays: tuple[np.ndarray, ...] = ()


def _native_params(gen: BaseGenerator) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    scalar, arrays = gen._get_batch_params()  # type: ignore[misc]
    return (
        np.ascontiguousarray(scalar, dtype=np.float64),
        tuple(np.ascontiguousarray(a, dtype=np.float64) for a in arrays),
    )


def _same_params(a: tuple, b: tuple) -> bool:
    (sa, aa), (sb, ab) = a, b
    return (
        np.array_equal(sa, sb)
        and len(aa) == len(ab)
        and all(np.array_equal(x, y) for x, y in zip(aa, ab, strict=True))
    )


def _resolve(spec: GeneratorSpec, freq: str | int) -> _Source:
    """Decide how ``spec`` draws; raises on parameters the generator or its
    native kernel rejects."""
    try:
        first = spec.build(_TEMPLATE_LENGTH, freq, 0)
        second = spec.build(_TEMPLATE_LENGTH, freq, 1)
    except Exception as exc:
        raise ValueError(f"{spec.name} rejects params {spec.params}: {exc}") from exc
    gen_type = _GEN_TYPE_MAP.get(spec.generator.__name__)
    library = spec.generator.__module__.startswith("synforecast.generators")
    if gen_type is None or not library or first._get_batch_params() is None:
        return _Source("python")
    params = _native_params(first)
    _, status, messages = _rs_groups.draw_series(
        np.asarray([gen_type], dtype=np.int32),
        [params[0]],
        [list(params[1])],
        np.zeros(1, dtype=np.uint32),
        np.asarray([_TEMPLATE_LENGTH], dtype=np.int64),
        np.zeros(1, dtype=np.uint64),
        1,
    )
    if status[0] == 2:
        raise ValueError(f"{spec.name} rejects params {spec.params}: {messages[0]}")
    if _same_params(params, _native_params(second)):
        return _Source("batch", gen_type, params[0], params[1])
    return _Source("per_draw", gen_type)


@dataclass
class _Plan:
    """Everything a group needs except its series values."""

    group_id: int
    attempt: int
    length: int
    total: int
    requests: list[tuple[GeneratorSpec, int]]
    data: dict[str, Any]


def _lagged(x: np.ndarray, lag: int) -> np.ndarray:
    """``x[t - lag]``, holding the first value before the start (no wrap)."""
    if lag == 0:
        return x
    out = np.empty_like(x)
    out[:lag] = x[0]
    out[lag:] = x[:-lag]
    return out


def _calibrated(
    x: np.ndarray, burn_in: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize rows with mean and std of the first ``burn_in`` steps."""
    mean = x[:, :burn_in].mean(axis=1, keepdims=True)
    std = x[:, :burn_in].std(axis=1, keepdims=True)
    std = np.where(std < MIN_STD, 1.0, std)
    return (x - mean) / std, mean[:, 0], std[:, 0]


def _ok(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all() and np.abs(values).max(initial=0.0) < MAX_ABS)


def _unit_ar1(shocks: np.ndarray, phi: float) -> np.ndarray:
    """Unit-variance AR(1) along axis 0 of ``shocks``, started from its
    stationary law."""
    if shocks.size == 0:
        return shocks
    shape = shocks.shape
    m = int(np.prod(shape[1:]))
    x = np.sqrt(1.0 - phi**2) * shocks.reshape(shape[0], m)
    x[0] = shocks.reshape(shape[0], m)[0]
    out = _rs_groups.ar_columns(x.ravel(), m, float(phi), np.zeros(m))
    return out.reshape(shape)


class GroupPrior(BaseModel):
    """Sample groups of aligned target and covariate channels.

    Every group is generated over ``burn_in + length`` steps. The first
    ``burn_in`` steps are a calibration and warm-up window: lags, distributed
    lags and AR filters read it, the standardization of drivers and the
    strength of the effect are fixed on it, and it is then discarded. No
    statistic of the emitted window feeds back into the composition, so the
    covariate effect at time ``t`` uses no later value. Latent series come
    from the wrapped generators as whole paths.

    Effect strength (``RegressionSpec``): under additive composition the
    centered effect is scaled so that its variance on the calibration window
    is ``share / (1 - share)`` times the base variance there; under
    multiplicative composition (positive target specs only) its log-space std
    on the calibration window is set directly. Both apply before target AR
    feedback. If the effect or the base is constant on the calibration window,
    the degenerate std is replaced by 1 and the recipe's ``calibration`` is
    ``'nominal_effect'`` or ``'nominal_base'``. The recipe's ``realized`` is
    the emitted-window variance share (additive) or log-space std
    (multiplicative).

    Channels are shuffled after construction, so channel position carries no
    information. A group is redrawn with a new attempt seed (up to
    ``max_redraws``) when a latent series (including the burn-in) or the
    emitted target is not finite or exceeds 1e8 in magnitude, or when a draw
    raises; a failing multiplicative composition first falls back to
    additive, which the recipe records. Persistent failure raises with every
    attempt's reason. Generator noise fallbacks (KernelSynth) are recorded in
    the recipe's ``generator_fallbacks`` (indices into ``channels``), or
    redrawn with
    ``on_generator_fallback='redraw'``.

    Recipes share one layout: a header (``version``, ``prior`` fingerprint,
    ``seed``, ``group_id``, ``attempt``, ``mechanism``, ``failures``,
    ``generator_fallbacks``) and ``channels``, one entry per latent channel
    with its emitted ``channel`` index and ``role`` (regression targets come
    first). Regression ``terms`` reference drivers by their index in
    ``channels`` and by emitted channel (``driver_channel``). Sampling with
    ``recipes=False`` keeps only the header fields :meth:`replay` needs.

    Sizes (lengths, burn-in, channel counts, VAR orders) are not capped; a
    prior built from untrusted configuration should bound them first.

    Seeding: group ``i`` of ``sample(n, seed, start_group)`` depends only on
    ``(seed, start_group + i)`` and the prior's configuration, never on ``n``,
    batching or ``n_jobs``, so groups can be sharded across workers and
    replayed individually with :meth:`replay`.

    Args:
        mechanism (RegressionSpec | FixedRegressionSpec | CoupledSpec | VARXSpec):
            How the channels are generated and related
            (default: ``RegressionSpec()``).
        length (tuple[int, int]): Inclusive range of emitted lengths
            (default: (256, 1024)).
        freq (str | int): Pandas offset alias or positive integer time step
            (default: 'h').
        start_datetime (str): First timestamp (default: '2000-01-01').
        burn_in (int): Calibration and warm-up steps; must exceed the
            mechanism's largest lag and be at least 16 (default: 128).
        max_redraws (int): Redraws allowed per group (default: 5).
        on_generator_fallback (str): 'record' or 'redraw' groups whose
            generator returned its noise fallback (default: 'record').

    Example:
        >>> prior = GroupPrior(length=(128, 128), freq="D")
        >>> batch = prior.sample(n_groups=4, seed=0)
        >>> arrays = batch.to_padded()
        >>> arrays["values"].shape[0]
        4
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism: Mechanism = Field(default_factory=RegressionSpec, discriminator="kind")
    length: tuple[int, int] = (256, 1024)
    freq: str | int = "h"
    start_datetime: str = "2000-01-01"
    burn_in: int = Field(default=128, ge=16)
    max_redraws: int = Field(default=5, ge=0)
    on_generator_fallback: Literal["record", "redraw"] = "record"

    _sources: dict[str, _Source] = PrivateAttr(default_factory=dict)
    _fingerprint: str = PrivateAttr(default="")
    _timestamps_cache: np.ndarray | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check(self) -> GroupPrior:
        lo, hi = self.length
        if not 2 <= lo <= hi:
            raise ValueError("length must satisfy 2 <= low <= high")
        if self.burn_in <= self.mechanism.largest_lag:
            raise ValueError(
                f"burn_in must exceed the largest lag ({self.mechanism.largest_lag})"
            )
        return self

    @field_validator("freq")
    @classmethod
    def _check_freq(cls, freq: str | int) -> str | int:
        if isinstance(freq, int):
            if freq < 1:
                raise ValueError("an integer freq must be a positive step")
            return freq
        try:
            pd.tseries.frequencies.to_offset(freq)
        except ValueError as exc:
            raise ValueError(f"invalid freq {freq!r}") from exc
        return freq

    @field_validator("start_datetime")
    @classmethod
    def _check_start(cls, start: str) -> str:
        try:
            pd.Timestamp(start)
        except ValueError as exc:
            raise ValueError(f"invalid start_datetime {start!r}") from exc
        return start

    def model_post_init(self, __context: Any) -> None:
        from synforecast import __version__

        sources = {}
        for spec in self.mechanism.generator_specs():
            if spec._key not in sources:
                sources[spec._key] = _resolve(spec, self.freq)
        self._sources = sources
        payload = json.dumps(
            {"prior": self.model_dump(mode="json"), "synforecast": __version__},
            sort_keys=True,
        )
        self._fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:32]

    @property
    def fingerprint(self) -> str:
        """Hash of the configuration and SynForecast version, in every recipe."""
        return self._fingerprint

    def _timestamps(self, length: int) -> np.ndarray:
        cache = self._timestamps_cache
        if cache is None:
            periods = self.length[1]
            if isinstance(self.freq, int):
                cache = np.arange(periods, dtype=np.int64) * self.freq
            else:
                cache = pd.date_range(
                    pd.Timestamp(self.start_datetime), periods=periods, freq=self.freq
                ).values
            self._timestamps_cache = cache
        return cache[:length]

    @property
    def _time_dtype(self) -> str:
        return "int64" if isinstance(self.freq, int) else "datetime64[ns]"

    # -- public API -----------------------------------------------------------

    def sample(
        self,
        n_groups: int,
        seed: int,
        start_group: int = 0,
        *,
        n_jobs: int = -1,
        recipes: bool = True,
        components: bool = False,
    ) -> GroupBatch:
        """Sample ``n_groups`` groups with IDs ``start_group, start_group + 1, ...``.

        Args:
            n_groups (int): Number of groups.
            seed (int): Base seed.
            start_group (int): First group ID (default: 0).
            n_jobs (int): Native worker threads; -1 uses all cores, 1 is
                single-threaded (use 1 inside data-loader worker processes).
                Results do not depend on it (default: -1).
            recipes (bool): Build full recipes. With False, recipes hold only
                what :meth:`replay` needs, which is faster (default: True).
            components (bool): Attach the target's base and applied effect to
                regression groups (default: False).
        """
        if n_groups < 0:
            raise ValueError("n_groups must be >= 0")
        ids = list(range(start_group, start_group + n_groups))
        groups = self._sample_ids(seed, ids, n_jobs, recipes, components)
        return GroupBatch(tuple(groups), self._time_dtype)

    def replay(self, recipe: dict, *, components: bool = False) -> Group:
        """Rebuild the group a recipe describes, with its full recipe; the
        prior must match the one that produced it."""
        if "control" in recipe:
            raise ValueError(
                "control groups cannot be replayed; replay recipe['source'] and "
                "the donor group instead"
            )
        if recipe.get("version") != RECIPE_VERSION:
            raise ValueError("recipe version does not match this SynForecast")
        if recipe.get("prior") != self.fingerprint:
            raise ValueError("recipe was produced by a different prior configuration")
        group = self._sample_ids(
            recipe["seed"], [recipe["group_id"]], -1, True, components
        )[0]
        if group.recipe["attempt"] != recipe["attempt"]:
            raise ValueError(
                "replay reached a different attempt than the recipe; the "
                "generators changed since the recipe was produced"
            )
        return group

    def draw_channel(self, spec: GeneratorSpec, length: int, seed: int) -> np.ndarray:
        """Draw one latent series exactly as sampling does.

        Recipes record every channel's spec and seed; with ``length`` equal to
        the recipe's ``burn_in + length`` this reproduces the channel's latent
        path, burn-in included.
        """
        plan = _Plan(-1, 0, length, length, [(spec, int(seed))], {})
        series, _, errors = self._draw([plan], 1)
        if errors[0] is not None:
            raise ValueError(errors[0])
        return series[0][0]

    def _source(self, spec: GeneratorSpec) -> _Source:
        source = self._sources.get(spec._key)
        if source is None:
            source = self._sources[spec._key] = _resolve(spec, self.freq)
        return source

    # -- batched sampling -------------------------------------------------------

    def _sample_ids(
        self,
        seed: int,
        ids: list[int],
        n_jobs: int,
        recipes: bool,
        components: bool,
    ) -> list[Group]:
        workers = 0 if n_jobs == -1 else max(int(n_jobs), 1)
        pending = dict.fromkeys(ids, 0)
        failures: dict[int, list[str]] = {gid: [] for gid in ids}
        done: dict[int, Group] = {}
        mech = self.mechanism
        while pending:
            plans: list[_Plan] = []
            retry: dict[int, int] = {}
            fail = self._failure_recorder(failures, retry)

            for gid, attempt in pending.items():
                rng = np.random.default_rng([seed, gid, attempt])
                try:
                    plans.append(_PLANNERS[mech.kind](self, mech, rng, gid, attempt))
                except _DRAW_ERRORS as exc:
                    fail(gid, attempt, f"plan: {exc}")

            series, flags, errors = self._draw(plans, workers)
            composed = (
                self._compose(plans, series, errors, workers)
                if mech.kind in ("regression", "fixed")
                else [None] * len(plans)
            )
            for plan, ser, flag, err, comp in zip(
                plans, series, flags, errors, composed, strict=True
            ):
                gid, attempt = plan.group_id, plan.attempt
                if err is not None:
                    fail(gid, attempt, err)
                    continue
                if any(flag) and self.on_generator_fallback == "redraw":
                    fail(gid, attempt, "generator noise fallback")
                    continue
                try:
                    result = _FINISHERS[mech.kind](
                        self, mech, plan, ser, comp, recipes, components
                    )
                except _DRAW_ERRORS as exc:
                    result = f"finish: {exc}"
                if isinstance(result, str):
                    fail(gid, attempt, result)
                    continue
                channels, roles, recipe, parts = result
                header = {
                    "version": RECIPE_VERSION,
                    "prior": self.fingerprint,
                    "seed": int(seed),
                    "group_id": int(gid),
                    "attempt": int(attempt),
                    "mechanism": mech.kind,
                }
                if recipes:
                    header["failures"] = list(failures[gid])
                    channel_of = plan.data.get("request_channels", [])
                    header["generator_fallbacks"] = [
                        channel_of[i] for i, f in enumerate(flag) if f
                    ]
                    header.update(recipe)
                done[gid] = Group(
                    group_id=gid,
                    timestamps=self._timestamps(channels.shape[1]),
                    values=channels,
                    roles=roles,
                    recipe=header,
                    components=parts,
                )
            pending = retry
        return [done[gid] for gid in ids]

    def _failure_recorder(
        self, failures: dict[int, list[str]], retry: dict[int, int]
    ) -> Any:
        """Record a failed attempt and schedule the next, or raise."""

        def fail(gid: int, attempt: int, reason: str) -> None:
            failures[gid].append(reason)
            if attempt + 1 > self.max_redraws:
                raise RuntimeError(
                    f"group {gid} failed after {attempt + 1} attempts: {failures[gid]}"
                )
            retry[gid] = attempt + 1

        return fail

    def _draw(
        self, plans: list[_Plan], workers: int
    ) -> tuple[list[list[Any]], list[list[bool]], list[str | None]]:
        """Draw every series every plan requests: native draws in one call."""
        gen_types: list[int] = []
        scalars: list[np.ndarray] = []
        arrays: list[list[np.ndarray]] = []
        table: dict[str, int] = {}
        native_idx: list[int] = []
        native_len: list[int] = []
        native_seed: list[int] = []
        slots: list[tuple[int, int]] = []
        series: list[list[Any]] = [[None] * len(p.requests) for p in plans]
        flags: list[list[bool]] = [[False] * len(p.requests) for p in plans]
        errors: list[str | None] = [None] * len(plans)

        for pi, plan in enumerate(plans):
            for ri, (spec, rseed) in enumerate(plan.requests):
                source = self._source(spec)
                try:
                    if source.mode == "python":
                        series[pi][ri] = spec.draw(plan.total, self.freq, rseed)
                        continue
                    if source.mode == "batch":
                        k = table.get(spec._key)
                        if k is None:
                            k = table[spec._key] = len(gen_types)
                            gen_types.append(source.gen_type)
                            scalars.append(source.scalar)
                            arrays.append(list(source.arrays))
                    else:
                        gen = spec.build(plan.total, self.freq, rseed)
                        scalar, arr = _native_params(gen)
                        k = len(gen_types)
                        gen_types.append(source.gen_type)
                        scalars.append(scalar)
                        arrays.append(list(arr))
                except _DRAW_ERRORS as exc:
                    errors[pi] = f"{spec.name}: {exc}"
                    break
                native_idx.append(k)
                native_len.append(plan.total)
                native_seed.append(rseed)
                slots.append((pi, ri))

        if slots:
            values, status, messages = _rs_groups.draw_series(
                np.asarray(gen_types, dtype=np.int32),
                scalars,
                arrays,
                np.asarray(native_idx, dtype=np.uint32),
                np.asarray(native_len, dtype=np.int64),
                np.asarray(native_seed, dtype=np.uint64),
                workers,
            )
            offsets = np.concatenate([[0], np.cumsum(native_len)])
            with np.errstate(invalid="ignore"):
                bad = ~(np.abs(values) < MAX_ABS)  # also catches NaN and inf
            bad_series = np.add.reduceat(bad, offsets[:-1]) > 0
            for j, (pi, ri) in enumerate(slots):
                series[pi][ri] = values[offsets[j] : offsets[j + 1]]
                flags[pi][ri] = bool(status[j] == 1)
                if errors[pi] is not None:
                    continue
                if status[j] == 2:
                    spec = plans[pi].requests[ri][0]
                    errors[pi] = f"{spec.name}: {messages[j]}"
                elif bad_series[j]:
                    errors[pi] = "latent series not finite or too large"
        for pi, plan in enumerate(plans):
            if errors[pi] is not None:
                continue
            for (spec, _), s in zip(plan.requests, series[pi], strict=True):
                if self._source(spec).mode == "python" and not _ok(s):
                    errors[pi] = "latent series not finite or too large"
                    break
        return series, flags, errors

    def _compose(
        self,
        plans: list[_Plan],
        series: list[list[Any]],
        errors: list[str | None],
        workers: int,
    ) -> list[dict | None]:
        """Compose every regression target in one native call."""
        calibrate = self.mechanism.kind == "regression"
        keep = [i for i, e in enumerate(errors) if e is None]
        out: list[dict | None] = [None] * len(plans)
        if not keep:
            return out
        base, drivers, n_drivers, term_counts = [], [], [], []
        cols: dict[str, list] = defaultdict(list)
        per_group: dict[str, list] = defaultdict(list)
        for i in keep:
            d, ser = plans[i].data, series[i]
            base.append(d["base_fn"](ser))
            drv = d["driver_fn"](ser)
            drivers.append(drv.ravel())
            n_drivers.append(drv.shape[0])
            term_counts.append(len(d["terms"]))
            for t in d["terms"]:
                cols["kind"].append(_KIND_CODES[t[0]])
                cols["driver"].append(t[1])
                cols["partner"].append(-1 if t[2] is None else t[2])
                cols["lag"].append(t[3])
                cols["partner_lag"].append(t[4])
                cols["coef"].append(t[5])
                cols["param"].append(np.nan if t[6] is None else t[6])
                cols["adstock"].append(np.nan if t[7] is None else t[7])
            for key in ("persistence", "target_ar", "share", "log_std", "composition"):
                per_group[key].append(d[key])
        lens = [b.shape[0] for b in base]
        t_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        res = _rs_groups.compose_batch(
            np.concatenate(base),
            t_off,
            np.concatenate(drivers),
            np.asarray(n_drivers, dtype=np.int64),
            np.concatenate([[0], np.cumsum(term_counts)]).astype(np.int64),
            np.asarray(cols["kind"], dtype=np.uint8),
            np.asarray(cols["driver"], dtype=np.int64),
            np.asarray(cols["partner"], dtype=np.int64),
            np.asarray(cols["lag"], dtype=np.int64),
            np.asarray(cols["partner_lag"], dtype=np.int64),
            np.asarray(cols["coef"], dtype=np.float64),
            np.asarray(cols["param"], dtype=np.float64),
            np.asarray(cols["adstock"], dtype=np.float64),
            np.asarray(per_group["persistence"], dtype=np.float64),
            np.asarray(per_group["target_ar"], dtype=np.float64),
            np.asarray(per_group["composition"], dtype=np.int32),
            np.asarray(per_group["share"], dtype=np.float64),
            np.asarray(per_group["log_std"], dtype=np.float64),
            self.burn_in,
            calibrate,
            workers,
        )
        y, effect, cal_mean, cal_std, center, scale, calib, comp, ok, realized = res
        d_off = np.concatenate([[0], np.cumsum(n_drivers)])
        for j, i in enumerate(keep):
            out[i] = {
                "base": base[j],
                "y": y[t_off[j] : t_off[j + 1]],
                "effect": effect[t_off[j] : t_off[j + 1]],
                "cal_mean": cal_mean[d_off[j] : d_off[j + 1]],
                "cal_std": cal_std[d_off[j] : d_off[j + 1]],
                "center": float(center[j]),
                "scale": float(scale[j]),
                "calibration": _CALIBRATIONS[int(calib[j])],
                "composition": _COMPOSITIONS[int(comp[j])],
                "ok": bool(ok[j]),
                "realized": float(realized[j]),
            }
        return out

    def _emit(
        self, latent: np.ndarray, roles: list[str], order: np.ndarray
    ) -> tuple[np.ndarray, tuple[str, ...], list[int]]:
        """Drop the burn-in and put the channels in the planned order.

        Returns the emitted channels, their roles and, for every latent
        channel, its emitted position.
        """
        channels = latent[order, self.burn_in :]
        position = np.empty(len(order), dtype=np.int64)
        position[order] = np.arange(len(order))
        return channels, tuple(roles[i] for i in order), position.tolist()

    def _draw_length(self, rng: np.random.Generator) -> tuple[int, int]:
        length = int(rng.integers(self.length[0], self.length[1] + 1))
        return length, self.burn_in + length


# ---------------------------------------------------------------------------
# Mechanisms: plan (all random choices) and finish (assemble the group)
# ---------------------------------------------------------------------------


def _draw_coef(rng: np.random.Generator) -> float:
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return sign * float(rng.uniform(*_COEF_RANGE))


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


def _seeds(rng: np.random.Generator, n: int) -> list[int]:
    return [int(s) for s in rng.integers(0, 2**63, size=n)]


def _stack_drivers(
    ser: list[np.ndarray], start: int, stop: int, total: int
) -> np.ndarray:
    if stop <= start:
        return np.empty((0, total))
    return np.stack(ser[start:stop])


def _plan_regression(
    prior: GroupPrior,
    spec: RegressionSpec,
    rng: np.random.Generator,
    gid: int,
    attempt: int,
) -> _Plan:
    length, total = prior._draw_length(rng)
    n_cov = int(rng.integers(spec.n_covariates[0], spec.n_covariates[1] + 1))
    is_null = bool(rng.random() < spec.p_null)
    n_inf = 0
    if not is_null and n_cov > 0:
        n_inf = min(
            n_cov, int(rng.integers(spec.n_informative[0], spec.n_informative[1] + 1))
        )
    target_spec = spec.target_pool[int(rng.integers(len(spec.target_pool)))]
    driver_specs = [
        spec.driver_pool[int(rng.integers(len(spec.driver_pool)))] for _ in range(n_cov)
    ]
    roles = [
        "future" if rng.random() < spec.p_known_future else "past" for _ in range(n_cov)
    ]
    seeds = _seeds(rng, 1 + n_cov)

    # [kind, driver, partner, lag, partner_lag, coef, param, adstock]
    terms: list[list[Any]] = []
    for j in range(n_inf):
        kind = spec.edge_kinds[int(rng.integers(len(spec.edge_kinds)))]
        param = None
        if kind == "saturating":
            lo, hi = np.log(_SOFTNESS_RANGE)
            param = float(np.exp(rng.uniform(lo, hi)))
        elif kind in ("threshold", "hinge"):
            param = float(rng.uniform(*_THRESHOLD_RANGE))
        terms.append(
            [kind, j, None, _draw_lag(spec.max_lag, rng), 0, _draw_coef(rng), param]
        )
    if n_inf >= 2 and rng.random() < spec.p_interaction:
        a, w = (int(v) for v in rng.choice(n_inf, size=2, replace=False))
        kind = _INTERACTION_KINDS[int(rng.integers(len(_INTERACTION_KINDS)))]
        terms.append(
            [
                kind,
                a,
                w,
                _draw_lag(spec.max_lag, rng),
                _draw_lag(spec.max_lag, rng),
                _draw_coef(rng),
                float(rng.uniform(*_THRESHOLD_RANGE)) if kind == "gate" else None,
            ]
        )
    for t in terms:
        t.append(
            float(rng.uniform(*spec.adstock_decay))
            if rng.random() < spec.p_adstock
            else None
        )
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
    multiplicative = bool(target_spec.positive and rng.random() < spec.p_multiplicative)
    share = float(rng.uniform(*spec.effect_share))
    log_std = float(rng.uniform(*spec.log_effect_std))
    order = rng.permutation(1 + n_cov)

    requests = [(target_spec, seeds[0])] + [
        (driver_specs[j], seeds[1 + j]) for j in range(n_cov)
    ]
    data = {
        "n_inf": n_inf,
        "target_spec": target_spec,
        "driver_specs": driver_specs,
        "roles": roles,
        "seeds": seeds,
        "terms": [tuple(t) for t in terms],
        "persistence": persistence,
        "target_ar": target_ar,
        "composition": int(multiplicative),
        "share": share,
        "log_std": log_std,
        "order": order,
        "request_channels": list(range(1 + n_cov)),
        "base_fn": lambda ser: ser[0],
        "driver_fn": lambda ser: _stack_drivers(ser, 1, 1 + n_cov, total),
    }
    return _Plan(gid, attempt, length, total, requests, data)


def _plan_fixed(
    prior: GroupPrior,
    spec: FixedRegressionSpec,
    rng: np.random.Generator,
    gid: int,
    attempt: int,
) -> _Plan:
    length, total = prior._draw_length(rng)
    n_inf = len(spec.drivers)
    specs = list(spec.drivers) + [
        spec.drivers[k % n_inf] for k in range(spec.n_distractors)
    ]
    n_drv = len(specs)
    seeds = _seeds(rng, n_drv)
    base_seed = int(rng.integers(0, 2**63)) if spec.target is not None else None
    noise = rng.normal(0.0, spec.noise_std, total) if spec.noise_std else None
    order = rng.permutation(1 + n_drv)
    requests = list(zip(specs, seeds, strict=True))
    if spec.target is not None:
        requests.append((spec.target, base_seed))

    def base_fn(ser: list[np.ndarray]) -> np.ndarray:
        base = ser[n_drv] if spec.target is not None else np.zeros(total)
        base = base + spec.intercept
        return base + noise if noise is not None else base

    terms = [
        (t.kind, t.driver, t.partner, t.lag, t.partner_lag, t.coef, t.param, None)
        for t in spec.terms
    ]
    data = {
        "specs": specs,
        "seeds": seeds,
        "base_seed": base_seed,
        "terms": terms,
        "persistence": 0.0,
        "target_ar": spec.target_ar,
        "composition": 0,
        "share": 0.0,
        "log_std": 0.0,
        "order": order,
        "request_channels": list(range(1, n_drv + 1))
        + ([0] if spec.target is not None else []),
        "base_fn": base_fn,
        "driver_fn": lambda ser: _stack_drivers(ser, 0, n_drv, total),
    }
    return _Plan(gid, attempt, length, total, requests, data)


def _term_records(
    terms: list[tuple], positions: list[int], offset: int
) -> list[dict[str, Any]]:
    """Recipe entries for terms whose drivers are recipe channels
    ``offset + driver``."""
    out = []
    for kind, driver, partner, lag, partner_lag, coef, param, adstock in terms:
        out.append(
            {
                "kind": kind,
                "driver": offset + driver,
                "partner": None if partner is None else offset + partner,
                "lag": lag,
                "partner_lag": partner_lag,
                "coef": coef,
                "param": param,
                "adstock": adstock,
                "driver_channel": positions[offset + driver],
                "partner_channel": None
                if partner is None
                else positions[offset + partner],
            }
        )
    return out


def _regression_parts(comp: dict, burn_in: int, components: bool) -> dict | None:
    if not components:
        return None
    return {
        "base": comp["base"][burn_in:].copy(),
        "effect": comp["effect"][burn_in:].copy(),
    }


def _finish_regression(
    prior: GroupPrior,
    spec: RegressionSpec,  # noqa: ARG001 (shared finisher signature)
    plan: _Plan,
    series: list[np.ndarray],
    comp: dict | None,
    recipes: bool,
    components: bool,
):
    if comp is None or not comp["ok"]:
        return "composed target not finite or too large"
    d = plan.data
    b = prior.burn_in
    latent = np.vstack([comp["y"][None], *[s[None] for s in series[1:]]])
    roles = ["target", *d["roles"]]
    channels, out_roles, position = prior._emit(latent, roles, d["order"])
    parts = _regression_parts(comp, b, components)
    if not recipes:
        return channels, out_roles, {}, parts
    multiplicative = comp["composition"] == "multiplicative"
    recipe = {
        "length": plan.length,
        "burn_in": b,
        "null": d["n_inf"] == 0,
        "driver_scale": "standardized",
        "channels": [
            {
                "channel": position[0],
                "role": "target",
                "kind": "target",
                "spec": d["target_spec"].describe(),
                "seed": d["seeds"][0],
            },
            *[
                {
                    "channel": position[1 + j],
                    "role": d["roles"][j],
                    "kind": "covariate",
                    "informative": j < d["n_inf"],
                    "spec": d["driver_specs"][j].describe(),
                    "seed": d["seeds"][1 + j],
                    "calibration_mean": float(comp["cal_mean"][j]),
                    "calibration_std": float(comp["cal_std"][j]),
                }
                for j in range(len(d["roles"]))
            ],
        ],
        "terms": _term_records(d["terms"], position, offset=1),
        "persistence": d["persistence"],
        "target_ar": d["target_ar"],
        "composition": comp["composition"],
        "fallback": "additive"
        if d["composition"] == 1 and not multiplicative
        else None,
        "calibration": comp["calibration"],
        "effect_center": comp["center"],
        "effect_scale": comp["scale"],
        "requested_share": None if multiplicative else d["share"],
        "requested_log_std": d["log_std"] if multiplicative else None,
        "realized": comp["realized"],
    }
    return channels, out_roles, recipe, parts


def _finish_fixed(
    prior: GroupPrior,
    spec: FixedRegressionSpec,
    plan: _Plan,
    series: list[np.ndarray],
    comp: dict | None,
    recipes: bool,
    components: bool,
):
    if comp is None or not comp["ok"]:
        return "composed target not finite or too large"
    d = plan.data
    b = prior.burn_in
    n_drv = len(d["specs"])
    latent = np.vstack([comp["y"][None], *[s[None] for s in series[:n_drv]]])
    roles = ["target"] + [spec.role] * n_drv
    channels, out_roles, position = prior._emit(latent, roles, d["order"])
    parts = _regression_parts(comp, b, components)
    if not recipes:
        return channels, out_roles, {}, parts
    used = {i for t in spec.terms for i in t.drivers()}
    recipe = {
        "length": plan.length,
        "burn_in": b,
        "null": not spec.terms,
        "driver_scale": "raw",
        "channels": [
            {
                "channel": position[0],
                "role": "target",
                "kind": "target",
                "spec": None if spec.target is None else spec.target.describe(),
                "seed": d["base_seed"],
            },
            *[
                {
                    "channel": position[1 + j],
                    "role": spec.role,
                    "kind": "covariate",
                    "informative": j in used,
                    "spec": d["specs"][j].describe(),
                    "seed": d["seeds"][j],
                }
                for j in range(n_drv)
            ],
        ],
        "terms": _term_records(d["terms"], position, offset=1),
        "intercept": spec.intercept,
        "noise_std": spec.noise_std,
        "target_ar": spec.target_ar,
        "fallback": None,
    }
    return channels, out_roles, recipe, parts


def _plan_coupled(
    prior: GroupPrior,
    spec: CoupledSpec,
    rng: np.random.Generator,
    gid: int,
    attempt: int,
) -> _Plan:
    length, total = prior._draw_length(rng)
    n = int(rng.integers(spec.n_channels[0], spec.n_channels[1] + 1))
    base_spec = spec.base_pool[int(rng.integers(len(spec.base_pool)))]
    seeds = _seeds(rng, n)
    mixing = None
    if n >= 2 and "mixing" in spec.couplings:
        strength = float(rng.uniform(*spec.mixing_strength))
        mixing = (strength, rng.normal(0.0, 1.0, (n, n)))
    leadlag = []
    if n >= 2 and "leadlag" in spec.couplings:
        chosen = rng.random(n - 1) < spec.p_leadlag
        if not chosen.any():
            chosen[-1] = True
        for j in range(1, n):
            if not chosen[j - 1]:
                continue
            leadlag.append(
                {
                    "src": int(rng.integers(0, j)),
                    "dst": j,
                    "lag": int(rng.integers(spec.lag_range[0], spec.lag_range[1] + 1)),
                    "sign": 1.0 if rng.random() < 0.5 else -1.0,
                    "noise": float(rng.uniform(*spec.noise_scale)),
                    "eps": rng.normal(size=total),
                }
            )
    if spec.targets == "all":
        roles = ["target"] * n
        target = None
    else:
        target = int(rng.integers(n))
        roles = [
            "target"
            if j == target
            else ("future" if rng.random() < spec.p_known_future else "past")
            for j in range(n)
        ]
    order = rng.permutation(n)
    data = {
        "base_spec": base_spec,
        "seeds": seeds,
        "mixing": mixing,
        "leadlag": leadlag,
        "roles": roles,
        "target": target,
        "order": order,
        "request_channels": list(range(n)),
    }
    return _Plan(gid, attempt, length, total, [(base_spec, s) for s in seeds], data)


def _finish_coupled(
    prior: GroupPrior,
    spec: CoupledSpec,  # noqa: ARG001 (shared finisher signature)
    plan: _Plan,
    series: list[np.ndarray],
    comp: dict | None,  # noqa: ARG001 (shared finisher signature)
    recipes: bool,
    components: bool,  # noqa: ARG001 (shared finisher signature)
):
    d = plan.data
    b = prior.burn_in
    raw = np.stack(series)
    positive = d["base_spec"].positive
    if positive:
        # Couple log sizes; zeros take the calibration-window mean log size
        occurs = raw > 0
        logs = np.log(np.where(occurs, raw, 1.0))
        fill = np.array(
            [
                logs[i, :b][occurs[i, :b]].mean() if occurs[i, :b].any() else 0.0
                for i in range(raw.shape[0])
            ]
        )
        work = np.where(occurs, logs, fill[:, None])
    else:
        work = raw
    z, level, scale = _calibrated(work, b)
    n = raw.shape[0]
    coupling: dict[str, Any] = {}
    if d["mixing"] is not None:
        strength, a = d["mixing"]
        gram = a @ a.T
        diag = np.sqrt(np.diag(gram))
        target = (1.0 - strength) * np.eye(n) + strength * gram / np.outer(diag, diag)
        mix = np.linalg.cholesky(target)
        z = _calibrated(mix @ z, b)[0]
        coupling["mixing"] = {"strength": strength, "matrix": mix.tolist()}
    root = list(range(n))
    ancestors: list[set[int]] = [set() for _ in range(n)]
    if d["leadlag"]:
        pairs = []
        for pair in d["leadlag"]:
            src, dst, lag = pair["src"], pair["dst"], pair["lag"]
            lead = pair["sign"] * _lagged(z[src], lag) + pair["noise"] * pair["eps"]
            z[dst] = _calibrated(lead[None], b)[0][0]
            root[dst] = root[src]
            ancestors[dst] = ancestors[src] | {src}
            pairs.append({k: v for k, v in pair.items() if k != "eps"})
        coupling["leadlag"] = pairs
    out = z * scale[:, None] + level[:, None]
    if positive:
        with np.errstate(over="ignore"):
            out = np.where(occurs, np.exp(out), 0.0)
    if not _ok(out[:, b:]):
        return "coupled channels not finite or too large"
    channels, out_roles, position = prior._emit(out, d["roles"], d["order"])
    if not recipes:
        return channels, out_roles, {}, None
    target = d["target"]
    entries = []
    for j in range(n):
        entry = {
            "channel": position[j],
            "role": d["roles"][j],
            "kind": "channel",
            "seed": d["seeds"][j],
            "calibration_mean": float(level[j]),
            "calibration_std": float(scale[j]),
            "leadlag_root": root[j],
        }
        if target is not None:
            entry["derived_from_target"] = target in ancestors[j]
        entries.append(entry)
    recipe = {
        "length": plan.length,
        "burn_in": b,
        "null": None,
        "spec": d["base_spec"].describe(),
        "space": "log" if positive else "level",
        "channels": entries,
        "coupling": coupling,
        "fallback": None,
    }
    return channels, out_roles, recipe, None


def _varx_lag_matrices(
    spec: VARXSpec, n: int, clusters: np.ndarray | None, rng: np.random.Generator
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
    return np.stack(stabilize(mats, VAR_STABILITY))


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


def _plan_varx(
    prior: GroupPrior,
    spec: VARXSpec,
    rng: np.random.Generator,
    gid: int,
    attempt: int,
) -> _Plan:
    length, total = prior._draw_length(rng)
    n = int(rng.integers(spec.n_targets[0], spec.n_targets[1] + 1))
    clusters = None if spec.n_clusters is None else np.arange(n) * spec.n_clusters // n
    if spec.coef_matrices is not None:
        lag_matrices = np.asarray(spec.coef_matrices, dtype=np.float64)
    else:
        lag_matrices = _varx_lag_matrices(spec, n, clusters, rng)
    data = {
        "n": n,
        "clusters": clusters,
        "lag_matrices": lag_matrices,
        "intercept": np.zeros(n)
        if spec.intercept is None
        else np.asarray(spec.intercept, dtype=np.float64),
        "covariance": np.eye(n)
        if spec.innovation_covariance is None
        else np.asarray(spec.innovation_covariance, dtype=np.float64),
        "shared_effects": rng.normal(0.0, spec.exog_scale, (n, spec.n_shared)),
        "future_effects": _varx_effects(spec, spec.n_future, n, rng),
        "past_effects": _varx_effects(spec, spec.n_past, n, rng),
        "shared_shocks": rng.standard_normal((total, spec.n_shared)),
        "future_shocks": rng.standard_normal((total, n, spec.n_future)),
        "past_shocks": rng.standard_normal((total, n, spec.n_past)),
        "innovations": rng.standard_normal((total, n)),
    }
    n_channels = n + spec.n_shared + n * (spec.n_future + spec.n_past)
    data["order"] = rng.permutation(n_channels)
    return _Plan(gid, attempt, length, total, [], data)


def _finish_varx(
    prior: GroupPrior,
    spec: VARXSpec,
    plan: _Plan,
    series: list[np.ndarray],  # noqa: ARG001 (shared finisher signature)
    comp: dict | None,  # noqa: ARG001 (shared finisher signature)
    recipes: bool,
    components: bool,  # noqa: ARG001 (shared finisher signature)
):
    d = plan.data
    n, total, phi = d["n"], plan.total, spec.exog_persistence
    shared = _unit_ar1(d["shared_shocks"], phi)
    future = _unit_ar1(d["future_shocks"], phi)
    past = _unit_ar1(d["past_shocks"], phi)
    innovations = d["innovations"] @ np.linalg.cholesky(d["covariance"]).T
    drive = d["intercept"] + shared @ d["shared_effects"].T + innovations
    if spec.n_future:
        drive += np.einsum("kij,tjk->ti", d["future_effects"], future)
    if spec.n_past:
        drive[1:] += np.einsum("kij,tjk->ti", d["past_effects"], past[:-1])
    y = _rs_groups.var_filter(
        np.ascontiguousarray(drive).ravel(),
        n,
        np.ascontiguousarray(d["lag_matrices"]).ravel(),
        spec.lag_order,
    ).reshape(total, n)
    if not _ok(y[prior.burn_in :]):
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
    for kind, values, role in (("future", future, "future"), ("past", past, "past")):
        for i in range(n):
            for k in range(values.shape[2]):
                latent.append(values[:, i, k])
                layout.append({"kind": kind, "series": i, "index": k})
                roles.append(role)
    channels, out_roles, position = prior._emit(np.stack(latent), roles, d["order"])
    if not recipes:
        return channels, out_roles, {}, None
    for j, entry in enumerate(layout):
        entry["channel"] = position[j]
        entry["role"] = roles[j]
    recipe = {
        "length": plan.length,
        "burn_in": prior.burn_in,
        "null": spec.n_shared + spec.n_future + spec.n_past == 0,
        "channels": layout,
        "process": _jsonable(
            {
                "lag_matrices": d["lag_matrices"],
                "intercept": d["intercept"],
                "shared_effects": d["shared_effects"],
                "future_effects": d["future_effects"],
                "past_effects": d["past_effects"],
                "innovation_covariance": d["covariance"],
                "exog_persistence": phi,
                "clusters": d["clusters"],
            }
        ),
        "fallback": None,
    }
    return channels, out_roles, recipe, None


_PLANNERS = {
    "regression": _plan_regression,
    "fixed": _plan_fixed,
    "coupled": _plan_coupled,
    "varx": _plan_varx,
}
_FINISHERS = {
    "regression": _finish_regression,
    "fixed": _finish_fixed,
    "coupled": _finish_coupled,
    "varx": _finish_varx,
}


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

    The task design follows the style of the controlled covariate study of
    Berthelier et al. (2026), https://arxiv.org/abs/2605.12200 (TabPFN-TS vs
    Chronos-2); the exact relations, ranges and driver processes are
    SynForecast's own and not a reproduction of it.

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
