"""Simons-parity transform layer and family weights for synthetic training streams.

Array-level port of the transform layer that the Simons TSFM recipe runs on
every synthetic draw (``simons/synth/augment.py``, ``base.py: post`` and
``observe.py``), for loaders that consume SynForecast draws as numpy rows:

- T1 :class:`BatchAugment` (mix, season, filter, lattice) acts on a batch,
  because mix takes donors from other rows of the same batch.
- T2 :class:`Noisers` (Poisson counts, lognormal shocks, Gaussian spikes).
- T3 :class:`RandomUnits` (group-shared time flip, sign, scale, positive
  shift or offset) and :class:`Gaps` (short holes, last patch observed).
- T4 :class:`DutyCycle` (Markov or periodic on/off gating).

:class:`SimonsParity` composes them per stage (:meth:`SimonsParity.from_stage`):
``batch_augment`` standardises each row on its finite values and applies T1;
``series_transforms`` applies ``post`` in Simons' order (time flip, noisers,
units, duty, gaps). Short-history padding is left to the loader.

NaN marks a missing value and stays missing: every statistic (mean, standard
deviation, minimum, maximum, order statistic) reads finite values only, and
no output position that was NaN is filled. Linear operations that need a
value at a missing position (mix, filter, cumulative sum) read the row mean
(0 after standardisation) there; a difference reads the last observed value.
Count rows (T2 counts) bypass units and duty, as in Simons.

Distributions follow the Simons stage configs; random streams differ (numpy
instead of torch), so outputs match in law, not bitwise.
"""

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from synforecast import _families

PARITY_VERSION = "simons-parity-v1"
"""Version tag of the transform semantics and family weights below."""

STAGES = ("stage1", "stage2", "stage3", "stage4", "stage4-long")
CATEGORIES = ("mix", "season", "filter", "lattice")
DAY_STEPS = (24.0, 96.0, 288.0, 48.0, 1.0)
DUTY_PERIODS = (
    24.0,
    24.0,
    48.0,
    96.0,
    96.0,
    144.0,
    288.0,
    168.0,
    12.0,
    7.0,
    52.0,
    60.0,
)
NOISE_KINDS = ("count", "count_lognormal", "shock", "spike")

SIMONS_MEMBERS = (
    "CauKerGenerator",
    "PolyrhythmGenerator",
    "AMFMWaveGenerator",
    "ForecastPFNGenerator",
    "EpidemicGenerator",
    "FluidQueueGenerator",
    "RainfallRunoffGenerator",
    "ThermalGenerator",
)
"""Generators appended by ``pretraining_pool(include_simons_parity=True)``, in pool order."""

SIMONS_FAMILY_WEIGHTS = {
    "stage1": {
        "kernel": 0.21056,
        "sarima": 0.21056,
        "sde": 0.12024,
        "step": 0.05512,
        "spike": 0.05512,
        "cauker": 0.08016,
        "rhythm": 0.04008,
        "wave": 0.03008,
        "fpfn": 0.03008,
        "ets": 0.068,
        "chaos": 0.024,
        "epi": 0.014,
        "queue": 0.014,
        "hydro": 0.009,
        "thermal": 0.009,
        "levy": 0.015,
        "pulse": 0.015,
    },
    "stage3": {
        "kernel": 0.14,
        "sarima": 0.14,
        "sde": 0.06,
        "step": 0.03,
        "spike": 0.03,
        "cauker": 0.04,
        "rhythm": 0.02,
        "wave": 0.02,
        "fpfn": 0.02,
    },
}
"""Simons-25M ``synthetic_family_weights`` (stage 2 uses stage 1's, stages 4 and 4-long stage 3's)."""

FAMILY_MEMBERS = {
    "cauker": "CauKerGenerator",
    "rhythm": "PolyrhythmGenerator",
    "wave": "AMFMWaveGenerator",
    "fpfn": "ForecastPFNGenerator",
    "epi": "EpidemicGenerator",
    "queue": "FluidQueueGenerator",
    "hydro": "RainfallRunoffGenerator",
    "thermal": "ThermalGenerator",
}
"""Simons family name of each ported generator."""


def _family_stage(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    return "stage1" if stage in ("stage1", "stage2") else "stage3"


def member_weights(stage: str) -> dict[str, float]:
    """Share of synthetic draws of each ported generator at ``stage``, by class name.

    Each family keeps its normalised Simons weight (its weight over the sum of
    the stage's weights): 22.64% in total in stages 1 and 2, 20% in stages 3,
    4 and 4-long, where the mechanistic families have no weight. The rest of
    the stream is the caller's.
    """
    weights = SIMONS_FAMILY_WEIGHTS[_family_stage(stage)]
    total = sum(weights.values())
    return {
        member: weights.get(family, 0.0) / total
        for family, member in FAMILY_MEMBERS.items()
    }


def regime_ou_share(stage: str) -> float:
    """Normalised Simons ``sde`` (regime-switching OU) share of synthetic draws at ``stage``."""
    weights = SIMONS_FAMILY_WEIGHTS[_family_stage(stage)]
    return weights["sde"] / sum(weights.values())


def exposure_counters() -> dict[str, dict[str, int]]:
    """Process-wide mechanistic simulation counters: draws, redraws (invalid paths) and fallbacks."""
    return {
        name: dict(counts) for name, counts in _families.MECHANISTIC_COUNTERS.items()
    }


def reset_exposure_counters() -> None:
    """Zero the process-wide mechanistic simulation counters."""
    for counts in _families.MECHANISTIC_COUNTERS.values():
        for key in counts:
            counts[key] = 0


def _rows(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a 2-D [n, length] array")
    return np.where(np.isinf(values), np.nan, values)


def _groups(group: np.ndarray | None, n: int) -> np.ndarray:
    if group is None:
        return np.arange(n)
    group = np.asarray(group)
    if group.shape != (n,) or not np.issubdtype(group.dtype, np.integer):
        raise ValueError("group must be an integer array with one entry per row")
    _, index = np.unique(group, return_inverse=True)
    return index.reshape(-1)


def _row_mean(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    count = np.maximum(observed.sum(-1, keepdims=True), 1)
    return np.where(observed, x, 0.0).sum(-1, keepdims=True) / count


def _row_std(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    count = observed.sum(-1, keepdims=True)
    centred = np.where(observed, x - _row_mean(x, observed), 0.0)
    var = (centred**2).sum(-1, keepdims=True) / np.maximum(count - 1, 1)
    return np.where(count >= 2, np.sqrt(var), 1.0)


def _row_min(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    low = np.where(observed, x, np.inf).min(-1, keepdims=True)
    return np.where(np.isfinite(low), low, 0.0)


def _row_max(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    high = np.where(observed, x, -np.inf).max(-1, keepdims=True)
    return np.where(np.isfinite(high), high, 0.0)


def standardise(values: np.ndarray) -> np.ndarray:
    """Rows standardised on their finite values (std floored at 1e-6); NaN stays NaN, inf becomes NaN."""
    x = _rows(values)
    observed = np.isfinite(x)
    out = (x - _row_mean(x, observed)) / np.maximum(_row_std(x, observed), 1e-6)
    return np.where(observed, out, np.nan)


def _standardise_filled(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    y = np.where(observed & np.isfinite(x), x, np.nan)
    ok = np.isfinite(y)
    out = (y - _row_mean(y, ok)) / np.maximum(_row_std(y, ok), 1e-6)
    return np.where(ok, out, 0.0)


def _forward_fill(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    n, length = x.shape
    index = np.where(observed, np.arange(length)[None], 0)
    np.maximum.accumulate(index, axis=1, out=index)
    return np.where(
        observed.any(-1, keepdims=True), x[np.arange(n)[:, None], index], 0.0
    )


def _first_observed(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    first = np.argmax(observed, -1)
    return x[np.arange(len(x)), first][:, None] * observed.any(-1, keepdims=True)


class BatchAugment(BaseModel):
    """T1: per-series augmentation of a standardised batch (Simons ``augment.apply``).

    A share ``rate`` of rows takes one category, or two without replacement
    with probability ``second_rate``, by Gumbel top-k on ``weights``. In the
    fixed order mix, season, filter, lattice, each chosen category replaces
    the row by its re-standardised output:

    - mix: a convex combination of the row and one or two donors (other
      rows of the batch), Dirichlet(1.5) weights, crossing over to a second
      weight vector along a sigmoid on half of mixes;
    - season: the row shifted positive times a weekend square wave (day of
      24, 96, 288, 48 or 1 steps) or a slow sine;
    - filter: a random smoothing (70%) or signed FIR filter of 2 to
      ``taps`` taps, replaced by a first difference (15%) or a cumulative
      sum (15%);
    - lattice: quantisation to a random grid (60%) or censoring below a
      random lower quantile (40%).

    With ``group``, activation, categories and every per-row parameter are
    shared within a group (donors stay per row).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rate: float = Field(default=0.4, ge=0, le=1)
    second_rate: float = Field(default=0.3, ge=0, le=1)
    weights: tuple[float, float, float, float] = (0.35, 0.25, 0.2, 0.2)
    taps: int = Field(default=32, ge=2)

    @model_validator(mode="after")
    def validate_weights(self) -> "BatchAugment":
        if min(self.weights) < 0 or sum(self.weights) <= 0:
            raise ValueError("weights must be nonnegative with a positive sum")
        return self

    def apply(
        self, values: np.ndarray, group: np.ndarray | None, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardise rows on finite values, then augment; returns (values, chosen [n, 4] bool in CATEGORIES order)."""
        x = _rows(values)
        n, length = x.shape
        observed = np.isfinite(x)
        z = _standardise_filled(x, observed).copy()
        g = _groups(group, n)

        def share(v):
            return v[g]

        active = share(rng.random((n, 1)) < self.rate)
        count = share(np.where(rng.random((n, 1)) < self.second_rate, 2, 1))
        w = np.asarray(self.weights, dtype=np.float64)[None]
        gumbel = -np.log(-np.log(np.clip(rng.random((n, 4)), 1e-12, 1 - 1e-7)))
        keys = share(np.log(np.maximum(w, 1e-12)) + gumbel)
        rank = np.argsort(np.argsort(-keys, -1, kind="stable"), -1, kind="stable")
        chosen = active & (rank < count) & (w > 0)
        makers = (self._mix, self._season, self._filter, self._lattice)
        for col, maker in enumerate(makers):
            idx = np.flatnonzero(chosen[:, col])
            if not len(idx):
                continue
            y = _standardise_filled(
                maker(z, observed, share, rng, length, idx), observed[idx]
            )
            z[idx] = y
        return np.where(observed, z, np.nan), chosen

    @staticmethod
    def _mix(z, _observed, share, rng, length, idx):
        n = len(z)
        p1, p2 = rng.permutation(n)[idx], rng.permutation(n)[idx]
        three = share(rng.random((n, 1)) < 0.5)[idx]
        two = np.array([1.0, 1.0, 0.0])
        wa = share(rng.gamma(1.5, size=(n, 3)))[idx]
        wb = share(rng.gamma(1.5, size=(n, 3)))[idx]
        wa = np.where(three, wa, wa * two)
        wb = np.where(three, wb, wb * two)
        wa, wb = wa / wa.sum(-1, keepdims=True), wb / wb.sum(-1, keepdims=True)
        t = np.arange(length, dtype=np.float64)[None] / length
        centre = share(_families.u(rng, (n, 1), 0.2, 0.8))[idx]
        width = share(_families.logu(rng, (n, 1), 0.01, 0.2))[idx]
        s = _families.sigmoid((t - centre) / width)
        s = np.where(share(rng.random((n, 1)) < 0.5)[idx], s, 0.0)
        y = (wa[:, :1] * (1 - s) + wb[:, :1] * s) * z[idx]
        y = y + (wa[:, 1:2] * (1 - s) + wb[:, 1:2] * s) * z[p1]
        return y + (wa[:, 2:] * (1 - s) + wb[:, 2:] * s) * z[p2]

    @staticmethod
    def _season(z, observed, share, rng, length, idx):
        n = len(z)
        t = np.arange(length, dtype=np.float64)[None]
        c = share(_families.u(rng, (n, 1), 0.2, 2.0))[idx]
        a = share(_families.u(rng, (n, 1), 0.2, 0.8))[idx]
        day = share(_families.choice(rng, DAY_STEPS, (n, 1)))[idx]
        phase = share(rng.random((n, 1)))[idx]
        weekend = np.remainder(t / (7 * day) + phase, 1.0) >= 5 / 7
        up = share(np.where(rng.random((n, 1)) < 0.25, 1.0, -1.0))[idx]
        square = 1 + up * a * weekend
        cycles = share(_families.u(rng, (n, 1), 1.0, 4.0))[idx]
        sine = 1 + a * np.sin(2 * math.pi * (cycles * t / length + phase))
        multiplier = np.where(share(rng.random((n, 1)) < 0.5)[idx], square, sine)
        return (z[idx] - _row_min(z[idx], observed[idx]) + c) * multiplier

    def _filter(self, z, observed, share, rng, _length, idx):
        n = len(z)
        taps = self.taps
        size = share(rng.integers(2, taps + 1, (n, 1)))
        live = np.arange(taps)[None] < size
        smooth_w = -np.log(np.maximum(rng.random((n, taps)), 1e-12)) * live
        smooth_w = smooth_w / smooth_w.sum(-1, keepdims=True)
        signed_w = rng.standard_normal((n, taps)) * live
        signed_w = signed_w / np.maximum(np.abs(signed_w).sum(-1, keepdims=True), 1e-6)
        w = share(np.where(rng.random((n, 1)) < 0.7, smooth_w, signed_w))[idx]
        kind = share(rng.random((n, 1)))[idx, 0]
        x, ok = z[idx], observed[idx]
        y = np.empty_like(x)
        rows = kind < 0.15
        if rows.any():
            held = _forward_fill(x[rows], ok[rows])
            y[rows] = np.concatenate(
                [np.zeros((int(rows.sum()), 1)), np.diff(held, axis=-1)], -1
            )
        rows = (kind >= 0.15) & (kind < 0.3)
        if rows.any():
            y[rows] = np.cumsum(x[rows], -1)
        first = _first_observed(x, ok)[:, 0]
        for r in np.flatnonzero(kind >= 0.3):
            padded = np.concatenate([np.full(taps - 1, first[r]), x[r]])
            y[r] = np.convolve(padded, w[r], mode="valid")
        return y

    @staticmethod
    def _lattice(z, observed, share, rng, length, idx):
        n = len(z)
        delta = share(_families.logu(rng, (n, 1), 0.05, 0.5))[idx]
        offset = share(rng.random((n, 1)))[idx]
        q = share(_families.u(rng, (n, 1), 0.05, 0.5))[idx]
        censor = share(rng.random((n, 1)) < 0.4)[idx]
        x, ok = z[idx], observed[idx]
        out = delta * np.floor(x / delta + offset)
        count = ok.sum(-1)
        position = np.minimum(
            (q[:, 0] * np.maximum(count - 1, 0)).astype(np.int64), length - 1
        )
        for r in np.flatnonzero(censor[:, 0]):
            values = np.where(ok[r], x[r], np.inf)
            floor = np.partition(values, position[r])[position[r]]
            out[r] = np.maximum(x[r], floor if np.isfinite(floor) else -np.inf)
        return out


class Noisers(BaseModel):
    """T2: observation noisers on a share ``rate`` of rows (Simons ``_noisers``).

    Each row draws one of four kinds uniformly: Poisson counts with rate
    LogU[0.05, 50] times the min-max scaled row; the same with a lognormal
    (sd 0.5) rate modulation; multiplicative lognormal shocks
    ``(unit + 0.1) * exp(sigma * z)`` with sigma U[0.05, 0.5]; or spikes at
    rate 1% with Gaussian amplitude of scale U[3, 10]. Count rows are
    flagged so later transforms leave their values alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rate: float = Field(default=0.3, ge=0, le=1)
    count_rate: tuple[float, float] = (0.05, 50.0)
    shock_sigma: tuple[float, float] = (0.05, 0.5)
    spike_rate: float = Field(default=0.01, ge=0, le=1)
    spike_scale: tuple[float, float] = (3.0, 10.0)

    def apply(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (values, is_count [n, 1] bool, kind [n] int: -1 inactive, else index into NOISE_KINDS)."""
        n, length = x.shape
        kind = rng.integers(0, 4, n)
        active = rng.random(n) < self.rate
        kind = np.where(active, kind, -1)
        out = x.copy()
        rows = np.flatnonzero(kind == 0)
        if len(rows):
            out[rows] = self._counts(x[rows], rng, False)
        rows = np.flatnonzero(kind == 1)
        if len(rows):
            out[rows] = self._counts(x[rows], rng, True)
        rows = np.flatnonzero(kind == 2)
        if len(rows):
            sigma = _families.u(rng, (len(rows), 1), *self.shock_sigma)
            noise = np.exp(sigma * rng.standard_normal((len(rows), length)))
            out[rows] = (self._unit(x[rows]) + 0.1) * noise
        rows = np.flatnonzero(kind == 3)
        if len(rows):
            scale = _families.u(rng, len(rows), *self.spike_scale)
            hit_row, hit_t = np.nonzero(
                rng.random((len(rows), length)) < self.spike_rate
            )
            spikes = rng.standard_normal(len(hit_row)) * scale[hit_row]
            out[rows[hit_row], hit_t] += spikes
        out = np.where(np.isfinite(x), out, x)
        return out, (kind >= 0)[:, None] & (kind <= 1)[:, None], kind

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        observed = np.isfinite(x)
        low, high = _row_min(x, observed), _row_max(x, observed)
        return np.where(observed, (x - low) / np.maximum(high - low, 1e-6), 0.0)

    def _counts(
        self, x: np.ndarray, rng: np.random.Generator, lognormal: bool
    ) -> np.ndarray:
        rate = _families.logu(rng, (len(x), 1), *self.count_rate) * self._unit(x)
        if lognormal:
            rate = rate * np.exp(0.5 * rng.standard_normal(x.shape))
        return rng.poisson(np.clip(rate, 0, 1e9)).astype(np.float64)


class RandomUnits(BaseModel):
    """T3: random time flip, sign, scale and level (Simons ``post``).

    The time flip (rate ``time_flip``) is shared within a group and runs
    first, before the noisers. The sign flips at ``sign_flip``; a share
    ``positive`` of rows is shifted to a zero minimum, the rest offset by
    U[-offset, offset]; then rows are scaled by LogU[``scale``]. Count rows
    keep their values (the draws are still made).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    time_flip: float = Field(default=0.25, ge=0, le=1)
    sign_flip: float = Field(default=0.25, ge=0, le=1)
    scale: tuple[float, float] = (1e-2, 1e4)
    positive: float = Field(default=0.3, ge=0, le=1)
    offset: float = Field(default=3.0, ge=0)

    def flip(
        self, x: np.ndarray, group: np.ndarray | None, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Time-reverse a group-shared share ``time_flip`` of rows; returns (values, flipped [n] bool)."""
        n = len(x)
        flipped = (rng.random(n) < self.time_flip)[_groups(group, n)]
        return np.where(flipped[:, None], x[:, ::-1], x), flipped

    def apply(
        self, x: np.ndarray, is_count: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Sign, shift/offset and scale non-count rows; returns (values, per-row flags)."""
        n = len(x)
        observed = np.isfinite(x)
        sign = np.where(rng.random((n, 1)) < self.sign_flip, -1.0, 1.0)
        scale = _families.logu(rng, (n, 1), *self.scale)
        offset = _families.u(rng, (n, 1), -self.offset, self.offset)
        positive = rng.random((n, 1)) < self.positive
        y = sign * x
        y = np.where(positive, y - _row_min(y, observed), y + offset)
        out = np.where(is_count, x, y * scale)
        flags = {
            "sign_flip": ((sign < 0) & ~is_count)[:, 0],
            "positive": (positive & ~is_count)[:, 0],
            "offset": (~positive & ~is_count)[:, 0],
        }
        return out, flags


class DutyCycle(BaseModel):
    """T4: on/off gating of a share ``rate`` of non-count rows (Simons ``observe`` with ``duty``).

    The row is shifted to a nonnegative minimum (only if it dips below zero)
    and multiplied by an on/off mask: Markov runs with mean on and off
    lengths LogU[``run``] over ``segments`` alternating segments, or with
    probability ``periodic`` a periodic mask (period from a bank of common
    periods, on-share U[0.3, 0.8], switch-off jitter up to ``jitter`` of the
    period).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rate: float = Field(default=0.1, ge=0, le=1)
    periodic: float = Field(default=0.5, ge=0, le=1)
    run: tuple[float, float] = (4.0, 512.0)
    segments: int = Field(default=256, ge=2)
    share: tuple[float, float] = (0.3, 0.8)
    jitter: float = Field(default=0.05, ge=0)

    def mask(
        self, rng: np.random.Generator, n: int, length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """On/off masks [n, length] and whether each row's mask is periodic [n]."""
        mean_on = _families.logu(rng, (n, 1), *self.run)
        mean_off = _families.logu(rng, (n, 1), *self.run)
        k = np.arange(self.segments)[None]
        start_on = rng.random((n, 1)) < mean_on / (mean_on + mean_off)
        on_k = (k % 2 == 0) == start_on
        mean = np.where(on_k, mean_on, mean_off)
        p = np.clip(1 / mean, 1e-6, 1 - 1e-6)
        uu = np.maximum(rng.random((n, self.segments)), 1e-12)
        runs = np.floor(np.log(uu) / np.log1p(-p)) + 1
        edges = np.cumsum(runs, -1)
        t = np.arange(length, dtype=np.float64)
        segment = np.stack([np.searchsorted(e, t, side="right") for e in edges])
        on = np.take_along_axis(on_k, np.minimum(segment, self.segments - 1), -1)
        periodic = np.zeros(n, dtype=bool)
        if self.periodic > 0:
            periodic = rng.random(n) < self.periodic
            on = np.where(periodic[:, None], self._periodic_on(rng, n, length), on)
        return on, periodic

    def _periodic_on(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        period = _families.choice(rng, DUTY_PERIODS, (n, 1))
        share = _families.u(rng, (n, 1), *self.share)
        phase = rng.random((n, 1))
        cycle_pos = np.arange(length, dtype=np.float64)[None] / period + phase
        cycle = np.floor(cycle_pos)
        seed = _families.u(rng, (n, 1), 0.0, 1000.0)
        jitter = self.jitter * np.sin(12.9898 * cycle + seed)
        return (cycle_pos - cycle) < (share + jitter)

    def apply(
        self, x: np.ndarray, is_count: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (values, duty rows [n] bool, periodic duty rows [n] bool)."""
        n, length = x.shape
        chosen = (rng.random(n) < self.rate) & ~is_count[:, 0]
        rows = np.flatnonzero(chosen)
        periodic_rows = np.zeros(n, dtype=bool)
        out = x.copy()
        if len(rows):
            on, periodic = self.mask(rng, len(rows), length)
            sub = x[rows]
            shifted = sub - np.minimum(_row_min(sub, np.isfinite(sub)), 0.0)
            y = shifted * on
            out[rows] = np.where(np.isfinite(y), y, sub)
            periodic_rows[rows] = periodic
        return out, chosen, periodic_rows


class Gaps(BaseModel):
    """T3: short holes in a share ``rate`` of rows (Simons ``post``).

    Hole starts arrive at a per-row rate LogU[``start_rate``]; each hole
    lasts U{1..``max_span``} steps (one span per row); the last
    ``keep_last`` steps stay observed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rate: float = Field(default=0.1, ge=0, le=1)
    start_rate: tuple[float, float] = (0.001, 0.02)
    max_span: int = Field(default=32, ge=1)
    keep_last: int = Field(default=32, ge=0)

    def apply(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (values with NaN holes, rows with holes drawn [n] bool)."""
        n, length = x.shape
        holes = rng.random(n) < self.rate
        rows = np.flatnonzero(holes)
        out = x.copy()
        if len(rows):
            t = np.arange(length)[None]
            rate = _families.logu(rng, (len(rows), 1), *self.start_rate)
            span = rng.integers(1, self.max_span + 1, (len(rows), 1))
            starts = rng.random((len(rows), length)) < rate
            latest = np.maximum.accumulate(np.where(starts, t, -(10**9)), -1)
            hole = t - latest < span
            if self.keep_last:
                hole[:, -self.keep_last :] = False
            out[rows] = np.where(hole, np.nan, x[rows])
        return out, holes


class SimonsParity(BaseModel):
    """The Simons transform layer for one training stage; a component set to None is off.

    Use :meth:`from_stage` for the Simons-25M stage settings. A loader calls
    :meth:`batch_augment` on a batch of draws (native units, NaN missing),
    then its own coupling or cropping, then :meth:`series_transforms`, then
    its own short-history padding.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch: BatchAugment | None = Field(default_factory=BatchAugment)
    noisers: Noisers | None = Field(default_factory=Noisers)
    units: RandomUnits | None = Field(default_factory=RandomUnits)
    duty: DutyCycle | None = Field(default_factory=DutyCycle)
    gaps: Gaps | None = Field(default_factory=Gaps)

    @classmethod
    def from_stage(
        cls, stage: Literal["stage1", "stage2", "stage3", "stage4", "stage4-long"]
    ) -> "SimonsParity":
        """Simons-25M settings: T1-T4 in stages 1 and 2 (duty 10%, half periodic); T1-T3 in stages 3, 4 and 4-long."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        if stage in ("stage1", "stage2"):
            return cls()
        return cls(duty=None)

    def rates(self) -> dict:
        """JSON-able settings of every component (None when off), with the version tag."""
        return {
            "version": PARITY_VERSION,
            **{
                name: None if spec is None else spec.model_dump(mode="json")
                for name, spec in (
                    ("batch", self.batch),
                    ("noisers", self.noisers),
                    ("units", self.units),
                    ("duty", self.duty),
                    ("gaps", self.gaps),
                )
            },
        }

    def batch_augment(
        self, values: np.ndarray, group: np.ndarray | None, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, int]]:
        """T1 on a batch [n, L]: rows standardised on finite values, then augmented; returns (values, counts)."""
        x = _rows(values)
        n = len(x)
        if self.batch is None:
            return standardise(x), {"series": n, "t1_active": 0}
        out, chosen = self.batch.apply(x, group, rng)
        counts = {"series": n, "t1_active": int(chosen.any(-1).sum())}
        for col, name in enumerate(CATEGORIES):
            counts[f"t1_{name}"] = int(chosen[:, col].sum())
        counts["t1_two"] = int((chosen.sum(-1) == 2).sum())
        for i, first in enumerate(CATEGORIES):
            for j in range(i + 1, len(CATEGORIES)):
                pair = chosen[:, i] & chosen[:, j]
                counts[f"t1_{first}+{CATEGORIES[j]}"] = int(pair.sum())
        return out, counts

    def series_transforms(
        self, values: np.ndarray, group: np.ndarray | None, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Simons ``post`` without padding: time flip, noisers, units, duty, gaps; returns (values, counts)."""
        x = standardise(values)
        n = len(x)
        counts = {"series": n}
        if self.units is not None:
            x, flipped = self.units.flip(x, group, rng)
            counts["t3_time_flip"] = int(flipped.sum())
        is_count = np.zeros((n, 1), dtype=bool)
        if self.noisers is not None:
            x, is_count, kind = self.noisers.apply(x, rng)
            counts["t2_active"] = int((kind >= 0).sum())
            for index, name in enumerate(NOISE_KINDS):
                counts[f"t2_{name}"] = int((kind == index).sum())
        if self.units is not None:
            x, flags = self.units.apply(x, is_count, rng)
            for name, flag in flags.items():
                counts[f"t3_{name}"] = int(flag.sum())
        observed = np.isfinite(x) | np.isinf(x)
        x = np.where(observed & ~np.isfinite(x), 0.0, x)
        if self.duty is not None:
            x, duty_rows, periodic_rows = self.duty.apply(x, is_count, rng)
            counts["t4_duty"] = int(duty_rows.sum())
            counts["t4_duty_periodic"] = int(periodic_rows.sum())
        if self.gaps is not None:
            x, holed = self.gaps.apply(x, rng)
            counts["t3_gaps"] = int(holed.sum())
        return x, counts
