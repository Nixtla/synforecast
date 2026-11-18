"""Simons-style coupling mechanisms for groups of synthetic channels.

Array-level port of the coupling stage of the Simons TSFM recipe
(``simons/synth/multi.py``: ``plan`` and ``couple``), for loaders that build
coupled groups from SynForecast base draws as numpy rows. Each group of
``size`` channels draws one mechanism, uniformly over ``mechanisms``
(a group of one channel is always ``identity``):

- identity: the channels stay independent;
- mix: every channel is a signed combination of one or two drivers;
- lag: the first channels observe a driver, the others follow a driver or an
  earlier follower at a lag LogU[4, ``lag_max``], optionally smoothed by an
  exponential filter;
- coint: channels are combinations of one or two random-walk drivers (with or
  without drift) plus their own Lorentzian (OU-like) stationary noise;
- func: channels are a nonlinear map (tanh, softplus, abs, clip, rounding,
  hinge) of a lagged, optionally smoothed driver;
- event: channels respond with an exponential decay, at a lag, to an
  impulse train or a jittered periodic pulse train.

Drivers are two rows per group: a base draw (``driver_bases``, supplied by
the caller from the same source as the channels), a random walk, a drifting
random walk, an impulse train or a pulse train. Each channel is
``sqrt(rho) * signal + sqrt(1 - rho) * own``, where ``signal`` is the
standardised response to its parents and ``own`` is its base draw (or OU
noise for coint). Responses are linear convolutions on a grid twice the
padded length, so lags never wrap: the caller draws bases at
:meth:`SimonsCoupling.padded_length` and gets the last ``length`` steps back.
A second pass lets chain followers read their parents' first-pass outputs.

Distributions follow ``multi.py``; random streams differ (numpy instead of
torch), so outputs match in law, not bitwise. ``plan`` consumes the numpy
stream exactly as Simons' per-group loop does. Simons draws every channel's
base and the base drivers from its family mixture and reuses base-driver and
OU-noise rows round-robin across a pool; here bases and drivers come from
the caller and every OU row is drawn per channel.
"""

import dataclasses
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

COUPLING_VERSION = "simons-coupling-v1"
"""Version tag of the coupling semantics below."""

MECHANISMS = ("identity", "mix", "lag", "coint", "func", "event")
DRIVER_KINDS = ("base", "walk", "drift", "impulses", "pulses")
FUNC_KINDS = ("none", "tanh", "softplus", "abs", "clip", "round", "hinge")
PULSE_PERIODS = (7, 12, 24, 48, 52, 96, 168, 288, 336)
DRIVERS = 2
EDGES = 3


def fft_length(n: int) -> int:
    """Smallest even 7-smooth integer >= n (Simons' FFT grid length)."""
    m = int(n) + (int(n) % 2)
    while True:
        r = m
        for p in (2, 3, 5, 7):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 2


def _logu(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _standardise(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(-1, keepdims=True)
    std = (
        x.std(-1, ddof=1, keepdims=True)
        if x.shape[-1] > 1
        else np.ones_like(x[..., :1])
    )
    return x / np.maximum(std, 1e-6)


@dataclasses.dataclass(frozen=True)
class CouplingPlan:
    """Structure of one coupled group (Simons ``plan`` for one group).

    Edge parents index ``[driver 0, driver 1, channel 0, ..., channel size-1]``
    (-1 = no parent); a channel parent reads that channel's first-pass output.
    """

    mechanism: int
    driver_kind: np.ndarray
    driver_param: np.ndarray
    own_kind: np.ndarray
    ou: np.ndarray
    rho: np.ndarray
    func_kind: np.ndarray
    func_param: np.ndarray
    parent: np.ndarray
    tau: np.ndarray
    decay: np.ndarray
    coef: np.ndarray
    root: np.ndarray

    @property
    def size(self) -> int:
        return len(self.rho)

    @property
    def mechanism_name(self) -> str:
        return MECHANISMS[self.mechanism]

    @property
    def base_drivers(self) -> int:
        """Number of drivers the caller supplies as base draws (``driver_bases`` rows)."""
        return int((self.driver_kind == 0).sum())

    def counts(self) -> dict[str, int]:
        """Mechanism, driver kinds, OU and nonlinear channels, roots and edges of this group."""
        out = {f"mechanism_{self.mechanism_name}": 1, "channels": self.size}
        for kind in self.driver_kind[self.driver_kind >= 0]:
            key = f"driver_{DRIVER_KINDS[int(kind)]}"
            out[key] = out.get(key, 0) + 1
        out["own_ou"] = int((self.own_kind == 1).sum())
        out["func_channels"] = int((self.func_kind > 0).sum())
        out["roots"] = int(self.root.sum())
        out["edges"] = int((self.parent >= 0).sum())
        out["chain_edges"] = int((self.parent >= DRIVERS).sum())
        return out


class SimonsCoupling(BaseModel):
    """Simons ``multi.py`` coupling mechanisms for one group of channels at a time.

    A loader calls :meth:`plan` for a group, draws ``size`` channel bases and
    ``plan.base_drivers`` driver bases at :meth:`padded_length` (standardised,
    finite), then :meth:`couple`, which returns the standardised last
    ``length`` steps of every channel.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanisms: tuple[str, ...] = MECHANISMS
    lag_max: int = Field(default=256, ge=4)

    @model_validator(mode="after")
    def validate_mechanisms(self) -> "SimonsCoupling":
        if not self.mechanisms or not set(self.mechanisms) <= set(MECHANISMS):
            raise ValueError(f"mechanisms must list some of {MECHANISMS}")
        if len(set(self.mechanisms)) != len(self.mechanisms):
            raise ValueError("mechanisms must not repeat")
        return self

    def settings(self) -> dict:
        """JSON-able settings with the version tag."""
        return {
            "version": COUPLING_VERSION,
            "mechanisms": list(self.mechanisms),
            "lag_max": self.lag_max,
            "pulse_periods": list(PULSE_PERIODS),
        }

    def padded_length(self, length: int) -> int:
        """Base length for an output of ``length`` steps: the FFT grid length of ``length + lag_max``."""
        if length < 2:
            raise ValueError("length must be at least 2")
        return fft_length(int(length) + self.lag_max)

    def plan(self, size: int, rng: np.random.Generator) -> CouplingPlan:
        """Sample the structure of one group of ``size`` channels (Simons ``plan``, one group, same stream use)."""
        q = int(size)
        if q < 1:
            raise ValueError("size must be positive")
        lag_max = self.lag_max
        ids = [MECHANISMS.index(m) for m in self.mechanisms]
        cum = np.cumsum(np.full(len(ids), 1.0 / len(ids)))
        m = (
            ids[
                min(int(np.searchsorted(cum, rng.random(), side="right")), len(cum) - 1)
            ]
            if q > 1
            else 0
        )
        driver_kind = np.full(DRIVERS, -1, np.int64)
        driver_param = np.zeros((DRIVERS, 3))
        own_kind = np.zeros(q, np.int64)
        ou = np.ones(q)
        rho = np.zeros(q)
        func_kind = np.zeros(q, np.int64)
        func_param = np.zeros((q, 4))
        parent = np.full((q, EDGES), -1, np.int64)
        tau = np.zeros((q, EDGES))
        decay = np.zeros((q, EDGES))
        coef = np.zeros((q, EDGES))
        root = np.zeros(q, bool)
        k_drivers = 1 if m in (4, 5) else int(rng.integers(1, 3))
        for k in range(DRIVERS):
            if m == 0 or k >= k_drivers:
                kind = -1
            elif m == 3:
                kind = 1 if rng.random() < 0.6 else 2
            elif m == 5:
                kind = 3 if rng.random() < 0.6 else 4
            else:
                kind = 0
            if kind >= 0:
                driver_kind[k] = kind
                driver_param[k] = (
                    _logu(rng, 0.003, 0.05),
                    float(PULSE_PERIODS[int(rng.integers(len(PULSE_PERIODS)))]),
                    float(rng.integers(1, 5)),
                )
        plain = []
        observed_root = m == 2 or m == 5 or (m == 4 and rng.random() < 0.5)
        for j in range(q):
            if m == 0:
                continue
            if observed_root and j < (k_drivers if m == 2 else 1):
                parent[j, 0], coef[j, 0] = j, 1.0
                rho[j] = rng.uniform(0.85, 1.0)
                root[j] = True
                plain.append(j)
                continue
            if m == 1:
                for k in range(k_drivers):
                    parent[j, k] = k
                    coef[j, k] = rng.normal() * (
                        1.0 if k == 0 else rng.uniform(0.2, 1.0)
                    )
                rho[j] = rng.uniform(0.3, 0.95)
            elif m == 2:
                if plain and rng.random() < 0.4:
                    parent[j, 0] = DRIVERS + plain[int(rng.integers(len(plain)))]
                else:
                    parent[j, 0] = int(rng.integers(k_drivers))
                    plain.append(j)
                tau[j, 0] = round(_logu(rng, 4, lag_max))
                decay[j, 0] = 0.0 if rng.random() < 0.6 else _logu(rng, 1, 48)
                coef[j, 0] = (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(
                    0.5, 1.5
                )
                if rng.random() < 0.3:
                    parent[j, 1] = int(rng.integers(k_drivers))
                    tau[j, 1] = round(_logu(rng, 4, lag_max))
                    decay[j, 1] = 0.0 if rng.random() < 0.6 else _logu(rng, 1, 48)
                    coef[j, 1] = (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(
                        0.3, 1.0
                    )
                rho[j] = rng.uniform(0.4, 0.95)
            elif m == 3:
                for k in range(k_drivers):
                    parent[j, k] = k
                    tau[j, k] = 0.0 if rng.random() < 0.7 else round(_logu(rng, 1, 64))
                    coef[j, k] = (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(
                        0.5, 2.0
                    )
                own_kind[j], ou[j] = 1, _logu(rng, 2, 100)
                rho[j] = rng.uniform(0.6, 0.97)
            elif m == 4:
                parent[j, 0], coef[j, 0] = 0, 1.0
                tau[j, 0] = 0.0 if rng.random() < 0.4 else round(_logu(rng, 4, lag_max))
                decay[j, 0] = 0.0 if rng.random() < 0.7 else _logu(rng, 1, 32)
                func_kind[j] = int(rng.integers(1, 7))
                func_param[j] = (
                    _logu(rng, 0.5, 3.0),
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                    rng.uniform(-2, 2),
                )
                rho[j] = rng.uniform(0.5, 0.95)
            else:
                for e in range(1 + (rng.random() < 0.3)):
                    parent[j, e] = 0
                    tau[j, e] = round(_logu(rng, 1, 128)) if rng.random() < 0.8 else 0.0
                    decay[j, e] = _logu(rng, 1, 100)
                    coef[j, e] = (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(
                        0.5, 2.0
                    )
                rho[j] = rng.uniform(0.3, 0.9)
        return CouplingPlan(
            mechanism=m,
            driver_kind=driver_kind,
            driver_param=driver_param,
            own_kind=own_kind,
            ou=ou,
            rho=rho,
            func_kind=func_kind,
            func_param=func_param,
            parent=parent,
            tau=tau,
            decay=decay,
            coef=coef,
            root=root,
        )

    def couple(
        self,
        plan: CouplingPlan,
        own: np.ndarray,
        driver_bases: np.ndarray,
        length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Couple one group: ``own`` [size, P] and ``driver_bases`` [base_drivers, P] (P >= length + lag_max, finite) -> [size, length], standardised."""
        own = np.asarray(own, dtype=np.float64)
        q = plan.size
        if own.ndim != 2 or own.shape[0] != q:
            raise ValueError(f"own must be [{q}, P], got {own.shape}")
        full = own.shape[1]
        if length < 2 or full < int(length) + self.lag_max:
            raise ValueError(
                f"bases need at least length + lag_max = {int(length) + self.lag_max} steps, got {full}"
            )
        bases = (
            np.asarray(driver_bases, dtype=np.float64).reshape(-1, full)
            if plan.base_drivers
            else np.zeros((0, full))
        )
        if bases.shape[0] != plan.base_drivers:
            raise ValueError(
                f"the plan needs {plan.base_drivers} driver bases, got {bases.shape[0]}"
            )
        if not (np.isfinite(own).all() and np.isfinite(bases).all()):
            raise ValueError("bases must be finite")
        own = _standardise(own)
        if plan.mechanism == 0:
            return _standardise(own[:, -int(length) :])
        drivers = np.zeros((DRIVERS, full))
        b = 0
        t = np.arange(full, dtype=np.float64)
        for k in range(DRIVERS):
            kind = int(plan.driver_kind[k])
            if kind < 0:
                continue
            rate, period, width = plan.driver_param[k]
            if kind == 0:
                row = bases[b]
                b += 1
            else:
                steps = rng.standard_normal(full)
                if kind == 1:
                    row = _standardise(np.cumsum(steps))
                elif kind == 2:
                    row = _standardise(
                        np.cumsum(steps + 4 * (rate - 0.025) / 0.025 * 0.05)
                    )
                elif kind == 3:
                    row = _standardise((rng.random(full) < rate) * np.exp(0.5 * steps))
                else:
                    phase = rng.random() * period
                    row = _standardise(
                        (np.remainder(t + phase, period) < width) * np.exp(0.3 * steps)
                    )
            drivers[k] = _standardise(row)
        grid = 2 * full
        omega = 2 * math.pi * np.arange(grid // 2 + 1) / grid
        ou_rows = np.flatnonzero(plan.own_kind == 1)
        if len(ou_rows):
            spectrum = 1.0 / (1.0 + (omega[None] * plan.ou[ou_rows, None]) ** 2)
            amp = np.sqrt(np.maximum(spectrum, 0))
            amp[:, 0] = 0
            xi = rng.standard_normal(amp.shape) + 1j * rng.standard_normal(amp.shape)
            own[ou_rows] = _standardise(np.fft.irfft(amp * xi, n=grid)[:, :full])
        used = np.flatnonzero((plan.parent >= 0).any(0))
        parent = np.maximum(plan.parent[:, used], 0)
        tau, decay = plan.tau[:, used], plan.decay[:, used]
        weight = np.where(plan.parent[:, used] >= 0, plan.coef[:, used], 0.0)
        r = np.where(decay > 0, np.exp(-1.0 / np.maximum(decay, 1e-3)), 0.0)
        shift = np.exp(-1j * omega)
        response = (
            (weight * (1 - r))[..., None]
            * np.exp(-1j * tau[..., None] * omega)
            / (1 - r[..., None] * shift)
        )
        spec_d = np.fft.rfft(drivers, n=grid)
        rho = plan.rho[:, None]
        mix_own = np.sqrt(1 - rho) * own

        def combine(slots):
            table = np.concatenate([spec_d, np.fft.rfft(slots, n=grid)])
            mixed = (table[parent] * response).sum(1)
            signal = np.fft.irfft(mixed, n=grid)[:, :full]
            signal = _standardise(_transform(signal, plan.func_kind, plan.func_param))
            return np.sqrt(rho) * signal + mix_own

        x = combine(own)
        # The second pass only changes channels whose parent is another channel (chain followers).
        if (plan.parent >= DRIVERS).any():
            x = combine(x)
        return _standardise(x[:, -int(length) :])


def _transform(y: np.ndarray, kind: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-row nonlinear maps (Simons ``_transform``): p [n, 4] = gain, b, s0, s1; kind 0 is identity."""
    out = y.copy()
    for k in np.unique(kind[kind > 0]):
        rows = np.flatnonzero(kind == k)
        v = y[rows]
        g, b, s0, s1 = (p[rows, i : i + 1] for i in range(4))
        if k == 1:
            out[rows] = np.tanh(g * v)
        elif k == 2:
            out[rows] = np.logaddexp(0.0, g * v)
        elif k == 3:
            out[rows] = np.abs(v)
        elif k == 4:
            clip = 0.2 + np.abs(b) * 1.3
            out[rows] = np.minimum(np.maximum(v, -clip), clip)
        elif k == 5:
            step = 1 + 3 * np.abs(b)
            out[rows] = np.round(v * step) / step
        elif k == 6:
            out[rows] = s0 * v + s1 * np.maximum(v - b, 0.0)
    return out
