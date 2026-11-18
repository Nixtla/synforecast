"""TempoPFN-style regime-switching SDE generator."""

import numpy as np
from pydantic import Field, model_validator

from synforecast import _tempo_sarsim
from synforecast.generators._block import BlockGenerator


class TempoSDEGenerator(BlockGenerator):
    """Generate regime-switching, time-inhomogeneous OU paths with an optional fBm driver (TempoPFN SDE prior).

    Each series follows dy = theta(t, r)(mu(t, r) - y) dt + sigma(t, r) dW
    by Euler-Maruyama at step ``dt`` (TempoPFN, arXiv:2510.25502, Sec. 3.2,
    App. C.1, Table 6). The regime r is a two-state Markov chain with
    per-step stay probabilities p00, p11 ~ U[``stay_min``, ``stay_max``]
    (mean regime durations of about 7 to 1,000 steps) and a uniform initial
    regime. Regime 0 is calm (theta U[1, 5], mu N(-2, 1), sigma
    LogN(log 0.3, 0.3)); regime 1 is turbulent (theta U[0.05, 0.5],
    mu N(2, 1), sigma LogN(log 1.5, 0.5)). Each regime has its own trend on
    mu (p = 0.7), theta (0.2) and sigma (0.3), drawn from TempoPFN's trend
    bank (none, linear, exponential, logistic, sinusoidal, piecewise linear,
    polynomial) plus up to two smooth structural changes, and its own one or
    two seasonal sinusoids on mu (0.6) and sigma (0.3) at periods
    {7, 30, 90, 182.6, 365.25} time units (jitter 5%, amplitude U[0.5, 3],
    drift U[-0.001, 0.001]). theta and sigma are scaled by (1 + trend [+
    seasonal]); theta is clipped to [1e-3, 0.5 / dt] and sigma taken in
    absolute value. On ``stochastic_vol_rate`` of series sigma is further
    scaled by exp(h / 2), h a regime-dependent log-variance OU (Table 6
    volatility process). On ``fbm_rate`` of series the Brownian increments
    are replaced by exact fractional Gaussian noise (Davies-Harte) with
    Hurst H ~ U[``hurst_min``, ``hurst_max``], scaled by dt^H. The path is
    then shifted by U[-100, 100], scaled by U[0.1, 50] and given Gaussian
    measurement noise of standard deviation U[0, 0.1].

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index.
        dt (float): Euler-Maruyama step in model time units (default: 0.01).
        fbm_rate (float): Share of series driven by fractional Brownian motion (default: 0.25).
        hurst_min (float): Lower bound of the Hurst exponent (default: 0.3).
        hurst_max (float): Upper bound of the Hurst exponent (default: 0.8).
        stay_min (float): Lower bound of the per-step regime stay probability (default: 0.85).
        stay_max (float): Upper bound of the per-step regime stay probability (default: 0.999).
        stochastic_vol_rate (float): Share of series with a stochastic volatility factor (default: 0.0).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    dt: float = Field(default=0.01, gt=0.0, le=0.1, description="Euler-Maruyama step")
    fbm_rate: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Share of fBm-driven series"
    )
    hurst_min: float = Field(
        default=0.3, gt=0.0, lt=1.0, description="Lower Hurst bound"
    )
    hurst_max: float = Field(
        default=0.8, gt=0.0, lt=1.0, description="Upper Hurst bound"
    )
    stay_min: float = Field(
        default=0.85, ge=0.0, lt=1.0, description="Lower regime stay probability"
    )
    stay_max: float = Field(
        default=0.999, ge=0.0, lt=1.0, description="Upper regime stay probability"
    )
    stochastic_vol_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of series with stochastic volatility",
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> "TempoSDEGenerator":
        if self.hurst_min > self.hurst_max:
            raise ValueError("hurst_min must not exceed hurst_max")
        if self.stay_min > self.stay_max:
            raise ValueError("stay_min must not exceed stay_max")
        return self

    def _kernel(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        return _tempo_sarsim.tempo_sde(
            rng,
            n,
            length,
            self.dt,
            self.fbm_rate,
            self.hurst_min,
            self.hurst_max,
            self.stay_min,
            self.stay_max,
            self.stochastic_vol_rate,
        )
