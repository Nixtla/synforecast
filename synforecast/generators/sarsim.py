"""SarSim SARIMA / SARIMA-2 generator with level-scaled noisers."""

import numpy as np
from pydantic import Field

from synforecast import _tempo_sarsim
from synforecast.generators._block import BlockGenerator


class SarSimGenerator(BlockGenerator):
    """Generate stable SARIMA and SARIMA-2 paths with level-scaled observation noise (SarSim prior).

    Follows SarSim (arXiv:2601.00970, Sec. 3, Table 9). A SARIMA path
    draws its season s ~ U{0, ..., ``max_season``} (s < 2: non-seasonal),
    orders p ~ U{0..10}, P ~ U{0..2}, q ~ U{0..3}, Q ~ U{0..2}, and a fair
    coin sets p = 0 or P = 0, so at most one AR side is active; lags combine
    additively (SarSim Eq. 6: theta(B) + Theta(B^s) - 1 on the MA side).
    Roots are drawn
    inside the unit circle (radius U[0, ``ar_radius``], or
    U[0, ``seasonal_ar_radius``] for seasonal factors; uniform angle; in
    conjugate pairs plus one real root for odd orders) and expanded into
    polynomials, so every AR part is stable and every MA part invertible.
    The stationary ARMA path is synthesised in the frequency domain with a
    discarded warm-up of the largest active MA lag plus the time for the
    slowest sampled pole to decay by e^-40 (decay part capped at 65,536
    steps; at least 1,024 steps in all), then integrated: fractionally with d ~ U[0, 1]
    (Hosking binomial weights) and seasonally (D = 1) when s >= 2.
    On ``sarima2_rate`` of series the path is SARIMA-2: a base SARIMA at
    the first season of a pair from {(24, 7), (7, 52), (0, 7), (0, 3),
    (0, 24), (0, 52)} and an envelope SARIMA at the second, drawn at a
    rate slower by the base season (or an integer LogU[4, 64] factor for a
    non-seasonal base) and linearly upsampled; they combine additively
    (base + envelope) or, on ``multiplicative_rate``, as
    (1 + w e) x base with e the envelope min-max scaled to [-1, 1] and
    w ~ U[0, 1]. On ``noiser_rate`` of series a noiser chosen uniformly
    from Poisson, generalised gamma and lognormal replaces the values by
    observation draws at rate lambda_t = lambda0 x minmax(y_t): Poisson
    (lambda0 LogU[0.1, 100]); Gamma(shape kappa, scale lambda_t)^zeta, so
    the coefficient of variation before the power is 1 / sqrt(kappa)
    (lambda0 LogU[0.1, 100], kappa LogU[1, 50], zeta U[0.5, 1.5]);
    LogNormal(log-mean lambda_t, log-sd kappa) (lambda0 LogU[0.1, 5], kappa
    LogU[1, 3]). A noised row that comes out constant keeps its pre-noise
    values.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index.
        sarima2_rate (float): Share of SARIMA-2 series (default: 0.5).
        multiplicative_rate (float): Share of SARIMA-2 series combined multiplicatively (default: 0.5).
        noiser_rate (float): Share of series passed through a noiser (default: 0.75).
        max_season (int): Largest SARIMA season (default: 52).
        ar_radius (float): Largest root radius of non-seasonal polynomials (default: 0.9).
        seasonal_ar_radius (float): Largest root radius of seasonal polynomials (default: 0.1).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    sarima2_rate: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Share of SARIMA-2 series"
    )
    multiplicative_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Share of multiplicative SARIMA-2 series",
    )
    noiser_rate: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Share of noised series"
    )
    max_season: int = Field(
        default=52, ge=2, le=1024, description="Largest SARIMA season"
    )
    ar_radius: float = Field(
        default=0.9, gt=0.0, lt=1.0, description="Non-seasonal root radius bound"
    )
    seasonal_ar_radius: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Seasonal root radius bound"
    )

    def _kernel(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        return _tempo_sarsim.sarsim(
            rng,
            n,
            length,
            self.sarima2_rate,
            self.multiplicative_rate,
            self.noiser_rate,
            self.max_season,
            self.ar_radius,
            self.seasonal_ar_radius,
        )
