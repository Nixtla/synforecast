"""Regime-switching Ornstein-Uhlenbeck generator."""

import numpy as np

from synforecast import _families
from synforecast.generators._block import BlockGenerator


class RegimeOUGenerator(BlockGenerator):
    """Generate regime-switching OU paths with drifting mean, reversion speed and volatility.

    Two regimes (calm: fast reversion, low volatility around one level;
    turbulent: slow reversion, high volatility around the opposite level)
    alternate at a log-uniform hazard in [1/8192, 1/256]. The mean adds a
    trend on 70% and a seasonal Fourier term on 60% of series; the
    reversion speed drifts on 20%; the volatility drifts (30%), cycles
    (30%) and follows a slow GP (50%). Innovations are fractional Gaussian
    noise (Hurst 0.3 to 0.8) on 20% of series. The exact discretisation
    uses a log-uniform model step in [0.005, 0.05]. Port of the Simons TSFM
    ``sde`` family.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index.
        seed (int | None): Random seed for reproducibility (default: None).
    """

    def _kernel(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        return _families.regime_ou(rng, n, length)
