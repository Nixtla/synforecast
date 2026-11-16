"""ForecastPFN-style multiplicative prior generator."""

import numpy as np
from pydantic import Field

from synforecast import _families
from synforecast.generators._block import BlockGenerator


class ForecastPFNGenerator(BlockGenerator):
    """Generate positive trend x seasonality x noise series (ForecastPFN prior).

    The trend is linear times exponential (flat on 20% of series). Up to
    two seasonal factors follow a granularity drawn from hourly (24, 168),
    daily (7, 365.25), weekly (52.18), monthly (12), hourly-only (24) or
    daily-only (7), each a ``harmonics``-term Fourier series of amplitude
    up to 0.4 floored at 0.05. Noise is ``1 + level * (Weibull - mean)``,
    floored at 0.01. Port of the Simons TSFM ``fpfn`` family.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index.
        harmonics (int): Fourier terms per seasonal factor (default: 3).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    harmonics: int = Field(
        default=3, ge=1, description="Fourier terms per seasonal factor"
    )

    def _kernel(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        return _families.forecast_pfn(rng, n, length, self.harmonics)
