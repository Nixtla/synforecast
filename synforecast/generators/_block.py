"""Shared base of generators whose kernels draw a whole block of series at once."""

from abc import abstractmethod

import numpy as np

from synforecast.base import BaseGenerator


class BlockGenerator(BaseGenerator):
    """A generator backed by a vectorised kernel ``(rng, n, length) -> [n, length]``.

    :meth:`generate_block` draws ``n_series`` series of one length from the
    generator's stream in a single kernel call, without overlays; the
    per-series API (``generate``, ``generate_values``, ``Multivariatizer``)
    draws one series at a time through :meth:`generate_single_series`.
    """

    @abstractmethod
    def _kernel(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        """Draw float64 [n, length] values from ``rng``."""

    def generate_block(self, n_series: int, length: int) -> np.ndarray:
        """Draw ``n_series`` series of ``length`` steps as a float64 [n_series, length] array (no overlays).

        Args:
            n_series (int): Number of series.
            length (int): Steps per series.

        Returns:
            Array of shape (n_series, length).
        """
        if n_series < 1 or length < 2:
            raise ValueError("generate_block needs n_series >= 1 and length >= 2")
        values = np.asarray(self._kernel(self.rng, n_series, length), dtype=np.float64)
        return np.where(np.isfinite(values), values, 0.0)

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate values for a single time series.

        Args:
            length (int): The length of the series to generate.

        Returns:
            Array of time series values.
        """
        return self.generate_block(1, length)[0]
