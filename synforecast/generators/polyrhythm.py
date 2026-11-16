"""Polyrhythm (layered beat train) generator."""

import numpy as np
from pydantic import Field

from synforecast import _families
from synforecast.generators._block import BlockGenerator


class PolyrhythmGenerator(BlockGenerator):
    """Generate layered beat trains with decaying envelopes (polyrhythm prior).

    A base period (60% from a bank of common periods, else log-uniform in
    [4, 400]) is split into 3 to ``layers`` layers, each beating at the
    period divided by 1, 2, 3, 4, 6 or 8 with a random offset and firing
    probability. Every beat adds a log-uniform amplitude (with lognormal
    jitter on half the layers) to an exponentially decaying envelope. 30%
    of series are modulated by a slow sinusoidal carrier, 40% get a trend
    or slow drift and 60% light noise. Without drift or noise, values before
    the first beat are zero. Port of the Simons TSFM ``rhythm`` family.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index.
        layers (int): Most beat layers per series, at least 3 (default: 5).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    layers: int = Field(default=5, ge=3, description="Most beat layers per series")

    def _kernel(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        return _families.polyrhythm(rng, n, length, self.layers)
