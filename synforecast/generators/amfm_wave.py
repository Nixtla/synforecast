"""AM/FM waveform-mixture generator."""

import numpy as np
from pydantic import Field

from synforecast import _families
from synforecast.generators._block import BlockGenerator


class AMFMWaveGenerator(BlockGenerator):
    """Generate sums of amplitude- and frequency-modulated waveforms.

    Each of 1 to ``components`` components is a sine, soft square,
    triangle or sawtooth wave whose period (60% from a bank of common
    periods, else log-uniform in [2, 700], at least 3) drifts by up to 30%
    and whose amplitude drifts by up to 50% along slow Gaussian-process
    paths. Half the series add a linear trend; all add light noise. Port of
    the Simons TSFM ``wave`` family.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index.
        components (int): Most waveform components per series (default: 3).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    components: int = Field(
        default=3, ge=1, description="Most waveform components per series"
    )

    def _kernel(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        return _families.amfm_wave(rng, n, length, self.components)
