"""Mechanistic simulators: epidemics, fluid queues, rainfall-runoff and thermal systems."""

import numpy as np
from pydantic import Field, model_validator

from synforecast import _families
from synforecast.generators._block import BlockGenerator

_SIMULATION_ARGS = """
        simulation_length (int): Steps simulated per draw after burn-in; the
            series is a random crop of one channel, strided 1 to 4 times
            (pooled or decimated) when it fits, and mirrored when longer than
            the simulation, so ``max_length`` may be up to twice this
            (default: 9216).
        burn (int): Burn-in steps discarded before recording (default: 512).
        max_attempts (int): Simulations tried per draw until one is finite,
            bounded and still moving in its second half (default: 4).
            The path is divided by each channel's RMS and rounded to float16
            before cropping, as Simons stores its trajectory bank."""


class _Simulated(BlockGenerator):
    simulation_length: int = Field(
        default=9216, ge=16, description="Steps simulated per draw"
    )
    burn: int = Field(default=512, ge=0, description="Burn-in steps")
    max_attempts: int = Field(default=4, ge=1, description="Simulations tried per draw")

    @model_validator(mode="after")
    def validate_simulation_length(self) -> "_Simulated":
        if self.max_length > 2 * self.simulation_length:
            raise ValueError("max_length must be at most twice simulation_length")
        return self

    def _kernel(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        return _families.mechanistic(
            rng,
            self._simulator,
            n,
            length,
            self.simulation_length,
            self.burn,
            self.max_attempts,
        )


class EpidemicGenerator(_Simulated):
    __doc__ = (
        """Generate observed epidemic series from a stochastic compartmental model.

    A daily (65%) or weekly (35%) SEIRS model (SIRS on 30% of series) with
    population log-uniform in [1e4, 1e7], R0 in [1.1, 4], waning immunity
    (70%), births, up to three seasonal harmonics of transmission, up to
    four interventions scaling transmission, overdispersed transmission
    noise and Poisson importations. Transitions are normal approximations of
    binomial flows. Reported cases are Poisson draws of incidence times a
    sigmoid-varying reporting rate, a geometric reporting delay (60%) and
    weekday effects (daily data); the second channel is a noisy occupancy
    proxy of prevalence. One channel is cropped and observed with light
    additive or multiplicative noise and an occasional trend. Port of the
    Simons TSFM ``epi`` family, simulated per draw.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index."""
        + _SIMULATION_ARGS
        + """
        seed (int | None): Random seed for reproducibility (default: None).
    """
    )

    _simulator = "epidemic"


class RainfallRunoffGenerator(_Simulated):
    __doc__ = (
        """Generate streamflow or catchment-storage series from a bucket rainfall-runoff model.

    Hourly (40%) or daily rain follows a seasonal two-state Markov chain
    with gamma amounts; a snow store (35% of catchments) accumulates below
    freezing and melts with degree days; soil storage loses seasonal
    evaporation, spills over capacity and drains nonlinearly into a linear
    routing store and a slow groundwater store. Channels: streamflow (plus a
    small base level) and soil plus snow storage. One channel is cropped and
    observed with light additive or multiplicative noise and an occasional
    trend. Port of the Simons TSFM ``hydro`` family, simulated per draw.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index."""
        + _SIMULATION_ARGS
        + """
        seed (int | None): Random seed for reproducibility (default: None).
    """
    )

    _simulator = "rainfall_runoff"


class ThermalGenerator(_Simulated):
    __doc__ = (
        """Generate temperature or energy series from a two-node thermal network.

    Ambient temperature has yearly and daily cycles plus AR(1) weather at
    hourly (65%) or 15-minute resolution; internal heat follows a daily and
    weekly load profile (linear or quadratic in load); 80% of systems are
    cooled by a hysteresis thermostat or (30%) proportional control.
    Channels: node temperature and energy use (cooling plus heat). One
    channel is cropped and observed with light additive or multiplicative
    noise and an occasional trend. Port of the Simons TSFM ``thermal``
    family, simulated per draw.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index."""
        + _SIMULATION_ARGS
        + """
        seed (int | None): Random seed for reproducibility (default: None).
    """
    )

    _simulator = "thermal"


class FluidQueueGenerator(BlockGenerator):
    """Generate fluid-queue observables under daily and weekly load.

    Poisson arrivals follow a daily profile (288, 96 or 24 steps per day)
    with a weekend dip, AR(1) load noise and, on 60% of series, Markov
    surge regimes; capacity drops in occasional outages (30%). The series
    is one of utilisation, backlog, latency, throughput, or a
    traffic-speed curve of occupancy, observed with light additive or
    multiplicative noise and an occasional trend. Port of the Simons TSFM
    ``queue`` family.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index.
        seed (int | None): Random seed for reproducibility (default: None).
    """

    def _kernel(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        return _families.observe(
            rng, _families.finite(_families.fluid_queue(rng, n, length))
        )
