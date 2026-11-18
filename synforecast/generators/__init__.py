"""Time series generators."""

from synforecast.generators.amfm_wave import AMFMWaveGenerator
from synforecast.generators.bounded_process import BoundedProcessGenerator
from synforecast.generators.cauker import CauKerGenerator
from synforecast.generators.chaotic_system import ChaoticSystemGenerator
from synforecast.generators.clickstream import ClickstreamGenerator
from synforecast.generators.copula import CopulaGenerator
from synforecast.generators.cyclic import CyclicGenerator
from synforecast.generators.daily_active_users import DailyActiveUsersGenerator
from synforecast.generators.energy_load import EnergyLoadGenerator
from synforecast.generators.ets import ETSGenerator
from synforecast.generators.forecast_pfn import ForecastPFNGenerator
from synforecast.generators.fractional_brownian_motion import (
    FractionalBrownianMotionGenerator,
)
from synforecast.generators.garch import GARCHGenerator
from synforecast.generators.gaussian_process import GaussianProcessGenerator
from synforecast.generators.geometric_brownian_motion import (
    GeometricBrownianMotionGenerator,
)
from synforecast.generators.hawkes_process import HawkesProcessGenerator
from synforecast.generators.inar import INARGenerator
from synforecast.generators.intermittent_demand import IntermittentDemandGenerator
from synforecast.generators.iot_sensor import IoTSensorGenerator
from synforecast.generators.jump_diffusion import JumpDiffusionGenerator
from synforecast.generators.kernel_synth import KernelSynthGenerator
from synforecast.generators.levy_process import LevyProcessGenerator
from synforecast.generators.mar import MARGenerator
from synforecast.generators.mechanistic import (
    EpidemicGenerator,
    FluidQueueGenerator,
    RainfallRunoffGenerator,
    ThermalGenerator,
)
from synforecast.generators.ornstein_uhlenbeck import OrnsteinUhlenbeckGenerator
from synforecast.generators.poisson_process import PoissonProcessGenerator
from synforecast.generators.polyrhythm import PolyrhythmGenerator
from synforecast.generators.random_walk import RandomWalkGenerator
from synforecast.generators.regime_ou import RegimeOUGenerator
from synforecast.generators.regime_switching import RegimeSwitchingGenerator
from synforecast.generators.sarima import SARIMAGenerator
from synforecast.generators.sarsim import SarSimGenerator
from synforecast.generators.seasonal import SeasonalGenerator
from synforecast.generators.state_space import StateSpaceGenerator
from synforecast.generators.stochastic_volatility import StochasticVolatilityGenerator
from synforecast.generators.tcm import TCMGenerator
from synforecast.generators.tempo_sde import TempoSDEGenerator
from synforecast.generators.tsi import TSIGenerator
from synforecast.generators.var import VARGenerator
from synforecast.generators.vital_signs import VitalSignsGenerator

__all__ = [
    # Statistical
    "RandomWalkGenerator",
    "SeasonalGenerator",
    "SARIMAGenerator",
    "ETSGenerator",
    "INARGenerator",
    # Stochastic
    "GARCHGenerator",
    "OrnsteinUhlenbeckGenerator",
    "GeometricBrownianMotionGenerator",
    "JumpDiffusionGenerator",
    "PoissonProcessGenerator",
    "CyclicGenerator",
    "FractionalBrownianMotionGenerator",
    "HawkesProcessGenerator",
    "StochasticVolatilityGenerator",
    "RegimeSwitchingGenerator",
    "ChaoticSystemGenerator",
    "BoundedProcessGenerator",
    "LevyProcessGenerator",
    # Multivariate
    "CopulaGenerator",
    "VARGenerator",
    "GaussianProcessGenerator",
    # Domain-Specific
    "IntermittentDemandGenerator",
    "IoTSensorGenerator",
    "EnergyLoadGenerator",
    "StateSpaceGenerator",
    "DailyActiveUsersGenerator",
    "VitalSignsGenerator",
    "ClickstreamGenerator",
    # Composition / causal (foundation-model pretraining)
    "TSIGenerator",
    "TCMGenerator",
    "KernelSynthGenerator",
    "MARGenerator",
    # Ported from the Simons TSFM recipe
    "CauKerGenerator",
    "PolyrhythmGenerator",
    "AMFMWaveGenerator",
    "ForecastPFNGenerator",
    "RegimeOUGenerator",
    "EpidemicGenerator",
    "FluidQueueGenerator",
    "RainfallRunoffGenerator",
    "ThermalGenerator",
    # TempoPFN regime SDE and SarSim SARIMA-2
    "TempoSDEGenerator",
    "SarSimGenerator",
]
