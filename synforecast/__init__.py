"""SynForecast: Synthetic Time Series Generation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version(distribution_name="synforecast")
except PackageNotFoundError:
    __version__ = "0+unknown"

try:
    from synforecast import _lib as _lib
except ImportError as exc:
    raise ImportError(
        "SynForecast requires its native Rust extension. Reinstall the package "
        "using a supported wheel, or install a Rust toolchain before building "
        "from source."
    ) from exc

from synforecast.augment import (
    DBA,
    AmplitudeEnvelope,
    BaseAugmenter,
    Censor,
    Downsample,
    FitAndSimulate,
    Mixup,
    MovingBlockBootstrap,
    Negate,
    PowerWarp,
    Reverse,
    Stretch,
)
from synforecast.base import BaseGenerator
from synforecast.dataset import SynAugment, SynSet
from synforecast.evaluation import (
    CoverageDiagnostics,
    CoverageResult,
    compare_feature_coverage,
    compute_features,
    feature_coverage,
)
from synforecast.groups import (
    CoupledSpec,
    FixedRegressionSpec,
    GeneratorSpec,
    Group,
    GroupBatch,
    GroupOverlaySpec,
    GroupPrior,
    ObservationSpec,
    RegressionSpec,
    Term,
    VARXSpec,
    VARXTruth,
    default_driver_pool,
    default_target_pool,
    probe_suite,
)
from synforecast.multivariatize import Multivariatizer
from synforecast.overlays import (
    Anomalies,
    Calendar,
    CalendarCycle,
    Changepoints,
    Halt,
    Missing,
    Observation,
    OverlayConfig,
    Waveform,
)
from synforecast.presets import (
    balanced_pool,
    calendar_overlays,
    halt_overlays,
    interpretable_pool,
    observability_overlays,
    peak_calendar_overlays,
    pretraining_overlays,
    pretraining_pool,
    seasonal_pool,
    step_cycle_overlays,
)
from synforecast.real_augment import RealWindowAugment, WindowPlan
from synforecast.simons_parity import SimonsParity
from synforecast.utils import generate_series

__all__ = [
    "AmplitudeEnvelope",
    "Anomalies",
    "BaseAugmenter",
    "BaseGenerator",
    "Calendar",
    "CalendarCycle",
    "Censor",
    "Changepoints",
    "CoupledSpec",
    "CoverageDiagnostics",
    "CoverageResult",
    "DBA",
    "Downsample",
    "FitAndSimulate",
    "FixedRegressionSpec",
    "GeneratorSpec",
    "Group",
    "GroupBatch",
    "GroupOverlaySpec",
    "GroupPrior",
    "Halt",
    "Missing",
    "Mixup",
    "MovingBlockBootstrap",
    "Multivariatizer",
    "Negate",
    "Observation",
    "ObservationSpec",
    "OverlayConfig",
    "PowerWarp",
    "RealWindowAugment",
    "RegressionSpec",
    "Reverse",
    "SimonsParity",
    "Stretch",
    "SynAugment",
    "SynSet",
    "Term",
    "VARXSpec",
    "VARXTruth",
    "Waveform",
    "WindowPlan",
    "balanced_pool",
    "calendar_overlays",
    "compare_feature_coverage",
    "compute_features",
    "default_driver_pool",
    "default_target_pool",
    "feature_coverage",
    "generate_series",
    "halt_overlays",
    "interpretable_pool",
    "observability_overlays",
    "peak_calendar_overlays",
    "pretraining_overlays",
    "pretraining_pool",
    "probe_suite",
    "seasonal_pool",
    "step_cycle_overlays",
]
