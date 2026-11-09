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
    BaseAugmenter,
    FitAndSimulate,
    Mixup,
    MovingBlockBootstrap,
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
from synforecast.exogenous import CalendarFeatures, DiagnosticsConfig
from synforecast.groups import (
    CoupledSpec,
    FixedRegressionSpec,
    GeneratorSpec,
    Group,
    GroupBatch,
    GroupPrior,
    RegressionSpec,
    RoleSpec,
    Term,
    VARXSpec,
    VARXTruth,
    probe_suite,
)
from synforecast.overlays import (
    Anomalies,
    Calendar,
    CalendarCycle,
    Changepoints,
    Missing,
    Observation,
    OverlayConfig,
    Waveform,
)
from synforecast.presets import (
    balanced_pool,
    calendar_overlays,
    interpretable_pool,
    observability_overlays,
    pretraining_overlays,
    pretraining_pool,
    seasonal_pool,
)
from synforecast.utils import generate_series

__all__ = [
    "DBA",
    "DiagnosticsConfig",
    "Anomalies",
    "BaseAugmenter",
    "BaseGenerator",
    "Calendar",
    "CalendarFeatures",
    "CalendarCycle",
    "Changepoints",
    "CoupledSpec",
    "CoverageDiagnostics",
    "CoverageResult",
    "FitAndSimulate",
    "FixedRegressionSpec",
    "GeneratorSpec",
    "Group",
    "GroupBatch",
    "GroupPrior",
    "Missing",
    "Mixup",
    "MovingBlockBootstrap",
    "Observation",
    "OverlayConfig",
    "RegressionSpec",
    "RoleSpec",
    "SynAugment",
    "SynSet",
    "Term",
    "VARXSpec",
    "VARXTruth",
    "Waveform",
    "balanced_pool",
    "calendar_overlays",
    "compare_feature_coverage",
    "compute_features",
    "feature_coverage",
    "generate_series",
    "interpretable_pool",
    "observability_overlays",
    "pretraining_overlays",
    "pretraining_pool",
    "probe_suite",
    "seasonal_pool",
]
