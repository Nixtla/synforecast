"""Validation contracts shared by all public configuration models."""

import inspect

import pytest
from pydantic import ValidationError

from synforecast.exogenous import CalendarFeatures, DiagnosticsConfig
from synforecast.generators import RandomWalkGenerator

BASE = {"min_length": 10, "max_length": 10, "freq": "D"}


def test_generator_rejects_unknown_parameters() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RandomWalkGenerator(**BASE, volatilty=2.0)


def test_runtime_state_is_not_constructor_configuration() -> None:
    signature = inspect.signature(RandomWalkGenerator)
    assert "rng" not in signature.parameters

    with pytest.raises(ValidationError, match="extra_forbidden"):
        RandomWalkGenerator(**BASE, rng="not a generator")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"engine": "numpy"}, "engine"),
        ({"id_col": "y"}, "must be distinct"),
        ({"time_col": ""}, "must not be empty"),
        (
            {"innovation_distribution": "normal", "innovation_params": {"df": 5}},
            "unsupported innovation_params",
        ),
        (
            {"innovation_distribution": "t", "innovation_params": {"df": 2}},
            "must be > 2",
        ),
        (
            {
                "innovation_distribution": "t",
                "innovation_params": {"df": float("nan")},
            },
            "must be > 2",
        ),
        ({"changepoint_locations": [-0.1]}, r"values in \[0, 1\]"),
        ({"changepoint_variance_changes": [0.0]}, "must be positive"),
    ],
)
def test_generator_rejects_invalid_shared_configuration(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        RandomWalkGenerator(**BASE, **kwargs)


def test_column_configs_reject_unknown_and_empty_settings() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CalendarFeatures(datetime_features=True)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DiagnosticsConfig(anomaly_flags=True)
    with pytest.raises(ValidationError, match="pass None"):
        CalendarFeatures(basic=False)
    with pytest.raises(ValidationError, match="pass None"):
        DiagnosticsConfig()


def test_removed_exogenous_argument_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RandomWalkGenerator(**BASE, exogenous=None)


def test_label_columns_do_not_collide_with_output_columns() -> None:
    with pytest.raises(ValidationError, match="collide with output columns"):
        RandomWalkGenerator(
            **{**BASE, "target_col": "anomaly_flag"},
            diagnostics=DiagnosticsConfig(anomalies=True),
        )
