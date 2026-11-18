"""The Simons synthetic covariate curriculum as an opt-in sampler.

Stage 2 (12k synthetic steps) gives a phase-dependent share of coupled
groups covariate roles (0.1, 0.2, 0.4 at run fractions 0.27 / 0.53) and
replaces a phase-dependent share of rows by synthetic effect rows (0.01, 0.02,
0.04) whose tasks come from five families (generic 0.2, classical 0.3,
energy 0.2, retail 0.15, lagged 0.15). Stages 3, 4 and 4-long keep a constant
role share of 0.4 and add a covariate loss pass (weight 0.025) whose windows
are synthetic effects on corpus series (0.63), sibling pairs (0.25) and real
known-future windows (0.12); only the synthetic-effect windows are generated
here (:func:`synthetic_window`), sibling and real windows are real data.
"""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from synforecast.simons_covariates import effects, families, roles

CURRICULUM_VERSION = "simons-covariates-v1"
"""Version tag of the covariate mechanisms and curriculum below."""

PASS_KINDS = ("synthetic", "sibling", "real")
Stage = Literal["stage2", "stage3", "stage4", "stage4-long"]


class CovariateCurriculum(BaseModel):
    """Role/effect shares per phase, effect family mix and covariate-pass settings of one training stage.

    Args:
        boundaries (list[float]): Phase boundaries as fractions of the stage's steps.
        role_share (list[float]): Share of coupled groups given covariate roles, one per phase.
        effect_share (list[float]): Share of rows replaced by effect rows, one per phase.
        effect_mix (dict[str, float] | None): Weights of the effect families (generic, classical, energy, retail, lagged).
        pass_weight (float): Loss weight of the covariate pass (0: no pass).
        pass_mix (dict[str, float]): Window kinds of the covariate pass (synthetic, sibling, real).
        long_weight (float): Loss weight of the long-history pass (real windows only; 0: no pass).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    boundaries: list[float] = Field(default_factory=list)
    role_share: list[float] = Field(default_factory=lambda: [0.0])
    effect_share: list[float] = Field(default_factory=lambda: [0.0])
    effect_mix: dict[str, float] | None = None
    pass_weight: float = Field(default=0.0, ge=0.0)
    pass_mix: dict[str, float] = Field(
        default_factory=lambda: {"synthetic": 0.0, "sibling": 0.0, "real": 0.0}
    )
    long_weight: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check(self):
        for name in ("boundaries", "role_share", "effect_share"):
            values = getattr(self, name)
            if any(not 0.0 <= v <= 1.0 for v in values):
                raise ValueError(f"{name} must hold fractions from 0 to 1")
        if list(self.boundaries) != sorted(self.boundaries):
            raise ValueError("boundaries must be increasing")
        phases = len(self.boundaries) + 1
        if len(self.role_share) not in (1, phases) or len(self.effect_share) not in (
            1,
            phases,
        ):
            raise ValueError(
                "role_share and effect_share need one value, or one value per phase"
            )
        if max(self.effect_share) > 0 and self.effect_mix is None:
            raise ValueError("effect rows (effect_share > 0) need effect_mix")
        if self.effect_mix is not None:
            families.parse_mix(self.effect_mix)
        if set(self.pass_mix) != set(PASS_KINDS) or min(self.pass_mix.values()) < 0:
            raise ValueError(f"pass_mix needs nonnegative weights for {PASS_KINDS}")
        if self.pass_weight > 0 and sum(self.pass_mix.values()) <= 0:
            raise ValueError(
                "a covariate pass (pass_weight > 0) needs a positive pass_mix"
            )
        return self

    @classmethod
    def from_simons_stage(cls, stage: Stage) -> "CovariateCurriculum":
        """The Simons-25M settings of ``stage`` (configs/simons-25m-*.yaml; Simons calls stage 2 ``stage1``)."""
        if stage == "stage2":
            return cls(boundaries=[0.27, 0.53], role_share=[0.1, 0.2, 0.4], effect_share=[0.01, 0.02, 0.04],
                       effect_mix={"generic": 0.2, "classical": 0.3, "energy": 0.2, "retail": 0.15, "lagged": 0.15})  # fmt: skip
        if stage not in ("stage3", "stage4", "stage4-long"):
            raise ValueError(f"unknown Simons stage {stage!r}")
        return cls(role_share=[0.4], effect_share=[0.0], pass_weight=0.025,
                   pass_mix={"synthetic": 0.63, "sibling": 0.25, "real": 0.12},
                   long_weight=0.0 if stage == "stage3" else 0.025)  # fmt: skip

    @classmethod
    def off(cls) -> "CovariateCurriculum":
        """No roles, no effect rows, no synthetic covariate windows (the default)."""
        return cls()

    @property
    def enabled(self) -> bool:
        return (
            max(self.role_share) > 0
            or max(self.effect_share) > 0
            or (self.pass_weight > 0 and self.pass_mix["synthetic"] > 0)
        )

    def _per_phase(self, values):
        phases = len(self.boundaries) + 1
        return list(values) * phases if len(values) == 1 else list(values)

    def role_spec(self, effect_workers: int = 4) -> dict | None:
        """The role settings (``roles.settings`` input; Simons configs run 4 effect workers), or None when no roles and no effect rows are on."""
        role, effect = (
            self._per_phase(self.role_share),
            self._per_phase(self.effect_share),
        )
        if max(role) <= 0 and max(effect) <= 0:
            return None
        return {"boundaries": list(self.boundaries), "p_role": role, "p_eff": effect,
                "effect_mix": dict(self.effect_mix) if self.effect_mix is not None else None,
                "effect_workers": int(effect_workers)}  # fmt: skip

    def p_role(self, fraction: float) -> float:
        return float(
            roles.phase_value(
                self._per_phase(self.role_share), self.boundaries, fraction
            )
        )

    def p_effect(self, fraction: float) -> float:
        return float(
            roles.phase_value(
                self._per_phase(self.effect_share), self.boundaries, fraction
            )
        )

    def effect_rows(self, rows: int, fraction: float, rng: np.random.Generator) -> int:
        """Effect rows among ``rows`` at ``fraction`` of the stage: floor(p * rows) plus one with the remainder's probability."""
        share = self.p_effect(fraction) * rows
        n = int(np.floor(share) + (rng.random() < share - np.floor(share)))
        return min(n, rows)


class BaseWindows:
    """Base series for :func:`synthetic_window`: ``uni`` is a list of (values, frequency) pairs."""

    def __init__(self, uni):
        self.uni = list(uni)


def synthetic_window(base, rng: np.random.Generator, hist: int, horizon: int):
    """(target float32 [hist + horizon], covariates float32 [C, hist + horizon]): synthetic effects on a base window.

    ``base.uni`` lists (values, frequency) pairs (corpus series in Simons'
    covariate pass); the target is a random window of one of them plus the
    effects of 1-6 generated or real covariates (event, smooth, step, calendar,
    issued forecast, another base window), 35% of draws with no relevant one.
    """
    return effects.synthetic(base, rng, hist, horizon)


class EffectRowStream:
    """Endless effect rows of one worker: ``window(n)`` supplies base windows, the generator is the worker's own."""

    def __init__(self, spec: dict, rng: np.random.Generator, window, length: int):
        self.s = roles.settings(spec)
        if self.s["effect_mix"] is None:
            raise ValueError("effect rows need effect_mix")
        self.rng, self.window, self.length = rng, window, int(length)

    def __iter__(self):
        return self

    def __next__(self):
        return roles.effect_row(self.s, self.rng, self.window, self.length)
