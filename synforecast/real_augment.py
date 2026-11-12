"""Per-window augmentation of real series for training loaders.

:class:`RealWindowAugment` composes the real-window specs of
:mod:`synforecast.augment` (``Downsample``, ``Stretch``, ``PowerWarp``,
``AmplitudeEnvelope``, ``Censor``, ``Negate``, ``Reverse``) with pairwise
mixup, each applied at its own rate. A loader calls it per series on numpy
arrays with its own ``numpy.random.Generator``:

1. ``plan = aug.draw(rng, length, want)`` draws every data-independent
   parameter, consuming the same draws whatever the rates, so switching one
   spec on or off never moves another's parameters.
2. If ``plan.resample`` is set, the loader reads the raw slice
   ``plan.resample.start:stop`` of its series; otherwise its own crop.
3. If ``plan.mixup`` is set, the loader supplies a partner window of the
   same length (another real series).
4. ``values, mask, applied = aug.transform(values, mask, plan, partner,
   partner_mask)`` applies, in order: resampling (downsample, else stretch),
   power warp, mixup, censor, amplitude envelope, negate, reverse.

Data-dependent quantities (power-warp scale, censor quantile, mixup scales,
bridge step scale) read only the first ``prefix`` observed values: the
bridge scale those of the raw slice, the value steps only positions before
the end of the first ``prefix`` observed values of the window entering them,
and the partner's scale its own first ``prefix``. Values after that prefix
never change a parameter (``trace`` returns them). Values before it can
still shape labels inside it; that is stated, not hidden.
"""

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from synforecast import _window
from synforecast.augment import (
    AmplitudeEnvelope,
    Censor,
    Downsample,
    Float,
    Int,
    PowerWarp,
    Stretch,
)

Rate = Float
"""A probability in [0, 1]."""


@dataclass(frozen=True)
class WindowPlan:
    """Data-independent draws for one window; None or False means the spec is off."""

    resample: _window.Resample | None = None
    power: float | None = None
    mixup: float | None = None
    censor: tuple[bool, float] | None = None
    envelope: tuple[np.ndarray, np.ndarray] | None = field(default=None, compare=False)
    negate: bool = False
    reverse: bool = False

    @property
    def active(self) -> bool:
        """Whether any spec was drawn."""
        return any(
            (
                self.resample is not None,
                self.power is not None,
                self.mixup is not None,
                self.censor is not None,
                self.envelope is not None,
                self.negate,
                self.reverse,
            )
        )


class RealWindowAugment(BaseModel):
    """Rates and settings of per-window real-series augmentation.

    Stretch is drawn only when downsampling is not, so its effective rate is
    ``stretch`` times the share of windows not downsampled. Downsampling a
    series too short for ``min_points`` buckets is not applied (and leaves
    stretch its chance). Mixup's weight is lam ~ Beta(``mixup_alpha``,
    ``mixup_alpha``) clipped to [``mixup_min``, ``mixup_max``] on the anchor.

    Args:
        downsample, stretch, power_warp, amplitude_envelope, censor, mixup,
            negate, reverse (float): Per-window probabilities (default: 0).
        mixup_alpha (float): Beta concentration of the mixup weight (default: 1.5).
        mixup_min, mixup_max (float): Clip range of the weight (default: 0.2, 0.8).
        prefix (int): Observed values read by the mixup scales (default: 512).
        downsample_spec, stretch_spec, power_warp_spec, amplitude_envelope_spec,
            censor_spec: Settings of each spec (their defaults follow Simons).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    downsample: Rate = Field(default=0.0, ge=0, le=1)
    stretch: Rate = Field(default=0.0, ge=0, le=1)
    power_warp: Rate = Field(default=0.0, ge=0, le=1)
    amplitude_envelope: Rate = Field(default=0.0, ge=0, le=1)
    censor: Rate = Field(default=0.0, ge=0, le=1)
    mixup: Rate = Field(default=0.0, ge=0, le=1)
    negate: Rate = Field(default=0.0, ge=0, le=1)
    reverse: Rate = Field(default=0.0, ge=0, le=1)
    mixup_alpha: Float = Field(default=1.5, gt=0)
    mixup_min: Float = Field(default=0.2, ge=0, le=1)
    mixup_max: Float = Field(default=0.8, ge=0, le=1)
    prefix: Int = Field(default=_window.PREFIX, ge=1)
    downsample_spec: Downsample = Field(default_factory=Downsample)
    stretch_spec: Stretch = Field(default_factory=Stretch)
    power_warp_spec: PowerWarp = Field(default_factory=PowerWarp)
    amplitude_envelope_spec: AmplitudeEnvelope = Field(
        default_factory=AmplitudeEnvelope
    )
    censor_spec: Censor = Field(default_factory=Censor)

    @property
    def active(self) -> bool:
        """Whether any rate is positive."""
        return any(
            rate > 0
            for rate in (
                self.downsample,
                self.stretch,
                self.power_warp,
                self.amplitude_envelope,
                self.censor,
                self.mixup,
                self.negate,
                self.reverse,
            )
        )

    def draw(self, rng: np.random.Generator, length: int, want: int) -> WindowPlan:
        """Every data-independent draw for one window of a ``length``-point series.

        ``want`` is the longest window the loader accepts; a resampling draw
        returns at most ``want`` points. The draws consumed do not depend on
        the rates or on ``length``.
        """
        gates = rng.random(8)
        ds = self.downsample_spec.draw(rng)
        st = self.stretch_spec.draw(rng)
        power = self.power_warp_spec.draw(rng)
        censor = self.censor_spec.draw(rng)
        envelope = self.amplitude_envelope_spec.draw(rng)
        lam = float(
            np.clip(
                rng.beta(self.mixup_alpha, self.mixup_alpha),
                self.mixup_min,
                self.mixup_max,
            )
        )
        resample = None
        if gates[0] < self.downsample:
            resample = self.downsample_spec.plan(ds, length, want)
        if resample is None and gates[1] < self.stretch and length >= 2:
            resample = self.stretch_spec.plan(st, length, want)
        return WindowPlan(
            resample=resample,
            power=power if gates[2] < self.power_warp else None,
            mixup=lam if gates[3] < self.mixup else None,
            censor=censor if gates[4] < self.censor else None,
            envelope=envelope if gates[5] < self.amplitude_envelope else None,
            negate=bool(gates[6] < self.negate),
            reverse=bool(gates[7] < self.reverse),
        )

    def transform(
        self,
        values: np.ndarray,
        mask: np.ndarray | None,
        plan: WindowPlan,
        partner: np.ndarray | None = None,
        partner_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Apply ``plan`` to a window; returns values (NaN where unobserved), mask and applied specs.

        With ``plan.resample`` set, ``values`` must be the raw slice it names.
        Mixup is skipped without a partner of the (resampled) window's
        length. A spec that cannot act on this window (power warp on a prefix
        with a negative value, censor below two observations) is skipped and
        left out of ``applied``.
        """
        x, ok, applied, _ = self.trace(values, mask, plan, partner, partner_mask)
        return x, ok, applied

    def trace(
        self,
        values: np.ndarray,
        mask: np.ndarray | None,
        plan: WindowPlan,
        partner: np.ndarray | None = None,
        partner_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, float | None]]:
        """``transform`` plus every data-dependent quantity it read.

        The value steps read only positions before ``limit``, the end of the
        first ``prefix`` observed values of the window entering them (after
        resampling), so later values never change a quantity. The mixup
        partner's scale reads its own first ``prefix`` observed values.
        """
        x, ok = _window.as_pair(values, mask)
        applied: list[str] = []
        used: dict[str, float | None] = {}
        if plan.resample is not None:
            prefix = self.stretch_spec.prefix
            if plan.resample.kind == "up":
                used["bridge_sigma"] = _window.bridge_sigma(x, ok, prefix)
            x, ok = _window.resample(x, ok, plan.resample, prefix)
            applied.append(
                "downsample" if plan.resample.kind == "downsample" else "stretch"
            )
        limit = _window.prefix_end(ok, self.prefix)
        used["limit"] = limit
        if plan.power is not None:
            prefix = self.power_warp_spec.prefix
            result = _window.power_warp(x, ok, plan.power, prefix, limit)
            used["power_scale"] = _window.mean_abs(x, ok, prefix, limit)
            if result is not None:
                x, ok = result
                applied.append("power_warp")
        if plan.mixup is not None and partner is not None:
            xb, okb = _window.as_pair(partner, partner_mask)
            if len(xb) == len(x):
                used["mixup_scale"] = _window.mean_abs(x, ok, self.prefix, limit)
                used["mixup_partner_scale"] = _window.mean_abs(xb, okb, self.prefix)
                result = _window.mixup(x, ok, xb, okb, plan.mixup, self.prefix, limit)
                if result is not None:
                    x, ok = result
                    applied.append("mixup")
        if plan.censor is not None:
            prefix = self.censor_spec.prefix
            used["censor_level"] = _window.censor_level(
                x, ok, plan.censor[1], prefix, limit
            )
            result = _window.censor(x, ok, *plan.censor, prefix, limit)
            if result is not None:
                x, ok = result
                applied.append("censor")
        if plan.envelope is not None:
            x, ok = _window.amplitude_envelope(x, ok, *plan.envelope)
            applied.append("amplitude_envelope")
        if plan.negate:
            x, ok = _window.negate(x, ok)
            applied.append("negate")
        if plan.reverse:
            x, ok = _window.reverse(x, ok)
            applied.append("reverse")
        return x, ok, tuple(applied), used
