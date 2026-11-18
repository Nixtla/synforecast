"""Simons synthetic covariate mechanisms and curriculum (opt-in; needs scipy).

Port of the covariate side of the Simons TSFM recipe (Apache-2.0,
``simons/synth/covariates`` and ``simons/training/covariates/roles.py``):

- :func:`synthetic_window` (``effects``): synthetic covariate effects on a base
  window (the stage-3/4 covariate pass on corpus series).
- ``families.make_task`` with five effect families: ``generic`` (forecast,
  weather, promo, closure, price, schedule, setpoint, decoy covariates),
  ``classical`` (transfer functions, interventions, dynamic regression,
  weather, marketing, capacity, interactions, issued forecasts), ``energy``
  (load, PV, wind, price, tariff from latent weather with issued forecasts),
  ``retail`` (SCAN*PRO-style price and promotion lifts) and ``lagged``
  (lagged, thresholded and differenced responses).
- :func:`roles.effect_row`: 32-slot rows of effect tasks (target slot, then
  its known-future covariate slots, some cut before the horizon end).
- :class:`roles.RolePlanner`: covariate roles, end blocks and future cuts for
  coupled groups (roots of lag/event/func groups, random siblings otherwise).
- :class:`CovariateCurriculum`: per-stage role/effect shares, family mix and
  covariate-pass mix (:meth:`CovariateCurriculum.from_simons_stage`).

Nothing here runs unless called: existing generators and presets are unchanged.
The numpy random stream is consumed as in the reference, so the same generator
state and base windows give the same outputs.
"""

try:
    import scipy  # noqa: F401
except ImportError as error:
    raise ImportError(
        "synforecast.simons_covariates needs scipy (pip install scipy)"
    ) from error

from synforecast.simons_covariates import (
    classical,
    effects,
    energy,
    families,
    generic,
    roles,
)
from synforecast.simons_covariates.curriculum import (
    CURRICULUM_VERSION,
    PASS_KINDS,
    BaseWindows,
    CovariateCurriculum,
    EffectRowStream,
    synthetic_window,
)
from synforecast.simons_covariates.roles import (
    RolePlanner,
    effect_row,
    role_meta_from_roles,
    structure_from_recipes,
)

__all__ = [
    "CURRICULUM_VERSION",
    "PASS_KINDS",
    "BaseWindows",
    "CovariateCurriculum",
    "EffectRowStream",
    "RolePlanner",
    "classical",
    "effect_row",
    "effects",
    "energy",
    "families",
    "generic",
    "role_meta_from_roles",
    "roles",
    "structure_from_recipes",
    "synthetic_window",
]
