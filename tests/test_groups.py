"""Tests for grouped covariate tasks: the regression mechanisms and the
sampling machinery of synforecast.groups."""

import hashlib
import json
import subprocess
import sys
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from synforecast import (
    FixedRegressionSpec,
    GeneratorSpec,
    GroupPrior,
    RegressionSpec,
    Term,
    default_driver_pool,
    default_target_pool,
    probe_suite,
)
from synforecast._lib import groups as rs
from synforecast.base import BaseGenerator
from synforecast.generators import (
    EnergyLoadGenerator,
    IntermittentDemandGenerator,
    KernelSynthGenerator,
    RandomWalkGenerator,
    SARIMAGenerator,
)

WHITE = {"p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "noise_std": 1.0}
WHITE_SPEC = GeneratorSpec(generator=SARIMAGenerator, params=WHITE)


class _Constant(BaseGenerator):
    """Constant series: a legitimate channel with zero variance."""

    def generate_single_series(self, length: int) -> np.ndarray:
        return np.full(length, 3.0)


class _Broken(BaseGenerator):
    """Always non-finite output."""

    def generate_single_series(self, length: int) -> np.ndarray:
        return np.full(length, np.inf)


class _Flaky(BaseGenerator):
    """Non-finite on about half of the draws."""

    def generate_single_series(self, length: int) -> np.ndarray:
        if self.rng.random() < 0.5:
            return np.full(length, np.inf)
        return self.rng.normal(size=length)


class _Raising(BaseGenerator):
    """Raises on about half of the draws."""

    def generate_single_series(self, length: int) -> np.ndarray:
        if self.rng.random() < 0.5:
            raise ValueError("unlucky draw")
        return self.rng.normal(size=length)


class _Bump(BaseGenerator):
    """Positive noise, plus 50 from ``bump_at`` on (a class switch)."""

    bump_at: ClassVar[int | None] = None

    def generate_single_series(self, length: int) -> np.ndarray:
        x = self.rng.normal(size=length) + 3.0
        if type(self).bump_at is not None:
            x[type(self).bump_at :] += 50.0
        return np.abs(x)


class _Ramp(BaseGenerator):
    """Deterministic ramp from -3 to 3."""

    def generate_single_series(self, length: int) -> np.ndarray:
        return np.linspace(-3.0, 3.0, length)


class _Wave(BaseGenerator):
    """Deterministic sine wave."""

    def generate_single_series(self, length: int) -> np.ndarray:
        return np.sin(np.arange(length) * 0.37)


def _linear_prior(burn_in: int = 256, **overrides) -> GroupPrior:
    """Stationary pools, linear edges only, no filters: identifiable OLS."""
    spec = {
        "target_pool": [WHITE_SPEC],
        "driver_pool": [WHITE_SPEC],
        "n_covariates": (3, 3),
        "n_informative": (2, 2),
        "p_null": 0.0,
        "edge_kinds": ["linear"],
        "max_lag": 4,
        "p_interaction": 0.0,
        "p_adstock": 0.0,
        "p_persistence": 0.0,
        "p_target_ar": 0.0,
        "p_multiplicative": 0.0,
        "effect_share": (0.8, 0.8),
    }
    spec.update(overrides)
    return GroupPrior(
        mechanism=RegressionSpec(**spec), length=(4000, 4000), freq="D", burn_in=burn_in
    )


def _spec_for(prior: GroupPrior, description: dict) -> GeneratorSpec:
    return next(
        s for s in prior.mechanism.generator_specs() if s.describe() == description
    )


def _fixed(terms, drivers, **kw) -> GroupPrior:
    mech = FixedRegressionSpec(drivers=drivers, terms=terms, **kw)
    return GroupPrior(mechanism=mech, length=(60, 60), freq="D", burn_in=32)


def _compose_one(base, drivers, terms, persistence=0.0, target_ar=0.0):
    """Call the native composition on one group (raw drivers, no calibration)."""
    drivers = np.atleast_2d(np.asarray(drivers, dtype=float))
    t = len(base)
    kinds = {"linear": 0, "saturating": 1, "quadratic": 2, "product": 7, "gate": 8}
    res = rs.compose_batch(
        np.asarray(base, dtype=float),
        np.array([0, t], dtype=np.int64),
        drivers.ravel(),
        np.array([drivers.shape[0]], dtype=np.int64),
        np.array([0, len(terms)], dtype=np.int64),
        np.array([kinds[k["kind"]] for k in terms], dtype=np.uint8),
        np.array([k.get("driver", 0) for k in terms], dtype=np.int64),
        np.array([k.get("partner", -1) for k in terms], dtype=np.int64),
        np.array([k.get("lag", 0) for k in terms], dtype=np.int64),
        np.array([k.get("partner_lag", 0) for k in terms], dtype=np.int64),
        np.array([k.get("coef", 1.0) for k in terms], dtype=float),
        np.array([k.get("param", np.nan) for k in terms], dtype=float),
        np.array([k.get("adstock", np.nan) for k in terms], dtype=float),
        np.array([persistence]),
        np.array([target_ar]),
        np.array([0], dtype=np.int32),
        np.array([0.5]),
        np.array([0.1]),
        8,
        False,
        1,
    )
    return res[0]


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mechanism",
    [
        RegressionSpec(
            target_pool=[GeneratorSpec(generator=_Bump, positive=True)],
            driver_pool=[GeneratorSpec(generator=_Bump)],
            p_null=0.0,
            p_multiplicative=0.0,
        ),
        RegressionSpec(
            target_pool=[GeneratorSpec(generator=_Bump, positive=True)],
            driver_pool=[GeneratorSpec(generator=_Bump)],
            p_null=0.0,
            p_multiplicative=1.0,
            log_effect_std=(0.05, 0.05),
        ),
        FixedRegressionSpec(
            drivers=[GeneratorSpec(generator=_Bump)],
            terms=[Term(kind="linear", driver=0, coef=1.0, lag=2)],
            target=GeneratorSpec(generator=_Bump),
            target_ar=0.5,
        ),
    ],
    ids=["additive", "multiplicative", "fixed"],
)
def test_groups_are_prefix_invariant(mechanism, monkeypatch):
    """A change in the latent series at step `cut` leaves every emitted value
    before it untouched."""
    prior = GroupPrior(mechanism=mechanism, length=(200, 200), freq="D", burn_in=96)
    monkeypatch.setattr(_Bump, "bump_at", None)
    plain = prior.sample(4, seed=3)
    monkeypatch.setattr(_Bump, "bump_at", 96 + 120)
    bumped = prior.sample(4, seed=3)
    for a, b in zip(plain, bumped, strict=True):
        np.testing.assert_array_equal(a.values[:, :120], b.values[:, :120])
        assert not np.allclose(a.values[:, 120:], b.values[:, 120:])


def test_lags_do_not_wrap():
    prior = _fixed(
        [Term(kind="linear", driver=0, coef=1.0, lag=3)],
        [GeneratorSpec(generator=_Ramp)],
    )
    g = prior.sample(1, seed=0)[0]
    x = g.values[g.covariate_indices[0]]
    np.testing.assert_allclose(g.values[g.target_index][3:], x[:-3])
    # The first emitted values read the burn-in, not the end of the series
    assert g.values[g.target_index][0] < x[0]


# ---------------------------------------------------------------------------
# Kernels against closed forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "param", "formula"),
    [
        ("linear", None, lambda x: x),
        ("saturating", None, np.tanh),
        ("saturating", 2.0, lambda x: 2.0 * np.tanh(x / 2.0)),
        ("quadratic", None, lambda x: x**2),
        ("signed_sqrt", None, lambda x: np.sign(x) * np.sqrt(np.abs(x))),
        ("abs_sqrt", None, lambda x: np.sqrt(np.abs(x))),
        ("threshold", 0.5, lambda x: (x > 0.5).astype(float)),
        ("threshold", None, lambda x: (x > 0.0).astype(float)),
        ("hinge", -0.2, lambda x: np.maximum(x + 0.2, 0.0)),
    ],
)
def test_edge_kernels_match_closed_forms(kind, param, formula):
    prior = _fixed(
        [Term(kind=kind, driver=0, coef=1.5, param=param)],
        [GeneratorSpec(generator=_Ramp)],
    )
    g = prior.sample(1, seed=0)[0]
    x = g.values[g.covariate_indices[0]]
    np.testing.assert_allclose(g.values[g.target_index], 1.5 * formula(x), atol=1e-12)


@pytest.mark.parametrize("kind", ["product", "gate"])
def test_interactions_match_closed_forms(kind):
    term = Term(
        kind=kind,
        driver=0,
        partner=1,
        coef=2.0,
        lag=1,
        partner_lag=3,
        param=0.1 if kind == "gate" else None,
    )
    prior = _fixed(
        [term], [GeneratorSpec(generator=_Ramp), GeneratorSpec(generator=_Wave)]
    )
    g = prior.sample(1, seed=0)[0]
    rec = g.recipe["terms"][0]
    x, w = g.values[rec["driver_channel"]], g.values[rec["partner_channel"]]
    y = g.values[g.target_index]
    other = w[:-3] if kind == "product" else (w[:-3] > 0.1)
    np.testing.assert_allclose(y[3:], 2.0 * x[2:-1] * other, atol=1e-12)


def test_adstock_and_persistence_closed_forms():
    const = np.ones((1, 40))
    adstock = _compose_one(np.zeros(40), const, [{"kind": "linear", "adstock": 0.7}])
    np.testing.assert_allclose(adstock, 1.0)
    persistent = _compose_one(
        np.zeros(40), const, [{"kind": "linear"}], persistence=0.5
    )
    np.testing.assert_allclose(persistent, 2.0)
    impulse = np.zeros((1, 40))
    impulse[0, 10] = 1.0
    out = _compose_one(np.zeros(40), impulse, [{"kind": "linear", "adstock": 0.6}])
    np.testing.assert_allclose(out[10:20], 0.4 * 0.6 ** np.arange(10))
    np.testing.assert_allclose(out[:10], 0.0)


# ---------------------------------------------------------------------------
# Recipes and reproducibility
# ---------------------------------------------------------------------------


def test_recipe_reconstructs_every_channel():
    prior = GroupPrior(length=(64, 256), freq="D", burn_in=96)
    batch = prior.sample(40, seed=3, components=True)
    compositions = set()
    for g in batch:
        r = g.recipe
        b, total = r["burn_in"], r["burn_in"] + r["length"]
        compositions.add(r["composition"])
        for ch in r["channels"]:
            latent = prior.draw_channel(_spec_for(prior, ch["spec"]), total, ch["seed"])
            if ch["kind"] == "covariate":
                np.testing.assert_array_equal(latent[b:], g.values[ch["channel"]])
                assert ch["calibration_mean"] == pytest.approx(latent[:b].mean())
            else:
                np.testing.assert_array_equal(latent[b:], g.components["base"])
        if r["target_ar"] == 0:
            base, effect = g.components["base"], g.components["effect"]
            multiplicative = r["composition"] == "multiplicative"
            expect = base * np.exp(effect) if multiplicative else base + effect
            np.testing.assert_allclose(g.values[g.target_index], expect, rtol=1e-12)
    assert compositions == {"additive", "multiplicative"}


@pytest.mark.parametrize(
    "mechanism",
    [RegressionSpec(), probe_suite()["sum/phi0.3/past/d2/smooth"].mechanism],
    ids=["regression", "fixed"],
)
def test_recipes_survive_json_round_trip(mechanism):
    prior = GroupPrior(mechanism=mechanism, length=(48, 48), burn_in=128)
    for g in prior.sample(3, seed=5):
        loaded = json.loads(json.dumps(g.recipe, allow_nan=False))
        again = prior.replay(loaded)
        assert again.recipe == g.recipe
        np.testing.assert_array_equal(again.values, g.values)


def test_groups_do_not_depend_on_batching_or_workers():
    prior = GroupPrior(length=(32, 96), freq="h")
    batch = prior.sample(10, seed=11)
    serial = prior.sample(10, seed=11, n_jobs=1)
    for i, g in enumerate(batch):
        single = prior.sample(1, seed=11, start_group=i, n_jobs=1)[0]
        np.testing.assert_array_equal(single.values, g.values)
        np.testing.assert_array_equal(serial[i].values, g.values)
        assert single.recipe == g.recipe
    assert not np.array_equal(batch[0].values[0], batch[1].values[0])


def test_minimal_recipes_replay_to_full_ones():
    prior = GroupPrior(length=(32, 64))
    full = prior.sample(3, seed=2)
    light = prior.sample(3, seed=2, recipes=False)
    for f, g in zip(full, light, strict=True):
        np.testing.assert_array_equal(f.values, g.values)
        assert set(g.recipe) == {
            "version",
            "prior",
            "seed",
            "group_id",
            "attempt",
            "mechanism",
        }
        assert prior.replay(g.recipe).recipe == f.recipe


def test_cross_process_determinism():
    code = (
        "import hashlib, numpy as np, synforecast as sf;"
        "b = sf.GroupPrior(length=(32, 64)).sample(3, seed=7);"
        "v = np.concatenate([g.values.ravel() for g in b]);"
        "print(hashlib.sha256(v.tobytes()).hexdigest())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "123"},
    ).stdout.strip()
    batch = GroupPrior(length=(32, 64)).sample(3, seed=7)
    local = hashlib.sha256(np.concatenate([g.values.ravel() for g in batch]).tobytes())
    assert out == local.hexdigest()


def test_default_pools_draw_deterministically():
    prior = GroupPrior(mechanism=RegressionSpec(), length=(32, 32))
    for spec in default_driver_pool() + default_target_pool():
        a = prior.draw_channel(spec, 200, 1)
        np.testing.assert_array_equal(a, prior.draw_channel(spec, 200, 1))
        assert not np.array_equal(a, prior.draw_channel(spec, 200, 2))


def test_replay_rejects_mismatches():
    prior = GroupPrior(length=(32, 64))
    recipe = prior.sample(1, seed=0)[0].recipe
    with pytest.raises(ValueError, match="different prior"):
        GroupPrior(length=(32, 65)).replay(recipe)
    with pytest.raises(ValueError, match="version"):
        prior.replay({**recipe, "version": -1})
    with pytest.raises(ValueError, match="different attempt"):
        prior.replay({**recipe, "attempt": recipe["attempt"] + 1})


def test_constructor_randomized_generators_draw_fresh_parameters():
    spec = GeneratorSpec(generator=SARIMAGenerator, params={"p": 2, "q": 1})
    prior = GroupPrior(
        mechanism=RegressionSpec(target_pool=[spec], driver_pool=[spec]),
        length=(64, 64),
    )
    assert prior._source(spec).mode == "randomized"
    a = prior.draw_channel(spec, 200, 1)
    np.testing.assert_array_equal(a, prior.draw_channel(spec, 200, 1))
    assert not np.allclose(a, prior.draw_channel(spec, 200, 2))
    walk = GeneratorSpec(generator=RandomWalkGenerator)
    shared = GroupPrior(
        mechanism=RegressionSpec(target_pool=[walk], driver_pool=[walk])
    )
    assert shared._source(walk).mode == "batch"


def test_generator_spec_validation():
    with pytest.raises(ValidationError, match="prior owns"):
        GeneratorSpec(generator=SARIMAGenerator, params={"seed": 1})
    with pytest.raises(ValidationError):
        GeneratorSpec(generator=int)
    with pytest.raises(ValidationError, match="JSON-serializable"):
        GeneratorSpec(generator=SARIMAGenerator, params={"mean": object()})
    spec = GeneratorSpec(
        generator=SARIMAGenerator,
        params={
            "p": 1,
            "q": 0,
            "ar_params": np.array([0.5]),
            "noise_std": np.float64(1),
        },
    )
    assert spec.params == {"p": 1, "q": 0, "ar_params": [0.5], "noise_std": 1.0}
    bad = GeneratorSpec(generator=SARIMAGenerator, params={"p": 2, "ar_params": [0.5]})
    with pytest.raises(ValidationError, match="rejects params"):
        GroupPrior(mechanism=RegressionSpec(target_pool=[bad]))


def test_generator_identity_includes_module():
    def make(module: str) -> type[BaseGenerator]:
        return type("SameName", (_Constant,), {"__module__": module})

    a = GeneratorSpec(generator=make("pkg_a"))
    b = GeneratorSpec(generator=make("pkg_b"))
    assert a.describe()["generator"] != b.describe()["generator"]
    prior_a = GroupPrior(mechanism=RegressionSpec(target_pool=[a], driver_pool=[a]))
    prior_b = GroupPrior(mechanism=RegressionSpec(target_pool=[b], driver_pool=[b]))
    assert prior_a.fingerprint != prior_b.fingerprint
    with pytest.raises(ValueError, match="different prior"):
        prior_b.replay(prior_a.sample(1, seed=0)[0].recipe)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_linear_coefficients_are_recovered():
    group = _linear_prior().sample(1, seed=5)[0]
    recipe = group.recipe
    y = group.values[group.target_index]
    cols, truth = [], []
    for term in recipe["terms"]:
        cov = recipe["channels"][term["driver"]]
        assert cov["informative"] and cov["channel"] == term["driver_channel"]
        x = (group.values[cov["channel"]] - cov["calibration_mean"]) / cov[
            "calibration_std"
        ]
        cols.append(np.roll(x, term["lag"]))
        truth.append(term["coef"] * recipe["effect_scale"])
    design = np.column_stack([np.ones_like(y), *cols])[8:]
    coef, *_ = np.linalg.lstsq(design, y[8:], rcond=None)
    np.testing.assert_allclose(coef[1:], truth, rtol=0.1)


def test_realized_share_tracks_requested_share():
    for share in (0.2, 0.5, 0.8):
        prior = _linear_prior(effect_share=(share, share))
        realized = [g.recipe["realized"] for g in prior.sample(20, seed=2)]
        assert np.mean(realized) == pytest.approx(share, abs=0.05)


def test_null_groups_have_no_effect():
    prior = GroupPrior(
        mechanism=RegressionSpec(p_null=1.0, p_target_ar=0.0, n_covariates=(2, 4)),
        length=(64, 64),
        freq="D",
    )
    batch = prior.sample(10, seed=0, components=True)
    assert batch.diagnostics["n_null"] == 10
    for group in batch:
        recipe = group.recipe
        assert recipe["null"] and recipe["terms"] == []
        assert recipe["calibration"] == "null"
        assert not any(c.get("informative") for c in recipe["channels"])
        np.testing.assert_array_equal(
            group.values[group.target_index], group.components["base"]
        )


def test_covariate_free_groups():
    prior = GroupPrior(mechanism=RegressionSpec(n_covariates=(0, 0)), length=(32, 32))
    for g in prior.sample(4, seed=0):
        assert g.roles == ("target",) and g.recipe["null"]
        assert g.recipe["calibration"] == "null"


def test_roles_are_independent_of_informativeness():
    prior = GroupPrior(
        mechanism=RegressionSpec(n_covariates=(4, 6), p_known_future=0.3),
        length=(32, 32),
    )
    future = {True: [], False: []}
    target_positions = set()
    for group in prior.sample(600, seed=4):
        target_positions.add(group.target_index)
        for cov in group.recipe["channels"][1:]:
            future[cov["informative"]].append(cov["role"] == "future")
            assert group.roles[cov["channel"]] == cov["role"]
    for flags in future.values():
        se = np.sqrt(0.3 * 0.7 / len(flags))
        assert abs(np.mean(flags) - 0.3) < 5 * se
    assert len(target_positions) > 3


def test_multiplicative_preserves_zeros():
    spec = GeneratorSpec(generator=IntermittentDemandGenerator, positive=True)
    prior = GroupPrior(
        mechanism=RegressionSpec(
            target_pool=[spec], p_multiplicative=1.0, p_target_ar=0.0
        ),
        length=(128, 128),
        freq="D",
    )
    for group in prior.sample(10, seed=1, components=True):
        assert group.recipe["composition"] == "multiplicative"
        y = group.values[group.target_index]
        assert (y >= 0).all()
        np.testing.assert_array_equal(y == 0, group.components["base"] == 0)


def test_multiplicative_falls_back_to_additive():
    spec = GeneratorSpec(generator=EnergyLoadGenerator, positive=True)
    prior = GroupPrior(
        mechanism=RegressionSpec(
            target_pool=[spec],
            driver_pool=[WHITE_SPEC],
            p_null=0.0,
            edge_kinds=["linear"],
            p_multiplicative=1.0,
            log_effect_std=(60.0, 60.0),
            p_target_ar=0.0,
        ),
        length=(128, 128),
        freq="h",
    )
    batch = prior.sample(5, seed=0, components=True)
    assert batch.diagnostics["n_fallback"] == 5
    for g in batch:
        r = g.recipe
        assert r["composition"] == "additive" and r["fallback"] == "additive"
        assert r["requested_log_std"] is None and r["requested_share"] is not None
        np.testing.assert_allclose(
            g.values[g.target_index], g.components["base"] + g.components["effect"]
        )


def test_redraws_are_recorded_and_replayable():
    flaky = GeneratorSpec(generator=_Flaky)
    prior = GroupPrior(
        mechanism=RegressionSpec(target_pool=[flaky], driver_pool=[WHITE_SPEC]),
        length=(32, 32),
        max_redraws=10,
    )
    batch = prior.sample(20, seed=0)
    redrawn = [g for g in batch if g.recipe["attempt"] > 0]
    assert redrawn and batch.diagnostics["n_redrawn"] == len(redrawn)
    for g in redrawn:
        assert len(g.recipe["failures"]) == g.recipe["attempt"]
        np.testing.assert_array_equal(prior.replay(g.recipe).values, g.values)


def test_raising_draws_are_redrawn_and_reported():
    raising = GeneratorSpec(generator=_Raising)
    prior = GroupPrior(
        mechanism=RegressionSpec(target_pool=[raising], driver_pool=[WHITE_SPEC]),
        length=(32, 32),
        max_redraws=10,
    )
    failures = [f for g in prior.sample(10, seed=0) for f in g.recipe["failures"]]
    assert failures and all("_Raising: unlucky draw" in f for f in failures)


def test_persistent_failures_raise():
    prior = GroupPrior(
        mechanism=RegressionSpec(target_pool=[GeneratorSpec(generator=_Broken)]),
        length=(32, 32),
        max_redraws=2,
    )
    with pytest.raises(RuntimeError, match="3 attempts"):
        prior.sample(1, seed=0)


def test_constant_drivers_give_a_nominal_effect():
    const = GeneratorSpec(generator=_Constant)
    prior = GroupPrior(
        mechanism=RegressionSpec(driver_pool=[const], n_covariates=(2, 2), p_null=0.0),
        length=(32, 32),
    )
    group = prior.sample(1, seed=0, components=True)[0]
    assert group.recipe["attempt"] == 0
    assert group.recipe["calibration"] == "nominal_effect"
    np.testing.assert_allclose(group.components["effect"], 0.0, atol=1e-12)


def test_constant_target_gives_a_nominal_base():
    prior = GroupPrior(
        mechanism=RegressionSpec(
            target_pool=[GeneratorSpec(generator=_Constant)],
            driver_pool=[WHITE_SPEC],
            p_null=0.0,
            p_multiplicative=0.0,
            p_target_ar=0.0,
            effect_share=(0.5, 0.5),
        ),
        length=(64, 64),
    )
    group = prior.sample(1, seed=0)[0]
    assert group.recipe["calibration"] == "nominal_base"
    assert group.recipe["realized"] == pytest.approx(1.0)


def test_every_sampler_branch_is_reachable():
    prior = GroupPrior(
        mechanism=RegressionSpec(
            n_covariates=(3, 4),
            n_informative=(2, 3),
            p_null=0.0,
            p_interaction=1.0,
            p_adstock=1.0,
            p_persistence=1.0,
            p_target_ar=1.0,
        ),
        length=(64, 64),
    )
    spec = prior.mechanism
    for g in prior.sample(20, seed=0):
        r = g.recipe
        n_inf = sum(c.get("informative", False) for c in r["channels"])
        inter = [t for t in r["terms"] if t["partner"] is not None]
        assert len(inter) == 1
        t = inter[0]
        assert t["partner"] != t["driver"] and max(t["driver"], t["partner"]) <= n_inf
        if t["kind"] == "gate":
            assert -1.0 <= t["param"] <= 1.0
        else:
            assert t["param"] is None
        for term in r["terms"]:
            assert spec.adstock_decay[0] <= term["adstock"] <= spec.adstock_decay[1]
        assert spec.persistence_phi[0] <= r["persistence"] <= spec.persistence_phi[1]
        assert spec.target_ar_phi[0] <= r["target_ar"] <= spec.target_ar_phi[1]


@pytest.mark.parametrize(
    "kind",
    [
        "linear",
        "saturating",
        "quadratic",
        "signed_sqrt",
        "abs_sqrt",
        "threshold",
        "hinge",
    ],
)
def test_each_edge_kind_samples_and_replays(kind):
    prior = GroupPrior(
        mechanism=RegressionSpec(edge_kinds=[kind], p_null=0.0), length=(48, 48)
    )
    for g in prior.sample(3, seed=1):
        assert {t["kind"] for t in g.recipe["terms"] if t["partner"] is None} == {kind}
        np.testing.assert_array_equal(prior.replay(g.recipe).values, g.values)


def test_generator_noise_fallback_is_recorded_or_redrawn():
    params = {
        "seasonal_periods": [],
        "rbf_length_scales": [],
        "rational_quadratic_alphas": [],
        "linear_sigmas": [],
        "white_noise_levels": [],
        "include_constant": True,
        "jitter": 0.0,
        "max_kernels": 1,
    }
    spec = GeneratorSpec(generator=KernelSynthGenerator, params=params)
    mech = RegressionSpec(
        target_pool=[spec], driver_pool=[WHITE_SPEC], n_covariates=(1, 1)
    )
    batch = GroupPrior(mechanism=mech, length=(32, 32)).sample(3, seed=0)
    assert batch.diagnostics["n_generator_fallback"] == 3
    assert all(g.recipe["generator_fallbacks"] == [0] for g in batch)
    strict = GroupPrior(
        mechanism=mech, length=(32, 32), on_generator_fallback="redraw", max_redraws=1
    )
    with pytest.raises(RuntimeError, match="generator noise fallback"):
        strict.sample(1, seed=0)


def test_validation():
    with pytest.raises(ValidationError, match="burn_in"):
        GroupPrior(mechanism=RegressionSpec(max_lag=100), burn_in=64)
    with pytest.raises(ValidationError, match="partner"):
        Term(kind="product", driver=0, coef=1.0)
    with pytest.raises(ValidationError, match="takes no param"):
        Term(kind="linear", driver=0, coef=1.0, param=0.5)
    with pytest.raises(ValidationError, match="partner_lag"):
        Term(kind="linear", driver=0, coef=1.0, partner_lag=2)
    with pytest.raises(ValidationError, match="does not exist"):
        FixedRegressionSpec(
            drivers=[WHITE_SPEC], terms=[Term(kind="linear", driver=1, coef=1.0)]
        )
    with pytest.raises(ValidationError, match="n_informative"):
        RegressionSpec(n_informative=(0, 2))
    with pytest.raises(ValidationError, match="positive step"):
        GroupPrior(freq=0)
    with pytest.raises(ValidationError, match="invalid freq"):
        GroupPrior(freq="@@")
    with pytest.raises(ValidationError, match="start_datetime"):
        GroupPrior(start_datetime="not a date")
    with pytest.raises(ValueError, match="n_groups"):
        GroupPrior(length=(16, 16)).sample(-1, seed=0)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["pandas", "polars", "pyarrow"])
def test_to_long(engine):
    batch = GroupPrior(length=(16, 32), freq="D").sample(6, seed=0, start_group=10)
    df = batch.to_long(engine=engine)
    pdf = df if isinstance(df, pd.DataFrame) else df.to_pandas()
    assert list(pdf.columns) == ["group_id", "unique_id", "ds", "y", "role"]
    assert len(pdf) == sum(g.values.size for g in batch)
    assert set(pdf["group_id"]) == set(range(10, 16))
    first = batch[0]
    rows = pdf[pdf["unique_id"] == f"{first.group_id}_{first.target_index}"]
    np.testing.assert_array_equal(
        rows["y"].to_numpy(), first.values[first.target_index]
    )
    with pytest.raises(ValueError, match="distinct"):
        batch.to_long(id_col="y")


def test_integer_freq_and_empty_batches_share_dtypes():
    prior = GroupPrior(freq=7, length=(16, 16))
    g = prior.sample(1, seed=0)[0]
    np.testing.assert_array_equal(g.timestamps, np.arange(16) * 7)
    full = prior.sample(2, seed=0).to_long()
    empty = prior.sample(0, seed=0).to_long()
    assert len(empty) == 0 and empty["ds"].dtype == full["ds"].dtype
    dated = GroupPrior(length=(16, 16)).sample(0, seed=0).to_long()
    assert dated["ds"].dtype.kind == "M"


def test_to_padded():
    batch = GroupPrior(length=(20, 40)).sample(6, seed=0)
    out = batch.to_padded()
    b, c, t = out["values"].shape
    assert (b, c, t) == (
        6,
        max(g.values.shape[0] for g in batch),
        max(g.length for g in batch),
    )
    for i, g in enumerate(batch):
        k, length = g.values.shape
        np.testing.assert_allclose(
            out["values"][i, :k, t - length :], g.values, rtol=1e-6
        )
        assert out["observed"][i, :k, t - length :].all()
        assert not out["observed"][i, :, : t - length].any()
        assert not out["channel_mask"][i, k:].any()
        assert out["target_mask"][i].sum() == 1
        assert out["target_mask"][i, g.target_index]
        np.testing.assert_array_equal(
            out["known_future_mask"][i, :k], np.asarray(g.roles) == "future"
        )
    start = batch.to_padded(max_length=10, align="start")
    k0 = batch[0].values.shape[0]
    np.testing.assert_allclose(
        start["values"][0, :k0], batch[0].values[:, :10], rtol=1e-6
    )
    with pytest.raises(ValueError, match="channels"):
        batch.to_padded(max_channels=1)


def test_shuffle_covariates_fixed_length():
    prior = probe_suite()["sum/phi0.0/future/d2/iid"]
    batch = prior.sample(8, seed=0)
    shuffled = batch.shuffle_covariates(seed=1)
    assert shuffled.diagnostics["n_controls"] == 8
    for original, control in zip(batch, shuffled, strict=True):
        recipe = control.recipe
        assert recipe["control"] == "shuffled_covariates"
        assert recipe["source"] == original.recipe
        assert control.target_index == original.target_index
        np.testing.assert_array_equal(
            control.values[control.target_index],
            original.values[original.target_index],
        )
        donor = batch[recipe["covariates_from"]]
        assert donor.group_id != original.group_id
        for source in recipe["covariate_sources"]:
            np.testing.assert_array_equal(
                control.values[source["channel"]],
                donor.values[source["donor_channel"]],
            )
            assert (
                control.roles[source["channel"]] == donor.roles[source["donor_channel"]]
            )
        with pytest.raises(ValueError, match="cannot be replayed"):
            prior.replay(recipe)


def test_shuffle_covariates_variable_length():
    prior = GroupPrior(mechanism=RegressionSpec(n_covariates=(2, 2)), length=(32, 96))
    batch = prior.sample(12, seed=0)
    shuffled = batch.shuffle_covariates(seed=3)
    for original, control in zip(batch, shuffled, strict=True):
        donor = batch[control.recipe["covariates_from"]]
        length = control.length
        if "cropped_to" in control.recipe:
            assert length == donor.length < original.length
        else:
            assert length == original.length <= donor.length
        np.testing.assert_array_equal(
            control.values[control.target_index],
            original.values[original.target_index, :length],
        )
    with pytest.raises(ValueError, match="bucket"):
        GroupPrior(length=(16, 16)).sample(1, seed=0).shuffle_covariates(0)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def test_probe_suite_shapes():
    probes = probe_suite(context=24, horizon=12)
    assert len(probes) == 4 * 4 * 2 * 3 * 2
    group = probes["nonlinear/phi0.3/past/d4/smooth"].sample(1, seed=0)[0]
    assert group.values.shape == (1 + 3 + 4, 36)
    assert sorted(set(group.roles)) == ["past", "target"]
    assert {t["lag"] for t in group.recipe["terms"]} == {12}


def test_probe_suite_needs_context_for_past_only_lags():
    with pytest.raises(ValueError, match="context must be at least"):
        probe_suite(context=12, horizon=24)
    with pytest.raises(ValueError, match="positive"):
        probe_suite(context=0, horizon=0)
    group = probe_suite()["identity/phi0.0/past/d0/iid"].sample(1, seed=0)[0]
    y = group.values[group.target_index]
    x = group.values[group.covariate_indices[0]]
    np.testing.assert_allclose(y[24:], x[:24], atol=0.5)


def test_identity_probe_is_exact_up_to_noise():
    probe = probe_suite(noise_std=0.0)["identity/phi0.0/future/d0/iid"]
    group = probe.sample(1, seed=0)[0]
    np.testing.assert_allclose(
        group.values[group.target_index], group.values[group.covariate_indices[0]]
    )


def test_ar_probe_follows_recursion():
    probe = probe_suite(noise_std=0.0)["sum/phi0.6/past/d0/smooth"]
    group = probe.sample(1, seed=0)[0]
    y = group.values[group.target_index]
    x, w = (group.values[i] for i in group.covariate_indices)
    np.testing.assert_allclose(y[24:] - 0.6 * y[23:-1], (x + w)[:24], atol=1e-10)


def test_random_walk_drivers_stay_finite_over_long_windows():
    prior = GroupPrior(
        mechanism=RegressionSpec(
            driver_pool=[GeneratorSpec(generator=RandomWalkGenerator)],
            edge_kinds=["quadratic"],
        ),
        length=(2048, 2048),
    )
    assert all(np.isfinite(g.values).all() for g in prior.sample(10, seed=0))


# ---------------------------------------------------------------------------
# Robustness of the native boundary and of cached configuration
# ---------------------------------------------------------------------------


def test_native_draw_failures_are_per_request():
    from synforecast.generators import ETSGenerator

    gen = ETSGenerator(
        min_length=64, max_length=64, freq="h", seed=0, seasonal_type="add"
    )
    scalar, arrays = gen._get_batch_params()
    scalar = np.asarray(scalar, dtype=float).copy()
    scalar[scalar == 12] = 0  # zero seasonal period: the kernel would panic
    walk = RandomWalkGenerator(min_length=8, max_length=8, freq="h", seed=0)
    walk_scalar, _ = walk._get_batch_params()
    values, status, messages = rs.draw_series(
        np.array([3, 0], dtype=np.int32),
        [scalar, np.asarray(walk_scalar, dtype=float)],
        [[np.asarray(a, dtype=float) for a in arrays], []],
        np.array([0, 1], dtype=np.uint32),
        np.array([16, 16]),
        np.array([1, 2], dtype=np.uint64),
        1,
    )
    assert status.tolist() == [2, 0]
    assert messages[0] and not messages[1]
    assert np.isfinite(values[16:]).all()
    with pytest.raises(ValueError, match="allocate|overflow"):
        rs.draw_series(
            np.array([0], dtype=np.int32),
            [np.asarray(walk_scalar, dtype=float)],
            [[]],
            np.zeros(3, dtype=np.uint32),
            np.array([2**62] * 3),
            np.zeros(3, dtype=np.uint64),
            1,
        )


def test_spec_params_are_snapshotted():
    spec = GeneratorSpec(generator=RandomWalkGenerator, params={"start_value": 1.0})
    prior = GroupPrior(
        mechanism=RegressionSpec(target_pool=[spec], driver_pool=[spec]),
        length=(32, 32),
    )
    before = prior.sample(2, seed=0)
    fingerprint = prior.fingerprint
    spec.params["start_value"] = 9.0
    after = prior.sample(2, seed=0)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a.values, b.values)
    assert prior.fingerprint == fingerprint
    assert spec.describe()["params"] == {"start_value": 1.0}
    rebuilt = GroupPrior(
        mechanism=RegressionSpec(target_pool=[spec], driver_pool=[spec]),
        length=(32, 32),
    )
    assert rebuilt.fingerprint == fingerprint
    with pytest.raises(AttributeError):
        prior.mechanism.target_pool.append(spec)


def test_pickled_and_copied_priors_reuse_their_sources(monkeypatch):
    import copy
    import pickle

    import synforecast.groups as groups_mod

    prior = GroupPrior(length=(32, 64))
    reference = prior.sample(5, seed=1)
    calls = []
    original = groups_mod._resolve
    monkeypatch.setattr(
        groups_mod, "_resolve", lambda *a: calls.append(a) or original(*a)
    )
    for clone in (pickle.loads(pickle.dumps(prior)), copy.deepcopy(prior)):
        batch = clone.sample(5, seed=1)
        for a, b in zip(reference, batch, strict=True):
            np.testing.assert_array_equal(a.values, b.values)
        assert clone.fingerprint == prior.fingerprint
    assert calls == []


def test_nonfinite_configuration_is_rejected():
    with pytest.raises(ValidationError):
        Term(kind="threshold", driver=0, coef=1.0, param=float("nan"))
    with pytest.raises(ValidationError):
        Term(kind="linear", driver=0, coef=float("inf"))
    with pytest.raises(ValidationError, match="finite"):
        GeneratorSpec(generator=SARIMAGenerator, params={"mean": float("nan")})
    with pytest.raises(ValidationError):
        FixedRegressionSpec(
            drivers=[WHITE_SPEC],
            terms=[Term(kind="linear", driver=0, coef=1.0)],
            intercept=float("nan"),
        )


def test_fixed_fallbacks_index_recipe_channels():
    params = {
        "seasonal_periods": [],
        "rbf_length_scales": [],
        "rational_quadratic_alphas": [],
        "linear_sigmas": [],
        "white_noise_levels": [],
        "include_constant": True,
        "jitter": 0.0,
        "max_kernels": 1,
    }
    spec = GeneratorSpec(generator=KernelSynthGenerator, params=params)
    prior = _fixed([Term(kind="linear", driver=0, coef=1.0)], [spec])
    group = prior.sample(1, seed=0)[0]
    assert group.recipe["generator_fallbacks"] == [1]
    entry = group.recipe["channels"][1]
    assert entry["spec"]["generator"].endswith("KernelSynthGenerator")


def test_to_padded_validates_arguments():
    batch = GroupPrior(length=(20, 40)).sample(2, seed=0)
    with pytest.raises(ValueError, match="max_length"):
        batch.to_padded(max_length=0)
    with pytest.raises(ValueError, match="align"):
        batch.to_padded(align="middle")


@pytest.mark.parametrize(
    ("generator", "params"),
    [
        ("SARIMAGenerator", {}),
        ("SARIMAGenerator", {"p": 3, "P": 0, "Q": 0}),
        ("SARIMAGenerator", {"p": 2, "ar_params": [0.3, 0.2]}),
        ("ETSGenerator", {"seasonal_type": "add"}),
        ("ETSGenerator", {"seasonal_type": "mul", "error_type": "mul"}),
        ("GARCHGenerator", {}),
        ("INARGenerator", {"p": 2}),
        ("StateSpaceGenerator", {"state_dim": 3}),
    ],
)
def test_native_parameter_sampling_matches_constructors(generator, params):
    """Natively re-sampled parameters follow the Python constructors'
    distributions, and every other parameter stays the template's."""
    from synforecast import generators
    from synforecast.groups import _resolve

    spec = GeneratorSpec(generator=getattr(generators, generator), params=params)
    source = _resolve(spec, "h")
    assert source.mode == "randomized"
    n = 1000
    python = [spec.build(64, "h", s)._get_batch_params() for s in range(n)]
    native = [
        rs.sample_params(
            source.gen_type, source.scalar, list(source.arrays), source.randomizer, s
        )
        for s in range(n)
    ]
    for scalar, arrays in native:
        assert len(arrays) == len(python[0][1])
        assert np.isfinite(scalar).all() and all(np.isfinite(a).all() for a in arrays)
    for k in range(len(python[0][1])):
        size = len(python[0][1][k])
        if size == 0:
            continue
        for i in {0, size - 1}:
            a = np.array([np.asarray(p[1][k])[i] for p in python])
            b = np.array([np.asarray(q[1][k])[i] for q in native])
            if a.std() == 0:
                np.testing.assert_allclose(b, a)
                continue
            se = np.sqrt((a.var() + b.var()) / n)
            assert abs(a.mean() - b.mean()) < 5 * se
            assert b.std() == pytest.approx(a.std(), rel=0.12)
    scalars_py = np.array([np.asarray(p[0]) for p in python])
    scalars_nat = np.array([np.asarray(q[0]) for q in native])
    np.testing.assert_allclose(
        scalars_nat.mean(axis=0), scalars_py.mean(axis=0), rtol=0.05
    )


def test_seed_dependent_generators_without_a_sampler_are_rebuilt(monkeypatch):
    import synforecast.groups as groups_mod

    monkeypatch.setattr(groups_mod, "_randomizer", lambda _gen: None)
    spec = GeneratorSpec(generator=SARIMAGenerator)
    prior = GroupPrior(
        mechanism=RegressionSpec(target_pool=[spec], driver_pool=[spec]),
        length=(48, 48),
    )
    assert prior._source(spec).mode == "per_draw"
    batch = prior.sample(3, seed=0)
    for g in batch:
        np.testing.assert_array_equal(prior.replay(g.recipe).values, g.values)


def test_randomized_draws_vary_and_replay():
    spec = GeneratorSpec(generator=SARIMAGenerator)
    prior = GroupPrior(
        mechanism=RegressionSpec(target_pool=[spec], driver_pool=[spec]),
        length=(64, 64),
    )
    source = prior._source(spec)
    params = [
        rs.sample_params(
            source.gen_type, source.scalar, list(source.arrays), source.randomizer, s
        )[1][0]
        for s in (1, 2)
    ]
    assert not np.allclose(params[0], params[1])
    for g in prior.sample(4, seed=3):
        np.testing.assert_array_equal(prior.replay(g.recipe).values, g.values)
