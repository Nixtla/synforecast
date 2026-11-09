"""Tests for grouped covariate tasks (synforecast.groups)."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from synforecast import (
    FixedRegressionSpec,
    GeneratorSpec,
    GroupPrior,
    RegressionSpec,
    RoleSpec,
    Term,
    probe_suite,
)
from synforecast.base import BaseGenerator
from synforecast.generators import (
    IntermittentDemandGenerator,
    RandomWalkGenerator,
    SARIMAGenerator,
)
from synforecast.groups import _compose, _raw_effect

WHITE = {"p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "noise_std": 1.0}
WHITE_SPEC = GeneratorSpec(generator=SARIMAGenerator, params=WHITE)


class _ConstantGenerator(BaseGenerator):
    """Constant series: a legitimate channel with zero variance."""

    def generate_single_series(self, length: int) -> np.ndarray:
        return np.full(length, 3.0)


class _BrokenGenerator(BaseGenerator):
    """Always non-finite output."""

    def generate_single_series(self, length: int) -> np.ndarray:
        return np.full(length, np.inf)


def _linear_prior(**overrides) -> GroupPrior:
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
        mechanism=RegressionSpec(**spec), length=(4000, 4000), freq="D", burn_in=64
    )


def _rebuild(prior: GroupPrior, group) -> np.ndarray:
    """Recompute the target from the recipe alone (latent paths via seeds)."""
    recipe = group.recipe
    b, total = recipe["burn_in"], recipe["burn_in"] + recipe["length"]
    mech = prior.mechanism
    target_spec = next(
        s for s in mech.target_pool if s.describe() == recipe["target"]["spec"]
    )
    base = target_spec.draw(total, prior.freq, recipe["target"]["seed"])
    z = []
    for cov in recipe["covariates"]:
        spec = next(s for s in mech.driver_pool if s.describe() == cov["spec"])
        raw = spec.draw(total, prior.freq, cov["seed"])
        np.testing.assert_allclose(raw[b:], group.values[cov["channel"]])
        z.append((raw - cov["calibration_mean"]) / cov["calibration_std"])
    z = np.array(z) if z else np.empty((0, total))
    y, _, info = _compose(
        base,
        z,
        [Term(**t) for t in recipe["terms"]],
        recipe["adstock"],
        recipe["persistence"],
        recipe["target_ar"],
        b,
        recipe["composition"],
        recipe["requested_share"] or 0.5,
        recipe["requested_log_std"] or 0.1,
    )
    assert info["scale"] == pytest.approx(recipe["effect_scale"])
    return y[b:]


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_compose_is_prefix_invariant():
    """Changing inputs after time t never changes the target at or before t."""
    rng = np.random.default_rng(0)
    total, b = 400, 64
    base = rng.normal(size=total)
    z = rng.normal(size=(3, total))
    terms = [
        Term(kind="linear", driver=0, coef=0.7, lag=3),
        Term(kind="saturating", driver=1, coef=-0.5, param=1.5),
        Term(kind="gate", driver=0, partner=2, coef=0.4, lag=1, param=0.2),
        Term(kind="quadratic", driver=2, coef=0.3, lag=10),
    ]
    adstock = [0.6, None, 0.4, None]
    for composition in ("additive", "multiplicative"):
        args = (terms, adstock, 0.5, 0.7, b, composition, 0.6, 0.3)
        y, _, _ = _compose(np.abs(base) + 1, z, *args)
        cut = 200
        base2, z2 = np.abs(base) + 1, z.copy()
        base2[cut:] += 50.0
        z2[:, cut:] = rng.normal(size=(3, total - cut)) * 10
        y2, _, _ = _compose(base2, z2, *args)
        np.testing.assert_allclose(y[:cut], y2[:cut])
        assert not np.allclose(y[cut:], y2[cut:])


def test_lags_do_not_wrap():
    z = np.arange(10.0)[None, :]
    effect = _raw_effect(
        z, [Term(kind="linear", driver=0, coef=1.0, lag=3)], [None], 0.0, 10
    )
    np.testing.assert_allclose(effect, [0, 0, 0, 0, 1, 2, 3, 4, 5, 6])


# ---------------------------------------------------------------------------
# Recipes and reproducibility
# ---------------------------------------------------------------------------


def test_recipe_reconstructs_target_exactly():
    prior = GroupPrior(length=(64, 256), freq="D", burn_in=96)
    batch = prior.sample(40, seed=3)
    assert {g.recipe["composition"] for g in batch} == {"additive", "multiplicative"}
    for group in batch:
        np.testing.assert_allclose(
            _rebuild(prior, group), group.values[group.target_index], rtol=1e-12
        )


def test_groups_are_reproducible_individually():
    prior = GroupPrior(length=(32, 64), freq="h")
    batch = prior.sample(5, seed=11)
    again = prior.sample(5, seed=11)
    single = prior.sample(1, seed=11, start_group=3)[0]
    for a, b in zip(batch, again, strict=True):
        np.testing.assert_array_equal(a.values, b.values)
        assert a.roles == b.roles
        assert a.recipe == b.recipe
    np.testing.assert_array_equal(single.values, batch[3].values)
    replayed = prior.replay(batch[2].recipe)
    np.testing.assert_array_equal(replayed.values, batch[2].values)
    assert not np.array_equal(batch[0].values[0], batch[1].values[0])


def test_replay_rejects_other_configuration():
    recipe = GroupPrior(length=(32, 64)).sample(1, seed=0)[0].recipe
    with pytest.raises(ValueError, match="different prior"):
        GroupPrior(length=(32, 65)).replay(recipe)
    with pytest.raises(ValueError, match="version"):
        GroupPrior(length=(32, 64)).replay({**recipe, "version": -1})


def test_constructor_randomized_generators_draw_fresh_parameters():
    """SARIMA samples coefficients in its constructor; each draw rebuilds it."""
    spec = GeneratorSpec(generator=SARIMAGenerator, params={"p": 2, "q": 1})
    a = spec.draw(200, "D", seed=1)
    b = spec.draw(200, "D", seed=2)
    np.testing.assert_array_equal(a, spec.draw(200, "D", seed=1))
    assert not np.allclose(a, b)


def test_generator_spec_rejects_reserved_params():
    with pytest.raises(ValidationError, match="prior owns"):
        GeneratorSpec(generator=SARIMAGenerator, params={"seed": 1})
    with pytest.raises(ValidationError):
        GeneratorSpec(generator=int)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_linear_coefficients_are_recovered():
    group = _linear_prior().sample(1, seed=5)[0]
    recipe = group.recipe
    y = group.values[group.target_index]
    cols, truth = [], []
    for term in recipe["terms"]:
        cov = recipe["covariates"][term["driver"]]
        assert cov["informative"]
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
        realized = [g.recipe["realized"] for g in prior.sample(5, seed=2)]
        assert np.mean(realized) == pytest.approx(share, abs=0.08)


def test_null_groups_have_no_effect():
    prior = GroupPrior(
        mechanism=RegressionSpec(p_null=1.0, p_target_ar=0.0, n_covariates=(2, 4)),
        length=(64, 64),
        freq="D",
        burn_in=96,
    )
    for group in prior.sample(10, seed=0):
        recipe = group.recipe
        assert recipe["null"] and recipe["terms"] == []
        assert not any(c["informative"] for c in recipe["covariates"])
        assert len(group.covariate_indices) >= 2
        target_spec = GeneratorSpec(
            **{
                **recipe["target"]["spec"],
                "generator": _generator_class(recipe["target"]["spec"]["generator"]),
            }
        )
        base = target_spec.draw(160, "D", recipe["target"]["seed"])
        np.testing.assert_allclose(group.values[group.target_index], base[96:])


def _generator_class(path: str):
    import importlib

    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def test_roles_are_independent_of_informativeness():
    prior = GroupPrior(
        mechanism=RegressionSpec(n_covariates=(4, 6)),
        roles=RoleSpec(p_known_future=0.3),
        length=(32, 32),
        burn_in=96,
    )
    future = {True: [], False: []}
    target_positions = set()
    for group in prior.sample(150, seed=4):
        target_positions.add(group.target_index)
        for cov in group.recipe["covariates"]:
            future[cov["informative"]].append(cov["role"] == "future")
            assert group.roles[cov["channel"]] == cov["role"]
    assert np.mean(future[True]) == pytest.approx(0.3, abs=0.07)
    assert np.mean(future[False]) == pytest.approx(0.3, abs=0.07)
    assert len(target_positions) > 3


def test_multiplicative_preserves_zeros():
    spec = GeneratorSpec(generator=IntermittentDemandGenerator, positive=True)
    prior = GroupPrior(
        mechanism=RegressionSpec(
            target_pool=[spec], p_multiplicative=1.0, p_target_ar=0.0
        ),
        length=(128, 128),
        freq="D",
        burn_in=96,
    )
    for group in prior.sample(10, seed=1):
        assert group.recipe["composition"] == "multiplicative"
        y = group.values[group.target_index]
        base = spec.draw(224, "D", group.recipe["target"]["seed"])[96:]
        assert (y >= 0).all()
        np.testing.assert_array_equal(y == 0, base == 0)


def test_constant_channels_are_allowed():
    const = GeneratorSpec(generator=_ConstantGenerator)
    prior = GroupPrior(
        mechanism=RegressionSpec(driver_pool=[const], n_covariates=(2, 2)),
        length=(32, 32),
        burn_in=96,
    )
    group = prior.sample(1, seed=0)[0]
    assert group.recipe["attempt"] == 0
    assert group.recipe["calibration"] in {"nominal_effect", "null"}


def test_persistent_failures_raise():
    broken = GeneratorSpec(generator=_BrokenGenerator)
    prior = GroupPrior(
        mechanism=RegressionSpec(target_pool=[broken]),
        length=(32, 32),
        burn_in=96,
        max_redraws=2,
    )
    with pytest.raises(RuntimeError, match="3 attempts"):
        prior.sample(1, seed=0)


def test_validation():
    with pytest.raises(ValidationError, match="burn_in"):
        GroupPrior(mechanism=RegressionSpec(max_lag=100), burn_in=64)
    with pytest.raises(ValidationError, match="partner"):
        Term(kind="product", driver=0, coef=1.0)
    with pytest.raises(ValidationError, match="does not exist"):
        FixedRegressionSpec(
            drivers=[WHITE_SPEC], terms=[Term(kind="linear", driver=1, coef=1.0)]
        )
    with pytest.raises(ValidationError, match="n_informative"):
        RegressionSpec(n_informative=(0, 2))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_to_long(engine):
    batch = GroupPrior(length=(16, 32), freq="D").sample(6, seed=0, start_group=10)
    df = batch.to_long(engine=engine)
    pdf = df if isinstance(df, pd.DataFrame) else df.to_pandas()
    assert list(pdf.columns) == ["group_id", "unique_id", "ds", "y", "role"]
    assert len(pdf) == sum(g.values.size for g in batch)
    assert set(pdf["group_id"]) == set(range(10, 16))
    for group in batch:
        rows = pdf[(pdf["group_id"] == group.group_id) & (pdf["role"] == "target")]
        assert len(rows) == group.length
    first = batch[0]
    rows = pdf[pdf["unique_id"] == f"{first.group_id}_{first.target_index}"]
    np.testing.assert_array_equal(
        rows["y"].to_numpy(), first.values[first.target_index]
    )


def test_shuffle_covariates():
    prior = probe_suite()["sum/phi0.0/future/d2/iid"]
    batch = prior.sample(8, seed=0)
    shuffled = batch.shuffle_covariates(seed=1)
    assert shuffled.diagnostics["n_controls"] == 8
    for original, control in zip(batch, shuffled, strict=True):
        recipe = control.recipe
        assert recipe["control"] == "shuffled_covariates"
        assert recipe["source"] == original.recipe
        # Targets keep their position and values
        assert control.target_index == original.target_index
        np.testing.assert_array_equal(
            control.values[control.target_index], original.values[original.target_index]
        )
        donor = batch[recipe["covariates_from"]]
        assert donor.group_id != original.group_id
        assert [s["channel"] for s in recipe["covariate_sources"]] == (
            original.covariate_indices
        )
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
    lone = GroupPrior(length=(16, 64)).sample(3, seed=0)
    if len({(g.length, len(g.covariate_indices)) for g in lone}) == 3:
        with pytest.raises(ValueError, match="bucket"):
            lone.shuffle_covariates(seed=0)


def test_numpy_params_are_normalized():
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
    prior = GroupPrior(
        mechanism=RegressionSpec(target_pool=[spec], driver_pool=[spec]),
        length=(32, 32),
        burn_in=96,
    )
    group = prior.sample(1, seed=0)[0]
    assert prior.replay(group.recipe).values.tolist() == group.values.tolist()
    with pytest.raises(ValidationError, match="JSON-serializable"):
        GeneratorSpec(generator=SARIMAGenerator, params={"mean": object()})


def test_generator_identity_includes_module():
    def make(module: str) -> type[BaseGenerator]:
        cls = type("SameName", (_ConstantGenerator,), {"__module__": module})
        return cls

    a = GeneratorSpec(generator=make("pkg_a"))
    b = GeneratorSpec(generator=make("pkg_b"))
    assert a.describe()["generator"] != b.describe()["generator"]
    prior_a = GroupPrior(
        mechanism=RegressionSpec(target_pool=[a], driver_pool=[a]),
        length=(16, 16),
        burn_in=96,
    )
    prior_b = GroupPrior(
        mechanism=RegressionSpec(target_pool=[b], driver_pool=[b]),
        length=(16, 16),
        burn_in=96,
    )
    assert prior_a.fingerprint != prior_b.fingerprint
    with pytest.raises(ValueError, match="different prior"):
        prior_b.replay(prior_a.sample(1, seed=0)[0].recipe)


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
    probes = probe_suite(context=24, horizon=24)
    group = probes["identity/phi0.0/past/d0/iid"].sample(1, seed=0)[0]
    y, x = group.values[group.target_index], group.values[group.covariate_indices[0]]
    # Every forecast lead reads a covariate value inside the context window
    np.testing.assert_allclose(y[24:], x[:24], atol=0.5)


def test_identity_probe_is_exact_up_to_noise():
    group = probe_suite(noise_std=0.0)["identity/phi0.0/future/d0/iid"].sample(
        1, seed=0
    )[0]
    x = group.values[group.covariate_indices[0]]
    np.testing.assert_allclose(group.values[group.target_index], x)


def test_ar_probe_follows_recursion():
    prior = probe_suite(noise_std=0.0)["sum/phi0.6/past/d0/smooth"]
    group = prior.sample(1, seed=0)[0]
    y = group.values[group.target_index]
    # y[t] - 0.6 y[t-1] = x[t-24] + w[t-24] for t >= 24 inside the window
    x, w = (group.values[i] for i in group.covariate_indices)
    np.testing.assert_allclose(y[24:] - 0.6 * y[23:-1], (x + w)[:24], atol=1e-10)


def test_random_walk_drivers_stay_finite_over_long_windows():
    prior = GroupPrior(
        mechanism=RegressionSpec(
            driver_pool=[GeneratorSpec(generator=RandomWalkGenerator)],
            edge_kinds=["quadratic"],
        ),
        length=(2048, 2048),
        burn_in=96,
    )
    batch = prior.sample(10, seed=0)
    assert all(np.isfinite(g.values).all() for g in batch)
