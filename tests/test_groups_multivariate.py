"""Tests for the multivariate group mechanisms: CoupledSpec and VARXSpec."""

import json
from typing import ClassVar

import numpy as np
import pytest
from pydantic import ValidationError

from synforecast import (
    CoupledSpec,
    GeneratorSpec,
    GroupPrior,
    VARXSpec,
    VARXTruth,
)
from synforecast.base import BaseGenerator
from synforecast.generators import (
    IntermittentDemandGenerator,
    SARIMAGenerator,
    TSIGenerator,
)
from tests.helpers import assert_std
from tests.test_tsi import pool_metrics

STATIONARY = GeneratorSpec(
    generator=TSIGenerator,
    params={
        "trend_types": ["none"],
        "n_seasonal_range": (0, 0),
        "irregular_types": ["gaussian"],
    },
)


class _Bump(BaseGenerator):
    """Gaussian noise, plus 50 from ``bump_at`` on (a class switch)."""

    bump_at: ClassVar[int | None] = None

    def generate_single_series(self, length: int) -> np.ndarray:
        x = self.rng.normal(size=length)
        if type(self).bump_at is not None:
            x[type(self).bump_at :] += 50.0
        return x


def _coupled(**overrides) -> GroupPrior:
    spec = {"base_pool": [STATIONARY], "n_channels": (4, 4)}
    spec.update(overrides.pop("spec", {}))
    kwargs = {"length": (512, 512), "freq": "h", "burn_in": 128}
    kwargs.update(overrides)
    return GroupPrior(mechanism=CoupledSpec(**spec), **kwargs)


def _latent_order(group) -> np.ndarray:
    """Channels in the order the recipe's coupling indices refer to."""
    return np.stack([group.values[c["channel"]] for c in group.recipe["channels"]])


def _mean_abs_offdiag_corr(x: np.ndarray) -> float:
    k = x.shape[0]
    corr = np.corrcoef(x)
    return float(np.abs(corr[np.triu_indices(k, 1)]).mean())


# ---------------------------------------------------------------------------
# CoupledSpec
# ---------------------------------------------------------------------------


class TestCoupled:
    def test_all_targets_by_default(self) -> None:
        group = _coupled().sample(1, seed=0)[0]
        assert group.roles == ("target",) * 4
        assert group.target_indices == [0, 1, 2, 3]
        with pytest.raises(ValueError, match="target_indices"):
            _ = group.target_index

    @pytest.mark.parametrize(("p", "role"), [(1.0, "future"), (0.0, "past")])
    def test_one_target_roles(self, p: float, role: str) -> None:
        prior = _coupled(
            spec={"targets": "one", "n_channels": (5, 5), "p_known_future": p}
        )
        for group in prior.sample(5, seed=1):
            assert sorted(group.roles) == sorted([role] * 4 + ["target"])

    def test_one_target_role_rate_and_positions(self) -> None:
        prior = _coupled(
            spec={"targets": "one", "n_channels": (5, 5), "p_known_future": 0.3},
            length=(32, 32),
        )
        batch = prior.sample(300, seed=2)
        future = [r == "future" for g in batch for r in g.roles if r != "target"]
        se = np.sqrt(0.3 * 0.7 / len(future))
        assert abs(np.mean(future) - 0.3) < 5 * se
        assert {g.target_index for g in batch} == set(range(5))
        for g in batch:
            for c in g.recipe["channels"]:
                assert g.roles[c["channel"]] == c["role"]

    def test_reproducible_and_replayable(self) -> None:
        prior = _coupled(length=(64, 128))
        batch = prior.sample(4, seed=2)
        again = prior.sample(4, seed=2)
        for a, b in zip(batch, again, strict=True):
            np.testing.assert_array_equal(a.values, b.values)
        loaded = json.loads(json.dumps(batch[1].recipe, allow_nan=False))
        np.testing.assert_array_equal(prior.replay(loaded).values, batch[1].values)
        other = prior.sample(1, seed=3)[0]
        assert not np.allclose(other.values, batch[0].values[:, : other.length])

    def test_single_channel(self) -> None:
        group = _coupled(spec={"n_channels": (1, 1)}).sample(1, seed=0)[0]
        assert group.values.shape == (1, 512)
        assert group.recipe["coupling"] == {}

    def test_wraps_constructor_randomized_generators(self) -> None:
        spec = GeneratorSpec(generator=SARIMAGenerator, params={"p": 2, "q": 1})
        prior = _coupled(spec={"base_pool": [spec], "couplings": ["leadlag"]})
        assert np.isfinite(prior.sample(1, seed=0)[0].values).all()

    def test_default_pool_output_guards(self) -> None:
        prior = GroupPrior(mechanism=CoupledSpec(), length=(256, 256))
        for group in prior.sample(30, seed=0):
            assert np.isfinite(group.values).all()
            assert np.abs(group.values).max() < 1e8

    def test_positive_channels_keep_their_zeros(self) -> None:
        spec = GeneratorSpec(generator=IntermittentDemandGenerator, positive=True)
        prior = _coupled(spec={"base_pool": [spec]}, freq="D")
        for group in prior.sample(5, seed=0):
            assert group.recipe["space"] == "log"
            assert (group.values >= 0).all()
            r = group.recipe
            total = r["burn_in"] + r["length"]
            for c in r["channels"]:
                raw = prior.draw_channel(spec, total, c["seed"])[r["burn_in"] :]
                np.testing.assert_array_equal(group.values[c["channel"]] == 0, raw == 0)

    def test_lagged_target_copies_are_labelled(self) -> None:
        prior = _coupled(
            spec={
                "targets": "one",
                "n_channels": (2, 2),
                "couplings": ["leadlag"],
                "p_leadlag": 1.0,
            },
            length=(64, 64),
        )
        seen = set()
        for group in prior.sample(20, seed=0):
            channels = group.recipe["channels"]
            target = next(i for i, c in enumerate(channels) if c["role"] == "target")
            for j, c in enumerate(channels):
                if j == target:
                    assert (
                        "derived_from_target" not in c or not c["derived_from_target"]
                    )
                    continue
                assert c["derived_from_target"] == (c["leadlag_root"] == target)
                seen.add(c["derived_from_target"])
        assert seen == {True, False}

    def test_shuffle_keeps_every_target(self) -> None:
        prior = _coupled(spec={"targets": "one"})
        batch = prior.sample(6, seed=0)
        shuffled = batch.shuffle_covariates(seed=0)
        for original, control in zip(batch, shuffled, strict=True):
            np.testing.assert_array_equal(
                control.values[control.target_index],
                original.values[original.target_index],
            )

    def test_validation(self) -> None:
        with pytest.raises(ValidationError):
            CoupledSpec(couplings=["unknown"])
        with pytest.raises(ValidationError, match="couplings"):
            CoupledSpec(couplings=[])
        for bad in [(0.5, 1.0), (0.5, 0.2)]:
            with pytest.raises(ValidationError, match="mixing_strength"):
                CoupledSpec(mixing_strength=bad)
        for bad in [(0, 5), (5, 2)]:
            with pytest.raises(ValidationError, match="lag_range"):
                CoupledSpec(lag_range=bad)
        for bad in [(0.0, 0.1), (0.2, 0.1)]:
            with pytest.raises(ValidationError, match="noise_scale"):
                CoupledSpec(noise_scale=bad)
        with pytest.raises(ValidationError, match="n_channels"):
            CoupledSpec(n_channels=(0, 2))
        with pytest.raises(ValidationError, match="burn_in"):
            GroupPrior(mechanism=CoupledSpec(lag_range=(1, 100)), burn_in=64)
        chained = CoupledSpec(n_channels=(6, 6), lag_range=(24, 24))
        assert chained.largest_lag == 120
        with pytest.raises(ValidationError, match="burn_in"):
            GroupPrior(mechanism=chained, burn_in=100)
        GroupPrior(mechanism=chained, burn_in=121)


@pytest.mark.parametrize("couplings", [["mixing"], ["leadlag"], ["mixing", "leadlag"]])
def test_coupled_groups_are_prefix_invariant(couplings, monkeypatch) -> None:
    prior = GroupPrior(
        mechanism=CoupledSpec(
            base_pool=[GeneratorSpec(generator=_Bump)],
            n_channels=(3, 3),
            couplings=couplings,
            p_leadlag=1.0,
            lag_range=(2, 5),
        ),
        length=(200, 200),
        burn_in=64,
    )
    monkeypatch.setattr(_Bump, "bump_at", None)
    plain = prior.sample(3, seed=4)
    monkeypatch.setattr(_Bump, "bump_at", 64 + 120)
    bumped = prior.sample(3, seed=4)
    for a, b in zip(plain, bumped, strict=True):
        np.testing.assert_array_equal(a.values[:, :120], b.values[:, :120])
        assert not np.allclose(a.values[:, 120:], b.values[:, 120:])


@pytest.mark.stats
class TestCoupledDependence:
    def test_mixing_correlates_channels_above_baseline(self) -> None:
        coupled = _coupled(
            spec={"couplings": ["mixing"], "mixing_strength": (0.5, 0.9)}
        )
        mixed = np.mean(
            [_mean_abs_offdiag_corr(g.values) for g in coupled.sample(10, seed=0)]
        )
        rng = np.random.default_rng(1)
        baseline = np.mean(
            [
                _mean_abs_offdiag_corr(
                    np.stack(
                        [
                            coupled.draw_channel(
                                STATIONARY, 512, int(rng.integers(2**31))
                            )
                            for _ in range(4)
                        ]
                    )
                )
                for _ in range(10)
            ]
        )
        assert mixed > 0.15, f"coupled mean |corr| {mixed:.3f}"
        assert mixed > 4.0 * baseline, f"{mixed:.3f} vs baseline {baseline:.3f}"

    def test_leadlag_correlation_peaks_at_sampled_lag(self) -> None:
        lag_corrs, zero_corrs = [], []
        for group in _coupled(spec={"couplings": ["leadlag"]}).sample(10, seed=3):
            x = _latent_order(group)
            pairs = group.recipe["coupling"]["leadlag"]
            assert len(pairs) >= 1
            for pair in pairs:
                src, dst, lag = pair["src"], pair["dst"], pair["lag"]
                zero_corrs.append(abs(np.corrcoef(x[src], x[dst])[0, 1]))
                lag_corrs.append(abs(np.corrcoef(x[src, :-lag], x[dst, lag:])[0, 1]))
        assert min(lag_corrs) > 0.8, f"min lag corr {min(lag_corrs):.3f}"
        assert np.median(lag_corrs) - np.median(zero_corrs) > 0.5

    def test_leadlag_channel_follows_its_source(self) -> None:
        """Without noise a lead-lag channel is an affine copy of its source
        ``lag`` steps earlier: the destination lags, it never leads."""
        prior = _coupled(
            spec={
                "couplings": ["leadlag"],
                "p_leadlag": 1.0,
                "noise_scale": (1e-9, 1e-9),
            }
        )
        group = prior.sample(1, seed=4)[0]
        x = _latent_order(group)
        for pair in group.recipe["coupling"]["leadlag"]:
            src, dst, lag = pair["src"], pair["dst"], pair["lag"]
            corr = np.corrcoef(x[src, :-lag], x[dst, lag:])[0, 1]
            assert corr == pytest.approx(pair["sign"], abs=1e-6)

    def test_chained_leadlag_stays_inside_the_burn_in(self) -> None:
        """At the chain bound, no emitted channel starts with the constant
        padding that a too-short burn-in would leak."""
        prior = GroupPrior(
            mechanism=CoupledSpec(
                base_pool=[STATIONARY],
                n_channels=(6, 6),
                couplings=["leadlag"],
                p_leadlag=1.0,
                lag_range=(24, 24),
                noise_scale=(1e-9, 1e-9),
            ),
            length=(200, 200),
            burn_in=121,
        )
        for group in prior.sample(3, seed=0):
            for channel in group.values:
                assert np.abs(np.diff(channel[:120])).min() > 1e-6

    def test_coupled_tsi_pool_diversity(self) -> None:
        """Coupled TSI channels keep the synthetic-pool diversity targets."""
        prior = GroupPrior(
            mechanism=CoupledSpec(
                base_pool=[GeneratorSpec(generator=TSIGenerator)], n_channels=(5, 5)
            ),
            length=(512, 512),
            freq="D",
        )
        rows = np.array(
            [
                pool_metrics(channel)
                for g in prior.sample(60, seed=0)
                for channel in g.values
            ]
        )
        assert np.median(rows[:, 0]) >= 1.0
        assert np.median(np.abs(rows[:, 1])) <= 0.5
        assert np.median(rows[:, 2]) >= 0.8


# ---------------------------------------------------------------------------
# VARXSpec
# ---------------------------------------------------------------------------


def _varx(**overrides) -> GroupPrior:
    spec = {
        "n_targets": (4, 4),
        "n_shared": 1,
        "n_future": 2,
        "n_past": 1,
        "exog_cross_scale": 0.5,
        "n_clusters": 2,
    }
    spec.update(overrides.pop("spec", {}))
    kwargs = {"length": (200, 200), "freq": "D", "burn_in": 100}
    kwargs.update(overrides)
    return GroupPrior(mechanism=VARXSpec(**spec), **kwargs)


def _residuals(truth: VARXTruth, parts: dict[str, np.ndarray]) -> np.ndarray:
    """Targets minus everything the truth explains, from the second period."""
    y = parts["y"]
    fitted = (
        truth.intercept
        + y[:-1] @ truth.lag_matrices[0].T
        + parts["shared"][1:] @ truth.shared_effects.T
        + np.einsum("kij,tjk->ti", truth.future_effects, parts["future"][1:])
        + np.einsum("kij,tjk->ti", truth.past_effects, parts["past"][:-1])
    )
    return y[1:] - fitted


class TestVARX:
    def test_layout_and_roles(self) -> None:
        group = _varx().sample(1, seed=0)[0]
        assert group.values.shape == (4 + 1 + 4 * 2 + 4 * 1, 200)
        assert group.roles.count("target") == 4
        assert group.roles.count("future") == 1 + 8
        assert group.roles.count("past") == 4
        for c in group.recipe["channels"]:
            assert group.roles[c["channel"]] == c["role"]
        parts = VARXTruth.split(group)
        assert parts["y"].shape == (200, 4)
        assert parts["future"].shape == (200, 4, 2)
        assert parts["past"].shape == (200, 4, 1)

    def test_truth_shapes_and_clusters(self) -> None:
        group = _varx(spec={"n_targets": (6, 6)}).sample(1, seed=1)[0]
        truth = VARXTruth.from_recipe(json.loads(json.dumps(group.recipe)))
        assert truth.lag_matrices.shape == (1, 6, 6)
        assert truth.shared_effects.shape == (6, 1)
        assert truth.future_effects.shape == (2, 6, 6)
        assert truth.past_effects.shape == (1, 6, 6)
        np.testing.assert_array_equal(truth.clusters, [0, 0, 0, 1, 1, 1])

    def test_process_follows_the_truth_exactly_without_noise(self) -> None:
        prior = _varx(
            spec={
                "n_targets": (5, 5),
                "innovation_covariance": (1e-24 * np.eye(5)).tolist(),
            }
        )
        group = prior.sample(1, seed=2)[0]
        truth = VARXTruth.from_recipe(group.recipe)
        resid = _residuals(truth, VARXTruth.split(group))
        np.testing.assert_allclose(resid, 0.0, atol=1e-9)

    def test_user_coefficients_are_kept(self) -> None:
        a = [[[0.5, 0.2], [0.0, 0.4]]]
        prior = _varx(
            spec={
                "n_targets": (2, 2),
                "coef_matrices": a,
                "cross_structure": "sparse",
                "n_clusters": None,
            }
        )
        truth = VARXTruth.from_recipe(prior.sample(1, seed=0)[0].recipe)
        np.testing.assert_array_equal(truth.lag_matrices, a)
        assert truth.clusters is None

    def test_own_effects_only_by_default(self) -> None:
        group = _varx(spec={"exog_cross_scale": 0.0}).sample(1, seed=0)[0]
        effects = VARXTruth.from_recipe(group.recipe).future_effects[0]
        np.testing.assert_array_equal(effects, np.diag(np.diag(effects)))

    def test_cluster_structure(self) -> None:
        prior = _varx(
            spec={"n_targets": (9, 9), "cross_structure": "clusters", "n_clusters": 3}
        )
        truth = VARXTruth.from_recipe(prior.sample(1, seed=0)[0].recipe)
        a, clusters = truth.lag_matrices[0], truth.clusters
        for i in range(9):
            for c in range(3):
                others = [j for j in np.flatnonzero(clusters == c) if j != i]
                np.testing.assert_allclose(a[i, others], a[i, others[0]])

    def test_sparse_structure(self) -> None:
        prior = _varx(
            spec={
                "n_targets": (40, 40),
                "cross_structure": "sparse",
                "sparsity": 0.2,
                "n_clusters": None,
            }
        )
        a = VARXTruth.from_recipe(prior.sample(1, seed=0)[0].recipe).lag_matrices[0]
        off = a[~np.eye(40, dtype=bool)]
        assert 0.1 < np.mean(off != 0) < 0.3
        assert np.all(np.diag(a) > 0)

    @pytest.mark.parametrize("structure", ["dense", "sparse", "clusters"])
    def test_lag_two_processes_stay_stable(self, structure: str) -> None:
        prior = _varx(
            spec={
                "n_targets": (40, 40),
                "lag_order": 2,
                "cross_structure": structure,
                "n_clusters": 4,
            }
        )
        y = VARXTruth.split(prior.sample(1, seed=0)[0])["y"]
        assert np.isfinite(y).all() and np.abs(y).max() < 1e3

    def test_validation(self) -> None:
        with pytest.raises(ValidationError, match="requires n_clusters"):
            VARXSpec(cross_structure="clusters")
        with pytest.raises(ValidationError, match="n_clusters exceeds"):
            VARXSpec(n_targets=(3, 6), n_clusters=5)
        with pytest.raises(ValidationError, match="fixes n_targets"):
            VARXSpec(n_targets=(2, 4), intercept=[0.0, 1.0])
        with pytest.raises(ValidationError, match="unstable"):
            VARXSpec(n_targets=(1, 1), coef_matrices=[[[1.2]]])
        with pytest.raises(ValidationError, match="shape"):
            VARXSpec(n_targets=(2, 2), coef_matrices=[[[0.2], [0.1]]])
        with pytest.raises(ValidationError, match="shape"):
            VARXSpec(n_targets=(1, 1), lag_order=2, coef_matrices=[[[0.2]]])
        with pytest.raises(ValidationError, match="symmetric"):
            VARXSpec(n_targets=(2, 2), innovation_covariance=[[1, 10], [0, 1]])
        with pytest.raises(ValidationError, match="positive definite"):
            VARXSpec(n_targets=(2, 2), innovation_covariance=[[1, 2], [2, 1]])
        with pytest.raises(ValidationError, match="finite"):
            VARXSpec(n_targets=(1, 1), intercept=[float("nan")])

    def test_recipe_decoding_is_validated(self) -> None:
        group = _varx().sample(1, seed=0)[0]
        recipe = group.recipe
        with pytest.raises(ValueError, match="not a VARX"):
            VARXTruth.from_recipe({"mechanism": "coupled"})
        with pytest.raises(ValueError, match="version"):
            VARXTruth.from_recipe({**recipe, "version": -1})
        tampered = json.loads(json.dumps(recipe))
        tampered["process"]["intercept"] = [0.0]
        with pytest.raises(ValueError, match="inconsistent shapes"):
            VARXTruth.from_recipe(tampered)
        light = _varx().sample(1, seed=0, recipes=False)[0]
        with pytest.raises(ValueError, match="recipes=True"):
            VARXTruth.from_recipe(light.recipe)
        with pytest.raises(ValueError, match="split needs"):
            VARXTruth.split(light)

    def test_shuffled_controls_have_no_process(self) -> None:
        batch = _varx(spec={"n_clusters": None}).sample(4, seed=0)
        control = batch.shuffle_covariates(seed=0)[0]
        with pytest.raises(ValueError, match="control"):
            VARXTruth.from_recipe(control.recipe)
        with pytest.raises(ValueError, match="split needs"):
            VARXTruth.split(control)
        truth = VARXTruth.from_recipe(control.recipe["source"])
        assert truth.lag_matrices.shape == (1, 4, 4)

    def test_oracle_forecast_needs_room(self) -> None:
        group = _varx().sample(1, seed=0)[0]
        truth = VARXTruth.from_recipe(group.recipe)
        with pytest.raises(ValueError, match="origin"):
            truth.forecast_group(group, origin=199, h=5)

    def test_oracle_multistep_matches_a_noiseless_process(self) -> None:
        """Without innovations and past-only regressors, the h-step oracle is
        the process itself, for lag order 2 and every horizon step."""
        prior = _varx(
            spec={
                "n_targets": (3, 3),
                "lag_order": 2,
                "n_past": 0,
                "n_clusters": None,
                "innovation_covariance": (1e-24 * np.eye(3)).tolist(),
            }
        )
        group = prior.sample(1, seed=5)[0]
        truth = VARXTruth.from_recipe(group.recipe)
        y = VARXTruth.split(group)["y"]
        np.testing.assert_allclose(
            truth.forecast_group(group, origin=150, h=10), y[150:160], atol=1e-8
        )

    def test_oracle_past_regressors_decay_with_persistence(self) -> None:
        truth = VARXTruth(
            lag_matrices=np.zeros((1, 1, 1)),
            intercept=np.zeros(1),
            shared_effects=np.zeros((1, 0)),
            future_effects=np.zeros((0, 1, 1)),
            past_effects=np.full((1, 1, 1), 2.0),
            innovation_covariance=np.eye(1),
            exog_persistence=0.5,
            clusters=None,
        )
        out = truth.forecast(
            np.zeros((1, 1)),
            np.zeros((3, 0)),
            np.zeros((3, 1, 0)),
            np.ones((1, 1)),
            3,
        )
        np.testing.assert_allclose(out[:, 0], [2.0, 1.0, 0.5])


@pytest.mark.stats
class TestVARXStats:
    def test_residuals_are_the_innovations(self) -> None:
        group = _varx(length=(3000, 3000)).sample(1, seed=0)[0]
        truth = VARXTruth.from_recipe(group.recipe)
        for e in _residuals(truth, VARXTruth.split(group)).T:
            assert_std(e, expected=1.0)

    def test_regressors_are_unit_variance_ar1(self) -> None:
        prior = _varx(
            spec={"n_targets": (2, 2), "exog_persistence": 0.6, "n_clusters": None},
            length=(5000, 5000),
        )
        x = VARXTruth.split(prior.sample(1, seed=0)[0])["future"][:, 0, 0]
        # AR(1) samples: thin to near independence before the iid variance test
        assert_std(x[::8], expected=1.0)
        assert abs(np.corrcoef(x[1:], x[:-1])[0, 1] - 0.6) < 0.05

    def test_oracle_one_step_error_is_the_innovation(self) -> None:
        group = _varx(
            spec={"n_targets": (3, 3), "n_clusters": None}, length=(600, 600)
        ).sample(1, seed=0)[0]
        truth = VARXTruth.from_recipe(group.recipe)
        y = VARXTruth.split(group)["y"]
        errors = [
            y[t] - truth.forecast_group(group, origin=t, h=1)[0]
            for t in range(100, 600)
        ]
        assert_std(np.concatenate(errors), expected=1.0)


def test_derived_from_target_follows_the_whole_chain() -> None:
    from synforecast.generators import RandomWalkGenerator

    prior = GroupPrior(
        mechanism=CoupledSpec(
            base_pool=[GeneratorSpec(generator=RandomWalkGenerator)],
            n_channels=(4, 4),
            couplings=["leadlag"],
            p_leadlag=1.0,
            targets="one",
        ),
        length=(64, 64),
    )
    checked = 0
    for group in prior.sample(30, seed=0):
        channels = group.recipe["channels"]
        target = next(i for i, c in enumerate(channels) if c["role"] == "target")
        ancestors: dict[int, set[int]] = {i: set() for i in range(4)}
        for pair in group.recipe["coupling"]["leadlag"]:
            ancestors[pair["dst"]] = ancestors[pair["src"]] | {pair["src"]}
        for j, c in enumerate(channels):
            if j != target:
                assert c["derived_from_target"] == (target in ancestors[j])
                checked += (
                    c["derived_from_target"]
                    and channels[target]["leadlag_root"] != target
                )
    assert checked > 0  # chains whose target is itself a copy are covered
