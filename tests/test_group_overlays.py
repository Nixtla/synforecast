"""Tests for group overlays (GroupOverlaySpec) and covariate observation plans
(ObservationSpec, Group.observe, GroupBatch.observe)."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from synforecast import (
    CoupledSpec,
    FixedRegressionSpec,
    GeneratorSpec,
    GroupOverlaySpec,
    GroupPrior,
    ObservationSpec,
    RegressionSpec,
    Term,
    VARXSpec,
    VARXTruth,
)
from synforecast.generators import SARIMAGenerator
from synforecast.overlays import Changepoints, OverlayConfig, apply_plan, native_plan
from tests._group_reference import digest, reference_priors

FIXTURES = Path(__file__).parent / "fixtures"
WHITE = GeneratorSpec(
    generator=SARIMAGenerator,
    params={"p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "noise_std": 1.0},
)
MECHANISMS = {
    "regression": RegressionSpec(),
    "coupled": CoupledSpec(targets="one", n_channels=(2, 5)),
    "varx": VARXSpec(n_targets=(2, 3), n_future=1),
}
FULL = {
    "shared_rate": 1.0,
    "local_rate": 1.0,
    "calendar_rate": 0.5,
    "observability_rate": 0.2,
}


def _prior(mechanism="regression", length=(96, 160), freq="h", **overlays):
    return GroupPrior(
        mechanism=MECHANISMS[mechanism] if isinstance(mechanism, str) else mechanism,
        length=length,
        freq=freq,
        overlays=GroupOverlaySpec(**overlays),
    )


def _identity_prior(apply_to, channels, length=200):
    """y = x exactly (zero base, no noise): a driver event reaches the target only through the edge."""
    mechanism = FixedRegressionSpec(
        drivers=(WHITE,), terms=(Term(kind="linear", driver=0, coef=1.0),)
    )
    return GroupPrior(
        mechanism=mechanism,
        length=(length, length),
        freq="D",
        overlays=GroupOverlaySpec(
            shared_rate=1.0, local_rate=0.0, shared_channels=channels, apply_to=apply_to
        ),
    )


def _same(a, b):
    return all(
        np.array_equal(x.values, y.values, equal_nan=True) and x.recipe == y.recipe
        for x, y in zip(a, b, strict=True)
    )


# -- overlays=None and zero rates -------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() != "x86_64",
    reason="the fixture hashes floats drawn on linux x86_64",
)
def test_no_overlays_draws_bitwise_as_3b33546e():
    reference = json.loads((FIXTURES / "group_draws_3b33546e.json").read_text())
    for name, prior in reference_priors().items():
        assert digest(prior) == reference[name], name


def test_no_overlays_keeps_the_fingerprint_and_recipes():
    plain = GroupPrior(length=(64, 64))
    explicit = GroupPrior(length=(64, 64), overlays=None)
    assert plain.fingerprint == explicit.fingerprint
    assert "overlays" not in plain.sample(2, seed=1)[0].recipe
    overlaid = GroupPrior(length=(64, 64), overlays=GroupOverlaySpec())
    assert overlaid.fingerprint != plain.fingerprint


@pytest.mark.parametrize("mechanism", sorted(MECHANISMS))
def test_zero_rates_change_no_value(mechanism):
    plain = GroupPrior(mechanism=MECHANISMS[mechanism], length=(96, 160), freq="h")
    zero = _prior(mechanism, shared_rate=0.0, local_rate=0.0)
    for a, b in zip(plain.sample(6, seed=2), zero.sample(6, seed=2), strict=True):
        np.testing.assert_array_equal(a.values, b.values)
        assert (
            b.recipe["overlays"]["shared"] is None
            and b.recipe["overlays"]["local"] == []
        )


# -- determinism and replay ---------------------------------------------------------


@pytest.mark.parametrize(
    ("mechanism", "apply_to"),
    [
        ("regression", "emitted"),
        ("regression", "drivers"),
        ("coupled", "drivers"),
        ("varx", "emitted"),
    ],
)
def test_overlaid_groups_depend_only_on_seed_and_group_id(mechanism, apply_to):
    prior = _prior(mechanism, apply_to=apply_to, **FULL)
    batch = prior.sample(8, seed=11)
    assert _same(batch, prior.sample(8, seed=11, n_jobs=1))
    for i, group in enumerate(batch):
        assert _same([group], prior.sample(1, seed=11, start_group=i, n_jobs=1).groups)
    assert not _same(batch, prior.sample(8, seed=12))


@pytest.mark.parametrize("apply_to", ["emitted", "drivers"])
def test_replay_rebuilds_overlays_exactly(apply_to):
    prior = _prior("regression", apply_to=apply_to, **FULL)
    full = prior.sample(6, seed=4)
    light = prior.sample(6, seed=4, recipes=False)
    for f, g in zip(full, light, strict=True):
        np.testing.assert_array_equal(f.values, g.values)
        again = prior.replay(json.loads(json.dumps(f.recipe)))
        np.testing.assert_array_equal(again.values, f.values)
        assert again.recipe == f.recipe
        assert prior.replay(g.recipe).recipe == f.recipe


def test_overlays_draw_from_their_own_streams():
    plain = GroupPrior(length=(128, 128), freq="h")
    overlaid = _prior(length=(128, 128), shared_rate=1.0, local_rate=1.0)
    for a, b in zip(plain.sample(5, seed=3), overlaid.sample(5, seed=3), strict=True):
        assert a.roles == b.roles
        a_rec, b_rec = dict(a.recipe), dict(b.recipe)
        b_rec.pop("overlays")
        a_rec.pop("prior"), b_rec.pop("prior")
        assert a_rec == b_rec


# -- shared events ------------------------------------------------------------------


def _deltas(spec_kwargs, n=40, seed=0):
    """(overlaid group, per-channel change against the same group without overlays)."""
    mechanism = RegressionSpec(n_covariates=(3, 5), p_multiplicative=0.0)
    plain = GroupPrior(mechanism=mechanism, length=(300, 300), freq="D")
    overlaid = GroupPrior(
        mechanism=mechanism,
        length=(300, 300),
        freq="D",
        overlays=GroupOverlaySpec(**spec_kwargs),
    )
    for a, b in zip(
        plain.sample(n, seed=seed), overlaid.sample(n, seed=seed), strict=True
    ):
        yield a, b, b.values - a.values


def test_shared_events_coincide_across_channels_and_follow_shared_scale():
    checked, signs = 0, set()
    for plain, group, delta in _deltas(
        {"shared_rate": 1.0, "local_rate": 0.0, "shared_scale": (0.5, 1.5)}
    ):
        shared = group.recipe["overlays"]["shared"]
        assert shared["items"] == list(range(group.values.shape[0]))
        assert all(0.5 <= abs(s) <= 1.5 for s in shared["scales"])
        signs.update(np.sign(shared["scales"]).tolist())
        support = delta != 0
        events = sorted(shared["changepoints"] + shared["anomalies"])
        if not events:
            # A low anomaly fraction can place no event on a short series, as for a single series.
            assert not support.any()
            continue
        # Every channel changes from the first event on, or one step later (trend breaks add slope * 0 at their index).
        assert all(int(np.flatnonzero(row)[0]) - events[0] in (0, 1) for row in support)
        cp = shared["config"].get("changepoints")
        if shared["anomalies"] or cp is None or cp["kind"] != "level":
            continue
        # A level-only schedule: the final change is (sum of level jumps) * channel std * channel factor.
        unit = delta[:, -1] / (plain.values.std(axis=1) * np.asarray(shared["scales"]))
        np.testing.assert_allclose(unit, unit[0], rtol=1e-9)
        checked += 1
    assert checked >= 3 and signs == {-1.0, 1.0}


@pytest.mark.parametrize(
    ("channels", "touched"),
    [("covariates", "covariate"), ("target_and_informative", "target")],
)
def test_shared_channel_selection(channels, touched):
    for _, group, delta in _deltas(
        {"shared_rate": 1.0, "local_rate": 0.0, "shared_channels": channels}, n=20
    ):
        shared = group.recipe["overlays"]["shared"]
        if not shared["changepoints"] + shared["anomalies"]:
            continue
        target = group.target_index
        moved = (delta != 0).any(axis=1)
        if touched == "covariate":
            assert (
                not moved[target]
                and moved[[c for c in range(len(moved)) if c != target]].all()
            )
        else:
            informative = {
                e["channel"] for e in group.recipe["channels"] if e.get("informative")
            } | {target}
            assert set(np.flatnonzero(moved)) == informative


def test_driver_events_reach_the_target_only_through_the_edges():
    emitted = _identity_prior("emitted", "covariates").sample(10, seed=1)
    drivers = _identity_prior("drivers", "covariates").sample(10, seed=1)
    plain = GroupPrior(
        mechanism=_identity_prior("emitted", "all").mechanism,
        length=(200, 200),
        freq="D",
    )
    for base, e, d in zip(plain.sample(10, seed=1), emitted, drivers, strict=True):
        t, x = base.target_index, base.covariate_indices[0]
        np.testing.assert_allclose(base.values[t], base.values[x])
        np.testing.assert_array_equal(e.values[t], base.values[t])
        if (
            e.recipe["overlays"]["shared"]["changepoints"]
            + e.recipe["overlays"]["shared"]["anomalies"]
        ):
            assert not np.array_equal(e.values[x], base.values[x])
        np.testing.assert_allclose(d.values[t], d.values[x])
        if (
            d.recipe["overlays"]["shared"]["changepoints"]
            + d.recipe["overlays"]["shared"]["anomalies"]
        ):
            assert not np.array_equal(d.values[x], base.values[x])
        assert d.recipe["overlays"]["shared"]["items_are"] == "latent requests"


# -- local overlays -------------------------------------------------------------------


def test_local_overlay_rates_and_independence():
    prior = _prior(
        "regression",
        length=(256, 256),
        shared_rate=0.0,
        local_rate=1.0,
        calendar_rate=0.5,
        observability_rate=0.25,
    )
    groups = prior.sample(300, seed=6)
    rows = []
    for group in groups:
        local = {e["channel"]: e["config"] for e in group.recipe["overlays"]["local"]}
        for c in range(group.values.shape[0]):
            config = local.get(c, {})
            rows.append(
                (
                    group.group_id,
                    c,
                    "calendar" in config,
                    "waveform" in config,
                    bool({"changepoints", "anomalies", "missing"} & set(config)),
                )
            )
    calendar = np.mean([r[2] for r in rows])
    waveform = np.mean([r[3] for r in rows])
    assert abs(calendar - 0.5) < 0.05 and abs(waveform - 0.25) < 0.05
    # Channels 0 and 1 draw independently: P(both calendars) is close to the product of the marginals.
    first = {r[0]: r[2] for r in rows if r[1] == 0}
    second = {r[0]: r[2] for r in rows if r[1] == 1}
    both = np.mean([first[g] and second[g] for g in first])
    assert (
        abs(both - np.mean(list(first.values())) * np.mean(list(second.values())))
        < 0.06
    )
    seeds = [e["seeds"] for g in groups[:20] for e in g.recipe["overlays"]["local"]]
    assert len({tuple(s) for s in seeds}) == len(seeds)


def test_local_rate_reproduces_pretraining_overlays_stage_rates():
    prior = _prior("regression", length=(256, 256), shared_rate=0.0, local_rate=1.0)
    configs = [
        {e["channel"]: e["config"] for e in g.recipe["overlays"]["local"]}
        for g in prior.sample(300, seed=8)
    ]
    channels = [
        (i, c)
        for i, g in enumerate(prior.sample(300, seed=8))
        for c in range(g.values.shape[0])
    ]
    share = {
        stage: np.mean([stage in configs[i].get(c, {}) for i, c in channels])
        for stage in ("changepoints", "anomalies", "missing")
    }
    assert abs(share["changepoints"] - 0.3) < 0.05
    assert abs(share["anomalies"] - 0.25) < 0.05
    assert abs(share["missing"] - 0.1) < 0.04


def test_overlays_move_marginals_toward_pretraining_overlays():
    def kurtosis(values):
        out = []
        for row in values:
            d = np.diff(row[np.isfinite(row)])
            out.append(((d - d.mean()) ** 4).mean() / d.var() ** 2 - 3)
        return np.median(out)

    plain = GroupPrior(length=(512, 512), freq="h").sample(80, seed=2)
    overlaid = _prior(length=(512, 512), shared_rate=0.3, local_rate=1.0).sample(
        80, seed=2
    )
    assert (
        kurtosis([g.values[g.target_index] for g in overlaid])
        > kurtosis([g.values[g.target_index] for g in plain]) + 0.3
    )


def test_shuffled_controls_keep_target_overlays_and_take_donor_covariate_overlays():
    prior = GroupPrior(
        mechanism=FixedRegressionSpec(
            drivers=(WHITE, WHITE), terms=(Term(kind="linear", driver=0, coef=1.0),)
        ),
        length=(128, 128),
        freq="h",
        overlays=GroupOverlaySpec(shared_rate=1.0, local_rate=1.0),
    )
    batch = prior.sample(6, seed=0)
    for original, control in zip(batch, batch.shuffle_covariates(seed=1), strict=True):
        np.testing.assert_array_equal(
            control.values[control.target_index], original.values[original.target_index]
        )
        donor = batch[control.recipe["covariates_from"]]
        for source in control.recipe["covariate_sources"]:
            np.testing.assert_array_equal(
                control.values[source["channel"]], donor.values[source["donor_channel"]]
            )
        assert control.recipe["source"]["overlays"] == original.recipe["overlays"]


# -- overlay helpers and validation -----------------------------------------------------


def test_apply_plan_places_the_same_events_on_every_series():
    config = OverlayConfig(
        changepoints=Changepoints(count=2, kind="level", scale="relative")
    )
    plan = native_plan(config, "D", pd.Timestamp("2000-01-01"), None, 100)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=100), 5.0 * rng.normal(size=100)
    moved_a, ia = apply_plan(a, plan, (1, 2, 3))
    moved_b, ib = apply_plan(b, plan, (1, 2, 3))
    np.testing.assert_array_equal(ia["changepoints"], ib["changepoints"])
    np.testing.assert_allclose((moved_b - b)[-1] / b.std(), (moved_a - a)[-1] / a.std())
    assert (
        native_plan(OverlayConfig(), "D", pd.Timestamp("2000-01-01"), None, 100) is None
    )


def test_overlay_validation():
    with pytest.raises(ValidationError):
        GroupOverlaySpec(shared_scale=(1.5, 0.5))
    with pytest.raises(ValidationError):
        GroupOverlaySpec(local_rate=1.5)
    with pytest.raises(ValidationError):
        GroupOverlaySpec(unknown=1)
    with pytest.raises(ValidationError, match="drivers"):
        GroupPrior(mechanism=VARXSpec(), overlays=GroupOverlaySpec(apply_to="drivers"))
    group = _prior("varx", local_rate=1.0).sample(1, seed=0)[0]
    with pytest.raises(ValueError, match="overlaid"):
        VARXTruth.from_recipe(group.recipe)


# -- observation plans ----------------------------------------------------------------


OBSERVATION = ObservationSpec(
    p_drop_all=0.15, p_drop_channel=0.15, p_missing_channel=0.1, slots=5
)


def test_observation_plan_is_bitwise_tsfm_structured_prior_observation():
    fixture = json.loads((FIXTURES / "observation_plan_tsfm.json").read_text())
    spec = ObservationSpec(
        p_drop_all=fixture["p_drop_all"],
        p_drop_channel=fixture["p_drop_channel"],
        p_missing_channel=fixture["p_missing_channel"],
        missing_rate=tuple(fixture["missing_rate"]),
        missing_block=tuple(fixture["missing_block"]),
        slots=fixture["slots"],
    )
    assert len(fixture["plans"]) == 1000
    for group_id, (length, drop_all, dropped, count, digest16) in enumerate(
        fixture["plans"]
    ):
        flag, drops, missing = spec.plan(fixture["seed"], group_id, length)
        assert (int(flag), "".join(str(int(x)) for x in drops)) == (
            drop_all,
            dropped,
        ), group_id
        assert int(missing.sum()) == count, group_id
        assert (
            hashlib.sha256(np.packbits(missing).tobytes()).hexdigest()[:16] == digest16
        ), group_id


def test_observe_rates_and_targets_untouched():
    prior = GroupPrior(
        mechanism=RegressionSpec(n_covariates=(5, 5)), length=(200, 200), freq="h"
    )
    batch = prior.sample(400, seed=1).observe(OBSERVATION, 7)
    drop_all = np.mean([g.recipe["observation"]["drop_all"] for g in batch])
    unavailable = [
        len(g.recipe["observation"]["unavailable"])
        for g in batch
        if not g.recipe["observation"]["drop_all"]
    ]
    missing = [
        n > 0
        for g in batch
        if not g.recipe["observation"]["drop_all"]
        for n in g.recipe["observation"]["missing"].values()
    ]
    assert abs(drop_all - 0.15) < 0.04
    assert abs(np.mean(unavailable) / 5 - 0.15) < 0.03
    assert abs(np.mean(missing) - 0.1) < 0.03
    for g in batch:
        assert np.isfinite(g.values[g.target_indices]).all()
        for channel in g.recipe["observation"]["unavailable"]:
            assert np.isnan(g.values[channel]).all()


def test_observe_depends_only_on_seed_group_id_and_slot():
    prior = GroupPrior(mechanism=RegressionSpec(n_covariates=(5, 5)), length=(150, 150))
    batch = prior.sample(12, seed=3).observe(OBSERVATION, 5)
    for i, group in enumerate(batch):
        single = prior.sample(1, seed=3, start_group=i).observe(OBSERVATION, 5)[0]
        np.testing.assert_array_equal(single.values, group.values)
        assert single.recipe == group.recipe
        assert prior.replay(group.recipe).recipe == group.recipe
        np.testing.assert_array_equal(prior.replay(group.recipe).values, group.values)
    other = prior.sample(12, seed=3).observe(OBSERVATION, 6)
    assert any(
        not np.array_equal(a.values, b.values, equal_nan=True)
        for a, b in zip(batch, other, strict=True)
    )


def test_observe_after_shuffle_applies_the_receivers_plan():
    prior = GroupPrior(
        mechanism=FixedRegressionSpec(
            drivers=(WHITE, WHITE, WHITE),
            terms=(Term(kind="linear", driver=0, coef=1.0),),
        ),
        length=(120, 120),
    )
    spec = ObservationSpec(p_drop_channel=0.4, p_missing_channel=0.5, slots=3)
    batch = prior.sample(10, seed=2)
    controls = batch.shuffle_covariates(seed=4)
    observed = controls.observe(spec, 9)
    for control, group in zip(controls, observed, strict=True):
        drop_all, dropped, missing = spec.plan(9, control.group_id, control.length)
        for slot, channel in enumerate(control.covariate_indices):
            expected = (
                np.full(control.length, np.nan)
                if drop_all or dropped[slot]
                else np.where(missing[slot], np.nan, control.values[channel])
            )
            np.testing.assert_array_equal(group.values[channel], expected)
        np.testing.assert_array_equal(
            group.values[group.target_index], control.values[control.target_index]
        )


def test_observe_validation():
    group = GroupPrior(
        mechanism=RegressionSpec(n_covariates=(3, 3)), length=(50, 50)
    ).sample(1, seed=0)[0]
    with pytest.raises(ValueError, match="observation slots"):
        group.observe(ObservationSpec(slots=2), 0)
    with pytest.raises(ValueError, match="already observed"):
        group.observe(ObservationSpec(slots=3), 0).observe(ObservationSpec(slots=3), 0)
    with pytest.raises(ValidationError):
        ObservationSpec(slots=3, missing_block=(0, 4))
    with pytest.raises(ValidationError):
        ObservationSpec(slots=3, p_drop_all=1.5)
