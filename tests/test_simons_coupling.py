"""Tests for the Simons-style coupling mechanisms (synforecast.simons_coupling)."""

import numpy as np
import pytest

from synforecast import SimonsCoupling
from synforecast.simons_coupling import (
    COUPLING_VERSION,
    DRIVERS,
    MECHANISMS,
    CouplingPlan,
    fft_length,
)

LENGTH = 512


def _bases(rng, n, padded):
    x = np.cumsum(rng.standard_normal((n, padded)), -1) * 0.05 + rng.standard_normal(
        (n, padded)
    )
    x = x - x.mean(-1, keepdims=True)
    return x / x.std(-1, ddof=1, keepdims=True)


def _group(coupling, size, seed, length=LENGTH):
    rng = np.random.default_rng(seed)
    plan = coupling.plan(size, rng)
    padded = coupling.padded_length(length)
    own = _bases(rng, size, padded)
    drivers = _bases(rng, plan.base_drivers, padded)
    return plan, own, drivers, rng


def _only(mechanism, **kwargs):
    return SimonsCoupling(mechanisms=(mechanism,), **kwargs)


def test_fft_length_is_even_seven_smooth_and_minimal():
    for n in (1, 2, 7, 97, 4864, 8960, 9001):
        m = fft_length(n)
        assert m >= n and m % 2 == 0
        r = m
        for p in (2, 3, 5, 7):
            while r % p == 0:
                r //= p
        assert r == 1
        assert all(not _smooth_even(k) for k in range(n, m))


def _smooth_even(k):
    if k % 2:
        return False
    for p in (2, 3, 5, 7):
        while k % p == 0:
            k //= p
    return k == 1


def test_padded_length_covers_lag_max():
    c = SimonsCoupling(lag_max=256)
    assert c.padded_length(4608) >= 4608 + 256
    assert c.padded_length(8704) == fft_length(8704 + 256)
    with pytest.raises(ValueError):
        c.padded_length(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mechanisms": ()},
        {"mechanisms": ("mixing",)},
        {"mechanisms": ("mix", "mix")},
        {"lag_max": 3},
    ],
)
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        SimonsCoupling(**kwargs)


def test_settings_are_json_able_and_versioned():
    s = SimonsCoupling().settings()
    assert (
        s["version"] == COUPLING_VERSION
        and s["mechanisms"] == list(MECHANISMS)
        and s["lag_max"] == 256
    )


def test_singleton_groups_are_identity():
    c = SimonsCoupling()
    for seed in range(20):
        plan = c.plan(1, np.random.default_rng(seed))
        assert (
            plan.mechanism == 0 and plan.base_drivers == 0 and (plan.parent == -1).all()
        )


def test_plan_is_deterministic_and_well_formed():
    c = SimonsCoupling()
    seen = set()
    for seed in range(400):
        size = int(np.random.default_rng(seed + 1000).integers(2, 17))
        a = c.plan(size, np.random.default_rng(seed))
        b = c.plan(size, np.random.default_rng(seed))
        for field in (
            "driver_kind",
            "driver_param",
            "parent",
            "tau",
            "decay",
            "coef",
            "rho",
            "func_kind",
            "func_param",
        ):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        seen.add(a.mechanism_name)
        assert a.size == size
        assert (a.parent < DRIVERS + size).all()
        follower = a.parent >= DRIVERS
        assert (a.parent[follower] - DRIVERS != np.nonzero(follower)[0]).all()
        assert ((a.tau >= 0) & (a.tau <= c.lag_max)).all()
        assert ((a.rho >= 0) & (a.rho <= 1)).all()
        used = a.driver_kind >= 0
        assert set(np.unique(a.parent[(a.parent >= 0) & (a.parent < DRIVERS)])) <= set(
            np.nonzero(used)[0]
        )
    assert seen == set(MECHANISMS)


def test_mechanism_choice_is_uniform_over_the_listed_mechanisms():
    c = SimonsCoupling(mechanisms=("mix", "event"))
    rng = np.random.default_rng(0)
    names = [c.plan(4, rng).mechanism_name for _ in range(4000)]
    share = names.count("mix") / len(names)
    assert set(names) == {"mix", "event"} and abs(share - 0.5) < 0.03


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_couple_output_shape_finite_and_standardised(mechanism):
    c = _only(mechanism)
    for seed in range(6):
        plan, own, drivers, rng = _group(c, 5, seed)
        out = c.couple(plan, own, drivers, LENGTH, rng)
        assert out.shape == (5, LENGTH) and np.isfinite(out).all()
        np.testing.assert_allclose(out.mean(-1), 0, atol=1e-9)
        np.testing.assert_allclose(out.std(-1, ddof=1), 1, atol=1e-9)


def test_identity_returns_the_cropped_bases():
    c = _only("identity")
    plan, own, drivers, rng = _group(c, 4, 3)
    out = c.couple(plan, own, drivers, LENGTH, rng)
    tail = own[:, -LENGTH:]
    expected = (tail - tail.mean(-1, keepdims=True)) / tail.std(
        -1, ddof=1, keepdims=True
    )
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_couple_is_deterministic_given_the_stream():
    c = SimonsCoupling()
    for seed in range(10):
        plan, own, drivers, _ = _group(c, 6, seed)
        a = c.couple(plan, own, drivers, LENGTH, np.random.default_rng(7))
        b = c.couple(plan, own, drivers, LENGTH, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)


def test_couple_does_not_modify_its_inputs():
    c = SimonsCoupling()
    plan, own, drivers, rng = _group(c, 6, 11)
    own0, drivers0 = own.copy(), drivers.copy()
    c.couple(plan, own, drivers, LENGTH, rng)
    np.testing.assert_array_equal(own, own0)
    np.testing.assert_array_equal(drivers, drivers0)


def test_lag_followers_track_their_driver_at_the_planned_lag():
    c = _only("lag", lag_max=64)
    padded = c.padded_length(LENGTH)
    checked = 0
    for seed in range(300):
        rng = np.random.default_rng(seed)
        plan = c.plan(3, rng)
        own = rng.standard_normal((3, padded))
        drivers = rng.standard_normal((plan.base_drivers, padded))
        out = c.couple(plan, own, drivers, LENGTH, rng)
        for j in range(3):
            single = (
                plan.parent[j, 1] < 0
                and 0 <= plan.parent[j, 0] < DRIVERS
                and plan.decay[j, 0] == 0
            )
            if plan.root[j] or not single or plan.rho[j] < 0.6:
                continue
            k = int(plan.parent[j, 0])
            corr = [
                abs(
                    float(
                        np.corrcoef(
                            out[j], drivers[k][padded - LENGTH - lag : padded - lag]
                        )[0, 1]
                    )
                )
                for lag in range(c.lag_max + 1)
            ]
            assert int(np.argmax(corr)) == int(plan.tau[j, 0]), (seed, j)
            assert corr[int(plan.tau[j, 0])] > 0.6
            checked += 1
    assert checked >= 10


def test_roots_observe_their_driver():
    c = _only("lag")
    checked = 0
    for seed in range(60):
        plan, own, drivers, rng = _group(c, 4, seed)
        out = c.couple(plan, own, drivers, LENGTH, rng)
        for j in np.nonzero(plan.root)[0]:
            k = int(plan.parent[j, 0])
            corr = float(np.corrcoef(out[j], drivers[k][-LENGTH:])[0, 1])
            assert corr > 0.85 * np.sqrt(plan.rho[j]) - 0.05, (seed, j, corr)
            checked += 1
    assert checked >= 20


def test_mix_channels_share_their_drivers():
    c = _only("mix")
    corrs = []
    for seed in range(40):
        plan, own, drivers, rng = _group(c, 4, seed)
        out = c.couple(plan, own, drivers, LENGTH, rng)
        cm = np.corrcoef(out)
        corrs.append(np.abs(cm[np.triu_indices(4, 1)]).mean())
    assert np.mean(corrs) > 0.3


def test_event_channels_respond_to_impulses_or_pulses():
    c = _only("event")
    for seed in range(10):
        plan, own, drivers, rng = _group(c, 3, seed)
        assert plan.base_drivers == 0 and plan.driver_kind[0] in (3, 4)
        out = c.couple(plan, own, drivers, LENGTH, rng)
        assert np.isfinite(out).all()


def test_coint_uses_walk_drivers_and_ou_noise():
    c = _only("coint")
    for seed in range(10):
        plan, own, drivers, rng = _group(c, 3, seed)
        assert plan.base_drivers == 0 and set(
            plan.driver_kind[plan.driver_kind >= 0]
        ) <= {1, 2}
        assert (
            (plan.own_kind == 1).all()
            and (plan.ou >= 2).all()
            and (plan.ou <= 100).all()
        )
        out = c.couple(plan, own, drivers, LENGTH, rng)
        assert np.isfinite(out).all()


def test_func_channels_carry_a_nonlinear_map():
    c = _only("func")
    kinds = set()
    for seed in range(60):
        plan = c.plan(4, np.random.default_rng(seed))
        kinds |= set(plan.func_kind[~plan.root].tolist())
    assert kinds == {1, 2, 3, 4, 5, 6}


def test_chain_followers_read_first_pass_outputs():
    c = _only("lag")
    found = 0
    for seed in range(300):
        plan = c.plan(8, np.random.default_rng(seed))
        found += int((plan.parent >= DRIVERS).any())
    assert found > 0


def test_counts_summarise_the_plan():
    c = SimonsCoupling()
    plan = c.plan(6, np.random.default_rng(5))
    counts = plan.counts()
    assert counts[f"mechanism_{plan.mechanism_name}"] == 1 and counts["channels"] == 6
    assert counts["edges"] == int((plan.parent >= 0).sum())


@pytest.mark.parametrize(
    "bad",
    [
        {"own_rows": 3},
        {"short": True},
        {"nan": True},
        {"drivers": 5},
    ],
)
def test_couple_rejects_bad_inputs(bad):
    c = _only("mix")
    plan, own, drivers, rng = _group(c, 4, 1)
    if "own_rows" in bad:
        own = own[: bad["own_rows"]]
    if "short" in bad:
        own, drivers = own[:, : LENGTH + 10], drivers[:, : LENGTH + 10]
    if "nan" in bad:
        own = own.copy()
        own[0, 5] = np.nan
    if "drivers" in bad:
        drivers = np.zeros((bad["drivers"], own.shape[1]))
    with pytest.raises(ValueError):
        c.couple(plan, own, drivers, LENGTH, rng)


def test_plan_is_a_frozen_record():
    plan = SimonsCoupling().plan(3, np.random.default_rng(0))
    assert isinstance(plan, CouplingPlan)
    with pytest.raises(AttributeError):
        plan.mechanism = 2


def test_existing_draws_are_unchanged_by_the_new_module():
    from synforecast import Multivariatizer
    from synforecast.generators import TSIGenerator

    base = TSIGenerator(min_length=64, max_length=64, freq="h", seed=1)
    a = Multivariatizer(base=base, seed=3).generate(n_series=3)
    import synforecast.simons_coupling  # noqa: F401

    b = Multivariatizer(base=base, seed=3).generate(n_series=3)
    assert a.equals(b)


def test_chain_followers_track_their_parent_channel_at_the_planned_lag():
    c = _only("lag", lag_max=64)
    padded = c.padded_length(LENGTH + c.lag_max)
    checked = 0
    for seed in range(400):
        rng = np.random.default_rng(seed)
        plan = c.plan(6, rng)
        own = rng.standard_normal((6, padded))
        drivers = rng.standard_normal((plan.base_drivers, padded))
        out = c.couple(plan, own, drivers, LENGTH + c.lag_max, rng)
        for j in range(6):
            i = int(plan.parent[j, 0]) - DRIVERS
            if (
                i < 0
                or plan.parent[j, 1] >= 0
                or plan.decay[j, 0] > 0
                or plan.rho[j] < 0.7
                or plan.rho[i] < 0.7
            ):
                continue
            tau = int(plan.tau[j, 0])
            n = LENGTH
            corr = abs(
                float(
                    np.corrcoef(out[j, -n:], out[i, -n - tau : out.shape[1] - tau])[
                        0, 1
                    ]
                )
            )
            assert corr > 0.5, (seed, j, i, corr)
            checked += 1
    assert checked >= 5
