"""Tests for the real-window augmentation specs and their per-window array API."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from synforecast import (
    AmplitudeEnvelope,
    Censor,
    Downsample,
    Negate,
    PowerWarp,
    RealWindowAugment,
    Reverse,
    Stretch,
    WindowPlan,
    _window,
)
from tests.helpers import to_pandas

FULL = RealWindowAugment(
    downsample=0.4,
    stretch=0.25,
    power_warp=0.2,
    amplitude_envelope=0.2,
    censor=0.2,
    mixup=0.2,
    negate=0.2,
    reverse=0.1,
)
ALWAYS = RealWindowAugment(
    downsample=1.0,
    stretch=1.0,
    power_warp=1.0,
    amplitude_envelope=1.0,
    censor=1.0,
    mixup=1.0,
    negate=1.0,
    reverse=1.0,
)
SPECS = [
    Downsample(seed=3),
    Stretch(seed=3),
    PowerWarp(seed=3),
    AmplitudeEnvelope(seed=3),
    Censor(seed=3),
    Negate(seed=3),
    Reverse(seed=3),
]


def _series(n, seed=0, missing=0.1, positive=True):
    rng = np.random.default_rng(seed)
    x = 10 + np.cumsum(rng.normal(size=n)) * 0.1 + np.sin(np.arange(n) / 5)
    x = np.abs(x) if positive else x - x.mean()
    x[rng.random(n) < missing] = np.nan
    return x


def _frame(lengths=(2000, 1300, 600), seed=0):
    parts = []
    for i, n in enumerate(lengths):
        parts.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": pd.date_range("2020-01-01", periods=n, freq="h"),
                    "y": _series(n, seed + i),
                    "x": np.arange(n, dtype=float),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _window_pass(aug, x, seed, want=3041):
    rng = np.random.default_rng(seed)
    plan = aug.draw(rng, len(x), want)
    raw = (
        x[plan.resample.start : plan.resample.stop]
        if plan.resample
        else x[-min(len(x), want) :]
    )
    n = plan.resample.out if plan.resample else len(raw)
    partner = _series(n, seed + 100)
    return plan, aug.transform(raw, None, plan, partner, None)


class TestShapesAndMissing:
    @pytest.mark.parametrize("seed", range(40))
    def test_composite_output_contract(self, seed):
        x = _series(9000, seed)
        plan, (y, ok, applied) = _window_pass(FULL, x, seed)
        assert y.dtype == np.float64 and ok.dtype == bool and y.shape == ok.shape
        assert np.array_equal(ok, np.isfinite(y))
        assert len(y) <= 3041
        assert plan.active == bool(applied) or plan.mixup is not None

    def test_value_kernels_keep_length_and_missing(self):
        x, ok = _window.as_pair(_series(800, 1))
        for out in (
            _window.power_warp(x, ok, 1.7),
            _window.censor(x, ok, True, 0.3),
            _window.censor(x, ok, False, 0.8),
            _window.amplitude_envelope(
                x, ok, np.array([0, 0.3, 0.6, 1.0]), np.array([1, 2, 0.5, 1.0])
            ),
            _window.negate(x, ok),
        ):
            y, yo = out
            assert len(y) == len(x) and np.array_equal(yo, ok)
            assert np.isnan(y[~ok]).all() and np.isfinite(y[ok]).all()

    def test_mask_argument_hides_finite_values(self):
        x = np.arange(10.0)
        mask = np.ones(10, bool)
        mask[3] = False
        y, ok = Negate().transform(x, mask)
        assert not ok[3] and np.isnan(y[3]) and y[4] == -4.0

    def test_reverse_keeps_outer_gaps(self):
        x = np.array([np.nan, 1.0, 2.0, np.nan, 4.0, np.nan])
        y, ok = Reverse().transform(x)
        np.testing.assert_array_equal(y, [np.nan, 4.0, np.nan, 2.0, 1.0, np.nan])
        np.testing.assert_array_equal(ok, np.isfinite(y))

    def test_mixup_missing_where_either_is(self):
        x, ok = _window.as_pair(np.array([1.0, np.nan, 3.0, 4.0]))
        xb, okb = _window.as_pair(np.array([2.0, 2.0, np.nan, 2.0]))
        y, yo = _window.mixup(x, ok, xb, okb, 0.5)
        np.testing.assert_array_equal(yo, [True, False, False, True])
        sa, sb = np.mean([1.0, 3.0, 4.0]), 2.0
        assert y[0] == pytest.approx(sa * (0.5 / sa + 0.5 * 2.0 / sb))

    def test_infinite_values_reject_apply(self):
        df = _frame((700,))
        df.loc[5, "y"] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            Negate(seed=1).apply(df)


class TestPrefixInvariance:
    @pytest.mark.parametrize("seed", range(25))
    def test_quantities_ignore_values_after_prefix(self, seed):
        x = _series(3041, seed)
        partner = _series(3041, seed + 7)
        drawn = ALWAYS.draw(np.random.default_rng(seed), 10_000, 3041)
        plan = WindowPlan(
            power=drawn.power,
            mixup=drawn.mixup,
            censor=drawn.censor,
            envelope=drawn.envelope,
            negate=True,
        )
        end = _window.prefix_end(np.isfinite(x))
        assert np.isfinite(x[:end]).sum() == _window.PREFIX
        changed = x.copy()
        changed[end:] = changed[end:] * 50 + 1000
        partner_changed = partner.copy()
        partner_changed[max(end, _window.prefix_end(np.isfinite(partner))) :] *= -30
        ya, _, applied_a, used_a = ALWAYS.trace(x, None, plan, partner)
        yb, _, applied_b, used_b = ALWAYS.trace(changed, None, plan, partner_changed)
        assert used_a == used_b and applied_a == applied_b
        assert {
            "power_scale",
            "censor_level",
            "mixup_scale",
            "mixup_partner_scale",
        } <= set(used_a)
        np.testing.assert_array_equal(ya[:end], yb[:end])

    @pytest.mark.parametrize("seed", range(10))
    def test_resampled_quantities_ignore_raw_values_after_prefix(self, seed):
        x = _series(20_000, seed, missing=0.02)
        for trial in range(20):
            plan = ALWAYS.draw(np.random.default_rng(1000 * seed + trial), len(x), 3041)
            raw = x[plan.resample.start : plan.resample.stop]
            if plan.resample.kind != "up":
                continue
            changed = raw.copy()
            changed[_window.prefix_end(np.isfinite(raw)) :] += 77.0
            n = plan.resample.out
            used_a = ALWAYS.trace(raw, None, plan, _series(n, 1))[3]
            used_b = ALWAYS.trace(changed, None, plan, _series(n, 1))[3]
            assert used_a["bridge_sigma"] == used_b["bridge_sigma"]

    def test_bridge_scale_reads_only_the_prefix(self):
        x, ok = _window.as_pair(_series(3000, 4))
        changed = x.copy()
        end = _window.prefix_end(ok)
        changed[end:] += np.random.default_rng(0).normal(size=len(x) - end) * 100
        assert _window.bridge_sigma(x, ok) == _window.bridge_sigma(changed, ok)

    def test_power_warp_gate_reads_only_the_prefix(self):
        x, ok = _window.as_pair(_series(2000, 5, missing=0.0))
        x[1500] = -3.0
        y, _ = _window.power_warp(x, ok, 2.0)
        s = _window.mean_abs(x, ok)
        assert y[1500] == pytest.approx(-s * (3.0 / s) ** 2)


class TestDeterminism:
    def test_same_seed_same_plan_and_window(self):
        x = _series(20_000, 2)
        for seed in range(10):
            p1, (y1, o1, a1) = _window_pass(FULL, x, seed)
            p2, (y2, o2, a2) = _window_pass(FULL, x, seed)
            assert p1 == p2 and a1 == a2
            np.testing.assert_array_equal(y1, y2)
            np.testing.assert_array_equal(o1, o2)

    def test_different_seeds_differ(self):
        plans = {ALWAYS.draw(np.random.default_rng(s), 20_000, 3041) for s in range(20)}
        assert len(plans) == 20

    def test_rates_do_not_move_other_draws(self):
        for seed in range(20):
            full = ALWAYS.draw(np.random.default_rng(seed), 20_000, 3041)
            only = RealWindowAugment(censor=1.0).draw(
                np.random.default_rng(seed), 20_000, 3041
            )
            assert only.censor == full.censor and only.resample is None
            follow = np.random.default_rng(seed)
            FULL.draw(follow, 20_000, 3041)
            other = np.random.default_rng(seed)
            RealWindowAugment().draw(other, 50, 64)
            assert follow.random() == other.random()

    @pytest.mark.parametrize("spec", SPECS, ids=lambda s: type(s).__name__)
    def test_apply_is_seeded(self, spec):
        df = _frame()
        a = to_pandas(spec.apply(df, include_original=False))
        b = to_pandas(spec.apply(df, include_original=False))
        pd.testing.assert_frame_equal(a, b)
        if not isinstance(spec, Negate | Reverse):
            c = to_pandas(spec.apply(df, seed=99, include_original=False))
            assert not a["y"].equals(c["y"])


class TestDownsample:
    @pytest.mark.parametrize("length", [1023, 1024, 1500, 4096, 5000, 100_000])
    def test_cap_keeps_min_points(self, length):
        spec = Downsample()
        for seed in range(30):
            plan = spec.plan(spec.draw(np.random.default_rng(seed)), length, length)
            if length < 1024:
                assert plan is None
                continue
            k = int(plan.factor)
            assert 2 <= k <= min(8, length // 512)
            assert length // k >= 512 and plan.out == length // k
            assert plan.stop - plan.start == plan.out * k and plan.stop <= length

    def test_draws_every_allowed_factor(self):
        spec = Downsample()
        ks = {
            int(spec.plan(spec.draw(np.random.default_rng(s)), 10_000, 3041).factor)
            for s in range(400)
        }
        assert ks == set(range(2, 9))

    def test_bucket_mean_of_observed(self):
        x = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 5.0, 7.0])
        y, ok = _window.bucket_mean(*_window.as_pair(x), 2)
        np.testing.assert_array_equal(y, [1.0, 3.0, np.nan, 6.0])
        np.testing.assert_array_equal(ok, [True, True, False, True])

    def test_short_series_are_skipped(self):
        df = _frame((2000, 600))
        out = to_pandas(Downsample(seed=1).apply(df, include_original=False))
        assert set(out["unique_id"].astype(str)) == {"s0_downsample_0"}
        with pytest.raises(ValueError, match="no series"):
            Downsample(seed=1).apply(_frame((600,)))

    def test_copies_end_aligned_on_source_rows(self):
        df = _frame((2000,))
        out = to_pandas(Downsample(seed=2).apply(df, include_original=False))
        n = len(out)
        assert n in (1000, 666, 500)
        np.testing.assert_array_equal(
            out["x"].to_numpy(), np.arange(2000 - n, 2000, dtype=float)
        )


class TestStretch:
    def test_up_nan_where_a_neighbour_is_nan(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        plan = _window.Resample("up", 2.0, 0, 8, 15)
        y, ok = _window.resample(*_window.as_pair(x), plan)
        t = 7 - (14 - np.arange(15)) / 2.0
        between = (np.floor(t) == 1) | (np.floor(t) == 2)
        on_knot = t == np.floor(t)
        assert np.isnan(y[between & ~on_knot]).all() and np.isnan(y[t == 2]).all()
        np.testing.assert_array_equal(
            y[on_knot & (t != 2)], x[t[on_knot & (t != 2)].astype(int)]
        )
        np.testing.assert_array_equal(ok, np.isfinite(y))

    def test_up_bridge_only_between_distinct_knots(self):
        x, ok = _window.as_pair(_series(600, 3, missing=0.0))
        plan = _window.Resample("up", 2.5, 0, 600, 1498, bridge=1.0, bridge_seed=7)
        y, _ = _window.resample(x, ok, plan)
        plain, _ = _window.resample(x, ok, _window.Resample("up", 2.5, 0, 600, 1498))
        t = 599 - (1497 - np.arange(1498)) / 2.5
        knots = np.abs(t - np.round(t)) < 1e-9
        np.testing.assert_array_equal(y[knots], plain[knots])
        assert not np.allclose(y[~knots], plain[~knots])

    def test_down_nan_without_coverage(self):
        x = np.ones(40)
        x[10:20] = np.nan
        plan = _window.Resample("down", 2.5, 0, 40, 16, 0.0)
        y, ok = _window.resample(*_window.as_pair(x), plan)
        edges = 2.5 * np.arange(17)
        empty = (edges[:-1] >= 10) & (edges[1:] <= 20)
        assert np.isnan(y[empty]).all() and np.allclose(y[ok], 1.0)

    def test_plans_respect_ranges_and_slice(self):
        spec = Stretch()
        for seed in range(200):
            plan = spec.plan(spec.draw(np.random.default_rng(seed)), 6000, 3041)
            assert plan.out <= 3041 and 0 <= plan.start < plan.stop <= 6000
            if plan.kind == "up":
                assert 1.15 <= plan.factor <= 3.0
            else:
                assert (
                    1.15 <= plan.factor <= min(6.0, 6000 / 512)
                    and 6000 / plan.factor >= 512
                )

    def test_short_series_only_stretch_up(self):
        spec = Stretch()
        kinds = {
            spec.plan(spec.draw(np.random.default_rng(s)), 500, 500).kind
            for s in range(50)
        }
        assert kinds == {"up"}


class TestPowerWarp:
    def test_only_nonnegative_prefix(self):
        x, ok = _window.as_pair(_series(900, 6, positive=False))
        assert (x[ok][:512] < 0).any()
        assert _window.power_warp(x, ok, 2.0) is None
        df = _frame((700,))
        df["y"] -= df["y"].mean()
        with pytest.raises(ValueError, match="no series"):
            PowerWarp(seed=1).apply(df)

    @pytest.mark.parametrize("p", [0.5, 0.8, 1.6, 2.5])
    def test_scale_restored(self, p):
        x, ok = _window.as_pair(_series(1500, 8))
        s = _window.mean_abs(x, ok)
        y, _ = _window.power_warp(x, ok, p)
        np.testing.assert_allclose(y[ok], s * (x[ok] / s) ** p)
        z, _ = _window.power_warp(np.array([s, 2 * s, 0.0]), np.ones(3, bool), p)
        assert z[0] == pytest.approx(s) and z[2] == 0.0

    def test_exponent_draws(self):
        spec = PowerWarp()
        ps = np.array([spec.draw(np.random.default_rng(s)) for s in range(2000)])
        convex = ps > 1
        assert (ps[convex] >= 1.25).all() and (ps[convex] <= 2.5).all()
        assert (1 / ps[~convex] >= 1.25).all() and (1 / ps[~convex] <= 2.5).all()
        assert 0.6 < convex.mean() < 0.7


class TestEnvelope:
    def test_knots_and_levels(self):
        knots, levels = np.array([0.0, 0.25, 0.5, 1.0]), np.array([1.0, 2.0, 0.5, 3.0])
        e = _window.envelope(101, 1, knots, levels)
        assert e[0] == 1.0 and e[1] == 1.0
        assert (
            e[26] == pytest.approx(2.0)
            and e[51] == pytest.approx(0.5)
            and e[100] == pytest.approx(2.95)
        )
        assert e[13] == pytest.approx(1.48)

    def test_draws(self):
        spec = AmplitudeEnvelope()
        for seed in range(50):
            knots, levels = spec.draw(np.random.default_rng(seed))
            assert (
                knots[0] == 0
                and knots[-1] == 1
                and np.all(np.diff(knots) >= 0)
                and len(knots) == 4
            )
            assert len(levels) == 4 and (levels > 0).all()

    def test_starts_at_first_observation(self):
        x = np.array([np.nan, np.nan, 2.0, 2.0, 2.0])
        y, _ = AmplitudeEnvelope().transform(
            x, None, (np.array([0.0, 0.3, 0.6, 1.0]), np.array([3.0, 1.0, 1.0, 1.0]))
        )
        assert y[2] == 6.0


class TestCensor:
    def test_floor_and_cap_at_prefix_quantile(self):
        x, ok = _window.as_pair(_series(1200, 9))
        head = np.sort(x[ok][:512])
        y, _ = _window.censor(x, ok, True, 0.25)
        level = head[round(0.25 * 511)]
        assert y[ok].min() == level and np.array_equal(y[ok], np.maximum(x[ok], level))
        y, _ = _window.censor(x, ok, False, 0.9)
        assert y[ok].max() == head[round(0.9 * 511)]

    def test_draws(self):
        spec = Censor()
        draws = [spec.draw(np.random.default_rng(s)) for s in range(500)]
        assert all(q <= 0.4 for bottom, q in draws if bottom)
        assert all(0.6 <= q <= 1.0 for bottom, q in draws if not bottom)

    def test_needs_two_observations(self):
        assert (
            Censor().transform(np.array([np.nan, 1.0, np.nan]), None, (True, 0.2))
            is None
        )


class TestApply:
    @pytest.mark.parametrize("spec", SPECS, ids=lambda s: type(s).__name__)
    def test_ids_and_originals(self, spec):
        df = _frame((2000, 1300))
        out = to_pandas(spec.apply(df, include_original=True, source_col="src"))
        ids = set(out["unique_id"].astype(str))
        assert {"s0", "s1", f"s0_{spec._suffix}_0", f"s1_{spec._suffix}_0"} <= ids
        copy = out[out["unique_id"].astype(str) == f"s0_{spec._suffix}_0"]
        assert (copy["src"].astype(str) == "s0").all()
        if not isinstance(spec, Downsample | Stretch):
            assert len(copy) == 2000

    def test_validation(self):
        with pytest.raises(ValidationError):
            Downsample(k_min=5, k_max=3)
        with pytest.raises(ValidationError):
            RealWindowAugment(censor=1.5)
        with pytest.raises(ValidationError):
            RealWindowAugment(mixup=True)
        assert not RealWindowAugment().active and FULL.active
