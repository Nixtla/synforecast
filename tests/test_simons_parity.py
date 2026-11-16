"""Tests for the Simons-parity generators, transform layer and pool wiring."""

import json
import math

import numpy as np
import pytest

from synforecast import Multivariatizer, SimonsParity, pretraining_pool, simons_parity
from synforecast import _families as F
from synforecast.generators import (
    AMFMWaveGenerator,
    CauKerGenerator,
    EpidemicGenerator,
    FluidQueueGenerator,
    ForecastPFNGenerator,
    OrnsteinUhlenbeckGenerator,
    PolyrhythmGenerator,
    RainfallRunoffGenerator,
    RegimeOUGenerator,
    ThermalGenerator,
)
from synforecast.simons_parity import (
    BatchAugment,
    DutyCycle,
    Gaps,
    Noisers,
    RandomUnits,
    member_weights,
    regime_ou_share,
)
from tests.helpers import Z, assert_long_format, to_pandas

FAST = (
    CauKerGenerator,
    PolyrhythmGenerator,
    AMFMWaveGenerator,
    ForecastPFNGenerator,
    RegimeOUGenerator,
    FluidQueueGenerator,
)
SIMULATED = (EpidemicGenerator, RainfallRunoffGenerator, ThermalGenerator)
SIM_KW = {"simulation_length": 256, "burn": 32}


def _gen(cls, length=128, **overrides):
    kwargs = {"min_length": length, "max_length": length, "freq": "D", "seed": 3}
    if cls in SIMULATED:
        kwargs.update(SIM_KW)
    kwargs.update(overrides)
    return cls(**kwargs)


def _rate_ok(count: int, n: int, p: float) -> bool:
    return abs(count / n - p) <= Z * math.sqrt(p * (1 - p) / n) + 1e-12


class TestGenerators:
    @pytest.mark.parametrize("cls", FAST + SIMULATED)
    def test_long_format(self, cls, engine: str) -> None:
        df = _gen(cls, engine=engine).generate(n_series=3, n_jobs=1)
        assert_long_format(df, n_series=3, min_length=128, max_length=128)

    @pytest.mark.parametrize("cls", FAST + SIMULATED)
    def test_seed_determinism_and_finite(self, cls) -> None:
        a = _gen(cls).generate_values(4, n_jobs=2)
        b = _gen(cls).generate_values(4, n_jobs=1)
        for x, y in zip(a, b, strict=False):
            assert x.shape == (128,)
            assert np.isfinite(x).all()
            np.testing.assert_array_equal(x, y)

    @pytest.mark.parametrize("cls", FAST + SIMULATED)
    def test_generate_block(self, cls) -> None:
        block = _gen(cls).generate_block(5, 200)
        assert block.shape == (5, 200)
        assert np.isfinite(block).all()
        assert (block.std(1) > 0).all()

    def test_generate_block_rejects_bad_sizes(self) -> None:
        with pytest.raises(ValueError):
            _gen(CauKerGenerator).generate_block(0, 64)

    @pytest.mark.parametrize("cls", FAST + SIMULATED)
    def test_multivariatizer(self, cls) -> None:
        mv = Multivariatizer(base=_gen(cls), seed=1)
        df = to_pandas(mv.generate(3))
        assert df["unique_id"].nunique() == 3
        assert np.isfinite(df["y"]).all()

    def test_forecast_pfn_positive(self) -> None:
        block = _gen(ForecastPFNGenerator).generate_block(200, 300)
        assert (block > 0).all()

    def test_cauker_rejects_roots_at_nodes(self) -> None:
        with pytest.raises(ValueError):
            CauKerGenerator(
                min_length=8, max_length=8, freq="D", max_nodes=4, max_roots=4
            )

    def test_simulated_length_limit(self) -> None:
        with pytest.raises(ValueError):
            EpidemicGenerator(
                min_length=8, max_length=600, freq="D", simulation_length=256
            )
        values = _gen(ThermalGenerator, length=500).generate_single_series(500)
        assert values.shape == (500,) and np.isfinite(values).all()

    def test_crop_mirrors_and_pools(self) -> None:
        path = np.arange(20.0)[:, None]
        rng = np.random.default_rng(0)
        long = F.crop(rng, path, 30)
        np.testing.assert_array_equal(
            long, np.r_[np.arange(10.0)[::-1], np.arange(20.0)]
        )
        short = F.crop(rng, path, 5)
        assert short.shape == (5,) and np.all(np.diff(short) > 0)

    def test_mechanistic_counters(self) -> None:
        simons_parity.reset_exposure_counters()
        _gen(EpidemicGenerator).generate_block(3, 64)
        counts = simons_parity.exposure_counters()["epidemic"]
        assert counts["draws"] == 3
        assert counts["redraws"] >= 0 and counts["fallbacks"] >= 0

    def test_fluid_queue_kinds_nonnegative_before_observation(self) -> None:
        raw = F.fluid_queue(np.random.default_rng(0), 50, 400)
        assert np.isfinite(raw).all() and (raw >= -1e-9).all()

    def test_ou_scan_matches_loop(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.uniform(0.4, 1.0, (3, 700))
        d = rng.normal(size=(3, 700))
        y0 = rng.normal(size=3)
        out = F.ou_scan(a, d, y0)
        ref = np.empty_like(d)
        prev = y0.copy()
        for t in range(700):
            prev = a[:, t] * prev + d[:, t]
            ref[:, t] = prev
        np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)

    def test_periodic_harmonic_cap(self) -> None:
        x = F.periodic(np.random.default_rng(0), np.array([4.0, 24.0]), 96)
        spectrum = np.abs(np.fft.rfft(x[0]))
        assert spectrum.argmax() in (24, 48)
        np.testing.assert_allclose(x.mean(1), 0, atol=1e-9)


class TestRegimeOU:
    def test_rate_zero_is_byte_identical(self) -> None:
        base = {"min_length": 64, "max_length": 64, "freq": "D", "seed": 11}
        a = OrnsteinUhlenbeckGenerator(**base).generate_values(5)
        b = OrnsteinUhlenbeckGenerator(**base, regime_rate=0.0).generate_values(5)
        for x, y in zip(a, b, strict=False):
            np.testing.assert_array_equal(x, y)

    def test_rate_one_draws_regime_paths(self) -> None:
        gen = OrnsteinUhlenbeckGenerator(
            min_length=256, max_length=256, freq="D", seed=1, regime_rate=1.0
        )
        assert gen._get_batch_params() is None
        values = gen.generate_values(3, n_jobs=1)
        assert all(np.isfinite(v).all() for v in values)

    def test_pool_rate(self) -> None:
        plain = pretraining_pool(min_length=32, max_length=32, include_extended=True)
        tuned = pretraining_pool(
            min_length=32,
            max_length=32,
            include_extended=True,
            simons_regime_ou_rate=0.5,
        )
        ou = [i for i, g in enumerate(plain) if type(g) is OrnsteinUhlenbeckGenerator]
        assert len(ou) >= 1
        assert tuned[ou[-1]].regime_rate == 0.5 and plain[ou[-1]].regime_rate == 0.0
        with pytest.raises(ValueError):
            pretraining_pool(simons_regime_ou_rate=1.5)


class TestPool:
    def test_appended_after_existing_members(self) -> None:
        kwargs = {
            "min_length": 64,
            "max_length": 64,
            "include_extended": True,
            "overlay_seed": 5,
            "observability_rate": 0.2,
        }
        base = pretraining_pool(**kwargs)
        full = pretraining_pool(**kwargs, include_simons_parity=True)
        assert len(full) == len(base) + len(simons_parity.SIMONS_MEMBERS)
        assert (
            tuple(type(g).__name__ for g in full[len(base) :])
            == simons_parity.SIMONS_MEMBERS
        )
        for i in (0, len(base) - 1):
            np.testing.assert_array_equal(
                base[i].generate_values(2, n_jobs=1)[0],
                full[i].generate_values(2, n_jobs=1)[0],
            )

    def test_member_selection_matches_full_pool(self) -> None:
        kwargs = {
            "min_length": 64,
            "max_length": 64,
            "include_extended": True,
            "include_simons_parity": True,
            "overlay_seed": 2,
        }
        full = pretraining_pool(**kwargs)
        index = len(full) - 8
        alone = pretraining_pool(**kwargs, members=[index])[0]
        np.testing.assert_array_equal(
            full[index].generate_values(1)[0], alone.generate_values(1)[0]
        )

    def test_default_pool_unchanged_and_appends_without_extended(self) -> None:
        names = [
            type(g).__name__ for g in pretraining_pool(min_length=32, max_length=32)
        ]
        with_parity = [
            type(g).__name__
            for g in pretraining_pool(
                min_length=32, max_length=32, include_simons_parity=True
            )
        ]
        assert with_parity[: len(names)] == names
        assert tuple(with_parity[len(names) :]) == simons_parity.SIMONS_MEMBERS


class TestWeights:
    def test_stage_one_and_two(self) -> None:
        w = member_weights("stage1")
        assert w == member_weights("stage2")
        assert tuple(w) == simons_parity.SIMONS_MEMBERS
        assert sum(w.values()) == pytest.approx(0.2264, abs=1e-12)
        assert w["CauKerGenerator"] == pytest.approx(0.08016, abs=1e-12)

    @pytest.mark.parametrize("stage", ["stage3", "stage4", "stage4-long"])
    def test_late_stages(self, stage) -> None:
        w = member_weights(stage)
        assert sum(w.values()) == pytest.approx(0.20, abs=1e-12)
        assert w["CauKerGenerator"] == pytest.approx(0.04 / 0.5, abs=1e-12)
        assert w["EpidemicGenerator"] == 0.0
        assert regime_ou_share(stage) == pytest.approx(0.12, abs=1e-12)

    def test_unknown_stage(self) -> None:
        with pytest.raises(ValueError):
            member_weights("stage9")
        with pytest.raises(ValueError):
            SimonsParity.from_stage("stage9")


def _batch(n=400, length=96, seed=0, nan_share=0.1):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=(n, length)), 1) * rng.uniform(
        0.1, 100, (n, 1)
    ) + rng.normal(0, 50, (n, 1))
    x[rng.random((n, length)) < nan_share] = np.nan
    return x


class TestTransformInvariants:
    def test_standardise_on_finite_values(self) -> None:
        x = _batch()
        z = simons_parity.standardise(x)
        np.testing.assert_array_equal(np.isnan(z), np.isnan(x))
        np.testing.assert_allclose(np.nanmean(z, 1), 0, atol=1e-9)
        np.testing.assert_allclose(np.nanstd(z, 1, ddof=1), 1, atol=1e-9)

    def test_batch_augment_keeps_masks(self) -> None:
        x = _batch()
        out, counts = SimonsParity().batch_augment(x, None, np.random.default_rng(1))
        np.testing.assert_array_equal(np.isnan(out), np.isnan(x))
        assert np.isfinite(out[~np.isnan(x)]).all()
        np.testing.assert_allclose(np.nanmean(out, 1), 0, atol=1e-9)
        assert counts["series"] == len(x)

    def test_batch_augment_off_returns_standardised(self) -> None:
        x = _batch()
        out, counts = SimonsParity(batch=None).batch_augment(
            x, None, np.random.default_rng(1)
        )
        np.testing.assert_array_equal(out, simons_parity.standardise(x), strict=False)
        assert counts["t1_active"] == 0

    def test_batch_group_sharing(self) -> None:
        x = _batch(n=320, nan_share=0.0)
        group = np.repeat(np.arange(40), 8)
        _, chosen = BatchAugment(rate=0.5).apply(x, group, np.random.default_rng(2))
        for g in range(40):
            rows = chosen[group == g]
            assert (rows == rows[0]).all()

    def test_series_transforms_masks_and_order(self) -> None:
        x = _batch(nan_share=0.05)
        parity = SimonsParity(gaps=None)
        out, counts = parity.series_transforms(x, None, np.random.default_rng(3))
        flipped_mask = np.isnan(x)
        rng = np.random.default_rng(3)
        flips = rng.random(len(x)) < parity.units.time_flip
        expected = np.where(flips[:, None], flipped_mask[:, ::-1], flipped_mask)
        np.testing.assert_array_equal(np.isnan(out), expected)
        assert counts["t3_time_flip"] == int(flips.sum())

    def test_gaps_keep_last_patch_and_only_add_nan(self) -> None:
        x = _batch(n=2000, length=200, nan_share=0.0)
        out, holed = Gaps(rate=1.0).apply(x, np.random.default_rng(4))
        assert not np.isnan(out[:, -32:]).any()
        assert np.isnan(out).any()
        np.testing.assert_array_equal(out[~np.isnan(out)], x[~np.isnan(out)])
        assert holed.all()

    def test_time_flip_shared_in_group(self) -> None:
        x = _batch(n=200, nan_share=0.0)
        group = np.repeat(np.arange(50), 4)
        _, flipped = RandomUnits(time_flip=0.5).flip(x, group, np.random.default_rng(5))
        assert (flipped.reshape(50, 4) == flipped.reshape(50, 4)[:, :1]).all()

    def test_count_rows_bypass_units_and_duty(self) -> None:
        z = simons_parity.standardise(_batch(n=3000, nan_share=0.05))
        x, is_count, kind = Noisers(rate=1.0).apply(z, np.random.default_rng(6))
        rows = is_count[:, 0]
        assert rows.any()
        counts = x[rows]
        finite = counts[np.isfinite(counts)]
        np.testing.assert_array_equal(finite, np.round(finite))
        assert (finite >= 0).all()
        units, _ = RandomUnits().apply(x, is_count, np.random.default_rng(7))
        np.testing.assert_array_equal(units[rows], x[rows])
        duty, duty_rows, _ = DutyCycle(rate=1.0).apply(
            units, is_count, np.random.default_rng(8)
        )
        np.testing.assert_array_equal(duty[rows], x[rows])
        assert not duty_rows[rows].any() and duty_rows[~rows].all()

    def test_duty_rows_gate_to_zero(self) -> None:
        x = simons_parity.standardise(_batch(n=300, nan_share=0.0))
        out, rows, _ = DutyCycle(rate=1.0).apply(
            x, np.zeros((300, 1), bool), np.random.default_rng(9)
        )
        assert rows.all()
        assert (out >= -1e-12).all()
        assert ((out == 0).mean(1) > 0).mean() > 0.9

    def test_lattice_quantises_to_grid(self) -> None:
        z = simons_parity.standardise(_batch(n=50, nan_share=0.0))
        rng = np.random.default_rng(10)
        out = BatchAugment._lattice(
            z, np.isfinite(z), lambda v: v, rng, z.shape[1], np.arange(len(z))
        )
        distinct = [len(np.unique(np.round(row, 9))) for row in out]
        assert np.median(distinct) < z.shape[1]

    def test_filter_diff_reads_last_observed(self) -> None:
        z = np.array([[0.0, 1.0, np.nan, 3.0]])
        held = simons_parity._forward_fill(np.nan_to_num(z), np.isfinite(z))
        np.testing.assert_array_equal(held, [[0.0, 1.0, 1.0, 3.0]])

    def test_infinite_input_becomes_missing(self) -> None:
        x = _batch(n=20, nan_share=0.0)
        x[0, 5] = np.inf
        out, _ = SimonsParity.from_stage("stage3").series_transforms(
            x, None, np.random.default_rng(11)
        )
        assert np.isfinite(out[np.isfinite(out)]).all()
        assert np.isnan(out).sum() >= 1

    def test_deterministic(self) -> None:
        x = _batch()
        parity = SimonsParity()
        a = parity.series_transforms(
            parity.batch_augment(x, None, np.random.default_rng(1))[0],
            None,
            np.random.default_rng(2),
        )
        b = parity.series_transforms(
            parity.batch_augment(x, None, np.random.default_rng(1))[0],
            None,
            np.random.default_rng(2),
        )
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]

    def test_rates_json(self) -> None:
        late = SimonsParity.from_stage("stage4-long")
        assert late.duty is None and SimonsParity.from_stage("stage2").duty is not None
        dumped = json.loads(json.dumps(late.rates()))
        assert dumped["version"] == simons_parity.PARITY_VERSION
        assert dumped["duty"] is None and dumped["batch"]["rate"] == 0.4

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError):
            SimonsParity().batch_augment(np.zeros(5), None, np.random.default_rng(0))
        with pytest.raises(ValueError):
            SimonsParity().series_transforms(
                np.zeros((4, 8)), np.zeros(3, int), np.random.default_rng(0)
            )


class TestTransformRates:
    N = 20000

    def test_stage_one_rates(self) -> None:
        x = _batch(n=self.N, length=64, nan_share=0.0)
        parity = SimonsParity.from_stage("stage1")
        y, t1 = parity.batch_augment(x, None, np.random.default_rng(20))
        _, post = parity.series_transforms(y, None, np.random.default_rng(21))
        n = self.N
        assert _rate_ok(t1["t1_active"], n, 0.4)
        assert _rate_ok(t1["t1_two"], n, 0.4 * 0.3)
        assert _rate_ok(post["t3_time_flip"], n, 0.25)
        assert _rate_ok(post["t2_active"], n, 0.3)
        for name in simons_parity.NOISE_KINDS:
            assert _rate_ok(post[f"t2_{name}"], n, 0.075)
        non_count = n - post["t2_count"] - post["t2_count_lognormal"]
        assert _rate_ok(post["t3_sign_flip"], non_count, 0.25)
        assert _rate_ok(post["t3_positive"], non_count, 0.3)
        assert _rate_ok(post["t4_duty"], non_count, 0.1)
        assert _rate_ok(post["t4_duty_periodic"], non_count, 0.05)
        assert _rate_ok(post["t3_gaps"], n, 0.1)

    def test_category_marginals_follow_gumbel_top_k(self) -> None:
        _, chosen = BatchAugment(rate=1.0, second_rate=0.0).apply(
            _batch(n=self.N, length=32, nan_share=0.0), None, np.random.default_rng(22)
        )
        for col, p in enumerate((0.35, 0.25, 0.2, 0.2)):
            assert _rate_ok(int(chosen[:, col].sum()), self.N, p)
        assert (chosen.sum(1) == 1).all()

    def test_spike_rate(self) -> None:
        z = simons_parity.standardise(_batch(n=2000, length=500, nan_share=0.0))
        noisers = Noisers(rate=1.0)
        out, _, kind = noisers.apply(z, np.random.default_rng(23))
        rows = kind == 3
        changed = (out[rows] != z[rows]).mean()
        assert abs(changed - 0.01) < 0.002


# Pure-Python reference recurrences of the mechanistic simulators (the native loops must match them exactly).
def _epi_loop(x, beta, p_const, p_birth, pop, imports, z, burn, length):
    s_, e_, i_, r_ = x
    p_e, p_i, p_w = p_const
    keep = 1 - p_birth
    born = p_birth * pop
    inc = [0.0] * length
    prev = [0.0] * length
    probs = (0.0, p_e, p_i, p_w)
    for step in range(burn + length):
        p_s = 1 - math.exp(-beta[step] * i_)
        zs = z[step]
        flows = []
        for value, p, zk in zip((s_, e_, i_, r_), (p_s,) + probs[1:], zs, strict=False):
            mean = value * p
            spread = math.sqrt(max(mean * (1 - p), 0.0))
            flows.append(min(max(mean + spread * zk, 0.0), value))
        f_s, f_e, f_i, f_r = flows
        s_, e_, i_, r_ = s_ - f_s + f_r, e_ - f_e + f_s, i_ - f_i + f_e, r_ - f_r + f_i
        s_, e_, i_, r_ = (
            s_ * keep + born,
            e_ * keep + imports[step],
            i_ * keep,
            r_ * keep,
        )
        if step >= burn:
            inc[step - burn] = f_e
            prev[step - burn] = i_
    return inc, prev


def _hydro_loop(state, params, series, burn, length):
    soil, smax, k, a, alpha, fg, g, p11 = state + params
    occ, p01_t, amount, snow_frac, melt_rate, pet = series
    snow = route = ground = 0.0
    wet = False
    flow = [0.0] * length
    moist = [0.0] * length
    for step in range(burn + length):
        wet = occ[step] < p11 if wet else occ[step] < p01_t[step]
        rain = amount[step] if wet else 0.0
        sf = snow_frac[step]
        melt = min(snow, melt_rate[step])
        snow = snow + rain * sf - melt
        soil = soil + rain * (1 - sf) + melt
        soil = max(soil - pet[step] * min(soil / smax, 1.0), 0.0)
        over = max(soil - smax, 0.0)
        soil = soil - over
        q = min(k * smax * (soil / smax) ** a, soil)
        soil = soil - q
        ground = ground + fg * q - g * ground
        route = route + alpha * ((1 - fg) * q + over - route)
        if step >= burn:
            flow[step - burn] = route + g * ground
            moist[step - burn] = soil + snow
    return flow, moist


def _thermal_loop(t1, params, ambient, heat, burn, length):
    r12, r1a, r2a, cap, setpoint, band, power, prop = params
    t2 = t1
    on = False
    temp = [0.0] * length
    energy = [0.0] * length
    for step in range(burn + length):
        on = (t1 > setpoint + band) or (on and t1 > setpoint - band)
        if prop:
            cool = power * min(max((t1 - setpoint + band) / (2 * band), 0.0), 1.0)
        else:
            cool = power if on else 0.0
        amb = ambient[step]
        h = heat[step]
        t1, t2 = (
            t1 + r12 * (t2 - t1) + r1a * (amb - t1) + h - cool,
            t2 + r12 * cap * (t1 - t2) + r2a * (amb - t2),
        )
        if step >= burn:
            temp[step - burn] = t1
            energy[step - burn] = cool + h
    return temp, energy


class TestNativeMechanistic:
    def test_epidemic_matches_reference(self) -> None:
        from synforecast._lib import mechanistic

        rng = np.random.default_rng(0)
        for _ in range(5):
            burn, length = 37, 211
            total = burn + length
            pop = float(F.logu(rng, (), 1e4, 1e7))
            x0 = (rng.dirichlet(np.ones(4)) * pop).tolist()
            beta = rng.uniform(0, 3, total) / pop
            p_const = rng.uniform(0.01, 0.99, 3).tolist()
            imports = rng.poisson(0.5, total).astype(float)
            z = rng.normal(size=(total, 4))
            ref = _epi_loop(
                x0,
                beta.tolist(),
                p_const,
                0.001,
                pop,
                imports.tolist(),
                z.tolist(),
                burn,
                length,
            )
            out = mechanistic.epidemic(
                x0, beta, p_const, 0.001, pop, imports, z.reshape(-1), burn, length
            )
            np.testing.assert_array_equal(out[0], ref[0])
            np.testing.assert_array_equal(out[1], ref[1])

    def test_rainfall_runoff_matches_reference(self) -> None:
        from synforecast._lib import mechanistic

        rng = np.random.default_rng(1)
        burn, length = 50, 300
        total = burn + length
        params = [120.0, 0.05, 2.2, 0.4, 0.3, 0.004, 0.7]
        series = [
            rng.random(total),
            rng.uniform(0.01, 0.3, total),
            rng.gamma(0.8, 5.0, total),
            rng.random(total) * (rng.random(total) < 0.5),
            rng.uniform(0, 3, total),
            rng.uniform(0, 2, total),
        ]
        ref = _hydro_loop([60.0], params, [v.tolist() for v in series], burn, length)
        out = mechanistic.rainfall_runoff(60.0, params, *series, burn, length)
        np.testing.assert_array_equal(out[0], ref[0])
        np.testing.assert_array_equal(out[1], ref[1])

    @pytest.mark.parametrize("proportional", [False, True])
    def test_thermal_matches_reference(self, proportional) -> None:
        from synforecast._lib import mechanistic

        rng = np.random.default_rng(2)
        burn, length = 40, 260
        total = burn + length
        ambient = 10 + 5 * rng.normal(size=total).cumsum() / 10
        heat = rng.uniform(0, 0.5, total)
        params = [0.1, 0.02, 0.01, 0.5, 20.0, 1.5, 0.8]
        ref = _thermal_loop(
            15.0, params + [proportional], ambient.tolist(), heat.tolist(), burn, length
        )
        out = mechanistic.thermal(
            15.0, params, proportional, ambient, heat, burn, length
        )
        np.testing.assert_array_equal(out[0], ref[0])
        np.testing.assert_array_equal(out[1], ref[1])

    def test_rejects_short_inputs(self) -> None:
        from synforecast._lib import mechanistic

        with pytest.raises(ValueError):
            mechanistic.thermal(1.0, [0.1] * 7, False, np.zeros(5), np.zeros(5), 3, 4)

    def test_float16_storage(self) -> None:
        values = _gen(EpidemicGenerator).generate_block(4, 128)
        assert np.isfinite(values).all()


class TestKernelEquivalence:
    def test_periodic_matches_direct_formula(self) -> None:
        period = np.array([7.0, 24.0, 365.0, 3.3, 100.5])
        x = F.periodic(np.random.default_rng(1), period, 3000)
        r = np.random.default_rng(1)
        k = np.arange(1, 13.0)
        smooth = F.u(r, (5, 1), 1, 6)
        var = np.exp(-(k**2) / (2 * smooth**2)) * (
            k[None] <= np.floor(period[:, None] / 2)
        )
        a = F.randn(r, (5, 12)) * np.sqrt(var)
        b = F.randn(r, (5, 12)) * np.sqrt(var)
        t = np.arange(3000.0)
        phase = (
            2
            * np.pi
            * np.remainder(t[None, None] * k[None, :, None] / period[:, None, None], 1)
        )
        ref = F.standardise(
            (a[..., None] * np.cos(phase) + b[..., None] * np.sin(phase)).sum(1)
        )
        np.testing.assert_allclose(x, ref, atol=1e-9)

    def test_filter_branches_match_loop_formulas(self) -> None:
        z = simons_parity.standardise(_batch(n=64, length=120, nan_share=0.05))
        z = np.where(np.isfinite(z), z, 0.0)
        observed = np.isfinite(_batch(n=64, length=120, nan_share=0.05))
        idx = np.arange(64)
        spec = BatchAugment()
        out = spec._filter(z, observed, lambda v: v, np.random.default_rng(3), 120, idx)
        r = np.random.default_rng(3)
        taps = spec.taps
        size = r.integers(2, taps + 1, (64, 1))
        live = np.arange(taps)[None] < size
        smooth_w = -np.log(np.maximum(r.random((64, taps)), 1e-12)) * live
        smooth_w = smooth_w / smooth_w.sum(-1, keepdims=True)
        signed_w = r.standard_normal((64, taps)) * live
        signed_w = signed_w / np.maximum(np.abs(signed_w).sum(-1, keepdims=True), 1e-6)
        w = np.where(r.random((64, 1)) < 0.7, smooth_w, signed_w)
        kind = r.random((64, 1))
        first = simons_parity._first_observed(z, observed)
        padded = np.concatenate([np.repeat(first, taps - 1, 1), z], -1)
        fir = sum(
            w[:, k : k + 1] * padded[:, taps - 1 - k : taps - 1 - k + 120]
            for k in range(taps)
        )
        held = simons_parity._forward_fill(z, observed)
        diff = np.concatenate([np.zeros((64, 1)), held[:, 1:] - held[:, :-1]], -1)
        ref = np.where(kind < 0.15, diff, np.where(kind < 0.3, np.cumsum(z, -1), fir))
        np.testing.assert_allclose(out, ref, atol=1e-12)

    def test_lattice_floor_is_sorted_order_statistic(self) -> None:
        z = simons_parity.standardise(_batch(n=40, length=90, nan_share=0.1))
        observed = np.isfinite(z)
        filled = np.where(observed, z, 0.0)
        out = BatchAugment._lattice(
            filled, observed, lambda v: v, np.random.default_rng(4), 90, np.arange(40)
        )
        r = np.random.default_rng(4)
        delta = F.logu(r, (40, 1), 0.05, 0.5)
        offset = r.random((40, 1))
        q = F.u(r, (40, 1), 0.05, 0.5)
        censor = r.random((40, 1)) < 0.4
        ordered = np.sort(np.where(observed, filled, np.inf), -1)
        position = np.minimum(
            (q * (observed.sum(-1, keepdims=True) - 1)).astype(int), 89
        )
        floor = np.take_along_axis(ordered, position, -1)
        ref = np.where(
            censor, np.maximum(filled, floor), delta * np.floor(filled / delta + offset)
        )
        np.testing.assert_array_equal(out, ref)


class TestNativeHook:
    def _tcm(self, **overrides):
        from synforecast.generators import TCMGenerator

        return TCMGenerator(
            min_length=256, max_length=256, freq="h", seed=5, **overrides
        )

    def test_native_series_matches_batch_kernel_and_trace(self) -> None:
        from synforecast._lib import batch

        gen = self._tcm()
        seed = int(
            np.random.default_rng(5).integers(0, 2**63, size=1, dtype=np.uint64)[0]
        )
        values = gen.generate_native_series(256)
        scalars, arrays = gen._get_batch_params()
        traced, info = batch.tcm_trace(256, scalars, arrays[0], arrays[1], seed)
        np.testing.assert_array_equal(values, traced)
        assert info["n_vars"] >= 1 and info["rescale_branch"] in (0, 1, 2, 3)
        assert info["redraws"] >= 0 and isinstance(info["fallback"], bool)

    def test_native_series_seeded(self) -> None:
        a = self._tcm().generate_native_series(128)
        b = self._tcm().generate_native_series(128)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (128,) and np.isfinite(a).all()

    def test_native_series_unavailable(self) -> None:
        with pytest.raises(NotImplementedError):
            self._tcm(multivariate=True).generate_native_series(64)
        with pytest.raises(NotImplementedError):
            _gen(CauKerGenerator).generate_native_series(64)

    def test_multivariatizer_default_unchanged_and_native_opt_in(self) -> None:
        plain = Multivariatizer(base=self._tcm(), seed=3).generate(3)
        explicit = Multivariatizer(
            base=self._tcm(), seed=3, native_bases=False
        ).generate(3)
        assert to_pandas(plain).equals(to_pandas(explicit))
        native = to_pandas(
            Multivariatizer(base=self._tcm(), seed=3, native_bases=True).generate(3)
        )
        assert np.isfinite(native["y"]).all()
        assert not native["y"].equals(to_pandas(plain)["y"])
