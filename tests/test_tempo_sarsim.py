"""Tests for the TempoPFN regime SDE and SarSim SARIMA-2 generators and their pool wiring."""

import math

import numpy as np
import pytest

from synforecast import Multivariatizer, pretraining_pool, simons_parity, tempo_sarsim
from synforecast import _tempo_sarsim as T
from synforecast.generators import SarSimGenerator, TempoSDEGenerator
from tests.helpers import Z, assert_long_format, to_pandas

GENERATORS = (TempoSDEGenerator, SarSimGenerator)


def _gen(cls, length=128, **overrides):
    kwargs = {"min_length": length, "max_length": length, "freq": "D", "seed": 3}
    kwargs.update(overrides)
    return cls(**kwargs)


def _rate_ok(count: int, n: int, p: float) -> bool:
    return abs(count / n - p) <= Z * math.sqrt(p * (1 - p) / n) + 1e-12


class TestGenerators:
    @pytest.mark.parametrize("cls", GENERATORS)
    def test_long_format(self, cls, engine: str) -> None:
        df = _gen(cls, engine=engine).generate(n_series=3, n_jobs=1)
        assert_long_format(df, n_series=3, min_length=128, max_length=128)

    @pytest.mark.parametrize("cls", GENERATORS)
    def test_seed_determinism_and_finite(self, cls) -> None:
        a = _gen(cls).generate_values(4, n_jobs=2)
        b = _gen(cls).generate_values(4, n_jobs=1)
        for x, y in zip(a, b, strict=True):
            assert x.shape == (128,)
            assert np.isfinite(x).all()
            np.testing.assert_array_equal(x, y)

    @pytest.mark.parametrize("cls", GENERATORS)
    def test_block_determinism_and_seed_sensitivity(self, cls) -> None:
        a = _gen(cls).generate_block(8, 300)
        np.testing.assert_array_equal(a, _gen(cls).generate_block(8, 300))
        assert not np.array_equal(a, _gen(cls, seed=4).generate_block(8, 300))

    @pytest.mark.parametrize("cls", GENERATORS)
    @pytest.mark.parametrize("length", [2, 3, 200, 4608])
    def test_generate_block(self, cls, length) -> None:
        block = _gen(cls).generate_block(16, length)
        assert block.shape == (16, length)
        assert np.isfinite(block).all()
        if length >= 200:
            assert (block.std(1) > 0).all()

    @pytest.mark.parametrize("cls", GENERATORS)
    def test_multivariatizer(self, cls) -> None:
        mv = Multivariatizer(base=_gen(cls), seed=1)
        df = to_pandas(mv.generate(3))
        assert df["unique_id"].nunique() == 3
        assert np.isfinite(df["y"]).all()

    def test_tempo_rejects_bad_ranges(self) -> None:
        with pytest.raises(ValueError):
            _gen(TempoSDEGenerator, hurst_min=0.9, hurst_max=0.5)
        with pytest.raises(ValueError):
            _gen(TempoSDEGenerator, stay_min=0.99, stay_max=0.9)
        with pytest.raises(ValueError):
            _gen(TempoSDEGenerator, hurst_max=1.0)
        with pytest.raises(ValueError):
            _gen(TempoSDEGenerator, dt=0.5)

    def test_sarsim_rejects_unstable_radius(self) -> None:
        with pytest.raises(ValueError):
            _gen(SarSimGenerator, ar_radius=1.0)

    def test_options_change_draws(self) -> None:
        base = _gen(TempoSDEGenerator, fbm_rate=0.0).generate_block(8, 256)
        fbm = _gen(TempoSDEGenerator, fbm_rate=1.0).generate_block(8, 256)
        sv = _gen(
            TempoSDEGenerator, fbm_rate=0.0, stochastic_vol_rate=1.0
        ).generate_block(8, 256)
        assert not np.array_equal(base, fbm)
        assert not np.array_equal(base, sv)
        assert np.isfinite(sv).all()


class TestTempoSDE:
    def test_regime_switch_rate(self) -> None:
        rng = np.random.default_rng(0)
        n, length, p = 64, 4096, 0.05
        regime = T.markov_regimes(rng, n, length, 1 - p, 1 - p)
        switches = int((np.diff(regime.astype(np.int8), axis=1) != 0).sum())
        assert _rate_ok(switches, n * (length - 1), p)

    def test_regime_durations_geometric_and_first_uniform(self) -> None:
        rng = np.random.default_rng(1)
        regime = T.markov_regimes(rng, 4000, 64, 0.9, 0.9)
        assert _rate_ok(int(regime[:, 0].sum()), 4000, 0.5)
        stays = int((regime[:, 1:] == regime[:, :-1]).sum())
        assert _rate_ok(stays, 4000 * 63, 0.9)

    @pytest.mark.parametrize("hurst", [0.3, 0.5, 0.8])
    def test_fgn_lag_one_autocorrelation(self, hurst) -> None:
        rng = np.random.default_rng(2)
        x = T.fgn(rng, np.full((64, 1), hurst), 4096)
        assert abs(x.var() - 1.0) < 0.05
        rho = (x[:, 1:] * x[:, :-1]).mean() / x.var()
        assert abs(rho - (2 ** (2 * hurst - 1) - 1)) < 0.03

    def test_fgn_variance_scaling_recovers_hurst(self) -> None:
        rng = np.random.default_rng(3)
        for hurst in (0.3, 0.8):
            path = np.cumsum(T.fgn(rng, np.full((128, 1), hurst), 4096), 1)
            lags = np.array([4, 16, 64, 256])
            var = [np.var(path[:, lag:] - path[:, :-lag]) for lag in lags]
            slope = np.polyfit(np.log(lags), np.log(var), 1)[0]
            assert abs(slope / 2 - hurst) < 0.03

    def test_hurst_draws_within_range_and_fbm_rate(self, monkeypatch) -> None:
        seen = []
        real = T.fgn

        def spy(rng, hurst, length):
            seen.append(hurst.copy())
            return real(rng, hurst, length)

        monkeypatch.setattr(T, "fgn", spy)
        _gen(
            TempoSDEGenerator, fbm_rate=0.25, hurst_min=0.4, hurst_max=0.6
        ).generate_block(4000, 16)
        hurst = np.concatenate(seen).ravel()
        assert hurst.min() >= 0.4 and hurst.max() < 0.6
        assert _rate_ok(hurst.size, 4000, 0.25)
        seen.clear()
        _gen(TempoSDEGenerator, fbm_rate=0.0).generate_block(100, 16)
        assert not seen

    def test_volatility_clustering(self) -> None:
        g = _gen(TempoSDEGenerator, fbm_rate=0.0, seed=5)
        x = g.generate_block(64, 2048)
        dx = np.diff(x, axis=1)
        dx = (dx - dx.mean(1, keepdims=True)) / dx.std(1, keepdims=True)
        sq = dx**2 - 1
        acf = (sq[:, 10:] * sq[:, :-10]).mean(1) / np.maximum((sq**2).mean(1), 1e-12)
        assert np.median(acf) > 0.02

    def test_global_affine_ranges(self) -> None:
        x = _gen(TempoSDEGenerator, seed=6).generate_block(256, 512)
        assert np.abs(x.mean(1)).max() < 100 + 50 * 40
        assert (x.std(1) > 0).all()


class TestSarSim:
    @pytest.mark.parametrize(
        ("order", "radius"), [(10, 0.9), (9, 0.9), (3, 0.9), (2, 0.1), (1, 0.1)]
    )
    def test_roots_inside_radius(self, order, radius) -> None:
        rng = np.random.default_rng(order)
        coef = T.root_polynomial(rng, np.full(200, order), max(order, 2), radius)
        for row in coef:
            inverse = 1.0 / np.roots(row[: order + 1][::-1])
            assert np.abs(inverse).max() < radius + 1e-9
            assert len(inverse) == order

    def test_zero_order_is_identity(self) -> None:
        coef = T.root_polynomial(
            np.random.default_rng(0), np.zeros(5, dtype=int), 10, 0.9
        )
        np.testing.assert_array_equal(coef, np.eye(1, 11).repeat(5, 0))

    @pytest.mark.parametrize(("p", "big_p"), [(4, 0), (0, 2), (0, 0)])
    def test_arma_matches_sarsim_recursion(self, p, big_p) -> None:
        rng = np.random.default_rng(7 + p + big_p)
        s, q, big_q, grid, steps = 5, 3, 2, 4096, 80
        phi = T.root_polynomial(rng, np.array([p]), 10, 0.9)[0]
        big_phi = T.root_polynomial(rng, np.array([big_p]), 2, 0.5)[0]
        theta = T.root_polynomial(rng, np.array([q]), 3, 0.9)[0]
        big_theta = T.root_polynomial(rng, np.array([big_q]), 2, 0.5)[0]
        impulse = np.zeros((1, grid))
        impulse[0, 0] = 1.0
        out = T.arma_filter(
            impulse,
            phi[None],
            big_phi[None],
            theta[None],
            big_theta[None],
            np.array([s]),
        )[0][:steps]
        e = np.zeros(steps)
        e[0] = 1.0
        ref = np.zeros(steps)
        for t in range(steps):
            acc = e[t]
            acc += sum(theta[i] * e[t - i] for i in range(1, 4) if t - i >= 0)
            acc += sum(
                big_theta[j] * e[t - j * s] for j in range(1, 3) if t - j * s >= 0
            )
            acc -= sum(phi[i] * ref[t - i] for i in range(1, 11) if t - i >= 0)
            acc -= sum(
                big_phi[j] * ref[t - j * s] for j in range(1, 3) if t - j * s >= 0
            )
            ref[t] = acc
        np.testing.assert_allclose(out, ref, atol=1e-8)

    @pytest.mark.parametrize("season", [0, 2, 7])
    def test_integration_matches_loop(self, season) -> None:
        rng = np.random.default_rng(13)
        length, d = 40, 0.37
        x = rng.standard_normal((1, length))
        out = T.integrate(x, np.array([[d]]), np.array([season]))[0]
        k = np.arange(1, length)
        psi = np.r_[1.0, np.cumprod((k - 1 + d) / k)]
        frac = np.array(
            [sum(psi[j] * x[0, t - j] for j in range(t + 1)) for t in range(length)]
        )
        ref = frac.copy()
        if season >= 2:
            for t in range(season, length):
                ref[t] = ref[t - season] + frac[t]
        np.testing.assert_allclose(out, ref, atol=1e-9)

    def test_integration_impulse_has_no_wraparound(self) -> None:
        x = np.zeros((1, 4))
        x[0, 3] = 1.0
        out = T.integrate(x, np.array([[0.5]]), np.array([2]))
        np.testing.assert_allclose(out, [[0.0, 0.0, 0.0, 1.0]], atol=1e-12)

    def test_warmup_grows_with_slow_seasonal_poles(self) -> None:
        none = np.zeros((1, 3))
        none[0, 0] = 1.0
        assert T.arma_warmup(np.array([52]), none, True, 0.9, 0) == T.ARMA_WARMUP
        slow = np.array([[1.0, -0.9, 0.0]])
        assert T.arma_warmup(np.array([64]), slow, False, 0.9, 0) > 64 * 40 / -math.log(
            0.9
        )
        pair = np.array([[1.0, 0.0, 0.81]])
        assert T.arma_warmup(np.array([64]), pair, False, 0.9, 0) > 64 * 40 / -math.log(
            0.9
        )

    def test_warmup_ignores_inactive_sides_and_is_capped(self) -> None:
        none = np.tile([1.0, 0.0, 0.0], (4, 1))
        assert T.arma_warmup(np.full(4, 52), none, False, 0.999, 0) == T.ARMA_WARMUP
        near = np.tile([1.0, -0.9999, 0.0], (4, 1))
        assert T.arma_warmup(np.full(4, 52), near, True, 0.9999, 0) == T.ARMA_MAX_WARMUP
        x = T.sarima_paths(np.random.default_rng(0), np.full(8, 52), 64, 0.9, 0.999)
        assert x.shape == (8, 64) and np.isfinite(x).all()

    def test_warmup_covers_long_seasonal_ma_lags(self) -> None:
        none = np.tile([1.0, 0.0, 0.0], (1, 1))
        assert T.arma_warmup(np.array([1024]), none, False, 0.9, 2048) == 2048
        near = np.tile([1.0, -0.9999, 0.0], (1, 1))
        assert T.arma_warmup(np.array([52]), near, True, 0.9999, 70000) == (
            70000 + T.ARMA_MAX_WARMUP
        )

    def test_long_seasonal_ma_has_no_wraparound(self) -> None:
        rng = np.random.default_rng(3)
        s, length = 1024, 1024
        theta = np.array([[1.0, 0.5, 0.0, 0.0]])
        big_theta = np.array([[1.0, 0.3, 0.6]])
        grid = length + T.arma_warmup(np.array([s]), np.eye(1, 3), False, 0.9, 2 * s)
        noise = rng.standard_normal((1, grid))
        y = T.arma_filter(
            noise, np.eye(1, 11), np.eye(1, 3), theta, big_theta, np.array([s])
        )
        e = noise[0]
        t = np.arange(grid - length, grid)
        ref = e[t] + 0.5 * e[t - 1] + 0.3 * e[t - s] + 0.6 * e[t - 2 * s]
        np.testing.assert_allclose(y[0, grid - length :], ref, atol=1e-8)

    def test_ar_tail_after_seasonal_ma_lag_has_no_wraparound(self) -> None:
        s, length = 1024, 1024
        phi = np.zeros((1, 11))
        phi[0, :2] = [1.0, -0.8]
        big_phi = np.eye(1, 3)
        theta = np.eye(1, 4)
        big_theta = np.array([[1.0, 0.0, 0.64]])
        warm = T.arma_warmup(np.array([s]), big_phi, True, 0.9, 2 * s)
        impulse = np.zeros((1, warm + length))
        impulse[0, -1] = 1.0
        out = T.arma_filter(impulse, phi, big_phi, theta, big_theta, np.array([s]))
        assert np.abs(out[0, warm:-1]).max() < 1e-12
        assert abs(out[0, -1] - 1.0) < 1e-12

    def test_long_season_slow_pole_variance_is_stationary(self) -> None:
        rng = np.random.default_rng(14)
        s, r = 64, 0.9
        big_phi = np.array([1.0, -r, 0.0])
        none = np.eye(1, 3)[0]
        grid = 256 + T.arma_warmup(np.array([s]), big_phi[None], False, 0.9, 0)
        noise = rng.standard_normal((512, grid))
        y = T.arma_filter(
            noise, np.eye(1, 11), big_phi[None], np.eye(1, 4), none[None], np.array([s])
        )[:, -256:]
        assert abs(y.var() / (1 / (1 - r**2)) - 1) < 0.1

    def test_seasonal_spectral_peaks(self) -> None:
        rng = np.random.default_rng(8)
        x = T.sarima_paths(rng, np.full(64, 12), 1200, 0.9, 0.1)
        dx = np.diff(x, axis=1)
        power = (np.abs(np.fft.rfft(dx - dx.mean(1, keepdims=True), axis=1)) ** 2).mean(
            0
        )
        bins = 1199 / 12
        for j in range(1, 6):
            peak = int(round(j * bins))
            near = np.r_[power[peak - 12 : peak - 3], power[peak + 4 : peak + 13]]
            assert power[peak - 1 : peak + 2].max() > 5 * np.median(near)

    def test_fractional_integration_weights(self) -> None:
        rng = np.random.default_rng(9)
        d = 0.4
        k = np.arange(1, 6)
        psi = np.r_[1.0, np.cumprod((k - 1 + d) / k)]
        ref = [
            math.gamma(j + d) / (math.gamma(d) * math.gamma(j + 1)) for j in range(6)
        ]
        np.testing.assert_allclose(psi, ref)
        assert np.isfinite(
            T.sarima_paths(rng, np.zeros(4, dtype=int), 64, 0.9, 0.1)
        ).all()

    def test_upsample_is_linear_interpolation(self) -> None:
        env = np.array([[0.0, 2.0, 4.0, 0.0]])
        out = T.upsample(env, np.array([2.0]), 6)
        np.testing.assert_allclose(out, [[0.0, 1.0, 2.0, 3.0, 4.0, 2.0]])

    def test_gamma_noiser_cv_is_level_free(self) -> None:
        rng = np.random.default_rng(15)
        x = np.tile(np.linspace(0.0, 1.0, 30000), (16, 1))
        out = T.noisers(rng, x, np.full(16, 1))
        mid, high = out[:, 9000:12000], out[:, 27000:30000]
        cv_mid = mid.std(1) / mid.mean(1)
        cv_high = high.std(1) / high.mean(1)
        np.testing.assert_allclose(cv_mid, cv_high, rtol=0.15)

    def test_noisers_level_scaled(self) -> None:
        rng = np.random.default_rng(10)
        x = np.tile(np.linspace(0.0, 1.0, 8192), (32, 1))
        for kind in (0, 1, 2):
            out = T.noisers(rng, x, np.full(32, kind))
            assert (out >= 0).all()
            low, high = out[:, :819], out[:, -819:]
            assert (high.mean(1) > low.mean(1)).mean() > 0.9

    def test_noiser_tails_heavier_than_gaussian(self) -> None:
        rng = np.random.default_rng(11)
        x = np.tile(np.sin(np.linspace(0, 20, 4096)), (64, 1))
        out = T.noisers(rng, x, np.full(64, 2))
        z = (out - out.mean(1, keepdims=True)) / out.std(1, keepdims=True)
        kurtosis = (z**4).mean(1) - 3
        assert np.median(kurtosis) > 10

    def test_passthrough_and_constant_fallback(self) -> None:
        rng = np.random.default_rng(12)
        x = np.random.default_rng(0).standard_normal((4, 64))
        np.testing.assert_array_equal(T.noisers(rng, x, np.full(4, -1)), x)
        flat = np.zeros((2, 16))
        flat[:, -1] = 1e-9
        out = T.noisers(rng, flat, np.zeros(2, dtype=int))
        assert np.isfinite(out).all()

    def test_no_noiser_no_sarima2_is_plain_sarima(self) -> None:
        g = _gen(SarSimGenerator, sarima2_rate=0.0, noiser_rate=0.0)
        x = g.generate_block(16, 512)
        assert (x < 0).any()


class TestRawKernels:
    def test_kernels_finite_without_floating_point_errors(self) -> None:
        rng = np.random.default_rng(16)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            a = T.tempo_sde(rng, 128, 2048, 0.01, 0.5, 0.3, 0.8, 0.85, 0.999, 0.5)
            b = T.sarsim(rng, 128, 2048, 0.5, 0.5, 0.75, 52, 0.9, 0.1)
        assert np.isfinite(a).all() and np.isfinite(b).all()

    def test_stochastic_vol_starts_at_regime_level(self, monkeypatch) -> None:
        seen = []
        real = T.ou_scan

        def spy(a, drive, y0):
            seen.append((a.shape, y0.copy()))
            return real(a, drive, y0)

        monkeypatch.setattr(T, "ou_scan", spy)
        T.tempo_sde(
            np.random.default_rng(17), 4, 64, 0.01, 0.0, 0.3, 0.8, 0.85, 0.999, 1.0
        )
        assert [shape for shape, _ in seen] == [(4, 63), (4, 63)]


class TestPool:
    def test_appended_after_every_existing_member(self) -> None:
        kwargs = {
            "min_length": 64,
            "max_length": 64,
            "include_extended": True,
            "include_simons_parity": True,
            "overlay_seed": 5,
            "observability_rate": 0.2,
        }
        base = pretraining_pool(**kwargs)
        full = pretraining_pool(**kwargs, include_tempo_sarsim=True)
        assert len(full) == len(base) + 2
        assert (
            tuple(type(g).__name__ for g in full[len(base) :])
            == tempo_sarsim.TEMPO_SARSIM_MEMBERS
        )
        for i in range(len(base)):
            np.testing.assert_array_equal(
                base[i].generate_values(1, n_jobs=1)[0],
                full[i].generate_values(1, n_jobs=1)[0],
            )

    def test_default_is_off(self) -> None:
        kwargs = {
            "min_length": 32,
            "max_length": 32,
            "include_extended": True,
            "overlay_seed": 1,
        }
        default = pretraining_pool(**kwargs)
        off = pretraining_pool(**kwargs, include_tempo_sarsim=False)
        assert [type(g).__name__ for g in default] == [type(g).__name__ for g in off]
        assert not {"TempoSDEGenerator", "SarSimGenerator"} & {
            type(g).__name__ for g in default
        }
        for a, b in zip(default, off, strict=True):
            np.testing.assert_array_equal(
                a.generate_values(1, n_jobs=1)[0], b.generate_values(1, n_jobs=1)[0]
            )

    def test_appends_without_simons_and_member_selection(self) -> None:
        kwargs = {
            "min_length": 64,
            "max_length": 64,
            "include_tempo_sarsim": True,
            "overlay_seed": 2,
        }
        full = pretraining_pool(**kwargs)
        assert (
            tuple(type(g).__name__ for g in full[-2:])
            == tempo_sarsim.TEMPO_SARSIM_MEMBERS
        )
        for index in (len(full) - 2, len(full) - 1):
            alone = pretraining_pool(**kwargs, members=[index])[0]
            np.testing.assert_array_equal(
                full[index].generate_values(1)[0], alone.generate_values(1)[0]
            )


class TestWeights:
    @pytest.mark.parametrize("stage", simons_parity.STAGES)
    def test_defaults(self, stage) -> None:
        w = tempo_sarsim.member_weights(stage)
        assert tuple(w) == tempo_sarsim.TEMPO_SARSIM_MEMBERS
        assert w["TempoSDEGenerator"] == (
            0.06 if stage in ("stage1", "stage2") else 0.03
        )

    def test_explicit_shares_and_removal(self) -> None:
        w = tempo_sarsim.member_weights("stage1", tempo_sde_share=0.1, sarsim_share=0.0)
        assert w == {"TempoSDEGenerator": 0.1, "SarSimGenerator": 0.0}

    def test_rejects_bad(self) -> None:
        with pytest.raises(ValueError):
            tempo_sarsim.member_weights("stage9")
        with pytest.raises(ValueError):
            tempo_sarsim.member_weights("stage1", tempo_sde_share=-0.1)
        with pytest.raises(ValueError):
            tempo_sarsim.member_weights("stage1", tempo_sde_share=0.6, sarsim_share=0.5)
