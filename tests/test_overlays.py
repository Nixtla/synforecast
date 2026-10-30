"""Relative overlay magnitudes, the pretraining overlay prior and the extended pretraining pool."""

import numpy as np
import pytest

from synforecast import calendar_overlays, pretraining_overlays, pretraining_pool
from synforecast._core import _overlay_scale
from synforecast._lib import pattern_injection as rs_pi

EMPTY = np.array([], dtype=np.float64)


def _series(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n)) + 0.1 * np.arange(n)


@pytest.mark.parametrize("factor", [1e-3, 1.0, 250.0])
def test_relative_anomalies_scale_with_the_series(factor: float) -> None:
    base = _series()
    out = []
    for values in (base.copy(), factor * base):
        injected, _ = rs_pi.add_anomalies(
            values,
            7,
            ["spike", "dip", "level_shift"],
            0.03,
            5.0,
            -5.0,
            2.0,
            10,
            _overlay_scale(values),
        )
        out.append(injected)
    np.testing.assert_allclose(out[1], factor * out[0], rtol=1e-12, atol=1e-12 * factor)


@pytest.mark.parametrize("kind", ["level", "trend", "variance", "mixed"])
def test_relative_changepoints_scale_with_the_series(kind: str) -> None:
    base = _series(seed=3)
    out = []
    for values in (base.copy(), 40.0 * base):
        injected, _ = rs_pi.add_changepoints(
            values,
            11,
            2,
            EMPTY,
            kind,
            EMPTY,
            EMPTY,
            EMPTY,
            True,
            _overlay_scale(values),
        )
        out.append(injected)
    np.testing.assert_allclose(out[1], 40.0 * out[0], rtol=1e-10, atol=1e-9)


def test_relative_trend_change_is_total_drift_in_std_units() -> None:
    values = np.zeros(200)
    values[::2] = 1.0
    scale = _overlay_scale(values)
    injected, meta = rs_pi.add_changepoints(
        values.copy(),
        0,
        1,
        np.array([0.5]),
        "trend",
        EMPTY,
        np.array([2.0]),
        EMPTY,
        True,
        scale,
    )
    pos = int(meta["changepoint_indices"][0])
    drift = (injected - values)[-1]
    assert drift == pytest.approx(
        2.0 * scale * (len(values) - pos - 1) / (len(values) - pos)
    )


def test_absolute_mode_is_the_default_and_unscaled() -> None:
    values = _series()
    a, _ = rs_pi.add_anomalies(values.copy(), 3, ["spike"], 0.05, 10.0, -10.0, 20.0, 10)
    b, _ = rs_pi.add_anomalies(
        values.copy(), 3, ["spike"], 0.05, 10.0, -10.0, 20.0, 10, 1.0
    )
    np.testing.assert_array_equal(a, b)
    assert sorted(set(np.round(a - values, 9)) - {0.0}) == [10.0]


@pytest.mark.parametrize(
    "values", [np.full(50, 3.0), np.array([1.0]), np.array([np.nan, 2.0])]
)
def test_overlay_scale_falls_back_to_one(values: np.ndarray) -> None:
    assert _overlay_scale(values) == 1.0


def test_overlay_prior_is_deterministic_and_documented_rates() -> None:
    first = [pretraining_overlays(np.random.default_rng([5, i])) for i in range(50)]
    again = [pretraining_overlays(np.random.default_rng([5, i])) for i in range(50)]
    assert first == again
    draws = [pretraining_overlays(np.random.default_rng([9, i])) for i in range(20000)]
    rates = {
        "innovation_distribution": 0.3,
        "changepoints": 0.3,
        "anomalies": 0.25,
        "missing_data": 0.1,
    }
    for key, rate in rates.items():
        assert np.mean([key in d for d in draws]) == pytest.approx(rate, abs=0.015)
    assert all(d.get("overlay_scale", "relative") == "relative" for d in draws)
    assert any(not d for d in draws)


def test_member_built_alone_matches_full_pool_with_overlays() -> None:
    kwargs = {
        "min_length": 64,
        "max_length": 64,
        "freq": "D",
        "seed": 4,
        "include_extended": True,
        "overlay_seed": 123,
    }
    full = pretraining_pool(**kwargs)
    for i in (0, 17, 45, 50, 60, len(full) - 1):
        (alone,) = pretraining_pool(**kwargs, members=[i])
        assert (
            type(alone) is type(full[i]) and alone.model_dump() == full[i].model_dump()
        )


def test_extended_entries_are_appended() -> None:
    plain = pretraining_pool(min_length=32, max_length=32, freq="h", seed=1)
    extended = pretraining_pool(
        min_length=32, max_length=32, freq="h", seed=1, include_extended=True
    )
    assert len(extended) == len(plain) + 11
    assert [g.model_dump() for g in extended[: len(plain)]] == [
        g.model_dump() for g in plain
    ]


def test_entries_with_own_changepoints_keep_them() -> None:
    kwargs = {
        "min_length": 96,
        "max_length": 96,
        "freq": "MS",
        "seed": 2,
        "include_seasonal": True,
    }
    plain = pretraining_pool(**kwargs)
    own = [i for i, g in enumerate(plain) if g.changepoints or g.anomalies]
    assert own
    for overlay_seed in range(40):
        drawn = pretraining_pool(**kwargs, overlay_seed=overlay_seed, members=own)
        for g, ref in zip(drawn, (plain[i] for i in own), strict=True):
            assert g.overlay_scale == "absolute"
            assert g.changepoint_trend_changes == ref.changepoint_trend_changes
            assert g.anomalies == ref.anomalies


@pytest.mark.parametrize(
    "freq,length", [("D", 400), ("h", 1024), ("YS", 2273), ("MS", 128)]
)
def test_every_member_draws_valid_series_with_overlays(freq: str, length: int) -> None:
    n = len(
        pretraining_pool(min_length=8, max_length=8, freq=freq, include_extended=True)
    )
    rng = np.random.default_rng(0)
    for draw in range(3 * n):
        (gen,) = pretraining_pool(
            min_length=length,
            max_length=length,
            freq=freq,
            seed=int(rng.integers(1 << 40)),
            include_extended=True,
            overlay_seed=int(rng.integers(1 << 40)),
            members=[draw % n],
        )
        (values,) = gen.generate_values(1, n_jobs=1)
        assert values.shape == (length,) and not np.isinf(values).any()
        assert np.isfinite(values).sum() >= 2, (draw % n, type(gen).__name__)


@pytest.mark.parametrize("kind", ["level", "trend", "variance", "mixed"])
def test_relative_overlays_stay_on_the_series_scale(kind: str) -> None:
    overlays = {
        "changepoints": True,
        "num_changepoints": 3,
        "changepoint_type": kind,
        "anomalies": True,
        "anomaly_types": ["spike", "dip", "level_shift"],
        "anomaly_fraction": 0.03,
        "spike_magnitude": 10.0,
        "dip_magnitude": -10.0,
        "level_shift_magnitude": 4.0,
        "level_shift_duration": 50,
        "overlay_scale": "relative",
    }
    n = len(
        pretraining_pool(min_length=8, max_length=8, freq="D", include_extended=True)
    )
    for member in range(n):
        kwargs = {
            "min_length": 512,
            "max_length": 512,
            "freq": "D",
            "seed": 11,
            "include_extended": True,
            "members": [member],
        }
        (base,) = pretraining_pool(**kwargs)[0].generate_values(1, n_jobs=1)
        (both,) = pretraining_pool(**kwargs, **overlays)[0].generate_values(1, n_jobs=1)
        sigma = _overlay_scale(base)
        if np.nanmax(np.abs(base)) >= 1e11:
            continue
        # Spikes reach 10 sigma, drift and level changes 3 sigma each, variance changes at most 2x per break.
        assert np.nanmax(np.abs(both - base)) <= 200 * sigma, (member, kind)


@pytest.mark.parametrize("freq", ["h", "D", "MS"])
def test_extended_entries_stay_bounded_on_long_series(freq: str) -> None:
    n = len(pretraining_pool(min_length=8, max_length=8, freq=freq))
    rng = np.random.default_rng(1)
    for member in range(n, n + 11):
        for _ in range(25):
            (gen,) = pretraining_pool(
                min_length=2273,
                max_length=2273,
                freq=freq,
                seed=int(rng.integers(1 << 40)),
                include_extended=True,
                members=[member],
            )
            (values,) = gen.generate_values(1, n_jobs=1)
            assert np.nanmax(np.abs(values)) < 1e8, (member, type(gen).__name__)


@pytest.mark.parametrize("scale", ["absolute", "relative"])
def test_variance_changepoints_keep_sensor_gaps_local(scale: str) -> None:
    # IoTSensor series have failure gaps; variance changepoints must not turn the rest of the series into NaN.
    member = next(
        i
        for i, g in enumerate(pretraining_pool(min_length=8, max_length=8, freq="D"))
        if type(g).__name__ == "IoTSensorGenerator"
    )
    rng = np.random.default_rng(2)
    for _ in range(60):
        kwargs = {
            "min_length": 128,
            "max_length": 128,
            "freq": "D",
            "seed": int(rng.integers(1 << 40)),
            "members": [member],
        }
        (base,) = pretraining_pool(**kwargs)[0].generate_values(1, n_jobs=1)
        (both,) = pretraining_pool(
            **kwargs,
            changepoints=True,
            num_changepoints=3,
            changepoint_type="variance",
            overlay_scale=scale,
        )[0].generate_values(1, n_jobs=1)
        np.testing.assert_array_equal(np.isnan(both), np.isnan(base))


def _calendar_values(**kwargs):
    from synforecast.generators import RandomWalkGenerator

    settings = {"min_length": 24 * 21, "max_length": 24 * 21, "freq": "h", "seed": 3}
    settings.update(kwargs)
    (values,) = RandomWalkGenerator(**settings).generate_values(1, n_jobs=1)
    return values


def test_calendar_off_draws_exactly_as_before() -> None:
    base = _calendar_values()
    np.testing.assert_array_equal(base, _calendar_values(calendar_amplitude=3.0))
    kwargs = {"min_length": 64, "max_length": 64, "freq": "h", "seed": 4}
    kwargs.update(include_extended=True, overlay_seed=9)
    plain = pretraining_pool(**kwargs)
    zero = pretraining_pool(**kwargs, calendar_rate=0.0)
    assert [g.model_dump() for g in plain] == [g.model_dump() for g in zero]


def test_additive_daily_cycle_is_periodic_with_the_given_peak() -> None:
    base = _calendar_values()
    cycled = _calendar_values(calendar_cycles=True, calendar_amplitude=2.0)
    added = cycled - base
    np.testing.assert_allclose(added[24:], added[:-24], atol=1e-9)
    assert np.abs(added).max() == pytest.approx(2.0 * _overlay_scale(base), rel=0.01)
    assert abs(added[:24].mean()) < 1e-9


def test_calendar_cycle_follows_the_timestamps() -> None:
    kwargs = {"calendar_cycles": True, "calendar_weekly": 0.7}
    midnight = _calendar_values(**kwargs) - _calendar_values()
    shifted = _calendar_values(start_datetime="2000-01-01 05:00", **kwargs)
    shifted -= _calendar_values(start_datetime="2000-01-01 05:00")
    scale = _overlay_scale(_calendar_values())
    np.testing.assert_allclose(shifted[:-5] / scale, midnight[5:] / scale, atol=1e-9)


def test_weekly_cycle_on_daily_data_and_none_on_monthly() -> None:
    daily = {"min_length": 70, "max_length": 70, "freq": "D"}
    added = _calendar_values(**daily, calendar_cycles=True, calendar_weekly=1.0)
    added -= _calendar_values(**daily)
    np.testing.assert_allclose(added[7:], added[:-7], atol=1e-9)
    assert np.abs(added).max() > 0
    np.testing.assert_array_equal(
        _calendar_values(**daily, calendar_cycles=True), _calendar_values(**daily)
    )
    monthly = {"min_length": 48, "max_length": 48, "freq": "MS"}
    np.testing.assert_array_equal(
        _calendar_values(**monthly, calendar_cycles=True, calendar_weekly=1.0),
        _calendar_values(**monthly),
    )


def test_multiplicative_cycle_scales_positive_series_only() -> None:
    from synforecast.generators import PoissonProcessGenerator

    kwargs = {"min_length": 24 * 14, "max_length": 24 * 14, "freq": "h", "seed": 5}
    (counts,) = PoissonProcessGenerator(**kwargs).generate_values(1, n_jobs=1)
    counts = counts + 1.0
    from synforecast.base import BaseGenerator

    class Positive(BaseGenerator):
        def generate_single_series(self, length: int) -> np.ndarray:
            return counts[:length].copy()

    mult = {"calendar_cycles": True, "calendar_mode": "multiplicative"}
    (scaled,) = Positive(**kwargs, **mult, calendar_swing=0.6).generate_values(
        1, n_jobs=1
    )
    ratio = scaled / counts
    np.testing.assert_allclose(ratio[24:], ratio[:-24], rtol=1e-9)
    assert ratio.min() >= 0.4 - 1e-9 and ratio.max() <= 1.6 + 1e-9
    base = _calendar_values()
    fallback = _calendar_values(**mult, calendar_amplitude=2.5, calendar_swing=0.6)
    additive = _calendar_values(calendar_cycles=True, calendar_amplitude=2.5)
    assert base.min() <= 0
    np.testing.assert_array_equal(fallback, additive)
    with pytest.raises(ValueError, match="calendar_swing"):
        _calendar_values(**mult, calendar_swing=1.0)


def test_calendar_cycle_keeps_missing_values_and_matches_generate() -> None:
    from synforecast.generators import RandomWalkGenerator

    kwargs = {"min_length": 200, "max_length": 200, "freq": "30min", "seed": 8}
    kwargs.update(missing_data=True, missing_rate=0.1)
    plain = _calendar_values(**kwargs)
    cycled = _calendar_values(**kwargs, calendar_cycles=True, calendar_weekly=0.5)
    np.testing.assert_array_equal(np.isnan(cycled), np.isnan(plain))
    df = RandomWalkGenerator(
        **kwargs, calendar_cycles=True, calendar_weekly=0.5
    ).generate(1, n_jobs=1)
    np.testing.assert_array_equal(df["y"].to_numpy(), cycled)


def test_synset_applies_calendar_cycles_like_generate() -> None:
    from synforecast import SynSet

    kwargs = {"min_length": 24 * 7, "max_length": 24 * 8, "freq": "h", "seed": 6}
    kwargs.update(overlay_seed=11, engine="pandas")
    n = 3

    def per_generator(rate: float) -> list[np.ndarray]:
        pool = pretraining_pool(**kwargs, calendar_rate=rate)
        return [
            g.generate(n, start_id=i * n, n_jobs=1)["y"].to_numpy()
            for i, g in enumerate(pool)
        ]

    pool = pretraining_pool(**kwargs, calendar_rate=1.0)
    df = SynSet(pool).generate(n_series_per_generator=n, n_jobs=2)
    ids = df["unique_id"].astype(int).to_numpy()
    expected = per_generator(1.0)
    for i, values in enumerate(expected):
        block = df["y"].to_numpy()[(ids >= i * n) & (ids < (i + 1) * n)]
        np.testing.assert_array_equal(block, values, err_msg=type(pool[i]).__name__)
    for i, (on, off) in enumerate(zip(expected, per_generator(0.0), strict=True)):
        assert not np.array_equal(on, off, equal_nan=True), type(pool[i]).__name__


def test_calendar_prior_is_deterministic_with_the_given_rate() -> None:
    kwargs = {"min_length": 16, "max_length": 16, "freq": "h", "seed": 1}
    kwargs.update(include_extended=True, calendar_rate=0.8)
    rates = []
    for overlay_seed in range(40):
        pool = pretraining_pool(**kwargs, overlay_seed=overlay_seed)
        again = pretraining_pool(**kwargs, overlay_seed=overlay_seed)
        assert [g.model_dump() for g in pool] == [g.model_dump() for g in again]
        rates += [g.calendar_cycles for g in pool]
    assert np.mean(rates) == pytest.approx(0.8, abs=0.03)
    draws = [calendar_overlays(np.random.default_rng([2, i])) for i in range(5000)]
    assert np.mean([d["calendar_mode"] == "multiplicative" for d in draws]) == (
        pytest.approx(0.5, abs=0.03)
    )
    assert np.mean([d["calendar_weekly"] == 0 for d in draws]) == pytest.approx(
        0.3, abs=0.03
    )
    with pytest.raises(ValueError, match="overlay_seed"):
        pretraining_pool(min_length=16, max_length=16, calendar_rate=0.5)
    with pytest.raises(ValueError, match="calendar_rate"):
        pretraining_pool(min_length=16, max_length=16, overlay_seed=1, calendar_rate=2)


@pytest.mark.parametrize("freq", ["h", "10min", "D"])
def test_every_member_draws_valid_series_with_calendar_cycles(freq: str) -> None:
    n = len(
        pretraining_pool(min_length=8, max_length=8, freq=freq, include_extended=True)
    )
    rng = np.random.default_rng(4)
    for draw in range(2 * n):
        (gen,) = pretraining_pool(
            min_length=1024,
            max_length=1024,
            freq=freq,
            seed=int(rng.integers(1 << 40)),
            include_extended=True,
            overlay_seed=int(rng.integers(1 << 40)),
            calendar_rate=1.0,
            members=[draw % n],
        )
        assert gen.calendar_cycles
        (values,) = gen.generate_values(1, n_jobs=1)
        assert values.shape == (1024,) and not np.isinf(values).any()
        assert np.isfinite(values).sum() >= 2, (draw % n, type(gen).__name__)
