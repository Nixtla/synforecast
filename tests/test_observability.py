"""Waveform and observation overlays and the observability prior."""

import numpy as np
import pytest

from synforecast import SynSet, observability_overlays, pretraining_pool
from synforecast._core import _overlay_scale
from synforecast.base import BaseGenerator, _waveform_on_steps
from synforecast.generators import (
    DailyActiveUsersGenerator,
    RandomWalkGenerator,
)

LENGTH = 1200


def _values(**kwargs) -> np.ndarray:
    settings = {"min_length": LENGTH, "max_length": LENGTH, "freq": "5min", "seed": 3}
    settings.update(kwargs)
    (values,) = RandomWalkGenerator(**settings).generate_values(1, n_jobs=1)
    return values


def _template(**kwargs) -> np.ndarray:
    """Exact template: base removed, unit-free amplitude."""
    values = _values(waveform=True, waveform_base=0.0, **kwargs)
    return values - values.min()


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starts and lengths of the True runs of a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


class Constant(BaseGenerator):
    """Returns a fixed array, to test overlays on known input."""

    def generate_single_series(self, length: int) -> np.ndarray:
        return np.asarray(BASE[:length], dtype=np.float64).copy()


BASE = np.sin(np.arange(LENGTH) / 7.0) + 0.01 * np.arange(LENGTH)


def _constant(**kwargs) -> np.ndarray:
    settings = {"min_length": LENGTH, "max_length": LENGTH, "freq": "h", "seed": 1}
    settings.update(kwargs)
    (values,) = Constant(**settings).generate_values(1, n_jobs=1)
    return values


def test_waveform_off_draws_exactly_as_before() -> None:
    base = _values()
    np.testing.assert_array_equal(base, _values(waveform_amplitude=9.0))
    np.testing.assert_array_equal(base, _values(observation_quantum=0.5))
    np.testing.assert_array_equal(base, _values(waveform=True, waveform_amplitude=0.0))


@pytest.mark.parametrize("period", [12, 97, 360])
@pytest.mark.parametrize("duty", [0.01, 0.1, 0.5])
def test_periodic_pulse_has_the_period_and_duty_in_steps(
    period: int, duty: float
) -> None:
    values = _template(waveform_period=period, waveform_duty=duty)
    levels = np.unique(values)
    assert levels.size == 2
    on = values == levels[1]
    np.testing.assert_array_equal(on[period:], on[:-period])
    width = int(np.clip(round(duty * period), 1, period - 1))
    _, lengths = _runs(on)
    assert set(lengths[1:-1].tolist()) == {width}
    assert np.ptp(values) == pytest.approx(3.0 * _overlay_scale(_values()))


@pytest.mark.parametrize("width", [1, 2, 3])
def test_spikes_are_width_steps_on_a_flat_floor(width: int) -> None:
    values = _template(waveform_shape="spike", waveform_width=width, waveform_period=96)
    on = values > 0
    starts, lengths = _runs(on)
    assert set(lengths[1:-1].tolist()) == {width}
    assert set(np.diff(starts).tolist()) == {96}
    assert np.all(values[~on] == 0)


def test_sawtooth_ramps_and_resets_every_period() -> None:
    values = _template(waveform_shape="sawtooth", waveform_period=50)
    steps = np.diff(values)
    resets = np.flatnonzero(steps < 0)
    assert set(np.diff(resets).tolist()) == {50}
    rises = np.delete(steps, resets)
    np.testing.assert_allclose(rises, rises[0])
    assert rises[0] > 0


def test_trapezoid_has_short_ramps_and_a_flat_top() -> None:
    values = _template(
        waveform_shape="trapezoid",
        waveform_period=100,
        waveform_duty=0.4,
        waveform_ramp=0.1,
    )
    top = values.max()
    on = values > 0
    _, lengths = _runs(on)
    assert set(lengths[1:-1].tolist()) == {40}
    _, flat = _runs(values == top)
    assert set(flat[1:-1].tolist()) == {32}
    assert np.unique(values).size == 6


@pytest.mark.parametrize("shape", ["levels", "staircase"])
def test_level_shapes_are_exact_plateaus_on_a_ladder(shape: str) -> None:
    values = _template(waveform_shape=shape, waveform_levels=5, waveform_period=40)
    levels = np.unique(values)
    assert levels.size == 5
    np.testing.assert_allclose(np.diff(levels), np.diff(levels)[0])
    change = np.flatnonzero(np.diff(values) != 0)
    assert set(np.diff(change).tolist()) == {40}
    moves = np.round(np.diff(values)[change] / np.diff(levels)[0]).astype(int)
    if shape == "staircase":
        assert set(np.abs(moves).tolist()) == {1}
    else:
        assert np.abs(moves).max() > 1


def test_jitter_moves_cycle_starts_but_keeps_pulse_width() -> None:
    values = _template(waveform_period=60, waveform_duty=0.2, waveform_timing="jitter")
    starts, lengths = _runs(values > 0)
    gaps = np.diff(starts)
    assert np.unique(gaps).size > 1
    assert abs(gaps.mean() - 60) < 3
    assert np.all(np.abs(gaps - 60) <= 60)
    assert np.median(lengths) == 12


@pytest.mark.parametrize("renewal", ["gamma", "lognormal"])
def test_renewal_durations_have_the_nominal_means(renewal: str) -> None:
    on, off = [], []
    for seed in range(20):
        values = _template(
            seed=seed,
            waveform_period=40,
            waveform_duty=0.25,
            waveform_timing="renewal",
            waveform_renewal=renewal,
            waveform_cv=0.5,
        )
        starts, lengths = _runs(values > 0)
        on += lengths[1:-1].tolist()
        off += (np.diff(starts) - lengths[:-1])[1:].tolist()
    assert np.mean(on) == pytest.approx(10, rel=0.1)
    assert np.mean(off) == pytest.approx(30, rel=0.1)
    assert np.std(on) / np.mean(on) == pytest.approx(0.5, abs=0.12)


def test_dips_and_multiplicative_pulses() -> None:
    base = _constant() + 5.0
    dip = _constant(waveform=True, waveform_direction="dip", waveform_amplitude=2.0)
    added = dip - _constant()
    assert added.max() == pytest.approx(0, abs=1e-12)
    assert added.min() == pytest.approx(-2.0 * _overlay_scale(_constant()))

    class Positive(BaseGenerator):
        def generate_single_series(self, length: int) -> np.ndarray:
            return base[:length].copy()

    kwargs = {"min_length": LENGTH, "max_length": LENGTH, "freq": "h", "seed": 1}
    mult = {"waveform": True, "waveform_mode": "multiplicative", "waveform_swing": 0.4}
    (peaks,) = Positive(**kwargs, **mult).generate_values(1, n_jobs=1)
    assert set(np.round(peaks / base, 12).tolist()) == {1.0, 1.4}
    (dips,) = Positive(**kwargs, **mult, waveform_direction="dip").generate_values(
        1, n_jobs=1
    )
    assert set(np.round(dips / base, 12).tolist()) == {1.0, 0.6}
    with pytest.raises(ValueError, match="waveform_swing"):
        Positive(**kwargs, **{**mult, "waveform_swing": 1.5}, waveform_direction="dip")


def test_base_weight_shrinks_the_base_around_its_mean() -> None:
    full = _constant(waveform=True)
    half = _constant(waveform=True, waveform_base=0.5)
    template = full - BASE
    np.testing.assert_allclose(
        half, BASE.mean() + 0.5 * (BASE - BASE.mean()) + template
    )


def test_quantization_puts_every_reading_on_the_grid() -> None:
    values = _values(observation=True, observation_quantum=0.3)
    base = _values()
    quantum = 0.3 * _overlay_scale(base)
    ratio = values / quantum
    np.testing.assert_allclose(ratio, np.round(ratio), atol=1e-9)
    assert np.abs(values - base).max() <= quantum / 2 + 1e-9
    assert np.unique(values).size < np.unique(base).size


def test_saturation_bounds_are_fractions_of_the_range() -> None:
    base = _constant()
    values = _constant(observation=True, observation_floor=0.2, observation_cap=0.9)
    low, span = base.min(), np.ptp(base)
    assert values.min() == pytest.approx(low + 0.2 * span)
    assert values.max() == pytest.approx(low + 0.9 * span)
    inside = (base > low + 0.2 * span) & (base < low + 0.9 * span)
    np.testing.assert_array_equal(values[inside], base[inside])
    with pytest.raises(ValueError, match="observation_floor"):
        _constant(observation=True, observation_floor=0.6, observation_cap=0.5)


@pytest.mark.parametrize("hold", [2, 5, 13])
def test_sample_and_hold_repeats_each_block_start(hold: int) -> None:
    base = _constant()
    values = _constant(observation=True, observation_hold=hold)
    change = np.flatnonzero(np.diff(values) != 0) + 1
    assert set(np.diff(change).tolist()) == {hold}
    np.testing.assert_array_equal(values[change], base[change])


def test_frozen_windows_repeat_the_last_reading() -> None:
    base = _constant()
    values = _constant(
        observation=True, observation_stuck_fraction=0.1, observation_stuck_length=30
    )
    frozen = values != base
    assert 0.03 < frozen.mean() < 0.2
    starts, _ = _runs(frozen)
    for start in starts:
        assert values[start] == values[start - 1]


def test_observation_never_spreads_missing_values() -> None:
    gappy = BASE.copy()
    gappy[100:140] = np.nan
    gappy[500] = np.nan

    class Gappy(BaseGenerator):
        def generate_single_series(self, length: int) -> np.ndarray:
            return gappy[:length].copy()

    kwargs = {"min_length": LENGTH, "max_length": LENGTH, "freq": "h", "seed": 2}
    full = {
        "waveform": True,
        "observation": True,
        "observation_floor": 0.1,
        "observation_cap": 0.9,
        "observation_quantum": 0.2,
        "observation_hold": 7,
        "observation_stuck_fraction": 0.3,
        "observation_stuck_length": 50,
        "changepoints": True,
        "changepoint_type": "variance",
        "overlay_scale": "relative",
    }
    (values,) = Gappy(**kwargs, **full).generate_values(1, n_jobs=1)
    np.testing.assert_array_equal(np.isnan(values), np.isnan(gappy))

    class Empty(BaseGenerator):
        def generate_single_series(self, length: int) -> np.ndarray:
            return np.full(length, np.nan)

    (empty,) = Empty(**kwargs, **full).generate_values(1, n_jobs=1)
    assert np.isnan(empty).all()


def test_overlays_run_in_order_with_missingness_last() -> None:
    kwargs = {
        "observation": True,
        "observation_quantum": 0.25,
        "observation_cap": 0.8,
        "anomalies": True,
        "anomaly_fraction": 0.02,
        "anomaly_types": ["spike"],
        "spike_magnitude": 20.0,
        "overlay_scale": "relative",
        "missing_data": True,
        "missing_rate": 0.1,
    }
    values = _values(waveform=True, **kwargs)
    observed = np.isfinite(values)
    assert 0.05 < 1 - observed.mean() < 0.15
    finite = values[observed]
    # Quantization runs after the anomalies, so every reading, spikes included, is on the grid.
    on_grid = _values(waveform=True, **{**kwargs, "missing_data": False})
    quantum = np.diff(np.unique(on_grid)).min()
    np.testing.assert_allclose(finite / quantum, np.round(finite / quantum), atol=1e-6)
    # Saturation also runs after them, so several spikes are pinned at the cap.
    assert np.sum(on_grid == on_grid.max()) > 1
    no_cap = _values(waveform=True, **{**kwargs, "observation_cap": None})
    assert np.nanmax(no_cap) > on_grid.max()


def test_calendar_runs_before_the_waveform_on_the_ordered_path() -> None:
    kwargs = {"freq": "h", "waveform": True, "waveform_amplitude": 0.0}
    calendar = {"calendar_cycles": True, "calendar_amplitude": 2.0}
    scale = _overlay_scale(_values(freq="h"))
    shrunk = _values(**kwargs, **calendar, waveform_base=0.5)
    shrunk -= _values(**kwargs, waveform_base=0.5)
    np.testing.assert_allclose(shrunk[24:], shrunk[:-24], atol=1e-9)
    # The base weight halves the cycle, so it was added before the waveform.
    assert 0.5 * scale < np.ptp(shrunk) / 2 <= 1.0 * scale + 1e-9
    removed = _values(**kwargs, **calendar, waveform_base=0.0)
    assert np.unique(removed).size == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"waveform": True, "waveform_timing": "renewal", "waveform_shape": "levels"},
        {"waveform": True, "observation": True, "observation_hold": 4},
        {"observation": True, "observation_quantum": 0.2, "changepoints": True},
    ],
)
def test_draws_are_deterministic_and_generate_matches_values(kwargs: dict) -> None:
    np.testing.assert_array_equal(_values(**kwargs), _values(**kwargs))
    assert not np.array_equal(_values(**kwargs), _values(**kwargs, seed=4))
    settings = {"min_length": 300, "max_length": 400, "freq": "h", "seed": 9, **kwargs}
    df = RandomWalkGenerator(**settings).generate(3, n_jobs=2)
    values = RandomWalkGenerator(**settings).generate_values(3, n_jobs=1)
    np.testing.assert_array_equal(df["y"].to_numpy(), np.concatenate(values))


def test_threaded_path_gets_the_overlays() -> None:
    settings = {"min_length": 400, "max_length": 400, "freq": "D", "seed": 5}
    wave = {"waveform": True, "waveform_shape": "spike", "waveform_base": 0.0}
    (values,) = DailyActiveUsersGenerator(**settings, **wave).generate_values(
        1, n_jobs=1
    )
    assert np.unique(values).size == 2
    np.testing.assert_array_equal(values[96:], values[:-96])


def test_synset_matches_generate_with_the_observability_prior() -> None:
    kwargs = {"min_length": 200, "max_length": 300, "freq": "5min", "seed": 6}
    kwargs.update(overlay_seed=11, calendar_rate=0.5, observability_rate=1.0)
    n = 2
    pool = pretraining_pool(**kwargs)
    df = SynSet(pool).generate(n_series_per_generator=n, n_jobs=2)
    ids = df["unique_id"].astype(int).to_numpy()
    for i, gen in enumerate(pretraining_pool(**kwargs)):
        expected = gen.generate(n, start_id=i * n, n_jobs=1)["y"].to_numpy()
        block = df["y"].to_numpy()[(ids >= i * n) & (ids < (i + 1) * n)]
        np.testing.assert_array_equal(block, expected, err_msg=type(gen).__name__)


def test_observability_prior_rates_and_validation() -> None:
    kwargs = {"min_length": 96, "max_length": 96, "freq": "10s", "seed": 1}
    kwargs.update(include_extended=True)
    plain = pretraining_pool(**kwargs, overlay_seed=3, calendar_rate=0.5)
    zero = pretraining_pool(
        **kwargs, overlay_seed=3, calendar_rate=0.5, observability_rate=0
    )
    assert [g.model_dump() for g in plain] == [g.model_dump() for g in zero]
    picked = []
    for overlay_seed in range(40):
        pool = pretraining_pool(
            **kwargs, overlay_seed=overlay_seed, observability_rate=0.3
        )
        again = pretraining_pool(
            **kwargs, overlay_seed=overlay_seed, observability_rate=0.3
        )
        assert [g.model_dump() for g in pool] == [g.model_dump() for g in again]
        picked += [g.waveform for g in pool]
        for g in pretraining_pool(
            **kwargs, overlay_seed=overlay_seed, observability_rate=1
        ):
            assert g.waveform and 12 <= g.waveform_period <= 32
    assert np.mean(picked) == pytest.approx(0.3, abs=0.03)
    with pytest.raises(ValueError, match="overlay_seed"):
        pretraining_pool(min_length=16, max_length=16, observability_rate=0.5)
    with pytest.raises(ValueError, match="observability_rate"):
        pretraining_pool(
            min_length=16, max_length=16, overlay_seed=1, observability_rate=2
        )
    with pytest.raises(ValueError, match="max_period"):
        observability_overlays(np.random.default_rng(0), max_period=5)


def test_observability_prior_components() -> None:
    draws = [observability_overlays(np.random.default_rng([4, i])) for i in range(6000)]
    layered = np.mean([d["changepoints"] for d in draws])
    assert layered == pytest.approx(0.25, abs=0.02)
    for d in draws:
        assert d["waveform"] and 12 <= d["waveform_period"] <= 720
        if d["changepoints"]:
            assert d["observation"] and d["anomalies"] and d["waveform_base"] >= 0.5
    always = set.intersection(*(set(d) for d in draws))
    clean_like = [
        d
        for d in draws
        if not (d["changepoints"] or d["anomalies"] or d["missing_data"])
        and d["waveform_base"] <= 0.1
        and set(d) - always <= {"observation_quantum"}
    ]
    # Clean draws plus mild draws whose only perturbation is quantization.
    assert len(clean_like) / len(draws) == pytest.approx(0.4 + 0.35 / 14, abs=0.02)
    for d in draws:
        if d.get("observation_hold", 1) > 1:
            on = _waveform_on_steps(
                d["waveform_shape"],
                d["waveform_period"],
                d["waveform_duty"],
                d["waveform_width"],
            )
            assert d["observation_hold"] <= on
    shapes = [d["waveform_shape"] for d in draws]
    assert np.mean([s == "pulse" for s in shapes]) == pytest.approx(0.25, abs=0.02)
    periods = np.log([d["waveform_period"] for d in draws])
    assert np.median(periods) == pytest.approx(np.log(np.sqrt(12 * 720)), abs=0.1)


@pytest.mark.parametrize("freq", ["10s", "5min", "D"])
def test_every_member_draws_valid_series_with_the_observability_prior(
    freq: str,
) -> None:
    n = len(pretraining_pool(min_length=8, max_length=8, include_extended=True))
    rng = np.random.default_rng(7)
    for draw in range(2 * n):
        (gen,) = pretraining_pool(
            min_length=1024,
            max_length=1024,
            freq=freq,
            seed=int(rng.integers(1 << 40)),
            include_extended=True,
            overlay_seed=int(rng.integers(1 << 40)),
            calendar_rate=0.8,
            observability_rate=1.0,
            start_datetime="2000-01-03 06:00",
            members=[draw % n],
        )
        (values,) = gen.generate_values(1, n_jobs=1)
        assert values.shape == (1024,) and not np.isinf(values).any()
        assert np.isfinite(values).sum() >= 2, (draw % n, type(gen).__name__)
