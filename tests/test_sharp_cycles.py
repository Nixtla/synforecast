import numpy as np
import pytest

from synforecast import pretraining_pool
from synforecast.generators.random_walk import RandomWalkGenerator
from synforecast.overlays import Calendar, CalendarCycle, OverlayConfig
from synforecast.presets import (
    _steps_per_day,
    peak_calendar_overlays,
    step_cycle_overlays,
)

SUBDAILY = {"h": 24, "30min": 48, "15min": 96, "10min": 144, "5min": 288}


def _calendar() -> Calendar:
    weekly = CalendarCycle(
        phase="week", profile="levels", values=[0.0] * 5 + [1.0] * 2, weight=0.5
    )
    return Calendar(
        cycles=[CalendarCycle(phase="day", harmonics=3), weekly],
        amplitude=1.0,
        scale="absolute",
    )


def _render(
    calendar: Calendar, freq: str, length: int, start: str = "2000-01-03"
) -> np.ndarray:
    gen = RandomWalkGenerator(
        min_length=length,
        max_length=length,
        freq=freq,
        seed=0,
        start_datetime=start,
        overlays=OverlayConfig(base_weight=0.0, calendar=calendar),
    )
    (values,) = gen.generate_values(1, n_jobs=1)
    return np.asarray(values, dtype=np.float64)


def _sharpness(profile: np.ndarray) -> float:
    p = profile - np.median(profile)
    return max(p.max(), -p.min()) / p.std()


def _day_draws(n: int, count: int = 60) -> list[Calendar]:
    draws = [
        peak_calendar_overlays(np.random.default_rng(seed), _calendar(), n)
        for seed in range(count)
    ]
    return [c for c in draws if c.cycles[0].phase == "day"]


def test_steps_per_day_covers_fixed_subdaily_frequencies_only() -> None:
    assert {f: _steps_per_day(f) for f in SUBDAILY} == SUBDAILY
    assert _steps_per_day("min") == 1440
    assert [_steps_per_day(f) for f in ("D", "W", "MS", "QS", 7)] == [None] * 5


@pytest.mark.parametrize(("freq", "n"), SUBDAILY.items())
@pytest.mark.parametrize("start", ["2000-01-03", "2000-01-05 07:00"])
def test_peak_profile_renders_its_template_exactly(
    freq: str, n: int, start: str
) -> None:
    sharp, widths = [], []
    for cal in _day_draws(n):
        template = np.asarray(cal.cycles[0].values)
        y = _render(
            cal.model_copy(update={"cycles": [cal.cycles[0]]}), freq, 6 * n, start
        )
        offset = int(
            round(
                (np.datetime64(start) - np.datetime64(start[:10]))
                / np.timedelta64(86400 // n, "s")
            )
        )
        profile = np.roll(y.reshape(6, n).mean(0), offset)
        assert np.corrcoef(profile, template)[0, 1] > 0.9999
        sharp.append(_sharpness(profile))
        a = np.abs(profile - profile.mean())
        widths.append(int((a >= a.max() / 2).sum()))
    assert np.median(sharp) >= 2.5
    assert min(widths) >= 2


def test_peak_calendar_keeps_mode_amplitude_and_weekly_cycle() -> None:
    base = _calendar().model_copy(
        update={"mode": "multiplicative", "swing": 0.4, "amplitude": 3.0}
    )
    for seed in range(40):
        cal = peak_calendar_overlays(np.random.default_rng(seed), base, 24)
        assert (cal.mode, cal.swing, cal.amplitude) == ("multiplicative", 0.4, 3.0)
        if cal.cycles[0].phase == "day":
            assert cal.cycles[0].profile == "levels" and cal.cycles[0].bins == 24
            assert cal.cycles[1] == base.cycles[1]
        else:
            assert len(cal.cycles) == 1 and cal.cycles[0].bins == 7 * 24


def test_weekend_variant_changes_the_shape_of_the_day() -> None:
    weekly = [
        c
        for c in (
            peak_calendar_overlays(np.random.default_rng(seed), _calendar(), 48)
            for seed in range(60)
        )
        if c.cycles[0].phase == "week"
    ]
    assert weekly
    for cal in weekly:
        days = np.asarray(cal.cycles[0].values).reshape(7, 48)
        assert all(np.array_equal(days[0], days[d]) for d in range(1, 5))
        assert np.array_equal(days[5], days[6]) and not np.allclose(days[0], days[5])


def test_weekly_variant_is_skipped_when_a_week_exceeds_the_bin_limit() -> None:
    assert all(
        c.cycles[0].phase == "day"
        for c in (
            peak_calendar_overlays(np.random.default_rng(s), _calendar(), 1440)
            for s in range(40)
        )
    )


@pytest.mark.parametrize("length", [1407, 1408, 2273, 3041, 8192])
def test_step_cycle_period_leaves_four_repeats_before_a_1024_horizon(
    length: int,
) -> None:
    for seed in range(30):
        rng = np.random.default_rng(seed)
        alone = step_cycle_overlays(rng, None, length)
        if length < 1408:
            assert alone is None
            continue
        period = alone.cycles[0].period
        assert 96 <= period <= min(400, (length - 1024) / 4) + 0.5
        assert alone.cycles[0].phase == "step" and alone.cycles[0].profile == "fourier"


def test_step_cycle_is_appended_with_twice_the_existing_weight() -> None:
    base = _calendar()
    cal = step_cycle_overlays(np.random.default_rng(0), base, 3041)
    assert cal.cycles[:2] == base.cycles
    assert cal.cycles[2].phase == "step" and cal.cycles[2].weight == pytest.approx(3.0)
    assert (cal.mode, cal.amplitude) == (base.mode, base.amplitude)
    assert step_cycle_overlays(np.random.default_rng(0), base, 1000) is base


def _pool(freq: str, **rates: float) -> list:
    kwargs = {
        "min_length": 3041,
        "max_length": 3041,
        "freq": freq,
        "include_extended": True,
        "overlay_seed": 7,
        "calendar_rate": 0.8,
        "observability_rate": 0.2,
    }
    if freq in ("h", "5min"):
        kwargs["start_datetime"] = "2000-01-03"
    return pretraining_pool(**kwargs, **rates)


@pytest.mark.parametrize("freq", ["D", "W", "MS"])
def test_new_priors_leave_daily_and_coarser_entries_unchanged(freq: str) -> None:
    off, on = _pool(freq), _pool(freq, peak_calendar_rate=1.0, step_cycle_rate=1.0)
    for a, b in zip(off, on, strict=True):
        np.testing.assert_array_equal(
            a.generate_values(1, n_jobs=1)[0], b.generate_values(1, n_jobs=1)[0]
        )


def test_new_priors_change_only_the_entries_they_draw_for() -> None:
    off, on = _pool("h"), _pool("h", peak_calendar_rate=0.5, step_cycle_rate=0.5)
    changed = 0
    for a, b in zip(off, on, strict=True):
        same_calendar = getattr(a, "overlays", None) == getattr(b, "overlays", None)
        va, vb = a.generate_values(1, n_jobs=1)[0], b.generate_values(1, n_jobs=1)[0]
        if same_calendar:
            np.testing.assert_array_equal(va, vb)
        else:
            changed += 1
    assert 0 < changed < len(off)


def test_member_built_alone_matches_full_pool_with_new_priors() -> None:
    full = _pool("5min", peak_calendar_rate=0.5, step_cycle_rate=0.5)
    for member in (0, 7, len(full) - 1):
        alone = pretraining_pool(
            min_length=3041,
            max_length=3041,
            freq="5min",
            include_extended=True,
            overlay_seed=7,
            calendar_rate=0.8,
            observability_rate=0.2,
            peak_calendar_rate=0.5,
            step_cycle_rate=0.5,
            members=[member],
            start_datetime="2000-01-03",
        )[0]
        np.testing.assert_array_equal(
            full[member].generate_values(1, n_jobs=1)[0],
            alone.generate_values(1, n_jobs=1)[0],
        )


@pytest.mark.parametrize("name", ["peak_calendar_rate", "step_cycle_rate"])
def test_new_rates_are_validated(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        pretraining_pool(**{name: 1.5}, overlay_seed=1)
    with pytest.raises(ValueError, match="overlay_seed"):
        pretraining_pool(**{name: 0.5})
