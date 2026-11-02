"""Golden hashes of pretraining_pool draws, recorded before the waveform and observation overlays existed.

New overlays must leave every existing draw byte-identical: these cases cover the
default pool, the extended pool with the overlay prior, the calendar prior,
sub-hourly to monthly frequencies, the lengths a downstream pretraining stream
draws, and both the per-generator and the SynSet batch paths. The hashes pin
bit-exact output of the Linux x86_64 build; other platforms can differ in the
last ulp of transcendental functions, so the test only runs there.
"""

import hashlib
import platform

import numpy as np
import pytest

from synforecast import SynSet, pretraining_pool

pytestmark = pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine() != "x86_64",
    reason="bit-exact goldens are pinned on Linux x86_64",
)

EXTENDED = pretraining_pool(include_extended=True)
# DailyActiveUsers adds an event column, which SynSet cannot concatenate with the others.
NO_EVENTS = [
    i for i, g in enumerate(EXTENDED) if type(g).__name__ != "DailyActiveUsersGenerator"
]

# Pretraining-stream draws: (freq, length, seed, overlay_seed, calendar_rate, start_datetime).
CONSUMER_CASES = {
    "10s-2273-11": ("10s", 2273, 11, 101, 0.8, "2000-01-03 05:00:10"),
    "5min-3041-12": ("5min", 3041, 12, 102, 0.8, "2000-01-05 17:35:00"),
    "h-128-13": ("h", 128, 13, 103, 0.8, "2000-01-01 07:00:00"),
    "D-2273-14": ("D", 2273, 14, 104, 0.25, "2000-01-04"),
    "MS-128-15": ("MS", 128, 15, 105, 0.0, None),
    "h-3041-16": ("h", 3041, 16, 106, 0.0, None),
    "5min-128-wide-seed": (
        "5min",
        128,
        2305843009213693969,
        2305843009213694059,
        0.8,
        "2000-01-02 23:55:00",
    ),
}
# sha256 over every extended member's draw, per case.
CONSUMER_GOLDEN = {
    "10s-2273-11": "286e05b8182540e239692d82672a97441ec30a9bd48b20bcf2d81f6655c9f678",
    "5min-3041-12": "31a4bd55e354e6b125438a4a4866bc5b653ff711a72841d26776a632adcfd8ef",
    "h-128-13": "27861fcbc43d1555d755ca7cec555587ab467c29a72df7a87bf1dfecd4006e78",
    "D-2273-14": "b7d1ed27a71db2de72ec1d2ff8d77b1cf456ab0ce970b4e58b4e309810b64418",
    "MS-128-15": "95badb229885863b673a513ed1f9a95da098019ec7a10da06d85bfbd521e670a",
    "h-3041-16": "55efd5c8cfb89ceafd322fa62b3e176f74891295ce9e45000949d238e4c40941",
    "5min-128-wide-seed": "17208a82dbb6ef1dfa8f81b6818a375d6bf8d8918c4635a89d6211dadf00464f",
}

SYNSET_GOLDEN = {
    "default": "d48732a61ca50bd43d6e57c10305086a6afca169615702f5e8240268db5cc90c",
    "overlay": "fc5a0b4190bed1a973a000c6dbd1f83c6544b671d5a9bf608d610a679f8829a9",
    "calendar": "eaa034742ae077852a7258e44c85ae63dd5ee453e6e2d60bde8885c8cf184c6d",
}

SYNSET_KWARGS = {
    "default": {"min_length": 150, "max_length": 260, "freq": "h", "seed": 3},
    "overlay": {
        "min_length": 150,
        "max_length": 260,
        "freq": "h",
        "seed": 4,
        "include_extended": True,
        "overlay_seed": 5,
        "members": NO_EVENTS,
    },
    "calendar": {
        "min_length": 150,
        "max_length": 260,
        "freq": "5min",
        "seed": 6,
        "include_extended": True,
        "overlay_seed": 7,
        "calendar_rate": 0.8,
        "start_datetime": "2000-01-06 13:05:00",
        "members": NO_EVENTS,
    },
}

DEFAULT_GOLDEN = "fd72709857eb01faec0e7b74298514bf52a9808f301e0f034c7bd21f7c17e515"


def _update(digest: "hashlib._Hash", values: np.ndarray) -> None:
    digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())


def _consumer_digest(
    freq: str,
    length: int,
    seed: int,
    overlay_seed: int,
    rate: float,
    start: str | None,
) -> str:
    """Draw each member as a pretraining stream does: built alone, one series, one job."""
    digest = hashlib.sha256()
    calendar = {} if start is None else {"calendar_rate": rate, "start_datetime": start}
    for member in range(len(EXTENDED)):
        (gen,) = pretraining_pool(
            min_length=length,
            max_length=length,
            freq=freq,
            seed=seed,
            members=[member],
            include_extended=True,
            overlay_seed=overlay_seed,
            **calendar,
        )
        _update(digest, gen.generate_values(1, n_jobs=1)[0])
    return digest.hexdigest()


def _synset_digest(name: str) -> str:
    df = SynSet(pretraining_pool(**SYNSET_KWARGS[name])).generate(
        n_series_per_generator=2, n_jobs=2
    )
    digest = hashlib.sha256()
    _update(digest, df["y"].to_numpy())
    _update(digest, df["unique_id"].astype(int).to_numpy())
    return digest.hexdigest()


def _default_digest() -> str:
    digest = hashlib.sha256()
    for gen in pretraining_pool():
        for values in gen.generate_values(2, n_jobs=1):
            _update(digest, values)
    return digest.hexdigest()


@pytest.mark.parametrize("name", list(CONSUMER_CASES))
def test_consumer_draws_are_unchanged(name: str) -> None:
    assert _consumer_digest(*CONSUMER_CASES[name]) == CONSUMER_GOLDEN[name]


@pytest.mark.parametrize("name", list(SYNSET_GOLDEN))
def test_synset_draws_are_unchanged(name: str) -> None:
    assert _synset_digest(name) == SYNSET_GOLDEN[name]


def test_default_pool_draws_are_unchanged() -> None:
    assert _default_digest() == DEFAULT_GOLDEN


@pytest.mark.parametrize("name", list(CONSUMER_CASES)[:2])
def test_zero_observability_rate_is_a_no_op(name: str) -> None:
    freq, length, seed, overlay_seed, rate, start = CONSUMER_CASES[name]
    kwargs = {
        "min_length": length,
        "max_length": length,
        "freq": freq,
        "seed": seed,
        "include_extended": True,
        "overlay_seed": overlay_seed,
        "calendar_rate": rate,
        "start_datetime": start,
    }
    plain = pretraining_pool(**kwargs)
    zero = pretraining_pool(**kwargs, observability_rate=0.0)
    assert [g.model_dump() for g in plain] == [g.model_dump() for g in zero]
