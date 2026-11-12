"""Numpy kernels for augmenting real series windows.

Every kernel takes ``values`` (float64, NaN where unobserved) and ``observed``
(bool) of equal length and returns the same pair. Missing values stay
missing: a missing input never becomes observed except through resampling,
whose NaN rules are stated per kernel. Data-dependent quantities read only
the first ``prefix`` observed values (:func:`prefix_values`), optionally
only before a position ``limit``, so no parameter depends on values after
that prefix.

The resampling, warp and real-window augmentation recipe follows the Simons
training code (``training/augment.py`` and ``training/warp.py``).
"""

import math
from dataclasses import dataclass

import numpy as np

PREFIX = 512
"""Observed values read by every data-dependent quantity."""
SCALE_FLOOR = 1e-6


@dataclass(frozen=True)
class Resample:
    """A data-independent resampling draw over a raw series of ``length`` points.

    ``start:stop`` is the raw slice to read; the kernel returns ``out`` points.
    ``kind`` is 'downsample' (integer bucket mean), 'up' (interpolation) or
    'down' (fractional aggregation). ``offset`` is the fractional start of the
    first 'down' bucket inside the slice. ``bridge`` scales the Brownian-bridge
    fill of an 'up' draw, whose noise comes from ``bridge_seed``.
    """

    kind: str
    factor: float
    start: int
    stop: int
    out: int
    offset: float = 0.0
    bridge: float = 0.0
    bridge_seed: int = 0


def as_pair(
    values: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Values as float64 with NaN where unobserved, and the observed mask."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    observed = np.isfinite(values)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("mask must have the shape of values")
        observed &= mask
    return np.where(observed, values, np.nan), observed


def prefix_end(observed: np.ndarray, prefix: int = PREFIX) -> int:
    """Position just past the ``prefix``-th observed value (or the end)."""
    positions = np.flatnonzero(observed)
    if len(positions) <= prefix:
        return len(observed)
    return int(positions[prefix - 1]) + 1


def prefix_values(
    values: np.ndarray,
    observed: np.ndarray,
    prefix: int = PREFIX,
    limit: int | None = None,
) -> np.ndarray:
    """The first ``prefix`` observed values, only among positions before ``limit``."""
    return values[np.flatnonzero(observed[:limit])[:prefix]]


def mean_abs(
    values: np.ndarray,
    observed: np.ndarray,
    prefix: int = PREFIX,
    limit: int | None = None,
) -> float:
    """Mean absolute value of the prefix, floored at 1e-6 (1.0 without observations)."""
    head = prefix_values(values, observed, prefix, limit)
    if not len(head):
        return 1.0
    return max(float(np.abs(head).mean()), SCALE_FLOOR)


def bridge_sigma(
    values: np.ndarray, observed: np.ndarray, prefix: int = PREFIX
) -> float:
    """Robust step scale (1.4826 x MAD of finite first differences) inside the prefix."""
    head = values[: prefix_end(observed, prefix)]
    d = np.diff(head)
    d = d[np.isfinite(d)]
    if d.size < 4:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


def censor_level(
    values: np.ndarray,
    observed: np.ndarray,
    q: float,
    prefix: int = PREFIX,
    limit: int | None = None,
) -> float | None:
    """The ``q`` order statistic (nearest rank) of the prefix, or None below two observations."""
    head = np.sort(prefix_values(values, observed, prefix, limit))
    if len(head) < 2:
        return None
    return float(head[int(round(q * (len(head) - 1)))])


def downsample_cap(length: int, k_max: int, min_points: int) -> int:
    """Largest factor that leaves at least ``min_points`` buckets."""
    return min(k_max, length // max(1, min_points))


def plan_downsample(
    u_k: float,
    u_start: float,
    length: int,
    want: int,
    k_min: int,
    k_max: int,
    min_points: int,
) -> Resample | None:
    """Integer factor k ~ U{k_min..cap}; a ``want``-point crop of the bucketed series."""
    cap = downsample_cap(length, k_max, min_points)
    if cap < k_min:
        return None
    k = k_min + min(int(u_k * (cap - k_min + 1)), cap - k_min)
    buckets = length // k
    out = min(buckets, want)
    first = min(int(u_start * (buckets - out + 1)), buckets - out)
    return Resample("downsample", float(k), first * k, (first + out) * k, out)


def plan_stretch(
    u_share: float,
    u_factor: float,
    u_bridge: float,
    u_start: float,
    bridge_seed: int,
    length: int,
    want: int,
    up_share: float,
    up_range: tuple[float, float],
    down_range: tuple[float, float],
    min_points: int,
) -> Resample:
    """Up by a log-uniform factor (share ``up_share``), else down by fractional aggregation."""
    down_hi = min(down_range[1], length / max(1, min_points))
    if u_share < up_share or down_hi < down_range[0] * 1.01:
        r = _log_uniform(u_factor, *up_range)
        raw = int(min(length, math.ceil((want - 1) / r) + 1))
        start = min(int(u_start * (length - raw + 1)), length - raw)
        out = int(min(want, math.floor((raw - 1) * r + 1e-9) + 1))
        return Resample("up", r, start, start + raw, out, 0.0, u_bridge, bridge_seed)
    k = _log_uniform(u_factor, down_range[0], down_hi)
    out = int(min(want, math.floor(length / k)))
    span = out * k
    s = u_start * max(0.0, length - span)
    lo, hi = math.floor(s), int(min(length, math.ceil(s + span)))
    return Resample("down", k, lo, hi, out, s - lo)


def _log_uniform(u: float, lo: float, hi: float) -> float:
    return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


def resample(
    values: np.ndarray, observed: np.ndarray, plan: Resample, prefix: int = PREFIX
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a resampling draw to the raw slice ``start:stop`` it names."""
    if len(values) != plan.stop - plan.start:
        raise ValueError("resample expects exactly the planned raw slice")
    if plan.kind == "downsample":
        return bucket_mean(values, observed, int(plan.factor))
    if plan.kind == "up":
        return upsample(values, observed, plan, prefix)
    if plan.kind == "down":
        return aggregate(values, observed, plan)
    raise ValueError(f"unknown resampling kind {plan.kind!r}")


def bucket_mean(
    values: np.ndarray, observed: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the observed values of each k-point bucket; NaN where a bucket has none."""
    n = len(values) // k * k
    ok = observed[:n].reshape(-1, k)
    counts = ok.sum(1)
    sums = np.where(ok, values[:n].reshape(-1, k), 0.0).sum(1)
    out = np.full(len(counts), np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out, counts > 0


def upsample(
    values: np.ndarray, observed: np.ndarray, plan: Resample, prefix: int = PREFIX
) -> tuple[np.ndarray, np.ndarray]:
    """End-aligned linear interpolation at rate ``factor`` plus a Brownian-bridge fill.

    An output between two knots is NaN when either knot is; a point on a knot
    copies it. The fill has scale ``bridge`` x the prefix step scale and is
    clipped at zero only between two nonnegative knots of a series whose
    prefix is nonnegative.
    """
    x = np.where(observed, values, np.nan)
    raw, r, out = len(x), plan.factor, plan.out
    t = np.clip((raw - 1) - (out - 1 - np.arange(out)) / r, 0.0, raw - 1)
    i0 = np.minimum(np.floor(t + 1e-9).astype(np.int64), raw - 1)
    u = np.clip(t - i0, 0.0, 1.0)
    on_knot = u < 1e-9
    i1 = np.minimum(i0 + 1, raw - 1)
    a, b = x[i0], x[i1]
    with np.errstate(invalid="ignore"):
        y = np.where(on_knot, a, (1 - u) * a + u * b)
    sigma = bridge_sigma(x, observed, prefix)
    if plan.bridge > 0 and out > 1 and sigma > 0:
        rng = np.random.default_rng(plan.bridge_seed)
        fill = plan.bridge * sigma * _bridge(i0, u, on_knot, rng)
        with np.errstate(invalid="ignore"):
            fill = np.where((a == b) | on_knot, 0.0, fill)
            y = y + fill
            head = prefix_values(x, observed, prefix)
            if len(head) and not (head < 0).any():
                y = np.where((a >= 0) & (b >= 0), np.maximum(y, 0.0), y)
    finite = np.isfinite(y)
    return np.where(finite, y, np.nan), finite


def _bridge(
    i0: np.ndarray, u: np.ndarray, on_knot: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Brownian bridges between consecutive knots, sampled at fractional positions ``u``."""
    out = len(u)
    prev_u = np.concatenate([[0.0], u[:-1]])
    new_seg = np.concatenate([[True], i0[1:] != i0[:-1]])
    du = np.where(new_seg, u, u - prev_u).clip(min=0.0)
    cs = np.cumsum(np.sqrt(du) * rng.standard_normal(out))
    seg_start = np.maximum.accumulate(np.where(new_seg, np.arange(out), 0))
    base = np.where(seg_start > 0, cs[np.maximum(seg_start - 1, 0)], 0.0)
    w = cs - base
    last = np.concatenate([i0[1:] != i0[:-1], [True]])
    end_idx = np.flatnonzero(last)
    seg_id = np.cumsum(new_seg) - 1
    w1_last = w[end_idx] + np.sqrt(
        np.clip(1 - u[end_idx], 0, None)
    ) * rng.standard_normal(len(end_idx))
    bridge = w - u * w1_last[seg_id]
    return np.where(on_knot, 0.0, bridge)


def aggregate(
    values: np.ndarray, observed: np.ndarray, plan: Resample
) -> tuple[np.ndarray, np.ndarray]:
    """Fractional-width bucket means; NaN where observed coverage is at most a quarter step."""
    k, out, n = plan.factor, plan.out, len(values)
    vals = np.where(observed, values, 0.0)
    cv = np.concatenate([[0.0], np.cumsum(vals)])
    cn = np.concatenate([[0.0], np.cumsum(observed)])
    edges = np.clip(plan.offset + k * np.arange(out + 1), 0.0, n)
    e0 = np.minimum(np.floor(edges).astype(np.int64), n - 1)
    frac = edges - e0
    num = np.diff(cv[e0] + frac * vals[e0])
    cnt = np.diff(cn[e0] + frac * observed[e0])
    ok = cnt > 0.25
    y = np.full(out, np.nan)
    np.divide(num, np.maximum(cnt, 1e-9), out=y, where=ok)
    return y, ok


def power_warp(
    values: np.ndarray,
    observed: np.ndarray,
    p: float,
    prefix: int = PREFIX,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Scale-preserving power warp ``s * sign(x) * |x / s| ** p``, s the prefix mean-abs.

    Applies only when the prefix has observations and none is negative
    (None otherwise); the sign is kept, so a later negative value is warped
    symmetrically instead of being clipped.
    """
    head = prefix_values(values, observed, prefix, limit)
    if not len(head) or (head < 0).any():
        return None
    s = mean_abs(values, observed, prefix, limit)
    with np.errstate(invalid="ignore"):
        y = s * np.sign(values) * np.abs(values / s) ** p
    return np.where(observed, y, np.nan), observed.copy()


def envelope(
    length: int, first: int, knots: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    """Piecewise-linear envelope over the span from ``first`` to the end.

    ``knots`` holds the 4 relative positions (0, two interior, 1) and
    ``levels`` the multiplier at each.
    """
    span = max(length - first, 1)
    t = np.clip((np.arange(length) - first) / span, 0.0, 1.0)
    return np.interp(t, knots, levels)


def amplitude_envelope(
    values: np.ndarray, observed: np.ndarray, knots: np.ndarray, levels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Multiply by an envelope starting at the first observed position."""
    positions = np.flatnonzero(observed)
    first = int(positions[0]) if len(positions) else 0
    y = values * envelope(len(values), first, knots, levels)
    return np.where(observed, y, np.nan), observed.copy()


def censor(
    values: np.ndarray,
    observed: np.ndarray,
    bottom: bool,
    q: float,
    prefix: int = PREFIX,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Saturate at the prefix ``q`` quantile: a floor (bottom) or a cap (top).

    As in ``Observation``'s floor and cap, observed values are clamped and
    missing values stay missing. None below two prefix observations.
    """
    level = censor_level(values, observed, q, prefix, limit)
    if level is None:
        return None
    y = np.maximum(values, level) if bottom else np.minimum(values, level)
    return np.where(observed, y, np.nan), observed.copy()


def mixup(
    values: np.ndarray,
    observed: np.ndarray,
    partner: np.ndarray,
    partner_observed: np.ndarray,
    lam: float,
    prefix: int = PREFIX,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pairwise mixup in the anchor's scale: ``sa * (lam x / sa + (1 - lam) xb / sb)``.

    ``sa`` and ``sb`` are the prefix mean-abs scales; a point is observed when
    both series observe it. None when either series has no observation.
    """
    if len(partner) != len(values):
        raise ValueError("mixup partner must have the anchor's length")
    if not observed.any() or not partner_observed.any():
        return None
    sa = mean_abs(values, observed, prefix, limit)
    sb = mean_abs(partner, partner_observed, prefix)
    both = observed & partner_observed
    with np.errstate(invalid="ignore"):
        y = sa * (lam * values / sa + (1 - lam) * partner / sb)
    return np.where(both, y, np.nan), both


def negate(values: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip the sign."""
    return np.where(observed, -values, np.nan), observed.copy()


def reverse(values: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reverse time between the first and last observed points; outer gaps stay in place."""
    positions = np.flatnonzero(observed)
    y, ok = np.where(observed, values, np.nan), observed.copy()
    if len(positions) < 2:
        return y, ok
    lo, hi = int(positions[0]), int(positions[-1]) + 1
    y[lo:hi], ok[lo:hi] = y[lo:hi][::-1].copy(), ok[lo:hi][::-1].copy()
    return y, ok
