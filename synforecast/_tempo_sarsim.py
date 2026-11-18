"""Numpy kernels of the TempoPFN regime SDE and the SarSim SARIMA-2 families.

Both are ``fn(rng, n, length, ...) -> float64 [n, length]`` drawn from a
``numpy.random.Generator`` and vectorised over rows:

- :func:`tempo_sde`: the regime-switching, time-inhomogeneous OU process of
  TempoPFN (Moroshan et al., 2025, arXiv:2510.25502, Sec. 3.2, App. C.1,
  Table 6), with an optional exact fractional-Brownian-motion driver.
- :func:`sarsim`: SarSim's stable SARIMA, SARIMA-2 (fast base with a slow
  envelope) and level-scaled noisers (arXiv:2601.00970, Sec. 3, Table 9).

Design choices and priors are documented in the generator classes.
"""

import math

import numpy as np

from synforecast._families import TWO_PI, categorical, logu, ou_scan, u

TREND_PROBABILITIES = (0.3, 0.15, 0.15, 0.1, 0.15, 0.1, 0.05)
"""TempoPFN trend kinds: none, linear, exponential, logistic, sinusoidal, piecewise linear, polynomial."""

SEASONAL_PERIODS = (7.0, 30.0, 90.0, 365.25, 182.625)
"""TempoPFN seasonal periods, in model time units (steps x dt)."""

MIN_SEGMENT_STEPS = 200
SV_REVERSION = ((2.0, 5.0), (0.5, 2.0))
SV_MEAN = ((0.2, 0.4), (0.8, 1.2))
SV_VOL = ((0.1, 0.3), (0.3, 0.5))

SARIMA2_SEASONS = ((24, 7), (7, 52), (0, 7), (0, 3), (0, 24), (0, 52))
"""SarSim (base, envelope) season pairs; a base period of 0 means a non-seasonal base."""

ARMA_WARMUP = 1024
"""Minimum discarded steps of the circular ARMA synthesis."""

ARMA_MAX_WARMUP = 65536
"""Cap on the pole-decay part of the warm-up, bounding memory for poles near the unit circle (the MA-lag part is added uncapped)."""
NOISER_KINDS = ("poisson", "gamma", "lognormal")


def markov_regimes(
    rng: np.random.Generator, n: int, length: int, stay_lo: float, stay_hi: float
) -> np.ndarray:
    """Two-state Markov regime paths [n, L] (True = regime 1) with stay probabilities p00, p11 ~ U[stay_lo, stay_hi]."""
    stay = u(rng, (n, 2), stay_lo, stay_hi)
    first = rng.integers(0, 2, n)
    leave = 1.0 - stay
    chunk = int(math.ceil(length * float(leave.max()) * 1.25)) + 16
    ends = np.zeros(n)
    bounds = []
    segment = 0
    while (ends < length).any():
        j = segment + np.arange(chunk)
        state = (first[:, None] + j[None]) % 2
        q = np.take_along_axis(leave, state, 1)
        cum = ends[:, None] + np.cumsum(rng.geometric(q), 1)
        bounds.append(cum)
        ends = cum[:, -1]
        segment += chunk
    cum = np.concatenate(bounds, 1)
    flips = np.zeros((n, length), dtype=np.int64)
    rows, cols = np.nonzero(cum < length)
    flips[rows, cum[rows, cols].astype(np.int64)] = 1
    return ((first[:, None] + np.cumsum(flips, 1)) % 2).astype(bool)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def trend_paths(rng: np.random.Generator, n: int, t: np.ndarray) -> np.ndarray:
    """TempoPFN trend functions [n, L] of model time ``t`` plus up to two smooth structural changes."""
    length, horizon = t.shape[0], float(max(t[-1], 1e-12))
    kind = categorical(rng, TREND_PROBABILITIES, n)
    out = np.zeros((n, length))
    tt = t[None]
    rows = np.flatnonzero(kind == 1)
    if rows.size:
        out[rows] = u(rng, (rows.size, 1), -0.01, 0.01) * tt
    rows = np.flatnonzero(kind == 2)
    if rows.size:
        rate = u(rng, (rows.size, 1), -0.005, 0.005)
        out[rows] = u(rng, (rows.size, 1), -5.0, 5.0) * (1.0 - np.exp(-rate * tt))
    rows = np.flatnonzero(kind == 3)
    if rows.size:
        growth = u(rng, (rows.size, 1), 0.01, 0.1)
        capacity = u(rng, (rows.size, 1), 5.0, 20.0)
        mid = u(rng, (rows.size, 1), 0.3, 0.7) * horizon
        out[rows] = capacity * _logistic(growth * (tt - mid))
    rows = np.flatnonzero(kind == 4)
    if rows.size:
        amplitude = u(rng, (rows.size, 1), 1.0, 5.0)
        period = u(rng, (rows.size, 1), 0.1, 0.5) * horizon
        phase = u(rng, (rows.size, 1), 0.0, TWO_PI)
        out[rows] = amplitude * np.sin(TWO_PI * tt / period + phase)
    rows = np.flatnonzero(kind == 5)
    if rows.size:
        m = rows.size
        segments = rng.integers(2, 5, m)
        inner = u(rng, (m, 3), 0.0, horizon)
        inner = np.where(np.arange(3)[None] < (segments - 1)[:, None], inner, horizon)
        edges = np.concatenate(
            [np.zeros((m, 1)), np.sort(inner, 1), np.full((m, 1), horizon)], 1
        )
        slopes = u(rng, (m, 4), -0.02, 0.02)
        width = np.diff(edges, axis=1)
        acc = np.zeros((m, length))
        for j in range(4):
            acc += slopes[:, j : j + 1] * np.clip(
                tt - edges[:, j : j + 1], 0.0, width[:, j : j + 1]
            )
        out[rows] = acc
    rows = np.flatnonzero(kind == 6)
    if rows.size:
        coef = u(rng, (rows.size, 3), -1e-6, 1e-6)
        out[rows] = coef[:, :1] + coef[:, 1:2] * tt + coef[:, 2:3] * tt**2
    min_segment = MIN_SEGMENT_STEPS * (t[1] - t[0] if length > 1 else 1.0)
    count = rng.integers(0, 3, n)
    when = u(rng, (n, 2), min_segment, max(horizon - min_segment, min_segment))
    size = u(rng, (n, 2), 1.0, 5.0)
    room = max(int(horizon / min_segment) - 1, 0) if horizon > 2 * min_segment else 0
    count = np.minimum(count, room)
    width = 0.1 * min_segment
    for j in range(2):
        on = (j < count)[:, None]
        out += (
            on
            * size[:, j : j + 1]
            * _logistic(10.0 * (tt - when[:, j : j + 1]) / width)
        )
    return out


def seasonal_paths(rng: np.random.Generator, n: int, t: np.ndarray) -> np.ndarray:
    """One or two TempoPFN sinusoids [n, L] at jittered periods from :data:`SEASONAL_PERIODS`, with drifting amplitude."""
    count = rng.integers(1, 3, n)
    pick = np.argsort(rng.random((n, len(SEASONAL_PERIODS))), 1)[:, :2]
    period = np.asarray(SEASONAL_PERIODS)[pick] * (1.0 + u(rng, (n, 2), -0.05, 0.05))
    amplitude = u(rng, (n, 2), 0.5, 3.0)
    phase = u(rng, (n, 2), 0.0, TWO_PI)
    evolution = u(rng, (n, 2), -0.001, 0.001)
    tt = t[None]
    out = np.zeros((n, t.shape[0]))
    for j in range(2):
        on = (j < count)[:, None]
        amp = amplitude[:, j : j + 1] + evolution[:, j : j + 1] * tt
        out += (
            on * amp * np.sin(TWO_PI * tt / period[:, j : j + 1] + phase[:, j : j + 1])
        )
    return out


def fgn(rng: np.random.Generator, hurst: np.ndarray, length: int) -> np.ndarray:
    """Exact unit-variance fractional Gaussian noise [m, L] by Davies-Harte circulant embedding, one Hurst per row."""
    m = hurst.shape[0]
    k = np.arange(length + 1, dtype=np.float64)[None]
    two_h = 2.0 * hurst.reshape(m, 1)
    gamma = 0.5 * (np.abs(k + 1) ** two_h - 2.0 * k**two_h + np.abs(k - 1) ** two_h)
    circulant = np.concatenate([gamma, gamma[:, -2:0:-1]], 1)
    size = circulant.shape[1]
    eigen = np.maximum(np.fft.fft(circulant, axis=1).real, 0.0)
    z = rng.standard_normal((m, size)) + 1j * rng.standard_normal((m, size))
    return np.fft.fft(np.sqrt(eigen / size) * z, axis=1).real[:, :length]


def _regime_pair(regime: np.ndarray, a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    return np.where(regime, a1, a0)


def _maybe(rng: np.random.Generator, n: int, p: float, draw) -> np.ndarray | float:
    on = np.flatnonzero(rng.random(n) < p)
    if not on.size:
        return 0.0
    out = draw(on.size)
    full = np.zeros((n, out.shape[1]))
    full[on] = out
    return full


def tempo_sde(
    rng: np.random.Generator,
    n: int,
    length: int,
    dt: float,
    fbm_rate: float,
    hurst_lo: float,
    hurst_hi: float,
    stay_lo: float,
    stay_hi: float,
    stochastic_vol_rate: float,
) -> np.ndarray:
    """TempoPFN regime-switching time-inhomogeneous OU paths [n, L] (Euler-Maruyama at step ``dt``)."""
    t = np.arange(length, dtype=np.float64) * dt
    regime = markov_regimes(rng, n, length, stay_lo, stay_hi)
    theta_b = (u(rng, (n, 1), 1.0, 5.0), u(rng, (n, 1), 0.05, 0.5))
    mu_b = (rng.normal(-2.0, 1.0, (n, 1)), rng.normal(2.0, 1.0, (n, 1)))
    sigma_b = (
        rng.lognormal(math.log(0.3), 0.3, (n, 1)),
        rng.lognormal(math.log(1.5), 0.5, (n, 1)),
    )
    x0 = rng.normal(0.0, 2.0, n)

    def trend(p):
        return [_maybe(rng, n, p, lambda m: trend_paths(rng, m, t)) for _ in range(2)]

    def season(p):
        on = rng.random(n) < p
        if not on.any():
            return [0.0, 0.0]
        rows = np.flatnonzero(on)
        out = []
        for _ in range(2):
            full = np.zeros((n, length))
            full[rows] = seasonal_paths(rng, rows.size, t)
            out.append(full)
        return out

    d_mu, d_theta, d_sigma = trend(0.7), trend(0.2), trend(0.3)
    s_mu, s_sigma = season(0.6), season(0.3)
    theta = _regime_pair(
        regime, theta_b[0] * (1.0 + d_theta[0]), theta_b[1] * (1.0 + d_theta[1])
    )
    theta = np.clip(theta, 1e-3, 0.5 / dt)
    mu = _regime_pair(regime, mu_b[0] + d_mu[0] + s_mu[0], mu_b[1] + d_mu[1] + s_mu[1])
    sigma = _regime_pair(
        regime,
        sigma_b[0] * np.abs(1.0 + d_sigma[0] + s_sigma[0]),
        sigma_b[1] * np.abs(1.0 + d_sigma[1] + s_sigma[1]),
    )
    sv = np.flatnonzero(rng.random(n) < stochastic_vol_rate)
    if sv.size:
        m = sv.size
        kappa = [u(rng, (m, 1), *SV_REVERSION[r]) for r in (0, 1)]
        level = [np.log(u(rng, (m, 1), *SV_MEAN[r])) for r in (0, 1)]
        xi = [u(rng, (m, 1), *SV_VOL[r]) for r in (0, 1)]
        reg = regime[sv]
        kap = _regime_pair(reg, kappa[0], kappa[1])
        drive = kap * dt * _regime_pair(reg, level[0], level[1]) + _regime_pair(
            reg, xi[0], xi[1]
        ) * math.sqrt(dt) * rng.standard_normal((m, length))
        h0 = np.where(reg[:, 0], level[1][:, 0], level[0][:, 0])
        h = np.empty((m, length))
        h[:, 0] = h0
        if length > 1:
            h[:, 1:] = ou_scan(1.0 - kap[:, :-1] * dt, drive[:, :-1], h0)
        sigma[sv] = sigma[sv] * np.exp(0.5 * h)
    increments = math.sqrt(dt) * rng.standard_normal((n, length))
    fbm_rows = np.flatnonzero(rng.random(n) < fbm_rate)
    if fbm_rows.size:
        hurst = u(rng, (fbm_rows.size, 1), hurst_lo, hurst_hi)
        increments[fbm_rows] = dt**hurst * fgn(rng, hurst, length)
    a = 1.0 - theta[:, :-1] * dt
    drive = theta[:, :-1] * mu[:, :-1] * dt + sigma[:, :-1] * increments[:, :-1]
    path = np.empty((n, length))
    path[:, 0] = x0
    if length > 1:
        path[:, 1:] = ou_scan(a, drive, x0)
    level_shift = u(rng, (n, 1), -100.0, 100.0)
    scale = u(rng, (n, 1), 0.1, 50.0)
    noise = u(rng, (n, 1), 0.0, 0.1)
    return level_shift + scale * path + noise * rng.standard_normal((n, length))


def root_polynomial(
    rng: np.random.Generator, order: np.ndarray, max_order: int, radius: float
) -> np.ndarray:
    """Coefficients [n, max_order + 1] of prod(1 - r_j z) with roots r_j inside |z| < ``radius`` (conjugate pairs, one real root if odd)."""
    n = order.shape[0]
    pairs = max_order // 2
    coef = np.zeros((n, max_order + 1))
    coef[:, 0] = 1.0
    r = u(rng, (n, max(pairs, 1)), 0.0, radius)
    angle = u(rng, (n, max(pairs, 1)), 0.0, TWO_PI)
    real = u(rng, n, 0.0, radius) * np.where(rng.random(n) < 0.5, -1.0, 1.0)
    for j in range(pairs):
        on = (j < order // 2)[:, None]
        c1 = np.where(on[:, 0], -2.0 * r[:, j] * np.cos(angle[:, j]), 0.0)[:, None]
        c2 = np.where(on[:, 0], r[:, j] ** 2, 0.0)[:, None]
        shifted1 = np.pad(coef[:, :-1], ((0, 0), (1, 0)))
        shifted2 = np.pad(coef[:, :-2], ((0, 0), (2, 0)))
        coef = coef + c1 * shifted1 + c2 * shifted2
    c1 = np.where(order % 2 == 1, -real, 0.0)[:, None]
    coef = coef + c1 * np.pad(coef[:, :-1], ((0, 0), (1, 0)))
    return coef


def _poly_response(coef: np.ndarray, z: np.ndarray) -> np.ndarray:
    out = np.zeros(z.shape, dtype=np.complex128) + coef[:, -1:]
    for j in range(coef.shape[1] - 2, -1, -1):
        out = out * z + coef[:, j : j + 1]
    return out


def sarima_paths(
    rng: np.random.Generator,
    season: np.ndarray,
    length: int,
    ar_radius: float,
    seasonal_ar_radius: float,
) -> np.ndarray:
    """SarSim stable SARIMA paths [n, L]: additive (lacunary) ARMA from sampled roots, then fractional integration d ~ U[0, 1] and seasonal integration (D = 1)."""
    n = season.shape[0]
    seasonal = season >= 2
    period = np.maximum(season, 1)
    seasonal_ar = rng.random(n) < 0.5
    p = np.where(seasonal_ar, 0, rng.integers(0, 11, n))
    big_p = np.where(seasonal_ar & seasonal, rng.integers(0, 3, n), 0)
    q = rng.integers(0, 4, n)
    big_q = np.where(seasonal, rng.integers(0, 3, n), 0)
    phi = root_polynomial(rng, p, 10, ar_radius)
    big_phi = root_polynomial(rng, big_p, 2, seasonal_ar_radius)
    theta = root_polynomial(rng, q, 3, ar_radius)
    big_theta = root_polynomial(rng, big_q, 2, seasonal_ar_radius)
    ma_lag = int(max(q.max(initial=0), (big_q * period).max(initial=0)))
    grid = length + arma_warmup(period, big_phi, bool((p > 0).any()), ar_radius, ma_lag)
    noise = rng.standard_normal((n, grid))
    arma = arma_filter(noise, phi, big_phi, theta, big_theta, period)
    d = u(rng, (n, 1), 0.0, 1.0)
    return integrate(arma[:, grid - length :], d, np.where(seasonal, period, 0))


def arma_filter(
    noise: np.ndarray,
    phi: np.ndarray,
    big_phi: np.ndarray,
    theta: np.ndarray,
    big_theta: np.ndarray,
    period: np.ndarray,
) -> np.ndarray:
    """Circular ARMA filter of ``noise`` [n, G]: (phi(B) Phi(B^s)) y = (theta(B) + Theta(B^s) - 1) e, coefficients in ascending powers."""
    grid = noise.shape[1]
    omega = TWO_PI * np.arange(grid // 2 + 1, dtype=np.float64) / grid
    z = np.exp(-1j * omega)[None]
    zs = np.exp(-1j * omega[None] * period[:, None])
    # At most one AR side is non-trivial, so the product equals SarSim's additive lacunary AR polynomial.
    response = (_poly_response(theta, z) + _poly_response(big_theta, zs) - 1.0) / (
        _poly_response(phi, z) * _poly_response(big_phi, zs)
    )
    return np.fft.irfft(np.fft.rfft(noise, axis=1) * response, n=grid, axis=1)


def arma_warmup(
    period: np.ndarray,
    big_phi: np.ndarray,
    plain_active: bool,
    ar_radius: float,
    ma_lag: int,
) -> int:
    """Discarded steps of the circular ARMA synthesis: the largest active MA lag ``ma_lag`` plus the time for the slowest sampled pole to decay by e^-40 (the decay part capped at ``ARMA_MAX_WARMUP``), at least ``ARMA_WARMUP``."""
    plain = 64 + math.ceil(40.0 / -math.log(ar_radius)) if plain_active else 0
    radius = np.where(
        big_phi[:, 2] != 0, np.sqrt(np.abs(big_phi[:, 2])), np.abs(big_phi[:, 1])
    )
    active = radius > 0
    seasonal = 0
    if active.any():
        lags = 2.0 + np.ceil(40.0 / -np.log(radius[active]))
        seasonal = int((period[active] * lags).max())
    # The impulse response runs to the last MA lag, then decays at the AR poles: both must fit before the kept window.
    return max(ARMA_WARMUP, int(ma_lag) + min(max(plain, seasonal), ARMA_MAX_WARMUP))


def integrate(x: np.ndarray, d: np.ndarray, season: np.ndarray) -> np.ndarray:
    """Apply (1 - B)^-d and, on rows with season s >= 2, (1 - B^s)^-1 from zero pre-sample values (linear FFT convolution)."""
    n, length = x.shape
    k = np.arange(1, length, dtype=np.float64)[None]
    psi = np.concatenate([np.ones((n, 1)), np.cumprod((k - 1.0 + d) / k, 1)], 1)
    lags = np.arange(length)[None]
    comb = np.where(
        (season >= 2)[:, None],
        (lags % np.maximum(season, 1)[:, None]) == 0,
        lags == 0,
    ).astype(np.float64)
    size = 2 * length
    kernel = np.fft.irfft(
        np.fft.rfft(psi, n=size, axis=1) * np.fft.rfft(comb, n=size, axis=1),
        n=size,
        axis=1,
    )[:, :length]
    spectrum = np.fft.rfft(x, n=size, axis=1) * np.fft.rfft(kernel, n=size, axis=1)
    return np.fft.irfft(spectrum, n=size, axis=1)[:, :length]


def upsample(env: np.ndarray, factor: np.ndarray, length: int) -> np.ndarray:
    """Linear interpolation of rows of ``env`` at positions t / factor for t < length."""
    pos = np.arange(length, dtype=np.float64)[None] / factor[:, None]
    lo = np.minimum(np.floor(pos).astype(np.int64), env.shape[1] - 2)
    frac = pos - lo
    return (
        np.take_along_axis(env, lo, 1) * (1.0 - frac)
        + np.take_along_axis(env, lo + 1, 1) * frac
    )


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-12)


def noisers(rng: np.random.Generator, x: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """SarSim level-scaled observation noise; rows of kind 0/1/2 become Poisson(lambda_t), Gamma(shape kappa, scale lambda_t)^zeta or LogNormal(lambda_t, kappa) draws, lambda_t = lambda0 * minmax(x); kind -1 passes through."""
    out = x.copy()
    n = x.shape[0]
    lam_wide = logu(rng, (n, 1), 0.1, 100.0)
    lam_narrow = logu(rng, (n, 1), 0.1, 5.0)
    kappa_gamma = logu(rng, (n, 1), 1.0, 50.0)
    kappa_lognormal = logu(rng, (n, 1), 1.0, 3.0)
    zeta = u(rng, (n, 1), 0.5, 1.5)
    rows = np.flatnonzero(kind == 0)
    if rows.size:
        out[rows] = rng.poisson(lam_wide[rows] * _minmax(x[rows])).astype(np.float64)
    rows = np.flatnonzero(kind == 1)
    if rows.size:
        scale = lam_wide[rows] * _minmax(x[rows])
        out[rows] = (
            rng.gamma(kappa_gamma[rows] * np.ones_like(scale), scale) ** zeta[rows]
        )
    rows = np.flatnonzero(kind == 2)
    if rows.size:
        out[rows] = rng.lognormal(
            lam_narrow[rows] * _minmax(x[rows]), kappa_lognormal[rows]
        )
    # A noised row that came out constant (e.g. an all-zero Poisson draw) keeps its pre-noise values.
    flat = (kind >= 0) & (out.max(1) == out.min(1))
    out[flat] = x[flat]
    return out


def sarsim(
    rng: np.random.Generator,
    n: int,
    length: int,
    sarima2_rate: float,
    multiplicative_rate: float,
    noiser_rate: float,
    max_season: int,
    ar_radius: float,
    seasonal_ar_radius: float,
) -> np.ndarray:
    """SarSim draws [n, L]: SARIMA or SARIMA-2 (base plus/times upsampled envelope), then an optional level-scaled noiser."""
    two = rng.random(n) < sarima2_rate
    pair = rng.integers(0, len(SARIMA2_SEASONS), n)
    base_season = np.asarray(SARIMA2_SEASONS)[pair, 0]
    env_season = np.asarray(SARIMA2_SEASONS)[pair, 1]
    free = np.exp(u(rng, n, math.log(4.0), math.log(65.0))).astype(np.int64)
    factor = np.where(base_season >= 2, base_season, free).astype(np.float64)
    season = np.where(two, base_season, rng.integers(0, max_season + 1, n))
    x = sarima_paths(rng, season, length, ar_radius, seasonal_ar_radius)
    rows = np.flatnonzero(two)
    if rows.size:
        env_length = int(math.ceil((length - 1) / factor[rows].min())) + 2
        env = sarima_paths(
            rng, env_season[rows], env_length, ar_radius, seasonal_ar_radius
        )
        env = upsample(env, factor[rows], length)
        mult = rng.random(rows.size) < multiplicative_rate
        depth = u(rng, (rows.size, 1), 0.0, 1.0)
        scaled = 2.0 * _minmax(env) - 1.0
        x[rows] = np.where(
            mult[:, None], (1.0 + depth * scaled) * x[rows], x[rows] + env
        )
    kind = np.where(
        rng.random(n) < noiser_rate, rng.integers(0, len(NOISER_KINDS), n), -1
    )
    return noisers(rng, x, kind)
