"""Numpy kernels of the batched synthetic families ported from the Simons TSFM recipe.

Every family is ``fn(rng, n, length) -> float64 [n, length]`` drawn from a
``numpy.random.Generator``. The draws follow the Simons stage-1 priors
(``simons/synth/{base,nonstat,causal,mech}.py``) in distribution; the random
streams differ (numpy instead of torch), so values match only in law.
"""

import math

import numpy as np

from synforecast._lib import mechanistic as _mechanistic

PERIOD_BANK = (
    4,
    6,
    7,
    10,
    12,
    14,
    24,
    26,
    30,
    40,
    48,
    52,
    60,
    96,
    168,
    288,
    336,
    365,
    672,
    730,
)
MULTIPLES = (2, 4, 7, 12, 24, 52)
CHUNK = 256
TWO_PI = 2 * math.pi


def u(rng: np.random.Generator, shape, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return rng.uniform(lo, hi, shape)


def logu(rng: np.random.Generator, shape, lo: float, hi: float) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo), math.log(hi), shape))


def randn(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.standard_normal(shape)


def choice(rng: np.random.Generator, values, shape) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[rng.integers(len(values), size=shape)]


def categorical(rng: np.random.Generator, probabilities, n: int) -> np.ndarray:
    edges = np.cumsum(probabilities)[:-1]
    return (rng.random((n, 1)) > edges[None]).sum(-1)


def standardise(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(-1, keepdims=True)
    return x / np.maximum(x.std(-1, ddof=1, keepdims=True), 1e-6)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def periods(rng: np.random.Generator, n: int) -> np.ndarray:
    bank = choice(rng, PERIOD_BANK, n)
    free = logu(rng, n, 2.0, 700.0)
    return np.where(rng.random(n) < 0.6, bank, free)


def periodic(
    rng: np.random.Generator, period: np.ndarray, length: int, harmonics: int = 12
) -> np.ndarray:
    """Standardised Fourier series [n, L] with min(12, p/2) harmonics and Gaussian-decaying variance."""
    n = period.shape[0]
    k = np.arange(1, harmonics + 1, dtype=np.float64)
    smooth = u(rng, (n, 1), 1.0, 6.0)
    var = np.exp(-(k**2) / (2 * smooth**2)) * (k[None] <= np.floor(period[:, None] / 2))
    a = randn(rng, (n, harmonics)) * np.sqrt(var)
    b = randn(rng, (n, harmonics)) * np.sqrt(var)
    t = np.arange(length, dtype=np.float64)
    base = np.exp(1j * TWO_PI * np.remainder(t[None] / period[:, None], 1.0))
    power = base.copy()
    x = np.zeros((n, length))
    for h in range(harmonics):
        x += a[:, h : h + 1] * power.real + b[:, h : h + 1] * power.imag
        if h + 1 < harmonics:
            power *= base
    return standardise(x)


def spectral(rng: np.random.Generator, length: int, spectrum) -> np.ndarray:
    """Standardised stationary paths [n, L] with power spectrum ``spectrum(omega) -> [n, L + 1]`` on a 2L grid."""
    grid = 2 * length
    omega = TWO_PI * np.arange(grid // 2 + 1, dtype=np.float64) / grid
    amp = np.sqrt(np.maximum(spectrum(omega), 0.0))
    amp[:, 0] = 0.0
    xi = randn(rng, amp.shape) + 1j * randn(rng, amp.shape)
    return standardise(np.fft.irfft(amp * xi, n=grid)[:, :length])


def smooth_gp(rng: np.random.Generator, n: int, length: int) -> np.ndarray:
    """RBF or Matern GP paths with nu in {0.5, 1.5, 2.5}."""
    scale = logu(rng, (n, 1), 0.003, 1.0) * length
    nu = choice(rng, (0.5, 1.5, 2.5, 100.0), (n, 1))
    capped = np.minimum(nu, 10.0)

    def spectrum(omega):
        w = omega[None] * scale
        matern = (1 + w**2 / (2 * capped)) ** (-(capped + 0.5))
        return np.where(nu > 10, np.exp(-(w**2) / 2), matern)

    return spectral(rng, length, spectrum)


def slow(
    rng: np.random.Generator, n: int, length: int, lo: float = 0.02, hi: float = 0.5
) -> np.ndarray:
    scale = logu(rng, (n, 1), lo, hi) * length
    return spectral(rng, length, lambda w: np.exp(-((w[None] * scale) ** 2) / 2))


def kernel_paths(
    rng: np.random.Generator, n: int, length: int, max_components: int = 5
) -> np.ndarray:
    """KernelSynth-style sums and products of periodic, GP, noise and trend components [n, L]."""
    slots = max_components
    m = n * slots
    t = np.arange(length, dtype=np.float64) / length
    count = rng.integers(1, slots + 1, n)
    kind = categorical(rng, (0.4, 0.25, 0.15, 0.2), m)
    period = periods(rng, m).reshape(slots, n)
    multiple = period[:1] * choice(rng, MULTIPLES, (slots, n))
    harmonic = rng.random((slots, n)) < 0.4
    harmonic[0] = False
    period = np.where(harmonic, multiple, period).reshape(-1)
    sigma = np.where(rng.random((m, 1)) < 0.5, 0.1, 1.0)
    slope = randn(rng, (m, 1)) * 2
    ramp = u(rng, (m, 1), 0.05, 1.0) + t[None] * np.sign(slope)
    ramp = ramp - np.minimum(ramp.min(-1, keepdims=True), 0.0)
    amplitude = logu(rng, (m, 1), 0.1, 3.0)
    component = slope * (t[None] - 0.5) * 2 * np.ones((m, 1))
    rows = kind == 0
    if rows.any():
        component[rows] = periodic(rng, period[rows], length) * amplitude[rows]
    rows = kind == 1
    if rows.any():
        component[rows] = smooth_gp(rng, int(rows.sum()), length) * amplitude[rows]
    rows = kind == 2
    if rows.any():
        component[rows] = randn(rng, (int(rows.sum()), length)) * sigma[rows]
    factor = np.where(kind[:, None] == 3, ramp, component / amplitude)
    component = component.reshape(slots, n, length)
    factor = factor.reshape(slots, n, length)
    product = rng.random((slots, n, 1)) < 0.2
    acc = component[0]
    for slot in range(1, slots):
        combined = np.where(product[slot], acc * factor[slot], acc + component[slot])
        acc = np.where((slot < count)[:, None], combined, acc)
    return acc


def trend(rng: np.random.Generator, n: int, length: int) -> np.ndarray:
    s = np.arange(length, dtype=np.float64)[None] / length
    kind = rng.integers(0, 4, (n, 1))
    c1, c2 = randn(rng, (n, 1)), randn(rng, (n, 1))
    poly = c1 * (s - 0.5) * 2 + c2 * ((s - 0.5) * 2) ** 2
    centre, rate = u(rng, (n, 1), 0.1, 0.9), logu(rng, (n, 1), 5.0, 50.0)
    logistic = 2 * c1 * sigmoid(rate * (s - centre))
    cycles = logu(rng, (n, 1), 0.3, 3.0)
    sine = c1 * np.sin(TWO_PI * cycles * s + u(rng, (n, 1), 0, TWO_PI))
    knot = u(rng, (n, 1), 0.1, 0.9)
    piece = c1 * s + (c2 - c1) * np.maximum(s - knot, 0.0)
    return np.where(
        kind == 0, poly, np.where(kind == 1, logistic, np.where(kind == 2, sine, piece))
    )


def ou_scan(a: np.ndarray, drive: np.ndarray, y0: np.ndarray) -> np.ndarray:
    """Solve y_t = a_t y_{t-1} + drive_t from y_{-1} = y0 in float64 chunks (a_t >= e^-1)."""
    n, length = a.shape
    pad = (-length) % CHUNK
    a = np.pad(a, ((0, 0), (0, pad)), constant_values=1.0).reshape(n, -1, CHUNK)
    drive = np.pad(drive, ((0, 0), (0, pad))).reshape(n, -1, CHUNK)
    log_p = np.cumsum(np.log(a), -1)
    p = np.exp(log_p)
    local = p * np.cumsum(drive * np.exp(-log_p), -1)
    carry = np.asarray(y0, dtype=np.float64).copy()
    starts = np.empty((n, a.shape[1]))
    for c in range(a.shape[1]):
        starts[:, c] = carry
        carry = local[:, c, -1] + p[:, c, -1] * carry
    return (local + p * starts[..., None]).reshape(n, -1)[:, :length]


def regime_ou(rng: np.random.Generator, n: int, length: int) -> np.ndarray:
    """Regime-switching OU paths with drifting mean, mean-reversion and volatility (Simons ``sde``)."""
    t = np.arange(length, dtype=np.float64)[None]
    hazard = logu(rng, (n, 1), 1 / 8192, 1 / 256)
    switches = np.cumsum(rng.random((n, length)) < hazard, -1)
    first = rng.integers(0, 2, (n, 1))
    regime = ((switches + first) % 2).astype(bool)
    theta = np.where(regime, u(rng, (n, 1), 0.05, 0.5), u(rng, (n, 1), 1.0, 5.0))
    side = np.where(rng.random((n, 1)) < 0.5, -1.0, 1.0)
    mu0 = side * 2 + randn(rng, (n, 1))
    mu1 = -side * 2 + randn(rng, (n, 1))
    mu = np.where(regime, mu1, mu0)
    s0 = 0.3 * np.exp(0.5 * randn(rng, (n, 1)))
    s1 = 1.5 * np.exp(0.5 * randn(rng, (n, 1)))
    sigma = np.where(regime, s1, s0)

    def use(p):
        return rng.random((n, 1)) < p

    mu = mu + use(0.7) * 2 * trend(rng, n, length)
    period = choice(rng, PERIOD_BANK, n)
    mu = mu + use(0.6) * logu(rng, (n, 1), 0.2, 2.0) * periodic(rng, period, length)
    theta = theta * np.exp(use(0.2) * randn(rng, (n, 1)) * (t / length - 0.5) * 2)
    sigma = sigma * np.exp(use(0.3) * 0.7 * np.clip(trend(rng, n, length), -2, 2))
    period2 = choice(rng, PERIOD_BANK, n)
    phase = u(rng, (n, 1), 0, TWO_PI)
    sigma = sigma * np.exp(
        use(0.3)
        * u(rng, (n, 1), 0.2, 0.8)
        * np.sin(TWO_PI * t / period2[:, None] + phase)
    )
    sigma = sigma * np.exp(
        use(0.5) * u(rng, (n, 1), 0.0, 0.7) * smooth_gp(rng, n, length)
    )
    hurst = u(rng, (n, 1), 0.3, 0.8)
    fgn = spectral(rng, length, lambda w: np.maximum(w[None], 1e-4) ** (1 - 2 * hurst))
    eps = np.where(use(0.2), fgn, randn(rng, (n, length)))
    dt = logu(rng, (n, 1), 0.005, 0.05)
    theta = np.minimum(np.maximum(theta, 1e-3), 0.25 / dt)
    a = np.exp(-theta * dt)
    drive = (1 - a) * mu + sigma * np.sqrt((1 - a**2) / (2 * theta)) * eps
    y0 = mu[:, 0] + sigma[:, 0] / np.sqrt(2 * theta[:, 0]) * randn(rng, n)
    return ou_scan(a, drive, y0)


def cauker(
    rng: np.random.Generator,
    n: int,
    length: int,
    max_nodes: int = 8,
    max_roots: int = 4,
) -> np.ndarray:
    """Causal-DAG series: kernel roots with mean functions, nonlinear children, one node observed."""
    t = np.arange(length, dtype=np.float64)[None] / length
    roots = standardise(
        kernel_paths(rng, max_roots * n, length, max_components=3)
    ).reshape(max_roots, n, length)
    m = max_roots * n
    kind = rng.integers(0, 3, (m, 1))
    use = rng.random((m, 1)) < 0.5
    a, b = randn(rng, (m, 1)), randn(rng, (m, 1))
    linear = a * 2 * (t - 0.5) + 0.5 * b
    rate = u(rng, (m, 1), -2.0, 2.0)
    expo = a * np.exp(rate * t)
    expo = expo - expo.mean(-1, keepdims=True)
    impulses = (rng.random((m, length)) < 0.002) * u(rng, (m, length), -5.0, 5.0)
    mean = np.where(kind == 0, linear, np.where(kind == 1, expo, impulses))
    roots = roots + (use * mean).reshape(max_roots, n, length)
    count = rng.integers(2, max_roots + 1, (n, 1))
    nodes: list[np.ndarray] = []
    for v in range(max_nodes):
        root = roots[v] if v < max_roots else None
        if v < 2:
            nodes.append(root)
            continue
        stack = np.stack(nodes, 1)
        parents = rng.integers(1, 4, (n, 1))
        index = np.minimum((rng.random((n, 3)) * v).astype(np.int64), v - 1)
        valid = np.arange(3)[None] < parents
        weight = randn(rng, (n, 3)) * valid / np.sqrt(parents)
        picked = np.take_along_axis(stack, index[..., None], 1)
        z = (weight[..., None] * picked).sum(1) + randn(rng, (n, 1))
        act = categorical(rng, (0.2, 0.25, 0.15, 0.2, 0.15, 0.05), n)[:, None]
        slope = u(rng, (n, 1), 0.01, 0.3)
        k = logu(rng, (n, 1), 0.5, 3.0)
        omega = logu(rng, (n, 1), 0.5, 3.0)
        modulus = u(rng, (n, 1), 1.0, 5.0)
        z = np.select(
            [act == 0, act == 1, act == 2, act == 3, act == 4],
            [
                z,
                np.maximum(z, 0.0),
                np.where(z > 0, z, slope * z),
                sigmoid(k * z),
                np.sin(omega * z),
            ],
            np.remainder(z, modulus),
        )
        z = standardise(z)
        if root is not None:
            z = np.where(v < count, root, z)
        nodes.append(z)
    stack = np.stack(nodes, 1)
    pick = count + (rng.random((n, 1)) * (max_nodes - count)).astype(np.int64)
    pick = np.minimum(pick, max_nodes - 1)
    out = np.take_along_axis(stack, pick[..., None], 1)[:, 0]
    out = np.where(out.std(-1, ddof=1, keepdims=True) < 1e-3, roots[0], out)
    noise = logu(rng, (n, 1), 0.003, 0.1) * (rng.random((n, 1)) < 0.5)
    return out + noise * randn(rng, (n, length))


def polyrhythm(
    rng: np.random.Generator, n: int, length: int, layers: int = 5
) -> np.ndarray:
    """Layered beat trains with decaying envelopes, optional carrier, drift and noise."""
    tf = np.arange(length, dtype=np.float64)[None]
    bank = choice(rng, PERIOD_BANK, (n, 1))
    period = np.where(rng.random((n, 1)) < 0.6, bank, logu(rng, (n, 1), 4.0, 400.0))
    count = rng.integers(3, layers + 1, (n, 1))
    m = n * layers
    p = np.tile(period, (layers, 1))
    k = choice(rng, (1.0, 2.0, 3.0, 4.0, 6.0, 8.0), (m, 1))
    spacing = np.maximum(p / k, 2.0)
    offset = rng.random((m, 1)) * spacing
    beat = np.floor((tf - offset) / spacing)
    fired = beat > np.floor((tf - 1 - offset) / spacing)
    q = u(rng, (m, 1), 0.3, 1.0)
    fired = fired & (rng.random((m, length)) < q)
    amp = logu(rng, (m, 1), 0.2, 5.0)
    jitter = np.exp(0.25 * randn(rng, (m, length)) * (rng.random((m, 1)) < 0.5))
    impulse = fired * amp * jitter
    tau = np.maximum(np.exp(rng.random((m, 1)) * np.log(np.maximum(spacing, 1.0))), 1.0)
    decay = np.broadcast_to(np.exp(-1.0 / tau), (m, length))
    env = ou_scan(decay, impulse, np.zeros(m)).reshape(layers, n, length)
    active = np.arange(layers)[:, None, None] < count[None]
    y = (env * active).sum(0)
    carrier = 1 + 0.5 * np.sin(
        TWO_PI * tf / (period * choice(rng, (2.0, 4.0, 7.0, 12.0), (n, 1)))
        + u(rng, (n, 1), 0, TWO_PI)
    )
    y = np.where(rng.random((n, 1)) < 0.3, y * carrier, y)
    scale = np.maximum(y.std(-1, ddof=1, keepdims=True), 1e-3)
    drift = np.where(
        rng.random((n, 1)) < 0.5, trend(rng, n, length), slow(rng, n, length)
    )
    y = y + (rng.random((n, 1)) < 0.4) * u(rng, (n, 1), 0.1, 1.0) * scale * drift
    noisy = rng.random((n, 1)) < 0.6
    return y + noisy * logu(rng, (n, 1), 0.003, 0.2) * scale * randn(rng, (n, length))


def amfm_wave(
    rng: np.random.Generator, n: int, length: int, components: int = 3
) -> np.ndarray:
    """Sums of sine/square/triangle/saw waves with slowly modulated frequency and amplitude."""
    m = n * components
    tf = np.arange(length, dtype=np.float64)[None]
    period = np.maximum(periods(rng, m)[:, None], 3.0)
    delta = u(rng, (m, 1), 0.0, 0.3)
    eps = u(rng, (m, 1), 0.0, 0.5)
    freq = np.exp(delta * slow(rng, m, length)) / period
    cycles = np.remainder(np.cumsum(freq, -1) + rng.random((m, 1)), 1.0)
    sine = np.sin(TWO_PI * cycles)
    shape = rng.integers(0, 4, (m, 1))
    kappa = logu(rng, (m, 1), 1.0, 10.0)
    square = np.tanh(kappa * sine) / np.tanh(kappa)
    triangle = 2 / math.pi * np.arcsin(np.clip(sine, -1, 1))
    saw = 2 * cycles - 1
    wave = np.where(
        shape == 0,
        sine,
        np.where(shape == 1, square, np.where(shape == 2, triangle, saw)),
    )
    amplitude = logu(rng, (m, 1), 0.2, 2.0) * np.exp(eps * slow(rng, m, length))
    count = rng.integers(1, components + 1, (n, 1))
    active = np.arange(components)[:, None, None] < count[None]
    y = ((amplitude * wave).reshape(components, n, length) * active).sum(0)
    line = randn(rng, (n, 1)) * (tf / length - 0.5) * 2
    y = y + (rng.random((n, 1)) < 0.5) * line
    return y + logu(rng, (n, 1), 0.005, 0.3) * randn(rng, (n, length))


_GRANULARITY = np.array(
    ((24.0, 168.0), (7.0, 365.25), (52.18, 0.0), (12.0, 0.0), (24.0, 0.0), (7.0, 0.0)),
    dtype=np.float64,
)


def forecast_pfn(
    rng: np.random.Generator, n: int, length: int, harmonics: int = 3
) -> np.ndarray:
    """Positive trend x seasonal x (1 + Weibull noise) series (ForecastPFN prior)."""
    tf = np.arange(length, dtype=np.float64)[None]
    s = tf / length
    pair = _GRANULARITY[rng.integers(0, len(_GRANULARITY), n)]
    m_l = np.maximum(randn(rng, (n, 1)), -0.9)
    log_me = randn(rng, (n, 1)) * 0.7
    level_trend = (1 + m_l * s) * np.exp(log_me * s)
    level_trend = np.where(rng.random((n, 1)) < 0.2, 1.0, level_trend)
    season = np.ones((n, length))
    start = u(rng, (n, 1), 0.0, 1000.0)
    for j in range(2):
        p = pair[:, j : j + 1]
        used = p > 0
        p = np.maximum(p, 1.0)
        level = u(rng, (n, 1), 0.0, 0.4)
        acc = np.zeros((n, length))
        for f in range(1, harmonics + 1):
            a = randn(rng, (n, 1)) * level / f
            b = randn(rng, (n, 1)) * level / f
            arg = TWO_PI * f * np.remainder((tf + start) / p, 1.0)
            acc = acc + a * np.sin(arg) + b * np.cos(arg)
        season = season * np.where(used, np.maximum(1 + acc, 0.05), 1.0)
    shape = u(rng, (n, 1), 1.0, 5.0)
    uu = u(rng, (n, length), 1e-6, 1 - 1e-6)
    weibull = (-np.log1p(-uu)) ** (1 / shape)
    mean = np.array([math.gamma(1 + 1 / v) for v in shape[:, 0]])[:, None]
    level = logu(rng, (n, 1), 0.01, 0.3)
    noise = np.maximum(1 + level * (weibull - mean), 0.01)
    return level_trend * season * noise


def ar1(
    rng: np.random.Generator, n: int, length: int, phi: np.ndarray, sd: np.ndarray
) -> np.ndarray:
    """AR(1) paths [n, L] from zero with per-row phi and asymptotic sd [n, 1]."""
    z = randn(rng, (n, length))
    a = np.broadcast_to(phi, (n, length))
    return ou_scan(a, z * np.sqrt(1 - phi**2), np.zeros(n)) * sd


def epidemic_paths(
    rng: np.random.Generator, n: int, length: int, burn: int = 512
) -> np.ndarray:
    """Stochastic SEIRS/SIRS paths with seasonal forcing, interventions, reporting delay and weekday effects.

    Returns [n, length, 2]: reported cases and an occupancy proxy.
    """
    total = burn + length
    t = np.arange(total, dtype=np.float64)[None]
    weekly = rng.random((n, 1)) < 0.35
    dt = np.where(weekly, 7.0, 1.0)
    year = 365.25 / dt
    pop = logu(rng, (n, 1), 1e4, 1e7)
    gamma = 1 / u(rng, (n, 1), 2.0, 10.0)
    sir = rng.random((n, 1)) < 0.3
    sigma = np.where(sir, 50.0, 1 / u(rng, (n, 1), 1.0, 6.0))
    r0 = u(rng, (n, 1), 1.1, 4.0)
    waning = np.where(
        rng.random((n, 1)) < 0.7, 1 / logu(rng, (n, 1), 60.0, 2000.0), 0.0
    )
    births = np.where(
        (rng.random((n, 1)) < 0.4) | (waning == 0),
        1 / logu(rng, (n, 1), 5 * 365.0, 70 * 365.0),
        0.0,
    )
    season = np.ones((n, total))
    for k, (hi, p) in enumerate(((0.45, 1.0), (0.15, 0.5), (0.1, 0.3))):
        a = u(rng, (n, 1), 0.0, hi) * (rng.random((n, 1)) < p)
        ph = u(rng, (n, 1), 0.0, TWO_PI)
        season = season + a * np.cos(TWO_PI * (k + 1) * t / year + ph)
    logm = np.zeros((n, total))
    count = rng.integers(0, 5, (n, 1))
    for e in range(4):
        start = u(rng, (n, 1), 0.0, float(total))
        dur = logu(rng, (n, 1), 10.0, 600.0) / dt
        m = np.log(logu(rng, (n, 1), 0.35, 1.6))
        logm = logm + m * ((t >= start) & (t < start + dur) & (count > e))
    od = u(rng, (n, 1), 0.0, 0.4)
    noise = np.exp(od * randn(rng, (n, total)) - od**2 / 2)
    beta = (r0 * gamma) * np.maximum(season, 0.05) * np.exp(logm) * noise * dt / pop
    imports = rng.poisson(
        np.broadcast_to(logu(rng, (n, 1), 0.01, 3.0) * dt, (n, total))
    ).astype(np.float64)
    z = randn(rng, (n, total, 4))
    p_const = 1 - np.exp(-np.concatenate([sigma, gamma, waning], 1) * dt)
    p_birth = 1 - np.exp(-births * dt)
    s0 = u(rng, (n, 1), 0.2, 1.0)
    i0 = logu(rng, (n, 1), 1e-5, 1e-2)
    x = np.concatenate([s0 * (1 - i0), i0 / 2, i0 / 2, (1 - s0) * (1 - i0)], 1) * pop
    inc = np.empty((n, length))
    prev = np.empty((n, length))
    for r in range(n):
        inc[r], prev[r] = _mechanistic.epidemic(
            x[r].tolist(),
            np.ascontiguousarray(beta[r]),
            p_const[r].tolist(),
            float(p_birth[r, 0]),
            float(pop[r, 0]),
            np.ascontiguousarray(imports[r]),
            np.ascontiguousarray(z[r]).reshape(-1),
            burn,
            length,
        )
    tt = t[:, :length]
    r_a, r_b = logu(rng, (n, 1), 0.02, 0.8), logu(rng, (n, 1), 0.02, 0.8)
    centre, width = u(rng, (n, 1), 0.0, float(length)), logu(rng, (n, 1), 5.0, 500.0)
    report = r_a + (r_b - r_a) * sigmoid((tt - centre) / width)
    q = u(rng, (n, 1), math.exp(-0.25), 0.97)
    inc = np.clip(np.where(np.isfinite(inc), inc, 0.0), 0, 1e9)
    delayed = ou_scan(np.broadcast_to(q, (n, length)), (1 - q) * inc, np.zeros(n))
    delayed = np.where(rng.random((n, 1)) < 0.6, delayed, inc)
    wd = np.exp(u(rng, (n, 1), 0.0, 0.4) * randn(rng, (n, 7)))
    wd = np.where(weekly, 1.0, wd)
    weekday = np.take_along_axis(
        wd, np.broadcast_to(tt.astype(np.int64) % 7, (n, length)), 1
    )
    obs_sd = u(rng, (n, 1), 0.0, 0.5)
    mean = (
        report
        * delayed
        * weekday
        * np.exp(obs_sd * randn(rng, (n, length)) - obs_sd**2 / 2)
    )
    cases = rng.poisson(
        np.clip(np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0), 0, 1e8)
    ).astype(np.float64)
    occupancy = (
        prev * logu(rng, (n, 1), 0.001, 0.3) * np.exp(0.05 * randn(rng, (n, length)))
    )
    return np.stack([cases, occupancy], -1)


def rainfall_runoff_paths(
    rng: np.random.Generator, n: int, length: int, burn: int = 512
) -> np.ndarray:
    """Bucket rainfall-runoff model with Markov rain, snow, evaporation and routing.

    Returns [n, length, 2]: streamflow and storage (soil moisture plus snow).
    """
    total = burn + length
    t = np.arange(total, dtype=np.float64)[None]
    hourly = rng.random((n, 1)) < 0.4
    per_day = np.where(hourly, 24.0, 1.0)
    year = 365.25 * per_day
    seas = np.cos(TWO_PI * t / year + u(rng, (n, 1), 0.0, TWO_PI))
    p01 = np.where(hourly, logu(rng, (n, 1), 0.003, 0.04), logu(rng, (n, 1), 0.03, 0.4))
    p11 = np.where(hourly, u(rng, (n, 1), 0.6, 0.95), u(rng, (n, 1), 0.2, 0.8))
    p01_t = np.clip(p01 * (1 + u(rng, (n, 1), 0.0, 0.8) * seas), 0, 1)
    shape = u(rng, (n, 1), 0.5, 1.3)
    amount = rng.gamma(np.broadcast_to(shape, (n, total))) / shape
    amount = amount * np.where(
        hourly, u(rng, (n, 1), 0.5, 3.0), u(rng, (n, 1), 3.0, 15.0)
    )
    amount = amount * (1 + u(rng, (n, 1), 0.0, 0.5) * seas)
    occ = rng.random((n, total))
    snowy = rng.random((n, 1)) < 0.35
    temp = (
        u(rng, (n, 1), -5.0, 12.0)
        + u(rng, (n, 1), 5.0, 15.0) * seas
        + np.where(hourly, u(rng, (n, 1), 0.0, 6.0), 0.0)
        * np.cos(TWO_PI * t / 24 + u(rng, (n, 1), 0.0, 6.3))
        + ar1(rng, n, total, np.where(hourly, 0.99, 0.8), np.full((n, 1), 3.0))
    )
    snow_frac = np.where(snowy, sigmoid(-temp), 0.0)
    melt_rate = u(rng, (n, 1), 1.0, 5.0) / per_day * np.maximum(temp, 0.0) * snowy
    smax = logu(rng, (n, 1), 20.0, 300.0)
    k = logu(rng, (n, 1), 2e-3, 0.15) / per_day
    a = u(rng, (n, 1), 1.0, 3.0)
    pet = u(rng, (n, 1), 0.5, 5.0) / per_day * np.maximum(1 - 0.8 * seas, 0.0)
    pet = pet * np.where(
        hourly, np.maximum(math.pi * np.sin(TWO_PI * t / 24), 0.0), 1.0
    )
    alpha = np.where(hourly, logu(rng, (n, 1), 0.01, 0.3), logu(rng, (n, 1), 0.1, 0.9))
    fg = u(rng, (n, 1), 0.1, 0.7)
    g = logu(rng, (n, 1), 2e-4, 2e-2) / per_day * np.where(hourly, 4.0, 1.0)
    soil = smax[:, 0] * u(rng, n, 0.2, 0.8)
    flow = np.empty((n, length))
    moist = np.empty((n, length))
    for r in range(n):
        flow[r], moist[r] = _mechanistic.rainfall_runoff(
            float(soil[r]),
            [float(v[r, 0]) for v in (smax, k, a, alpha, fg, g, p11)],
            *(
                np.ascontiguousarray(np.broadcast_to(v[r], (total,)), dtype=np.float64)
                for v in (occ, p01_t, amount, snow_frac, melt_rate, pet)
            ),
            burn,
            length,
        )
    return np.stack([flow + 1e-3 * smax, moist], -1)


def thermal_paths(
    rng: np.random.Generator, n: int, length: int, burn: int = 512
) -> np.ndarray:
    """Two-node thermal network with load-driven heat and hysteresis or proportional cooling.

    Returns [n, length, 2]: node temperature and energy use.
    """
    total = burn + length
    t = np.arange(total, dtype=np.float64)[None]
    fine = rng.random((n, 1)) < 0.35
    day = np.where(fine, 96.0, 24.0)
    year = 365.25 * day
    ambient = (
        u(rng, (n, 1), -5.0, 25.0)
        + u(rng, (n, 1), 0.0, 12.0)
        * np.cos(TWO_PI * t / year + u(rng, (n, 1), 0.0, 6.3))
        + u(rng, (n, 1), 0.0, 6.0) * np.cos(TWO_PI * t / day + u(rng, (n, 1), 0.0, 6.3))
        + ar1(rng, n, total, np.where(fine, 0.995, 0.98), u(rng, (n, 1), 0.5, 3.0))
    )
    hour = TWO_PI * t / day
    profile = (
        1
        + u(rng, (n, 1), 0.0, 0.6) * np.cos(hour + u(rng, (n, 1), 0.0, 6.3))
        + u(rng, (n, 1), 0.0, 0.3) * np.cos(2 * hour + u(rng, (n, 1), 0.0, 6.3))
    )
    weekend = (np.floor(t / day) % 7) >= 5
    profile = profile * np.where(weekend, 1 - u(rng, (n, 1), 0.0, 0.5), 1.0)
    load = np.maximum(
        profile
        * np.exp(
            ar1(rng, n, total, np.where(fine, 0.99, 0.95), u(rng, (n, 1), 0.05, 0.4))
        ),
        0.0,
    )
    quad = rng.random((n, 1)) < 0.5
    heat = np.where(quad, load**2, load) * u(rng, (n, 1), 0.02, 0.5)
    r12 = logu(rng, (n, 1), 0.01, 0.3)
    r1a = logu(rng, (n, 1), 0.003, 0.08)
    r2a = logu(rng, (n, 1), 0.001, 0.05)
    cap = logu(rng, (n, 1), 0.1, 1.0)
    cooled = rng.random((n, 1)) < 0.8
    prop = rng.random((n, 1)) < 0.3
    setpoint = ambient.mean(1, keepdims=True) + u(rng, (n, 1), 2.0, 15.0)
    band = u(rng, (n, 1), 0.3, 3.0)
    power = (heat.mean(1, keepdims=True) + r1a * 10) * u(rng, (n, 1), 1.2, 3.0) * cooled
    temp = np.empty((n, length))
    energy = np.empty((n, length))
    for r in range(n):
        temp[r], energy[r] = _mechanistic.thermal(
            float(ambient[r, 0] + 5),
            [float(v[r, 0]) for v in (r12, r1a, r2a, cap, setpoint, band, power)],
            bool(prop[r, 0]),
            np.ascontiguousarray(ambient[r]),
            np.ascontiguousarray(heat[r]),
            burn,
            length,
        )
    return np.stack([temp, energy], -1)


def fluid_queue(rng: np.random.Generator, n: int, length: int) -> np.ndarray:
    """Fluid-queue observables (utilisation, backlog, latency, throughput or speed) under daily load."""
    t = np.arange(length, dtype=np.float64)[None]
    day = np.array([288.0, 96.0, 24.0])[rng.integers(0, 3, (n, 1))]
    hour = TWO_PI * (t + u(rng, (n, 1), 0.0, 1e4)) / day
    logp = (
        u(rng, (n, 1), 0.0, 1.0) * np.cos(hour)
        + u(rng, (n, 1), 0.0, 0.4) * np.cos(2 * hour + u(rng, (n, 1), 0.0, 6.3))
        - u(rng, (n, 1), 0.0, 0.6) * ((np.floor(t / day) % 7) >= 5)
        + ar1(rng, n, length, u(rng, (n, 1), 0.9, 0.999), u(rng, (n, 1), 0.02, 0.3))
    )
    hazard = logu(rng, (n, 1), 1e-3, 3e-2)
    regime = (np.cumsum(rng.random((n, length)) < hazard, -1) % 2).astype(bool)
    regime = regime & (rng.random((n, 1)) < 0.6)
    logp = logp + np.log(logu(rng, (n, 1), 1.2, 3.0)) * regime
    profile = np.exp(logp - logp.mean(1, keepdims=True))
    cap = logu(rng, (n, 1), 5.0, 500.0)
    dip = (np.cumsum(rng.random((n, length)) < 5e-4, -1) % 2).astype(bool)
    dip = dip & (rng.random((n, 1)) < 0.3)
    cap_t = cap * np.where(dip, u(rng, (n, 1), 0.3, 0.8), 1.0)
    rho = u(rng, (n, 1), 0.5, 1.1)
    arrivals = rng.poisson(np.clip(cap * rho * profile, 0, 1e6)).astype(np.float64)
    s = np.cumsum(arrivals - cap_t, -1)
    backlog = s - np.minimum(np.minimum.accumulate(s, -1), 0.0)
    before = np.concatenate([np.zeros((n, 1)), backlog[:, :-1]], 1)
    served = np.minimum(before + arrivals, cap_t)
    util = served / cap_t
    latency = (backlog / cap_t + 1) * np.exp(0.1 * randn(rng, (n, length)))
    occ = np.clip((arrivals + backlog) / (cap * u(rng, (n, 1), 1.5, 3.0)), 0.01, 0.98)
    crit = u(rng, (n, 1), 0.2, 0.5)
    speed = np.where(occ < crit, 1.0, crit / (1 - crit) * (1 - occ) / occ) * u(
        rng, (n, 1), 40, 120
    )
    kind = rng.integers(0, 5, (n, 1))
    return np.select(
        [kind == 0, kind == 1, kind == 2, kind == 3],
        [util, backlog, latency, served],
        speed,
    )


def valid_paths(x: np.ndarray) -> np.ndarray:
    """Rows [n] whose first channel is finite, bounded and still moving in its second half."""
    x0 = x[..., 0]
    ok = np.isfinite(x0).all(1) & (
        np.abs(np.where(np.isfinite(x0), x0, 0.0)).max(1) < 6e4
    )
    x0 = np.where(np.isfinite(x0), x0, 0.0)
    tail = x0[:, x0.shape[1] // 2 :].std(1, ddof=1)
    return (
        ok
        & (tail > 0.05 * x0.std(1, ddof=1))
        & (tail > 1e-6 * (np.abs(x0).mean(1) + 1e-6))
    )


def rms_normalise(x: np.ndarray) -> np.ndarray:
    """Finite, clipped multichannel paths [n, L, C] divided by each channel's RMS."""
    x = np.clip(np.where(np.isfinite(x), x, 0.0), -6e4, 6e4)
    rms = np.maximum(np.sqrt((x**2).mean(1, keepdims=True)), 1e-12)
    return np.clip(x / rms, -6e4, 6e4)


def crop(rng: np.random.Generator, path: np.ndarray, length: int) -> np.ndarray:
    """A random channel of one simulated path [size, C], strided (pooled or decimated) and mirrored if short."""
    size, channels = path.shape
    if length > 2 * size:
        raise ValueError(
            f"{length}-step series need a simulation length of at least {(length + 1) // 2}"
        )
    span = min(length, size)
    ch = int(rng.integers(0, channels))
    max_stride = max(1, min(4, size // span))
    stride = int(rng.integers(1, max_stride + 1))
    pool = rng.random() < 0.5
    room = max(size - stride * span, 0)
    offset = min(int(rng.random() * (room + 1)), size - 1)
    pos = offset + stride * np.arange(span)
    x = np.zeros(span)
    for j in range(max_stride):
        weight = float(j < stride) / stride if pool else float(j == 0)
        x = x + weight * path[np.minimum(pos + j, size - 1), ch]
    if span < length:
        x = np.concatenate([x[::-1], x])[-length:]
    return x


def observe(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    """Measurement layer: additive or multiplicative noise and an occasional trend, on [n, L]."""
    n, length = x.shape
    z = standardise(x)
    level = u(rng, (n, 1), 0.0, 0.1) * (rng.random((n, 1)) < 0.7)
    eps = randn(rng, (n, length))
    mult = rng.random((n, 1)) < 0.3
    base = x / np.maximum(x.std(1, ddof=1, keepdims=True), 1e-6)
    out = np.where(mult, base * (1 + level * eps), z + level * eps)
    t = np.linspace(0, 1, length)[None]
    drift = u(rng, (n, 1), -1.0, 1.0) * t + u(rng, (n, 1), 0.0, 0.5) * np.sin(
        TWO_PI * t * u(rng, (n, 1), 0.2, 1.5)
    )
    return out + drift * (rng.random((n, 1)) < 0.2)


SIMULATORS = {
    "epidemic": epidemic_paths,
    "rainfall_runoff": rainfall_runoff_paths,
    "thermal": thermal_paths,
}
MECHANISTIC_COUNTERS = {
    name: {"draws": 0, "redraws": 0, "fallbacks": 0} for name in SIMULATORS
}


def mechanistic(
    rng: np.random.Generator,
    name: str,
    n: int,
    length: int,
    simulation_length: int,
    burn: int,
    max_attempts: int,
) -> np.ndarray:
    """Per-draw mechanistic series: simulate ``simulation_length`` steps, keep a valid path, crop, observe.

    The RMS-normalised path is rounded to float16 before cropping, as Simons stores its trajectory bank.
    """
    counters = MECHANISTIC_COUNTERS[name]
    out = np.empty((n, length))
    for r in range(n):
        for attempt in range(max_attempts):
            path = SIMULATORS[name](rng, 1, simulation_length, burn)
            if valid_paths(path)[0]:
                break
            if attempt + 1 < max_attempts:
                counters["redraws"] += 1
            else:
                counters["fallbacks"] += 1
        counters["draws"] += 1
        stored = rms_normalise(path)[0].astype(np.float16).astype(np.float64)
        out[r] = crop(rng, stored, length)
    return finite(observe(rng, out))


def finite(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, 0.0)
