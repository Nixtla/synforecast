"""KernelSynth generator: sample from randomly composed GP kernels."""

import numpy as np
from pydantic import Field, model_validator

from synforecast.base import BaseGenerator

# Divergence guards, mirroring the other pretraining generators (TSI/TCM).
_MAX_ABS = 1e8
_MIN_STD = 1e-8
_MAX_RETRIES = 8
# Tenfold jitter escalations on a Levinson breakdown before the composition is redrawn.
_MAX_REGULARIZATIONS = 4

# Shared with rust/src/generators/kernel_synth.rs.
_KIND_IDS = {
    "rbf": 0,
    "rational_quadratic": 1,
    "periodic": 2,
    "linear": 3,
    "white": 4,
    "constant": 5,
}

Terms = list[np.ndarray | None]


def _add_terms(a: Terms, b: Terms) -> Terms:
    """Sum of two kernel polynomials in ``x x'``, coefficient by coefficient."""
    out: Terms = []
    for degree in range(max(len(a), len(b))):
        left = a[degree] if degree < len(a) else None
        right = b[degree] if degree < len(b) else None
        if left is None or right is None:
            out.append(right if left is None else left)
        else:
            out.append(left + right)
    return out


def _multiply_terms(a: Terms, b: Terms) -> Terms:
    """Product of two kernel polynomials: degrees add, lag coefficients multiply elementwise."""
    out: Terms = [None] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        if left is None:
            continue
        for j, right in enumerate(b):
            if right is None:
                continue
            product = left * right
            out[i + j] = product if out[i + j] is None else out[i + j] + product
    return out


class KernelSynthGenerator(BaseGenerator):
    """Generate series by sampling from randomly composed Gaussian-process kernels.

    This adapts the KernelSynth recipe introduced for pretraining the Chronos
    forecasting models (Ansari et al. 2024,
    "Chronos: Learning the Language of Time Series",
    https://arxiv.org/abs/2403.07815) and its Apache-2.0-licensed reference
    implementation (https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/kernel-synth.py).
    For each series the
    generator draws ``1..max_kernels`` base kernels (with replacement) from a
    fixed bank, folds them together with randomly chosen binary operators
    (``+`` or ``*``), and samples one path from the resulting GP prior on the
    normalized grid ``x = linspace(0, 1, length)``. Kernel addition mixes
    behaviors (e.g. trend + seasonality); kernel multiplication modulates them
    (e.g. locally periodic, amplitude-varying seasonality). SynForecast makes
    the bank configurable, expresses seasonal periods in time steps on a
    normalized grid, and adds bounded retries, divergence guards, and optional
    standardization.

    Base kernels (r = |x_i - x_j|, all on the normalized grid):
        - rbf: ``exp(-r^2 / (2 l^2))`` — smooth, length-scale ``l``
        - rational_quadratic: ``(1 + r^2 / (2 a))^(-a)`` — scale mixture of
          RBFs, shape ``a``
        - periodic (ExpSineSquared): ``exp(-2 sin^2(pi r / p_norm))`` with
          ``p_norm = period / length`` so ``period`` is expressed in time
          steps
        - linear (DotProduct): ``s^2 + x_i x_j`` — trend / drift
        - white: ``w`` on the diagonal — independent noise
        - constant: a constant offset

    Each composition is a polynomial ``K = sum_d (x x')^d T_d(x - x')`` with
    stationary PSD ``T_d`` (only the linear kernel is non-stationary), sampled
    exactly as ``sum_d x^d * f_d`` with independent O(n^2) Durbin-Levinson
    draws ``f_d ~ GP(0, T_d)``. Bulk ``generate()`` runs in the Rust batch
    path, whose RNG stream differs from NumPy's.

    Because a composed kernel can be near-degenerate or produce an exploding
    scale, non-finite, near-constant, or ``|y| >= 1e8`` draws are redrawn a
    bounded number of times before falling back to Gaussian noise.

    Args:
        max_kernels (int): Maximum number of base kernels composed per series;
            the count is drawn uniformly from ``1..max_kernels`` (default: 5).
        seasonal_periods (list[float]): Periodic-kernel periods, in time steps,
            forming the periodic entries of the bank (default: a broad set from
            4 up to 730 covering common hourly/daily/weekly/quarterly/yearly
            seasonalities).
        rbf_length_scales (list[float]): RBF length scales on the normalized
            grid (default: [0.1, 1.0, 10.0]).
        rational_quadratic_alphas (list[float]): Rational-quadratic shape
            parameters (default: [0.1, 1.0, 10.0]).
        linear_sigmas (list[float]): ``sigma_0`` offsets for the linear
            (DotProduct) kernel (default: [0.0, 1.0, 10.0]).
        white_noise_levels (list[float]): Diagonal noise levels for the white
            kernel (default: [0.1, 1.0]).
        include_constant (bool): Include a constant kernel in the bank
            (default: True).
        jitter (float): Diagonal jitter added to each term ``T_d``, escalated
            if the recursion still breaks down (default: 1e-6).
        standardize (bool): Standardize each sampled series to zero mean and
            unit variance. Kernel compositions span extreme scales, so
            standardization keeps the pool comparable for pretraining
            (default: True).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    max_kernels: int = Field(
        default=5, ge=1, description="Max base kernels composed per series"
    )
    seasonal_periods: list[float] = Field(
        default=[
            4.0,
            6.0,
            7.0,
            10.0,
            12.0,
            14.0,
            24.0,
            26.0,
            30.0,
            48.0,
            52.0,
            96.0,
            168.0,
            336.0,
            365.0,
            730.0,
        ],
        description="Periodic-kernel periods in time steps",
    )
    rbf_length_scales: list[float] = Field(
        default=[0.1, 1.0, 10.0], description="RBF length scales (normalized grid)"
    )
    rational_quadratic_alphas: list[float] = Field(
        default=[0.1, 1.0, 10.0], description="Rational-quadratic shape parameters"
    )
    linear_sigmas: list[float] = Field(
        default=[0.0, 1.0, 10.0], description="Linear (DotProduct) sigma_0 offsets"
    )
    white_noise_levels: list[float] = Field(
        default=[0.1, 1.0], description="White-kernel diagonal noise levels"
    )
    include_constant: bool = Field(
        default=True, description="Include a constant kernel in the bank"
    )
    jitter: float = Field(
        default=1e-6, ge=0, description="Diagonal jitter for numerical stability"
    )
    standardize: bool = Field(
        default=True, description="Standardize each series to zero mean, unit variance"
    )

    # Bank of (kind, param) specs; built once so rng.integers can index it.
    _bank: list[tuple[str, float]] = []

    @model_validator(mode="after")
    def build_kernel_bank(self) -> "KernelSynthGenerator":
        """Validate pools and assemble the kernel bank in a stable order."""
        if any(p <= 1 for p in self.seasonal_periods):
            raise ValueError("seasonal_periods must all be > 1")
        for name in ("rbf_length_scales", "rational_quadratic_alphas"):
            values = getattr(self, name)
            if any(v <= 0 for v in values):
                raise ValueError(f"{name} values must be positive")
        if any(w < 0 for w in self.white_noise_levels):
            raise ValueError("white_noise_levels must be non-negative")
        if any(s < 0 for s in self.linear_sigmas):
            raise ValueError("linear_sigmas must be non-negative")

        bank: list[tuple[str, float]] = []
        bank += [("rbf", float(v)) for v in self.rbf_length_scales]
        bank += [
            ("rational_quadratic", float(v)) for v in self.rational_quadratic_alphas
        ]
        bank += [("periodic", float(v)) for v in self.seasonal_periods]
        bank += [("linear", float(v)) for v in self.linear_sigmas]
        bank += [("white", float(v)) for v in self.white_noise_levels]
        if self.include_constant:
            bank.append(("constant", 1.0))
        if not bank:
            raise ValueError("kernel bank is empty; enable at least one kernel family")

        object.__setattr__(self, "_bank", bank)
        return self

    def _get_batch_params(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Encode the bank for ``GEN_KERNEL_SYNTH`` in ``rust/src/batch.rs``.

        scalars: [max_kernels, jitter, standardize]
        arrays:  [kernel kind ids (``_KIND_IDS``), kernel parameters]
        """
        kinds = np.array([_KIND_IDS[kind] for kind, _ in self._bank], dtype=float)
        params = np.array([param for _, param in self._bank], dtype=float)
        scalars = np.array(
            [float(self.max_kernels), self.jitter, float(self.standardize)]
        )
        return scalars, [kinds, params]

    def _lag_kernel(
        self, spec: tuple[str, float], length: int, lag: np.ndarray
    ) -> np.ndarray:
        """Stationary base kernel as a function of the lag."""
        kind, param = spec
        if kind == "rbf":
            return np.exp(-0.5 * (lag * lag) / param**2)
        if kind == "rational_quadratic":
            return (1.0 + (lag * lag) / (2.0 * param)) ** (-param)
        if kind == "periodic":
            p_norm = param / length
            return np.exp(-2.0 * np.sin(np.pi * lag / p_norm) ** 2)
        if kind == "white":
            values = np.zeros(length)
            values[0] = param
            return values
        return np.full(length, param)

    def _base_terms(
        self, spec: tuple[str, float], length: int, lag: np.ndarray
    ) -> list[np.ndarray | None]:
        """One bank entry as polynomial terms: ``s^2 + x x'`` for the linear kernel, a degree-0 term otherwise."""
        kind, param = spec
        if kind == "linear":
            return [np.full(length, param**2) if param > 0 else None, np.ones(length)]
        return [self._lag_kernel(spec, length, lag)]

    def _kernel_terms(
        self, indices: np.ndarray, ops: list[bool], length: int, lag: np.ndarray
    ) -> list[np.ndarray | None]:
        """Coefficients ``T_d`` of one composition; ``ops[j]`` is True to add ``bank[indices[j + 1]]``, False to multiply."""
        terms = self._base_terms(self._bank[indices[0]], length, lag)
        for index, add in zip(indices[1:], ops, strict=True):
            comp = self._base_terms(self._bank[index], length, lag)
            terms = _add_terms(terms, comp) if add else _multiply_terms(terms, comp)
        return terms

    @staticmethod
    def _levinson_sample(c: np.ndarray, z: np.ndarray) -> np.ndarray | None:
        """L @ z for the Cholesky factor L of Toeplitz(c), by Durbin-Levinson in O(n^2); None on breakdown."""
        n = len(c)
        x = np.empty(n)
        phi = np.empty(n)
        v = c[0]
        if not v > 0:
            return None
        x[0] = np.sqrt(v) * z[0]
        for t in range(1, n):
            kappa = (c[t] - phi[: t - 1] @ c[t - 1 : 0 : -1]) / v
            if not abs(kappa) < 1.0:
                return None
            phi[: t - 1] = phi[: t - 1] - kappa * phi[t - 2 :: -1] if t > 1 else phi[:0]
            phi[t - 1] = kappa
            v *= 1.0 - kappa * kappa
            if not v > 0:
                return None
            x[t] = phi[:t] @ x[t - 1 :: -1] + np.sqrt(v) * z[t]
        return x

    def _sample_stationary(self, c: np.ndarray) -> np.ndarray | None:
        """One draw from N(0, Toeplitz(c) + jitter I); the jitter escalates on a numerical breakdown."""
        n = len(c)
        scale = max(abs(float(c[0])), np.finfo(float).tiny)
        if np.all(np.abs(c - c[0]) <= 1e-12 * scale):
            # Constant (rank-one) kernel: one shared normal, which Levinson cannot factor.
            level = np.sqrt(max(float(c[0]), 0.0)) * self.rng.standard_normal()
            return level + np.sqrt(self.jitter) * self.rng.standard_normal(n)
        if not np.any(c[1:]):
            sd = np.sqrt(max(float(c[0]) + self.jitter, 0.0))
            return sd * self.rng.standard_normal(n)
        z = self.rng.standard_normal(n)
        extra = self.jitter
        for _ in range(_MAX_REGULARIZATIONS):
            shifted = c.copy()
            shifted[0] += extra
            values = self._levinson_sample(shifted, z)
            if values is not None and np.all(np.isfinite(values)):
                return values
            extra = max(10.0 * extra, 1e-10 * scale)
        return None

    def _compose_and_sample(self, length: int, x: np.ndarray) -> np.ndarray | None:
        """Draw a kernel composition and sample one path from its GP prior; None on breakdown."""
        n_kernels = int(self.rng.integers(1, self.max_kernels + 1))
        indices = self.rng.integers(0, len(self._bank), size=n_kernels)
        ops = [bool(self.rng.integers(0, 2) == 0) for _ in range(n_kernels - 1)]
        terms = self._kernel_terms(indices, ops, length, x - x[0])
        values = np.zeros(length)
        for degree, c in enumerate(terms):
            if c is None or not np.any(c):
                continue
            part = self._sample_stationary(c)
            if part is None:
                return None
            values += x**degree * part
        return values

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate one KernelSynth series.

        Args:
            length (int): The length of the series to generate

        Returns:
            Array of time series values
        """
        x = np.linspace(0.0, 1.0, length) if length > 1 else np.zeros(1)

        for _ in range(_MAX_RETRIES):
            values = self._compose_and_sample(length, x)
            if values is None or not np.all(np.isfinite(values)):
                continue
            if np.abs(values).max() >= _MAX_ABS:
                continue
            std = float(values.std()) if length > 1 else 1.0
            if length > 1 and std <= _MIN_STD:
                continue
            if self.standardize and length > 1:
                values = (values - values.mean()) / std
            return values

        # Retry budget exhausted (pathological bank): fall back to noise.
        return self.rng.normal(0.0, 1.0, length)
