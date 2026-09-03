"""MAR (mixture autoregressive) generator from the GRATIS recipe."""

from typing import Any

import numpy as np
from pydantic import Field, model_validator

from synforecast._features import compute_features
from synforecast.base import BaseGenerator

_MAX_ABS = 1e8
_MIN_STD = 1e-8
_MAX_RETRIES = 8
_SUPPORTED_FEATURES = {
    "spectral_entropy",
    "trend_strength",
    "seasonal_strength",
    "acf1",
}


class MARGenerator(BaseGenerator):
    """Generate GRATIS-style mixtures of autoregressive components.

    This adapts the MAR simulation used by GRATIS (Kang, Hyndman, and Li,
    2020, "GRATIS: GeneRAting TIme Series with diverse and controllable
    characteristics", https://arxiv.org/abs/1903.02787). At each step a
    component ``k`` is drawn from the mixture weights, then
    ``y_t = c_k + sum_i phi[k, i] y[t-i] + sigma_k eps_t``. The parameter
    sampling, partial-autocorrelation reparameterization via Durbin-Levinson,
    defaults, and stability guards are SynForecast's own design rather than a
    reproduction of the reference ``gratis`` R package.

    Random mode samples one to ``max_components`` components for every series.
    Fixed mode supplies all four of ``weights``, ``ar_coefficients``,
    ``intercepts``, and ``noise_scales`` and is used by feature targeting.
    Moment handling is explicit: ``standardize=True`` (the default) maps every
    accepted multi-observation draw to zero mean and unit standard deviation,
    rather than retaining the MAR model's component-implied level and scale.
    Set ``standardize=False`` to preserve those raw simulated moments. This
    output normalization is SynForecast's own design and is not source-series
    moment matching.
    """

    max_components: int = Field(
        default=3, ge=1, description="Maximum number of AR mixture components"
    )
    max_ar_order: int = Field(
        default=5, ge=1, description="Maximum non-seasonal AR order"
    )
    seasonal_period: int | None = Field(
        default=None, description="Optional multiplicative seasonal AR period"
    )
    weights_concentration: float = Field(
        default=1.0, gt=0, description="Symmetric Dirichlet concentration"
    )
    intercept_scale: float = Field(
        default=1.0, ge=0, description="Standard deviation of component intercepts"
    )
    noise_scale_range: tuple[float, float] = Field(
        default=(0.1, 2.0), description="Minimum and maximum innovation scales"
    )
    burn_in: int = Field(default=100, ge=0, description="Discarded simulation steps")
    standardize: bool = Field(
        default=True, description="Standardize each generated series"
    )
    weights: list[float] | None = Field(
        default=None, description="Fixed component probabilities"
    )
    ar_coefficients: list[list[float]] | None = Field(
        default=None, description="Fixed AR coefficients by component"
    )
    intercepts: list[float] | None = Field(
        default=None, description="Fixed component intercepts"
    )
    noise_scales: list[float] | None = Field(
        default=None, description="Fixed positive innovation scales"
    )

    @model_validator(mode="after")
    def validate_mar_parameters(self) -> "MARGenerator":
        """Validate seasonal, noise-range, and fixed-mode configuration."""
        if self.seasonal_period is not None and self.seasonal_period < 2:
            raise ValueError("seasonal_period must be >= 2 when provided")
        lo, hi = self.noise_scale_range
        if not 0 < lo <= hi:
            raise ValueError("noise_scale_range must satisfy 0 < low <= high")

        fixed = (
            self.weights,
            self.ar_coefficients,
            self.intercepts,
            self.noise_scales,
        )
        if all(value is None for value in fixed):
            return self
        if any(value is None for value in fixed):
            raise ValueError(
                "weights, ar_coefficients, intercepts, and noise_scales must be "
                "provided together"
            )

        weights = self.weights
        coefficients = self.ar_coefficients
        intercepts = self.intercepts
        scales = self.noise_scales
        assert weights is not None
        assert coefficients is not None
        assert intercepts is not None
        assert scales is not None
        counts = {len(weights), len(coefficients), len(intercepts), len(scales)}
        if len(counts) != 1 or not weights:
            raise ValueError("fixed parameter component counts must match and be >= 1")
        if any(not np.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("weights must be finite and positive")
        if any(not np.isfinite(scale) or scale <= 0 for scale in scales):
            raise ValueError("noise_scales must be finite and positive")
        if any(not np.isfinite(value) for value in intercepts):
            raise ValueError("intercepts must be finite")
        for component in coefficients:
            if not component or not np.all(np.isfinite(component)):
                raise ValueError(
                    "each AR coefficient list must be non-empty and finite"
                )
            if not self._is_stationary(np.asarray(component, dtype=float)):
                raise ValueError("every fixed AR component must be stationary")
        total = float(sum(weights))
        object.__setattr__(
            self, "weights", [float(weight / total) for weight in weights]
        )
        return self

    @staticmethod
    def _is_stationary(coefficients: np.ndarray) -> bool:
        """Check AR stationarity through companion-matrix spectral radius."""
        order = len(coefficients)
        companion = np.zeros((order, order))
        companion[0] = coefficients
        if order > 1:
            companion[1:, :-1] = np.eye(order - 1)
        return bool(np.max(np.abs(np.linalg.eigvals(companion))) < 1.0 - 1e-10)

    @staticmethod
    def _pacf_to_ar(pacf: np.ndarray) -> np.ndarray:
        """Map partial autocorrelations to stationary AR coefficients."""
        pacf = np.asarray(pacf, dtype=float)
        if np.any(np.abs(pacf) >= 1):
            raise ValueError(
                "partial autocorrelations must lie strictly inside (-1, 1)"
            )
        coefficients = np.empty(0, dtype=float)
        for reflection in pacf:
            previous = coefficients.copy()
            coefficients = np.empty(len(previous) + 1)
            if len(previous):
                coefficients[:-1] = previous - reflection * previous[::-1]
            coefficients[-1] = reflection
        return coefficients

    @staticmethod
    def _apply_seasonal_factor(
        coefficients: np.ndarray, seasonal_period: int, seasonal_phi: float
    ) -> np.ndarray:
        """Multiply an AR polynomial by one seasonal AR factor."""
        base = np.concatenate(([1.0], -coefficients))
        seasonal = np.zeros(seasonal_period + 1)
        seasonal[0] = 1.0
        seasonal[-1] = -seasonal_phi
        return -np.convolve(base, seasonal)[1:]

    def _sample_params(
        self,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Sample one stable random MAR configuration."""
        n_components = int(self.rng.integers(1, self.max_components + 1))
        weights = self.rng.dirichlet(np.full(n_components, self.weights_concentration))
        coefficients: list[np.ndarray] = []
        for _ in range(n_components):
            order = int(self.rng.integers(1, self.max_ar_order + 1))
            pacf = self.rng.uniform(-0.9, 0.9, order)
            component = self._pacf_to_ar(pacf)
            if self.seasonal_period is not None and self.rng.random() < 0.5:
                seasonal_phi = float(self.rng.uniform(-0.9, 0.9))
                component = self._apply_seasonal_factor(
                    component, self.seasonal_period, seasonal_phi
                )
            coefficients.append(component)
        intercepts = self.rng.normal(0.0, self.intercept_scale, n_components)
        low, high = np.log(self.noise_scale_range)
        scales = np.exp(self.rng.uniform(low, high, n_components))
        return weights, coefficients, intercepts, scales

    def _fixed_params(
        self,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """Return fixed fields as simulation arrays."""
        assert self.weights is not None
        assert self.ar_coefficients is not None
        assert self.intercepts is not None
        assert self.noise_scales is not None
        return (
            np.asarray(self.weights, dtype=float),
            [np.asarray(values, dtype=float) for values in self.ar_coefficients],
            np.asarray(self.intercepts, dtype=float),
            np.asarray(self.noise_scales, dtype=float),
        )

    def _simulate(
        self,
        length: int,
        params: tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Simulate a MAR path and discard burn-in observations."""
        weights, coefficients, intercepts, scales = params
        max_order = max(len(component) for component in coefficients)
        total = length + self.burn_in
        values = np.empty(total + max_order)
        values[:max_order] = self.rng.normal(0.0, 1.0, max_order)
        components = self.rng.choice(len(weights), size=total, p=weights)
        innovations = self._sample_innovations(total)
        for step, component_index in enumerate(components, start=max_order):
            component = coefficients[component_index]
            past = values[step - len(component) : step][::-1]
            values[step] = (
                intercepts[component_index]
                + float(component @ past)
                + scales[component_index] * innovations[step - max_order]
            )
        return values[max_order + self.burn_in :]

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate one finite MAR series of exactly ``length`` observations."""
        fixed_mode = self.weights is not None
        for _ in range(_MAX_RETRIES):
            params = self._fixed_params() if fixed_mode else self._sample_params()
            values = self._simulate(length, params)
            if not np.all(np.isfinite(values)):
                continue
            if np.abs(values).max() >= _MAX_ABS:
                continue
            std = float(values.std()) if length > 1 else 1.0
            if length > 1 and std <= _MIN_STD:
                continue
            if self.standardize and length > 1:
                values = (values - values.mean()) / std
            return values
        return self.rng.normal(0.0, 1.0, length)

    @classmethod
    def tune_to_features(
        cls,
        target_features: dict[str, float],
        min_length: int,
        max_length: int,
        freq: str | int,
        seasonal_period: int | None = None,
        n_generations: int = 15,
        population_size: int = 30,
        n_draws_per_candidate: int = 3,
        tolerance: float = 0.05,
        seed: int | None = None,
    ) -> "MARGenerator":
        """Tune fixed MAR parameters toward requested GRATIS-style features.

        Supported names are ``spectral_entropy``, ``trend_strength``,
        ``seasonal_strength``, and ``acf1``, as defined by
        :func:`synforecast._features.compute_features`. This implements
        GRATIS feature-targeted generation (Kang et al. 2020,
        https://arxiv.org/abs/1903.02787) as a seeded evolutionary search over
        MAR parameters. The elitism, PACF-space Gaussian mutation, and minimal
        feature set are SynForecast's own design rather than a reproduction of
        the GRATIS genetic algorithm. Candidate draw lengths are spaced evenly
        across ``min_length..max_length`` so the result targets the configured
        length range rather than one endpoint. The default budget evaluates
        roughly ``15 * 30 * 3`` series. Infeasible targets return the best
        candidate after the fixed budget.
        """
        cls._validate_targeting(
            target_features,
            min_length,
            max_length,
            seasonal_period,
            n_generations,
            population_size,
            n_draws_per_candidate,
            tolerance,
        )
        rng = np.random.default_rng(seed)
        evaluation_lengths = cls._evaluation_lengths(
            min_length, max_length, n_draws_per_candidate
        )
        population = [
            cls._random_candidate(rng, seasonal_period) for _ in range(population_size)
        ]
        best_candidate = population[0]
        best_distance = np.inf
        for _ in range(n_generations):
            ranked: list[tuple[float, dict[str, Any]]] = []
            for candidate in population:
                distance = cls._candidate_fitness(
                    candidate,
                    target_features,
                    evaluation_lengths,
                    freq,
                    seasonal_period,
                    rng,
                )
                ranked.append((distance, candidate))
            ranked.sort(key=lambda item: item[0])
            if ranked[0][0] < best_distance:
                best_distance, best_candidate = ranked[0]
            if best_distance <= tolerance:
                break
            n_elites = max(1, population_size // 4)
            elites = [candidate for _, candidate in ranked[:n_elites]]
            n_mutated = (population_size - n_elites + 1) // 2
            population = elites + [
                cls._mutate_candidate(elites[index % n_elites], rng)
                for index in range(n_mutated)
            ]
            while len(population) < population_size:
                population.append(cls._random_candidate(rng, seasonal_period))

        fixed = cls._candidate_to_fixed(best_candidate, seasonal_period)
        return cls(
            min_length=min_length,
            max_length=max_length,
            freq=freq,
            seed=seed,
            seasonal_period=seasonal_period,
            weights=fixed[0],
            ar_coefficients=fixed[1],
            intercepts=fixed[2],
            noise_scales=fixed[3],
        )

    @classmethod
    def _validate_targeting(
        cls,
        target_features: dict[str, float],
        min_length: int,
        max_length: int,
        seasonal_period: int | None,
        n_generations: int,
        population_size: int,
        n_draws_per_candidate: int,
        tolerance: float,
    ) -> None:
        """Validate feature targeting arguments."""
        if min_length < 3:
            raise ValueError("feature targeting requires min_length >= 3")
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length")
        if not target_features:
            raise ValueError("target_features must not be empty")
        unknown = set(target_features) - _SUPPORTED_FEATURES
        if unknown:
            supported = ", ".join(sorted(_SUPPORTED_FEATURES))
            raise ValueError(f"unsupported feature names; supported names: {supported}")
        for name, value in target_features.items():
            lower = -1.0 if name == "acf1" else 0.0
            if not lower <= value <= 1.0:
                raise ValueError(f"target {name} must be in [{lower:g}, 1]")
        if "seasonal_strength" in target_features and seasonal_period is None:
            raise ValueError("targeting seasonal_strength requires seasonal_period")
        if seasonal_period is not None and seasonal_period < 2:
            raise ValueError("seasonal_period must be >= 2 when provided")
        if (
            "seasonal_strength" in target_features
            and seasonal_period is not None
            and min_length < 2 * seasonal_period
        ):
            raise ValueError(
                "targeting seasonal_strength requires min_length >= 2 * seasonal_period"
            )
        if min(n_generations, population_size, n_draws_per_candidate) < 1:
            raise ValueError("search counts must all be >= 1")
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")

    @staticmethod
    def _evaluation_lengths(
        min_length: int, max_length: int, n_draws: int
    ) -> np.ndarray:
        """Return deterministic draw lengths spanning the configured range."""
        if n_draws == 1:
            return np.asarray([(min_length + max_length) // 2], dtype=int)
        return np.rint(np.linspace(min_length, max_length, n_draws)).astype(int)

    @classmethod
    def _random_candidate(
        cls, rng: np.random.Generator, seasonal_period: int | None
    ) -> dict[str, Any]:
        """Sample an unconstrained evolutionary candidate."""
        n_components = int(rng.integers(1, 4))
        weights = rng.dirichlet(np.ones(n_components))
        return {
            "weight_logits": np.log(weights),
            "pacf": [
                rng.uniform(-0.9, 0.9, int(rng.integers(1, 6)))
                for _ in range(n_components)
            ],
            "seasonal": [
                float(rng.uniform(-0.9, 0.9))
                if seasonal_period is not None and rng.random() < 0.5
                else None
                for _ in range(n_components)
            ],
            "intercepts": rng.normal(0.0, 1.0, n_components),
            "log_scales": rng.uniform(np.log(0.1), np.log(2.0), n_components),
        }

    @classmethod
    def _mutate_candidate(
        cls, candidate: dict[str, Any], rng: np.random.Generator
    ) -> dict[str, Any]:
        """Apply Gaussian mutation in logits, PACF, and scale space."""
        pacf = [
            np.clip(
                values + rng.normal(0.0, 0.15, len(values)),
                -0.9,
                0.9,
            )
            for values in candidate["pacf"]
        ]
        seasonal = [
            None
            if value is None
            else float(np.clip(value + rng.normal(0.0, 0.15), -0.9, 0.9))
            for value in candidate["seasonal"]
        ]
        return {
            "weight_logits": candidate["weight_logits"]
            + rng.normal(0.0, 0.2, len(candidate["weight_logits"])),
            "pacf": pacf,
            "seasonal": seasonal,
            "intercepts": candidate["intercepts"]
            + rng.normal(0.0, 0.2, len(candidate["intercepts"])),
            "log_scales": candidate["log_scales"]
            + rng.normal(0.0, 0.15, len(candidate["log_scales"])),
        }

    @classmethod
    def _candidate_to_fixed(
        cls, candidate: dict[str, Any], seasonal_period: int | None
    ) -> tuple[list[float], list[list[float]], list[float], list[float]]:
        """Convert one unconstrained candidate into fixed MAR fields."""
        logits = np.asarray(candidate["weight_logits"], dtype=float)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        coefficients: list[list[float]] = []
        for pacf, seasonal_phi in zip(
            candidate["pacf"], candidate["seasonal"], strict=True
        ):
            component = cls._pacf_to_ar(np.asarray(pacf, dtype=float))
            if seasonal_period is not None and seasonal_phi is not None:
                component = cls._apply_seasonal_factor(
                    component, seasonal_period, seasonal_phi
                )
            coefficients.append(component.tolist())
        scales = np.exp(np.clip(candidate["log_scales"], np.log(0.03), np.log(5.0)))
        return (
            weights.tolist(),
            coefficients,
            np.asarray(candidate["intercepts"], dtype=float).tolist(),
            scales.tolist(),
        )

    @classmethod
    def _candidate_fitness(
        cls,
        candidate: dict[str, Any],
        target_features: dict[str, float],
        eval_lengths: np.ndarray,
        freq: str | int,
        seasonal_period: int | None,
        rng: np.random.Generator,
    ) -> float:
        """Return realized L2 feature distance for one candidate."""
        fixed = cls._candidate_to_fixed(candidate, seasonal_period)
        generator = cls(
            min_length=int(eval_lengths.min()),
            max_length=int(eval_lengths.max()),
            freq=freq,
            seed=int(rng.integers(0, 2**63)),
            seasonal_period=seasonal_period,
            weights=fixed[0],
            ar_coefficients=fixed[1],
            intercepts=fixed[2],
            noise_scales=fixed[3],
        )
        realized = dict.fromkeys(target_features, 0.0)
        for eval_length in eval_lengths:
            features = compute_features(
                generator.generate_single_series(int(eval_length)), seasonal_period
            )
            for name in realized:
                realized[name] += features[name] / len(eval_lengths)
        differences = [
            realized[name] - value for name, value in target_features.items()
        ]
        return float(np.linalg.norm(differences))
