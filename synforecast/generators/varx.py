"""VARX generator: a VAR panel driven by exogenous regressors with known effects."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from synforecast.exogenous import SeriesMetadata
from synforecast.generators.var import _BURN_IN, _STABILITY_TARGET, VARGenerator


@dataclass(frozen=True)
class VARXTruth:
    """The process behind one ``VARXGenerator`` draw.

    Shapes use ``N`` series, ``p`` lags and ``S``/``F``/``H`` shared, future-known
    and history-only regressors::

        y[t] = intercept + sum_l lag_matrices[l] @ y[t - 1 - l]
               + shared_effects @ shared[t]
               + sum_k future_effects[k] @ future[t, :, k]
               + sum_k hist_effects[k] @ hist[t - 1, :, k] + e[t]

    Every regressor is a unit-variance AR(1) with coefficient
    ``exog_persistence``. ``groups`` holds the group of every series (None
    without ``n_groups``). The truth describes the process before missing
    data, anomalies and changepoints are injected.
    """

    lag_matrices: np.ndarray  # (p, N, N)
    intercept: np.ndarray  # (N,)
    shared_effects: np.ndarray  # (N, S)
    future_effects: np.ndarray  # (F, N, N)
    hist_effects: np.ndarray  # (H, N, N)
    innovation_covariance: np.ndarray  # (N, N)
    exog_persistence: float
    groups: np.ndarray | None  # (N,)

    def forecast(
        self,
        y: np.ndarray,
        shared: np.ndarray,
        future: np.ndarray,
        hist: np.ndarray,
        h: int,
    ) -> np.ndarray:
        """Conditional-mean forecast ``(h, N)`` after the last row of ``y``.

        The optimal squared-error forecast of the process, an oracle for
        benchmarking estimators. ``y`` is the history ``(T, N)`` with
        ``T >= p``; ``shared`` ``(h, S)`` and ``future`` ``(h, N, F)`` hold the
        future-known regressors of the forecast periods; ``hist`` ``(N, H)``
        holds the history-only regressors at the last period, whose later values
        are replaced by their AR(1) expectation.
        """
        p = self.lag_matrices.shape[0]
        if y.shape[0] < p:
            raise ValueError(f"y needs at least {p} rows, got {y.shape[0]}")
        state = list(y[-p:])
        hist_now = np.asarray(hist, dtype=np.float64)
        out = np.empty((h, y.shape[1]))
        for s in range(h):
            value = self.intercept + self.shared_effects @ shared[s]
            for lag in range(p):
                value = value + self.lag_matrices[lag] @ state[-1 - lag]
            for k in range(self.future_effects.shape[0]):
                value = value + self.future_effects[k] @ future[s, :, k]
            for k in range(self.hist_effects.shape[0]):
                value = value + self.hist_effects[k] @ hist_now[:, k]
            hist_now = hist_now * self.exog_persistence
            out[s] = value
            state.append(value)
        return out


class VARXGenerator(VARGenerator):
    """Generate a VAR panel driven by exogenous regressors with known effects.

    Extends :class:`VARGenerator` with three kinds of regressors, each a
    unit-variance AR(1) with coefficient ``exog_persistence``:

    - shared (``exog_shared_k``): one value per period for every series, such
      as a promotion calendar; series ``i`` responds with effect
      ``shared_effects[i, k]``;
    - future-known (``exog_future_k``): one value per series and period, known
      for the forecast periods, such as a plan or price; it moves its own
      series and, with ``exog_cross_scale > 0``, the other series;
    - history-only (``exog_hist_k``): one value per series and period, known
      up to the forecast origin, such as an order book; it moves the series
      one period later.

    Lag matrices are random with a configurable structure unless
    ``coef_matrices`` is given: ``"dense"`` (as ``VARGenerator``),
    ``"sparse"`` (each off-diagonal entry nonzero with probability
    ``sparsity``) or ``"groups"`` (every series responds to its own lag and to
    the lagged total of each of ``n_groups`` groups). All are rescaled to a
    stable process. With ``n_groups``, series ``i`` belongs to group
    ``i * n_groups // n_series`` and ``group_col`` holds it.

    After :meth:`generate` (or :meth:`generate_values`), :attr:`truth`
    returns the coefficients of that draw and its oracle forecast.

    Args:
        n_shared_exog (int): Shared regressors (default: 0).
        n_future_exog (int): Future-known regressors per series (default: 1).
        n_hist_exog (int): History-only regressors per series (default: 0).
        exog_persistence (float): AR(1) coefficient of every regressor in
            [0, 1) (default: 0.5).
        exog_scale (float): Standard deviation of the shared effects and of a
            regressor's effect on its own series (default: 1.0).
        exog_cross_scale (float): Standard deviation of the total effect of
            the other series' regressors on one series; each entry has
            standard deviation ``exog_cross_scale / sqrt(n_series - 1)``
            (default: 0.0, own effects only).
        cross_structure (str): Random lag structure: 'dense', 'sparse' or
            'groups' (default: 'dense').
        sparsity (float): Probability that an off-diagonal lag coefficient is
            nonzero under 'sparse' (default: 0.1).
        n_groups (int | None): Number of contiguous series groups; required
            for 'groups' (default: None).
        group_col (str): Column holding the group of each series
            (default: 'group').
    """

    n_shared_exog: int = Field(default=0, ge=0, description="Shared regressors")
    n_future_exog: int = Field(default=1, ge=0, description="Future-known regressors")
    n_hist_exog: int = Field(default=0, ge=0, description="History-only regressors")
    exog_persistence: float = Field(
        default=0.5, ge=0, lt=1, description="AR(1) coefficient of every regressor"
    )
    exog_scale: float = Field(
        default=1.0, ge=0, description="Std of shared and own regressor effects"
    )
    exog_cross_scale: float = Field(
        default=0.0, ge=0, description="Std of the total cross-series regressor effect"
    )
    cross_structure: Literal["dense", "sparse", "groups"] = Field(
        default="dense", description="Random lag structure"
    )
    sparsity: float = Field(
        default=0.1, gt=0, le=1, description="Nonzero share of off-diagonal lags"
    )
    n_groups: int | None = Field(default=None, ge=1, description="Series groups")
    group_col: str = Field(default="group", description="Group column name")

    _truth: VARXTruth | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_varx_params(self) -> "VARXGenerator":
        """Check that the group structure has its groups."""
        if self.cross_structure == "groups" and self.n_groups is None:
            raise ValueError("cross_structure='groups' requires n_groups")
        return self

    @property
    def truth(self) -> VARXTruth:
        """The process of the last draw."""
        if self._truth is None:
            raise ValueError("generate first: the truth is drawn with the data")
        return self._truth

    def _groups(self, n_series: int) -> np.ndarray | None:
        if self.n_groups is None:
            return None
        if self.n_groups > n_series:
            raise ValueError(f"n_groups={self.n_groups} exceeds n_series={n_series}")
        return np.arange(n_series) * self.n_groups // n_series

    def _generate_stable_coefficients(self, n_variables: int) -> list[np.ndarray]:
        """Random lag matrices of the configured structure, rescaled to stability."""
        if self.cross_structure == "dense":
            return super()._generate_stable_coefficients(n_variables)
        n_series, groups = n_variables, self._groups(n_variables)
        mats = []
        for lag in range(self.lag_order):
            own = np.diag(self.rng.uniform(0.2, 0.6, n_series))
            if self.cross_structure == "sparse":
                mask = self.rng.random((n_series, n_series)) < self.sparsity
                np.fill_diagonal(mask, False)
                cross = mask * self.rng.uniform(-0.5, 0.5, (n_series, n_series))
            else:
                assert groups is not None
                sizes = np.bincount(groups)
                totals = self.rng.uniform(-0.5, 0.5, (n_series, len(sizes)))
                cross = (totals / sizes)[:, groups]
            mats.append((own + cross) * 0.7**lag)
        radius = self._companion_spectral_radius(mats)
        if radius >= _STABILITY_TARGET:
            s = _STABILITY_TARGET / radius
            mats = [mat * s ** (lag + 1) for lag, mat in enumerate(mats)]
        return mats

    def _effects(self, n_regressors: int, n_series: int) -> np.ndarray:
        """``(n_regressors, N, N)`` per-series regressor effects: own effects on
        the diagonal, cross effects off it."""
        out = np.zeros((n_regressors, n_series, n_series))
        for k in range(n_regressors):
            if self.exog_cross_scale > 0 and n_series > 1:
                scale = self.exog_cross_scale / np.sqrt(n_series - 1)
                out[k] = self.rng.normal(0.0, scale, (n_series, n_series))
            np.fill_diagonal(out[k], self.rng.normal(0.0, self.exog_scale, n_series))
        return out

    def _ar1(self, shape: tuple[int, ...]) -> np.ndarray:
        """Unit-variance AR(1) along axis 0, started from its stationary law."""
        phi = self.exog_persistence
        shocks = self.rng.standard_normal(shape)
        out = np.empty(shape)
        out[0] = shocks[0]
        root = np.sqrt(1.0 - phi**2)
        for t in range(1, shape[0]):
            out[t] = phi * out[t - 1] + root * shocks[t]
        return out

    def _simulate(self, length: int, n_series: int) -> tuple[np.ndarray, ...]:
        """Draw the truth and simulate ``length`` periods after the burn-in.

        Returns:
            (y (T, N), shared (T, S), future (T, N, F), hist (T, N, H)).
        """
        groups = self._groups(n_series)
        lag_matrices, intercept, covariance = self._resolve_parameters(n_series)
        shared_effects = self.rng.normal(
            0.0, self.exog_scale, (n_series, self.n_shared_exog)
        )
        future_effects = self._effects(self.n_future_exog, n_series)
        hist_effects = self._effects(self.n_hist_exog, n_series)

        total = length + _BURN_IN
        shared = self._ar1((total, self.n_shared_exog))
        future = self._ar1((total, n_series, self.n_future_exog))
        hist = self._ar1((total, n_series, self.n_hist_exog))
        innovations = (
            self._sample_innovations((total, n_series))
            @ np.linalg.cholesky(covariance).T
        )

        drive = intercept + shared @ shared_effects.T + innovations
        drive += np.einsum("kij,tjk->ti", future_effects, future)
        drive[1:] += np.einsum("kij,tjk->ti", hist_effects, hist[:-1])
        p = self.lag_order
        y = np.zeros((total, n_series))
        y[:p] = self.rng.normal(0, 0.1, (p, n_series))
        for t in range(p, total):
            value = drive[t].copy()
            for lag in range(p):
                value += lag_matrices[lag] @ y[t - lag - 1]
            y[t] = value

        self._truth = VARXTruth(
            lag_matrices=np.stack(lag_matrices),
            intercept=np.asarray(intercept, dtype=np.float64),
            shared_effects=shared_effects,
            future_effects=future_effects,
            hist_effects=hist_effects,
            innovation_covariance=np.asarray(covariance, dtype=np.float64),
            exog_persistence=self.exog_persistence,
            groups=groups,
        )
        keep = slice(_BURN_IN, None)
        return y[keep], shared[keep], future[keep], hist[keep]

    def generate_single_series(self, length: int) -> np.ndarray:
        """Generate the target of a one-series VARX.

        Args:
            length (int): The length of the series to generate.

        Returns:
            Array of time series values.
        """
        return self._simulate(length, 1)[0][:, 0]

    def _generate_metadata(
        self,
        n_series: int,
        start_id: int,
        n_jobs: int,  # noqa: ARG002
    ) -> list[SeriesMetadata]:
        """Draw n_series series and their regressors, sharing one length."""
        length = int(self.rng.integers(self.min_length, self.max_length + 1))
        y, shared, future, hist = self._simulate(length, n_series)
        groups = self.truth.groups

        pi_seeds = self._draw_pi_seeds(n_series)
        out: list[SeriesMetadata] = []
        for i in range(n_series):
            values, cp, anomalies, missing = self._apply_pattern_injection(
                np.ascontiguousarray(y[:, i]), pi_seeds[i]
            )
            extra = {f"exog_shared_{k}": shared[:, k] for k in range(shared.shape[1])}
            extra |= {
                f"exog_future_{k}": future[:, i, k] for k in range(future.shape[2])
            }
            extra |= {f"exog_hist_{k}": hist[:, i, k] for k in range(hist.shape[2])}
            if groups is not None:
                extra[self.group_col] = np.full(length, groups[i], dtype=np.int64)
            out.append(
                SeriesMetadata(
                    values=values,
                    timestamps=None,
                    series_id=start_id + i,
                    length=length,
                    anomaly_indices=anomalies,
                    changepoint_indices=cp,
                    missing_indices=missing,
                    extra_columns=extra,
                )
            )
        return out
