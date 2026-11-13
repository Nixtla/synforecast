"""Shared numeric guards and VAR helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Generated values must stay finite and below this magnitude
MAX_ABS = 1e8
# Standard deviations below this count as degenerate
MIN_STD = 1e-8
# Companion spectral radius random VAR coefficients are rescaled to
VAR_STABILITY = 0.95


def companion_spectral_radius(mats: Sequence[np.ndarray]) -> float:
    """Spectral radius of the VAR companion matrix of ``mats`` (one per lag).

    The VAR is stable iff the radius is below 1.
    """
    p = len(mats)
    k = mats[0].shape[0]
    companion = np.zeros((k * p, k * p))
    for lag, mat in enumerate(mats):
        companion[:k, lag * k : (lag + 1) * k] = mat
    if p > 1:
        companion[k:, :-k] = np.eye(k * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def stabilize(
    mats: list[np.ndarray], target: float = VAR_STABILITY
) -> list[np.ndarray]:
    """Rescale lag matrices so the companion spectral radius is at most ``target``.

    Scaling ``A_l`` by ``s**l`` maps every companion eigenvalue ``z`` to
    ``s * z``, so the rescale sets the radius exactly to the target.
    """
    radius = companion_spectral_radius(mats)
    if radius < target:
        return mats
    s = target / radius
    return [mat * s ** (lag + 1) for lag, mat in enumerate(mats)]


def validate_covariance(cov: object, n: int | None = None) -> np.ndarray:
    """Return ``cov`` as a finite, symmetric, positive-definite matrix.

    Raises:
        ValueError: If the matrix is not square (or not ``(n, n)``), not
            finite, not symmetric, or not positive definite.
    """
    arr = np.asarray(cov, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("innovation_covariance must be square")
    if n is not None and arr.shape != (n, n):
        raise ValueError(f"innovation_covariance must be ({n}, {n}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("innovation_covariance must be finite")
    if not np.allclose(arr, arr.T, atol=1e-8):
        raise ValueError("innovation_covariance must be symmetric")
    try:
        np.linalg.cholesky(arr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("innovation_covariance must be positive definite") from exc
    return arr
