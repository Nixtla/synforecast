"""DTW and DBA utilities backed by native Rust kernels.

The barycenter follows Petitjean, Ketterlin, and Gancarski (2011,
https://doi.org/10.1016/j.patcog.2010.09.013).
"""

import numpy as np

from synforecast._lib import augmentation as _rs_augmentation


def dtw_alignment(
    a: np.ndarray, b: np.ndarray, band: int | None
) -> tuple[float, np.ndarray]:
    """Return square-root DTW distance and an optimal alignment path."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) == 0 or len(b) == 0:
        raise ValueError("DTW inputs must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("DTW inputs must be finite")
    if band is not None and band < 0:
        raise ValueError("band must be non-negative when provided")

    distance, path = _rs_augmentation.dtw_alignment(
        np.ascontiguousarray(a), np.ascontiguousarray(b), band
    )
    return float(distance), np.asarray(path, dtype=int)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int | None) -> float:
    """Return square-root accumulated squared-Euclidean DTW distance."""
    distance, _ = dtw_alignment(a, b, band)
    return distance


def dba_barycenter(
    reference: np.ndarray,
    neighbors: list[np.ndarray],
    weights: np.ndarray,
    n_iterations: int,
    band: int | None,
) -> np.ndarray:
    """Compute a weighted DBA barycenter restricted to reference length."""
    series = [np.asarray(reference, dtype=float), *neighbors]
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(series) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and match all input series")
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")

    return np.asarray(
        _rs_augmentation.dba_barycenter(
            np.ascontiguousarray(series[0]),
            [np.ascontiguousarray(values) for values in series[1:]],
            np.ascontiguousarray(weights),
            n_iterations,
            band,
        )
    )
