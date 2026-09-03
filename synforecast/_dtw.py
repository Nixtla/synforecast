"""Self-contained DTW and DBA utilities.

The barycenter follows Petitjean, Ketterlin, and Gancarski (2011,
https://doi.org/10.1016/j.patcog.2010.09.013).
"""

import numpy as np


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

    n, m = len(a), len(b)
    width = max(n, m) if band is None else max(band, abs(n - m))
    previous_costs = {0: 0.0}
    parents: dict[tuple[int, int], int] = {}
    for i in range(1, n + 1):
        current_costs: dict[int, float] = {}
        lower = max(1, i - width)
        upper = min(m, i + width)
        for j in range(lower, upper + 1):
            options = (
                previous_costs.get(j - 1, np.inf),
                previous_costs.get(j, np.inf),
                current_costs.get(j - 1, np.inf),
            )
            direction = int(np.argmin(options))
            current_costs[j] = (a[i - 1] - b[j - 1]) ** 2 + options[direction]
            parents[(i - 1, j - 1)] = direction
        previous_costs = current_costs

    i, j = n - 1, m - 1
    path: list[tuple[int, int]] = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        direction = parents.get((i, j), -1)
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise RuntimeError("DTW alignment is infeasible for the requested band")
    path.reverse()
    return float(np.sqrt(previous_costs.get(m, np.inf))), np.asarray(path, dtype=int)


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

    barycenter = series[0].copy()
    for _ in range(n_iterations):
        sums = np.zeros_like(barycenter)
        totals = np.zeros_like(barycenter)
        for values, weight in zip(series, weights, strict=True):
            _, path = dtw_alignment(barycenter, values, band)
            for barycenter_index, values_index in path:
                sums[barycenter_index] += weight * values[values_index]
                totals[barycenter_index] += weight
        observed = totals > 0
        barycenter[observed] = sums[observed] / totals[observed]
    return barycenter
