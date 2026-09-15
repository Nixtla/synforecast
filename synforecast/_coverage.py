"""Numerical building blocks for feature-space coverage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

CONSTANT_ATOL = 1e-12


@dataclass
class EmbeddedFeatures:
    coordinates: list[np.ndarray]
    kept: np.ndarray
    constants: np.ndarray
    constant_values: np.ndarray
    explained_variance: tuple[float, float] | None
    parameters: dict[str, Any]


def fingerprint(value: Any) -> str:
    """Hash JSON-compatible measurement parameters in a canonical order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def embed_features(
    matrices: list[np.ndarray], embedding: str, fit: str, seed: int | None
) -> EmbeddedFeatures:
    """Fit on the specified population and preserve corpus boundaries."""
    population = matrices[0] if fit == "real" else np.concatenate(matrices)
    if len(population) < 2:
        raise ValueError(
            "The fitting population must contain at least two retained series"
        )
    # All inputs are finite; explicitly reject numerical overflow as well.
    with np.errstate(over="ignore", invalid="ignore"):
        mean = population.mean(axis=0)
        centered = population - mean
        constant = np.max(np.abs(centered), axis=0) <= CONSTANT_ATOL
        scale = np.sqrt(np.mean(centered**2, axis=0))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError(
            "Feature scaling overflowed; rescale the input feature columns"
        )
    kept = np.flatnonzero(~constant)
    if not len(kept):
        raise ValueError("No variable features remain in the fitting population")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = [(x[:, kept] - mean[kept]) / scale[kept] for x in matrices]
    if any(not np.all(np.isfinite(x)) for x in normalized):
        raise ValueError("Feature standardization produced non-finite values")
    parameters: dict[str, Any] = {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "kept_columns": kept.tolist(),
        "constant_atol": CONSTANT_ATOL,
        "numpy_version": np.__version__,
    }
    explained = None
    if embedding == "pca":
        training = normalized[0] if fit == "real" else np.concatenate(normalized)
        _, singular, vt = np.linalg.svd(training, full_matrices=False)
        tol = max(training.shape) * np.finfo(float).eps * singular[0]
        dimensions = min(2, int(np.count_nonzero(singular > tol)))
        components = np.zeros((2, len(kept)))
        for i in range(dimensions):
            loading = vt[i].copy()
            if loading[np.argmax(np.abs(loading))] < 0:
                loading *= -1
            components[i] = loading
        with np.errstate(over="ignore", invalid="ignore"):
            coordinates = [x @ components.T for x in normalized]
        explained_array = np.zeros(2)
        explained_array[:dimensions] = singular[:dimensions] ** 2 / np.sum(singular**2)
        explained = (float(explained_array[0]), float(explained_array[1]))
        parameters["components"] = components.tolist()
        parameters["rank_tolerance"] = float(tol)
    else:
        try:
            import sklearn
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError("t-SNE coverage requires scikit-learn") from exc
        training = np.concatenate(normalized)
        perplexity = 30.0
        if len(training) <= perplexity:
            raise ValueError(
                "t-SNE requires more than 30 retained series (perplexity=30)"
            )
        # Random initialization also permits a single retained feature column.
        coordinates_all = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=seed,
            n_jobs=1,
        ).fit_transform(training)
        coordinates = list(
            np.split(coordinates_all, np.cumsum([len(x) for x in matrices])[:-1])
        )
        parameters.update(
            sklearn_version=sklearn.__version__,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            seed=seed,
        )
    if any(not np.all(np.isfinite(x)) for x in coordinates):
        raise ValueError("Embedding produced non-finite coordinates")
    return EmbeddedFeatures(
        coordinates,
        kept,
        np.flatnonzero(constant),
        mean[constant],
        explained,
        parameters,
    )


def grid_edges(
    coordinates: list[np.ndarray], n_bins: int, grid_range: str
) -> tuple[np.ndarray, np.ndarray]:
    """Use a padded real range or the unpadded combined range."""
    population = coordinates[0] if grid_range == "real" else np.concatenate(coordinates)
    edges = []
    for axis in range(2):
        lower, upper = (
            float(population[:, axis].min()),
            float(population[:, axis].max()),
        )
        width = upper - lower
        padding = 0.01 * width if grid_range == "real" else 0.0
        if width == 0:
            padding = max(1.0, abs(lower)) * 0.01
        edge = np.linspace(lower - padding, upper + padding, n_bins + 1)
        if not np.all(np.isfinite(edge)) or not np.all(np.diff(edge) > 0):
            raise ValueError(
                "Grid edges are not finite and distinct; rescale feature inputs"
            )
        edges.append(edge)
    return edges[0], edges[1]


def grid_occupancy(
    coordinates: np.ndarray, edges: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return occupancy, cell indices, and the in-range mask (no clipping of outliers)."""
    bins = len(edges[0]) - 1
    inside = np.ones(len(coordinates), dtype=bool)
    cells = np.zeros((len(coordinates), 2), dtype=int)
    for axis, edge in enumerate(edges):
        values = coordinates[:, axis]
        inside &= (values >= edge[0]) & (values <= edge[-1])
        cells[:, axis] = np.minimum(
            np.searchsorted(edge, values, side="right") - 1, bins - 1
        )
    occupancy = np.zeros((bins, bins), dtype=bool)
    occupancy[cells[inside, 0], cells[inside, 1]] = True
    return occupancy, cells, inside


def occupancy_scores(
    real: np.ndarray, synthetic: np.ndarray
) -> tuple[float, float, float]:
    """Return the two all-cell measures and real-occupied-cell miscoverage."""
    uncovered = int(np.count_nonzero(real & ~synthetic))
    reverse = int(np.count_nonzero(synthetic & ~real))
    return uncovered / real.size, reverse / real.size, uncovered / int(real.sum())
