"""CauKer causal-DAG generator."""

import numpy as np
from pydantic import Field, model_validator

from synforecast import _families
from synforecast.generators._block import BlockGenerator


class CauKerGenerator(BlockGenerator):
    """Generate series from random nonlinear causal DAGs over kernel-composed roots (CauKer prior).

    Each series draws ``max_roots`` root paths from a KernelSynth-style
    composition (up to three periodic, GP, noise or trend components),
    adds a random linear, exponential or impulse mean function to half of
    them, and keeps 2 to ``max_roots`` as roots. Every later node is a
    random weighted sum of one to three earlier nodes plus a constant,
    passed through identity, ReLU, leaky ReLU, sigmoid, sine or modulo and
    standardised. One non-root node (or the last node) is returned, with
    light Gaussian noise on half the series. A constant output falls back
    to the first root. Port of the Simons TSFM ``cauker`` family.

    Args:
        min_length (int): Minimum length of each series.
        max_length (int): Maximum length of each series.
        freq (str | int): Pandas offset alias or an integer time index.
        max_nodes (int): Nodes in the DAG (default: 8).
        max_roots (int): Root nodes drawn per series, at least 2 (default: 4).
        seed (int | None): Random seed for reproducibility (default: None).
    """

    max_nodes: int = Field(default=8, ge=3, description="Nodes in the DAG")
    max_roots: int = Field(default=4, ge=2, description="Root nodes drawn per series")

    @model_validator(mode="after")
    def validate_roots(self) -> "CauKerGenerator":
        if self.max_roots >= self.max_nodes:
            raise ValueError("max_roots must be below max_nodes")
        return self

    def _kernel(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        return _families.cauker(rng, n, length, self.max_nodes, self.max_roots)
