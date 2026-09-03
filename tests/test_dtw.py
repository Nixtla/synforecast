"""Tests for the private DTW and DBA numeric utilities."""

import numpy as np
import pytest

from synforecast._dtw import dba_barycenter, dtw_alignment, dtw_distance


class TestDtwAlignment:
    def test_identical_series_use_zero_cost_diagonal(self) -> None:
        values = np.array([0.0, 1.0, 2.0])

        distance, path = dtw_alignment(values, values, band=0)

        assert distance == 0.0
        np.testing.assert_array_equal(path, [[0, 0], [1, 1], [2, 2]])

    def test_known_warped_distance_and_path(self) -> None:
        distance, path = dtw_alignment(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0]), band=None
        )

        assert distance == pytest.approx(1.0)
        assert tuple(path[0]) == (0, 0)
        assert tuple(path[-1]) == (2, 1)

    def test_distance_is_symmetric_for_unequal_lengths(self) -> None:
        first = np.array([0.0, 1.0, 1.5, 2.0])
        second = np.array([0.0, 0.5, 2.0])

        assert dtw_distance(first, second, 1) == pytest.approx(
            dtw_distance(second, first, 1)
        )

    @pytest.mark.parametrize(
        ("first", "second", "band", "message"),
        [
            (np.array([]), np.ones(2), None, "non-empty"),
            (np.ones((2, 2)), np.ones(2), None, "one-dimensional"),
            (np.array([0.0, np.nan]), np.ones(2), None, "finite"),
            (np.ones(2), np.ones(2), -1, "non-negative"),
        ],
    )
    def test_invalid_inputs_rejected(
        self,
        first: np.ndarray,
        second: np.ndarray,
        band: int | None,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            dtw_alignment(first, second, band)


class TestDbaBarycenter:
    def test_weighted_diagonal_average(self) -> None:
        result = dba_barycenter(
            reference=np.array([0.0, 2.0]),
            neighbors=[np.array([2.0, 4.0])],
            weights=np.array([0.25, 0.75]),
            n_iterations=1,
            band=0,
        )

        np.testing.assert_allclose(result, [1.5, 3.5])

    @pytest.mark.parametrize(
        "weights", [np.array([1.0]), np.array([1.0, -0.5]), np.array([0.0, 0.0])]
    )
    def test_invalid_weights_rejected(self, weights: np.ndarray) -> None:
        with pytest.raises(ValueError, match="weights"):
            dba_barycenter(np.ones(3), [np.ones(3)], weights, n_iterations=1, band=None)

    def test_invalid_iteration_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_iterations"):
            dba_barycenter(
                np.ones(3),
                [np.ones(3)],
                np.ones(2),
                n_iterations=0,
                band=None,
            )
