"""Tests for the private DTW and DBA numeric utilities."""

import numpy as np
import pytest

from synforecast._dtw import (
    dba_barycenter,
    dtw_alignment,
    dtw_distance,
    pairwise_dtw_distances,
)


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

    def test_distance_kernel_matches_alignment_distance(self) -> None:
        rng = np.random.default_rng(3)
        first = rng.normal(size=40).cumsum()
        second = rng.normal(size=37).cumsum()
        for band in (None, 3, 10):
            assert dtw_distance(first, second, band) == pytest.approx(
                dtw_alignment(first, second, band)[0], rel=1e-12
            )

    def test_band_constrains_alignment(self) -> None:
        first = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        second = np.roll(first, 2)
        assert dtw_distance(first, second, None) == pytest.approx(0.0)
        assert dtw_distance(first, second, 2) == pytest.approx(0.0)
        assert dtw_distance(first, second, 1) > 1.0

    def test_pairwise_matrix_matches_per_pair_distances(self) -> None:
        rng = np.random.default_rng(4)
        series = [rng.normal(size=n).cumsum() for n in (30, 25, 30, 41)]
        matrix = pairwise_dtw_distances(series, 0.1)
        assert matrix.shape == (4, 4)
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), 0.0)
        for i in range(4):
            for j in range(i + 1, 4):
                band = max(
                    int(np.ceil(0.1 * max(len(series[i]), len(series[j])))),
                    abs(len(series[i]) - len(series[j])) + 1,
                )
                assert matrix[i, j] == pytest.approx(
                    dtw_distance(series[i], series[j], band), rel=1e-12
                )

    @pytest.mark.parametrize("window_fraction", [0.0, 1.5])
    def test_pairwise_rejects_bad_window_fraction(self, window_fraction: float) -> None:
        with pytest.raises(ValueError, match="window_fraction"):
            pairwise_dtw_distances([np.ones(3), np.ones(3)], window_fraction)

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
