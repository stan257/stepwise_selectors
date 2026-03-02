import numpy as np
import pytest

from selection.selectors.routines_cv_scoring import (
    _aggregate_cv_rss_matrix,
    _build_fold_states,
    _cv_backward_scores,
    _cv_forward_scores,
    _cv_rss,
    _normalize_cv_aggregation,
)
from tests.helpers import explicit_cv_rss, make_cv_regression_gram


@pytest.mark.parametrize("seed", [601, 602])
def test_cv_rss_matches_explicit_for_active_set(seed: int):
    cv_data = make_cv_regression_gram(seed, folds=4, n=90, p=12)
    active_set = [0, 3, 5]

    fold_states = _build_fold_states(cv_data, tol=1e-10, active_set=active_set)
    got = _cv_rss(fold_states, cv_data)
    expected = explicit_cv_rss(cv_data, active_set)

    assert got == pytest.approx(expected, rel=1e-8, abs=1e-8)


def test_cv_forward_scores_match_explicit_candidate_rss():
    cv_data = make_cv_regression_gram(603, folds=5, n=100, p=14)
    active_set = [1, 4, 7]

    fold_states = _build_fold_states(cv_data, tol=1e-10, active_set=active_set)
    scored = _cv_forward_scores(fold_states, cv_data, tol=1e-10)
    assert scored is not None
    candidates, aggregated = scored

    expected = np.array(
        [explicit_cv_rss(cv_data, active_set + [int(c)]) for c in candidates],
        dtype=float,
    )
    np.testing.assert_allclose(aggregated, expected, atol=1e-8, rtol=1e-8)


def test_cv_backward_scores_match_explicit_candidate_rss():
    cv_data = make_cv_regression_gram(604, folds=4, n=120, p=12)
    active_set = [0, 1, 2, 3, 4]

    fold_states = _build_fold_states(cv_data, tol=1e-10, active_set=active_set)
    aggregated = _cv_backward_scores(fold_states, cv_data, tol=1e-10)
    assert aggregated is not None

    expected = np.array(
        [
            explicit_cv_rss(
                cv_data,
                [feat for pos, feat in enumerate(active_set) if pos != drop_pos],
            )
            for drop_pos in range(len(active_set))
        ],
        dtype=float,
    )
    np.testing.assert_allclose(aggregated, expected, atol=1e-8, rtol=1e-8)


def test_cv_aggregation_matrix_modes_can_change_candidate_ranking():
    rss_matrix = np.array(
        [
            [900.0, 990.0],
            [200.0, 150.0],
        ],
        dtype=float,
    )
    fold_sizes = np.array([900, 100], dtype=int)

    sum_scores = _aggregate_cv_rss_matrix(rss_matrix, fold_sizes, "sum_rss")
    mean_scores = _aggregate_cv_rss_matrix(rss_matrix, fold_sizes, "mean_mse")
    median_scores = _aggregate_cv_rss_matrix(rss_matrix, fold_sizes, "median_mse")

    assert int(np.argmin(sum_scores)) == 0
    assert int(np.argmin(mean_scores)) == 1
    assert int(np.argmin(median_scores)) == 1


def test_normalize_cv_aggregation_rejects_unknown_mode():
    with pytest.raises(
        ValueError,
        match=r"cv_aggregation must be one of: mean_mse, median_mse, sum_rss",
    ):
        _normalize_cv_aggregation("fold_weighted")
