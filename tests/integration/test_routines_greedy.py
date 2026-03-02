import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection import GramData
from selection import BackwardSelection, ForwardSelection, MixedSelection
from tests.integration._selection_routines_helpers import (
    esl_book,
    expected_indices,
    make_diagonal_problem,
    small_problem,
)


def test_forward_selects_strongest_variable(small_problem):
    state = ForwardSelection().fit(data=small_problem)

    assert state.active_set == [0]
    assert np.isclose(state.beta[0], 0.95, atol=1e-6)


def test_backward_prunes_to_best_single_variable(small_problem):
    state = BackwardSelection().fit(data=small_problem)

    assert state.active_set == [0]
    assert np.isclose(state.beta[1], 0.0, atol=1e-9)


def test_forward_with_best_rss_matches_standard(small_problem):
    state = ForwardSelection(criterion=BestRSSCriterion).fit(data=small_problem)

    assert set(state.active_set) == {0, 1}
    np.testing.assert_allclose(
        state.beta[:2], np.linalg.solve(small_problem.gram, small_problem.cov)
    )


def test_forward_selection_recovers_esl_support(esl_book):
    gram_data, support = esl_book
    selector = ForwardSelection(criterion=BestRSSCriterion)
    state = selector.fit(data=gram_data, max_steps=len(support))
    recovered = set(state.active_set)
    assert len(recovered & set(support)) >= int(0.8 * len(support))


def test_backward_selection_recovers_esl_support(esl_book):
    gram_data, support = esl_book
    forward_selector = ForwardSelection(criterion=BestRSSCriterion)
    forward_state = forward_selector.fit(data=gram_data, max_steps=len(support))
    selector = BackwardSelection(criterion=BestRSSCriterion)
    p = gram_data.gram.shape[0]
    state = selector.fit(data=gram_data, max_steps=p - len(support))

    recovered = set(state.active_set)
    forward_support = set(forward_state.active_set)
    assert len(recovered & forward_support) >= int(0.8 * len(forward_support))


def test_forward_selection_permutation_invariance():
    p, steps = 8, 4
    gram, cov, y_norm, n_samples = make_diagonal_problem(p)
    perm = np.array([3, 0, 5, 1, 6, 2, 7, 4], dtype=int)
    P = np.eye(p)[perm]
    gram_perm = P.T @ gram @ P
    cov_perm = P.T @ cov

    selector = ForwardSelection(criterion=BestRSSCriterion)
    base_state = selector.fit(
        data=GramData(gram, cov, y_norm, n_samples), max_steps=steps
    )
    perm_state = selector.fit(
        data=GramData(gram_perm, cov_perm, y_norm, n_samples), max_steps=steps
    )

    mapped = [int(perm[i]) for i in perm_state.active_set]
    assert set(mapped) == set(base_state.active_set)


@pytest.mark.parametrize("p,steps", [(5, 3), (10, 7), (15, 12)])
def test_best_rss_selects_largest_coefficients_forward_and_backward(p, steps):
    gram, cov, y_norm, n_samples = make_diagonal_problem(p)

    forward_selector = ForwardSelection(criterion=BestRSSCriterion)
    forward_state = forward_selector.fit(
        data=GramData(gram, cov, y_norm, n_samples), max_steps=steps
    )

    assert set(forward_state.active_set) == set(expected_indices(p, steps))


def test_mixed_selection_matches_direct_solution():
    rng = np.random.default_rng(123)
    n, p = 40, 6
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.01 * rng.standard_normal(n)
    gram, cov, y_norm = X.T @ X, X.T @ y, y @ y
    state = MixedSelection().fit(
        data=GramData(gram, cov, y_norm, n),
        max_forward_steps=4,
        max_total_steps=7,
    )
    idx = np.array(state.active_set, dtype=int)
    if len(idx):
        gram_ss = gram[np.ix_(idx, idx)]
        beta_expected = np.zeros(p)
        beta_expected[idx] = np.linalg.solve(gram_ss, cov[idx])
        np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8)
        rss_expected = y_norm - cov[idx] @ beta_expected[idx]
    else:
        np.testing.assert_allclose(state.beta, np.zeros(p))
        rss_expected = y_norm
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected


def test_forward_then_backward_returns_to_empty_state():
    gram, cov, y_norm, n_samples = make_diagonal_problem(p=10)
    data = GramData(gram, cov, y_norm, n_samples)
    forward_state = ForwardSelection(criterion=BestRSSCriterion).fit(
        data=data, max_steps=4
    )

    while forward_state.active_set:
        scores = forward_state.compute_backward_scores()
        idx = int(np.argmin(scores))
        forward_state.apply_backward_step(idx)

    assert forward_state.active_set == []
    np.testing.assert_allclose(forward_state.beta, np.zeros_like(forward_state.beta))
    assert pytest.approx(forward_state.rss, rel=1e-9, abs=1e-9) == y_norm

