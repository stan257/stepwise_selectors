import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.routines import (
    BeamCrossValForwardSelection,
    BeamForwardSelection,
    CrossValForwardSelection,
)


def _make_tied_problem(p: int = 6) -> GramData:
    gram = np.eye(p)
    cov = np.ones(p)
    y_norm = float(cov @ cov) + 1.0
    return GramData(gram, cov, y_norm, n_samples=50)


def _make_proportional_cv_data(
    base: GramData, scales: list[int], *, n_samples_unit: int = 40
) -> CrossValGramData:
    folds = []
    for scale in scales:
        s = float(scale)
        folds.append(
            GramData(
                gram=s * base.gram,
                cov=s * base.cov,
                y_norm=s * base.y_norm,
                n_samples=scale * n_samples_unit,
            )
        )
    return CrossValGramData(folds)


def test_cv_forward_is_invariant_to_fold_order():
    rng = np.random.default_rng(2028)
    folds = []
    for _ in range(5):
        X = rng.standard_normal((120, 10))
        beta = np.zeros(10)
        beta[:4] = 1.0
        y = X @ beta + 0.03 * rng.standard_normal(120)
        folds.append(GramData(X.T @ X, X.T @ y, float(y @ y), n_samples=120))

    cv_data = CrossValGramData(folds)
    permuted = CrossValGramData([folds[i] for i in [3, 0, 4, 1, 2]])

    state = CrossValForwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    state_perm = CrossValForwardSelection(criterion=BestRSSCriterion).fit(
        data=permuted, max_steps=4
    )

    assert state.active_set == state_perm.active_set
    assert state.rss_cv == pytest.approx(state_perm.rss_cv, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(state.beta, state_perm.beta, atol=1e-10, rtol=1e-10)


def test_beam_selection_is_deterministic_under_exact_ties():
    data = _make_tied_problem(p=6)
    cv_data = CrossValGramData([data, data, data])

    single_runs = []
    cv_runs = []
    for _ in range(5):
        single_state = BeamForwardSelection(
            beam_width=4, criterion=BestRSSCriterion
        ).fit(data=data, max_steps=3)
        cv_state = BeamCrossValForwardSelection(
            beam_width=4, criterion=BestRSSCriterion
        ).fit(data=cv_data, max_steps=3)
        single_runs.append((single_state.active_set, single_state.rss))
        cv_runs.append((cv_state.active_set, cv_state.rss_cv))

    expected_support = [0, 1, 2]
    assert all(support == expected_support for support, _ in single_runs)
    assert all(support == expected_support for support, _ in cv_runs)
    assert all(score == single_runs[0][1] for _, score in single_runs)
    assert all(score == cv_runs[0][1] for _, score in cv_runs)


def test_cv_forward_consistent_across_equivalent_proportional_decompositions():
    rng = np.random.default_rng(2029)
    X = rng.standard_normal((200, 8))
    beta = np.array([2.0, -1.5, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ beta + 0.05 * rng.standard_normal(200)
    base = GramData(X.T @ X, X.T @ y, float(y @ y), n_samples=200)

    # Both decompositions produce the same aggregate Gram/cov/y_norm.
    cv_a = _make_proportional_cv_data(base, [1, 1, 1, 1])
    cv_b = _make_proportional_cv_data(base, [1, 2, 1])

    state_a = CrossValForwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_a, max_steps=4
    )
    state_b = CrossValForwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_b, max_steps=4
    )

    assert state_a.active_set == state_b.active_set
    np.testing.assert_allclose(state_a.beta, state_b.beta, atol=1e-10, rtol=1e-10)
    assert state_a.rss_cv == pytest.approx(state_b.rss_cv, rel=1e-12, abs=1e-12)
