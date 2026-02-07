import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.grouped_routines import GroupForwardSelection
from selection.routines import (
    BeamCrossValBackwardSelection,
    BeamForwardSelection,
    CrossValForwardSelection,
    ForwardSelection,
    MixedSelection,
)


def test_forward_best_rss_matches_direct_ols_on_selected():
    rng = np.random.default_rng(123)
    n, p, k = 40, 6, 3
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.05 * rng.standard_normal(n)

    gram = X.T @ X
    cov = X.T @ y
    y_norm = float(y @ y)

    data = GramData(gram, cov, y_norm, n)
    state = ForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=k
    )

    assert len(state.active_set) == k
    idx = np.array(state.active_set, dtype=int)
    beta_expected = np.linalg.solve(gram[np.ix_(idx, idx)], cov[idx])
    beta_full = np.zeros(p)
    beta_full[idx] = beta_expected
    rss_expected = float(y_norm - cov[idx] @ beta_expected)

    np.testing.assert_allclose(state.beta, beta_full, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected


def test_cv_forward_matches_full_data_on_replicated_folds():
    rng = np.random.default_rng(321)
    n, p, folds, k = 60, 5, 4, 3
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    gram = X.T @ X
    cov = X.T @ y
    y_norm = float(y @ y)

    fold = GramData(gram, cov, y_norm, n)
    cv_data = CrossValGramData([fold for _ in range(folds)])
    full_data = cv_data.make_full_data()

    forward_state = ForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=full_data, max_steps=k
    )
    cv_state = CrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=k
    )

    assert set(cv_state.active_set) == set(forward_state.active_set)
    assert pytest.approx(cv_state.rss_cv, rel=1e-8, abs=1e-8) == forward_state.rss


def test_beam_search_can_beat_greedy_in_two_steps():
    rho = 0.6
    c0 = 1.2
    gram = np.array([[1.0, rho, rho], [rho, 1.0, 0.0], [rho, 0.0, 1.0]])
    cov = np.array([c0, 1.0, 1.0])
    y_norm = float(cov @ np.linalg.solve(gram, cov) + 1.0)

    data = GramData(gram, cov, y_norm, n_samples=50)

    greedy_state = ForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=2
    )
    beam_state = BeamForwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=2)

    assert 0 in greedy_state.active_set
    assert set(beam_state.active_set) == {1, 2}
    assert beam_state.rss < greedy_state.rss - 1e-9


def test_grouped_forward_matches_forward_with_singleton_groups():
    rng = np.random.default_rng(7)
    n, p = 50, 6
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    data = GramData(X.T @ X, X.T @ y, y @ y, n)
    groups = [[i] for i in range(p)]

    forward_state = ForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=3
    )
    grouped_state = GroupForwardSelection(
        groups, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=3)

    assert set(grouped_state.active_groups) == set(forward_state.active_set)
    np.testing.assert_allclose(grouped_state.beta, forward_state.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(grouped_state.rss, rel=1e-8, abs=1e-8) == forward_state.rss


# ---------------------------------------------------------------------------
# Harder integration checks (added last)
# ---------------------------------------------------------------------------

def test_mixed_selection_matches_exhaustive_aic_on_diagonal():
    rng = np.random.default_rng(99)
    p = 7
    cov = rng.uniform(0.5, 2.0, size=p)
    gram = np.eye(p)
    y_norm = float(np.sum(cov**2) + 5.0)
    n_samples = 100

    data = GramData(gram, cov, y_norm, n_samples)
    mixed_state = MixedSelection().fit(data=data)

    best_aic = None
    best_sets = []
    for mask in range(1 << p):
        idx = [i for i in range(p) if mask & (1 << i)]
        if idx:
            rss = y_norm - float(np.sum(cov[idx] ** 2))
        else:
            rss = y_norm
        aic = n_samples * np.log(rss / n_samples) + 2 * len(idx)
        if best_aic is None or aic < best_aic - 1e-12:
            best_aic = aic
            best_sets = [set(idx)]
        elif abs(aic - best_aic) <= 1e-12:
            best_sets.append(set(idx))

    assert set(mixed_state.active_set) in best_sets


def test_cv_backward_handles_near_collinearity_across_folds():
    rng = np.random.default_rng(2024)
    folds = 4
    n, p = 80, 3
    eps = 1e-3
    fold_data = []

    for _ in range(folds):
        x1 = rng.standard_normal(n)
        x2 = x1 + eps * rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        X = np.column_stack([x1, x2, x3])
        beta_true = np.array([1.0, 1.0, 0.5])
        y = X @ beta_true + 0.01 * rng.standard_normal(n)
        fold_data.append(GramData(X.T @ X, X.T @ y, y @ y, n))

    cv_data = CrossValGramData(fold_data)

    # Sanity: each fold Gram is full rank, but the aggregate is ill-conditioned.
    for fold in fold_data:
        assert np.linalg.matrix_rank(fold.gram) == p
    assert np.linalg.cond(cv_data.gram_total) > 1e6

    selector = BeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    )
    cv_state = selector.fit(data=cv_data, max_steps=1)

    assert len(cv_state.active_set) == p - 1
    assert np.isfinite(cv_state.rss_cv)

    idx = np.array(cv_state.active_set, dtype=int)
    for train_state in cv_state.train_states:
        gram_ss = train_state.data.gram[np.ix_(idx, idx)]
        inv = np.linalg.inv(gram_ss)
        assert np.isfinite(inv).all()
