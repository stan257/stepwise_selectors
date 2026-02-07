import numpy as np

from selection.criteria import BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.routines import (
    BeamForwardSelection,
    CrossValForwardSelection,
    ForwardSelection,
    MixedSelection,
)


def test_golden_forward_best_rss_active_set():
    rng = np.random.default_rng(111)
    n, p, k = 50, 7, 3
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    data = GramData(X.T @ X, X.T @ y, y @ y, n)
    state = ForwardSelection(criterion_cls=BestRSSCriterion).fit(data=data, max_steps=k)

    assert state.active_set == [2, 5, 6]


def test_golden_mixed_aic_active_set():
    rng = np.random.default_rng(222)
    n, p = 40, 6
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.2 * rng.standard_normal(n)

    data = GramData(X.T @ X, X.T @ y, y @ y, n)
    state = MixedSelection().fit(data=data, max_forward_steps=4, max_total_steps=6)

    assert state.active_set == [2, 1, 0, 3]


def test_golden_cv_forward_best_rss_active_set():
    rng = np.random.default_rng(333)
    folds = 3
    n = 30
    p = 5
    beta_true = rng.standard_normal(p)
    fold_data = []
    for _ in range(folds):
        X = rng.standard_normal((n, p))
        y = X @ beta_true + 0.1 * rng.standard_normal(n)
        fold_data.append(GramData(X.T @ X, X.T @ y, y @ y, n))

    cv_data = CrossValGramData(fold_data)
    cv_state = CrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=2
    )

    assert cv_state.active_set == [1, 3]


def test_golden_beam_forward_best_rss_active_set():
    rng = np.random.default_rng(444)
    n, p, k = 60, 8, 3
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    data = GramData(X.T @ X, X.T @ y, y @ y, n)
    state = BeamForwardSelection(beam_width=3, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=k
    )

    assert state.active_set == [6, 4, 5]
