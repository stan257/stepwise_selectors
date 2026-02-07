import itertools

import numpy as np
import pytest

from selection.branch_and_bound import BranchAndBoundSelection
from selection.criteria import AICCriterion, BestRSSCriterion
from selection.definitions import GramData


def _explicit_beta_rss(data: GramData, subset):
    p = data.gram.shape[0]
    if not subset:
        return np.zeros(p), float(data.y_norm)
    idx = np.array(subset, dtype=int)
    G_ss = data.gram[np.ix_(idx, idx)]
    cov_s = data.cov[idx]
    beta_s = np.linalg.solve(G_ss, cov_s)
    beta = np.zeros(p)
    beta[idx] = beta_s
    rss = float(data.y_norm - cov_s @ beta_s)
    return beta, rss


def _exhaustive_best(data: GramData, k_max: int, criterion) -> tuple[tuple[int, ...], float]:
    best_score = float("inf") if criterion.minimize else float("-inf")
    best_subset: tuple[int, ...] = ()
    p = data.gram.shape[0]
    for k in range(k_max + 1):
        for subset in itertools.combinations(range(p), k):
            _, rss = _explicit_beta_rss(data, subset)
            score = float(np.asarray(criterion.evaluate(rss, k)))
            if criterion.is_improvement(score, incumbent=best_score):
                best_score = score
                best_subset = subset
    return best_subset, best_score


def _exhaustive_best_exact(data: GramData, k: int, criterion) -> tuple[tuple[int, ...], float]:
    best_score = float("inf") if criterion.minimize else float("-inf")
    best_subset: tuple[int, ...] = ()
    p = data.gram.shape[0]
    for subset in itertools.combinations(range(p), k):
        _, rss = _explicit_beta_rss(data, subset)
        score = float(np.asarray(criterion.evaluate(rss, k)))
        if criterion.is_improvement(score, incumbent=best_score):
            best_score = score
            best_subset = subset
    return best_subset, best_score


def test_branch_and_bound_matches_exhaustive_best_rss():
    rng = np.random.default_rng(1234)
    n, p = 40, 8
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    criterion = BestRSSCriterion()
    subset, score = _exhaustive_best(data, k_max=3, criterion=criterion)

    state = BranchAndBoundSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_subset_size=3
    )
    chosen = tuple(sorted(state.active_set))
    assert chosen == subset
    assert pytest.approx(state.search_score, rel=1e-8, abs=1e-8) == score


def test_branch_and_bound_exact_k_matches_exhaustive():
    rng = np.random.default_rng(4321)
    n, p, k = 35, 7, 3
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    criterion = BestRSSCriterion()
    subset, score = _exhaustive_best_exact(data, k=k, criterion=criterion)

    state = BranchAndBoundSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_subset_size=k, exact_k=True
    )
    chosen = tuple(sorted(state.active_set))
    assert chosen == subset
    assert pytest.approx(state.search_score, rel=1e-8, abs=1e-8) == score


def test_branch_and_bound_matches_exhaustive_aic():
    rng = np.random.default_rng(2027)
    n, p = 50, 9
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.2 * rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    criterion = AICCriterion(n_samples=n)
    subset, score = _exhaustive_best(data, k_max=4, criterion=criterion)

    state = BranchAndBoundSelection(criterion_cls=AICCriterion).fit(
        data=data, max_subset_size=4
    )
    chosen = tuple(sorted(state.active_set))
    assert chosen == subset
    assert pytest.approx(state.search_score, rel=1e-8, abs=1e-8) == score
