import numpy as np
import pytest

from selection.definitions import CrossValGramData, GramData
from selection.fast_routines import (
    FastForwardState,
    _fast_cv_forward_scores,
    _fast_cv_rss,
)


def _make_explicit_cv(seed=123, folds=3, n=60, p=10):
    rng = np.random.default_rng(seed)
    beta = rng.standard_normal(p)
    fold_data = []
    X_folds = []
    y_folds = []
    for fold_seed in rng.integers(0, 1_000_000, size=folds):
        r = np.random.default_rng(int(fold_seed))
        X = r.standard_normal((n, p))
        y = X @ beta + 0.1 * r.standard_normal(n)
        X_folds.append(X)
        y_folds.append(y)
        fold_data.append(GramData(X.T @ X, X.T @ y, y @ y, n))
    return CrossValGramData(fold_data), X_folds, y_folds


def _explicit_cv_rss(X_folds, y_folds, active):
    total = 0.0
    for k in range(len(X_folds)):
        X_val = X_folds[k]
        y_val = y_folds[k]
        # Build train data by concatenating all other folds.
        X_train = np.vstack([X_folds[i] for i in range(len(X_folds)) if i != k])
        y_train = np.concatenate([y_folds[i] for i in range(len(X_folds)) if i != k])

        if not active:
            residuals = y_val
        else:
            idx = np.array(active, dtype=int)
            XtX = X_train[:, idx].T @ X_train[:, idx]
            XtY = X_train[:, idx].T @ y_train
            beta = np.linalg.solve(XtX, XtY)
            residuals = y_val - X_val[:, idx] @ beta
        total += float(residuals @ residuals)
    return total


def test_fast_cv_rss_matches_explicit_validation():
    cv_data, X_folds, y_folds = _make_explicit_cv(seed=321, folds=3, n=50, p=10)
    active = [0, 2, 4]

    fast_states = [
        FastForwardState.from_active_set(
            cv_data.train_data_for_fold(k), active, tol=1e-12
        )
        for k in range(cv_data.n_folds)
    ]
    fast_rss = _fast_cv_rss(fast_states, cv_data)
    explicit = _explicit_cv_rss(X_folds, y_folds, active)

    assert pytest.approx(fast_rss, rel=1e-8, abs=1e-8) == explicit


def test_fast_cv_forward_scores_match_explicit_single_feature():
    cv_data, X_folds, y_folds = _make_explicit_cv(seed=444, folds=3, n=60, p=8)
    fast_states = [
        FastForwardState.create(cv_data.train_data_for_fold(k), tol=1e-12)
        for k in range(cv_data.n_folds)
    ]
    scored = _fast_cv_forward_scores(fast_states, cv_data, tol=1e-12)
    assert scored is not None
    candidates, aggregated = scored

    # Validate a few candidate scores against explicit fold-by-fold evaluation.
    for candidate in candidates[:3]:
        explicit = _explicit_cv_rss(X_folds, y_folds, [int(candidate)])
        idx = candidates.index(candidate)
        assert pytest.approx(aggregated[idx], rel=1e-8, abs=1e-8) == explicit
