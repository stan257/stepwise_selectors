import numpy as np
import pytest

from benchmarks.baselines import (
    AdaptiveLassoBaseline,
    BASELINE_MAP,
    LassoCVBaseline,
    TopKAbsCovBaseline,
)


def _synthetic_regression(seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = 180, 10
    X = rng.standard_normal((n, p))
    beta = np.zeros(p, dtype=float)
    beta[1] = 2.5
    beta[4] = -2.0
    beta[7] = 1.5
    y = X @ beta + 0.2 * rng.standard_normal(n)
    return X, y


def test_baseline_map_exposes_new_entries():
    assert "TopKAbsCovBaseline" in BASELINE_MAP
    assert "LassoCVBaseline" in BASELINE_MAP
    assert "AdaptiveLassoBaseline" in BASELINE_MAP


def test_topk_baseline_shapes():
    X, y = _synthetic_regression()
    state = TopKAbsCovBaseline(k=3).fit(X_train=X, y_train=y)
    assert state.beta.shape == (X.shape[1],)
    assert len(state.active_set) == 3
    assert np.isfinite(state.rss)
    assert state.intercept == 0.0


def test_lasso_cv_baseline_recovers_signal_support():
    pytest.importorskip("sklearn")
    X, y = _synthetic_regression()
    state = LassoCVBaseline(cv_folds=3, random_state=0).fit(X_train=X, y_train=y)

    active = set(state.active_set)
    assert {1, 4, 7}.issubset(active)
    assert state.beta.shape == (X.shape[1],)
    assert np.isfinite(state.intercept)
    assert np.isfinite(state.rss)


def test_adaptive_lasso_baseline_recovers_signal_support():
    pytest.importorskip("sklearn")
    X, y = _synthetic_regression()
    state = AdaptiveLassoBaseline(cv_folds=3, random_state=0, gamma=1.0).fit(
        X_train=X, y_train=y
    )

    active = set(state.active_set)
    assert {1, 4, 7}.issubset(active)
    assert state.beta.shape == (X.shape[1],)
    assert np.isfinite(state.intercept)
    assert np.isfinite(state.rss)
