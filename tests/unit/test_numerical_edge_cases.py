import numpy as np
import pytest

from selection.constants import ABS_TOL
from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.routines import ForwardSelection
from selection.state_single import SelectionState


def test_forward_stops_after_duplicate_column():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.column_stack([x, x])
    y = x.copy()

    gram = X.T @ X
    cov = X.T @ y
    y_norm = float(y @ y)

    data = GramData(gram, cov, y_norm, n_samples=len(y))
    state = ForwardSelection().fit(data=data, max_steps=2)

    assert len(state.active_set) == 1
    assert state.active_set[0] in (0, 1)


def test_near_singular_candidate_is_filtered_by_tol():
    delta = ABS_TOL * 0.1
    gram = np.array([[1.0, 1.0 - delta], [1.0 - delta, 1.0]])
    cov = np.array([1.0, 1.0 - delta])
    y_norm = float(cov @ cov)

    state = SelectionState(GramData(gram, cov, y_norm, n_samples=10))
    cache = state.compute_forward_deltas()
    assert cache is not None
    idx0 = int(np.where(cache.candidates == 0)[0][0])
    state.apply_forward_step(cache, idx0)

    cache2 = state.compute_forward_deltas()
    assert cache2 is None or not cache2.candidates.size


def test_forward_handles_tiny_rss_without_nan():
    gram = np.array([[1.0]])
    cov = np.array([1.0])
    y_norm = 1.0

    data = GramData(gram, cov, y_norm, n_samples=5)
    state = ForwardSelection().fit(data=data)

    assert len(state.active_set) == 1
    assert np.isfinite(state.rss)
    assert 0.0 <= state.rss <= ABS_TOL * 1.1


def test_forward_near_singular_gram_remains_finite():
    rng = np.random.default_rng(123)
    n, p = 80, 6
    x1 = rng.standard_normal(n)
    # Create a nearly collinear column to stress conditioning.
    x2 = x1 + 1e-6 * rng.standard_normal(n)
    X = np.column_stack([x1, x2, rng.standard_normal((n, p - 2))])
    beta = rng.standard_normal(p)
    y = X @ beta + 0.01 * rng.standard_normal(n)

    data = GramData(X.T @ X, X.T @ y, y @ y, n_samples=n)
    state = ForwardSelection(criterion=BestRSSCriterion).fit(
        data=data, max_steps=3
    )

    assert np.isfinite(state.rss)
    assert np.isfinite(state.beta).all()
    if state.active_set:
        scores = state.compute_backward_scores()
        assert scores is not None
        assert np.isfinite(scores).all()


def test_forward_tiny_y_norm_stops_cleanly():
    gram = np.array([[1.0]])
    cov = np.array([1e-7])
    y_norm = 1e-12

    data = GramData(gram, cov, y_norm, n_samples=50)
    state = ForwardSelection().fit(data=data, max_steps=1)

    # AIC penalty should prevent adding the tiny-signal feature.
    assert state.active_set == []
    assert np.isfinite(state.rss)
    assert pytest.approx(state.rss, rel=1e-12, abs=0.0) == y_norm


def test_forward_large_p_zero_cov_stops_immediately():
    p = 300
    gram = np.eye(p)
    cov = np.zeros(p)
    y_norm = 10.0

    data = GramData(gram, cov, y_norm, n_samples=100)
    state = ForwardSelection().fit(data=data, max_steps=5)

    assert state.active_set == []
    assert np.isfinite(state.rss)
    assert pytest.approx(state.rss, rel=1e-12, abs=0.0) == y_norm
