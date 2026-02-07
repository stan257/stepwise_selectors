import numpy as np

from selection.constants import ABS_TOL
from selection.definitions import GramData
from selection.routines import ForwardSelection
from selection.state import SelectionState


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
