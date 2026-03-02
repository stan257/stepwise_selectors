import copy
import numpy as np
import pytest

from selection import GramData
from selection import CrossValGramData
from selection import CrossValSelectionState
from selection import SelectionState


def make_random_state(n=30, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, y @ y, n_samples=n)
    return SelectionState(data)


def test_selection_state_rejects_non_positive_block_size_at_init():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((20, 4))
    y = rng.standard_normal(20)
    data = GramData(X.T @ X, X.T @ y, y @ y, n_samples=20)
    with pytest.raises(ValueError, match="positive integer"):
        SelectionState(data, block_size=0)


def test_selection_state_rejects_non_positive_block_size_before_forward_cache():
    state = make_random_state(p=4)
    state.init_from_active_set([0])
    state.block_size = -1
    with pytest.raises(ValueError, match="positive integer"):
        state.compute_forward_deltas()


def test_selection_state_init_full_populates_active_and_beta():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 0.2, 0.9])
    data = GramData(X.T @ X, X.T @ y, y @ y, n_samples=X.shape[0])
    state = SelectionState(data)

    state.init_full()

    assert state.active_set == [0, 1]
    np.testing.assert_allclose(state.beta, np.array([0.9, 0.1]))
    assert pytest.approx(state.rss) == state.data.y_norm - state.data.cov @ np.array(
        [0.9, 0.1]
    )


def test_rank_one_add_then_remove_restores_state():
    state = make_random_state(p=4)
    original_rss = state.rss
    cache = state.compute_forward_deltas()
    matches = np.where(cache.candidates == 2)[0]
    state.apply_forward_step(cache, int(matches[0]))
    state.apply_backward_step(len(state.active_set) - 1)
    assert state.active_set == []
    assert state.K is None
    assert pytest.approx(state.rss) == original_rss


def test_backward_scores_match_individual_updates():
    state = make_random_state(p=5)
    state.init_full()

    scores = state.compute_backward_scores()
    assert scores is not None
    assert scores.shape == (5,)

    for i in range(5):
        clone = copy.deepcopy(state)
        clone.apply_backward_step(i)
        assert pytest.approx(scores[i]) == clone.rss


@pytest.mark.parametrize(
    "active_set,match",
    [
        ([-1], "out of range"),
        ([4], "out of range"),
        ([1, 1], "duplicate"),
    ],
)
def test_init_from_active_set_rejects_invalid_indices(active_set, match):
    state = make_random_state(p=4)
    with pytest.raises(ValueError, match=match):
        state.init_from_active_set(active_set)


def test_init_from_active_set_rejects_non_integer_indices():
    state = make_random_state(p=4)
    with pytest.raises(TypeError, match="integers"):
        state.init_from_active_set([0, 1.5])


def test_cv_state_rejects_desynced_fold_active_sets():
    rng = np.random.default_rng(987)
    fold_data = []
    for _ in range(2):
        X = rng.standard_normal((30, 3))
        y = rng.standard_normal(30)
        fold_data.append(GramData(X.T @ X, X.T @ y, y @ y, n_samples=30))

    cv_state = CrossValSelectionState(CrossValGramData(fold_data))
    cv_state.train_states[0].init_from_active_set([0])
    cv_state.train_states[1].init_from_active_set([1])

    with pytest.raises(RuntimeError, match="identical active_set across folds"):
        cv_state._sync_active_set()
