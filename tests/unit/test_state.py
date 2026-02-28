import copy
import numpy as np
import pytest

from selection.definitions import CrossValGramData, GramData
from selection.state import CrossValSelectionState, SelectionState


def make_random_state(n=30, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, y @ y, n_samples=n)
    return SelectionState(data)


def test_selection_state_validates_input_shapes():
    G = np.eye(2)
    c = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        SelectionState(GramData(G, c, y_norm=1.0, n_samples=10))


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


def test_forward_update_matches_direct_inverse():
    state = make_random_state()
    for j in [0, 1]:
        cache = state.compute_forward_deltas()
        assert cache is not None
        matches = np.where(cache.candidates == j)[0]
        assert matches.size
        idx = int(matches[0])
        applied = state.apply_forward_step(cache, idx)
        assert applied == j
        idx_arr = np.array(state.active_set, dtype=int)
        K_expected = np.linalg.inv(state.data.gram[np.ix_(idx_arr, idx_arr)])
        np.testing.assert_allclose(state.K, K_expected, atol=1e-10)
        rss_expected = state.data.y_norm - state.data.cov[idx_arr] @ (
            K_expected @ state.data.cov[idx_arr]
        )
        assert pytest.approx(state.rss) == rss_expected


def test_backward_update_matches_direct_inverse():
    state = make_random_state()
    state.init_full()
    state.apply_backward_step(1)
    idx = np.array(state.active_set, dtype=int)
    if len(idx):
        K_expected = np.linalg.inv(state.data.gram[np.ix_(idx, idx)])
        np.testing.assert_allclose(state.K, K_expected, atol=1e-10)
    else:
        assert state.K is None


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


def test_rank_one_remove_then_add_restores_state():
    state = make_random_state(p=4)
    state.init_full()
    original_K = state.K.copy()
    original_beta = state.beta.copy()
    original_rss = state.rss
    original_active = state.active_set.copy()
    idx = 1
    var = state.active_set[idx]
    state.apply_backward_step(idx)
    cache = state.compute_forward_deltas()
    matches = np.where(cache.candidates == var)[0]
    state.apply_forward_step(cache, int(matches[0]))
    perm = [original_active.index(v) for v in state.active_set]
    K_target = original_K[np.ix_(perm, perm)]
    np.testing.assert_allclose(state.K, K_target, atol=1e-10)
    np.testing.assert_allclose(state.beta, original_beta, atol=1e-10)
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


@pytest.mark.parametrize("pivot_mode", ["negative", "nan"])
def test_backward_rejects_unstable_pivot(pivot_mode):
    state = make_random_state(p=3)
    state.init_from_active_set([0])
    if pivot_mode == "negative":
        state.K[0, 0] = -abs(state.K[0, 0])
    else:
        state.K[0, 0] = np.nan

    scores = state.compute_backward_scores()
    assert scores is not None
    assert np.isinf(scores[0])

    with pytest.raises(np.linalg.LinAlgError):
        state.apply_backward_step(0)


@pytest.mark.parametrize("pivot_mode", ["negative", "nan"])
def test_cv_validation_backward_rejects_unstable_pivot(pivot_mode):
    rng = np.random.default_rng(987)
    fold_data = []
    for _ in range(2):
        X = rng.standard_normal((30, 3))
        y = rng.standard_normal(30)
        fold_data.append(GramData(X.T @ X, X.T @ y, y @ y, n_samples=30))

    cv_state = CrossValSelectionState(CrossValGramData(fold_data))
    for train_state in cv_state.train_states:
        train_state.init_from_active_set([0])
        if pivot_mode == "negative":
            train_state.K[0, 0] = -abs(train_state.K[0, 0])
        else:
            train_state.K[0, 0] = np.nan
    cv_state._sync_active_set()

    rss = cv_state.validation_rss_for_backward_candidate(0, 0, tol=1e-12)
    assert np.isinf(rss)


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
