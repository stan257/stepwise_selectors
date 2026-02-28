import numpy as np
import pytest
from selection.criteria import AICCriterion, BestRSSCriterion
from selection.definitions import GramData
from selection.grouped_routines import GroupBackwardSelection, GroupForwardSelection
from selection.state import GroupedSelectionState


class GroupRecordingCriterion(BestRSSCriterion):
    calls: list[tuple[int | None, int | None]] = []

    def __init__(self, *, n_samples: int | None = None, p: int | None = None, **kwargs):
        super().__init__(**kwargs)
        type(self).calls.append((n_samples, p))


def group_recording_criterion_factory(
    *, n_samples: int, p: int
) -> GroupRecordingCriterion:
    return GroupRecordingCriterion(n_samples=n_samples, p=p)


def make_group_problem():
    """
    Simple diagonal Gram with two groups:
    - Group 0: strong effects on features 0 and 1
    - Group 1: weaker effects on features 2 and 3
    """
    gram = np.eye(4)
    cov = np.array([3.0, 3.0, 0.1, 0.1])
    y_norm = float(cov @ cov) + 1.0  # leave some residual variance
    n_samples = 50
    groups = [[0, 1], [2, 3]]
    return GramData(gram, cov, y_norm, n_samples), groups


def _active_features(active_groups: list[int], groups: list[list[int]]) -> list[int]:
    active = []
    for group_idx in active_groups:
        active.extend(groups[group_idx])
    return sorted(active)


def test_group_forward_selects_strong_group_first():
    data, groups = make_group_problem()
    selector = GroupForwardSelection(groups, criterion_cls=BestRSSCriterion)
    state = selector.fit(data=data, max_steps=1)

    assert isinstance(state, GroupedSelectionState)
    assert state.data is data
    assert state.groups == tuple(tuple(g) for g in groups)
    assert state.active_groups == [0]
    assert state.active_set == _active_features(state.active_groups, groups)
    np.testing.assert_allclose(state.beta[:2], data.cov[:2])
    np.testing.assert_allclose(state.beta[2:], 0.0)
    assert state.rss < data.y_norm


def test_group_forward_adds_all_when_budget_allows():
    data, groups = make_group_problem()
    selector = GroupForwardSelection(groups, criterion_cls=BestRSSCriterion)
    state = selector.fit(data=data, max_steps=2)

    assert isinstance(state, GroupedSelectionState)
    assert state.data is data
    assert state.groups == tuple(tuple(g) for g in groups)
    assert set(state.active_groups) == {0, 1}
    assert state.active_set == _active_features(state.active_groups, groups)
    np.testing.assert_allclose(state.beta, data.cov)
    assert state.rss < data.y_norm


def test_group_backward_drops_weak_group():
    data, groups = make_group_problem()
    selector = GroupBackwardSelection(groups, criterion_cls=AICCriterion)
    state = selector.fit(data=data, max_steps=1)

    assert isinstance(state, GroupedSelectionState)
    assert state.data is data
    assert state.groups == tuple(tuple(g) for g in groups)
    assert state.active_groups == [0]
    assert state.active_set == _active_features(state.active_groups, groups)
    np.testing.assert_allclose(state.beta[:2], data.cov[:2])
    np.testing.assert_allclose(state.beta[2:], 0.0)
    assert state.rss < data.y_norm


def test_group_selection_rejects_overlapping_groups():
    with pytest.raises(ValueError, match="disjoint"):
        GroupForwardSelection([[0, 1], [1, 2]])


def test_group_selection_rejects_duplicate_feature_within_group():
    with pytest.raises(ValueError, match="unique features"):
        GroupForwardSelection([[0, 0], [1, 2]])


def test_group_selection_rejects_negative_feature_index():
    with pytest.raises(ValueError, match="non-negative"):
        GroupForwardSelection([[0, -1], [1, 2]])


def test_group_selection_rejects_non_integer_feature_index():
    with pytest.raises(TypeError, match="integers"):
        GroupForwardSelection([[0, 1.5], [1, 2]])


def test_group_selection_rejects_out_of_range_index_at_fit():
    data, _ = make_group_problem()
    selector = GroupForwardSelection([[0, 4], [1, 2]])
    with pytest.raises(ValueError, match="out of range"):
        selector.fit(data=data, max_steps=1)


def test_group_forward_accepts_criterion_factory_with_auto_params():
    data, groups = make_group_problem()
    GroupRecordingCriterion.calls.clear()
    selector = GroupForwardSelection(groups, criterion=group_recording_criterion_factory)
    selector.fit(data=data, max_steps=1)
    assert GroupRecordingCriterion.calls
    assert GroupRecordingCriterion.calls[-1] == (data.n_samples, data.gram.shape[0])
