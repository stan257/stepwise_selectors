import numpy as np
import pytest

from selection.criteria import AICCriterion, BestRSSCriterion
from selection.definitions import GramData
from selection.grouped_routines import (
    FastGroupBackwardSelection,
    FastGroupForwardSelection,
)
from selection.state import GroupedSelectionState
from tests.helpers import explicit_beta_rss


def make_group_problem():
    gram = np.eye(4)
    cov = np.array([3.0, 3.0, 0.1, 0.1])
    y_norm = float(cov @ cov) + 1.0
    n_samples = 50
    groups = [[0, 1], [2, 3]]
    return GramData(gram, cov, y_norm, n_samples), groups


def _active_features(active_groups: list[int], groups: list[list[int]]) -> list[int]:
    active = []
    for group_idx in active_groups:
        active.extend(groups[group_idx])
    return sorted(active)


def test_fast_group_forward_matches_explicit_solution():
    data, groups = make_group_problem()
    state = FastGroupForwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=2
    )
    assert isinstance(state, GroupedSelectionState)
    assert state.data is data
    assert state.groups == tuple(tuple(g) for g in groups)
    active_idx = _active_features(state.active_groups, groups)
    assert state.active_set == active_idx
    beta_expected, rss_expected = explicit_beta_rss(data, active_idx)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected


def test_fast_group_backward_matches_explicit_solution():
    data, groups = make_group_problem()
    state = FastGroupBackwardSelection(groups, criterion_cls=AICCriterion).fit(
        data=data, max_steps=1
    )
    assert isinstance(state, GroupedSelectionState)
    assert state.data is data
    assert state.groups == tuple(tuple(g) for g in groups)
    active_idx = _active_features(state.active_groups, groups)
    assert state.active_set == active_idx
    beta_expected, rss_expected = explicit_beta_rss(data, active_idx)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected
