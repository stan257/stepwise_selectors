import numpy as np
import pytest

from selection.criteria import AICCriterion, BestRSSCriterion
from selection.definitions import GramData
from selection.grouped_routines import FastGroupBackwardSelection, FastGroupForwardSelection
from selection.legacy_grouped_routines import (
    GroupBackwardSelection,
    GroupForwardSelection,
)


def make_group_problem():
    gram = np.eye(4)
    cov = np.array([3.0, 3.0, 0.1, 0.1])
    y_norm = float(cov @ cov) + 1.0
    n_samples = 50
    groups = [[0, 1], [2, 3]]
    return GramData(gram, cov, y_norm, n_samples), groups


def test_fast_group_forward_matches_standard():
    data, groups = make_group_problem()
    fast = FastGroupForwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=2
    )
    ref = GroupForwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=2
    )

    assert set(fast.active_groups) == set(ref.active_groups)
    np.testing.assert_allclose(fast.beta, ref.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast.rss, rel=1e-8, abs=1e-8) == ref.rss


def test_fast_group_backward_matches_standard():
    data, groups = make_group_problem()
    fast = FastGroupBackwardSelection(groups, criterion_cls=AICCriterion).fit(
        data=data, max_steps=1
    )
    ref = GroupBackwardSelection(groups, criterion_cls=AICCriterion).fit(
        data=data, max_steps=1
    )

    assert fast.active_groups == ref.active_groups
    np.testing.assert_allclose(fast.beta, ref.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast.rss, rel=1e-8, abs=1e-8) == ref.rss
