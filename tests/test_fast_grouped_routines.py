import numpy as np
import pytest

from selection.criteria import AICCriterion, BestRSSCriterion
from selection.definitions import GramData
from selection.grouped_routines import (
    FastGroupBackwardSelection,
    FastGroupForwardSelection,
)


def make_group_problem():
    gram = np.eye(4)
    cov = np.array([3.0, 3.0, 0.1, 0.1])
    y_norm = float(cov @ cov) + 1.0
    n_samples = 50
    groups = [[0, 1], [2, 3]]
    return GramData(gram, cov, y_norm, n_samples), groups


def _explicit_beta_rss(data: GramData, active_idx: list[int]):
    p = data.gram.shape[0]
    beta = np.zeros(p, dtype=float)
    if not active_idx:
        return beta, float(data.y_norm)
    idx = np.array(active_idx, dtype=int)
    gram_ss = data.gram[np.ix_(idx, idx)]
    cov_s = data.cov[idx]
    beta_s = np.linalg.solve(gram_ss, cov_s)
    beta[idx] = beta_s
    rss = float(data.y_norm - cov_s @ beta_s)
    return beta, rss


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
    active_idx = _active_features(state.active_groups, groups)
    beta_expected, rss_expected = _explicit_beta_rss(data, active_idx)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected


def test_fast_group_backward_matches_explicit_solution():
    data, groups = make_group_problem()
    state = FastGroupBackwardSelection(groups, criterion_cls=AICCriterion).fit(
        data=data, max_steps=1
    )
    active_idx = _active_features(state.active_groups, groups)
    beta_expected, rss_expected = _explicit_beta_rss(data, active_idx)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected
