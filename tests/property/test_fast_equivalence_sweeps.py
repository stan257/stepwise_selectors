import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.routines_core import (
    BackwardSelection,
    BeamBackwardSelection,
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
    BeamForwardSelection,
    BeamMixedSelection,
    CrossValBackwardSelection,
    CrossValForwardSelection,
    CrossValMixedSelection,
    ForwardSelection,
    MixedSelection,
)
from selection.grouped_routines import GroupBackwardSelection, GroupForwardSelection
from tests.helpers import (
    explicit_beta_rss,
    explicit_cv_rss,
    make_cv_regression_gram,
    make_regression_gram,
)


def _assert_state_consistent(state, data: GramData):
    beta, rss = explicit_beta_rss(data, state.active_set)
    np.testing.assert_allclose(state.beta, beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss


def _assert_group_state_consistent(state, data: GramData, groups):
    active_groups = state.active_groups
    idx = []
    for g in active_groups:
        idx.extend(groups[g])
    beta, rss = explicit_beta_rss(data, idx)
    np.testing.assert_allclose(state.beta, beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("p", [32, 80])  # include p > 64 to exercise capacity growth
def test_fast_greedy_consistency_sweep(seed: int, p: int):
    data = make_regression_gram(seed, n=140, p=p)
    max_steps = min(6, p // 2)

    fast_f = ForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=max_steps
    )
    _assert_state_consistent(fast_f, data)

    fast_b = BackwardSelection(allow_worse=True).fit(data=data, max_steps=4)
    _assert_state_consistent(fast_b, data)

    fast_m = MixedSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_forward_steps=3, max_total_steps=5
    )
    _assert_state_consistent(fast_m, data)


@pytest.mark.parametrize("seed", [2, 3])
def test_fast_beam_consistency_sweep(seed: int):
    data = make_regression_gram(seed, n=120, p=25)

    fast_f = BeamForwardSelection(
        beam_width=3, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=4)
    _assert_state_consistent(fast_f, data)

    fast_b = BeamBackwardSelection(
        beam_width=2, allow_worse=True, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=3)
    _assert_state_consistent(fast_b, data)

    fast_m = BeamMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_forward_steps=3, max_total_steps=6)
    _assert_state_consistent(fast_m, data)


@pytest.mark.parametrize("seed", [4, 5])
def test_fast_cv_consistency_sweep(seed: int):
    cv_data = make_cv_regression_gram(seed, folds=3, n=80, p=18)

    fast_f = CrossValForwardSelection(
        criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=4)
    assert pytest.approx(fast_f.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_f.active_set
    )

    fast_b = CrossValBackwardSelection(
        criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert pytest.approx(fast_b.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_b.active_set
    )

    fast_m = CrossValMixedSelection(
        criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    assert pytest.approx(fast_m.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_m.active_set
    )


@pytest.mark.parametrize("seed", [6, 7])
def test_fast_cv_beam_consistency_sweep(seed: int):
    cv_data = make_cv_regression_gram(seed, folds=3, n=70, p=16)

    fast_f = BeamCrossValForwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert pytest.approx(fast_f.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_f.active_set
    )

    fast_b = BeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert pytest.approx(fast_b.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_b.active_set
    )

    fast_m = BeamCrossValMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    assert pytest.approx(fast_m.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_m.active_set
    )


@pytest.mark.parametrize("seed", [8, 9])
def test_fast_grouped_consistency_sweep(seed: int):
    data = make_regression_gram(seed, n=100, p=12)
    groups = [list(range(i, i + 3)) for i in range(0, 12, 3)]

    fast_f = GroupForwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=3
    )
    _assert_group_state_consistent(fast_f, data, groups)

    fast_b = GroupBackwardSelection(groups, criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=2
    )
    _assert_group_state_consistent(fast_b, data, groups)
