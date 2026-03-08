import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection import CrossValGramData, GramData
from selection import (
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
from selection import GroupBackwardSelection, GroupForwardSelection
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


def _assert_states_match(lhs, rhs) -> None:
    assert lhs.active_set == rhs.active_set
    np.testing.assert_allclose(lhs.beta, rhs.beta, atol=1e-8, rtol=1e-8)
    assert lhs.rss == pytest.approx(rhs.rss, rel=1e-8, abs=1e-8)


def _assert_cv_states_match(lhs, rhs) -> None:
    assert lhs.active_set == rhs.active_set
    np.testing.assert_allclose(lhs.beta, rhs.beta, atol=1e-8, rtol=1e-8)
    assert lhs.rss_cv == pytest.approx(rhs.rss_cv, rel=1e-8, abs=1e-8)


def _make_strong_signal_diagonal_problem(p: int = 8, n_samples: int = 80) -> GramData:
    cov = np.linspace(3.0, 1.6, num=p)
    gram = np.eye(p)
    y_norm = float(np.sum(cov**2) + 5.0)
    return GramData(gram, cov, y_norm, n_samples)


def _make_strong_signal_cv_problem(
    p: int = 8, n_samples: int = 80, folds: int = 3
):
    base = _make_strong_signal_diagonal_problem(p=p, n_samples=n_samples)
    fold_data = [
        GramData(
            base.gram.copy(),
            base.cov.copy(),
            base.y_norm,
            base.n_samples,
            warn_if_uncentered=False,
        )
        for _ in range(folds)
    ]
    return CrossValGramData(fold_data)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("p", [32, 80])  # include p > 64 to exercise capacity growth
def test_greedy_consistency_sweep(seed: int, p: int):
    data = make_regression_gram(seed, n=140, p=p)
    max_steps = min(6, p // 2)

    fast_f = ForwardSelection(criterion=BestRSSCriterion).fit(
        data=data, max_steps=max_steps
    )
    _assert_state_consistent(fast_f, data)

    fast_b = BackwardSelection(allow_worse=True).fit(data=data, max_steps=4)
    _assert_state_consistent(fast_b, data)

    fast_m = MixedSelection(criterion=BestRSSCriterion).fit(
        data=data, max_forward_steps=3, max_total_steps=5
    )
    _assert_state_consistent(fast_m, data)


@pytest.mark.parametrize("seed", [2, 3])
def test_beam_consistency_sweep(seed: int):
    data = make_regression_gram(seed, n=120, p=25)

    fast_f = BeamForwardSelection(
        beam_width=3, criterion=BestRSSCriterion
    ).fit(data=data, max_steps=4)
    _assert_state_consistent(fast_f, data)

    fast_b = BeamBackwardSelection(
        beam_width=2, allow_worse=True, criterion=BestRSSCriterion
    ).fit(data=data, max_steps=3)
    _assert_state_consistent(fast_b, data)

    fast_m = BeamMixedSelection(
        beam_width=2, criterion=BestRSSCriterion
    ).fit(data=data, max_forward_steps=3, max_total_steps=6)
    _assert_state_consistent(fast_m, data)


@pytest.mark.parametrize("seed", [4, 5])
def test_cv_consistency_sweep(seed: int):
    cv_data = make_cv_regression_gram(seed, folds=3, n=80, p=18)

    fast_f = CrossValForwardSelection(
        criterion=BestRSSCriterion
    ).fit(data=cv_data, max_steps=4)
    assert pytest.approx(fast_f.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_f.active_set
    )

    fast_b = CrossValBackwardSelection(
        criterion=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert pytest.approx(fast_b.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_b.active_set
    )

    fast_m = CrossValMixedSelection(
        criterion=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    assert pytest.approx(fast_m.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_m.active_set
    )


@pytest.mark.parametrize("seed", [6, 7])
def test_cv_beam_consistency_sweep(seed: int):
    cv_data = make_cv_regression_gram(seed, folds=3, n=70, p=16)

    fast_f = BeamCrossValForwardSelection(
        beam_width=2, criterion=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert pytest.approx(fast_f.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_f.active_set
    )

    fast_b = BeamCrossValBackwardSelection(
        beam_width=2, criterion=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    assert pytest.approx(fast_b.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_b.active_set
    )

    fast_m = BeamCrossValMixedSelection(
        beam_width=2, criterion=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    assert pytest.approx(fast_m.rss_cv, rel=1e-8, abs=1e-8) == explicit_cv_rss(
        cv_data, fast_m.active_set
    )


@pytest.mark.parametrize("seed", [8, 9])
def test_grouped_consistency_sweep(seed: int):
    data = make_regression_gram(seed, n=100, p=12)
    groups = [list(range(i, i + 3)) for i in range(0, 12, 3)]

    fast_f = GroupForwardSelection(groups, criterion=BestRSSCriterion).fit(
        data=data, max_steps=3
    )
    _assert_group_state_consistent(fast_f, data, groups)

    fast_b = GroupBackwardSelection(groups, criterion=BestRSSCriterion).fit(
        data=data, max_steps=2
    )
    _assert_group_state_consistent(fast_b, data, groups)


@pytest.mark.parametrize("seed", [30, 31, 32])
def test_beam_width_one_matches_greedy_across_routines(seed: int):
    data = make_regression_gram(seed, n=120, p=24)

    greedy_f = ForwardSelection(criterion=BestRSSCriterion).fit(
        data=data, max_steps=5
    )
    beam_f = BeamForwardSelection(
        beam_width=1, criterion=BestRSSCriterion
    ).fit(data=data, max_steps=5)
    _assert_states_match(beam_f, greedy_f)

    greedy_b = BackwardSelection(
        criterion=BestRSSCriterion, allow_worse=True
    ).fit(data=data, max_steps=4)
    beam_b = BeamBackwardSelection(
        beam_width=1, criterion=BestRSSCriterion, allow_worse=True
    ).fit(data=data, max_steps=4)
    _assert_states_match(beam_b, greedy_b)

    greedy_m = MixedSelection(criterion=BestRSSCriterion).fit(
        data=data, max_forward_steps=3, max_total_steps=4
    )
    beam_m = BeamMixedSelection(
        beam_width=1, criterion=BestRSSCriterion
    ).fit(data=data, max_forward_steps=3, max_total_steps=4)
    _assert_states_match(beam_m, greedy_m)


@pytest.mark.parametrize("seed", [33, 34, 35])
def test_cv_beam_width_one_matches_greedy_across_routines(seed: int):
    cv_data = make_cv_regression_gram(seed, folds=4, n=90, p=18)

    greedy_f = CrossValForwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    beam_f = BeamCrossValForwardSelection(
        beam_width=1, criterion=BestRSSCriterion
    ).fit(data=cv_data, max_steps=4)
    _assert_cv_states_match(beam_f, greedy_f)

    greedy_b = CrossValBackwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_steps=3
    )
    beam_b = BeamCrossValBackwardSelection(
        beam_width=1, criterion=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    _assert_cv_states_match(beam_b, greedy_b)

    greedy_m = CrossValMixedSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_forward_steps=3, max_total_steps=4
    )
    beam_m = BeamCrossValMixedSelection(
        beam_width=1, criterion=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=4)
    _assert_cv_states_match(beam_m, greedy_m)


@pytest.mark.parametrize("seed", [40, 41])
@pytest.mark.parametrize("max_forward_steps,max_total_steps", [(2, 2), (3, 3), (3, 4)])
def test_beam_mixed_budget_matches_greedy_when_width_one(
    seed: int, max_forward_steps: int, max_total_steps: int
):
    data = make_regression_gram(seed, n=110, p=20)

    greedy = MixedSelection(criterion=BestRSSCriterion).fit(
        data=data,
        max_forward_steps=max_forward_steps,
        max_total_steps=max_total_steps,
    )
    beam = BeamMixedSelection(beam_width=1, criterion=BestRSSCriterion).fit(
        data=data,
        max_forward_steps=max_forward_steps,
        max_total_steps=max_total_steps,
    )
    _assert_states_match(beam, greedy)


@pytest.mark.parametrize("seed", [42, 43])
@pytest.mark.parametrize("max_forward_steps,max_total_steps", [(2, 2), (3, 4)])
def test_cv_beam_mixed_budget_matches_greedy_when_width_one(
    seed: int, max_forward_steps: int, max_total_steps: int
):
    cv_data = make_cv_regression_gram(seed, folds=4, n=80, p=16)

    greedy = CrossValMixedSelection(criterion=BestRSSCriterion).fit(
        data=cv_data,
        max_forward_steps=max_forward_steps,
        max_total_steps=max_total_steps,
    )
    beam = BeamCrossValMixedSelection(
        beam_width=1, criterion=BestRSSCriterion
    ).fit(
        data=cv_data,
        max_forward_steps=max_forward_steps,
        max_total_steps=max_total_steps,
    )
    _assert_cv_states_match(beam, greedy)


@pytest.mark.parametrize(
    "selector_factory,fit_kwargs",
    [
        pytest.param(
            lambda stop=False: ForwardSelection(
                criterion=BestRSSCriterion,
                stop_on_no_improvement=stop,
            ),
            {"max_steps": 3},
            id="forward",
        ),
        pytest.param(
            lambda stop=False: BeamForwardSelection(
                beam_width=2,
                criterion=BestRSSCriterion,
                stop_on_no_improvement=stop,
            ),
            {"max_steps": 3},
            id="beam_forward",
        ),
        pytest.param(
            lambda stop=False: MixedSelection(
                criterion=BestRSSCriterion,
                stop_on_no_improvement=stop,
            ),
            {"max_forward_steps": 3, "max_total_steps": 3},
            id="mixed",
        ),
        pytest.param(
            lambda stop=False: BeamMixedSelection(
                beam_width=2,
                criterion=BestRSSCriterion,
                stop_on_no_improvement=stop,
            ),
            {"max_forward_steps": 3, "max_total_steps": 3},
            id="beam_mixed",
        ),
    ],
)
def test_single_dataset_budget_mode_matches_legacy_when_steps_improve(
    selector_factory, fit_kwargs
):
    data = _make_strong_signal_diagonal_problem()

    budget = selector_factory().fit(data=data, **fit_kwargs)
    legacy = selector_factory(stop=True).fit(data=data, **fit_kwargs)

    _assert_states_match(budget, legacy)


def test_group_budget_mode_matches_legacy_when_steps_improve():
    data = _make_strong_signal_diagonal_problem(p=6)
    groups = [[idx] for idx in range(6)]

    budget = GroupForwardSelection(groups, criterion=BestRSSCriterion).fit(
        data=data, max_steps=3
    )
    legacy = GroupForwardSelection(
        groups,
        criterion=BestRSSCriterion,
        stop_on_no_improvement=True,
    ).fit(data=data, max_steps=3)

    assert budget.active_groups == legacy.active_groups
    _assert_group_state_consistent(budget, data, groups)
    _assert_group_state_consistent(legacy, data, groups)


@pytest.mark.parametrize(
    "selector_factory,fit_kwargs",
    [
        pytest.param(
            lambda stop=False: CrossValForwardSelection(
                criterion=BestRSSCriterion,
                stop_on_no_improvement=stop,
            ),
            {"max_steps": 3},
            id="cv_forward",
        ),
        pytest.param(
            lambda stop=False: BeamCrossValForwardSelection(
                beam_width=2,
                criterion=BestRSSCriterion,
                stop_on_no_improvement=stop,
            ),
            {"max_steps": 3},
            id="cv_beam_forward",
        ),
        pytest.param(
            lambda stop=False: CrossValMixedSelection(
                criterion=BestRSSCriterion,
                stop_on_no_improvement=stop,
            ),
            {"max_forward_steps": 3, "max_total_steps": 3},
            id="cv_mixed",
        ),
        pytest.param(
            lambda stop=False: BeamCrossValMixedSelection(
                beam_width=2,
                criterion=BestRSSCriterion,
                stop_on_no_improvement=stop,
            ),
            {"max_forward_steps": 3, "max_total_steps": 3},
            id="cv_beam_mixed",
        ),
    ],
)
def test_cv_budget_mode_matches_legacy_when_steps_improve(
    selector_factory, fit_kwargs
):
    cv_data = _make_strong_signal_cv_problem()

    budget = selector_factory().fit(data=cv_data, **fit_kwargs)
    legacy = selector_factory(stop=True).fit(data=cv_data, **fit_kwargs)

    _assert_cv_states_match(budget, legacy)
