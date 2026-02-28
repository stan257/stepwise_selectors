import numpy as np
import pytest

from selection.criteria import AICCriterion, BestRSSCriterion, GCVCriterion
from selection.definitions import CrossValGramData, GramData
from selection.routines_core import (
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
    CrossValBackwardSelection,
    CrossValForwardSelection,
    CrossValMixedSelection,
)
from selection.routines import (
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
    CrossValBackwardSelection,
    CrossValForwardSelection,
    CrossValMixedSelection,
)
from selection.state import CrossValSelectionState
from tests.helpers import explicit_beta_from_active, explicit_cv_rss, make_cv_problem


def test_crossvalgramdata_requires_at_least_two_folds():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 4))
    y = rng.standard_normal(20)
    fold = GramData(X.T @ X, X.T @ y, y @ y, n_samples=20)
    with pytest.raises(
        ValueError, match="requires at least two folds"
    ):
        CrossValGramData([fold])


def test_fast_cv_forward_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = CrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    expected_rss = explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_fast_cv_backward_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = CrossValBackwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    expected_rss = explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_fast_cv_mixed_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = CrossValMixedSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_forward_steps=3, max_total_steps=5
    )
    expected_rss = explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_fast_cv_state_exposes_full_data_postselection_beta():
    cv_data = make_cv_problem(seed=2026, folds=4, n=100, p=10, support=4)
    state = CrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    full_data = cv_data.make_full_data()
    expected_beta = explicit_beta_from_active(full_data, state.active_set)
    np.testing.assert_allclose(state.beta, expected_beta, atol=1e-8, rtol=1e-8)

    empty = CrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=0
    )
    np.testing.assert_allclose(empty.beta, np.zeros_like(empty.beta), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "selector_cls,selector_kwargs",
    [
        pytest.param(
            CrossValMixedSelection,
            {"criterion_cls": BestRSSCriterion},
            id="cv_mixed",
        ),
        pytest.param(
            BeamCrossValMixedSelection,
            {"criterion_cls": BestRSSCriterion, "beam_width": 2},
            id="cv_beam_mixed",
        ),
    ],
)
def test_cv_mixed_selectors_respect_zero_forward_budget(selector_cls, selector_kwargs):
    cv_data = make_cv_problem(seed=2027, folds=3, n=80, p=8, support=3)

    state = selector_cls(**selector_kwargs).fit(
        data=cv_data, max_forward_steps=0, max_total_steps=10
    )
    assert state.active_set == []


CV_SELECTOR_CASES = [
    pytest.param(
        CrossValForwardSelection,
        {"criterion_cls": BestRSSCriterion},
        {"max_steps": 3},
        id="cv_forward",
    ),
    pytest.param(
        CrossValBackwardSelection,
        {"criterion_cls": BestRSSCriterion},
        {"max_steps": 3},
        id="cv_backward",
    ),
    pytest.param(
        CrossValMixedSelection,
        {"criterion_cls": BestRSSCriterion},
        {"max_forward_steps": 3, "max_total_steps": 5},
        id="cv_mixed",
    ),
    pytest.param(
        BeamCrossValForwardSelection,
        {"criterion_cls": BestRSSCriterion, "beam_width": 2},
        {"max_steps": 3},
        id="cv_beam_forward",
    ),
    pytest.param(
        BeamCrossValBackwardSelection,
        {"criterion_cls": BestRSSCriterion, "beam_width": 2, "allow_worse": True},
        {"max_steps": 3},
        id="cv_beam_backward",
    ),
    pytest.param(
        BeamCrossValMixedSelection,
        {"criterion_cls": BestRSSCriterion, "beam_width": 2},
        {"max_forward_steps": 3, "max_total_steps": 5},
        id="cv_beam_mixed",
    ),
]


@pytest.mark.parametrize("selector_cls,selector_kwargs,fit_kwargs", CV_SELECTOR_CASES)
def test_cv_selector_rejects_mismatched_state_data(
    selector_cls, selector_kwargs, fit_kwargs
):
    cv_data = make_cv_problem(seed=111)
    other_data = make_cv_problem(seed=222)
    state = CrossValSelectionState(other_data)

    selector = selector_cls(**selector_kwargs)
    with pytest.raises(ValueError, match="state.data"):
        selector.fit(state=state, data=cv_data, **fit_kwargs)


@pytest.mark.parametrize("selector_cls,selector_kwargs,fit_kwargs", CV_SELECTOR_CASES)
def test_cv_selector_reuses_matching_state(
    selector_cls, selector_kwargs, fit_kwargs
):
    cv_data = make_cv_problem(seed=333)

    expected = selector_cls(**selector_kwargs).fit(data=cv_data, **fit_kwargs)
    target = CrossValSelectionState(cv_data)
    result = selector_cls(**selector_kwargs).fit(
        state=target, data=cv_data, **fit_kwargs
    )

    assert isinstance(result, CrossValSelectionState)
    assert result is target
    assert result.active_set == expected.active_set
    np.testing.assert_allclose(result.beta, expected.beta, atol=1e-8, rtol=1e-8)
    assert result.rss_cv == pytest.approx(expected.rss_cv, rel=1e-8, abs=1e-8)


@pytest.mark.parametrize(
    "selector_cls",
    [
        CrossValForwardSelection,
        CrossValBackwardSelection,
        CrossValMixedSelection,
        BeamCrossValForwardSelection,
        BeamCrossValBackwardSelection,
        BeamCrossValMixedSelection,
    ],
)
def test_cv_selectors_default_to_best_rss(selector_cls):
    selector = selector_cls()
    assert selector.criterion_cls is BestRSSCriterion


@pytest.mark.parametrize(
    "selector_cls",
    [
        CrossValForwardSelection,
        CrossValBackwardSelection,
        CrossValMixedSelection,
        BeamCrossValForwardSelection,
        BeamCrossValBackwardSelection,
        BeamCrossValMixedSelection,
    ],
)
@pytest.mark.parametrize("criterion_cls", [AICCriterion, GCVCriterion])
def test_cv_selectors_reject_disallowed_criteria(selector_cls, criterion_cls):
    with pytest.raises(
        ValueError,
        match=rf"{criterion_cls.__name__} is not supported for CV selection routines",
    ):
        selector_cls(criterion_cls=criterion_cls)
