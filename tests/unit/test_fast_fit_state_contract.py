import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.routines_core import (
    BackwardSelection,
    BeamBackwardSelection,
    BeamForwardSelection,
    BeamMixedSelection,
    ForwardSelection,
    MixedSelection,
)
from selection.state import SelectionState
from tests.helpers import make_regression_gram


NON_CV_SELECTOR_CASES = [
    pytest.param(
        ForwardSelection,
        {"criterion_cls": BestRSSCriterion},
        {"max_steps": 3},
        id="forward",
    ),
    pytest.param(
        BackwardSelection,
        {"criterion_cls": BestRSSCriterion, "allow_worse": True},
        {"max_steps": 3},
        id="backward",
    ),
    pytest.param(
        MixedSelection,
        {"criterion_cls": BestRSSCriterion},
        {"max_forward_steps": 3, "max_total_steps": 5},
        id="mixed",
    ),
    pytest.param(
        BeamForwardSelection,
        {"criterion_cls": BestRSSCriterion, "beam_width": 2},
        {"max_steps": 3},
        id="beam_forward",
    ),
    pytest.param(
        BeamBackwardSelection,
        {"criterion_cls": BestRSSCriterion, "beam_width": 2, "allow_worse": True},
        {"max_steps": 3},
        id="beam_backward",
    ),
    pytest.param(
        BeamMixedSelection,
        {"criterion_cls": BestRSSCriterion, "beam_width": 2},
        {"max_forward_steps": 3, "max_total_steps": 5},
        id="beam_mixed",
    ),
]


@pytest.mark.parametrize("selector_cls,selector_kwargs,fit_kwargs", NON_CV_SELECTOR_CASES)
def test_non_cv_selector_rejects_mismatched_state_data(
    selector_cls, selector_kwargs, fit_kwargs
):
    data = make_regression_gram(111, n=100, p=10)
    other_data = make_regression_gram(222, n=100, p=10)
    state = SelectionState(other_data)

    selector = selector_cls(**selector_kwargs)
    with pytest.raises(ValueError, match="state.data"):
        selector.fit(state=state, data=data, **fit_kwargs)


@pytest.mark.parametrize("selector_cls,selector_kwargs,fit_kwargs", NON_CV_SELECTOR_CASES)
def test_non_cv_selector_reuses_matching_state(
    selector_cls, selector_kwargs, fit_kwargs
):
    data = make_regression_gram(333, n=120, p=12)

    expected = selector_cls(**selector_kwargs).fit(data=data, **fit_kwargs)
    target = SelectionState(data)
    result = selector_cls(**selector_kwargs).fit(
        state=target, data=data, **fit_kwargs
    )

    assert result is target
    assert result.active_set == expected.active_set
    np.testing.assert_allclose(result.beta, expected.beta, atol=1e-8, rtol=1e-8)
    assert result.rss == pytest.approx(expected.rss, rel=1e-8, abs=1e-8)
