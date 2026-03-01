import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.routines import (
    BeamForwardSelection,
    BeamCrossValMixedSelection,
    BeamMixedSelection,
    ForwardSelection,
    MixedSelection,
)
from selection.state_single import SelectionState
from tests.helpers import make_cv_problem, make_regression_gram


class RecordingRSSCriterion(BestRSSCriterion):
    init_history: list[tuple[int | None, int | None]] = []

    def __init__(self, *, n_samples: int | None = None, p: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.p = p
        type(self).init_history.append((n_samples, p))


def recording_criterion_factory(*, n_samples: int, p: int) -> RecordingRSSCriterion:
    return RecordingRSSCriterion(n_samples=n_samples, p=p)


def recording_criterion_kwargs_factory(**kwargs) -> RecordingRSSCriterion:
    return RecordingRSSCriterion(**kwargs)


def raising_typeerror_factory(**kwargs):
    _ = kwargs
    raise TypeError("inner detail marker")


NON_CV_SELECTOR_CASES = [
    pytest.param(
        ForwardSelection,
        {"criterion": BestRSSCriterion},
        {"max_steps": 3},
        id="forward",
    ),
    pytest.param(
        BeamMixedSelection,
        {"criterion": BestRSSCriterion, "beam_width": 2},
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


@pytest.mark.parametrize(
    "selector_cls,selector_kwargs",
    [
        pytest.param(
            MixedSelection,
            {"criterion": BestRSSCriterion},
            id="mixed",
        ),
        pytest.param(
            BeamMixedSelection,
            {"criterion": BestRSSCriterion, "beam_width": 2},
            id="beam_mixed",
        ),
    ],
)
def test_mixed_selectors_respect_zero_forward_budget(selector_cls, selector_kwargs):
    data = GramData(
        gram=np.eye(4),
        cov=np.array([4.0, 3.0, 2.0, 1.0]),
        y_norm=40.0,
        n_samples=100,
    )

    state = selector_cls(**selector_kwargs).fit(
        data=data, max_forward_steps=0, max_total_steps=10
    )
    assert state.active_set == []


def test_beam_mixed_selector_respects_zero_total_budget():
    data = make_regression_gram(808, n=100, p=10)
    state = BeamMixedSelection(beam_width=3, criterion=BestRSSCriterion).fit(
        data=data, max_forward_steps=5, max_total_steps=0
    )
    assert state.active_set == []


def test_cv_beam_mixed_selector_respects_zero_total_budget():
    cv_data = make_cv_problem(seed=809, folds=4, n=80, p=10, support=3)
    state = BeamCrossValMixedSelection(
        beam_width=3, criterion=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=5, max_total_steps=0)
    assert state.active_set == []


def test_forward_selection_accepts_named_rss_criterion_key():
    data = make_regression_gram(4441, n=100, p=10)
    selector = ForwardSelection(criterion="rss")
    result = selector.fit(data=data, max_steps=3)
    assert isinstance(result, SelectionState)
    assert len(result.active_set) <= 3


def test_forward_selection_accepts_legacy_criterion_cls_alias():
    data = make_regression_gram(44410, n=100, p=10)
    selector = ForwardSelection(criterion_cls="rss")
    result = selector.fit(data=data, max_steps=3)
    assert isinstance(result, SelectionState)
    assert len(result.active_set) <= 3


def test_beam_forward_selection_accepts_legacy_criterion_cls_alias():
    data = make_regression_gram(44411, n=100, p=10)
    selector = BeamForwardSelection(beam_width=2, criterion_cls=BestRSSCriterion)
    result = selector.fit(data=data, max_steps=3)
    assert isinstance(result, SelectionState)
    assert len(result.active_set) <= 3


def test_forward_selection_rejects_conflicting_criterion_and_criterion_cls():
    with pytest.raises(
        ValueError, match=r"both `criterion` and legacy `criterion_cls`"
    ):
        ForwardSelection(criterion="rss", criterion_cls=BestRSSCriterion)


def test_forward_selection_rejects_unknown_criterion_key():
    data = make_regression_gram(4442, n=100, p=10)
    selector = ForwardSelection(criterion="not_a_real_criterion")
    with pytest.raises(ValueError, match=r"unknown criterion key"):
        selector.fit(data=data, max_steps=2)


def test_forward_selection_accepts_named_aic_criterion_key():
    data = make_regression_gram(4443, n=100, p=10)
    selector = ForwardSelection(criterion="aic")
    result = selector.fit(data=data, max_steps=3)
    assert isinstance(result, SelectionState)
    assert len(result.active_set) <= 3


def test_forward_selection_accepts_criterion_factory_with_auto_params():
    data = make_regression_gram(445, n=90, p=9)
    RecordingRSSCriterion.init_history.clear()
    selector = ForwardSelection(criterion=recording_criterion_factory)
    selector.fit(data=data, max_steps=2)
    assert RecordingRSSCriterion.init_history
    assert RecordingRSSCriterion.init_history[-1] == (data.n_samples, data.gram.shape[0])


def test_forward_selection_accepts_kwargs_only_criterion_factory():
    data = make_regression_gram(4451, n=95, p=11)
    RecordingRSSCriterion.init_history.clear()
    selector = ForwardSelection(criterion=recording_criterion_kwargs_factory)
    selector.fit(data=data, max_steps=2)
    assert RecordingRSSCriterion.init_history
    assert RecordingRSSCriterion.init_history[-1] == (data.n_samples, data.gram.shape[0])


def test_forward_selection_surfaces_factory_typeerror_detail():
    data = make_regression_gram(4452, n=90, p=9)
    selector = ForwardSelection(criterion=raising_typeerror_factory)

    with pytest.raises(TypeError, match="inner detail marker"):
        selector.fit(data=data, max_steps=1)


def test_forward_selection_rejects_kwargs_with_criterion_instance():
    data = make_regression_gram(446, n=80, p=8)
    selector = ForwardSelection(
        criterion=BestRSSCriterion(),
        criterion_kwargs={"abs_tol": 1e-6},
    )
    with pytest.raises(ValueError, match="criterion_kwargs"):
        selector.fit(data=data, max_steps=2)
