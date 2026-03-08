import numpy as np
import pytest

from selection import GramData
from selection import GroupForwardSelection
import selection as routines
from selection import (
    BackwardSelection,
    BeamCrossValForwardSelection,
    BeamBackwardSelection,
    BeamCrossValBackwardSelection,
    BeamCrossValMixedSelection,
    BeamForwardSelection,
    BeamMixedSelection,
    CrossValMixedSelection,
    CrossValForwardSelection,
    ForwardSelection,
    MixedSelection,
)
from tests.helpers import make_cv_problem, make_regression_gram


def test_routines_all_symbols_are_resolvable():
    for name in routines.__all__:
        assert hasattr(routines, name), f"selection missing {name}"


NON_CV_STEP_SELECTORS = [
    ForwardSelection(),
    BeamBackwardSelection(beam_width=2, allow_worse=True),
]

CV_STEP_SELECTORS = [
    CrossValForwardSelection(),
    BeamCrossValBackwardSelection(beam_width=2, allow_worse=True),
]

MIXED_SELECTORS = [
    pytest.param(lambda: MixedSelection(), "non_cv", id="mixed"),
    pytest.param(lambda: BeamMixedSelection(beam_width=2), "non_cv", id="beam_mixed"),
    pytest.param(lambda: CrossValMixedSelection(), "cv", id="cv_mixed"),
    pytest.param(
        lambda: BeamCrossValMixedSelection(beam_width=2), "cv", id="cv_beam_mixed"
    ),
]

FORWARD_REQUIRED_BUDGET_CASES = [
    pytest.param(
        lambda: ForwardSelection(),
        lambda: make_regression_gram(210, n=80, p=8),
        {},
        r"ForwardSelection defaults to budget-driven search; max_steps is required",
        id="forward",
    ),
    pytest.param(
        lambda: BeamForwardSelection(beam_width=2),
        lambda: make_regression_gram(211, n=80, p=8),
        {},
        r"BeamForwardSelection defaults to budget-driven search; max_steps is required",
        id="beam_forward",
    ),
    pytest.param(
        lambda: CrossValForwardSelection(),
        lambda: make_cv_problem(seed=212, folds=3, n=80, p=8),
        {},
        r"CrossValForwardSelection defaults to budget-driven search; max_steps is required",
        id="cv_forward",
    ),
    pytest.param(
        lambda: BeamCrossValForwardSelection(beam_width=2),
        lambda: make_cv_problem(seed=213, folds=3, n=80, p=8),
        {},
        r"BeamCrossValForwardSelection defaults to budget-driven search; max_steps is required",
        id="cv_beam_forward",
    ),
    pytest.param(
        lambda: MixedSelection(),
        lambda: make_regression_gram(214, n=80, p=8),
        {},
        r"MixedSelection defaults to budget-driven search; max_forward_steps is required",
        id="mixed",
    ),
    pytest.param(
        lambda: BeamMixedSelection(beam_width=2),
        lambda: make_regression_gram(215, n=80, p=8),
        {},
        r"BeamMixedSelection defaults to budget-driven search; max_forward_steps is required",
        id="beam_mixed",
    ),
    pytest.param(
        lambda: CrossValMixedSelection(),
        lambda: make_cv_problem(seed=216, folds=3, n=80, p=8),
        {},
        r"CrossValMixedSelection defaults to budget-driven search; max_forward_steps is required",
        id="cv_mixed",
    ),
    pytest.param(
        lambda: BeamCrossValMixedSelection(beam_width=2),
        lambda: make_cv_problem(seed=217, folds=3, n=80, p=8),
        {},
        r"BeamCrossValMixedSelection defaults to budget-driven search; max_forward_steps is required",
        id="cv_beam_mixed",
    ),
    pytest.param(
        lambda: GroupForwardSelection([[0], [1], [2], [3]]),
        lambda: make_regression_gram(218, n=80, p=4),
        {},
        r"GroupForwardSelection defaults to budget-driven search; max_steps is required",
        id="group_forward",
    ),
]

LEGACY_FORWARD_NO_BUDGET_CASES = [
    pytest.param(
        lambda: ForwardSelection(stop_on_no_improvement=True),
        lambda: make_regression_gram(310, n=80, p=8),
        {},
        id="forward",
    ),
    pytest.param(
        lambda: BeamForwardSelection(beam_width=2, stop_on_no_improvement=True),
        lambda: make_regression_gram(311, n=80, p=8),
        {},
        id="beam_forward",
    ),
    pytest.param(
        lambda: CrossValForwardSelection(stop_on_no_improvement=True),
        lambda: make_cv_problem(seed=312, folds=3, n=80, p=8),
        {},
        id="cv_forward",
    ),
    pytest.param(
        lambda: BeamCrossValForwardSelection(
            beam_width=2, stop_on_no_improvement=True
        ),
        lambda: make_cv_problem(seed=313, folds=3, n=80, p=8),
        {},
        id="cv_beam_forward",
    ),
    pytest.param(
        lambda: MixedSelection(stop_on_no_improvement=True),
        lambda: make_regression_gram(314, n=80, p=8),
        {"max_total_steps": 6},
        id="mixed",
    ),
    pytest.param(
        lambda: BeamMixedSelection(beam_width=2, stop_on_no_improvement=True),
        lambda: make_regression_gram(315, n=80, p=8),
        {"max_total_steps": 6},
        id="beam_mixed",
    ),
    pytest.param(
        lambda: CrossValMixedSelection(stop_on_no_improvement=True),
        lambda: make_cv_problem(seed=316, folds=3, n=80, p=8),
        {"max_total_steps": 6},
        id="cv_mixed",
    ),
    pytest.param(
        lambda: BeamCrossValMixedSelection(
            beam_width=2, stop_on_no_improvement=True
        ),
        lambda: make_cv_problem(seed=317, folds=3, n=80, p=8),
        {"max_total_steps": 6},
        id="cv_beam_mixed",
    ),
    pytest.param(
        lambda: GroupForwardSelection(
            [[0], [1], [2], [3]], stop_on_no_improvement=True
        ),
        lambda: make_regression_gram(318, n=80, p=4),
        {},
        id="group_forward",
    ),
]


def test_non_cv_step_selectors_reject_invalid_max_steps():
    data = make_regression_gram(101, n=80, p=8)
    for selector in NON_CV_STEP_SELECTORS:
        with pytest.raises(TypeError, match="max_steps must be an integer or None"):
            selector.fit(data=data, max_steps=1.5)
        with pytest.raises(ValueError, match="max_steps must be >= 0"):
            selector.fit(data=data, max_steps=-1)
        with pytest.raises(TypeError, match="max_steps must be an integer or None"):
            selector.fit(data=data, max_steps=True)


def test_cv_step_selectors_reject_invalid_max_steps():
    data = make_cv_problem(seed=102, folds=3, n=80, p=8)
    for selector in CV_STEP_SELECTORS:
        with pytest.raises(TypeError, match="max_steps must be an integer or None"):
            selector.fit(data=data, max_steps=1.5)
        with pytest.raises(ValueError, match="max_steps must be >= 0"):
            selector.fit(data=data, max_steps=-1)
        with pytest.raises(TypeError, match="max_steps must be an integer or None"):
            selector.fit(data=data, max_steps=True)


@pytest.mark.parametrize(
    "selector_factory,data_factory,fit_kwargs,match",
    FORWARD_REQUIRED_BUDGET_CASES,
)
def test_forward_like_selectors_require_budget_by_default(
    selector_factory, data_factory, fit_kwargs, match
):
    selector = selector_factory()
    data = data_factory()
    with pytest.raises(ValueError, match=match):
        selector.fit(data=data, **fit_kwargs)


@pytest.mark.parametrize(
    "selector_factory,data_factory,fit_kwargs",
    LEGACY_FORWARD_NO_BUDGET_CASES,
)
def test_forward_like_selectors_allow_legacy_no_budget(
    selector_factory, data_factory, fit_kwargs
):
    selector = selector_factory()
    data = data_factory()
    state = selector.fit(data=data, **fit_kwargs)
    assert state is not None


@pytest.mark.parametrize("selector_factory,kind", MIXED_SELECTORS)
def test_mixed_selectors_reject_invalid_budgets(selector_factory, kind):
    data = (
        make_regression_gram(103, n=80, p=8)
        if kind == "non_cv"
        else make_cv_problem(seed=103, folds=3, n=80, p=8)
    )
    selector = selector_factory()
    with pytest.raises(
        TypeError, match="max_forward_steps must be an integer or None"
    ):
        selector.fit(data=data, max_forward_steps=2.5, max_total_steps=5)
    with pytest.raises(ValueError, match="max_total_steps must be >= 0"):
        selector.fit(data=data, max_forward_steps=2, max_total_steps=-1)


@pytest.mark.parametrize(
    "selector_cls,kwargs",
    [
        pytest.param(BeamBackwardSelection, {"allow_worse": True}, id="beam_backward"),
        pytest.param(
            BeamCrossValBackwardSelection, {"allow_worse": True}, id="cv_beam_backward"
        ),
    ],
)
def test_beam_selectors_reject_invalid_beam_width(selector_cls, kwargs):
    with pytest.raises(TypeError, match="beam_width must be an integer"):
        selector_cls(beam_width=1.2, **kwargs)
    with pytest.raises(TypeError, match="beam_width must be an integer"):
        selector_cls(beam_width=True, **kwargs)
    with pytest.raises(ValueError, match="beam_width must be > 0"):
        selector_cls(beam_width=0, **kwargs)


@pytest.mark.parametrize(
    "selector_cls,kwargs",
    [
        pytest.param(BackwardSelection, {}, id="backward"),
        pytest.param(
            BeamCrossValBackwardSelection, {"beam_width": 2}, id="cv_beam_backward"
        ),
    ],
)
def test_backward_selectors_reject_non_bool_allow_worse(selector_cls, kwargs):
    with pytest.raises(TypeError, match="allow_worse must be a bool"):
        selector_cls(allow_worse=1, **kwargs)
    with pytest.raises(TypeError, match="allow_worse must be a bool"):
        selector_cls(allow_worse="yes", **kwargs)


@pytest.mark.parametrize(
    "selector_factory",
    [
        pytest.param(lambda: ForwardSelection(tol=0.0), id="forward_zero"),
        pytest.param(lambda: CrossValForwardSelection(tol=np.inf), id="cv_forward_inf"),
        pytest.param(
            lambda: GroupForwardSelection([[0], [1]], tol="1e-10"), id="group_bad_type"
        ),
    ],
)
def test_selectors_reject_invalid_tol(selector_factory):
    with pytest.raises((TypeError, ValueError), match="tol must"):
        selector_factory()


@pytest.mark.parametrize(
    "selector_factory",
    [
        pytest.param(lambda: ForwardSelection(stop_on_no_improvement=1), id="forward"),
        pytest.param(
            lambda: BeamForwardSelection(beam_width=2, stop_on_no_improvement="yes"),
            id="beam_forward",
        ),
        pytest.param(
            lambda: CrossValForwardSelection(stop_on_no_improvement=1),
            id="cv_forward",
        ),
        pytest.param(
            lambda: BeamCrossValForwardSelection(
                beam_width=2, stop_on_no_improvement="yes"
            ),
            id="cv_beam_forward",
        ),
        pytest.param(lambda: MixedSelection(stop_on_no_improvement=1), id="mixed"),
        pytest.param(
            lambda: BeamMixedSelection(beam_width=2, stop_on_no_improvement="yes"),
            id="beam_mixed",
        ),
        pytest.param(
            lambda: CrossValMixedSelection(stop_on_no_improvement=1), id="cv_mixed"
        ),
        pytest.param(
            lambda: BeamCrossValMixedSelection(
                beam_width=2, stop_on_no_improvement="yes"
            ),
            id="cv_beam_mixed",
        ),
        pytest.param(
            lambda: GroupForwardSelection([[0], [1]], stop_on_no_improvement="yes"),
            id="group_forward",
        ),
    ],
)
def test_forward_like_selectors_reject_non_bool_stop_on_no_improvement(
    selector_factory,
):
    with pytest.raises(TypeError, match="stop_on_no_improvement must be a bool"):
        selector_factory()


def test_gramdata_accepts_array_like_inputs():
    data = GramData(
        gram=[[2.0, 0.5], [0.5, 1.0]],
        cov=[1.0, -1.0],
        y_norm=3.0,
        n_samples=10,
    )

    assert isinstance(data.gram, np.ndarray)
    assert isinstance(data.cov, np.ndarray)
    assert data.gram.flags.c_contiguous
    assert data.cov.flags.c_contiguous


def test_gramdata_rejects_nonnumeric_array_like_inputs():
    with pytest.raises(TypeError, match="Gram matrix must contain numeric values"):
        GramData(
            gram=[["a", "b"], ["c", "d"]],
            cov=[1.0, 2.0],
            y_norm=5.0,
            n_samples=10,
        )


def test_gramdata_rejects_boolean_n_samples():
    with pytest.raises(TypeError, match="n_samples must be an integer"):
        GramData(
            gram=np.eye(2),
            cov=np.array([1.0, 2.0]),
            y_norm=5.0,
            n_samples=True,
        )
