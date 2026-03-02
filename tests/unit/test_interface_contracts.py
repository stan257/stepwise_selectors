import numpy as np
import pytest

from selection import GramData
from selection import GroupForwardSelection
import selection as routines
from selection import (
    BackwardSelection,
    BeamBackwardSelection,
    BeamCrossValBackwardSelection,
    BeamCrossValMixedSelection,
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
    pytest.param(
        lambda: BeamCrossValMixedSelection(beam_width=2), "cv", id="cv_beam_mixed"
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
