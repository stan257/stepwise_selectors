import numpy as np
import pytest

from selection.criteria import AICCriterion, BestRSSCriterion, GCVCriterion
from selection import CrossValGramData, GramData
from selection import (
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
    CrossValBackwardSelection,
    CrossValForwardSelection,
    CrossValMixedSelection,
)
from selection import CrossValSelectionState
from tests.helpers import explicit_beta_from_active, explicit_cv_rss, make_cv_problem


class IncompatibleCriterion(BestRSSCriterion):
    cv_compatible = False


def incompatible_criterion_factory(*, n_samples: int, p: int) -> IncompatibleCriterion:
    _ = (n_samples, p)
    return IncompatibleCriterion()


def make_cv_non_improving_forward_problem() -> CrossValGramData:
    x1 = np.array(
        [
            [0.12573, -0.132105],
            [0.640423, 0.1049],
            [-0.535669, 0.361595],
            [1.304, 0.947081],
            [-0.703735, -1.265421],
            [-0.623274, 0.041326],
            [-2.325031, -0.218792],
            [-1.245911, -0.732267],
        ]
    )
    x2 = np.array(
        [
            [-0.544259, -0.3163],
            [0.411631, 1.042513],
            [-0.128535, 1.366463],
            [-0.665195, 0.35151],
            [0.90347, 0.094012],
            [-0.743499, -0.921725],
            [-0.457726, 0.220195],
            [-1.009618, -0.209176],
        ]
    )
    y1 = np.array(
        [0.093885, 0.748592, -0.492738, 1.375075, -0.834501, -0.649197, -2.168236, -0.947225]
    )
    y2 = np.array(
        [-1.428672, 2.799442, 2.873567, 0.194088, 1.144386, -2.649735, 0.274269, -1.035918]
    )
    folds = [
        GramData(x1.T @ x1, x1.T @ y1, float(y1 @ y1), n_samples=len(y1), warn_if_uncentered=False),
        GramData(x2.T @ x2, x2.T @ y2, float(y2 @ y2), n_samples=len(y2), warn_if_uncentered=False),
    ]
    return CrossValGramData(folds)


def test_crossvalgramdata_requires_at_least_two_folds():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 4))
    y = rng.standard_normal(20)
    fold = GramData(X.T @ X, X.T @ y, y @ y, n_samples=20)
    with pytest.raises(
        ValueError, match="requires at least two folds"
    ):
        CrossValGramData([fold])


def test_cv_forward_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = CrossValForwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    expected_rss = explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_cv_backward_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = CrossValBackwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    expected_rss = explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_cv_mixed_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = CrossValMixedSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_forward_steps=3, max_total_steps=5
    )
    expected_rss = explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_cv_state_exposes_full_data_postselection_beta():
    cv_data = make_cv_problem(seed=2026, folds=4, n=100, p=10, support=4)
    state = CrossValForwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    full_data = cv_data.make_full_data()
    expected_beta = explicit_beta_from_active(full_data, state.active_set)
    np.testing.assert_allclose(state.beta, expected_beta, atol=1e-8, rtol=1e-8)

    empty = CrossValForwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_steps=0
    )
    np.testing.assert_allclose(empty.beta, np.zeros_like(empty.beta), atol=0.0, rtol=0.0)


def test_cv_forward_budget_mode_continues_through_non_improving_step():
    cv_data = make_cv_non_improving_forward_problem()

    budget_state = CrossValForwardSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_steps=2
    )
    legacy_state = CrossValForwardSelection(
        criterion=BestRSSCriterion, stop_on_no_improvement=True
    ).fit(data=cv_data)

    assert len(budget_state.active_set) == 2
    assert len(legacy_state.active_set) == 1
    assert budget_state.rss_cv > legacy_state.rss_cv


def test_cv_beam_forward_budget_mode_continues_through_non_improving_step():
    cv_data = make_cv_non_improving_forward_problem()

    budget_state = BeamCrossValForwardSelection(
        beam_width=2, criterion=BestRSSCriterion
    ).fit(data=cv_data, max_steps=2)
    legacy_state = BeamCrossValForwardSelection(
        beam_width=2,
        criterion=BestRSSCriterion,
        stop_on_no_improvement=True,
    ).fit(data=cv_data)

    assert len(budget_state.active_set) == 2
    assert len(legacy_state.active_set) == 1
    assert budget_state.rss_cv > legacy_state.rss_cv


def test_cv_mixed_budget_mode_continues_through_non_improving_forward_step():
    cv_data = make_cv_non_improving_forward_problem()

    budget_state = CrossValMixedSelection(criterion=BestRSSCriterion).fit(
        data=cv_data, max_forward_steps=2, max_total_steps=2
    )
    legacy_state = CrossValMixedSelection(
        criterion=BestRSSCriterion, stop_on_no_improvement=True
    ).fit(data=cv_data, max_total_steps=2)

    assert len(budget_state.active_set) == 2
    assert len(legacy_state.active_set) == 1
    assert budget_state.rss_cv > legacy_state.rss_cv


@pytest.mark.parametrize(
    "selector_cls,selector_kwargs",
    [
        pytest.param(
            CrossValMixedSelection,
            {"criterion": BestRSSCriterion},
            id="cv_mixed",
        ),
        pytest.param(
            BeamCrossValMixedSelection,
            {"criterion": BestRSSCriterion, "beam_width": 2},
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


CV_SELECTOR_STATE_CONTRACT_CASES = [
    pytest.param(
        CrossValForwardSelection,
        {"criterion": BestRSSCriterion},
        {"max_steps": 3},
        id="cv_forward",
    ),
    pytest.param(
        BeamCrossValForwardSelection,
        {"criterion": BestRSSCriterion, "beam_width": 2},
        {"max_steps": 3},
        id="cv_beam_forward",
    ),
]


CV_SELECTOR_SOLVER_COMPAT_CASES = [
    pytest.param(
        CrossValBackwardSelection,
        {"criterion": BestRSSCriterion},
        {"max_steps": 0},
        id="cv_backward",
    ),
    pytest.param(
        BeamCrossValBackwardSelection,
        {"criterion": BestRSSCriterion, "beam_width": 2},
        {"max_steps": 0},
        id="cv_beam_backward",
    ),
]


@pytest.mark.parametrize(
    "selector_cls,selector_kwargs,fit_kwargs", CV_SELECTOR_STATE_CONTRACT_CASES
)
def test_cv_selector_rejects_mismatched_state_data(
    selector_cls, selector_kwargs, fit_kwargs
):
    cv_data = make_cv_problem(seed=111)
    other_data = make_cv_problem(seed=222)
    state = CrossValSelectionState(other_data)

    selector = selector_cls(**selector_kwargs)
    with pytest.raises(ValueError, match="state.data"):
        selector.fit(state=state, data=cv_data, **fit_kwargs)


@pytest.mark.parametrize(
    "selector_cls,selector_kwargs,fit_kwargs", CV_SELECTOR_STATE_CONTRACT_CASES
)
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
    "selector_cls,selector_kwargs,fit_kwargs", CV_SELECTOR_SOLVER_COMPAT_CASES
)
def test_cv_selector_rejects_mismatched_state_solver_policy(
    selector_cls, selector_kwargs, fit_kwargs
):
    cv_data = make_cv_problem(seed=334)
    state = CrossValSelectionState(cv_data, solver_policy="strict")

    selector = selector_cls(
        **selector_kwargs,
        solver_policy="ridge",
        ridge_alpha=1e-6,
    )
    with pytest.raises(ValueError, match=r"state\.solver_policy"):
        selector.fit(state=state, data=cv_data, **fit_kwargs)


@pytest.mark.parametrize(
    "selector_cls,selector_kwargs,fit_kwargs", CV_SELECTOR_SOLVER_COMPAT_CASES
)
def test_cv_selector_rejects_mismatched_state_ridge_alpha(
    selector_cls, selector_kwargs, fit_kwargs
):
    cv_data = make_cv_problem(seed=335)
    state = CrossValSelectionState(
        cv_data,
        solver_policy="ridge",
        ridge_alpha=1e-3,
    )

    selector = selector_cls(
        **selector_kwargs,
        solver_policy="ridge",
        ridge_alpha=1e-6,
    )
    with pytest.raises(ValueError, match=r"state\.ridge_alpha"):
        selector.fit(state=state, data=cv_data, **fit_kwargs)


@pytest.mark.parametrize(
    "selector_cls,selector_kwargs,fit_kwargs", CV_SELECTOR_SOLVER_COMPAT_CASES
)
def test_cv_selector_rejects_mismatched_state_pinv_rcond(
    selector_cls, selector_kwargs, fit_kwargs
):
    cv_data = make_cv_problem(seed=336)
    state = CrossValSelectionState(
        cv_data,
        solver_policy="pinv",
        pinv_rcond=1e-6,
    )

    selector = selector_cls(
        **selector_kwargs,
        solver_policy="pinv",
        pinv_rcond=1e-10,
    )
    with pytest.raises(ValueError, match=r"state\.pinv_rcond"):
        selector.fit(state=state, data=cv_data, **fit_kwargs)


CV_SELECTOR_VALIDATION_CASES = [
    CrossValForwardSelection,
    BeamCrossValForwardSelection,
]


@pytest.mark.parametrize(
    "selector_cls",
    CV_SELECTOR_VALIDATION_CASES,
)
@pytest.mark.parametrize(
    "criterion_spec",
    [AICCriterion, GCVCriterion, "aic", "gcv"],
)
def test_cv_selectors_reject_disallowed_criteria(selector_cls, criterion_spec):
    with pytest.raises(
        ValueError,
        match=r"is not supported for CV selection routines",
    ):
        selector_cls(criterion=criterion_spec)


@pytest.mark.parametrize(
    "selector_cls",
    CV_SELECTOR_VALIDATION_CASES,
)
def test_cv_selectors_reject_cv_incompatible_custom_criterion(selector_cls):
    with pytest.raises(
        ValueError,
        match=r"IncompatibleCriterion is not supported for CV selection routines",
    ):
        selector_cls(criterion=IncompatibleCriterion)


@pytest.mark.parametrize("selector_cls", CV_SELECTOR_VALIDATION_CASES)
def test_cv_selectors_accept_named_best_rss_key(selector_cls):
    cv_data = make_cv_problem(seed=448, folds=3, n=60, p=6, support=2)
    state = selector_cls(criterion="rss").fit(data=cv_data, max_steps=2)
    assert isinstance(state, CrossValSelectionState)


@pytest.mark.parametrize("selector_cls", CV_SELECTOR_VALIDATION_CASES)
def test_cv_selectors_reject_legacy_criterion_cls_kwarg(selector_cls):
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'criterion_cls'"):
        selector_cls(criterion_cls=BestRSSCriterion)


@pytest.mark.parametrize(
    "selector_cls,fit_kwargs",
    [
        pytest.param(CrossValForwardSelection, {"max_steps": 2}, id="cv_forward"),
        pytest.param(
            BeamCrossValMixedSelection,
            {"max_forward_steps": 2, "max_total_steps": 3},
            id="cv_beam_mixed",
        ),
    ],
)
def test_cv_selectors_reject_cv_incompatible_factory_on_fit(selector_cls, fit_kwargs):
    cv_data = make_cv_problem(seed=447, folds=3, n=60, p=6, support=2)
    selector = selector_cls(criterion=incompatible_criterion_factory)
    with pytest.raises(
        ValueError,
        match=r"IncompatibleCriterion is not supported for CV selection routines",
    ):
        selector.fit(data=cv_data, **fit_kwargs)


CV_AGGREGATION_SELECTOR_CASES = [
    pytest.param(CrossValForwardSelection, {}, {"max_steps": 2}, id="cv_forward"),
    pytest.param(CrossValBackwardSelection, {}, {"max_steps": 2}, id="cv_backward"),
    pytest.param(
        CrossValMixedSelection,
        {},
        {"max_forward_steps": 2, "max_total_steps": 3},
        id="cv_mixed",
    ),
    pytest.param(
        BeamCrossValForwardSelection,
        {"beam_width": 2},
        {"max_steps": 2},
        id="cv_beam_forward",
    ),
    pytest.param(
        BeamCrossValBackwardSelection,
        {"beam_width": 2},
        {"max_steps": 2},
        id="cv_beam_backward",
    ),
    pytest.param(
        BeamCrossValMixedSelection,
        {"beam_width": 2},
        {"max_forward_steps": 2, "max_total_steps": 3},
        id="cv_beam_mixed",
    ),
]


@pytest.mark.parametrize("selector_cls,selector_kwargs,fit_kwargs", CV_AGGREGATION_SELECTOR_CASES)
def test_cv_selectors_accept_nondefault_cv_aggregation(
    selector_cls, selector_kwargs, fit_kwargs
):
    cv_data = make_cv_problem(seed=551, folds=4, n=80, p=8, support=3)
    selector = selector_cls(
        **selector_kwargs,
        criterion=BestRSSCriterion,
        cv_aggregation="mean_mse",
    )
    state = selector.fit(data=cv_data, **fit_kwargs)
    assert isinstance(state, CrossValSelectionState)


@pytest.mark.parametrize("selector_cls,selector_kwargs,_", CV_AGGREGATION_SELECTOR_CASES)
def test_cv_selectors_reject_invalid_cv_aggregation_value(
    selector_cls, selector_kwargs, _
):
    with pytest.raises(
        ValueError,
        match=r"cv_aggregation must be one of: mean_mse, median_mse, sum_rss",
    ):
        selector_cls(**selector_kwargs, cv_aggregation="weighted")


@pytest.mark.parametrize("selector_cls,selector_kwargs,_", CV_AGGREGATION_SELECTOR_CASES)
def test_cv_selectors_reject_non_string_cv_aggregation(selector_cls, selector_kwargs, _):
    with pytest.raises(TypeError, match=r"cv_aggregation must be a string"):
        selector_cls(**selector_kwargs, cv_aggregation=1)
