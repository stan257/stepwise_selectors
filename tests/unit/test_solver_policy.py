import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection import CrossValGramData, GramData
from selection import BackwardSelection, CrossValBackwardSelection, ForwardSelection
from selection import CrossValSelectionState, SelectionState


def _make_rank_deficient_data() -> GramData:
    X = np.array(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
            [3.0, 3.0, -1.0],
            [4.0, 4.0, 0.5],
        ]
    )
    y = np.array([1.0, 2.0, 3.0, 4.0])
    return GramData(X.T @ X, X.T @ y, float(y @ y), n_samples=X.shape[0])


def _make_rank_deficient_cv_data() -> CrossValGramData:
    base = _make_rank_deficient_data()
    return CrossValGramData([base, base])


def test_state_strict_solver_rejects_singular_full_active_set():
    data = _make_rank_deficient_data()
    state = SelectionState(data, solver_policy="strict")
    with pytest.raises(np.linalg.LinAlgError):
        state.init_full()


@pytest.mark.parametrize("solver_policy", ["ridge", "pinv"])
def test_state_non_strict_solver_handles_singular_full_active_set(solver_policy):
    data = _make_rank_deficient_data()
    state = SelectionState(
        data, solver_policy=solver_policy, ridge_alpha=1e-6, pinv_rcond=1e-10
    )
    state.init_full()
    assert np.isfinite(state.beta).all()
    assert np.isfinite(state.rss)


def test_backward_selection_strict_rejects_singular_full_active_set():
    data = _make_rank_deficient_data()
    selector = BackwardSelection(criterion=BestRSSCriterion)
    with pytest.raises(np.linalg.LinAlgError):
        selector.fit(data=data, max_steps=1)


@pytest.mark.parametrize("solver_policy", ["ridge", "pinv"])
def test_backward_selection_non_strict_handles_singular_full_active_set(solver_policy):
    data = _make_rank_deficient_data()
    selector = BackwardSelection(
        criterion=BestRSSCriterion,
        solver_policy=solver_policy,
        ridge_alpha=1e-6,
        pinv_rcond=1e-10,
    )
    result = selector.fit(data=data, max_steps=1)
    assert np.isfinite(result.beta).all()
    assert np.isfinite(result.rss)


@pytest.mark.parametrize("solver_policy", ["ridge", "pinv"])
def test_cv_backward_selection_non_strict_handles_singular_full_active_set(
    solver_policy,
):
    data = _make_rank_deficient_cv_data()
    selector = CrossValBackwardSelection(
        solver_policy=solver_policy,
        ridge_alpha=1e-6,
        pinv_rcond=1e-10,
    )
    result = selector.fit(data=data, max_steps=1)
    assert np.isfinite(result.beta).all()
    assert np.isfinite(result.rss_cv)


def test_cv_backward_selection_rejects_mismatched_state_solver_policy():
    data = _make_rank_deficient_cv_data()
    selector = CrossValBackwardSelection(
        solver_policy="ridge",
        ridge_alpha=1e-6,
        pinv_rcond=1e-10,
    )
    strict_state = CrossValSelectionState(data, solver_policy="strict")
    with pytest.raises(ValueError, match=r"state\.solver_policy"):
        selector.fit(state=strict_state, data=data, max_steps=0)


@pytest.mark.parametrize(
    "kwargs,error_type,match",
    [
        ({"solver_policy": "unknown"}, ValueError, "solver_policy must be one of"),
        ({"ridge_alpha": -1.0}, ValueError, "ridge_alpha must be >= 0"),
        ({"pinv_rcond": 0.0}, ValueError, "pinv_rcond must be > 0"),
    ],
)
def test_selector_rejects_invalid_solver_hyperparameters(kwargs, error_type, match):
    with pytest.raises(error_type, match=match):
        ForwardSelection(**kwargs)
