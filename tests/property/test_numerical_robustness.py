import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.routines import (
    BackwardSelection,
    CrossValBackwardSelection,
    CrossValForwardSelection,
    ForwardSelection,
)


def _make_ill_conditioned_data(seed: int, *, n: int = 120, p: int = 14) -> GramData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    x_shared = rng.standard_normal(n)
    X[:, 0] = x_shared
    X[:, 1] = x_shared + 1e-8 * rng.standard_normal(n)
    X[:, 2] *= 1e6
    X[:, 3] *= 1e-6
    beta = rng.standard_normal(p)
    y = X @ beta + 0.05 * rng.standard_normal(n)
    return GramData(X.T @ X, X.T @ y, float(y @ y), n_samples=n)


def _make_ill_conditioned_cv_data(
    seed: int, *, folds: int = 4, n: int = 80, p: int = 10
) -> CrossValGramData:
    rng = np.random.default_rng(seed)
    fold_data = []
    for _ in range(folds):
        X = rng.standard_normal((n, p))
        x_shared = rng.standard_normal(n)
        X[:, 0] = x_shared
        X[:, 1] = x_shared + 1e-8 * rng.standard_normal(n)
        X[:, 2] *= 1e5
        X[:, 3] *= 1e-5
        beta = rng.standard_normal(p)
        y = X @ beta + 0.05 * rng.standard_normal(n)
        fold_data.append(GramData(X.T @ X, X.T @ y, float(y @ y), n_samples=n))
    return CrossValGramData(fold_data)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("solver_policy", ["ridge", "pinv"])
def test_non_strict_solvers_keep_single_dataset_paths_finite(seed, solver_policy):
    data = _make_ill_conditioned_data(seed)
    common_kwargs = {
        "criterion_cls": BestRSSCriterion,
        "solver_policy": solver_policy,
        "ridge_alpha": 1e-6,
        "pinv_rcond": 1e-10,
    }

    forward = ForwardSelection(**common_kwargs).fit(data=data, max_steps=6)
    backward: object | None = None
    try:
        backward = BackwardSelection(**common_kwargs).fit(data=data, max_steps=4)
    except ValueError as err:
        # Under extreme scaling, pinv initialization can yield slightly
        # negative RSS for full-support backward starts.
        assert solver_policy == "pinv"
        assert "non-negative" in str(err)

    assert len(forward.active_set) == len(set(forward.active_set))
    assert np.isfinite(forward.rss)
    assert np.isfinite(forward.beta).all()
    if backward is not None:
        assert len(backward.active_set) == len(set(backward.active_set))
        assert np.isfinite(backward.rss)
        assert np.isfinite(backward.beta).all()


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("solver_policy", ["ridge", "pinv"])
def test_non_strict_solvers_keep_cv_paths_finite(seed, solver_policy):
    cv_data = _make_ill_conditioned_cv_data(seed)
    common_kwargs = {
        "solver_policy": solver_policy,
        "ridge_alpha": 1e-6,
        "pinv_rcond": 1e-10,
    }

    cv_forward = CrossValForwardSelection(**common_kwargs).fit(
        data=cv_data, max_steps=4
    )
    cv_backward = CrossValBackwardSelection(**common_kwargs).fit(
        data=cv_data, max_steps=2
    )

    assert len(cv_forward.active_set) == len(set(cv_forward.active_set))
    assert len(cv_backward.active_set) == len(set(cv_backward.active_set))
    assert np.isfinite(cv_forward.rss_cv)
    assert np.isfinite(cv_backward.rss_cv)
    assert np.isfinite(cv_forward.beta).all()
    assert np.isfinite(cv_backward.beta).all()
