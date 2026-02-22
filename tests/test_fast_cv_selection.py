import numpy as np
import pytest

from selection.criteria import AICCriterion, BestRSSCriterion, GCVCriterion
from selection.definitions import CrossValGramData, GramData
from selection.fast_routines import (
    FastCrossValBackwardSelection,
    FastCrossValForwardSelection,
    FastCrossValMixedSelection,
)
from selection.routines import (
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
    CrossValBackwardSelection,
    CrossValForwardSelection,
    CrossValMixedSelection,
)


def make_cv_problem(folds=4, n=120, p=12, support=4, seed=123):
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    beta[:support] = 1.0
    fold_data = []
    for fold_seed in rng.integers(0, 1_000_000, size=folds):
        r = np.random.default_rng(int(fold_seed))
        X = r.standard_normal((n, p))
        y = X @ beta + 0.05 * r.standard_normal(n)
        fold_data.append(GramData(X.T @ X, X.T @ y, y @ y, n))
    return CrossValGramData(fold_data)


def _explicit_cv_rss(cv_data: CrossValGramData, active_set: list[int]) -> float:
    if not active_set:
        return float(np.sum(cv_data.y_norm_folds))
    idx = np.array(active_set, dtype=int)
    total = 0.0
    for fold_idx in range(cv_data.n_folds):
        train = cv_data.train_data_for_fold(fold_idx)
        beta = np.linalg.solve(train.gram[np.ix_(idx, idx)], train.cov[idx])
        gram_val = cv_data.gram_folds[fold_idx]
        cov_val = cv_data.cov_folds[fold_idx]
        y_norm_val = cv_data.y_norm_folds[fold_idx]
        gram_val_ss = gram_val[np.ix_(idx, idx)]
        total += y_norm_val - 2.0 * float(beta @ cov_val[idx]) + float(
            beta @ (gram_val_ss @ beta)
        )
    return float(total)


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
    state = FastCrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    expected_rss = _explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_fast_cv_backward_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = FastCrossValBackwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    expected_rss = _explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_fast_cv_mixed_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = FastCrossValMixedSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_forward_steps=3, max_total_steps=5
    )
    expected_rss = _explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


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
