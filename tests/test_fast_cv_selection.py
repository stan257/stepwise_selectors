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


def test_crossvalgramdata_requires_at_least_two_folds():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 4))
    y = rng.standard_normal(20)
    fold = GramData(X.T @ X, X.T @ y, y @ y, n_samples=20)
    with pytest.raises(
        ValueError, match="requires at least two folds for cross-validation"
    ):
        CrossValGramData([fold])


def test_fast_cv_forward_matches_standard():
    cv_data = make_cv_problem()
    fast = FastCrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    ref = CrossValForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    assert fast.active_set == ref.active_set
    assert pytest.approx(fast.rss_cv, rel=1e-8, abs=1e-8) == ref.rss_cv


def test_fast_cv_backward_matches_standard():
    cv_data = make_cv_problem()
    fast = FastCrossValBackwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    ref = CrossValBackwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_steps=4
    )
    assert set(fast.active_set) == set(ref.active_set)
    assert pytest.approx(fast.rss_cv, rel=1e-8, abs=1e-8) == ref.rss_cv


def test_fast_cv_mixed_matches_standard():
    cv_data = make_cv_problem()
    fast = FastCrossValMixedSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_forward_steps=3, max_total_steps=5
    )
    ref = CrossValMixedSelection(criterion_cls=BestRSSCriterion).fit(
        data=cv_data, max_forward_steps=3, max_total_steps=5
    )
    assert fast.active_set == ref.active_set
    assert pytest.approx(fast.rss_cv, rel=1e-8, abs=1e-8) == ref.rss_cv


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
