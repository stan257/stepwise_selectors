import numpy as np
import pytest

from selection.beam_search import Beam
from selection.beam_utils import cv_beam_backward_children
from selection.criteria import BestRSSCriterion
from selection.cv_utils import CVBackwardScores
from selection.definitions import CrossValGramData, GramData
from selection.fast_routines import (
    FastBeamCrossValBackwardSelection,
    FastBeamCrossValForwardSelection,
    FastBeamCrossValMixedSelection,
)
from selection.routines import (
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
)


def make_cv_problem(folds=4, n=120, p=12, support=4, seed=321):
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


def test_fast_cv_beam_forward_matches_standard():
    cv_data = make_cv_problem()
    fast = FastBeamCrossValForwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=4)
    ref = BeamCrossValForwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=4)

    assert set(fast.active_set) == set(ref.active_set)
    assert pytest.approx(fast.rss_cv, rel=1e-8, abs=1e-8) == ref.rss_cv


def test_fast_cv_beam_backward_matches_standard():
    cv_data = make_cv_problem()
    fast = FastBeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    ref = BeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)

    assert set(fast.active_set) == set(ref.active_set)
    assert pytest.approx(fast.rss_cv, rel=1e-8, abs=1e-8) == ref.rss_cv


def test_fast_cv_beam_mixed_matches_standard():
    cv_data = make_cv_problem()
    fast = FastBeamCrossValMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    ref = BeamCrossValMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)

    assert set(fast.active_set) == set(ref.active_set)
    assert pytest.approx(fast.rss_cv, rel=1e-8, abs=1e-8) == ref.rss_cv


def test_cv_beam_backward_children_propagates_unexpected_errors(monkeypatch):
    class _DummyState:
        active_set = [0]

        def clone(self):
            return self

        def apply_backward_step(self, idx, tol):
            raise RuntimeError("unexpected failure")

    criterion = BestRSSCriterion()
    criterion.update_current(10.0)
    beam = Beam(_DummyState(), criterion, 10.0)

    def _fake_backward_scores(cv_state, tol):
        return CVBackwardScores(
            rss_matrix=np.array([[1.0]]),
            aggregated_rss=np.array([1.0]),
        )

    monkeypatch.setattr("selection.beam_utils.cv_backward_scores", _fake_backward_scores)

    with pytest.raises(RuntimeError, match="unexpected failure"):
        cv_beam_backward_children(beam, beam_width=1, tol=1e-10, allow_worse=True)
