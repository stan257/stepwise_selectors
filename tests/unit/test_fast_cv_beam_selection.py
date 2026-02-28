import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.routines_core import (
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
    CVBeam,
    _cv_beam_backward_children,
)
from tests.helpers import explicit_cv_rss, make_cv_problem


def _make_full_model_optimal_cv(*, folds: int = 3, n: int = 50, p: int = 6):
    """Build diagonal CV data where dropping any feature strictly worsens RSS."""
    fold_data = []
    gram = float(n) * np.eye(p)
    cov = float(n) * np.ones(p, dtype=float)
    y_norm = float(n) * (p + 1.0)
    for _ in range(folds):
        fold_data.append(GramData(gram.copy(), cov.copy(), y_norm, n))
    return CrossValGramData(fold_data)


def test_fast_cv_beam_forward_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = BeamCrossValForwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=4)
    expected_rss = explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_fast_cv_beam_backward_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = BeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    expected_rss = explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_fast_cv_beam_mixed_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = BeamCrossValMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    expected_rss = explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_cv_beam_backward_children_propagates_unexpected_errors(monkeypatch):
    class _DummyFastState:
        active_set = [0]

        def clone(self):
            return self

        def apply_backward(self, idx):
            raise RuntimeError("unexpected failure")

    criterion = BestRSSCriterion()
    criterion.update_current(10.0)
    beam = CVBeam([_DummyFastState()], criterion, 10.0)
    cv_data = make_cv_problem(folds=2, n=20, p=4, support=1, seed=777)

    monkeypatch.setattr(
        "selection.routines_core._cv_backward_scores",
        lambda states, data, tol: np.array([1.0]),
    )

    with pytest.raises(RuntimeError, match="unexpected failure"):
        _cv_beam_backward_children(
            beam, beam_width=1, data=cv_data, tol=1e-10, allow_worse=True
        )


def test_fast_cv_beam_backward_is_improvement_only_by_default():
    cv_data = _make_full_model_optimal_cv(folds=3, n=40, p=6)

    strict = BeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=1)
    relaxed = BeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion, allow_worse=True
    ).fit(data=cv_data, max_steps=1)

    assert len(strict.active_set) == cv_data.p
    assert len(relaxed.active_set) == cv_data.p - 1
    assert strict.rss_cv <= relaxed.rss_cv + 1e-12
