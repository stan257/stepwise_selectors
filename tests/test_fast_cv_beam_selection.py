import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import CrossValGramData, GramData
from selection.fast_routines import (
    FastBeamCrossValBackwardSelection,
    FastBeamCrossValForwardSelection,
    FastBeamCrossValMixedSelection,
    FastCVBeam,
    _fast_cv_beam_backward_children,
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


def test_fast_cv_beam_forward_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = FastBeamCrossValForwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=4)
    expected_rss = _explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_fast_cv_beam_backward_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = FastBeamCrossValBackwardSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_steps=3)
    expected_rss = _explicit_cv_rss(cv_data, state.active_set)
    assert pytest.approx(state.rss_cv, rel=1e-8, abs=1e-8) == expected_rss


def test_fast_cv_beam_mixed_matches_explicit_rss():
    cv_data = make_cv_problem()
    state = FastBeamCrossValMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=cv_data, max_forward_steps=3, max_total_steps=5)
    expected_rss = _explicit_cv_rss(cv_data, state.active_set)
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
    beam = FastCVBeam([_DummyFastState()], criterion, 10.0)
    cv_data = make_cv_problem(folds=2, n=20, p=4, support=1, seed=777)

    monkeypatch.setattr(
        "selection.fast_routines._fast_cv_backward_scores",
        lambda states, data, tol: np.array([1.0]),
    )

    with pytest.raises(RuntimeError, match="unexpected failure"):
        _fast_cv_beam_backward_children(
            beam, beam_width=1, data=cv_data, tol=1e-10, allow_worse=True
        )
