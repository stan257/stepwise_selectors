import numpy as np
import pytest

from selection.criteria import (
    AICCriterion,
    AICcCriterion,
    BICCriterion,
    BestRSSCriterion,
    EBICCriterion,
    GCVCriterion,
    HQICCriterion,
)
from selection.definitions import GramData
from selection.state import SelectionState


def make_state():
    G = np.eye(1)
    c = np.array([0.5])
    data = GramData(G, c, y_norm=1.0, n_samples=3)
    return SelectionState(data)


def test_aic_value_matches_formula():
    state = make_state()
    crit = AICCriterion(n_samples=state.data.n_samples)
    rss = 0.6
    value = crit.evaluate(rss, k=1)
    n = state.data.n_samples
    expected = n * np.log(rss / n) + 2
    assert pytest.approx(value) == expected


def test_is_improvement_uses_tolerance():
    state = make_state()
    crit = AICCriterion(n_samples=state.data.n_samples, abs_tol=1e-8)
    incumbent = 10.0
    assert crit.is_improvement(9.0, incumbent)
    assert not crit.is_improvement(incumbent - crit.abs_tol / 2, incumbent)


def test_tracking_current_value():
    state = make_state()
    crit = AICCriterion(n_samples=state.data.n_samples)
    initial = float(crit.evaluate(state.rss, k=0))
    crit.update_current(initial)
    assert crit.current_value == initial


def test_best_rss_criterion_prefers_lower_rss():
    state = make_state()
    crit = BestRSSCriterion()
    better = crit.evaluate(0.5, k=1)
    worse = crit.evaluate(0.6, k=1)
    assert crit.is_improvement(better, worse)
    assert not crit.is_improvement(worse, better)


def test_best_candidate_matches_argmin_for_aic():
    crit = AICCriterion(n_samples=50)
    rss_vals = np.array([0.2, 0.15, 0.5, 0.3])
    idx, score = crit.best_candidate(rss_vals, k=2)
    scores = crit.evaluate(rss_vals, k=2)
    assert idx == int(np.argmin(scores))
    assert pytest.approx(score) == scores[idx]


def test_best_candidate_matches_argmin_for_rss():
    crit = BestRSSCriterion()
    rss_vals = np.array([0.4, 0.1, 0.3])
    idx, score = crit.best_candidate(rss_vals, k=1)
    assert idx == 1
    assert pytest.approx(score) == 0.1


def test_best_rss_criterion_rejects_negative_rss():
    crit = BestRSSCriterion()
    with pytest.raises(ValueError, match="non-negative"):
        crit.evaluate(np.array([0.1, -0.1]), k=1)


@pytest.mark.parametrize(
    ("criterion_cls", "kwargs"),
    [
        (AICCriterion, {}),
        (BICCriterion, {}),
        (AICcCriterion, {}),
        (HQICCriterion, {}),
        (EBICCriterion, {"p": 5}),
        (GCVCriterion, {}),
    ],
)
def test_criteria_require_positive_n_samples(criterion_cls, kwargs):
    with pytest.raises(ValueError, match="n_samples must be positive"):
        criterion_cls(n_samples=0, **kwargs)


@pytest.mark.parametrize(
    ("criterion_cls", "kwargs"),
    [
        (AICCriterion, {}),
        (BICCriterion, {}),
        (AICcCriterion, {}),
        (HQICCriterion, {}),
        (EBICCriterion, {"p": 5}),
        (GCVCriterion, {}),
    ],
)
def test_criteria_reject_boolean_n_samples(criterion_cls, kwargs):
    with pytest.raises(TypeError, match="n_samples must be an integer"):
        criterion_cls(n_samples=True, **kwargs)


def test_ebic_rejects_boolean_p():
    with pytest.raises(TypeError, match="p must be an integer"):
        EBICCriterion(n_samples=10, p=True)


@pytest.mark.parametrize("tol_name", ["abs_tol", "rel_tol"])
def test_criteria_reject_negative_tolerances(tol_name):
    with pytest.raises(ValueError, match=rf"{tol_name} must be >= 0"):
        AICCriterion(n_samples=10, **{tol_name: -1e-9})


@pytest.mark.parametrize("tol_name", ["abs_tol", "rel_tol"])
def test_criteria_reject_non_finite_tolerances(tol_name):
    with pytest.raises(ValueError, match=rf"{tol_name} must be finite"):
        AICCriterion(n_samples=10, **{tol_name: np.inf})
