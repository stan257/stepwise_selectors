import numpy as np
import pytest

from selection.criteria import AICCriterion, BestRSSCriterion
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
