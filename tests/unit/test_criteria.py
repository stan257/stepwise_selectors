import math

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
from selection.state_single import SelectionState


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


def test_bic_value_matches_formula():
    crit = BICCriterion(n_samples=80)
    rss = 12.5
    k = 4
    value = float(np.asarray(crit.evaluate(rss, k=k)))
    expected = 80 * np.log(rss / 80) + k * np.log(80)
    assert value == pytest.approx(expected)


def test_aicc_value_matches_formula():
    crit = AICcCriterion(n_samples=30)
    rss = 10.0
    k = 5
    value = float(np.asarray(crit.evaluate(rss, k=k)))
    expected = 30 * np.log(rss / 30) + 2 * k + (2 * k * (k + 1)) / (30 - k - 1)
    assert value == pytest.approx(expected)


def test_aicc_returns_inf_at_boundary():
    crit = AICcCriterion(n_samples=10)
    assert np.isinf(float(np.asarray(crit.evaluate(1.0, k=9))))


def test_hqic_value_matches_formula():
    n = 100
    k = 6
    rss = 18.0
    crit = HQICCriterion(n_samples=n)
    value = float(np.asarray(crit.evaluate(rss, k=k)))
    expected = n * np.log(rss / n) + 2.0 * k * np.log(np.log(n))
    assert value == pytest.approx(expected)


def test_ebic_value_matches_formula():
    n = 120
    p = 25
    k = 4
    gamma = 0.3
    rss = 15.0
    crit = EBICCriterion(n_samples=n, p=p, gamma=gamma)
    value = float(np.asarray(crit.evaluate(rss, k=k)))
    log_choose = (
        math.lgamma(p + 1)
        - math.lgamma(k + 1)
        - math.lgamma(p - k + 1)
    )
    expected = n * np.log(rss / n) + k * np.log(n) + 2.0 * gamma * log_choose
    assert value == pytest.approx(expected)


def test_gcv_value_matches_formula():
    n = 90
    k = 7
    rss = 9.0
    crit = GCVCriterion(n_samples=n)
    value = float(np.asarray(crit.evaluate(rss, k=k)))
    expected = (rss / n) / ((n - k) / n) ** 2
    assert value == pytest.approx(expected)


def test_gcv_returns_inf_when_degrees_of_freedom_are_exhausted():
    crit = GCVCriterion(n_samples=8)
    assert np.isinf(float(np.asarray(crit.evaluate(1.0, k=8))))


def test_is_improvement_uses_tolerance():
    state = make_state()
    crit = AICCriterion(n_samples=state.data.n_samples, abs_tol=1e-8)
    incumbent = 10.0
    assert crit.is_improvement(9.0, incumbent)
    assert not crit.is_improvement(incumbent - crit.abs_tol / 2, incumbent)


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
