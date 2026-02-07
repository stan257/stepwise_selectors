import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.fast_routines import FastBeamBackwardSelection, FastBeamMixedSelection
from selection.legacy_routines import BeamBackwardSelection, BeamMixedSelection


def _make_problem(seed=777, n=200, p=18):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)
    return GramData(X.T @ X, X.T @ y, y @ y, n)


def test_fast_beam_backward_matches_standard_best_rss():
    data = _make_problem()
    fast = FastBeamBackwardSelection(
        beam_width=3, allow_worse=True, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=4)
    ref = BeamBackwardSelection(
        beam_width=3, allow_worse=True, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=4)

    assert set(fast.active_set) == set(ref.active_set)
    np.testing.assert_allclose(fast.beta, ref.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast.rss, rel=1e-8, abs=1e-8) == ref.rss


def test_fast_beam_mixed_matches_standard_best_rss():
    data = _make_problem(seed=778)
    fast = FastBeamMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_forward_steps=3, max_total_steps=6)
    ref = BeamMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_forward_steps=3, max_total_steps=6)

    assert set(fast.active_set) == set(ref.active_set)
    np.testing.assert_allclose(fast.beta, ref.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast.rss, rel=1e-8, abs=1e-8) == ref.rss
