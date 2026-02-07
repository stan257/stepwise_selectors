import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.fast_routines import FastBeamForwardSelection
from selection.legacy_routines import BeamForwardSelection


def test_fast_beam_forward_matches_standard_best_rss():
    rng = np.random.default_rng(456)
    n, p, k = 200, 25, 6
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    fast = FastBeamForwardSelection(
        beam_width=3, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=k)
    ref = BeamForwardSelection(
        beam_width=3, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=k)

    assert set(fast.active_set) == set(ref.active_set)
    np.testing.assert_allclose(fast.beta, ref.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast.rss, rel=1e-8, abs=1e-8) == ref.rss
