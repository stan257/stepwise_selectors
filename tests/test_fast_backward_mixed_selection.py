import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.fast_routines import FastBackwardSelection, FastMixedSelection
from selection.legacy_routines import BackwardSelection, MixedSelection


def test_fast_backward_matches_standard_best_rss():
    rng = np.random.default_rng(789)
    n, p, steps = 200, 20, 6
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    fast = FastBackwardSelection(criterion_cls=BestRSSCriterion, allow_worse=True).fit(
        data=data, max_steps=steps
    )
    ref = BackwardSelection(criterion_cls=BestRSSCriterion, allow_worse=True).fit(
        data=data, max_steps=steps
    )

    assert set(fast.active_set) == set(ref.active_set)
    np.testing.assert_allclose(fast.beta, ref.beta, atol=1e-8, rtol=1e-8)
    assert pytest.approx(fast.rss, rel=1e-8, abs=1e-8) == ref.rss


def test_fast_mixed_matches_standard_aic_diagonal():
    p = 10
    idx = np.arange(1, p + 1, dtype=float)
    gram = np.eye(p)
    true_beta = 2**idx
    cov = gram @ true_beta
    y_norm = float(true_beta @ true_beta)
    n_samples = 100

    data = GramData(gram, cov, y_norm, n_samples)

    fast = FastMixedSelection().fit(data=data, max_forward_steps=5, max_total_steps=8)
    ref = MixedSelection().fit(data=data, max_forward_steps=5, max_total_steps=8)

    assert fast.active_set == ref.active_set
    assert pytest.approx(fast.rss, rel=1e-8, abs=1e-8) == ref.rss
