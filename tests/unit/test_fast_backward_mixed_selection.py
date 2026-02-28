import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.routines_core import BackwardSelection, MixedSelection
from tests.helpers import explicit_beta_rss


def test_fast_backward_best_rss_matches_explicit_solution():
    rng = np.random.default_rng(789)
    n, p, steps = 200, 20, 6
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    state = BackwardSelection(criterion_cls=BestRSSCriterion, allow_worse=True).fit(
        data=data, max_steps=steps
    )
    beta_expected, rss_expected = explicit_beta_rss(data, state.active_set)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected


def test_fast_mixed_aic_diagonal_returns_explicitly_consistent_state():
    p = 10
    idx = np.arange(1, p + 1, dtype=float)
    gram = np.eye(p)
    true_beta = 2**idx
    cov = gram @ true_beta
    y_norm = float(true_beta @ true_beta)
    n_samples = 100

    data = GramData(gram, cov, y_norm, n_samples)

    state = MixedSelection().fit(data=data, max_forward_steps=5, max_total_steps=8)
    beta_expected, rss_expected = explicit_beta_rss(data, state.active_set)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected
