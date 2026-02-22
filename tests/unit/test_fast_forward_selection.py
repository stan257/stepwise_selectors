import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.fast_routines import FastForwardSelection
from tests.helpers import explicit_beta_rss


def test_fast_forward_best_rss_matches_explicit_solution():
    rng = np.random.default_rng(123)
    n, p, k = 200, 25, 10
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    state = FastForwardSelection(criterion_cls=BestRSSCriterion).fit(
        data=data, max_steps=k
    )
    beta_expected, rss_expected = explicit_beta_rss(data, state.active_set)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected


def test_fast_forward_aic_diagonal_returns_explicitly_consistent_state():
    p = 8
    idx = np.arange(1, p + 1, dtype=float)
    gram = np.eye(p)
    true_beta = 2**idx
    cov = gram @ true_beta
    y_norm = float(true_beta @ true_beta)
    n_samples = 100

    data = GramData(gram, cov, y_norm, n_samples)

    state = FastForwardSelection().fit(data=data, max_steps=5)
    beta_expected, rss_expected = explicit_beta_rss(data, state.active_set)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected
