import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.fast_routines import FastBeamForwardSelection
from tests.helpers import explicit_beta_rss


def test_fast_beam_forward_best_rss_matches_explicit_solution():
    rng = np.random.default_rng(456)
    n, p, k = 200, 25, 6
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    state = FastBeamForwardSelection(
        beam_width=3, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=k)
    beta_expected, rss_expected = explicit_beta_rss(data, state.active_set)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected
