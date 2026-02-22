import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.fast_routines import FastBeamBackwardSelection, FastBeamMixedSelection


def _explicit_beta_rss(data: GramData, active_set: list[int]):
    p = data.gram.shape[0]
    beta = np.zeros(p, dtype=float)
    if not active_set:
        return beta, float(data.y_norm)
    idx = np.array(active_set, dtype=int)
    gram_ss = data.gram[np.ix_(idx, idx)]
    cov_s = data.cov[idx]
    beta_s = np.linalg.solve(gram_ss, cov_s)
    beta[idx] = beta_s
    rss = float(data.y_norm - cov_s @ beta_s)
    return beta, rss


def _make_problem(seed=777, n=200, p=18):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)
    return GramData(X.T @ X, X.T @ y, y @ y, n)


def test_fast_beam_backward_best_rss_matches_explicit_solution():
    data = _make_problem()
    state = FastBeamBackwardSelection(
        beam_width=3, allow_worse=True, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=4)
    beta_expected, rss_expected = _explicit_beta_rss(data, state.active_set)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected


def test_fast_beam_mixed_best_rss_matches_explicit_solution():
    data = _make_problem(seed=778)
    state = FastBeamMixedSelection(
        beam_width=2, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_forward_steps=3, max_total_steps=6)
    beta_expected, rss_expected = _explicit_beta_rss(data, state.active_set)

    np.testing.assert_allclose(state.beta, beta_expected, atol=1e-8, rtol=1e-8)
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected
