import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection import GramData
from selection import ForwardSelection


@pytest.mark.parametrize("seed", range(24))
def test_best_rss_forward_monotone_and_matches_ols(seed):
    rng = np.random.default_rng(seed)
    n, p = 80, 8
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    gram = X.T @ X
    cov = X.T @ y
    y_norm = float(y @ y)
    data = GramData(gram, cov, y_norm, n)

    k_max = 5
    rss_vals = []

    for k in range(k_max + 1):
        state = ForwardSelection(criterion=BestRSSCriterion).fit(
            data=data, max_steps=k
        )
        rss_vals.append(state.rss)

        idx = np.array(state.active_set, dtype=int)
        if idx.size:
            beta_expected = np.linalg.solve(gram[np.ix_(idx, idx)], cov[idx])
            beta_full = np.zeros(p)
            beta_full[idx] = beta_expected
            rss_expected = y_norm - cov[idx] @ beta_expected
        else:
            beta_full = np.zeros(p)
            rss_expected = y_norm

        np.testing.assert_allclose(state.beta, beta_full, atol=1e-8, rtol=1e-8)
        assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == rss_expected

    assert all(
        later <= earlier + 1e-9 for earlier, later in zip(rss_vals, rss_vals[1:])
    )
