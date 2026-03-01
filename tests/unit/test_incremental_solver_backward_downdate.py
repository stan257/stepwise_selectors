import numpy as np

from selection.definitions import GramData
from selection.incremental_solver import IncrementalSolver


def test_backward_step_matches_direct_rss_for_active_set():
    rng = np.random.default_rng(135)
    n, p = 120, 12
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.05 * rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    # Build a forward state with a few steps.
    state = IncrementalSolver.create(data, tol=1e-10)
    for _ in range(6):
        cand, rss_new = state.candidate_scores()
        feat = int(cand[int(np.argmin(rss_new))])
        state.apply_forward(feat)

    # Remove one feature and validate against direct OLS on the remaining support.
    state.apply_backward(2)
    idx = np.array(state.active_set, dtype=int)
    if idx.size:
        beta = np.linalg.solve(data.gram[np.ix_(idx, idx)], data.cov[idx])
        rss_expected = float(data.y_norm - data.cov[idx] @ beta)
    else:
        rss_expected = float(data.y_norm)
    np.testing.assert_allclose(state.rss, rss_expected, atol=1e-8, rtol=1e-8)
