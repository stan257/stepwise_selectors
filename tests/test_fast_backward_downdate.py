import numpy as np
import pytest

from selection.definitions import GramData
from selection.fast_routines import FastForwardState


def test_fast_downdate_matches_rebuild():
    rng = np.random.default_rng(135)
    n, p = 120, 12
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.05 * rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    # Build a forward state with a few steps.
    state = FastForwardState.create(data, tol=1e-10)
    for _ in range(6):
        cand, rss_new = state.candidate_scores()
        feat = int(cand[int(np.argmin(rss_new))])
        state.apply_forward(feat)

    # Remove one feature and compare with a full rebuild.
    removed_pos = 2
    active_after = state.active_set[:removed_pos] + state.active_set[removed_pos + 1 :]

    state.apply_backward(removed_pos)
    rebuilt = FastForwardState.from_active_set(data, active_after, tol=1e-10)

    np.testing.assert_allclose(state.rss, rebuilt.rss, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state.r, rebuilt.r, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state.v, rebuilt.v, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state.K[: state.k, : state.k], rebuilt.K[: state.k, : state.k], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state.beta_S[: state.k], rebuilt.beta_S[: state.k], atol=1e-8, rtol=1e-8)
