import numpy as np
import pytest

from selection.definitions import GramData
from selection.routines_core import ForwardState


def test_fast_downdate_matches_rebuild():
    rng = np.random.default_rng(135)
    n, p = 120, 12
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.05 * rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    # Build a forward state with a few steps.
    state = ForwardState.create(data, tol=1e-10)
    for _ in range(6):
        cand, rss_new = state.candidate_scores()
        feat = int(cand[int(np.argmin(rss_new))])
        state.apply_forward(feat)

    # Remove one feature and compare with a full rebuild.
    removed_pos = 2
    active_after = state.active_set[:removed_pos] + state.active_set[removed_pos + 1 :]

    state.apply_backward(removed_pos)
    rebuilt = ForwardState.from_active_set(data, active_after, tol=1e-10)

    np.testing.assert_allclose(state.rss, rebuilt.rss, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state.r, rebuilt.r, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state.v, rebuilt.v, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state.K[: state.k, : state.k], rebuilt.K[: state.k, : state.k], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state.beta_S[: state.k], rebuilt.beta_S[: state.k], atol=1e-8, rtol=1e-8)


def test_fast_backward_failure_does_not_mutate_active_set():
    rng = np.random.default_rng(246)
    n, p = 50, 4
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    state = ForwardState.from_active_set(data, [0], tol=1e-12)
    state.K[0, 0] = -abs(state.K[0, 0])
    active_before = list(state.active_set)
    mask_before = state.active_mask.copy()
    k_before = state.k

    with pytest.raises(ValueError, match="near-singular pivot"):
        state.apply_backward(0)

    assert state.active_set == active_before
    np.testing.assert_array_equal(state.active_mask, mask_before)
    assert state.k == k_before
