import numpy as np
import pytest

from selection import GramData
from selection.core.incremental_solver import IncrementalSolver
from selection import SelectionState


def test_random_steps_preserve_active_set_and_rss_consistency():
    rng = np.random.default_rng(2024)
    n, p = 140, 40
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + 0.1 * rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, y @ y, n)

    state = IncrementalSolver.create(data, tol=1e-12)

    for _ in range(30):
        scored = state.candidate_scores()
        can_forward = scored is not None and scored[0].size
        # Prefer forward moves but allow backward steps to exercise downdates.
        do_forward = can_forward and (state.k == 0 or rng.random() < 0.6)

        if do_forward:
            candidates, _ = scored
            feat_idx = int(rng.choice(candidates))
            state.apply_forward(feat_idx)
        elif state.k:
            drop_idx = int(rng.integers(0, state.k))
            state.apply_backward(drop_idx)

        assert len(state.active_set) == len(set(state.active_set))
        assert all(0 <= feat < p for feat in state.active_set)

        # Reconstruct RSS from the active Gram block and compare.
        idx = np.array(state.active_set, dtype=int)
        if idx.size:
            G_ss = data.gram[np.ix_(idx, idx)]
            cov_s = data.cov[idx]
            beta_s = np.linalg.solve(G_ss, cov_s)
            rss = float(data.y_norm - cov_s @ beta_s)
            np.testing.assert_allclose(state.rss, rss, atol=1e-8, rtol=1e-8)
        else:
            np.testing.assert_allclose(state.rss, data.y_norm, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    "active_set,error_type,match",
    [
        ([-1], ValueError, "out of range"),
        ([4], ValueError, "out of range"),
        ([1, 1], ValueError, "duplicate"),
        ([0, 1.5], TypeError, "integers"),
    ],
)
def test_incremental_solver_from_active_set_rejects_invalid_indices(
    active_set, error_type, match
):
    data = GramData(
        gram=np.eye(4),
        cov=np.array([2.0, 1.0, 0.5, 0.25]),
        y_norm=10.0,
        n_samples=40,
    )

    with pytest.raises(error_type, match=match):
        IncrementalSolver.from_active_set(data, active_set, tol=1e-12)


@pytest.mark.parametrize("seed", [303, 304, 305])
def test_incremental_solver_from_active_set_matches_selection_state(seed: int):
    rng = np.random.default_rng(seed)
    n, p = 160, 24
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    data = GramData(X.T @ X, X.T @ y, float(y @ y), n_samples=n)

    for _ in range(10):
        k = int(rng.integers(0, 10))
        active = sorted(int(i) for i in rng.choice(p, size=k, replace=False))

        fstate = IncrementalSolver.from_active_set(data, active, tol=1e-12)
        sstate = SelectionState(data)
        sstate.init_from_active_set(active)

        beta_forward = np.zeros(p, dtype=float)
        if fstate.k:
            idx = np.array(fstate.active_set, dtype=int)
            beta_forward[idx] = fstate.beta_S[: fstate.k]

        np.testing.assert_allclose(beta_forward, sstate.beta, atol=1e-8, rtol=1e-8)
        assert fstate.rss == pytest.approx(sstate.rss, rel=1e-8, abs=1e-8)
