import itertools

import numpy as np
import pytest

from selection.criteria import BestRSSCriterion
from selection.definitions import GramData
from selection.fast_routines import FastBeamForwardSelection


def _explicit_rss(X: np.ndarray, y: np.ndarray, subset: tuple[int, ...]) -> tuple[float, np.ndarray]:
    if not subset:
        return float(y @ y), np.zeros(0)
    idx = np.array(subset, dtype=int)
    Xs = X[:, idx]
    beta = np.linalg.solve(Xs.T @ Xs, Xs.T @ y)
    resid = y - Xs @ beta
    return float(resid @ resid), beta


def test_fast_beam_forward_matches_exhaustive_best_subset():
    rng = np.random.default_rng(2026)
    n, p, k = 40, 8, 3
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p)
    y = X @ beta_true + 0.1 * rng.standard_normal(n)

    # Exhaustive oracle over all k-subsets.
    best_rss = np.inf
    best_sets: set[tuple[int, ...]] = set()
    for subset in itertools.combinations(range(p), k):
        rss, _ = _explicit_rss(X, y, subset)
        if rss < best_rss - 1e-10:
            best_rss = rss
            best_sets = {subset}
        elif abs(rss - best_rss) <= 1e-10:
            best_sets.add(subset)

    data = GramData(X.T @ X, X.T @ y, y @ y, n)
    # Beam width large enough to retain all subsets up to size k.
    beam_width = 100  # >= C(8, 3) = 56
    state = FastBeamForwardSelection(
        beam_width=beam_width, criterion_cls=BestRSSCriterion
    ).fit(data=data, max_steps=k)

    chosen = tuple(sorted(state.active_set))
    assert len(chosen) == k
    assert chosen in best_sets
    assert pytest.approx(state.rss, rel=1e-8, abs=1e-8) == best_rss

    # Validate coefficients against the oracle for the chosen subset.
    _, beta = _explicit_rss(X, y, chosen)
    beta_full = np.zeros(p)
    beta_full[np.array(chosen, dtype=int)] = beta
    np.testing.assert_allclose(state.beta, beta_full, atol=1e-8, rtol=1e-8)
