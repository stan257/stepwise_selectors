import itertools

import numpy as np
import pytest

from selection.criteria import (
    AICCriterion,
    AICcCriterion,
    BICCriterion,
    BestRSSCriterion,
    EBICCriterion,
    GCVCriterion,
    HQICCriterion,
)
from selection.definitions import GramData
from selection.routines import BeamForwardSelection, ForwardSelection


def _explicit_rss(X: np.ndarray, y: np.ndarray, subset: tuple[int, ...]) -> tuple[float, np.ndarray]:
    if not subset:
        return float(y @ y), np.zeros(0)
    idx = np.array(subset, dtype=int)
    Xs = X[:, idx]
    beta = np.linalg.solve(Xs.T @ Xs, Xs.T @ y)
    resid = y - Xs @ beta
    return float(resid @ resid), beta


def _rss_from_gram(data: GramData, subset: tuple[int, ...]) -> float:
    if not subset:
        return float(data.y_norm)
    idx = np.array(subset, dtype=int)
    gram_ss = data.gram[np.ix_(idx, idx)]
    cov_s = data.cov[idx]
    beta = np.linalg.solve(gram_ss, cov_s)
    return float(data.y_norm - cov_s @ beta)


def test_beam_forward_matches_exhaustive_best_subset():
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
    state = BeamForwardSelection(
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


def test_forward_matches_exhaustive_across_criteria():
    p = 8
    cov = np.array([2.5, 2.0, 1.6, 1.3, 1.0, 0.8, 0.6, 0.4])
    gram = np.eye(p)
    y_norm = float(np.sum(cov**2) + 3.0)
    n_samples = 50
    data = GramData(gram, cov, y_norm, n_samples)

    criteria = [
        ("AIC", AICCriterion(n_samples=n_samples), AICCriterion, {}),
        ("BIC", BICCriterion(n_samples=n_samples), BICCriterion, {}),
        ("AICc", AICcCriterion(n_samples=n_samples), AICcCriterion, {}),
        ("HQIC", HQICCriterion(n_samples=n_samples), HQICCriterion, {}),
        (
            "EBIC",
            EBICCriterion(n_samples=n_samples, p=p, gamma=0.5),
            EBICCriterion,
            {"gamma": 0.5},
        ),
        ("GCV", GCVCriterion(n_samples=n_samples), GCVCriterion, {}),
        ("BestRSS", BestRSSCriterion(), BestRSSCriterion, {}),
    ]

    for name, oracle_crit, crit_cls, kwargs in criteria:
        best_score = None
        best_sets: set[tuple[int, ...]] = set()
        for r in range(p + 1):
            for subset in itertools.combinations(range(p), r):
                rss = _rss_from_gram(data, subset)
                score = float(np.asarray(oracle_crit.evaluate(rss, r)))
                if best_score is None or score < best_score - 1e-10:
                    best_score = score
                    best_sets = {subset}
                elif abs(score - best_score) <= 1e-10:
                    best_sets.add(subset)

        state = ForwardSelection(
            criterion_cls=crit_cls, criterion_kwargs=kwargs
        ).fit(data=data, max_steps=p)

        chosen = tuple(sorted(state.active_set))
        assert chosen in best_sets, f"{name} expected {best_sets}, got {chosen}"

        chosen_rss = _rss_from_gram(data, chosen)
        chosen_score = float(
            np.asarray(oracle_crit.evaluate(chosen_rss, len(chosen)))
        )
        assert best_score is not None
        assert pytest.approx(chosen_score, rel=1e-10, abs=1e-10) == best_score
