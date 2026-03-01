"""Exact best-subset oracle utilities for small synthetic problems."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from .datasets import BenchmarkDataset


@dataclass(frozen=True)
class ExactSubsetOracleResult:
    """Exact support found by exhaustive train-RSS search at fixed subset size."""

    active_set: list[int]
    k: int
    n_features: int
    n_combinations: int
    train_rss: float
    train_mse: float
    val_mse: float
    test_mse: float


def _support_beta_from_gram(
    gram: np.ndarray,
    cov: np.ndarray,
    support: tuple[int, ...],
) -> np.ndarray:
    if not support:
        return np.zeros(0, dtype=float)
    idx = np.asarray(support, dtype=int)
    gram_ss = gram[np.ix_(idx, idx)]
    gram_ss = 0.5 * (gram_ss + gram_ss.T)
    cov_s = cov[idx]
    return np.linalg.pinv(gram_ss) @ cov_s


def _rss_from_gram(
    *,
    gram: np.ndarray,
    cov: np.ndarray,
    y_norm: float,
    support: tuple[int, ...],
    beta_s: np.ndarray,
) -> float:
    if not support:
        return float(y_norm)
    idx = np.asarray(support, dtype=int)
    gram_ss = gram[np.ix_(idx, idx)]
    cov_s = cov[idx]
    rss = float(y_norm - 2.0 * (cov_s @ beta_s) + beta_s @ (gram_ss @ beta_s))
    if rss < 0.0 and rss > -1e-10:
        return 0.0
    return rss


def _full_beta(n_features: int, support: tuple[int, ...], beta_s: np.ndarray) -> np.ndarray:
    beta = np.zeros(n_features, dtype=float)
    if support:
        beta[np.asarray(support, dtype=int)] = beta_s
    return beta


def _mse(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    residual = y - X @ beta
    return float((residual @ residual) / X.shape[0])


def exact_best_subset_train_rss(
    dataset: BenchmarkDataset,
    *,
    k: int,
    max_features: int = 18,
    max_combinations: int = 60000,
) -> ExactSubsetOracleResult | None:
    """Return exact best subset (train RSS) when exhaustive search is tractable.

    Returns `None` when the dataset is intentionally outside the oracle budget.
    """
    n_features = int(dataset.X_train.shape[1])
    if k < 0 or k > n_features:
        raise ValueError(f"k must be in [0, {n_features}], got {k}.")
    if n_features > max_features:
        return None

    n_combinations = comb(n_features, k)
    if n_combinations > max_combinations:
        return None

    gram = dataset.train_data.gram
    cov = dataset.train_data.cov
    y_norm = float(dataset.train_data.y_norm)

    best_support: tuple[int, ...] | None = None
    best_beta_s: np.ndarray | None = None
    best_rss = float("inf")

    for support in combinations(range(n_features), k):
        beta_s = _support_beta_from_gram(gram, cov, support)
        rss = _rss_from_gram(
            gram=gram,
            cov=cov,
            y_norm=y_norm,
            support=support,
            beta_s=beta_s,
        )
        if rss < best_rss:
            best_rss = rss
            best_support = support
            best_beta_s = beta_s

    if best_support is None or best_beta_s is None:
        raise RuntimeError("Exhaustive subset search failed to produce a result.")

    beta = _full_beta(n_features, best_support, best_beta_s)
    return ExactSubsetOracleResult(
        active_set=[int(i) for i in best_support],
        k=int(k),
        n_features=n_features,
        n_combinations=int(n_combinations),
        train_rss=float(best_rss),
        train_mse=float(best_rss / dataset.X_train.shape[0]),
        val_mse=_mse(dataset.X_val, dataset.y_val, beta),
        test_mse=_mse(dataset.X_test, dataset.y_test, beta),
    )


__all__ = [
    "ExactSubsetOracleResult",
    "exact_best_subset_train_rss",
]
