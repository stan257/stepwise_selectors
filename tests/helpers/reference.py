"""Reference computations reused across test modules."""

from __future__ import annotations

import numpy as np

from selection import CrossValGramData, GramData


def make_regression_gram(
    seed: int,
    *,
    n: int,
    p: int,
    noise_scale: float = 0.1,
) -> GramData:
    """Build a random regression problem directly as Gram statistics."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p)
    y = X @ beta + noise_scale * rng.standard_normal(n)
    return GramData(X.T @ X, X.T @ y, float(y @ y), n)


def make_cv_problem(
    *,
    folds: int = 4,
    n: int = 120,
    p: int = 12,
    support: int = 4,
    seed: int = 123,
    noise_scale: float = 0.05,
) -> CrossValGramData:
    """Build a synthetic CV problem with a shared sparse signal."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    beta[:support] = 1.0
    fold_data = []
    for fold_seed in rng.integers(0, 1_000_000, size=folds):
        r = np.random.default_rng(int(fold_seed))
        X = r.standard_normal((n, p))
        y = X @ beta + noise_scale * r.standard_normal(n)
        fold_data.append(GramData(X.T @ X, X.T @ y, float(y @ y), n))
    return CrossValGramData(fold_data)


def make_cv_regression_gram(
    seed: int,
    *,
    folds: int,
    n: int,
    p: int,
    noise_scale: float = 0.1,
) -> CrossValGramData:
    """Build CV folds from a shared random dense coefficient vector."""
    rng = np.random.default_rng(seed)
    beta = rng.standard_normal(p)
    fold_data = []
    for fold_seed in rng.integers(0, 1_000_000, size=folds):
        r = np.random.default_rng(int(fold_seed))
        X = r.standard_normal((n, p))
        y = X @ beta + noise_scale * r.standard_normal(n)
        fold_data.append(GramData(X.T @ X, X.T @ y, float(y @ y), n))
    return CrossValGramData(fold_data)


def explicit_beta_from_active(data: GramData, active_set: list[int] | tuple[int, ...]) -> np.ndarray:
    """Return the OLS coefficients on the given support, zero elsewhere."""
    beta = np.zeros(data.gram.shape[0], dtype=float)
    if not active_set:
        return beta
    idx = np.array(active_set, dtype=int)
    beta[idx] = np.linalg.solve(data.gram[np.ix_(idx, idx)], data.cov[idx])
    return beta


def explicit_beta_rss(
    data: GramData, active_set: list[int] | tuple[int, ...]
) -> tuple[np.ndarray, float]:
    """Return full-length OLS coefficients and RSS for a given support."""
    beta = explicit_beta_from_active(data, active_set)
    if not active_set:
        return beta, float(data.y_norm)
    idx = np.array(active_set, dtype=int)
    beta_s = beta[idx]
    rss = float(data.y_norm - data.cov[idx] @ beta_s)
    return beta, rss


def explicit_cv_rss(
    cv_data: CrossValGramData, active_set: list[int] | tuple[int, ...]
) -> float:
    """Compute summed CV validation RSS by explicit fold refits."""
    if not active_set:
        return float(np.sum(cv_data.y_norm_folds))
    idx = np.array(active_set, dtype=int)
    total = 0.0
    for fold_idx in range(cv_data.n_folds):
        train = cv_data.train_data_for_fold(fold_idx)
        beta = np.linalg.solve(train.gram[np.ix_(idx, idx)], train.cov[idx])
        gram_val = cv_data.gram_folds[fold_idx]
        cov_val = cv_data.cov_folds[fold_idx]
        y_norm_val = cv_data.y_norm_folds[fold_idx]
        gram_val_ss = gram_val[np.ix_(idx, idx)]
        total += y_norm_val - 2.0 * float(beta @ cov_val[idx]) + float(
            beta @ (gram_val_ss @ beta)
        )
    return float(total)
