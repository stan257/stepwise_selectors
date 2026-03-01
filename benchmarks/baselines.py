"""Simple baseline models for benchmark comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BaselineState:
    """State-like output matching the benchmark metrics contract."""

    active_set: list[int]
    beta: np.ndarray
    rss: float


class TopKAbsCovBaseline:
    """Select top-k features by |X^T y| on train data, then OLS refit."""

    def __init__(self, *, k: int):
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")
        if k <= 0:
            raise ValueError("k must be > 0.")
        self.k = int(k)

    def fit(self, *, X_train: np.ndarray, y_train: np.ndarray) -> BaselineState:
        n_features = int(X_train.shape[1])
        k = min(self.k, n_features)

        cov = X_train.T @ y_train
        order = np.argsort(-np.abs(cov), kind="stable")
        active = np.sort(order[:k]).astype(int)

        beta = np.zeros(n_features, dtype=float)
        if active.size:
            Xs = X_train[:, active]
            coeffs, *_ = np.linalg.lstsq(Xs, y_train, rcond=None)
            beta[active] = coeffs

        residual = y_train - X_train @ beta
        rss = float(residual @ residual)

        return BaselineState(
            active_set=[int(i) for i in active.tolist()],
            beta=beta,
            rss=rss,
        )


BASELINE_MAP = {
    "TopKAbsCovBaseline": TopKAbsCovBaseline,
}
