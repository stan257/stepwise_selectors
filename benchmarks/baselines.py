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
    intercept: float = 0.0


def _require_sklearn() -> None:
    try:
        from sklearn.linear_model import LassoCV, Ridge  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "scikit-learn is required for this baseline. "
            "Install it (e.g. `python -m pip install scikit-learn`) and retry."
        ) from exc


def _to_original_scale(
    *,
    coef_scaled: np.ndarray,
    intercept_scaled: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, float]:
    beta = coef_scaled / scale
    intercept = float(intercept_scaled - np.dot(mean / scale, coef_scaled))
    return beta, intercept


def _active_from_beta(beta: np.ndarray, *, coef_tol: float) -> np.ndarray:
    return np.flatnonzero(np.abs(beta) > coef_tol).astype(int)


def _train_rss(X: np.ndarray, y: np.ndarray, beta: np.ndarray, intercept: float) -> float:
    residual = y - (X @ beta + intercept)
    return float(residual @ residual)


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

        rss = _train_rss(X_train, y_train, beta, 0.0)
        return BaselineState(
            active_set=[int(i) for i in active.tolist()],
            beta=beta,
            rss=rss,
            intercept=0.0,
        )


class LassoCVBaseline:
    """LassoCV baseline with standardized features and intercept recovery."""

    def __init__(
        self,
        *,
        cv_folds: int = 5,
        random_state: int | None = 0,
        max_iter: int = 20000,
        coef_tol: float = 1e-8,
    ):
        if isinstance(cv_folds, bool) or not isinstance(cv_folds, int):
            raise TypeError("cv_folds must be an integer.")
        if cv_folds < 2:
            raise ValueError("cv_folds must be >= 2.")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise TypeError("random_state must be an integer or None.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer.")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0.")
        if coef_tol < 0:
            raise ValueError("coef_tol must be >= 0.")

        self.cv_folds = int(cv_folds)
        self.random_state = None if random_state is None else int(random_state)
        self.max_iter = int(max_iter)
        self.coef_tol = float(coef_tol)

    def fit(self, *, X_train: np.ndarray, y_train: np.ndarray) -> BaselineState:
        _require_sklearn()
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model = LassoCV(
            cv=self.cv_folds,
            random_state=self.random_state,
            max_iter=self.max_iter,
            fit_intercept=True,
        )
        model.fit(X_scaled, y_train)

        coef_scaled = np.asarray(model.coef_, dtype=float)
        beta, intercept = _to_original_scale(
            coef_scaled=coef_scaled,
            intercept_scaled=float(model.intercept_),
            mean=np.asarray(scaler.mean_, dtype=float),
            scale=np.asarray(scaler.scale_, dtype=float),
        )
        active = _active_from_beta(beta, coef_tol=self.coef_tol)
        rss = _train_rss(X_train, y_train, beta, intercept)
        return BaselineState(
            active_set=[int(i) for i in active.tolist()],
            beta=beta,
            rss=rss,
            intercept=intercept,
        )


class AdaptiveLassoBaseline:
    """Two-stage adaptive lasso using ridge weights plus LassoCV."""

    def __init__(
        self,
        *,
        cv_folds: int = 5,
        random_state: int | None = 0,
        max_iter: int = 20000,
        gamma: float = 1.0,
        weight_eps: float = 1e-6,
        ridge_alpha: float = 1e-3,
        coef_tol: float = 1e-8,
    ):
        if isinstance(cv_folds, bool) or not isinstance(cv_folds, int):
            raise TypeError("cv_folds must be an integer.")
        if cv_folds < 2:
            raise ValueError("cv_folds must be >= 2.")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise TypeError("random_state must be an integer or None.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer.")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0.")
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if weight_eps <= 0:
            raise ValueError("weight_eps must be > 0.")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0.")
        if coef_tol < 0:
            raise ValueError("coef_tol must be >= 0.")

        self.cv_folds = int(cv_folds)
        self.random_state = None if random_state is None else int(random_state)
        self.max_iter = int(max_iter)
        self.gamma = float(gamma)
        self.weight_eps = float(weight_eps)
        self.ridge_alpha = float(ridge_alpha)
        self.coef_tol = float(coef_tol)

    def fit(self, *, X_train: np.ndarray, y_train: np.ndarray) -> BaselineState:
        _require_sklearn()
        from sklearn.linear_model import LassoCV, Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # Stage 1: stable dense estimate for adaptive weights.
        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        ridge.fit(X_scaled, y_train)
        beta_init = np.asarray(ridge.coef_, dtype=float)
        weights = np.power(np.abs(beta_init) + self.weight_eps, -self.gamma)
        weights = np.clip(weights, 1e-4, 1e6)

        # Stage 2: weighted lasso via column rescaling.
        X_weighted = X_scaled / weights
        model = LassoCV(
            cv=self.cv_folds,
            random_state=self.random_state,
            max_iter=self.max_iter,
            fit_intercept=True,
        )
        model.fit(X_weighted, y_train)

        theta = np.asarray(model.coef_, dtype=float)
        coef_scaled = theta / weights
        beta, intercept = _to_original_scale(
            coef_scaled=coef_scaled,
            intercept_scaled=float(model.intercept_),
            mean=np.asarray(scaler.mean_, dtype=float),
            scale=np.asarray(scaler.scale_, dtype=float),
        )
        active = _active_from_beta(beta, coef_tol=self.coef_tol)
        rss = _train_rss(X_train, y_train, beta, intercept)
        return BaselineState(
            active_set=[int(i) for i in active.tolist()],
            beta=beta,
            rss=rss,
            intercept=intercept,
        )


BASELINE_MAP = {
    "TopKAbsCovBaseline": TopKAbsCovBaseline,
    "LassoCVBaseline": LassoCVBaseline,
    "AdaptiveLassoBaseline": AdaptiveLassoBaseline,
}
