"""Shared algebra helpers for state update routines."""

from __future__ import annotations

import numpy as np

from .solvers import normalize_solver_policy


def validate_solver_params(
    solver_policy: str, ridge_alpha: float, pinv_rcond: float
) -> tuple[str, float, float]:
    """Normalize and validate solver policy hyperparameters."""
    policy = normalize_solver_policy(solver_policy)
    ridge = float(ridge_alpha)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge_alpha must be finite and >= 0.")
    rcond = float(pinv_rcond)
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("pinv_rcond must be finite and > 0.")
    return policy, ridge, rcond


def backward_rss_scores(
    rss: float, beta_s: np.ndarray, gram_inv: np.ndarray, tol: float
) -> np.ndarray:
    """Return RSS after removing each active feature via Schur complement."""
    diag = np.diag(gram_inv)
    rss_new = np.full(beta_s.shape[0], np.inf, dtype=float)
    safe = np.isfinite(diag) & (diag > tol)
    if np.any(safe):
        rss_new[safe] = rss + (beta_s[safe] ** 2) / diag[safe]
    return rss_new


def backward_schur_downdate(
    gram_inv: np.ndarray, beta_s: np.ndarray, remove_idx: int, tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Compute inverse-Gram/beta downdate for removing one active feature."""
    k = beta_s.shape[0]
    if not 0 <= remove_idx < k:
        return None
    idx_keep = np.delete(np.arange(k), remove_idx)
    k_22 = gram_inv[remove_idx, remove_idx]
    if not np.isfinite(k_22) or k_22 <= tol:
        return None

    K_11 = gram_inv[np.ix_(idx_keep, idx_keep)]
    k_12 = gram_inv[idx_keep, remove_idx]
    K_new = K_11 - np.outer(k_12, k_12) / k_22

    beta_removed = beta_s[remove_idx]
    beta_keep = np.delete(beta_s, remove_idx)
    beta_new = (
        beta_keep - (beta_removed / k_22) * k_12 if beta_keep.size else np.zeros(0)
    )
    rss_delta = float((beta_removed**2) / k_22)
    return idx_keep, K_new, beta_new, rss_delta


__all__ = [
    "validate_solver_params",
    "backward_rss_scores",
    "backward_schur_downdate",
]
