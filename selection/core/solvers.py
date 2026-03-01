"""Shared linear-system helpers for active-set Gram solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SolverPolicy = Literal["strict", "ridge", "pinv"]


@dataclass(frozen=True)
class ForwardFactorization:
    """Numerical factors needed to initialize IncrementalSolver from an active set."""

    R: np.ndarray
    Z: np.ndarray
    qy: np.ndarray
    K: np.ndarray
    beta: np.ndarray


def solve_active_system(
    gram_ss: np.ndarray,
    cov_s: np.ndarray,
    *,
    solver_policy: SolverPolicy,
    ridge_alpha: float,
    pinv_rcond: float,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve coefficients and inverse Gram for one active support.

    Preconditions:
    - `gram_ss` is square and aligned with `cov_s`.
    - caller has already validated shapes/types at API boundary.
    - `solver_policy` and hyperparameters are valid.

    Returns:
    - `(beta, K)` where `K` approximates `(G_SS)^{-1}` under the chosen policy.

    Raises:
    - `np.linalg.LinAlgError` when strict/ridge solves are numerically unstable.
    - `ValueError` for unsupported solver policies.
    """
    match solver_policy:
        case "strict":
            return _solve_cholesky(gram_ss, cov_s, context=context, strict_pivot_check=True)
        case "ridge":
            regularized = gram_ss + ridge_alpha * np.eye(gram_ss.shape[0])
            return _solve_cholesky(
                regularized, cov_s, context=context, strict_pivot_check=False
            )
        case "pinv":
            gram_sym = 0.5 * (gram_ss + gram_ss.T)
            K = np.linalg.pinv(gram_sym, rcond=pinv_rcond)
            beta = K @ cov_s
            if not (np.isfinite(beta).all() and np.isfinite(K).all()):
                raise np.linalg.LinAlgError(
                    f"{context} produced non-finite pseudo-inverse solve."
                )
            return beta, K
        case _:
            raise ValueError(
                f"Unsupported solver policy '{solver_policy}'. "
                "Expected one of: strict, ridge, pinv."
            )


def build_forward_factorization(
    gram_ss: np.ndarray,
    cov_s: np.ndarray,
    gram_sfull: np.ndarray,
    *,
    solver_policy: SolverPolicy,
    ridge_alpha: float,
    pinv_rcond: float,
    context: str,
) -> ForwardFactorization:
    """Build initialization factors for `IncrementalSolver.from_active_set`.

    Preconditions:
    - caller passes support-restricted Gram blocks with consistent dimensions.
    - boundary validation has already normalized solver params.

    Postconditions:
    - returns factors (`R`, `Z`, `qy`, `K`, `beta`) consistent with the chosen
      solver policy for immediate use in incremental updates.
    """
    match solver_policy:
        case "strict" | "ridge":
            work = (
                gram_ss
                if solver_policy == "strict"
                else gram_ss + ridge_alpha * np.eye(gram_ss.shape[0])
            )
            try:
                L = np.linalg.cholesky(work)
            except np.linalg.LinAlgError as err:
                raise np.linalg.LinAlgError(
                    f"{context} is singular or ill-conditioned under {solver_policy} solver."
                ) from err
            R = L.T
            z = np.linalg.solve(R.T, gram_sfull)
            qy = np.linalg.solve(R.T, cov_s)
            invR = np.linalg.solve(R, np.eye(R.shape[0]))
            K = invR @ invR.T
            beta = np.linalg.solve(R, qy)
            return ForwardFactorization(R=R, Z=z, qy=qy, K=K, beta=beta)
        case "pinv":
            # pinv: use a PSD square root and pseudo-inverse projections.
            gram_sym = 0.5 * (gram_ss + gram_ss.T)
            evals, evecs = np.linalg.eigh(gram_sym)
            evals = np.clip(evals, 0.0, None)
            root = (evecs * np.sqrt(evals)) @ evecs.T
            # Convert to an upper-triangular factor: root = Q R => root.T @ root == R.T @ R.
            _, R = np.linalg.qr(root)
            pinv_rt = np.linalg.pinv(R.T, rcond=pinv_rcond)
            z = pinv_rt @ gram_sfull
            qy = pinv_rt @ cov_s
            K = np.linalg.pinv(gram_sym, rcond=pinv_rcond)
            beta = K @ cov_s
            if not (
                np.isfinite(R).all()
                and np.isfinite(z).all()
                and np.isfinite(qy).all()
                and np.isfinite(K).all()
                and np.isfinite(beta).all()
            ):
                raise np.linalg.LinAlgError(
                    f"{context} produced non-finite factors under pinv solver."
                )
            return ForwardFactorization(R=R, Z=z, qy=qy, K=K, beta=beta)
        case _:
            raise ValueError(
                f"Unsupported solver policy '{solver_policy}'. "
                "Expected one of: strict, ridge, pinv."
            )


def _solve_cholesky(
    gram_ss: np.ndarray,
    cov_s: np.ndarray,
    *,
    context: str,
    strict_pivot_check: bool,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        L = np.linalg.cholesky(gram_ss)
    except np.linalg.LinAlgError as err:
        raise np.linalg.LinAlgError(f"{context} is singular or ill-conditioned.") from err
    # Guard against semidefinite/near-singular cases that can slip through
    # floating-point Cholesky with tiny pivots.
    if strict_pivot_check:
        pivots = np.diag(L)
        pivot_tol = np.sqrt(np.finfo(float).eps) * max(
            1.0, float(np.max(np.diag(gram_ss)))
        )
        if np.any(pivots <= pivot_tol):
            raise np.linalg.LinAlgError(
                f"{context} is singular or near-singular under strict solver."
            )
    beta = np.linalg.solve(L.T, np.linalg.solve(L, cov_s))
    K = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(gram_ss.shape[0])))
    return beta, K


__all__ = [
    "SolverPolicy",
    "ForwardFactorization",
    "solve_active_system",
    "build_forward_factorization",
]
