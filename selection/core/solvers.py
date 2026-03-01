"""Shared linear-system helpers for active-set Gram solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, TypeVar

import numpy as np

SolverPolicy = Literal["strict", "ridge", "pinv"]
_TDispatch = TypeVar("_TDispatch")


@dataclass(frozen=True)
class ForwardFactorization:
    """Numerical factors needed to initialize IncrementalSolver from an active set."""

    R: np.ndarray
    Z: np.ndarray
    qy: np.ndarray
    K: np.ndarray
    beta: np.ndarray


def normalize_solver_policy(solver_policy: str) -> SolverPolicy:
    """Normalize and validate a solver policy token."""
    policy = str(solver_policy).strip().lower()
    match policy:
        case "strict" | "ridge" | "pinv":
            return policy
        case _:
            raise ValueError("solver_policy must be one of: pinv, ridge, strict.")


def _dispatch_solver(
    policy: str,
    *,
    strict_case: Callable[[], _TDispatch],
    ridge_case: Callable[[], _TDispatch],
    pinv_case: Callable[[], _TDispatch],
) -> _TDispatch:
    """Run the handler for one normalized solver policy."""
    match normalize_solver_policy(policy):
        case "strict":
            return strict_case()
        case "ridge":
            return ridge_case()
        case "pinv":
            return pinv_case()
    raise RuntimeError("Unreachable solver policy branch.")


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
    return _dispatch_solver(
        solver_policy,
        strict_case=lambda: _solve_cholesky(
            gram_ss,
            cov_s,
            context=context,
            strict_pivot_check=True,
        ),
        ridge_case=lambda: _solve_cholesky(
            gram_ss + ridge_alpha * np.eye(gram_ss.shape[0]),
            cov_s,
            context=context,
            strict_pivot_check=False,
        ),
        pinv_case=lambda: _solve_pinv_system(
            gram_ss,
            cov_s,
            pinv_rcond=pinv_rcond,
            context=context,
        ),
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
    return _dispatch_solver(
        solver_policy,
        strict_case=lambda: _build_cholesky_factorization(
            gram_ss,
            cov_s,
            gram_sfull,
            context=context,
            solver_name="strict",
        ),
        ridge_case=lambda: _build_cholesky_factorization(
            gram_ss + ridge_alpha * np.eye(gram_ss.shape[0]),
            cov_s,
            gram_sfull,
            context=context,
            solver_name="ridge",
        ),
        pinv_case=lambda: _build_pinv_factorization(
            gram_ss,
            cov_s,
            gram_sfull,
            pinv_rcond=pinv_rcond,
            context=context,
        ),
    )


def _build_cholesky_factorization(
    gram_ss: np.ndarray,
    cov_s: np.ndarray,
    gram_sfull: np.ndarray,
    *,
    context: str,
    solver_name: str,
) -> ForwardFactorization:
    try:
        L = np.linalg.cholesky(gram_ss)
    except np.linalg.LinAlgError as err:
        raise np.linalg.LinAlgError(
            f"{context} is singular or ill-conditioned under {solver_name} solver."
        ) from err
    R = L.T
    z = np.linalg.solve(R.T, gram_sfull)
    qy = np.linalg.solve(R.T, cov_s)
    invR = np.linalg.solve(R, np.eye(R.shape[0]))
    K = invR @ invR.T
    beta = np.linalg.solve(R, qy)
    return ForwardFactorization(R=R, Z=z, qy=qy, K=K, beta=beta)


def _build_pinv_factorization(
    gram_ss: np.ndarray,
    cov_s: np.ndarray,
    gram_sfull: np.ndarray,
    *,
    pinv_rcond: float,
    context: str,
) -> ForwardFactorization:
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


def _solve_pinv_system(
    gram_ss: np.ndarray,
    cov_s: np.ndarray,
    *,
    pinv_rcond: float,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    gram_sym = 0.5 * (gram_ss + gram_ss.T)
    K = np.linalg.pinv(gram_sym, rcond=pinv_rcond)
    beta = K @ cov_s
    if not (np.isfinite(beta).all() and np.isfinite(K).all()):
        raise np.linalg.LinAlgError(f"{context} produced non-finite pseudo-inverse solve.")
    return beta, K


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
    "normalize_solver_policy",
    "solve_active_system",
    "build_forward_factorization",
]
