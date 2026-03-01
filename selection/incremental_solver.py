"""State container and QR/Gram updates for fast selection routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .definitions import GramData
from .index_validation import validate_feature_indices
from .state_ops import (
    backward_rss_scores,
    backward_schur_downdate,
    validate_solver_params,
)
from .solvers import build_forward_factorization


@dataclass
class IncrementalSolver:
    data: GramData
    tol: float
    solver_policy: str
    ridge_alpha: float
    pinv_rcond: float
    active_set: list[int]
    active_mask: np.ndarray
    r: np.ndarray
    v: np.ndarray
    rss: float
    Z: np.ndarray
    qy: np.ndarray
    R: np.ndarray
    K: np.ndarray
    beta_S: np.ndarray
    k: int

    @classmethod
    def create(
        cls,
        data: GramData,
        tol: float,
        *,
        solver_policy: str = "strict",
        ridge_alpha: float = 1e-8,
        pinv_rcond: float = 1e-12,
    ) -> "IncrementalSolver":
        policy, ridge, rcond = validate_solver_params(
            solver_policy, ridge_alpha, pinv_rcond
        )
        p = data.gram.shape[0]
        cap = min(64, p)
        return cls(
            data=data,
            tol=tol,
            solver_policy=policy,
            ridge_alpha=ridge,
            pinv_rcond=rcond,
            active_set=[],
            active_mask=np.zeros(p, dtype=bool),
            r=data.cov.astype(float, copy=True),
            v=np.diag(data.gram).astype(float, copy=True),
            rss=float(data.y_norm),
            Z=np.empty((cap, p), dtype=float),
            qy=np.empty(cap, dtype=float),
            R=np.zeros((cap, cap), dtype=float),
            K=np.zeros((cap, cap), dtype=float),
            beta_S=np.empty(cap, dtype=float),
            k=0,
        )

    @classmethod
    def from_active_set(
        cls,
        data: GramData,
        active_set: list[int],
        tol: float,
        *,
        solver_policy: str = "strict",
        ridge_alpha: float = 1e-8,
        pinv_rcond: float = 1e-12,
    ) -> "IncrementalSolver":
        policy, ridge, rcond = validate_solver_params(
            solver_policy, ridge_alpha, pinv_rcond
        )
        p = data.gram.shape[0]
        normalized_active = validate_feature_indices(
            list(active_set), p, context="active_set"
        )
        if not normalized_active:
            return cls.create(
                data,
                tol,
                solver_policy=policy,
                ridge_alpha=ridge,
                pinv_rcond=rcond,
            )
        k = len(normalized_active)
        state = cls.create(
            data,
            tol,
            solver_policy=policy,
            ridge_alpha=ridge,
            pinv_rcond=rcond,
        )
        state._ensure_capacity(k - 1)
        state.active_set = normalized_active
        state.active_mask[:] = False
        state.active_mask[np.array(normalized_active, dtype=int)] = True
        state.k = k

        idx = np.array(normalized_active, dtype=int)
        G_S = data.gram[np.ix_(idx, idx)]
        G_Sfull = data.gram[np.ix_(idx, np.arange(p))]
        cov_S = data.cov[idx]
        factors = build_forward_factorization(
            G_S,
            cov_S,
            G_Sfull,
            solver_policy=policy,
            ridge_alpha=ridge,
            pinv_rcond=rcond,
            context="Active Gram matrix",
        )
        state.R[:k, :k] = factors.R
        state.Z[:k, :] = factors.Z
        state.qy[:k] = factors.qy

        state.r = data.cov - state.Z[:k, :].T @ state.qy[:k]
        state.v = np.diag(data.gram) - np.sum(state.Z[:k, :] ** 2, axis=0)
        state.rss = float(data.y_norm - np.dot(state.qy[:k], state.qy[:k]))

        state.K[:k, :k] = factors.K
        state.beta_S[:k] = factors.beta
        return state

    def clone(self) -> "IncrementalSolver":
        return IncrementalSolver(
            data=self.data,
            tol=self.tol,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
            active_set=list(self.active_set),
            active_mask=self.active_mask.copy(),
            r=self.r.copy(),
            v=self.v.copy(),
            rss=float(self.rss),
            Z=self.Z.copy(),
            qy=self.qy.copy(),
            R=self.R.copy(),
            K=self.K.copy(),
            beta_S=self.beta_S.copy(),
            k=self.k,
        )

    def _ensure_capacity(self, needed: int) -> None:
        if needed < self.Z.shape[0]:
            return
        new_cap = self.Z.shape[0]
        while new_cap <= needed:
            new_cap *= 2
        new_Z = np.empty((new_cap, self.Z.shape[1]), dtype=self.Z.dtype)
        new_qy = np.empty(new_cap, dtype=self.qy.dtype)
        new_R = np.zeros((new_cap, new_cap), dtype=self.R.dtype)
        new_K = np.zeros((new_cap, new_cap), dtype=self.K.dtype)
        new_beta = np.empty(new_cap, dtype=self.beta_S.dtype)
        new_Z[: self.k] = self.Z[: self.k]
        new_qy[: self.k] = self.qy[: self.k]
        new_R[: self.k, : self.k] = self.R[: self.k, : self.k]
        new_K[: self.k, : self.k] = self.K[: self.k, : self.k]
        new_beta[: self.k] = self.beta_S[: self.k]
        self.Z = new_Z
        self.qy = new_qy
        self.R = new_R
        self.K = new_K
        self.beta_S = new_beta

    def candidate_scores(self) -> tuple[np.ndarray, np.ndarray] | None:
        valid = (~self.active_mask) & (self.v > self.tol)
        if not np.any(valid):
            return None
        candidates = np.nonzero(valid)[0]
        r_valid = self.r[candidates]
        v_valid = self.v[candidates]
        rss_new = self.rss - (r_valid**2) / v_valid
        valid_rss = rss_new > -self.tol
        if not np.any(valid_rss):
            return None
        return candidates[valid_rss], rss_new[valid_rss]

    def _prepare_forward_step(
        self, feat_idx: int
    ) -> tuple[float, float, float, np.ndarray, np.ndarray, float]:
        if self.active_mask[feat_idx]:
            raise ValueError("Feature is already active.")
        resid_var = self.v[feat_idx]
        if resid_var <= self.tol:
            raise ValueError("Candidate variance is too small for a stable update.")
        resid_corr = self.r[feat_idx]
        denom = np.sqrt(resid_var)

        if self.k == 0:
            z_col = np.zeros(0)
            proj = 0.0
            qy_proj = 0.0
        else:
            # Project the candidate onto the existing orthonormal directions.
            z_col = self.Z[: self.k, feat_idx]
            proj = z_col @ self.Z[: self.k, :]
            qy_proj = z_col @ self.qy[: self.k]

        z_new = (self.data.gram[:, feat_idx] - proj) / denom
        qy_new = float((self.data.cov[feat_idx] - qy_proj) / denom)
        return resid_var, resid_corr, denom, z_col, z_new, qy_new

    def _update_forward_residuals(
        self, feat_idx: int, z_new: np.ndarray, qy_new: float
    ) -> None:
        # Update residual correlations/variances for all features in O(p).
        self.r -= z_new * qy_new
        self.v -= z_new * z_new
        self.r[feat_idx] = 0.0
        self.v[feat_idx] = 0.0
        self.rss -= qy_new * qy_new

    def _update_forward_basis(
        self, feat_idx: int, z_new: np.ndarray, qy_new: float, z_col: np.ndarray, denom: float
    ) -> None:
        self.active_set.append(int(feat_idx))
        self.active_mask[feat_idx] = True

        # Ensure buffers are large enough for the new basis row/column.
        self._ensure_capacity(self.k)
        self.Z[self.k, :] = z_new
        self.qy[self.k] = qy_new

        # Maintain QR factor R for downdates.
        if self.k:
            self.R[: self.k, self.k] = z_col
        self.R[self.k, self.k] = denom

    def _update_forward_inverse_gram(
        self, feat_idx: int, resid_var: float, resid_corr: float
    ) -> None:
        # Maintain K and beta_S for cheap backward scoring.
        beta_j = resid_corr / resid_var
        if self.k == 0:
            self.beta_S[0] = beta_j
            self.K[0, 0] = 1.0 / resid_var
            return

        idx = np.array(self.active_set[:-1], dtype=int)
        g = self.data.gram[np.ix_(idx, np.array([feat_idx]))].reshape(-1)
        Kk = self.K[: self.k, : self.k]
        proj_vec = Kk @ g
        self.beta_S[: self.k] -= proj_vec * beta_j
        self.beta_S[self.k] = beta_j

        K_new = self.K[: self.k + 1, : self.k + 1]
        K_new[: self.k, : self.k] = Kk + np.outer(proj_vec, proj_vec) / resid_var
        K_new[: self.k, self.k] = -proj_vec / resid_var
        K_new[self.k, : self.k] = -proj_vec / resid_var
        K_new[self.k, self.k] = 1.0 / resid_var

    def apply_forward(self, feat_idx: int) -> float:
        """Apply a forward step while updating QR/Gram state in O(p)."""
        resid_var, resid_corr, denom, z_col, z_new, qy_new = self._prepare_forward_step(
            feat_idx
        )
        self._update_forward_residuals(feat_idx, z_new, qy_new)
        self._update_forward_basis(feat_idx, z_new, qy_new, z_col, denom)
        self._update_forward_inverse_gram(feat_idx, resid_var, resid_corr)

        self.k += 1
        return self.rss

    def backward_scores(self) -> np.ndarray | None:
        if self.k == 0:
            return None
        return backward_rss_scores(
            self.rss,
            self.beta_S[: self.k],
            self.K[: self.k, : self.k],
            self.tol,
        )

    def _apply_backward_qr_downdate(self, idx: int) -> None:
        if idx >= self.k - 1:
            return
        self.R[: self.k, idx : self.k - 1] = self.R[: self.k, idx + 1 : self.k]
        for i in range(idx, self.k - 1):
            a = self.R[i, i]
            b = self.R[i + 1, i]
            if abs(b) <= self.tol:
                continue
            r = np.hypot(a, b)
            c = a / r
            s = b / r

            # QR downdate via Givens rotations; see Golub & Van Loan,
            # Matrix Computations (4th ed.), Givens-based QR updates/downdates.
            Ri = self.R[i, i : self.k - 1].copy()
            Rj = self.R[i + 1, i : self.k - 1].copy()
            self.R[i, i : self.k - 1] = c * Ri + s * Rj
            self.R[i + 1, i : self.k - 1] = -s * Ri + c * Rj

            Zi = self.Z[i, :].copy()
            Zj = self.Z[i + 1, :].copy()
            self.Z[i, :] = c * Zi + s * Zj
            self.Z[i + 1, :] = -s * Zi + c * Zj

            qi = self.qy[i]
            qj = self.qy[i + 1]
            self.qy[i] = c * qi + s * qj
            self.qy[i + 1] = -s * qi + c * qj

    def _refresh_residuals_from_basis(self) -> None:
        if self.k == 0:
            self.r = self.data.cov.astype(float, copy=True)
            self.v = np.diag(self.data.gram).astype(float, copy=True)
            self.rss = float(self.data.y_norm)
            return
        self.r = self.data.cov - self.Z[: self.k, :].T @ self.qy[: self.k]
        self.v = np.diag(self.data.gram) - np.sum(self.Z[: self.k, :] ** 2, axis=0)
        self.rss = float(self.data.y_norm - np.dot(self.qy[: self.k], self.qy[: self.k]))

    def apply_backward(self, idx: int) -> None:
        """Remove one active feature via inverse-Gram and QR downdates."""
        if not (0 <= idx < self.k):
            raise IndexError("Backward index out of range.")

        # Update K and beta_S via the standard inverse-Gram downdate.
        Kk = self.K[: self.k, : self.k]
        beta_k = self.beta_S[: self.k]
        downdate = backward_schur_downdate(Kk, beta_k, idx, self.tol)
        if downdate is None:
            raise ValueError("Backward update failed due to near-singular pivot.")
        _, K_new, beta_new, _ = downdate
        removed = self.active_set.pop(idx)
        self.active_mask[removed] = False

        self._apply_backward_qr_downdate(idx)

        # Reduce active dimension.
        self.k -= 1
        self.K[: self.k, : self.k] = K_new
        self.beta_S[: self.k] = beta_new

        # Recompute residual correlations/variances from the downdated basis.
        self._refresh_residuals_from_basis()


__all__ = ["IncrementalSolver"]
