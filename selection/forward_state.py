"""State container and QR/Gram updates for fast selection routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .definitions import GramData


@dataclass
class ForwardState:
    data: GramData
    tol: float
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
    def create(cls, data: GramData, tol: float) -> "ForwardState":
        p = data.gram.shape[0]
        cap = min(64, p)
        return cls(
            data=data,
            tol=tol,
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
        cls, data: GramData, active_set: list[int], tol: float
    ) -> "ForwardState":
        if not active_set:
            return cls.create(data, tol)
        p = data.gram.shape[0]
        k = len(active_set)
        state = cls.create(data, tol)
        state._ensure_capacity(k - 1)
        state.active_set = list(active_set)
        state.active_mask[:] = False
        state.active_mask[np.array(active_set, dtype=int)] = True
        state.k = k

        idx = np.array(active_set, dtype=int)
        G_S = data.gram[np.ix_(idx, idx)]
        # Use Cholesky to build the QR factor (R) for the active set.
        L = np.linalg.cholesky(G_S)
        R = L.T
        state.R[:k, :k] = R

        G_Sfull = data.gram[np.ix_(idx, np.arange(p))]
        # Z = Q^T X = R^{-T} G_Sfull
        state.Z[:k, :] = np.linalg.solve(R.T, G_Sfull)
        cov_S = data.cov[idx]
        state.qy[:k] = np.linalg.solve(R.T, cov_S)

        state.r = data.cov - state.Z[:k, :].T @ state.qy[:k]
        state.v = np.diag(data.gram) - np.sum(state.Z[:k, :] ** 2, axis=0)
        state.rss = float(data.y_norm - np.dot(state.qy[:k], state.qy[:k]))

        invR = np.linalg.solve(R, np.eye(k))
        state.K[:k, :k] = invR @ invR.T
        state.beta_S[:k] = np.linalg.solve(R, state.qy[:k])
        return state

    def clone(self) -> "ForwardState":
        return ForwardState(
            data=self.data,
            tol=self.tol,
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
        rss_new = np.clip(rss_new, self.tol, None)
        return candidates, rss_new

    def apply_forward(self, feat_idx: int) -> float:
        """Apply a forward step while updating QR/Gram state in O(p)."""
        if self.active_mask[feat_idx]:
            raise ValueError("Feature is already active.")
        resid_var = self.v[feat_idx]
        if resid_var <= self.tol:
            raise ValueError("Candidate variance is too small for a stable update.")
        resid_corr = self.r[feat_idx]
        denom = np.sqrt(resid_var)

        if self.k == 0:
            proj = 0.0
            qy_proj = 0.0
            z_col = np.zeros(0)
        else:
            # Project the candidate onto the existing orthonormal directions.
            z_col = self.Z[: self.k, feat_idx]
            proj = z_col @ self.Z[: self.k, :]
            qy_proj = z_col @ self.qy[: self.k]

        z_new = (self.data.gram[:, feat_idx] - proj) / denom
        qy_new = (self.data.cov[feat_idx] - qy_proj) / denom

        # Update residual correlations/variances for all features in O(p).
        self.r -= z_new * qy_new
        self.v -= z_new * z_new
        self.r[feat_idx] = 0.0
        self.v[feat_idx] = 0.0
        self.rss -= qy_new * qy_new

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

        # Maintain K and beta_S for cheap backward scoring.
        beta_j = resid_corr / resid_var
        if self.k == 0:
            self.beta_S[0] = beta_j
            self.K[0, 0] = 1.0 / resid_var
        else:
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

        self.k += 1
        return self.rss

    def backward_scores(self) -> np.ndarray | None:
        if self.k == 0:
            return None
        diag_K = np.diag(self.K[: self.k, : self.k])
        rss_new = np.full(self.k, np.inf, dtype=float)
        safe = diag_K > self.tol
        if np.any(safe):
            rss_new[safe] = self.rss + (self.beta_S[: self.k][safe] ** 2) / diag_K[safe]
        return rss_new

    def apply_backward(self, idx: int) -> None:
        """Remove one active feature via inverse-Gram and QR downdates."""
        if not (0 <= idx < self.k):
            raise IndexError("Backward index out of range.")

        # Update K and beta_S via the standard inverse-Gram downdate.
        Kk = self.K[: self.k, : self.k]
        beta_k = self.beta_S[: self.k]
        k_22 = Kk[idx, idx]
        if k_22 <= self.tol:
            raise ValueError("Backward update failed due to near-singular pivot.")
        removed = self.active_set.pop(idx)
        self.active_mask[removed] = False
        idx_keep = np.delete(np.arange(self.k), idx)
        K_11 = Kk[np.ix_(idx_keep, idx_keep)]
        k_12 = Kk[idx_keep, idx]
        K_new = K_11 - np.outer(k_12, k_12) / k_22

        beta_removed = beta_k[idx]
        beta_keep = np.delete(beta_k, idx)
        beta_new = beta_keep - (beta_removed / k_22) * k_12 if beta_keep.size else np.zeros(0)

        # Apply QR downdate to Z, qy, and R using Givens rotations.
        if idx < self.k - 1:
            self.R[: self.k, idx : self.k - 1] = self.R[: self.k, idx + 1 : self.k]
            for i in range(idx, self.k - 1):
                a = self.R[i, i]
                b = self.R[i + 1, i]
                if abs(b) <= self.tol:
                    continue
                r = np.hypot(a, b)
                c = a / r
                s = b / r

                # Rotate rows i and i+1 of R to restore upper-triangular form.
                Ri = self.R[i, i : self.k - 1].copy()
                Rj = self.R[i + 1, i : self.k - 1].copy()
                self.R[i, i : self.k - 1] = c * Ri + s * Rj
                self.R[i + 1, i : self.k - 1] = -s * Ri + c * Rj

                # Apply the same rotation to Z and qy (Q^T X and Q^T y).
                Zi = self.Z[i, :].copy()
                Zj = self.Z[i + 1, :].copy()
                self.Z[i, :] = c * Zi + s * Zj
                self.Z[i + 1, :] = -s * Zi + c * Zj

                qi = self.qy[i]
                qj = self.qy[i + 1]
                self.qy[i] = c * qi + s * qj
                self.qy[i + 1] = -s * qi + c * qj

        # Reduce active dimension.
        self.k -= 1
        self.K[: self.k, : self.k] = K_new
        self.beta_S[: self.k] = beta_new

        # Recompute residual correlations/variances from the downdated basis.
        if self.k == 0:
            self.r = self.data.cov.astype(float, copy=True)
            self.v = np.diag(self.data.gram).astype(float, copy=True)
            self.rss = float(self.data.y_norm)
        else:
            self.r = self.data.cov - self.Z[: self.k, :].T @ self.qy[: self.k]
            self.v = np.diag(self.data.gram) - np.sum(
                self.Z[: self.k, :] ** 2, axis=0
            )
            self.rss = float(self.data.y_norm - np.dot(self.qy[: self.k], self.qy[: self.k]))


__all__ = ["ForwardState"]
