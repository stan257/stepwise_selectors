"""Fast forward selection using Gram-only orthogonal updates."""

from __future__ import annotations

import inspect

import numpy as np

from .constants import ABS_TOL
from .criteria import AICCriterion, SelectionCriterion
from .definitions import GramData
from .state import SelectionState


class FastForwardSelection:
    """Forward selection with O(k·p) per-step candidate updates.

    This routine maintains residual correlations and variances for all features
    using an orthogonal basis derived from Gram data only. It avoids the
    O(k^2·p) cost of recomputing projections from scratch at each step.
    """

    def __init__(
        self,
        *,
        tol: float = ABS_TOL,
        criterion_cls=None,
        criterion_kwargs=None,
    ):
        self.tol = tol
        self.criterion_cls = criterion_cls or AICCriterion
        self.criterion_kwargs = dict(criterion_kwargs or {})

    def _init_criterion(self, data: GramData) -> SelectionCriterion:
        params = dict(self.criterion_kwargs)
        init_params = inspect.signature(self.criterion_cls.__init__).parameters
        if "n_samples" in init_params and "n_samples" not in params:
            params["n_samples"] = data.n_samples
        criterion = self.criterion_cls(**params)
        initial = float(np.asarray(criterion.evaluate(data.y_norm, 0)))
        criterion.update_current(initial)
        return criterion

    @staticmethod
    def _ensure_capacity(buf: np.ndarray, needed: int) -> np.ndarray:
        if buf.shape[0] > needed:
            return buf
        new_cap = max(1, buf.shape[0])
        while new_cap <= needed:
            new_cap *= 2
        new_buf = np.empty((new_cap, buf.shape[1]), dtype=buf.dtype)
        new_buf[: buf.shape[0]] = buf
        return new_buf

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError("FastForwardSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        gram = data.gram
        cov = data.cov

        p = gram.shape[0]
        r = cov.astype(float, copy=True)
        v = np.diag(gram).astype(float, copy=True)
        rss = float(data.y_norm)

        active_set: list[int] = []
        active_mask = np.zeros(p, dtype=bool)

        # Allocate orthogonal projections in a growable buffer to avoid reallocs.
        cap = min(max_steps or 64, p)
        if cap <= 0:
            cap = min(64, p)
        Z = np.empty((cap, p), dtype=float)
        qy = np.empty(cap, dtype=float)
        k = 0

        while max_steps is None or k < max_steps:
            valid = (~active_mask) & (v > self.tol)
            if not np.any(valid):
                break

            candidates = np.nonzero(valid)[0]
            r_valid = r[candidates]
            v_valid = v[candidates]
            rss_new = rss - (r_valid**2) / v_valid
            rss_new = np.clip(rss_new, self.tol, None)

            best_idx, best_score = criterion.best_candidate(rss_new, k + 1)
            if not criterion.is_improvement(best_score):
                break

            feat_idx = int(candidates[best_idx])
            denom = v[feat_idx]
            if denom <= self.tol:
                break
            denom = np.sqrt(denom)

            if k == 0:
                proj = 0.0
                qy_proj = 0.0
            else:
                # Project candidate onto existing orthonormal directions.
                z_col = Z[:k, feat_idx]
                proj = z_col @ Z[:k, :]
                qy_proj = z_col @ qy[:k]

            z_new = (gram[:, feat_idx] - proj) / denom
            qy_new = (cov[feat_idx] - qy_proj) / denom

            # Update residual correlations/variances for all features in O(p).
            r -= z_new * qy_new
            v -= z_new * z_new
            r[feat_idx] = 0.0
            v[feat_idx] = 0.0

            rss -= qy_new * qy_new
            active_set.append(feat_idx)
            active_mask[feat_idx] = True

            if k >= Z.shape[0]:
                Z = self._ensure_capacity(Z, k)
                qy = np.pad(qy, (0, Z.shape[0] - qy.shape[0]))

            Z[k, :] = z_new
            qy[k] = qy_new
            k += 1
            criterion.update_current(best_score)

        result = state if state is not None else SelectionState(data)
        result.init_from_active_set(active_set)
        return result
