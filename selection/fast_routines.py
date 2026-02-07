"""Fast forward selection using Gram-only orthogonal updates."""

from __future__ import annotations

import inspect

import numpy as np

from dataclasses import dataclass

from .constants import ABS_TOL
from .criteria import AICCriterion, SelectionCriterion
from .definitions import CrossValGramData, GramData
from .state import CrossValSelectionState, SelectionState
from .topk import topk_indices


@dataclass
class FastForwardState:
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
    def create(cls, data: GramData, tol: float) -> "FastForwardState":
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
    ) -> "FastForwardState":
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

    def clone(self) -> "FastForwardState":
        return FastForwardState(
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
        valid_rss = rss_new > -self.tol
        if not np.any(valid_rss):
            return None
        return candidates[valid_rss], rss_new[valid_rss]

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
        safe = np.abs(diag_K) > self.tol
        if np.any(safe):
            rss_new[safe] = self.rss + (self.beta_S[: self.k][safe] ** 2) / diag_K[safe]
        return rss_new

    def apply_backward(self, idx: int) -> None:
        """Remove one active feature via inverse-Gram and QR downdates."""
        if not (0 <= idx < self.k):
            raise IndexError("Backward index out of range.")
        removed = self.active_set.pop(idx)
        self.active_mask[removed] = False

        # Update K and beta_S via the standard inverse-Gram downdate.
        Kk = self.K[: self.k, : self.k]
        beta_k = self.beta_S[: self.k]
        k_22 = Kk[idx, idx]
        if abs(k_22) <= self.tol:
            raise ValueError("Backward update failed due to near-singular pivot.")
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
        if "p" in init_params and "p" not in params:
            params["p"] = data.gram.shape[0]
        criterion = self.criterion_cls(**params)
        initial = float(np.asarray(criterion.evaluate(data.y_norm, 0)))
        criterion.update_current(initial)
        return criterion

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
        fast_state = FastForwardState.create(data, self.tol)

        while max_steps is None or fast_state.k < max_steps:
            scored = fast_state.candidate_scores()
            if scored is None:
                break
            candidates, rss_new = scored
            best_idx, best_score = criterion.best_candidate(
                rss_new, fast_state.k + 1
            )
            if not criterion.is_improvement(best_score):
                break
            feat_idx = int(candidates[best_idx])
            fast_state.apply_forward(feat_idx)
            criterion.update_current(best_score)

        result = state if state is not None else SelectionState(data)
        result.init_from_active_set(fast_state.active_set)
        return result


class FastBackwardSelection(FastForwardSelection):
    """Backward selection using fast Gram-only updates."""

    def __init__(self, *, allow_worse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.allow_worse = allow_worse

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError("FastBackwardSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        full_active = list(range(data.gram.shape[0]))
        fast_state = FastForwardState.from_active_set(data, full_active, self.tol)
        initial = float(
            np.asarray(criterion.evaluate(fast_state.rss, len(fast_state.active_set)))
        )
        criterion.update_current(initial)

        steps = 0
        while fast_state.k and (max_steps is None or steps < max_steps):
            rss_values = fast_state.backward_scores()
            if rss_values is None:
                break
            best_idx, best_score = criterion.best_candidate(
                rss_values, fast_state.k - 1
            )
            if not self.allow_worse and not criterion.is_improvement(best_score):
                break
            fast_state.apply_backward(best_idx)
            criterion.update_current(best_score)
            steps += 1

        result = state if state is not None else SelectionState(data)
        result.init_from_active_set(fast_state.active_set)
        return result


class FastMixedSelection(FastForwardSelection):
    """Mixed forward/backward selection with fast updates."""

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_forward_steps: int | None = None,
        max_total_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError("FastMixedSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        fast_state = FastForwardState.create(data, self.tol)

        forward_steps = 0
        total_steps = 0
        while True:
            if max_total_steps is not None and total_steps >= max_total_steps:
                break
            scored = fast_state.candidate_scores()
            if scored is None:
                break
            candidates, rss_new = scored
            best_idx, best_score = criterion.best_candidate(
                rss_new, fast_state.k + 1
            )
            if not criterion.is_improvement(best_score):
                break
            feat_idx = int(candidates[best_idx])
            fast_state.apply_forward(feat_idx)
            criterion.update_current(best_score)
            forward_steps += 1
            total_steps += 1

            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break

            while True:
                if max_total_steps is not None and total_steps >= max_total_steps:
                    break
                rss_values = fast_state.backward_scores()
                if rss_values is None:
                    break
                best_idx, best_score = criterion.best_candidate(
                    rss_values, fast_state.k - 1
                )
                if not criterion.is_improvement(best_score):
                    break
                fast_state.apply_backward(best_idx)
                criterion.update_current(best_score)
                total_steps += 1

        result = state if state is not None else SelectionState(data)
        result.init_from_active_set(fast_state.active_set)
        return result


@dataclass
class FastBeam:
    state: FastForwardState
    criterion: SelectionCriterion
    score: float
    _signature: int = 0

    def __post_init__(self) -> None:
        sig = 0
        for idx in self.state.active_set:
            sig |= 1 << int(idx)
        self._signature = sig

    @property
    def signature(self) -> int:
        return self._signature


def _fast_beam_prune(beams: list[FastBeam], beam_limit: int) -> list[FastBeam]:
    seen = set()
    result: list[FastBeam] = []
    for beam in sorted(beams, key=lambda b: b.score):
        sig = beam.signature
        if sig in seen:
            continue
        seen.add(sig)
        result.append(beam)
        if len(result) >= beam_limit:
            break
    return result


def _fast_beam_forward_children(beam: FastBeam, beam_width: int) -> list[FastBeam]:
    scored = beam.state.candidate_scores()
    if scored is None:
        return []
    cand_idx, rss_new = scored
    crit_scores = np.asarray(beam.criterion.evaluate(rss_new, beam.state.k + 1))
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)
    children: list[FastBeam] = []
    for idx in order:
        candidate_score = float(crit_scores[idx])
        if not beam.criterion.is_improvement(candidate_score, beam.score):
            continue
        feat_idx = int(cand_idx[idx])
        child_state = beam.state.clone()
        child_state.apply_forward(feat_idx)
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(FastBeam(child_state, child_criterion, candidate_score))
    return children


def _fast_beam_backward_children(
    beam: FastBeam, beam_width: int, *, allow_worse: bool = False
) -> list[FastBeam]:
    rss_values = beam.state.backward_scores()
    if rss_values is None or not len(rss_values):
        return []
    crit_scores = np.asarray(
        beam.criterion.evaluate(rss_values, max(beam.state.k - 1, 0))
    )
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)
    children: list[FastBeam] = []
    for idx in order:
        candidate_score = float(crit_scores[idx])
        if not allow_worse and not beam.criterion.is_improvement(
            candidate_score, beam.score
        ):
            continue
        try:
            child_state = beam.state.clone()
            child_state.apply_backward(idx)
        except Exception:
            continue
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(FastBeam(child_state, child_criterion, candidate_score))
    return children


def _fast_beam_best_backward_child(beam: FastBeam) -> FastBeam | None:
    for child in _fast_beam_backward_children(beam, 1, allow_worse=False):
        if beam.criterion.is_improvement(child.score, beam.score):
            return child
    return None


class FastBeamForwardSelection(FastForwardSelection):
    """Beam-search forward selection using fast Gram-only updates."""

    def __init__(self, *, beam_width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError("FastBeamForwardSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        initial = FastBeam(FastForwardState.create(data, self.tol), criterion, criterion.current_value)
        beams = [initial]

        steps = 0
        while beams and (max_steps is None or steps < max_steps):
            candidates: list[FastBeam] = []
            for beam in beams:
                candidates.extend(_fast_beam_forward_children(beam, self.beam_width))

            if not candidates:
                break
            beams = _fast_beam_prune(candidates, self.beam_width)
            steps += 1

        best = min(beams, key=lambda b: b.score)
        result = state if state is not None else SelectionState(data)
        result.init_from_active_set(best.state.active_set)
        return result


class FastBeamBackwardSelection(FastForwardSelection):
    """Beam-search backward selection using fast Gram-only updates."""

    def __init__(self, *, beam_width: int = 1, allow_worse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))
        self.allow_worse = allow_worse

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError("FastBeamBackwardSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        full_active = list(range(data.gram.shape[0]))
        initial_state = FastForwardState.from_active_set(data, full_active, self.tol)
        initial_score = float(
            np.asarray(criterion.evaluate(initial_state.rss, initial_state.k))
        )
        criterion.update_current(initial_score)
        beams = [FastBeam(initial_state, criterion, criterion.current_value)]

        steps = 0
        while beams and (max_steps is None or steps < max_steps):
            candidates: list[FastBeam] = []
            for beam in beams:
                candidates.extend(
                    _fast_beam_backward_children(
                        beam, self.beam_width, allow_worse=self.allow_worse
                    )
                )
            if not candidates:
                break
            beams = _fast_beam_prune(candidates, self.beam_width)
            steps += 1

        best = min(beams, key=lambda b: b.score)
        result = state if state is not None else SelectionState(data)
        result.init_from_active_set(best.state.active_set)
        return result


class FastBeamMixedSelection(FastForwardSelection):
    """Beam-search mixed selection using fast Gram-only updates."""

    def __init__(self, *, beam_width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_forward_steps: int | None = None,
        max_total_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError("FastBeamMixedSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        initial = FastBeam(FastForwardState.create(data, self.tol), criterion, criterion.current_value)
        beams = [initial]
        best = initial

        forward_steps = 0
        total_ops = 0
        while True:
            if max_total_steps is not None and total_ops >= max_total_steps:
                break
            candidates: list[FastBeam] = []
            for beam in beams:
                candidates.extend(_fast_beam_forward_children(beam, self.beam_width))
            if not candidates:
                break
            beams = _fast_beam_prune(candidates, self.beam_width)
            best = min(beams, key=lambda b: b.score)
            forward_steps += 1
            total_ops += len(beams)
            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break

            new_beams: list[FastBeam] = []
            for beam in beams:
                while True:
                    if max_total_steps is not None and total_ops >= max_total_steps:
                        break
                    improved = _fast_beam_best_backward_child(beam)
                    if improved is None:
                        break
                    beam = improved
                    total_ops += 1
                best = min([best, beam], key=lambda b: b.score)
                new_beams.append(beam)
            beams = new_beams

        result = state if state is not None else SelectionState(data)
        result.init_from_active_set(best.state.active_set)
        return result


def _fast_cv_rss(fast_states: list[FastForwardState], data: CrossValGramData) -> float:
    """Compute summed validation RSS for the shared active set across folds."""
    if not fast_states or not fast_states[0].active_set:
        return float(np.sum(data.y_norm_folds))
    idx = np.array(fast_states[0].active_set, dtype=int)
    rss_sum = 0.0
    for fold_idx, fast_state in enumerate(fast_states):
        beta_S = fast_state.beta_S[: fast_state.k]
        G_val = data.gram_folds[fold_idx]
        c_val = data.cov_folds[fold_idx]
        y_norm_val = data.y_norm_folds[fold_idx]
        G_val_SS = G_val[np.ix_(idx, idx)]
        c_val_S = c_val[idx]
        rss_k = (
            y_norm_val
            - 2.0 * float(beta_S @ c_val_S)
            + float(beta_S @ (G_val_SS @ beta_S))
        )
        rss_sum += rss_k
    return float(rss_sum)


@dataclass
class FastCVBeam:
    states: list[FastForwardState]
    criterion: SelectionCriterion
    score: float
    _signature: int = 0

    def __post_init__(self) -> None:
        sig = 0
        for idx in self.states[0].active_set:
            sig |= 1 << int(idx)
        self._signature = sig

    @property
    def signature(self) -> int:
        return self._signature


def _fast_cv_beam_prune(
    beams: list[FastCVBeam], beam_limit: int
) -> list[FastCVBeam]:
    seen = set()
    result: list[FastCVBeam] = []
    for beam in sorted(beams, key=lambda b: b.score):
        sig = beam.signature
        if sig in seen:
            continue
        seen.add(sig)
        result.append(beam)
        if len(result) >= beam_limit:
            break
    return result


def _fast_cv_beam_forward_children(
    beam: FastCVBeam,
    beam_width: int,
    data: CrossValGramData,
    tol: float,
) -> list[FastCVBeam]:
    scored = _fast_cv_forward_scores(beam.states, data, tol)
    if scored is None:
        return []
    candidates, aggregated = scored
    crit_scores = np.asarray(
        beam.criterion.evaluate(aggregated, len(beam.states[0].active_set) + 1)
    )
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)
    children: list[FastCVBeam] = []
    for idx in order:
        candidate_score = float(crit_scores[idx])
        if not beam.criterion.is_improvement(candidate_score, beam.score):
            continue
        feat_idx = int(candidates[idx])
        child_states = [state.clone() for state in beam.states]
        for state_k in child_states:
            state_k.apply_forward(feat_idx)
        # Rebuild per-fold states to keep QR/K updates numerically stable.
        child_states = _rebuild_fast_states(data, child_states[0].active_set, tol)
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(FastCVBeam(child_states, child_criterion, candidate_score))
    return children


def _fast_cv_beam_backward_children(
    beam: FastCVBeam,
    beam_width: int,
    data: CrossValGramData,
    tol: float,
    *,
    allow_worse: bool = False,
) -> list[FastCVBeam]:
    aggregated = _fast_cv_backward_scores(beam.states, data, tol)
    if aggregated is None or not len(aggregated):
        return []
    crit_scores = np.asarray(
        beam.criterion.evaluate(aggregated, max(len(beam.states[0].active_set) - 1, 0))
    )
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)
    children: list[FastCVBeam] = []
    for idx in order:
        candidate_score = float(crit_scores[idx])
        if not allow_worse and not beam.criterion.is_improvement(
            candidate_score, beam.score
        ):
            continue
        try:
            child_states = [state.clone() for state in beam.states]
            for state_k in child_states:
                state_k.apply_backward(idx)
            # Rebuild per-fold states to keep QR/K updates numerically stable.
            child_states = _rebuild_fast_states(data, child_states[0].active_set, tol)
        except ValueError:
            continue
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(FastCVBeam(child_states, child_criterion, candidate_score))
    return children


def _fast_cv_beam_best_backward_child(
    beam: FastCVBeam, data: CrossValGramData, tol: float
) -> FastCVBeam | None:
    for child in _fast_cv_beam_backward_children(
        beam, 1, data, tol, allow_worse=False
    ):
        if beam.criterion.is_improvement(child.score, beam.score):
            return child
    return None


def _rebuild_fast_states(
    data: CrossValGramData, active_set: list[int], tol: float
) -> list[FastForwardState]:
    """Rebuild fast states from scratch to limit numerical drift."""
    return [
        FastForwardState.from_active_set(data.train_data_for_fold(k), active_set, tol)
        for k in range(data.n_folds)
    ]


def _build_cv_state_from_active_set(
    data: CrossValGramData, active_set: list[int]
) -> CrossValSelectionState:
    """Materialize a CrossValSelectionState from an active set."""
    cv_state = CrossValSelectionState(data)
    for fold_idx, fold_state in enumerate(cv_state.train_states):
        fold_state.init_from_active_set(active_set)
    cv_state._sync_active_set()
    cv_state.recompute_oos_rss()
    return cv_state


def _fast_cv_forward_scores(
    fast_states: list[FastForwardState], data: CrossValGramData, tol: float
) -> tuple[list[int], np.ndarray] | None:
    """Return common candidates and aggregated (summed) validation RSS.

    Uses Gram-only formulas to score candidates per fold without materializing
    design matrices, then sums fold RSS to match rss_cv scale.
    """
    candidate_lists = []
    for state in fast_states:
        scored = state.candidate_scores()
        if scored is None:
            return None
        candidates, _ = scored
        candidate_lists.append(set(int(c) for c in candidates))

    common = set.intersection(*candidate_lists)
    if not common:
        return None
    candidates = sorted(common)
    num_candidates = len(candidates)
    rss_matrix = np.empty((len(fast_states), num_candidates), dtype=float)

    for fold_idx, state in enumerate(fast_states):
        idx_S = np.array(state.active_set, dtype=int)
        k = state.k
        resid_corr = state.r[candidates]
        resid_var = state.v[candidates]
        beta_j = resid_corr / resid_var

        G_val = data.gram_folds[fold_idx]
        c_val = data.cov_folds[fold_idx]
        y_norm_val = data.y_norm_folds[fold_idx]
        c_val_j = c_val[candidates]
        g_val_jj = G_val[candidates, candidates]

        if k == 0:
            rss = y_norm_val - 2.0 * beta_j * c_val_j + (beta_j**2) * g_val_jj
            rss_matrix[fold_idx] = rss
            continue

        G_val_SS = G_val[np.ix_(idx_S, idx_S)]
        c_val_S = c_val[idx_S]
        g_val_Sc = G_val[np.ix_(idx_S, candidates)]

        g_train_Sc = state.data.gram[np.ix_(idx_S, candidates)]
        proj = state.K[:k, :k] @ g_train_Sc
        beta_S_new = state.beta_S[:k, None] - proj * beta_j[None, :]

        term1 = -2.0 * (beta_S_new * c_val_S[:, None]).sum(axis=0)
        term2 = -2.0 * beta_j * c_val_j
        term3 = (beta_S_new * (G_val_SS @ beta_S_new)).sum(axis=0)
        term4 = 2.0 * beta_j * (g_val_Sc * beta_S_new).sum(axis=0)
        term5 = (beta_j**2) * g_val_jj
        rss_matrix[fold_idx] = y_norm_val + term1 + term2 + term3 + term4 + term5

    # Sum across folds to match rss_cv scale.
    aggregated = np.sum(rss_matrix, axis=0)
    return candidates, aggregated


def _fast_cv_backward_scores(
    fast_states: list[FastForwardState], data: CrossValGramData, tol: float
) -> np.ndarray | None:
    """Compute aggregated CV backward scores using Gram-only downdates."""
    if not fast_states or not fast_states[0].active_set:
        return None
    idx_full = np.array(fast_states[0].active_set, dtype=int)
    k = len(idx_full)
    rss_matrix = np.empty((len(fast_states), k), dtype=float)

    for fold_idx, state in enumerate(fast_states):
        Kk = state.K[:k, :k]
        beta_k = state.beta_S[:k]
        G_val = data.gram_folds[fold_idx]
        c_val = data.cov_folds[fold_idx]
        y_norm_val = data.y_norm_folds[fold_idx]

        for local_idx in range(k):
            k_22 = Kk[local_idx, local_idx]
            if abs(k_22) <= tol:
                rss_matrix[fold_idx, local_idx] = np.inf
                continue
            idx_keep = np.delete(np.arange(k), local_idx)
            beta_removed = beta_k[local_idx]
            k_12 = Kk[idx_keep, local_idx]
            beta_keep = beta_k[idx_keep]
            beta_new = (
                beta_keep - (beta_removed / k_22) * k_12
                if beta_keep.size
                else np.zeros(0)
            )
            if not beta_new.size:
                rss_matrix[fold_idx, local_idx] = y_norm_val
                continue
            idx = idx_full[idx_keep]
            G_val_SS = G_val[np.ix_(idx, idx)]
            c_val_S = c_val[idx]
            rss = (
                y_norm_val
                - 2.0 * float(beta_new @ c_val_S)
                + float(beta_new @ (G_val_SS @ beta_new))
            )
            rss_matrix[fold_idx, local_idx] = rss

    # Sum across folds to match rss_cv scale.
    return np.sum(rss_matrix, axis=0)


class FastCrossValForwardSelection(FastForwardSelection):
    """Cross-validated forward selection using fast training updates."""

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError("FastCrossValForwardSelection does not support warm starts.")

        criterion = self._init_criterion(data.make_full_data())
        fast_states = [
            FastForwardState.create(data.train_data_for_fold(k), self.tol)
            for k in range(data.n_folds)
        ]
        criterion.update_current(
            float(np.asarray(criterion.evaluate(_fast_cv_rss(fast_states, data), 0)))
        )

        steps = 0
        while max_steps is None or steps < max_steps:
            scored = _fast_cv_forward_scores(fast_states, data, self.tol)
            if scored is None:
                break
            candidates, aggregated = scored
            best_idx, best_score = criterion.best_candidate(
                aggregated, len(fast_states[0].active_set) + 1
            )
            if not criterion.is_improvement(best_score):
                break
            feat_idx = candidates[best_idx]
            for state_k in fast_states:
                state_k.apply_forward(feat_idx)
            fast_states = _rebuild_fast_states(
                data, fast_states[0].active_set, self.tol
            )
            criterion.update_current(best_score)
            steps += 1

        return _build_cv_state_from_active_set(data, fast_states[0].active_set)


class FastCrossValBackwardSelection(FastForwardSelection):
    """Cross-validated backward selection using fast training updates."""

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError("FastCrossValBackwardSelection does not support warm starts.")

        criterion = self._init_criterion(data.make_full_data())
        full_active = list(range(data.gram_total.shape[0]))
        fast_states = [
            FastForwardState.from_active_set(data.train_data_for_fold(k), full_active, self.tol)
            for k in range(data.n_folds)
        ]
        criterion.update_current(
            float(
                np.asarray(
                    criterion.evaluate(_fast_cv_rss(fast_states, data), len(full_active))
                )
            )
        )

        steps = 0
        while fast_states[0].k and (max_steps is None or steps < max_steps):
            aggregated = _fast_cv_backward_scores(fast_states, data, self.tol)
            if aggregated is None:
                break
            best_idx, best_score = criterion.best_candidate(
                aggregated, fast_states[0].k - 1
            )
            if not criterion.is_improvement(best_score):
                break
            try:
                for state_k in fast_states:
                    state_k.apply_backward(best_idx)
                fast_states = _rebuild_fast_states(
                    data, fast_states[0].active_set, self.tol
                )
            except ValueError:
                break
            criterion.update_current(best_score)
            steps += 1

        return _build_cv_state_from_active_set(data, fast_states[0].active_set)


class FastCrossValMixedSelection(FastForwardSelection):
    """Cross-validated mixed selection using fast training updates."""

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: CrossValGramData,
        max_forward_steps: int | None = None,
        max_total_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError("FastCrossValMixedSelection does not support warm starts.")

        criterion = self._init_criterion(data.make_full_data())
        fast_states = [
            FastForwardState.create(data.train_data_for_fold(k), self.tol)
            for k in range(data.n_folds)
        ]
        criterion.update_current(
            float(np.asarray(criterion.evaluate(_fast_cv_rss(fast_states, data), 0)))
        )

        forward_steps = 0
        total_steps = 0
        while True:
            if max_total_steps is not None and total_steps >= max_total_steps:
                break
            scored = _fast_cv_forward_scores(fast_states, data, self.tol)
            if scored is None:
                break
            candidates, aggregated = scored
            best_idx, best_score = criterion.best_candidate(
                aggregated, len(fast_states[0].active_set) + 1
            )
            if not criterion.is_improvement(best_score):
                break
            feat_idx = candidates[best_idx]
            for state_k in fast_states:
                state_k.apply_forward(feat_idx)
            fast_states = _rebuild_fast_states(
                data, fast_states[0].active_set, self.tol
            )
            criterion.update_current(best_score)
            forward_steps += 1
            total_steps += 1

            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break

            while True:
                if max_total_steps is not None and total_steps >= max_total_steps:
                    break
                aggregated = _fast_cv_backward_scores(fast_states, data, self.tol)
                if aggregated is None:
                    break
                best_idx, best_score = criterion.best_candidate(
                    aggregated, fast_states[0].k - 1
                )
                if not criterion.is_improvement(best_score):
                    break
                try:
                    for state_k in fast_states:
                        state_k.apply_backward(best_idx)
                    fast_states = _rebuild_fast_states(
                        data, fast_states[0].active_set, self.tol
                    )
                except ValueError:
                    break
                criterion.update_current(best_score)
                total_steps += 1

        return _build_cv_state_from_active_set(data, fast_states[0].active_set)


class FastBeamCrossValForwardSelection(FastForwardSelection):
    """Beam-search forward selection with fast CV scoring."""

    def __init__(self, *, beam_width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError(
                "FastBeamCrossValForwardSelection does not support warm starts."
            )

        criterion = self._init_criterion(data.make_full_data())
        fast_states = [
            FastForwardState.create(data.train_data_for_fold(k), self.tol)
            for k in range(data.n_folds)
        ]
        initial_score = float(
            np.asarray(criterion.evaluate(_fast_cv_rss(fast_states, data), 0))
        )
        criterion.update_current(initial_score)
        beams = [FastCVBeam(fast_states, criterion, criterion.current_value)]

        steps = 0
        while beams and (max_steps is None or steps < max_steps):
            candidates: list[FastCVBeam] = []
            for beam in beams:
                candidates.extend(
                    _fast_cv_beam_forward_children(
                        beam, self.beam_width, data, self.tol
                    )
                )
            if not candidates:
                break
            beams = _fast_cv_beam_prune(candidates, self.beam_width)
            steps += 1

        best = min(beams, key=lambda b: b.score)
        return _build_cv_state_from_active_set(data, best.states[0].active_set)


class FastBeamCrossValBackwardSelection(FastForwardSelection):
    """Beam-search backward selection with fast CV scoring."""

    def __init__(self, *, beam_width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError(
                "FastBeamCrossValBackwardSelection does not support warm starts."
            )

        criterion = self._init_criterion(data.make_full_data())
        full_active = list(range(data.gram_total.shape[0]))
        fast_states = [
            FastForwardState.from_active_set(
                data.train_data_for_fold(k), full_active, self.tol
            )
            for k in range(data.n_folds)
        ]
        initial_score = float(
            np.asarray(criterion.evaluate(_fast_cv_rss(fast_states, data), len(full_active)))
        )
        criterion.update_current(initial_score)
        beams = [FastCVBeam(fast_states, criterion, criterion.current_value)]
        allow_worse = max_steps is not None

        steps = 0
        while beams and (max_steps is None or steps < max_steps):
            candidates: list[FastCVBeam] = []
            for beam in beams:
                candidates.extend(
                    _fast_cv_beam_backward_children(
                        beam,
                        self.beam_width,
                        data,
                        self.tol,
                        allow_worse=allow_worse,
                    )
                )
            if not candidates:
                break
            beams = _fast_cv_beam_prune(candidates, self.beam_width)
            steps += 1

        best = min(beams, key=lambda b: b.score)
        return _build_cv_state_from_active_set(data, best.states[0].active_set)


class FastBeamCrossValMixedSelection(FastForwardSelection):
    """Beam-search mixed selection with fast CV scoring."""

    def __init__(self, *, beam_width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: CrossValGramData,
        max_forward_steps: int | None = None,
        max_total_steps: int | None = None,
    ) -> SelectionState:
        if state is not None and state.active_set:
            raise ValueError(
                "FastBeamCrossValMixedSelection does not support warm starts."
            )

        criterion = self._init_criterion(data.make_full_data())
        fast_states = [
            FastForwardState.create(data.train_data_for_fold(k), self.tol)
            for k in range(data.n_folds)
        ]
        initial_score = float(
            np.asarray(criterion.evaluate(_fast_cv_rss(fast_states, data), 0))
        )
        criterion.update_current(initial_score)
        initial = FastCVBeam(fast_states, criterion, criterion.current_value)
        beams = [initial]
        best = initial

        forward_steps = 0
        total_ops = 0
        while True:
            if max_total_steps is not None and total_ops >= max_total_steps:
                break
            candidates: list[FastCVBeam] = []
            for beam in beams:
                candidates.extend(
                    _fast_cv_beam_forward_children(
                        beam, self.beam_width, data, self.tol
                    )
                )
            if not candidates:
                break
            beams = _fast_cv_beam_prune(candidates, self.beam_width)
            best = min(beams, key=lambda b: b.score)
            forward_steps += 1
            total_ops += len(beams)
            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break

            new_beams: list[FastCVBeam] = []
            for beam in beams:
                while True:
                    if max_total_steps is not None and total_ops >= max_total_steps:
                        break
                    improved = _fast_cv_beam_best_backward_child(beam, data, self.tol)
                    if improved is None:
                        break
                    beam = improved
                    total_ops += 1
                best = min([best, beam], key=lambda b: b.score)
                new_beams.append(beam)
            beams = new_beams

        return _build_cv_state_from_active_set(data, best.states[0].active_set)
