"""Fast forward selection using Gram-only orthogonal updates."""

from __future__ import annotations

import inspect

import numpy as np

from dataclasses import dataclass

from .constants import ABS_TOL
from .criteria import AICCriterion, SelectionCriterion
from .definitions import GramData
from .state import SelectionState


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
            k=0,
        )

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
        new_Z[: self.k] = self.Z[: self.k]
        new_qy[: self.k] = self.qy[: self.k]
        self.Z = new_Z
        self.qy = new_qy

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
        if self.active_mask[feat_idx]:
            raise ValueError("Feature is already active.")
        denom = self.v[feat_idx]
        if denom <= self.tol:
            raise ValueError("Candidate variance is too small for a stable update.")
        denom = np.sqrt(denom)

        if self.k == 0:
            proj = 0.0
            qy_proj = 0.0
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

        self._ensure_capacity(self.k)
        self.Z[self.k, :] = z_new
        self.qy[self.k] = qy_new
        self.k += 1
        return self.rss


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
                scored = beam.state.candidate_scores()
                if scored is None:
                    continue
                cand_idx, rss_new = scored
                crit_scores = np.asarray(
                    beam.criterion.evaluate(rss_new, beam.state.k + 1)
                )
                order = np.argsort(crit_scores)
                for idx in order[: self.beam_width]:
                    candidate_score = float(crit_scores[idx])
                    if not beam.criterion.is_improvement(candidate_score, beam.score):
                        continue
                    feat_idx = int(cand_idx[idx])
                    child_state = beam.state.clone()
                    child_state.apply_forward(feat_idx)
                    child_criterion = beam.criterion.clone()
                    child_criterion.update_current(candidate_score)
                    candidates.append(
                        FastBeam(child_state, child_criterion, candidate_score)
                    )

            if not candidates:
                break
            beams = _fast_beam_prune(candidates, self.beam_width)
            steps += 1

        best = min(beams, key=lambda b: b.score)
        result = state if state is not None else SelectionState(data)
        result.init_from_active_set(best.state.active_set)
        return result
