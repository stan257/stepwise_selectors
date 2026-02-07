"""Fast grouped selection routines (default API)."""

from __future__ import annotations

import inspect
from typing import Iterable, Sequence

import numpy as np

from .constants import ABS_TOL
from .criteria import AICCriterion, SelectionCriterion
from .definitions import GramData
from .fast_routines import FastForwardState


def _flatten_group_indices(groups: Iterable[int], group_map: Sequence[Sequence[int]]):
    idx: list[int] = []
    for g in groups:
        idx.extend(group_map[g])
    return sorted(idx)


def _beta_from_fast_state(state: FastForwardState, p: int) -> np.ndarray:
    beta = np.zeros(p, dtype=float)
    if state.k:
        idx = np.array(state.active_set, dtype=int)
        beta[idx] = state.beta_S[: state.k]
    return beta


def _apply_group_forward(state: FastForwardState, group: Sequence[int]) -> None:
    # Apply group members in a stable, deterministic order.
    for feat_idx in group:
        if state.active_mask[int(feat_idx)]:
            raise ValueError("Group feature already active.")
        state.apply_forward(int(feat_idx))


def _apply_group_backward(state: FastForwardState, group: Sequence[int]) -> None:
    # Remove in descending index order to avoid shifting positions mid-loop.
    pos_map = {feat: pos for pos, feat in enumerate(state.active_set)}
    positions = sorted((pos_map[int(feat)] for feat in group), reverse=True)
    for idx in positions:
        state.apply_backward(idx)


class BaseGroupedSelection:
    """Shared machinery for grouped greedy selection."""

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        *,
        tol: float = ABS_TOL,
        criterion_cls=None,
        criterion_kwargs=None,
    ):
        self.groups = [tuple(g) for g in groups]
        self.num_groups = len(self.groups)
        self.tol = tol
        self.criterion_cls = criterion_cls or AICCriterion
        self.criterion_kwargs = dict(criterion_kwargs or {})

    def _init_criterion(self, data: GramData) -> SelectionCriterion:
        params = dict(self.criterion_kwargs)
        init_params = inspect.signature(self.criterion_cls.__init__).parameters
        if "n_samples" in init_params and "n_samples" not in params:
            params["n_samples"] = data.n_samples
        return self.criterion_cls(**params)


class FastGroupForwardSelection(BaseGroupedSelection):
    """Greedy forward selection over groups using fast Gram-only updates."""

    def fit(self, *, data: GramData, max_steps: int | None = None):
        criterion = self._init_criterion(data)
        active: list[int] = []
        fast_state = FastForwardState.create(data, self.tol)
        initial = float(np.asarray(criterion.evaluate(fast_state.rss, 0)))
        criterion.update_current(initial)

        steps = 0
        while max_steps is None or steps < max_steps:
            best_group = None
            best_candidate_score = None
            best_candidate_state = None

            for g in range(self.num_groups):
                if g in active:
                    continue
                candidate_state = fast_state.clone()
                try:
                    _apply_group_forward(candidate_state, self.groups[g])
                except ValueError:
                    continue
                score_cand = float(
                    np.asarray(criterion.evaluate(candidate_state.rss, len(active) + 1))
                )
                if not criterion.is_improvement(score_cand):
                    continue
                if best_candidate_score is None or criterion.is_improvement(
                    score_cand, best_candidate_score
                ):
                    best_group = g
                    best_candidate_score = score_cand
                    best_candidate_state = candidate_state

            if best_group is None or best_candidate_state is None:
                break

            active.append(best_group)
            fast_state = best_candidate_state
            criterion.update_current(best_candidate_score)
            steps += 1

        result = type("GroupedState", (), {})()
        result.active_groups = list(active)
        result.beta = _beta_from_fast_state(fast_state, data.gram.shape[0])
        result.rss = float(fast_state.rss)
        return result


class FastGroupBackwardSelection(BaseGroupedSelection):
    """Greedy backward selection over groups using fast Gram-only updates."""

    def fit(self, *, data: GramData, max_steps: int | None = None):
        criterion = self._init_criterion(data)
        active = list(range(self.num_groups))
        full_idx = _flatten_group_indices(active, self.groups)
        fast_state = FastForwardState.from_active_set(data, full_idx, self.tol)
        initial = float(
            np.asarray(criterion.evaluate(fast_state.rss, len(active)))
        )
        criterion.update_current(initial)

        steps = 0
        while active and (max_steps is None or steps < max_steps):
            best_drop = None
            best_candidate_score = None
            best_candidate_state = None

            for g in active:
                candidate_state = fast_state.clone()
                try:
                    _apply_group_backward(candidate_state, self.groups[g])
                except ValueError:
                    continue
                score_cand = float(
                    np.asarray(criterion.evaluate(candidate_state.rss, len(active) - 1))
                )
                if not criterion.is_improvement(score_cand):
                    continue
                if best_candidate_score is None or criterion.is_improvement(
                    score_cand, best_candidate_score
                ):
                    best_drop = g
                    best_candidate_score = score_cand
                    best_candidate_state = candidate_state

            if best_drop is None or best_candidate_state is None:
                break

            active.remove(best_drop)
            fast_state = best_candidate_state
            criterion.update_current(best_candidate_score)
            steps += 1

        result = type("GroupedState", (), {})()
        result.active_groups = list(active)
        result.beta = _beta_from_fast_state(fast_state, data.gram.shape[0])
        result.rss = float(fast_state.rss)
        return result


# Default grouped API now points to the fast implementations.
GroupForwardSelection = FastGroupForwardSelection
GroupBackwardSelection = FastGroupBackwardSelection


__all__ = [
    "FastGroupForwardSelection",
    "FastGroupBackwardSelection",
    "GroupForwardSelection",
    "GroupBackwardSelection",
]
