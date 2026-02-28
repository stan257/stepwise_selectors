"""Fast grouped selection routines (default API)."""

from __future__ import annotations

import inspect
from numbers import Integral
from typing import Iterable, Sequence

import numpy as np

from .constants import ABS_TOL
from .criteria import AICCriterion, SelectionCriterion
from .definitions import GramData
from .fast_routines import FastForwardState
from .state import GroupedSelectionState


def _normalize_group_feature_index(feat: int) -> int:
    if isinstance(feat, bool) or not isinstance(feat, Integral):
        raise TypeError("Group feature indices must be integers.")
    idx = int(feat)
    if idx < 0:
        raise ValueError("Group feature indices must be non-negative.")
    return idx


def _validate_group_feature_bounds(groups: Sequence[Sequence[int]], p: int) -> None:
    for group_idx, group in enumerate(groups):
        for feat in group:
            if feat >= p:
                raise ValueError(
                    f"Group {group_idx} feature index {feat} is out of range for p={p}."
                )


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


def _build_grouped_state(
    *,
    data: GramData,
    groups: Sequence[Sequence[int]],
    active_groups: list[int],
    fast_state: FastForwardState,
) -> GroupedSelectionState:
    active_group_list = list(active_groups)
    active_set = _flatten_group_indices(active_group_list, groups)
    return GroupedSelectionState(
        data=data,
        groups=tuple(tuple(int(feat) for feat in group) for group in groups),
        active_groups=active_group_list,
        active_set=active_set,
        beta=_beta_from_fast_state(fast_state, data.gram.shape[0]),
        rss=float(fast_state.rss),
    )


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
        normalized_groups = []
        for group in groups:
            normalized_groups.append(tuple(_normalize_group_feature_index(f) for f in group))
        self.groups = normalized_groups
        seen = set()
        for g in self.groups:
            current = set()
            duplicates = set()
            for feat in g:
                if feat in current:
                    duplicates.add(feat)
                current.add(feat)
            if duplicates:
                raise ValueError(
                    f"Each group must contain unique features; group {g} repeats {sorted(duplicates)}."
                )
            overlap = seen & current
            if overlap:
                raise ValueError(
                    f"Groups must be disjoint; features {overlap} appear in multiple groups."
                )
            seen.update(current)
        self.num_groups = len(self.groups)
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
        return self.criterion_cls(**params)


class FastGroupForwardSelection(BaseGroupedSelection):
    """Greedy forward selection over groups using fast Gram-only updates."""

    def fit(
        self, *, data: GramData, max_steps: int | None = None
    ) -> GroupedSelectionState:
        _validate_group_feature_bounds(self.groups, data.gram.shape[0])
        criterion = self._init_criterion(data)
        active: list[int] = []
        fast_state = FastForwardState.create(data, self.tol)
        initial = float(np.asarray(criterion.evaluate(fast_state.rss, fast_state.k)))
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
                    np.asarray(criterion.evaluate(candidate_state.rss, candidate_state.k))
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

        return _build_grouped_state(
            data=data,
            groups=self.groups,
            active_groups=active,
            fast_state=fast_state,
        )


class FastGroupBackwardSelection(BaseGroupedSelection):
    """Greedy backward selection over groups using fast Gram-only updates."""

    def fit(
        self, *, data: GramData, max_steps: int | None = None
    ) -> GroupedSelectionState:
        _validate_group_feature_bounds(self.groups, data.gram.shape[0])
        criterion = self._init_criterion(data)
        active = list(range(self.num_groups))
        full_idx = _flatten_group_indices(active, self.groups)
        fast_state = FastForwardState.from_active_set(data, full_idx, self.tol)
        initial = float(
            np.asarray(criterion.evaluate(fast_state.rss, fast_state.k))
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
                    np.asarray(criterion.evaluate(candidate_state.rss, candidate_state.k))
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

        return _build_grouped_state(
            data=data,
            groups=self.groups,
            active_groups=active,
            fast_state=fast_state,
        )


# Default grouped API now points to the fast implementations.
GroupForwardSelection = FastGroupForwardSelection
GroupBackwardSelection = FastGroupBackwardSelection


__all__ = [
    "GroupedSelectionState",
    "FastGroupForwardSelection",
    "FastGroupBackwardSelection",
    "GroupForwardSelection",
    "GroupBackwardSelection",
]
