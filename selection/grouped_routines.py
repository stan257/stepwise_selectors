"""Grouped selection routines built on the core Gram-only state updates."""

from __future__ import annotations

from numbers import Integral
from typing import Callable, Iterable, Sequence

import numpy as np

from .constants import ABS_TOL
from .criteria import AICCriterion, CriterionProtocol
from .definitions import GramData
from .forward_state import ForwardState
from .interface_validation import (
    validate_choice,
    validate_non_negative_finite_float,
    validate_optional_non_negative_int,
    validate_positive_finite_float,
)
from .routines_base import _resolve_criterion
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


def _beta_from_state(state: ForwardState, p: int) -> np.ndarray:
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
    state: ForwardState,
) -> GroupedSelectionState:
    active_group_list = list(active_groups)
    active_set = _flatten_group_indices(active_group_list, groups)
    return GroupedSelectionState(
        data=data,
        groups=tuple(tuple(int(feat) for feat in group) for group in groups),
        active_groups=active_group_list,
        active_set=active_set,
        beta=_beta_from_state(state, data.gram.shape[0]),
        rss=float(state.rss),
    )


def _apply_group_forward(state: ForwardState, group: Sequence[int]) -> None:
    # Apply group members in a stable, deterministic order.
    for feat_idx in group:
        if state.active_mask[int(feat_idx)]:
            raise ValueError("Group feature already active.")
        state.apply_forward(int(feat_idx))


def _apply_group_backward(state: ForwardState, group: Sequence[int]) -> None:
    # Remove in descending index order to avoid shifting positions mid-loop.
    pos_map = {feat: pos for pos, feat in enumerate(state.active_set)}
    positions = sorted((pos_map[int(feat)] for feat in group), reverse=True)
    for idx in positions:
        state.apply_backward(idx)


def _select_best_group_move(
    *,
    state: ForwardState,
    candidate_groups: Iterable[int],
    groups: Sequence[Sequence[int]],
    criterion: CriterionProtocol,
    apply_group_move: Callable[[ForwardState, Sequence[int]], None],
) -> tuple[int | None, float | None, ForwardState | None]:
    """Evaluate candidate group moves and return the best improving move."""
    best_group = None
    best_candidate_score = None
    best_candidate_state = None

    for g in candidate_groups:
        candidate_state = state.clone()
        try:
            apply_group_move(candidate_state, groups[g])
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

    return best_group, best_candidate_score, best_candidate_state


class BaseGroupedSelection:
    """Shared machinery for grouped greedy selection."""

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        *,
        tol: float = ABS_TOL,
        solver_policy: str = "strict",
        ridge_alpha: float = 1e-8,
        pinv_rcond: float = 1e-12,
        criterion=None,
        criterion_cls=None,
        criterion_kwargs=None,
    ):
        if criterion is not None and criterion_cls is not None:
            raise ValueError(
                f"{type(self).__name__} accepts either `criterion` or `criterion_cls`, not both."
            )
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
        self.tol = validate_positive_finite_float(tol, name="tol")
        self.solver_policy = validate_choice(
            solver_policy,
            name="solver_policy",
            choices={"strict", "ridge", "pinv"},
        )
        self.ridge_alpha = validate_non_negative_finite_float(
            ridge_alpha, name="ridge_alpha"
        )
        self.pinv_rcond = validate_positive_finite_float(
            pinv_rcond, name="pinv_rcond"
        )
        self.criterion = criterion
        if criterion is None:
            self.criterion_cls = criterion_cls or AICCriterion
        else:
            self.criterion_cls = criterion if isinstance(criterion, type) else type(criterion)
        self.criterion_kwargs = dict(criterion_kwargs or {})

    def _init_criterion(self, data: GramData) -> CriterionProtocol:
        return _resolve_criterion(
            selector_name=type(self).__name__,
            data=data,
            criterion=self.criterion,
            criterion_cls=self.criterion_cls if self.criterion is None else None,
            default_criterion_cls=AICCriterion,
            criterion_kwargs=self.criterion_kwargs,
        )


class GroupForwardSelection(BaseGroupedSelection):
    """Greedy forward selection over groups using Gram-only updates."""

    def fit(
        self, *, data: GramData, max_steps: int | None = None
    ) -> GroupedSelectionState:
        max_steps = validate_optional_non_negative_int(max_steps, name="max_steps")
        _validate_group_feature_bounds(self.groups, data.gram.shape[0])
        criterion = self._init_criterion(data)
        active: list[int] = []
        state = ForwardState.create(
            data,
            self.tol,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        initial = float(np.asarray(criterion.evaluate(state.rss, state.k)))
        criterion.update_current(initial)

        steps = 0
        while max_steps is None or steps < max_steps:
            candidates = (g for g in range(self.num_groups) if g not in active)
            best_group, best_candidate_score, best_candidate_state = (
                _select_best_group_move(
                    state=state,
                    candidate_groups=candidates,
                    groups=self.groups,
                    criterion=criterion,
                    apply_group_move=_apply_group_forward,
                )
            )

            if best_group is None or best_candidate_state is None:
                break

            active.append(best_group)
            state = best_candidate_state
            criterion.update_current(best_candidate_score)
            steps += 1

        return _build_grouped_state(
            data=data,
            groups=self.groups,
            active_groups=active,
            state=state,
        )


class GroupBackwardSelection(BaseGroupedSelection):
    """Greedy backward selection over groups using Gram-only updates."""

    def fit(
        self, *, data: GramData, max_steps: int | None = None
    ) -> GroupedSelectionState:
        max_steps = validate_optional_non_negative_int(max_steps, name="max_steps")
        _validate_group_feature_bounds(self.groups, data.gram.shape[0])
        criterion = self._init_criterion(data)
        active = list(range(self.num_groups))
        full_idx = _flatten_group_indices(active, self.groups)
        state = ForwardState.from_active_set(
            data,
            full_idx,
            self.tol,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        initial = float(
            np.asarray(criterion.evaluate(state.rss, state.k))
        )
        criterion.update_current(initial)

        steps = 0
        while active and (max_steps is None or steps < max_steps):
            best_drop, best_candidate_score, best_candidate_state = (
                _select_best_group_move(
                    state=state,
                    candidate_groups=active,
                    groups=self.groups,
                    criterion=criterion,
                    apply_group_move=_apply_group_backward,
                )
            )

            if best_drop is None or best_candidate_state is None:
                break

            active.remove(best_drop)
            state = best_candidate_state
            criterion.update_current(best_candidate_score)
            steps += 1

        return _build_grouped_state(
            data=data,
            groups=self.groups,
            active_groups=active,
            state=state,
        )


__all__ = [
    "GroupedSelectionState",
    "GroupForwardSelection",
    "GroupBackwardSelection",
]
