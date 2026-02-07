"""
Deprecated reference implementations for grouped greedy selection.

These routines intentionally recompute block solves for transparency.
Use `selection.grouped_routines` (fast defaults) instead.
This module will be removed in a future release.
"""

from __future__ import annotations

import warnings

import inspect
from typing import Iterable, Sequence

import numpy as np

from .constants import ABS_TOL
from .criteria import AICCriterion, SelectionCriterion
from .definitions import GramData

warnings.warn(
    "selection.legacy_grouped_routines is deprecated and will be removed in a future "
    "release. Use selection.grouped_routines instead.",
    DeprecationWarning,
    stacklevel=2,
)


def _flatten_group_indices(groups: Iterable[int], group_map: Sequence[Sequence[int]]):
    idx: list[int] = []
    for g in groups:
        idx.extend(group_map[g])
    return sorted(idx)


def _rss_for_active_groups(
    active_groups, group_map, data: GramData
) -> tuple[np.ndarray, float]:
    """
    Solve the block least-squares for the given active_groups and return
    (beta_full, rss). Beta is full-length with zeros off-support.
    """
    p = data.gram.shape[0]
    beta_full = np.zeros(p, dtype=float)
    if not active_groups:
        return beta_full, float(data.y_norm)
    idx = np.array(_flatten_group_indices(active_groups, group_map), dtype=int)
    G_ss = data.gram[np.ix_(idx, idx)]
    c_s = data.cov[idx]
    try:
        beta_s = np.linalg.solve(G_ss, c_s)
    except np.linalg.LinAlgError as err:
        raise np.linalg.LinAlgError("Grouped solve failed (singular block).") from err
    beta_full[idx] = beta_s
    rss = float(data.y_norm - c_s @ beta_s)
    return beta_full, rss


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

    def _score_state(self, active_groups, data, criterion: SelectionCriterion):
        beta, rss = _rss_for_active_groups(active_groups, self.groups, data)
        score = float(np.asarray(criterion.evaluate(rss, len(active_groups))))
        return beta, rss, score


class GroupForwardSelection(BaseGroupedSelection):
    """Greedy forward selection over predefined feature groups."""

    def fit(self, *, data: GramData, max_steps: int | None = None):
        criterion = self._init_criterion(data)
        active: list[int] = []
        beta, rss, score = self._score_state(active, data, criterion)
        criterion.update_current(score)

        steps = 0
        while max_steps is None or steps < max_steps:
            best_group = None
            best_candidate_score = None
            best_candidate_beta = None
            best_candidate_rss = None

            for g in range(self.num_groups):
                if g in active:
                    continue
                candidate_groups = active + [g]
                try:
                    beta_cand, rss_cand, score_cand = self._score_state(
                        candidate_groups, data, criterion
                    )
                except np.linalg.LinAlgError:
                    continue
                if not criterion.is_improvement(score_cand):
                    continue
                if best_candidate_score is None or criterion.is_improvement(
                    score_cand, best_candidate_score
                ):
                    best_group = g
                    best_candidate_score = score_cand
                    best_candidate_beta = beta_cand
                    best_candidate_rss = rss_cand

            if best_group is None:
                break

            active.append(best_group)
            beta = best_candidate_beta
            rss = best_candidate_rss
            criterion.update_current(best_candidate_score)
            steps += 1

        result = type("GroupedState", (), {})()
        result.active_groups = list(active)
        result.beta = beta
        result.rss = float(rss)
        return result


class GroupBackwardSelection(BaseGroupedSelection):
    """Greedy backward selection over predefined feature groups."""

    def fit(self, *, data: GramData, max_steps: int | None = None):
        criterion = self._init_criterion(data)
        active = list(range(self.num_groups))
        beta, rss, score = self._score_state(active, data, criterion)
        criterion.update_current(score)

        steps = 0
        while active and (max_steps is None or steps < max_steps):
            best_drop = None
            best_candidate_score = None
            best_candidate_beta = None
            best_candidate_rss = None

            for pos, g in enumerate(active):
                candidate_groups = active[:pos] + active[pos + 1 :]
                try:
                    beta_cand, rss_cand, score_cand = self._score_state(
                        candidate_groups, data, criterion
                    )
                except np.linalg.LinAlgError:
                    continue
                # Backward can allow non-improving moves only if explicitly desired; here we stay improving.
                if not criterion.is_improvement(score_cand):
                    continue
                if best_candidate_score is None or criterion.is_improvement(
                    score_cand, best_candidate_score
                ):
                    best_drop = g
                    best_candidate_score = score_cand
                    best_candidate_beta = beta_cand
                    best_candidate_rss = rss_cand

            if best_drop is None:
                break

            active.remove(best_drop)
            beta = best_candidate_beta
            rss = best_candidate_rss
            criterion.update_current(best_candidate_score)
            steps += 1

        result = type("GroupedState", (), {})()
        result.active_groups = list(active)
        result.beta = beta
        result.rss = float(rss)
        return result


__all__ = [
    "GroupForwardSelection",
    "GroupBackwardSelection",
]
