"""Greedy selectors for single-dataset GramData."""

from __future__ import annotations

import inspect

import numpy as np

from .constants import ABS_TOL
from .criteria import AICCriterion, SelectionCriterion
from .definitions import GramData
from .routines_base import _DISALLOWED_CV_CRITERIA, _validate_state_target
from .forward_state import ForwardState
from .state import SelectionState


class ForwardSelection:
    """Forward selection with O(k·p) per-step candidate updates.

    This routine maintains residual correlations and variances for all features
    using an orthogonal basis derived from Gram data only. It avoids the
    O(k^2·p) cost of recomputing projections from scratch at each step.
    """

    _default_criterion = AICCriterion
    _reject_ic: bool = False

    def __init__(
        self,
        *,
        tol: float = ABS_TOL,
        criterion_cls=None,
        criterion_kwargs=None,
    ):
        self.tol = tol
        cls = criterion_cls or self._default_criterion
        if self._reject_ic and issubclass(cls, _DISALLOWED_CV_CRITERIA):
            raise ValueError(
                f"{type(self).__name__} uses cross-validation for regularisation; "
                f"{cls.__name__} is not supported for CV selection routines. "
                f"Use BestRSSCriterion (the default) instead."
            )
        self.criterion_cls = cls
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
        result_state = _validate_state_target(
            state, data, selector_name=type(self).__name__
        )
        if result_state is not None and result_state.active_set:
            raise ValueError("ForwardSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        work_state = ForwardState.create(data, self.tol)

        while max_steps is None or work_state.k < max_steps:
            scored = work_state.candidate_scores()
            if scored is None:
                break
            candidates, rss_new = scored
            best_idx, best_score = criterion.best_candidate(
                rss_new, work_state.k + 1
            )
            if not criterion.is_improvement(best_score):
                break
            feat_idx = int(candidates[best_idx])
            work_state.apply_forward(feat_idx)
            criterion.update_current(best_score)

        result = result_state if result_state is not None else SelectionState(data)
        result.init_from_active_set(work_state.active_set)
        return result


class BackwardSelection(ForwardSelection):
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
        result_state = _validate_state_target(
            state, data, selector_name=type(self).__name__
        )
        if result_state is not None and result_state.active_set:
            raise ValueError("BackwardSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        full_active = list(range(data.gram.shape[0]))
        work_state = ForwardState.from_active_set(data, full_active, self.tol)
        initial = float(
            np.asarray(criterion.evaluate(work_state.rss, len(work_state.active_set)))
        )
        criterion.update_current(initial)

        steps = 0
        while work_state.k and (max_steps is None or steps < max_steps):
            rss_values = work_state.backward_scores()
            if rss_values is None:
                break
            best_idx, best_score = criterion.best_candidate(
                rss_values, work_state.k - 1
            )
            if not self.allow_worse and not criterion.is_improvement(best_score):
                break
            work_state.apply_backward(best_idx)
            criterion.update_current(best_score)
            steps += 1

        result = result_state if result_state is not None else SelectionState(data)
        result.init_from_active_set(work_state.active_set)
        return result


class MixedSelection(ForwardSelection):
    """Mixed forward/backward selection with fast updates."""

    def fit(
        self,
        state: SelectionState | None = None,
        *,
        data: GramData,
        max_forward_steps: int | None = None,
        max_total_steps: int | None = None,
    ) -> SelectionState:
        result_state = _validate_state_target(
            state, data, selector_name=type(self).__name__
        )
        if result_state is not None and result_state.active_set:
            raise ValueError("MixedSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        work_state = ForwardState.create(data, self.tol)

        forward_steps = 0
        total_steps = 0
        while True:
            if max_total_steps is not None and total_steps >= max_total_steps:
                break
            scored = work_state.candidate_scores()
            if scored is None:
                break
            candidates, rss_new = scored
            best_idx, best_score = criterion.best_candidate(
                rss_new, work_state.k + 1
            )
            if not criterion.is_improvement(best_score):
                break
            feat_idx = int(candidates[best_idx])
            work_state.apply_forward(feat_idx)
            criterion.update_current(best_score)
            forward_steps += 1
            total_steps += 1

            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break

            while True:
                if max_total_steps is not None and total_steps >= max_total_steps:
                    break
                rss_values = work_state.backward_scores()
                if rss_values is None:
                    break
                best_idx, best_score = criterion.best_candidate(
                    rss_values, work_state.k - 1
                )
                if not criterion.is_improvement(best_score):
                    break
                work_state.apply_backward(best_idx)
                criterion.update_current(best_score)
                total_steps += 1

        result = result_state if result_state is not None else SelectionState(data)
        result.init_from_active_set(work_state.active_set)
        return result


__all__ = [
    "ForwardSelection",
    "BackwardSelection",
    "MixedSelection",
]
