"""Greedy selectors for single-dataset GramData."""

from __future__ import annotations

import numpy as np

from .constants import ABS_TOL
from .interface_validation import (
    validate_choice,
    validate_bool,
    validate_non_negative_finite_float,
    validate_optional_non_negative_int,
    validate_positive_finite_float,
)
from .criteria import AICCriterion, CriterionProtocol
from .definitions import GramData
from .selector_validation import (
    _resolve_criterion,
    _resolve_criterion_provider,
    _validate_cv_criterion,
    _validate_state_target,
)
from .incremental_solver import IncrementalSolver
from .state_single import SelectionState


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
        solver_policy: str = "strict",
        ridge_alpha: float = 1e-8,
        pinv_rcond: float = 1e-12,
        criterion=None,
        criterion_kwargs=None,
    ):
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
        self.criterion = (
            self._default_criterion if criterion is None else criterion
        )
        self.criterion_kwargs = dict(criterion_kwargs or {})
        if self._reject_ic:
            candidate = _resolve_criterion_provider(
                selector_name=type(self).__name__,
                criterion=self.criterion,
                default_criterion_cls=self._default_criterion,
            )
            if getattr(candidate, "cv_compatible", True) is False:
                name = (
                    candidate.__name__
                    if isinstance(candidate, type)
                    else type(candidate).__name__
                )
                raise ValueError(
                    f"{type(self).__name__} uses cross-validation for regularisation; "
                    f"{name} is not supported for CV selection routines. "
                    f"Use BestRSSCriterion (the default) instead."
                )

    def _init_criterion(self, data: GramData) -> CriterionProtocol:
        criterion = _resolve_criterion(
            selector_name=type(self).__name__,
            data=data,
            criterion=self.criterion,
            default_criterion_cls=self._default_criterion,
            criterion_kwargs=self.criterion_kwargs,
        )
        if self._reject_ic:
            _validate_cv_criterion(criterion, selector_name=type(self).__name__)
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

        max_steps = validate_optional_non_negative_int(max_steps, name="max_steps")
        criterion = self._init_criterion(data)
        work_state = IncrementalSolver.create(
            data,
            self.tol,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )

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

        result = (
            result_state
            if result_state is not None
            else SelectionState(
                data,
                solver_policy=self.solver_policy,
                ridge_alpha=self.ridge_alpha,
                pinv_rcond=self.pinv_rcond,
            )
        )
        result.init_from_active_set(work_state.active_set)
        return result


class BackwardSelection(ForwardSelection):
    """Backward selection using fast Gram-only updates."""

    def __init__(self, *, allow_worse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.allow_worse = validate_bool(allow_worse, name="allow_worse")

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

        max_steps = validate_optional_non_negative_int(max_steps, name="max_steps")
        criterion = self._init_criterion(data)
        full_active = list(range(data.gram.shape[0]))
        work_state = IncrementalSolver.from_active_set(
            data,
            full_active,
            self.tol,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
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

        result = (
            result_state
            if result_state is not None
            else SelectionState(
                data,
                solver_policy=self.solver_policy,
                ridge_alpha=self.ridge_alpha,
                pinv_rcond=self.pinv_rcond,
            )
        )
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

        max_forward_steps = validate_optional_non_negative_int(
            max_forward_steps, name="max_forward_steps"
        )
        max_total_steps = validate_optional_non_negative_int(
            max_total_steps, name="max_total_steps"
        )
        criterion = self._init_criterion(data)
        work_state = IncrementalSolver.create(
            data,
            self.tol,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )

        forward_steps = 0
        total_steps = 0
        while True:
            if max_total_steps is not None and total_steps >= max_total_steps:
                break
            if max_forward_steps is not None and forward_steps >= max_forward_steps:
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

        result = (
            result_state
            if result_state is not None
            else SelectionState(
                data,
                solver_policy=self.solver_policy,
                ridge_alpha=self.ridge_alpha,
                pinv_rcond=self.pinv_rcond,
            )
        )
        result.init_from_active_set(work_state.active_set)
        return result


__all__ = [
    "ForwardSelection",
    "BackwardSelection",
    "MixedSelection",
]
