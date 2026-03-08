"""Greedy selectors for single-dataset GramData."""

from __future__ import annotations

import numpy as np

from ..core.constants import ABS_TOL
from ..validation.interface_validation import (
    validate_choice,
    validate_bool,
    validate_non_negative_finite_float,
    validate_optional_non_negative_int,
    validate_positive_finite_float,
)
from ..criteria import AICCriterion, CriterionProtocol
from ..core.definitions import GramData
from ..validation.selector_validation import (
    _resolve_criterion,
    _resolve_criterion_provider,
    _validate_cv_criterion,
    _validate_state_target,
)
from ..core.incremental_solver import IncrementalSolver
from ..core.state_single import SelectionState
from .forward_budget_policy import (
    resolve_forward_budget,
    should_accept_forward_candidate,
)


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
        stop_on_no_improvement: bool = False,
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
        self.stop_on_no_improvement = validate_bool(
            stop_on_no_improvement, name="stop_on_no_improvement"
        )
        self.criterion = (
            self._default_criterion
            if criterion is None
            else criterion
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
        """Construct and initialize criterion state for this selector.

        Preconditions:
        - `data` is a validated `GramData` instance.
        - constructor-level criterion config has passed boundary validation.

        Postconditions:
        - returns a criterion satisfying `CriterionProtocol`.
        - criterion `current_value` is initialized at the empty model (`k=0`).
        """
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
        """Run greedy forward selection.

        Preconditions:
        - `data` must be a validated `GramData`.
        - if `state` is provided, it must be a `SelectionState` bound to `data`
          and currently empty (no warm start).
        - `max_steps` is `None` or a non-negative integer.

        Postconditions:
        - returns a `SelectionState` on the selected support.
        - returned state coefficients/RSS are recomputed from final support via
          stable solve (`SelectionState.init_from_active_set`).

        Raises:
        - `TypeError`/`ValueError` for contract violations at the API boundary.
        - `np.linalg.LinAlgError` if final support refit is numerically singular
          under the chosen solver policy.
        """
        result_state = _validate_state_target(
            state,
            data,
            selector_name=type(self).__name__,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        if result_state is not None and result_state.active_set:
            raise ValueError("ForwardSelection does not support warm starts.")

        max_steps = resolve_forward_budget(
            budget=max_steps,
            budget_name="max_steps",
            selector_name=type(self).__name__,
            stop_on_no_improvement=self.stop_on_no_improvement,
        )
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
            if not should_accept_forward_candidate(
                criterion=criterion,
                candidate_score=best_score,
                incumbent_score=criterion.current_value,
                stop_on_no_improvement=self.stop_on_no_improvement,
            ):
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
        """Run greedy backward elimination from full support.

        Preconditions:
        - same boundary requirements as `ForwardSelection.fit`.
        - `allow_worse=False` means each accepted backward step must improve the
          criterion; `allow_worse=True` allows forced removals under budget.

        Postconditions:
        - returns a `SelectionState` fit on the final active support.
        """
        result_state = _validate_state_target(
            state,
            data,
            selector_name=type(self).__name__,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
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
        """Run mixed forward-then-backward greedy search.

        Preconditions:
        - same data/state boundary requirements as other greedy selectors.
        - step budgets are `None` or non-negative integers.

        Postconditions:
        - applies at most one forward move per outer iteration, then zero or
          more improving backward moves.
        - returns a fully materialized `SelectionState` on final support.
        """
        result_state = _validate_state_target(
            state,
            data,
            selector_name=type(self).__name__,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        if result_state is not None and result_state.active_set:
            raise ValueError("MixedSelection does not support warm starts.")

        max_forward_steps = resolve_forward_budget(
            budget=max_forward_steps,
            budget_name="max_forward_steps",
            selector_name=type(self).__name__,
            stop_on_no_improvement=self.stop_on_no_improvement,
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
            if not should_accept_forward_candidate(
                criterion=criterion,
                candidate_score=best_score,
                incumbent_score=criterion.current_value,
                stop_on_no_improvement=self.stop_on_no_improvement,
            ):
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
