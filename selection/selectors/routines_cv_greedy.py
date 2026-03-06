"""Cross-validation greedy selectors."""

from __future__ import annotations

import numpy as np

from ..criteria import BestRSSCriterion
from ..core.definitions import CrossValGramData
from ..validation.interface_validation import validate_optional_non_negative_int
from .routines_cv_scoring import (
    _build_fold_states,
    _build_cv_state_from_active_set,
    _cv_backward_scores,
    _cv_forward_scores,
    _cv_rss,
    _normalize_cv_aggregation,
    _rebuild_states,
)
from .routines_greedy import ForwardSelection
from ..validation.selector_validation import _validate_cv_state_target
from ..core.state_cv import CrossValSelectionState


class _BaseCrossValSelection(ForwardSelection):
    """Shared constructor for CV selectors with configurable aggregation."""

    _default_criterion = BestRSSCriterion
    _reject_ic = True

    def __init__(self, *, cv_aggregation: str = "sum_rss", **kwargs):
        super().__init__(**kwargs)
        self.cv_aggregation = _normalize_cv_aggregation(cv_aggregation)


class CrossValForwardSelection(_BaseCrossValSelection):
    """Cross-validated forward selection using Gram-only training updates."""

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps: int | None = None,
    ) -> CrossValSelectionState:
        result_state = _validate_cv_state_target(
            state,
            data,
            selector_name=type(self).__name__,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        if result_state is not None and result_state.active_set:
            raise ValueError("CrossValForwardSelection does not support warm starts.")

        max_steps = validate_optional_non_negative_int(max_steps, name="max_steps")
        criterion = self._init_criterion(data.make_full_data())
        fold_states = _build_fold_states(
            data,
            tol=self.tol,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        criterion.update_current(
            float(
                np.asarray(
                    criterion.evaluate(
                        _cv_rss(
                            fold_states,
                            data,
                            cv_aggregation=self.cv_aggregation,
                        ),
                        0,
                    )
                )
            )
        )

        steps = 0
        while max_steps is None or steps < max_steps:
            scored = _cv_forward_scores(
                fold_states,
                data,
                self.tol,
                cv_aggregation=self.cv_aggregation,
            )
            if scored is None:
                break
            candidates, aggregated = scored
            best_idx, best_score = criterion.best_candidate(
                aggregated, len(fold_states[0].active_set) + 1
            )
            if not criterion.is_improvement(best_score):
                break
            feat_idx = candidates[best_idx]
            for state_k in fold_states:
                state_k.apply_forward(feat_idx)
            fold_states = _rebuild_states(
                data,
                fold_states[0].active_set,
                self.tol,
                solver_policy=self.solver_policy,
                ridge_alpha=self.ridge_alpha,
                pinv_rcond=self.pinv_rcond,
            )
            criterion.update_current(best_score)
            steps += 1

        return _build_cv_state_from_active_set(
            data,
            fold_states[0].active_set,
            state=result_state,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )


class CrossValBackwardSelection(_BaseCrossValSelection):
    """Cross-validated backward selection using Gram-only training updates."""

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps: int | None = None,
    ) -> CrossValSelectionState:
        result_state = _validate_cv_state_target(
            state,
            data,
            selector_name=type(self).__name__,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        if result_state is not None and result_state.active_set:
            raise ValueError("CrossValBackwardSelection does not support warm starts.")

        max_steps = validate_optional_non_negative_int(max_steps, name="max_steps")
        criterion = self._init_criterion(data.make_full_data())
        full_active = list(range(data.p))
        fold_states = _build_fold_states(
            data,
            tol=self.tol,
            active_set=full_active,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        criterion.update_current(
            float(
                np.asarray(
                    criterion.evaluate(
                        _cv_rss(
                            fold_states,
                            data,
                            cv_aggregation=self.cv_aggregation,
                        ),
                        len(full_active),
                    )
                )
            )
        )

        steps = 0
        while fold_states[0].k and (max_steps is None or steps < max_steps):
            aggregated = _cv_backward_scores(
                fold_states,
                data,
                self.tol,
                cv_aggregation=self.cv_aggregation,
            )
            if aggregated is None:
                break
            best_idx, best_score = criterion.best_candidate(
                aggregated, fold_states[0].k - 1
            )
            if not criterion.is_improvement(best_score):
                break
            try:
                for state_k in fold_states:
                    state_k.apply_backward(best_idx)
                fold_states = _rebuild_states(
                    data,
                    fold_states[0].active_set,
                    self.tol,
                    solver_policy=self.solver_policy,
                    ridge_alpha=self.ridge_alpha,
                    pinv_rcond=self.pinv_rcond,
                )
            except ValueError:
                break
            criterion.update_current(best_score)
            steps += 1

        return _build_cv_state_from_active_set(
            data,
            fold_states[0].active_set,
            state=result_state,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )


class CrossValMixedSelection(_BaseCrossValSelection):
    """Cross-validated mixed selection using Gram-only training updates."""

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_forward_steps: int | None = None,
        max_total_steps: int | None = None,
    ) -> CrossValSelectionState:
        result_state = _validate_cv_state_target(
            state,
            data,
            selector_name=type(self).__name__,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        if result_state is not None and result_state.active_set:
            raise ValueError("CrossValMixedSelection does not support warm starts.")

        max_forward_steps = validate_optional_non_negative_int(
            max_forward_steps, name="max_forward_steps"
        )
        max_total_steps = validate_optional_non_negative_int(
            max_total_steps, name="max_total_steps"
        )
        criterion = self._init_criterion(data.make_full_data())
        fold_states = _build_fold_states(
            data,
            tol=self.tol,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        criterion.update_current(
            float(
                np.asarray(
                    criterion.evaluate(
                        _cv_rss(
                            fold_states,
                            data,
                            cv_aggregation=self.cv_aggregation,
                        ),
                        0,
                    )
                )
            )
        )

        forward_steps = 0
        total_steps = 0
        while True:
            if max_total_steps is not None and total_steps >= max_total_steps:
                break
            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break
            scored = _cv_forward_scores(
                fold_states,
                data,
                self.tol,
                cv_aggregation=self.cv_aggregation,
            )
            if scored is None:
                break
            candidates, aggregated = scored
            best_idx, best_score = criterion.best_candidate(
                aggregated, len(fold_states[0].active_set) + 1
            )
            if not criterion.is_improvement(best_score):
                break
            feat_idx = candidates[best_idx]
            for state_k in fold_states:
                state_k.apply_forward(feat_idx)
            fold_states = _rebuild_states(
                data,
                fold_states[0].active_set,
                self.tol,
                solver_policy=self.solver_policy,
                ridge_alpha=self.ridge_alpha,
                pinv_rcond=self.pinv_rcond,
            )
            criterion.update_current(best_score)
            forward_steps += 1
            total_steps += 1

            while True:
                if max_total_steps is not None and total_steps >= max_total_steps:
                    break
                aggregated = _cv_backward_scores(
                    fold_states,
                    data,
                    self.tol,
                    cv_aggregation=self.cv_aggregation,
                )
                if aggregated is None:
                    break
                best_idx, best_score = criterion.best_candidate(
                    aggregated, fold_states[0].k - 1
                )
                if not criterion.is_improvement(best_score):
                    break
                try:
                    for state_k in fold_states:
                        state_k.apply_backward(best_idx)
                    fold_states = _rebuild_states(
                        data,
                        fold_states[0].active_set,
                        self.tol,
                        solver_policy=self.solver_policy,
                        ridge_alpha=self.ridge_alpha,
                        pinv_rcond=self.pinv_rcond,
                    )
                except ValueError:
                    break
                criterion.update_current(best_score)
                total_steps += 1

        return _build_cv_state_from_active_set(
            data,
            fold_states[0].active_set,
            state=result_state,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )


__all__ = [
    "CrossValForwardSelection",
    "CrossValBackwardSelection",
    "CrossValMixedSelection",
]
