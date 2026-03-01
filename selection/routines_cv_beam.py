"""Cross-validation beam selectors and beam-transition helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .beam_pruning import prune_unique_beams
from .criteria import BestRSSCriterion, CriterionProtocol
from .definitions import CrossValGramData
from .incremental_solver import IncrementalSolver
from .interface_validation import (
    validate_bool,
    validate_optional_non_negative_int,
    validate_positive_int,
)
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
from .selector_validation import _validate_cv_state_target
from .state_cv import CrossValSelectionState
from .topk import topk_indices


@dataclass
class CVBeam:
    states: list[IncrementalSolver]
    criterion: CriterionProtocol
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


def _cv_beam_forward_children(
    beam: CVBeam,
    beam_width: int,
    data: CrossValGramData,
    tol: float,
    *,
    cv_aggregation: str = "sum_rss",
) -> list[CVBeam]:
    scored = _cv_forward_scores(
        beam.states, data, tol, cv_aggregation=cv_aggregation
    )
    if scored is None:
        return []
    candidates, aggregated = scored
    crit_scores = np.asarray(
        beam.criterion.evaluate(aggregated, len(beam.states[0].active_set) + 1)
    )
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)
    children: list[CVBeam] = []
    for idx in order:
        candidate_score = float(crit_scores[idx])
        if not beam.criterion.is_improvement(candidate_score, beam.score):
            continue
        feat_idx = int(candidates[idx])
        child_states = [state.clone() for state in beam.states]
        for state_k in child_states:
            state_k.apply_forward(feat_idx)
        # Rebuild per-fold states to keep QR/K updates numerically stable.
        child_states = _rebuild_states(
            data,
            child_states[0].active_set,
            tol,
            solver_policy=child_states[0].solver_policy,
            ridge_alpha=child_states[0].ridge_alpha,
            pinv_rcond=child_states[0].pinv_rcond,
        )
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(CVBeam(child_states, child_criterion, candidate_score))
    return children


def _cv_beam_backward_children(
    beam: CVBeam,
    beam_width: int,
    data: CrossValGramData,
    tol: float,
    *,
    allow_worse: bool = False,
    cv_aggregation: str = "sum_rss",
) -> list[CVBeam]:
    aggregated = _cv_backward_scores(
        beam.states, data, tol, cv_aggregation=cv_aggregation
    )
    if aggregated is None or not len(aggregated):
        return []
    crit_scores = np.asarray(
        beam.criterion.evaluate(aggregated, max(len(beam.states[0].active_set) - 1, 0))
    )
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)
    children: list[CVBeam] = []
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
            child_states = _rebuild_states(
                data,
                child_states[0].active_set,
                tol,
                solver_policy=child_states[0].solver_policy,
                ridge_alpha=child_states[0].ridge_alpha,
                pinv_rcond=child_states[0].pinv_rcond,
            )
        except ValueError:
            continue
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(CVBeam(child_states, child_criterion, candidate_score))
    return children


def _cv_beam_best_backward_child(
    beam: CVBeam,
    data: CrossValGramData,
    tol: float,
    *,
    cv_aggregation: str = "sum_rss",
) -> CVBeam | None:
    for child in _cv_beam_backward_children(
        beam,
        1,
        data,
        tol,
        allow_worse=False,
        cv_aggregation=cv_aggregation,
    ):
        if beam.criterion.is_improvement(child.score, beam.score):
            return child
    return None


class BeamCrossValForwardSelection(ForwardSelection):
    """Beam-search forward selection with CV scoring."""

    _default_criterion = BestRSSCriterion
    _reject_ic = True

    def __init__(
        self, *, beam_width: int = 1, cv_aggregation: str = "sum_rss", **kwargs
    ):
        super().__init__(**kwargs)
        self.beam_width = validate_positive_int(beam_width, name="beam_width")
        self.cv_aggregation = _normalize_cv_aggregation(cv_aggregation)

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps: int | None = None,
    ) -> CrossValSelectionState:
        result_state = _validate_cv_state_target(
            state, data, selector_name=type(self).__name__
        )
        if result_state is not None and result_state.active_set:
            raise ValueError(
                "BeamCrossValForwardSelection does not support warm starts."
            )

        max_steps = validate_optional_non_negative_int(max_steps, name="max_steps")
        criterion = self._init_criterion(data.make_full_data())
        fold_states = _build_fold_states(
            data,
            tol=self.tol,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )
        initial_score = float(
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
        criterion.update_current(initial_score)
        beams = [CVBeam(fold_states, criterion, criterion.current_value)]

        steps = 0
        while beams and (max_steps is None or steps < max_steps):
            candidates: list[CVBeam] = []
            for beam in beams:
                candidates.extend(
                    _cv_beam_forward_children(
                        beam,
                        self.beam_width,
                        data,
                        self.tol,
                        cv_aggregation=self.cv_aggregation,
                    )
                )
            if not candidates:
                break
            beams = prune_unique_beams(candidates, self.beam_width)
            steps += 1

        sel = min if criterion.minimize else max
        best = sel(beams, key=lambda b: b.score)
        return _build_cv_state_from_active_set(
            data,
            best.states[0].active_set,
            state=result_state,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )


class BeamCrossValBackwardSelection(ForwardSelection):
    """Beam-search backward selection with CV scoring."""

    _default_criterion = BestRSSCriterion
    _reject_ic = True

    def __init__(
        self,
        *,
        beam_width: int = 1,
        allow_worse: bool = False,
        cv_aggregation: str = "sum_rss",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beam_width = validate_positive_int(beam_width, name="beam_width")
        self.allow_worse = validate_bool(allow_worse, name="allow_worse")
        self.cv_aggregation = _normalize_cv_aggregation(cv_aggregation)

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_steps: int | None = None,
    ) -> CrossValSelectionState:
        result_state = _validate_cv_state_target(
            state, data, selector_name=type(self).__name__
        )
        if result_state is not None and result_state.active_set:
            raise ValueError(
                "BeamCrossValBackwardSelection does not support warm starts."
            )

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
        initial_score = float(
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
        criterion.update_current(initial_score)
        beams = [CVBeam(fold_states, criterion, criterion.current_value)]

        steps = 0
        while beams and (max_steps is None or steps < max_steps):
            candidates: list[CVBeam] = []
            for beam in beams:
                candidates.extend(
                    _cv_beam_backward_children(
                        beam,
                        self.beam_width,
                        data,
                        self.tol,
                        allow_worse=self.allow_worse,
                        cv_aggregation=self.cv_aggregation,
                    )
                )
            if not candidates:
                break
            beams = prune_unique_beams(candidates, self.beam_width)
            steps += 1

        sel = min if criterion.minimize else max
        best = sel(beams, key=lambda b: b.score)
        return _build_cv_state_from_active_set(
            data,
            best.states[0].active_set,
            state=result_state,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )


class BeamCrossValMixedSelection(ForwardSelection):
    """Beam-search mixed selection with CV scoring."""

    _default_criterion = BestRSSCriterion
    _reject_ic = True

    def __init__(
        self, *, beam_width: int = 1, cv_aggregation: str = "sum_rss", **kwargs
    ):
        super().__init__(**kwargs)
        self.beam_width = validate_positive_int(beam_width, name="beam_width")
        self.cv_aggregation = _normalize_cv_aggregation(cv_aggregation)

    def fit(
        self,
        state: CrossValSelectionState | None = None,
        *,
        data: CrossValGramData,
        max_forward_steps: int | None = None,
        max_total_steps: int | None = None,
    ) -> CrossValSelectionState:
        result_state = _validate_cv_state_target(
            state, data, selector_name=type(self).__name__
        )
        if result_state is not None and result_state.active_set:
            raise ValueError(
                "BeamCrossValMixedSelection does not support warm starts."
            )

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
        initial_score = float(
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
        criterion.update_current(initial_score)
        initial = CVBeam(fold_states, criterion, criterion.current_value)
        beams = [initial]
        best = initial
        sel = min if criterion.minimize else max

        forward_steps = 0
        total_ops = 0
        while True:
            if max_total_steps is not None and total_ops >= max_total_steps:
                break
            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break
            candidates: list[CVBeam] = []
            for beam in beams:
                candidates.extend(
                    _cv_beam_forward_children(
                        beam,
                        self.beam_width,
                        data,
                        self.tol,
                        cv_aggregation=self.cv_aggregation,
                    )
                )
            if not candidates:
                break
            beams = prune_unique_beams(candidates, self.beam_width)
            best = sel(beams, key=lambda b: b.score)
            forward_steps += 1
            total_ops += len(beams)

            new_beams: list[CVBeam] = []
            for beam in beams:
                while True:
                    if max_total_steps is not None and total_ops >= max_total_steps:
                        break
                    improved = _cv_beam_best_backward_child(
                        beam,
                        data,
                        self.tol,
                        cv_aggregation=self.cv_aggregation,
                    )
                    if improved is None:
                        break
                    beam = improved
                    total_ops += 1
                best = sel([best, beam], key=lambda b: b.score)
                new_beams.append(beam)
            beams = new_beams

        return _build_cv_state_from_active_set(
            data,
            best.states[0].active_set,
            state=result_state,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
        )


__all__ = [
    "CVBeam",
    "_cv_beam_forward_children",
    "_cv_beam_backward_children",
    "_cv_beam_best_backward_child",
    "BeamCrossValForwardSelection",
    "BeamCrossValBackwardSelection",
    "BeamCrossValMixedSelection",
]
