"""Cross-validation selectors and CV beam helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .criteria import BestRSSCriterion, SelectionCriterion
from .definitions import CrossValGramData
from .routines_base import _validate_cv_state_target
from .routines_cv_scoring import (
    _build_cv_state_from_active_set,
    _cv_backward_scores,
    _cv_forward_scores,
    _cv_rss,
    _rebuild_states,
)
from .routines_greedy import ForwardSelection
from .forward_state import ForwardState
from .state import CrossValSelectionState
from .topk import topk_indices


@dataclass
class CVBeam:
    states: list[ForwardState]
    criterion: SelectionCriterion
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


def _cv_beam_prune(
    beams: list[CVBeam], beam_limit: int
) -> list[CVBeam]:
    if not beams:
        return []
    minimize = beams[0].criterion.minimize
    seen = set()
    result: list[CVBeam] = []
    for beam in sorted(beams, key=lambda b: b.score, reverse=not minimize):
        sig = beam.signature
        if sig in seen:
            continue
        seen.add(sig)
        result.append(beam)
        if len(result) >= beam_limit:
            break
    return result


def _cv_beam_forward_children(
    beam: CVBeam,
    beam_width: int,
    data: CrossValGramData,
    tol: float,
) -> list[CVBeam]:
    scored = _cv_forward_scores(beam.states, data, tol)
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
        child_states = _rebuild_states(data, child_states[0].active_set, tol)
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
) -> list[CVBeam]:
    aggregated = _cv_backward_scores(beam.states, data, tol)
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
            child_states = _rebuild_states(data, child_states[0].active_set, tol)
        except ValueError:
            continue
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(CVBeam(child_states, child_criterion, candidate_score))
    return children


def _cv_beam_best_backward_child(
    beam: CVBeam, data: CrossValGramData, tol: float
) -> CVBeam | None:
    for child in _cv_beam_backward_children(
        beam, 1, data, tol, allow_worse=False
    ):
        if beam.criterion.is_improvement(child.score, beam.score):
            return child
    return None


class CrossValForwardSelection(ForwardSelection):
    """Cross-validated forward selection using Gram-only training updates."""

    _default_criterion = BestRSSCriterion
    _reject_ic = True

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
            raise ValueError("CrossValForwardSelection does not support warm starts.")

        criterion = self._init_criterion(data.make_full_data())
        fold_states = [
            ForwardState.create(data.train_data_for_fold(k), self.tol)
            for k in range(data.n_folds)
        ]
        criterion.update_current(
            float(np.asarray(criterion.evaluate(_cv_rss(fold_states, data), 0)))
        )

        steps = 0
        while max_steps is None or steps < max_steps:
            scored = _cv_forward_scores(fold_states, data, self.tol)
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
                data, fold_states[0].active_set, self.tol
            )
            criterion.update_current(best_score)
            steps += 1

        return _build_cv_state_from_active_set(
            data, fold_states[0].active_set, state=result_state
        )


class CrossValBackwardSelection(ForwardSelection):
    """Cross-validated backward selection using Gram-only training updates."""

    _default_criterion = BestRSSCriterion
    _reject_ic = True

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
            raise ValueError("CrossValBackwardSelection does not support warm starts.")

        criterion = self._init_criterion(data.make_full_data())
        full_active = list(range(data.gram_total.shape[0]))
        fold_states = [
            ForwardState.from_active_set(
                data.train_data_for_fold(k), full_active, self.tol
            )
            for k in range(data.n_folds)
        ]
        criterion.update_current(
            float(
                np.asarray(
                    criterion.evaluate(_cv_rss(fold_states, data), len(full_active))
                )
            )
        )

        steps = 0
        while fold_states[0].k and (max_steps is None or steps < max_steps):
            aggregated = _cv_backward_scores(fold_states, data, self.tol)
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
                    data, fold_states[0].active_set, self.tol
                )
            except ValueError:
                break
            criterion.update_current(best_score)
            steps += 1

        return _build_cv_state_from_active_set(
            data, fold_states[0].active_set, state=result_state
        )


class CrossValMixedSelection(ForwardSelection):
    """Cross-validated mixed selection using Gram-only training updates."""

    _default_criterion = BestRSSCriterion
    _reject_ic = True

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
            raise ValueError("CrossValMixedSelection does not support warm starts.")

        criterion = self._init_criterion(data.make_full_data())
        fold_states = [
            ForwardState.create(data.train_data_for_fold(k), self.tol)
            for k in range(data.n_folds)
        ]
        criterion.update_current(
            float(np.asarray(criterion.evaluate(_cv_rss(fold_states, data), 0)))
        )

        forward_steps = 0
        total_steps = 0
        while True:
            if max_total_steps is not None and total_steps >= max_total_steps:
                break
            scored = _cv_forward_scores(fold_states, data, self.tol)
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
                data, fold_states[0].active_set, self.tol
            )
            criterion.update_current(best_score)
            forward_steps += 1
            total_steps += 1

            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break

            while True:
                if max_total_steps is not None and total_steps >= max_total_steps:
                    break
                aggregated = _cv_backward_scores(fold_states, data, self.tol)
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
                        data, fold_states[0].active_set, self.tol
                    )
                except ValueError:
                    break
                criterion.update_current(best_score)
                total_steps += 1

        return _build_cv_state_from_active_set(
            data, fold_states[0].active_set, state=result_state
        )


class BeamCrossValForwardSelection(ForwardSelection):
    """Beam-search forward selection with CV scoring."""

    _default_criterion = BestRSSCriterion
    _reject_ic = True

    def __init__(self, *, beam_width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))

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

        criterion = self._init_criterion(data.make_full_data())
        fold_states = [
            ForwardState.create(data.train_data_for_fold(k), self.tol)
            for k in range(data.n_folds)
        ]
        initial_score = float(
            np.asarray(criterion.evaluate(_cv_rss(fold_states, data), 0))
        )
        criterion.update_current(initial_score)
        beams = [CVBeam(fold_states, criterion, criterion.current_value)]

        steps = 0
        while beams and (max_steps is None or steps < max_steps):
            candidates: list[CVBeam] = []
            for beam in beams:
                candidates.extend(
                    _cv_beam_forward_children(
                        beam, self.beam_width, data, self.tol
                    )
                )
            if not candidates:
                break
            beams = _cv_beam_prune(candidates, self.beam_width)
            steps += 1

        sel = min if criterion.minimize else max
        best = sel(beams, key=lambda b: b.score)
        return _build_cv_state_from_active_set(
            data, best.states[0].active_set, state=result_state
        )


class BeamCrossValBackwardSelection(ForwardSelection):
    """Beam-search backward selection with CV scoring."""

    _default_criterion = BestRSSCriterion
    _reject_ic = True

    def __init__(
        self, *, beam_width: int = 1, allow_worse: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))
        self.allow_worse = bool(allow_worse)

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

        criterion = self._init_criterion(data.make_full_data())
        full_active = list(range(data.gram_total.shape[0]))
        fold_states = [
            ForwardState.from_active_set(
                data.train_data_for_fold(k), full_active, self.tol
            )
            for k in range(data.n_folds)
        ]
        initial_score = float(
            np.asarray(
                criterion.evaluate(_cv_rss(fold_states, data), len(full_active))
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
                    )
                )
            if not candidates:
                break
            beams = _cv_beam_prune(candidates, self.beam_width)
            steps += 1

        sel = min if criterion.minimize else max
        best = sel(beams, key=lambda b: b.score)
        return _build_cv_state_from_active_set(
            data, best.states[0].active_set, state=result_state
        )


class BeamCrossValMixedSelection(ForwardSelection):
    """Beam-search mixed selection with CV scoring."""

    _default_criterion = BestRSSCriterion
    _reject_ic = True

    def __init__(self, *, beam_width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))

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

        criterion = self._init_criterion(data.make_full_data())
        fold_states = [
            ForwardState.create(data.train_data_for_fold(k), self.tol)
            for k in range(data.n_folds)
        ]
        initial_score = float(
            np.asarray(criterion.evaluate(_cv_rss(fold_states, data), 0))
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
            candidates: list[CVBeam] = []
            for beam in beams:
                candidates.extend(
                    _cv_beam_forward_children(
                        beam, self.beam_width, data, self.tol
                    )
                )
            if not candidates:
                break
            beams = _cv_beam_prune(candidates, self.beam_width)
            best = sel(beams, key=lambda b: b.score)
            forward_steps += 1
            total_ops += len(beams)
            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break

            new_beams: list[CVBeam] = []
            for beam in beams:
                while True:
                    if max_total_steps is not None and total_ops >= max_total_steps:
                        break
                    improved = _cv_beam_best_backward_child(beam, data, self.tol)
                    if improved is None:
                        break
                    beam = improved
                    total_ops += 1
                best = sel([best, beam], key=lambda b: b.score)
                new_beams.append(beam)
            beams = new_beams

        return _build_cv_state_from_active_set(
            data, best.states[0].active_set, state=result_state
        )


__all__ = [
    "CVBeam",
    "_cv_beam_prune",
    "_cv_beam_forward_children",
    "_cv_beam_backward_children",
    "_cv_beam_best_backward_child",
    "CrossValForwardSelection",
    "CrossValBackwardSelection",
    "CrossValMixedSelection",
    "BeamCrossValForwardSelection",
    "BeamCrossValBackwardSelection",
    "BeamCrossValMixedSelection",
]
