"""Beam-search selectors for single-dataset GramData."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .criteria import SelectionCriterion
from .definitions import GramData
from .fast_base import _validate_state_target
from .fast_single import FastForwardSelection
from .fast_state import FastForwardState
from .state import SelectionState
from .topk import topk_indices


@dataclass
class FastBeam:
    state: FastForwardState
    criterion: SelectionCriterion
    score: float
    _signature: int = 0

    def __post_init__(self) -> None:
        sig = 0
        for idx in self.state.active_set:
            sig |= 1 << int(idx)
        self._signature = sig

    @property
    def signature(self) -> int:
        return self._signature


def _fast_beam_prune(beams: list[FastBeam], beam_limit: int) -> list[FastBeam]:
    if not beams:
        return []
    minimize = beams[0].criterion.minimize
    seen = set()
    result: list[FastBeam] = []
    for beam in sorted(beams, key=lambda b: b.score, reverse=not minimize):
        sig = beam.signature
        if sig in seen:
            continue
        seen.add(sig)
        result.append(beam)
        if len(result) >= beam_limit:
            break
    return result


def _fast_beam_forward_children(beam: FastBeam, beam_width: int) -> list[FastBeam]:
    scored = beam.state.candidate_scores()
    if scored is None:
        return []
    cand_idx, rss_new = scored
    crit_scores = np.asarray(beam.criterion.evaluate(rss_new, beam.state.k + 1))
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)
    children: list[FastBeam] = []
    for idx in order:
        candidate_score = float(crit_scores[idx])
        if not beam.criterion.is_improvement(candidate_score, beam.score):
            continue
        feat_idx = int(cand_idx[idx])
        child_state = beam.state.clone()
        child_state.apply_forward(feat_idx)
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(FastBeam(child_state, child_criterion, candidate_score))
    return children


def _fast_beam_backward_children(
    beam: FastBeam, beam_width: int, *, allow_worse: bool = False
) -> list[FastBeam]:
    rss_values = beam.state.backward_scores()
    if rss_values is None or not len(rss_values):
        return []
    crit_scores = np.asarray(
        beam.criterion.evaluate(rss_values, max(beam.state.k - 1, 0))
    )
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)
    children: list[FastBeam] = []
    for idx in order:
        candidate_score = float(crit_scores[idx])
        if not allow_worse and not beam.criterion.is_improvement(
            candidate_score, beam.score
        ):
            continue
        try:
            child_state = beam.state.clone()
            child_state.apply_backward(idx)
        except (ValueError, np.linalg.LinAlgError):
            continue
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(FastBeam(child_state, child_criterion, candidate_score))
    return children


def _fast_beam_best_backward_child(beam: FastBeam) -> FastBeam | None:
    for child in _fast_beam_backward_children(beam, 1, allow_worse=False):
        if beam.criterion.is_improvement(child.score, beam.score):
            return child
    return None


class FastBeamForwardSelection(FastForwardSelection):
    """Beam-search forward selection using fast Gram-only updates."""

    def __init__(self, *, beam_width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))

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
            raise ValueError("FastBeamForwardSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        initial = FastBeam(
            FastForwardState.create(data, self.tol), criterion, criterion.current_value
        )
        beams = [initial]

        steps = 0
        while beams and (max_steps is None or steps < max_steps):
            candidates: list[FastBeam] = []
            for beam in beams:
                candidates.extend(_fast_beam_forward_children(beam, self.beam_width))

            if not candidates:
                break
            beams = _fast_beam_prune(candidates, self.beam_width)
            steps += 1

        sel = min if criterion.minimize else max
        best = sel(beams, key=lambda b: b.score)
        result = result_state if result_state is not None else SelectionState(data)
        result.init_from_active_set(best.state.active_set)
        return result


class FastBeamBackwardSelection(FastForwardSelection):
    """Beam-search backward selection using fast Gram-only updates."""

    def __init__(self, *, beam_width: int = 1, allow_worse: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))
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
            raise ValueError("FastBeamBackwardSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        full_active = list(range(data.gram.shape[0]))
        initial_state = FastForwardState.from_active_set(data, full_active, self.tol)
        initial_score = float(
            np.asarray(criterion.evaluate(initial_state.rss, initial_state.k))
        )
        criterion.update_current(initial_score)
        beams = [FastBeam(initial_state, criterion, criterion.current_value)]

        steps = 0
        while beams and (max_steps is None or steps < max_steps):
            candidates: list[FastBeam] = []
            for beam in beams:
                candidates.extend(
                    _fast_beam_backward_children(
                        beam, self.beam_width, allow_worse=self.allow_worse
                    )
                )
            if not candidates:
                break
            beams = _fast_beam_prune(candidates, self.beam_width)
            steps += 1

        sel = min if criterion.minimize else max
        best = sel(beams, key=lambda b: b.score)
        result = result_state if result_state is not None else SelectionState(data)
        result.init_from_active_set(best.state.active_set)
        return result


class FastBeamMixedSelection(FastForwardSelection):
    """Beam-search mixed selection using fast Gram-only updates."""

    def __init__(self, *, beam_width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = max(1, int(beam_width))

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
            raise ValueError("FastBeamMixedSelection does not support warm starts.")

        criterion = self._init_criterion(data)
        initial = FastBeam(
            FastForwardState.create(data, self.tol), criterion, criterion.current_value
        )
        beams = [initial]
        best = initial
        sel = min if criterion.minimize else max

        forward_steps = 0
        total_ops = 0
        while True:
            if max_total_steps is not None and total_ops >= max_total_steps:
                break
            candidates: list[FastBeam] = []
            for beam in beams:
                candidates.extend(_fast_beam_forward_children(beam, self.beam_width))
            if not candidates:
                break
            beams = _fast_beam_prune(candidates, self.beam_width)
            best = sel(beams, key=lambda b: b.score)
            forward_steps += 1
            total_ops += len(beams)
            if max_forward_steps is not None and forward_steps >= max_forward_steps:
                break

            new_beams: list[FastBeam] = []
            for beam in beams:
                while True:
                    if max_total_steps is not None and total_ops >= max_total_steps:
                        break
                    improved = _fast_beam_best_backward_child(beam)
                    if improved is None:
                        break
                    beam = improved
                    total_ops += 1
                best = sel([best, beam], key=lambda b: b.score)
                new_beams.append(beam)
            beams = new_beams

        result = result_state if result_state is not None else SelectionState(data)
        result.init_from_active_set(best.state.active_set)
        return result


__all__ = [
    "FastBeam",
    "_fast_beam_prune",
    "_fast_beam_forward_children",
    "_fast_beam_backward_children",
    "_fast_beam_best_backward_child",
    "FastBeamForwardSelection",
    "FastBeamBackwardSelection",
    "FastBeamMixedSelection",
]
