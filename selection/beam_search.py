from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .constants import ABS_TOL
from .criteria import SelectionCriterion
from .state import SelectionState


@dataclass
class Beam:
    state: SelectionState
    criterion: SelectionCriterion
    score: float
    _signature: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        self._signature = tuple(sorted(self.state.active_set))

    @property
    def signature(self) -> tuple[int, ...]:
        return self._signature


@dataclass
class BeamManager:
    beams: list[Beam]
    num_beams: int

    def expand(self, expand_fn: Callable[[Beam], list[Beam]]) -> bool:
        candidates: list[Beam] = []
        for beam in self.beams:
            candidates.extend(expand_fn(beam))
        if not candidates:
            self.beams = []
            return False
        self.beams = beam_prune(candidates, self.num_beams)
        return bool(self.beams)

    def best_state(self) -> SelectionState:
        return min(self.beams, key=lambda b: b.score).state

    def __len__(self):
        return len(self.beams)


def beam_forward_children(
    beam: Beam, beam_width: int, tol: float = ABS_TOL
) -> list[Beam]:
    cache = beam.state.compute_forward_deltas(tol)
    if cache is None or not cache.candidates.size:
        return []
    rss_scores = np.asarray(cache.rss_new)
    crit_scores = np.asarray(beam.criterion.evaluate(rss_scores, cache.active_rk + 1))
    order = np.argsort(crit_scores)
    children: list[Beam] = []
    for idx in order[:beam_width]:
        candidate_score = float(crit_scores[idx])
        if not beam.criterion.is_improvement(candidate_score, beam.score):
            continue
        child_state = beam.state.clone()
        child_state.apply_forward_step(cache, idx)
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(Beam(child_state, child_criterion, candidate_score))
    return children


def beam_backward_children(
    beam: Beam, beam_width: int, tol: float = ABS_TOL, allow_worse: bool = False
) -> list[Beam]:
    rss_values = beam.state.compute_backward_scores()
    if rss_values is None or not len(rss_values):
        return []
    rss_scores = np.asarray(rss_values)
    k = max(len(beam.state.active_set) - 1, 0)
    crit_scores = np.asarray(beam.criterion.evaluate(rss_scores, k))
    order = np.argsort(crit_scores)
    children: list[Beam] = []
    for idx in order[:beam_width]:
        candidate_score = float(crit_scores[idx])
        if not allow_worse and not beam.criterion.is_improvement(
            candidate_score, beam.score
        ):
            continue
        try:
            child_state = beam.state.clone()
            child_state.apply_backward_step(idx, tol)
        except Exception:
            continue
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(Beam(child_state, child_criterion, candidate_score))
    return children


def beam_best_backward_child(beam: Beam, tol: float = ABS_TOL) -> Beam | None:
    children = beam_backward_children(beam, 1, tol)
    for child in children:
        if beam.criterion.is_improvement(child.score, beam.score):
            return child
    return None


def beam_prune(candidates: list[Beam], beam_limit: int) -> list[Beam]:
    seen = set()
    result: list[Beam] = []
    for beam in sorted(candidates, key=lambda b: b.score):
        sig = beam.signature
        if sig in seen:
            continue
        seen.add(sig)
        result.append(beam)
        if len(result) >= beam_limit:
            break
    return result
