from collections.abc import Callable

import numpy as np

from .beam_search import Beam, BeamManager
from .state import SelectionState
from .cv_utils import (
    CVBackwardScores,
    CVForwardScores,
    cv_backward_scores,
    cv_forward_scores,
)
from .state import CrossValSelectionState
from .topk import topk_indices


def run_beam_search(
    initial_beam: Beam,
    *,
    beam_width: int,
    max_steps: int | None,
    expand_fn: Callable[[Beam], list[Beam]],
    track_best=True,
) -> SelectionState:
    manager = BeamManager([initial_beam], beam_width)
    best_beam = manager.beams[0]
    sel = min if initial_beam.criterion.minimize else max
    steps = 0
    while True:
        if max_steps is not None and steps >= max_steps:
            break
        expanded = manager.expand(expand_fn)
        if not expanded:
            break
        if track_best:
            best_beam = sel(manager.beams, key=lambda b: b.score)
        steps += 1
    return best_beam.state


def run_beam_mixed(
    initial_beam: Beam,
    *,
    beam_width: int,
    max_forward_steps: int | None,
    max_total_steps: int | None,
    forward_expand: Callable[[Beam], list[Beam]],
    backward_improve: Callable[[Beam], Beam | None],
) -> SelectionState:
    manager = BeamManager([initial_beam], beam_width)
    best_beam = manager.beams[0]
    sel = min if initial_beam.criterion.minimize else max
    forward_steps = 0
    total_operations = 0
    while True:
        if max_total_steps is not None and total_operations >= max_total_steps:
            break
        expanded = manager.expand(forward_expand)
        if not expanded:
            break
        best_beam = sel(manager.beams, key=lambda b: b.score)
        forward_steps += 1
        total_operations += len(manager)
        if max_forward_steps is not None and forward_steps >= max_forward_steps:
            break

        new_beams: list[Beam] = []
        for beam in manager.beams:
            while True:
                if max_total_steps is not None and total_operations >= max_total_steps:
                    break
                improved_child = backward_improve(beam)
                if improved_child is None:
                    break
                beam = improved_child
                total_operations += 1
            best_beam = sel([best_beam, beam], key=lambda b: b.score)
            new_beams.append(beam)
        manager.beams = new_beams
    return best_beam.state


def cv_beam_forward_children(
    beam: Beam, beam_width: int, tol: float
) -> list[Beam]:
    cv_state = beam.state  # type: ignore[assignment]
    forward_data: CVForwardScores | None = cv_forward_scores(cv_state, tol)
    if forward_data is None:
        return []
    fold_caches = forward_data.fold_caches
    candidates = forward_data.candidates
    candidate_indices = forward_data.candidate_indices
    aggregated = forward_data.aggregated_rss
    crit_scores = np.asarray(
        beam.criterion.evaluate(aggregated, len(cv_state.active_set) + 1)
    )
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)

    children: list[Beam] = []
    for idx in order:
        candidate_score = float(crit_scores[idx])
        if not beam.criterion.is_improvement(candidate_score, beam.score):
            continue
        candidate = candidates[idx]

        child_state: CrossValSelectionState = cv_state.clone()
        for fold_idx, fold_state in enumerate(child_state.train_states):
            cache = fold_caches[fold_idx]
            idx_local = int(candidate_indices[fold_idx][idx])
            fold_state.apply_forward_step(cache, idx_local)
        child_state._sync_active_set()
        child_state.recompute_oos_rss()

        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(Beam(child_state, child_criterion, candidate_score))
    return children


def cv_beam_backward_children(
    beam: Beam, beam_width: int, tol: float, allow_worse: bool
) -> list[Beam]:
    """Build CV beam children by removing one active feature across all folds."""
    cv_state = beam.state  # type: ignore[assignment]
    backward_data: CVBackwardScores | None = cv_backward_scores(cv_state, tol)
    if backward_data is None:
        return []
    aggregated_rss = backward_data.aggregated_rss
    crit_scores = np.asarray(
        beam.criterion.evaluate(aggregated_rss, len(cv_state.active_set) - 1)
    )
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)

    children: list[Beam] = []
    for idx in order:
        candidate_score = float(crit_scores[idx])
        if not allow_worse and not beam.criterion.is_improvement(
            candidate_score, beam.score
        ):
            continue
        try:
            child_state = cv_state.clone()  # type: ignore[assignment]
            child_state.apply_backward_step(idx, tol)
        except Exception:
            continue
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(Beam(child_state, child_criterion, candidate_score))
    return children


def cv_beam_best_backward_child(beam: Beam, tol: float) -> Beam | None:
    """Return the first backward CV child that improves the given beam, if any."""
    for child in cv_beam_backward_children(beam, 1, tol, allow_worse=False):
        if beam.criterion.is_improvement(child.score, beam.score):
            return child
    return None
