"""Core selector facade.

The implementation is split across focused modules; this file re-exports the
shared selector surface and selected internal hooks.
"""

from __future__ import annotations

import numpy as np

from .definitions import CrossValGramData
from .routines_cv import (
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
    CVBeam,
    _cv_beam_best_backward_child,
    _cv_beam_forward_children,
    _cv_beam_prune,
)
from .routines_beam import (
    Beam,
    BeamBackwardSelection,
    BeamForwardSelection,
    BeamMixedSelection,
    _beam_backward_children,
    _beam_best_backward_child,
    _beam_forward_children,
    _beam_prune,
)
from .routines_cv_scoring import (
    _build_cv_state_from_active_set,
    _cv_backward_scores,
    _cv_forward_scores,
    _cv_rss,
    _rebuild_states,
)
from .routines_greedy import BackwardSelection, ForwardSelection, MixedSelection
from .forward_state import ForwardState
from .topk import topk_indices
from .routines_base import _validate_cv_state_target, _validate_state_target

# CV greedy selectors live with other CV routines.
from .routines_cv import (
    CrossValBackwardSelection,
    CrossValForwardSelection,
    CrossValMixedSelection,
)


def _cv_beam_backward_children(
    beam: CVBeam,
    beam_width: int,
    data: CrossValGramData,
    tol: float,
    *,
    allow_worse: bool = False,
) -> list[CVBeam]:
    """Compatibility wrapper for monkeypatchable CV backward beam scoring.

    Tests patch `selection.routines_core._cv_backward_scores`; this wrapper
    intentionally resolves that name from this module.
    """

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


__all__ = [
    "ForwardState",
    "ForwardSelection",
    "BackwardSelection",
    "MixedSelection",
    "Beam",
    "BeamForwardSelection",
    "BeamBackwardSelection",
    "BeamMixedSelection",
    "CrossValForwardSelection",
    "CrossValBackwardSelection",
    "CrossValMixedSelection",
    "CVBeam",
    "BeamCrossValForwardSelection",
    "BeamCrossValBackwardSelection",
    "BeamCrossValMixedSelection",
    "_validate_state_target",
    "_validate_cv_state_target",
    "_beam_prune",
    "_beam_forward_children",
    "_beam_backward_children",
    "_beam_best_backward_child",
    "_cv_rss",
    "_cv_beam_prune",
    "_cv_beam_forward_children",
    "_cv_beam_backward_children",
    "_cv_beam_best_backward_child",
    "_rebuild_states",
    "_build_cv_state_from_active_set",
    "_cv_forward_scores",
    "_cv_backward_scores",
]
