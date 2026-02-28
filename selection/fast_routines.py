"""Compatibility facade for fast selection routines.

The implementation is split across focused modules. This file preserves the
historical import surface (`selection.fast_routines`) for downstream users.
"""

from __future__ import annotations

import numpy as np

from .definitions import CrossValGramData
from .fast_beam_cv import (
    FastBeamCrossValBackwardSelection,
    FastBeamCrossValForwardSelection,
    FastBeamCrossValMixedSelection,
    FastCVBeam,
    _fast_cv_beam_best_backward_child,
    _fast_cv_beam_forward_children,
    _fast_cv_beam_prune,
)
from .fast_beam_single import (
    FastBeam,
    FastBeamBackwardSelection,
    FastBeamForwardSelection,
    FastBeamMixedSelection,
    _fast_beam_backward_children,
    _fast_beam_best_backward_child,
    _fast_beam_forward_children,
    _fast_beam_prune,
)
from .fast_cv_core import (
    _build_cv_state_from_active_set,
    _fast_cv_backward_scores,
    _fast_cv_forward_scores,
    _fast_cv_rss,
    _rebuild_fast_states,
)
from .fast_single import FastBackwardSelection, FastForwardSelection, FastMixedSelection
from .fast_state import FastForwardState
from .topk import topk_indices
from .fast_base import _validate_cv_state_target, _validate_state_target

# CV greedy selectors live with other CV routines.
from .fast_beam_cv import (
    FastCrossValBackwardSelection,
    FastCrossValForwardSelection,
    FastCrossValMixedSelection,
)


def _fast_cv_beam_backward_children(
    beam: FastCVBeam,
    beam_width: int,
    data: CrossValGramData,
    tol: float,
    *,
    allow_worse: bool = False,
) -> list[FastCVBeam]:
    """Compatibility wrapper for monkeypatchable CV backward beam scoring.

    Tests patch `selection.fast_routines._fast_cv_backward_scores`; this wrapper
    intentionally resolves that name from this module.
    """

    aggregated = _fast_cv_backward_scores(beam.states, data, tol)
    if aggregated is None or not len(aggregated):
        return []
    crit_scores = np.asarray(
        beam.criterion.evaluate(aggregated, max(len(beam.states[0].active_set) - 1, 0))
    )
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)
    children: list[FastCVBeam] = []
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
            child_states = _rebuild_fast_states(data, child_states[0].active_set, tol)
        except ValueError:
            continue
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(FastCVBeam(child_states, child_criterion, candidate_score))
    return children


__all__ = [
    "FastForwardState",
    "FastForwardSelection",
    "FastBackwardSelection",
    "FastMixedSelection",
    "FastBeam",
    "FastBeamForwardSelection",
    "FastBeamBackwardSelection",
    "FastBeamMixedSelection",
    "FastCrossValForwardSelection",
    "FastCrossValBackwardSelection",
    "FastCrossValMixedSelection",
    "FastCVBeam",
    "FastBeamCrossValForwardSelection",
    "FastBeamCrossValBackwardSelection",
    "FastBeamCrossValMixedSelection",
    "_validate_state_target",
    "_validate_cv_state_target",
    "_fast_beam_prune",
    "_fast_beam_forward_children",
    "_fast_beam_backward_children",
    "_fast_beam_best_backward_child",
    "_fast_cv_rss",
    "_fast_cv_beam_prune",
    "_fast_cv_beam_forward_children",
    "_fast_cv_beam_backward_children",
    "_fast_cv_beam_best_backward_child",
    "_rebuild_fast_states",
    "_build_cv_state_from_active_set",
    "_fast_cv_forward_scores",
    "_fast_cv_backward_scores",
]
