"""Exact-k forward-search helpers for benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from selection import CrossValGramData, GramData
from selection.core.incremental_solver import IncrementalSolver
from selection.core.state_cv import CrossValSelectionState
from selection.core.state_single import SelectionState
from selection.criteria import CriterionProtocol
from selection.selectors.beam_pruning import prune_unique_beams
from selection.selectors.routines_beam import BeamForwardSelection
from selection.selectors.routines_cv_greedy import CrossValForwardSelection
from selection.selectors.routines_cv_scoring import (
    _build_cv_state_from_active_set,
    _build_fold_states,
    _cv_forward_scores,
    _cv_rss,
    _rebuild_states,
)
from selection.selectors.routines_greedy import ForwardSelection
from selection.selectors.topk import topk_indices
from selection.validation.interface_validation import validate_optional_non_negative_int


@dataclass
class _ExactBeam:
    state: IncrementalSolver
    criterion: CriterionProtocol
    score: float
    _signature: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self._signature = frozenset(int(idx) for idx in self.state.active_set)

    @property
    def signature(self) -> frozenset[int]:
        return self._signature


def _exact_budget(max_steps: int | None) -> int:
    steps = validate_optional_non_negative_int(max_steps, name="max_steps")
    if steps is None:
        raise ValueError("Exact-k helpers require max_steps to be an integer budget.")
    return int(steps)


def fit_forward_exact_k(
    selector: ForwardSelection,
    *,
    data: GramData,
    max_steps: int,
) -> SelectionState:
    """Fit a forward selector to an exact feature budget.

    This follows the selector's per-step scoring rule but does not stop early on
    criterion non-improvement. It is intended for matched-k benchmark sweeps.
    """
    budget = _exact_budget(max_steps)
    criterion = selector._init_criterion(data)
    work_state = IncrementalSolver.create(
        data,
        selector.tol,
        solver_policy=selector.solver_policy,
        ridge_alpha=selector.ridge_alpha,
        pinv_rcond=selector.pinv_rcond,
    )

    while work_state.k < budget:
        scored = work_state.candidate_scores()
        if scored is None:
            break
        candidates, rss_new = scored
        best_idx, best_score = criterion.best_candidate(rss_new, work_state.k + 1)
        feat_idx = int(candidates[best_idx])
        work_state.apply_forward(feat_idx)
        criterion.update_current(best_score)

    result = SelectionState(
        data,
        solver_policy=selector.solver_policy,
        ridge_alpha=selector.ridge_alpha,
        pinv_rcond=selector.pinv_rcond,
    )
    result.init_from_active_set(work_state.active_set)
    return result


def fit_cv_forward_exact_k(
    selector: CrossValForwardSelection,
    *,
    data: CrossValGramData,
    max_steps: int,
) -> CrossValSelectionState:
    """Fit CV forward selection to an exact support size budget."""
    budget = _exact_budget(max_steps)
    criterion = selector._init_criterion(data.make_full_data())
    fold_states = _build_fold_states(
        data,
        tol=selector.tol,
        solver_policy=selector.solver_policy,
        ridge_alpha=selector.ridge_alpha,
        pinv_rcond=selector.pinv_rcond,
    )
    criterion.update_current(
        float(
            np.asarray(
                criterion.evaluate(
                    _cv_rss(
                        fold_states,
                        data,
                        cv_aggregation=selector.cv_aggregation,
                    ),
                    0,
                )
            )
        )
    )

    steps = 0
    while steps < budget:
        scored = _cv_forward_scores(
            fold_states,
            data,
            selector.tol,
            cv_aggregation=selector.cv_aggregation,
        )
        if scored is None:
            break
        candidates, aggregated = scored
        best_idx, best_score = criterion.best_candidate(
            aggregated,
            len(fold_states[0].active_set) + 1,
        )
        feat_idx = int(candidates[best_idx])
        for fold_state in fold_states:
            fold_state.apply_forward(feat_idx)
        fold_states = _rebuild_states(
            data,
            fold_states[0].active_set,
            selector.tol,
            solver_policy=selector.solver_policy,
            ridge_alpha=selector.ridge_alpha,
            pinv_rcond=selector.pinv_rcond,
        )
        criterion.update_current(best_score)
        steps += 1

    return _build_cv_state_from_active_set(
        data,
        fold_states[0].active_set,
        solver_policy=selector.solver_policy,
        ridge_alpha=selector.ridge_alpha,
        pinv_rcond=selector.pinv_rcond,
    )


def _beam_forward_children_exact_k(
    beam: _ExactBeam,
    *,
    beam_width: int,
) -> list[_ExactBeam]:
    scored = beam.state.candidate_scores()
    if scored is None:
        return []
    cand_idx, rss_new = scored
    crit_scores = np.asarray(beam.criterion.evaluate(rss_new, beam.state.k + 1))
    order = topk_indices(crit_scores, beam_width, minimize=beam.criterion.minimize)
    children: list[_ExactBeam] = []
    for idx in order:
        candidate_score = float(crit_scores[idx])
        feat_idx = int(cand_idx[idx])
        child_state = beam.state.clone()
        child_state.apply_forward(feat_idx)
        child_criterion = beam.criterion.clone()
        child_criterion.update_current(candidate_score)
        children.append(_ExactBeam(child_state, child_criterion, candidate_score))
    return children


def fit_beam_forward_exact_k(
    selector: BeamForwardSelection,
    *,
    data: GramData,
    max_steps: int,
) -> SelectionState:
    """Fit beam forward selection to an exact support size budget."""
    budget = _exact_budget(max_steps)
    criterion = selector._init_criterion(data)
    initial = _ExactBeam(
        IncrementalSolver.create(
            data,
            selector.tol,
            solver_policy=selector.solver_policy,
            ridge_alpha=selector.ridge_alpha,
            pinv_rcond=selector.pinv_rcond,
        ),
        criterion,
        criterion.current_value,
    )
    beams = [initial]

    steps = 0
    while beams and steps < budget:
        candidates: list[_ExactBeam] = []
        for beam in beams:
            candidates.extend(
                _beam_forward_children_exact_k(beam, beam_width=selector.beam_width)
            )
        if not candidates:
            break
        beams = prune_unique_beams(candidates, selector.beam_width)
        steps += 1

    sel = min if criterion.minimize else max
    best = sel(beams, key=lambda beam: beam.score)
    result = SelectionState(
        data,
        solver_policy=selector.solver_policy,
        ridge_alpha=selector.ridge_alpha,
        pinv_rcond=selector.pinv_rcond,
    )
    result.init_from_active_set(best.state.active_set)
    return result


__all__ = [
    "fit_beam_forward_exact_k",
    "fit_cv_forward_exact_k",
    "fit_forward_exact_k",
]
