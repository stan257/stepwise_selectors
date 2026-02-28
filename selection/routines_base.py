"""Shared validation and criterion guards for fast selection routines."""

from __future__ import annotations

from .criteria import (
    AICCriterion,
    AICcCriterion,
    BICCriterion,
    EBICCriterion,
    GCVCriterion,
    HQICCriterion,
)
from .definitions import CrossValGramData, GramData
from .state import CrossValSelectionState, SelectionState

_DISALLOWED_CV_CRITERIA = (
    AICCriterion,
    BICCriterion,
    AICcCriterion,
    HQICCriterion,
    EBICCriterion,
    GCVCriterion,
)


def _validate_state_target(
    state: SelectionState | None, data: GramData, *, selector_name: str
) -> SelectionState | None:
    if state is None:
        return None
    if not isinstance(state, SelectionState):
        raise TypeError(f"{selector_name} expects `state` to be a SelectionState.")
    if state.data is not data:
        raise ValueError(f"{selector_name} requires `state.data` to match `data`.")
    return state


def _validate_cv_state_target(
    state: CrossValSelectionState | None,
    data: CrossValGramData,
    *,
    selector_name: str,
) -> CrossValSelectionState | None:
    if state is None:
        return None
    if not isinstance(state, CrossValSelectionState):
        raise TypeError(
            f"{selector_name} expects `state` to be a CrossValSelectionState."
        )
    if state.data is not data:
        raise ValueError(f"{selector_name} requires `state.data` to match `data`.")
    return state


__all__ = [
    "_DISALLOWED_CV_CRITERIA",
    "_validate_state_target",
    "_validate_cv_state_target",
]
