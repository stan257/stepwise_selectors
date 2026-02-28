"""Compatibility facade re-exporting the core selector surface."""

from __future__ import annotations

from .routines_cv import (
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
)
from .routines_beam import (
    BeamBackwardSelection,
    BeamForwardSelection,
    BeamMixedSelection,
)
from .routines_greedy import BackwardSelection, ForwardSelection, MixedSelection
from .forward_state import ForwardState

# CV greedy selectors live with other CV routines.
from .routines_cv import (
    CrossValBackwardSelection,
    CrossValForwardSelection,
    CrossValMixedSelection,
)


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
    "BeamCrossValForwardSelection",
    "BeamCrossValBackwardSelection",
    "BeamCrossValMixedSelection",
]
