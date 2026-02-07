"""Fast selection routines (default API)."""

from .fast_routines import (
    FastBackwardSelection as BackwardSelection,
    FastBeamBackwardSelection as BeamBackwardSelection,
    FastBeamCrossValBackwardSelection as BeamCrossValBackwardSelection,
    FastBeamCrossValForwardSelection as BeamCrossValForwardSelection,
    FastBeamCrossValMixedSelection as BeamCrossValMixedSelection,
    FastBeamForwardSelection as BeamForwardSelection,
    FastBeamMixedSelection as BeamMixedSelection,
    FastCrossValBackwardSelection as CrossValBackwardSelection,
    FastCrossValForwardSelection as CrossValForwardSelection,
    FastCrossValMixedSelection as CrossValMixedSelection,
    FastForwardSelection as ForwardSelection,
    FastMixedSelection as MixedSelection,
)

__all__ = [
    "BackwardSelection",
    "BeamBackwardSelection",
    "BeamCrossValBackwardSelection",
    "BeamCrossValForwardSelection",
    "BeamCrossValMixedSelection",
    "BeamForwardSelection",
    "BeamMixedSelection",
    "CrossValBackwardSelection",
    "CrossValForwardSelection",
    "CrossValMixedSelection",
    "ForwardSelection",
    "MixedSelection",
]
