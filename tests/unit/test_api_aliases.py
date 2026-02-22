from selection import routines
from selection.fast_routines import (
    FastBackwardSelection,
    FastBeamBackwardSelection,
    FastBeamCrossValBackwardSelection,
    FastBeamCrossValForwardSelection,
    FastBeamCrossValMixedSelection,
    FastBeamForwardSelection,
    FastBeamMixedSelection,
    FastCrossValBackwardSelection,
    FastCrossValForwardSelection,
    FastCrossValMixedSelection,
    FastForwardSelection,
    FastMixedSelection,
)
from selection.grouped_routines import (
    FastGroupBackwardSelection,
    FastGroupForwardSelection,
    GroupBackwardSelection,
    GroupForwardSelection,
)


def test_routines_module_aliases_fast_classes():
    assert routines.ForwardSelection is FastForwardSelection
    assert routines.BackwardSelection is FastBackwardSelection
    assert routines.MixedSelection is FastMixedSelection
    assert routines.BeamForwardSelection is FastBeamForwardSelection
    assert routines.BeamBackwardSelection is FastBeamBackwardSelection
    assert routines.BeamMixedSelection is FastBeamMixedSelection
    assert routines.CrossValForwardSelection is FastCrossValForwardSelection
    assert routines.CrossValBackwardSelection is FastCrossValBackwardSelection
    assert routines.CrossValMixedSelection is FastCrossValMixedSelection
    assert routines.BeamCrossValForwardSelection is FastBeamCrossValForwardSelection
    assert routines.BeamCrossValBackwardSelection is FastBeamCrossValBackwardSelection
    assert routines.BeamCrossValMixedSelection is FastBeamCrossValMixedSelection


def test_grouped_aliases_fast_classes():
    assert GroupForwardSelection is FastGroupForwardSelection
    assert GroupBackwardSelection is FastGroupBackwardSelection
