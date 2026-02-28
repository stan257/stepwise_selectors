import selection.routines_core as fast


def test_fast_facade_exports_core_symbols():
    expected = [
        "ForwardState",
        "ForwardSelection",
        "BackwardSelection",
        "MixedSelection",
        "BeamForwardSelection",
        "BeamBackwardSelection",
        "BeamMixedSelection",
        "CrossValForwardSelection",
        "CrossValBackwardSelection",
        "CrossValMixedSelection",
        "BeamCrossValForwardSelection",
        "BeamCrossValBackwardSelection",
        "BeamCrossValMixedSelection",
        "CVBeam",
    ]
    for name in expected:
        assert hasattr(fast, name), f"selection.routines_core missing {name}"


def test_fast_facade_keeps_monkeypatchable_cv_backward_hook():
    assert callable(fast._cv_backward_scores)
    assert callable(fast._cv_beam_backward_children)
