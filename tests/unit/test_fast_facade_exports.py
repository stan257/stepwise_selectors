import selection.fast_routines as fast


def test_fast_facade_exports_core_symbols():
    expected = [
        "FastForwardState",
        "FastForwardSelection",
        "FastBackwardSelection",
        "FastMixedSelection",
        "FastBeamForwardSelection",
        "FastBeamBackwardSelection",
        "FastBeamMixedSelection",
        "FastCrossValForwardSelection",
        "FastCrossValBackwardSelection",
        "FastCrossValMixedSelection",
        "FastBeamCrossValForwardSelection",
        "FastBeamCrossValBackwardSelection",
        "FastBeamCrossValMixedSelection",
        "FastCVBeam",
    ]
    for name in expected:
        assert hasattr(fast, name), f"selection.fast_routines missing {name}"


def test_fast_facade_keeps_monkeypatchable_cv_backward_hook():
    assert callable(fast._fast_cv_backward_scores)
    assert callable(fast._fast_cv_beam_backward_children)
