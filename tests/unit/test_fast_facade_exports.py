import numpy as np

from selection.criteria import BestRSSCriterion
import selection.routines as public
import selection.routines_core as fast
from tests.helpers import make_cv_problem


def test_fast_facade_exports_match_public_aliases():
    expected_public = [
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
    for name in expected_public:
        assert name in fast.__all__, f"selection.routines_core missing {name} in __all__"
        if hasattr(public, name):
            assert getattr(fast, name) is getattr(public, name)


def test_fast_facade_keeps_monkeypatchable_cv_backward_hook(monkeypatch):
    cv_data = make_cv_problem(seed=404, folds=3, n=60, p=6, support=3)
    fold_states = [
        fast.ForwardState.from_active_set(
            cv_data.train_data_for_fold(k), [0], tol=1e-12
        )
        for k in range(cv_data.n_folds)
    ]
    criterion = BestRSSCriterion()
    base_score = float(
        np.asarray(criterion.evaluate(fast._cv_rss(fold_states, cv_data), 1))
    )
    criterion.update_current(base_score)
    beam = fast.CVBeam(fold_states, criterion, base_score)
    calls = {"n": 0}

    def fake_backward_scores(states, data, tol):
        calls["n"] += 1
        return np.array([base_score], dtype=float)

    monkeypatch.setattr(fast, "_cv_backward_scores", fake_backward_scores)
    children = fast._cv_beam_backward_children(
        beam, beam_width=1, data=cv_data, tol=1e-12, allow_worse=True
    )

    assert calls["n"] == 1
    assert len(children) == 1
    assert children[0].states[0].active_set == []
