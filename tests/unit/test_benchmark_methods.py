import pytest

from benchmarks.methods import _resolve_criterion_refs
from selection.criteria import AICCriterion


def test_resolve_criterion_refs_maps_criterion_string():
    resolved = _resolve_criterion_refs({"criterion": "AICCriterion", "beam_width": 2})
    assert resolved["criterion"] is AICCriterion
    assert resolved["beam_width"] == 2


def test_resolve_criterion_refs_rejects_criterion_cls_key():
    with pytest.raises(
        ValueError,
        match=r"unsupported key `criterion_cls`; use `criterion`",
    ):
        _resolve_criterion_refs({"criterion_cls": "AICCriterion"})


def test_resolve_criterion_refs_rejects_unknown_criterion_string():
    with pytest.raises(ValueError, match=r"Unknown criterion"):
        _resolve_criterion_refs({"criterion": "NotACriterion"})
