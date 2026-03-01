"""Shared beam-pruning helper for deterministic deduplication."""

from __future__ import annotations

from typing import Protocol, TypeVar


class _CriterionLike(Protocol):
    minimize: bool


class BeamLike(Protocol):
    score: float
    signature: int
    criterion: _CriterionLike


TBeam = TypeVar("TBeam", bound=BeamLike)


def prune_unique_beams(beams: list[TBeam], beam_limit: int) -> list[TBeam]:
    """Keep top-scoring beams while deduplicating by active-set signature."""
    if not beams:
        return []
    minimize = beams[0].criterion.minimize
    seen = set()
    result: list[TBeam] = []
    for beam in sorted(beams, key=lambda b: b.score, reverse=not minimize):
        sig = beam.signature
        if sig in seen:
            continue
        seen.add(sig)
        result.append(beam)
        if len(result) >= beam_limit:
            break
    return result


__all__ = ["prune_unique_beams", "BeamLike"]
