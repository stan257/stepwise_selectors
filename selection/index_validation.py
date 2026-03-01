"""Shared index-validation helpers."""

from __future__ import annotations

from numbers import Integral


def validate_feature_indices(
    indices: list[int], p: int, *, context: str
) -> list[int]:
    """Validate feature indices for active-set style APIs."""
    normalized: list[int] = []
    for idx in indices:
        if isinstance(idx, bool) or not isinstance(idx, Integral):
            raise TypeError(f"{context} indices must be integers.")
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= p:
            raise ValueError(
                f"{context} index {idx_int} is out of range for p={p}."
            )
        normalized.append(idx_int)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{context} contains duplicate feature indices.")
    return normalized


__all__ = ["validate_feature_indices"]
