"""Shared validation helpers for public-facing selection interfaces."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def validate_choice(value: str, *, name: str, choices: set[str]) -> str:
    """Validate a string option constrained to a finite set."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip().lower()
    if normalized not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {allowed}.")
    return normalized


def validate_optional_non_negative_int(
    value: int | None, *, name: str
) -> int | None:
    """Validate an optional non-negative integer hyperparameter."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer or None.")
    value_int = int(value)
    if value_int < 0:
        raise ValueError(f"{name} must be >= 0.")
    return value_int


def validate_positive_int(value: int, *, name: str) -> int:
    """Validate a strictly positive integer hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value_int


def validate_bool(value: bool, *, name: str) -> bool:
    """Validate a strict boolean flag (no truthy/falsy coercion)."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value


def validate_positive_finite_float(value: float, *, name: str) -> float:
    """Validate a finite positive float hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite.")
    if value_float <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return value_float


def validate_non_negative_finite_float(value: float, *, name: str) -> float:
    """Validate a finite non-negative float hyperparameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite.")
    if value_float < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return value_float


__all__ = [
    "validate_choice",
    "validate_optional_non_negative_int",
    "validate_positive_int",
    "validate_bool",
    "validate_positive_finite_float",
    "validate_non_negative_finite_float",
]
