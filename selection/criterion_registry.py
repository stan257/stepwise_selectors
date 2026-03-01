"""Built-in string aliases for criterion constructors."""

from __future__ import annotations

from .criteria import (
    AICCriterion,
    AICcCriterion,
    BICCriterion,
    BestRSSCriterion,
    EBICCriterion,
    GCVCriterion,
    HQICCriterion,
)

_CANONICAL_CRITERIA = {
    "aic": AICCriterion,
    "aicc": AICcCriterion,
    "bic": BICCriterion,
    "ebic": EBICCriterion,
    "gcv": GCVCriterion,
    "hqic": HQICCriterion,
    "rss": BestRSSCriterion,
}

_CRITERION_ALIASES = {
    **_CANONICAL_CRITERIA,
    "best_rss": BestRSSCriterion,
    "best_rss_criterion": BestRSSCriterion,
    "bestrss": BestRSSCriterion,
    "bestrsscriterion": BestRSSCriterion,
    "aiccriterion": AICCriterion,
    "aicccriterion": AICcCriterion,
    "biccriterion": BICCriterion,
    "ebiccriterion": EBICCriterion,
    "gcvcriterion": GCVCriterion,
    "hqiccriterion": HQICCriterion,
}


def normalize_criterion_key(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Criterion key must be a string.")
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        raise ValueError("Criterion key must be non-empty.")
    while "__" in key:
        key = key.replace("__", "_")
    return key


def resolve_builtin_criterion(name: str):
    key = normalize_criterion_key(name)
    criterion_cls = _CRITERION_ALIASES.get(key)
    if criterion_cls is None:
        raise KeyError(key)
    return criterion_cls


def available_builtin_criteria() -> tuple[str, ...]:
    return tuple(sorted(_CANONICAL_CRITERIA))


__all__ = [
    "available_builtin_criteria",
    "normalize_criterion_key",
    "resolve_builtin_criterion",
]
