"""Shared helpers for budget-driven forward search semantics."""

from __future__ import annotations

from ..criteria import CriterionProtocol
from ..validation.interface_validation import (
    validate_bool,
    validate_optional_non_negative_int,
)


def resolve_forward_budget(
    *,
    budget: int | None,
    budget_name: str,
    selector_name: str,
    stop_on_no_improvement: bool,
) -> int | None:
    """Validate forward budget and enforce explicit budgeting by default."""
    validated_budget = validate_optional_non_negative_int(budget, name=budget_name)
    if validated_budget is None and not stop_on_no_improvement:
        raise ValueError(
            f"{selector_name} defaults to budget-driven search; "
            f"{budget_name} is required unless stop_on_no_improvement=True."
        )
    return validated_budget


def should_accept_forward_candidate(
    *,
    criterion: CriterionProtocol,
    candidate_score: float,
    stop_on_no_improvement: bool,
    incumbent_score: float | None = None,
) -> bool:
    """Decide whether a forward candidate should be accepted."""
    validate_bool(stop_on_no_improvement, name="stop_on_no_improvement")
    if not stop_on_no_improvement:
        return True
    return criterion.is_improvement(candidate_score, incumbent_score)
