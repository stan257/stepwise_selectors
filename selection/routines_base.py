"""Shared validation and criterion guards for fast selection routines."""

from __future__ import annotations

import inspect
from typing import Any

from .criteria import (
    BestRSSCriterion,
    CriterionProtocol,
)
from .definitions import CrossValGramData, GramData
from .state_cv import CrossValSelectionState
from .state_single import SelectionState


def _validate_criterion_protocol(
    criterion: Any, *, selector_name: str
) -> CriterionProtocol:
    if isinstance(criterion, CriterionProtocol):
        return criterion
    raise TypeError(
        f"{selector_name} requires a criterion implementing "
        "`CriterionProtocol` (evaluate, best_candidate, is_improvement, "
        "update_current, clone, minimize, current_value)."
    )


def _inject_default_criterion_params(
    provider: Any, params: dict[str, Any], data: GramData
) -> dict[str, Any]:
    resolved = dict(params)
    try:
        sig = inspect.signature(provider)
    except (TypeError, ValueError):
        return resolved
    has_var_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )
    # Inject defaults for kwargs-only factories/callables, but not for classes.
    # Many criterion classes expose **kwargs for tolerance knobs and should not
    # receive n_samples/p unless they declare those names explicitly.
    accepts_kwargs = (not inspect.isclass(provider)) and has_var_kwargs
    if (
        ("n_samples" in sig.parameters or accepts_kwargs)
        and "n_samples" not in resolved
    ):
        resolved["n_samples"] = data.n_samples
    if ("p" in sig.parameters or accepts_kwargs) and "p" not in resolved:
        resolved["p"] = data.gram.shape[0]
    return resolved


def _resolve_criterion(
    *,
    selector_name: str,
    data: GramData,
    criterion: Any,
    criterion_cls: Any,
    default_criterion_cls: Any,
    criterion_kwargs: dict[str, Any] | None,
) -> CriterionProtocol:
    provider = criterion if criterion is not None else (criterion_cls or default_criterion_cls)
    params = dict(criterion_kwargs or {})

    if not inspect.isclass(provider) and isinstance(provider, CriterionProtocol):
        if params:
            raise ValueError(
                f"{selector_name} does not accept `criterion_kwargs` when `criterion` "
                "is an instantiated object."
            )
        resolved = provider.clone()
    else:
        if not callable(provider):
            raise TypeError(
                f"{selector_name} requires `criterion`/`criterion_cls` to be callable "
                "or a criterion instance."
            )
        call_params = _inject_default_criterion_params(provider, params, data)
        try:
            resolved = provider(**call_params)
        except TypeError as err:
            provider_name = getattr(provider, "__name__", type(provider).__name__)
            raise TypeError(
                f"{selector_name} could not construct criterion from "
                f"{provider_name}: {err}"
            ) from err

    return _validate_criterion_protocol(resolved, selector_name=selector_name)


def _validate_cv_criterion(
    criterion: CriterionProtocol, *, selector_name: str
) -> None:
    if bool(getattr(criterion, "cv_compatible", True)):
        return
    raise ValueError(
        f"{selector_name} uses cross-validation for regularisation; "
        f"{type(criterion).__name__} is not supported for CV selection routines. "
        f"Use {BestRSSCriterion.__name__} (the default) instead."
    )


def _validate_state_target(
    state: SelectionState | None, data: GramData, *, selector_name: str
) -> SelectionState | None:
    if state is None:
        return None
    if not isinstance(state, SelectionState):
        raise TypeError(f"{selector_name} expects `state` to be a SelectionState.")
    if state.data is not data:
        raise ValueError(f"{selector_name} requires `state.data` to match `data`.")
    return state


def _validate_cv_state_target(
    state: CrossValSelectionState | None,
    data: CrossValGramData,
    *,
    selector_name: str,
) -> CrossValSelectionState | None:
    if state is None:
        return None
    if not isinstance(state, CrossValSelectionState):
        raise TypeError(
            f"{selector_name} expects `state` to be a CrossValSelectionState."
        )
    if state.data is not data:
        raise ValueError(f"{selector_name} requires `state.data` to match `data`.")
    return state


__all__ = [
    "_inject_default_criterion_params",
    "_resolve_criterion",
    "_validate_criterion_protocol",
    "_validate_cv_criterion",
    "_validate_state_target",
    "_validate_cv_state_target",
]
