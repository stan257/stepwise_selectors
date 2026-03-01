"""Selector and criterion adapters for benchmark execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from selection.criteria import (
    AICCriterion,
    AICcCriterion,
    BICCriterion,
    BestRSSCriterion,
    EBICCriterion,
    GCVCriterion,
    HQICCriterion,
)
from selection.definitions import CrossValGramData, GramData
from selection.routines import (
    BackwardSelection,
    BeamBackwardSelection,
    BeamCrossValBackwardSelection,
    BeamCrossValForwardSelection,
    BeamCrossValMixedSelection,
    BeamForwardSelection,
    BeamMixedSelection,
    CrossValBackwardSelection,
    CrossValForwardSelection,
    CrossValMixedSelection,
    ForwardSelection,
    MixedSelection,
)

from .datasets import BenchmarkDataset


SELECTOR_MAP = {
    "ForwardSelection": ForwardSelection,
    "BackwardSelection": BackwardSelection,
    "MixedSelection": MixedSelection,
    "BeamForwardSelection": BeamForwardSelection,
    "BeamBackwardSelection": BeamBackwardSelection,
    "BeamMixedSelection": BeamMixedSelection,
    "CrossValForwardSelection": CrossValForwardSelection,
    "CrossValBackwardSelection": CrossValBackwardSelection,
    "CrossValMixedSelection": CrossValMixedSelection,
    "BeamCrossValForwardSelection": BeamCrossValForwardSelection,
    "BeamCrossValBackwardSelection": BeamCrossValBackwardSelection,
    "BeamCrossValMixedSelection": BeamCrossValMixedSelection,
}

CRITERION_MAP = {
    "AICCriterion": AICCriterion,
    "AICcCriterion": AICcCriterion,
    "BICCriterion": BICCriterion,
    "BestRSSCriterion": BestRSSCriterion,
    "EBICCriterion": EBICCriterion,
    "GCVCriterion": GCVCriterion,
    "HQICCriterion": HQICCriterion,
}

CV_SELECTORS = {
    "CrossValForwardSelection",
    "CrossValBackwardSelection",
    "CrossValMixedSelection",
    "BeamCrossValForwardSelection",
    "BeamCrossValBackwardSelection",
    "BeamCrossValMixedSelection",
}


@dataclass(frozen=True)
class MethodRunResult:
    method_name: str
    selector_name: str
    active_set: list[int]
    state: object


def _to_gram_data(X: np.ndarray, y: np.ndarray) -> GramData:
    return GramData(
        gram=np.ascontiguousarray(X.T @ X),
        cov=np.ascontiguousarray(X.T @ y),
        y_norm=float(y @ y),
        n_samples=int(X.shape[0]),
    )


def _build_cv_data(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> CrossValGramData:
    if n_folds < 2:
        raise ValueError("cv_folds must be >= 2.")
    if n_folds > X.shape[0]:
        raise ValueError("cv_folds must not exceed number of training samples.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    fold_indices = np.array_split(perm, n_folds)
    folds = [_to_gram_data(X[idx], y[idx]) for idx in fold_indices if idx.size > 0]

    if len(folds) < 2:
        raise ValueError("Unable to build at least two non-empty CV folds.")
    return CrossValGramData(folds)


def _resolve_criterion_refs(selector_params: dict) -> dict:
    resolved = dict(selector_params)
    for key in ("criterion", "criterion_cls"):
        value = resolved.get(key)
        if isinstance(value, str):
            if value not in CRITERION_MAP:
                raise ValueError(
                    f"Unknown criterion {value!r}. Supported: {sorted(CRITERION_MAP)}"
                )
            resolved[key] = CRITERION_MAP[value]
    return resolved


def run_method(method_config: dict, dataset: BenchmarkDataset) -> MethodRunResult:
    method_name = str(method_config["name"])
    selector_name = str(method_config["selector"])

    selector_cls = SELECTOR_MAP.get(selector_name)
    if selector_cls is None:
        raise ValueError(
            f"Unknown selector {selector_name!r}. Supported: {sorted(SELECTOR_MAP)}"
        )

    selector_params = _resolve_criterion_refs(method_config.get("selector_params", {}))
    fit_params = dict(method_config.get("fit_params", {}))

    selector = selector_cls(**selector_params)

    if selector_name in CV_SELECTORS:
        n_folds = int(method_config.get("cv_folds", 5))
        cv_seed = int(method_config.get("cv_seed", dataset.seed + 17))
        data_for_fit = _build_cv_data(dataset.X_train, dataset.y_train, n_folds, cv_seed)
    else:
        data_for_fit = dataset.train_data

    state = selector.fit(data=data_for_fit, **fit_params)
    active_set = [int(i) for i in state.active_set]

    return MethodRunResult(
        method_name=method_name,
        selector_name=selector_name,
        active_set=active_set,
        state=state,
    )
