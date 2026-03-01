"""Common benchmark metrics for model-selection experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import BenchmarkDataset


@dataclass(frozen=True)
class SplitMetrics:
    rss: float
    mse: float


def _regression_metrics(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> SplitMetrics:
    residual = y - X @ beta
    rss = float(residual @ residual)
    mse = rss / float(X.shape[0])
    return SplitMetrics(rss=rss, mse=mse)


def _support_metrics(selected: np.ndarray, true_support: np.ndarray) -> dict[str, float]:
    selected_set = set(int(i) for i in selected.tolist())
    true_set = set(int(i) for i in true_support.tolist())

    if not selected_set:
        precision = 1.0 if not true_set else 0.0
    else:
        precision = len(selected_set & true_set) / len(selected_set)

    recall = 1.0 if not true_set else len(selected_set & true_set) / len(true_set)

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
    }


def collect_metrics(state, dataset: BenchmarkDataset, elapsed_ms: float) -> dict:
    """Compute split-level error and support metrics for a fitted selector state."""
    beta = np.asarray(state.beta, dtype=float)
    if beta.shape != (dataset.X_train.shape[1],):
        raise ValueError(
            "State beta has incompatible shape "
            f"{beta.shape}; expected {(dataset.X_train.shape[1],)}."
        )

    active_set = np.asarray(state.active_set, dtype=int)

    train = _regression_metrics(dataset.X_train, dataset.y_train, beta)
    val = _regression_metrics(dataset.X_val, dataset.y_val, beta)
    test = _regression_metrics(dataset.X_test, dataset.y_test, beta)

    base_rss = getattr(state, "rss", None)
    cv_rss = getattr(state, "rss_cv", None)

    metrics = {
        "elapsed_ms": float(elapsed_ms),
        "n_selected": int(active_set.size),
        "state_rss": None if base_rss is None else float(base_rss),
        "state_rss_cv": None if cv_rss is None else float(cv_rss),
        "train_rss": train.rss,
        "train_mse": train.mse,
        "val_rss": val.rss,
        "val_mse": val.mse,
        "test_rss": test.rss,
        "test_mse": test.mse,
    }
    metrics.update(_support_metrics(active_set, dataset.true_support))
    return metrics
