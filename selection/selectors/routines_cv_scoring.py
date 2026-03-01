"""Core CV scoring helpers for selection routines."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..core.definitions import CrossValGramData
from ..core.incremental_solver import IncrementalSolver
from ..core.state_cv import CrossValSelectionState

CVAggregation = Literal["sum_rss", "mean_mse", "median_mse"]


def _normalize_cv_aggregation(cv_aggregation: str) -> CVAggregation:
    """Normalize CV aggregation mode used by CV selectors.

    Supported modes:
    - `sum_rss`: sum per-fold validation RSS (default/backward-compatible).
    - `mean_mse`: average per-fold validation MSE (equal fold weight).
    - `median_mse`: median per-fold validation MSE (robust across folds).
    """
    if not isinstance(cv_aggregation, str):
        raise TypeError("cv_aggregation must be a string.")
    normalized = cv_aggregation.strip().lower()
    match normalized:
        case "sum_rss" | "mean_mse" | "median_mse":
            return normalized
        case _:
            raise ValueError(
                "cv_aggregation must be one of: mean_mse, median_mse, sum_rss."
            )


def _aggregate_cv_rss_vector(
    fold_rss: np.ndarray,
    fold_sizes: np.ndarray,
    cv_aggregation: CVAggregation,
) -> float:
    """Aggregate per-fold RSS values into a scalar CV objective."""
    fold_sizes_f = np.asarray(fold_sizes, dtype=float)
    fold_rss_f = np.asarray(fold_rss, dtype=float)
    match cv_aggregation:
        case "sum_rss":
            return float(np.sum(fold_rss_f))
        case "mean_mse":
            return float(np.mean(fold_rss_f / fold_sizes_f))
        case "median_mse":
            return float(np.median(fold_rss_f / fold_sizes_f))
        case _:
            raise ValueError(
                "cv_aggregation must be one of: mean_mse, median_mse, sum_rss."
            )


def _aggregate_cv_rss_matrix(
    rss_matrix: np.ndarray,
    fold_sizes: np.ndarray,
    cv_aggregation: CVAggregation,
) -> np.ndarray:
    """Aggregate per-fold candidate RSS matrix into candidate objective values."""
    fold_sizes_f = np.asarray(fold_sizes, dtype=float)
    match cv_aggregation:
        case "sum_rss":
            return np.sum(rss_matrix, axis=0)
        case "mean_mse":
            return np.mean(rss_matrix / fold_sizes_f[:, None], axis=0)
        case "median_mse":
            return np.median(rss_matrix / fold_sizes_f[:, None], axis=0)
        case _:
            raise ValueError(
                "cv_aggregation must be one of: mean_mse, median_mse, sum_rss."
            )


def _build_fold_states(
    data: CrossValGramData,
    *,
    tol: float,
    active_set: list[int] | None = None,
    solver_policy: str = "strict",
    ridge_alpha: float = 1e-8,
    pinv_rcond: float = 1e-12,
) -> list[IncrementalSolver]:
    """Build per-fold `IncrementalSolver` objects for a shared active set.

    Preconditions:
    - `data` is validated `CrossValGramData`.
    - `active_set` is either `None` or a valid feature-index list for all folds.

    Postconditions:
    - returns one solver per fold with synchronized support when `active_set` is
      provided.
    """
    if active_set is None:
        return [
            IncrementalSolver.create(
                data.train_data_for_fold(k),
                tol,
                solver_policy=solver_policy,
                ridge_alpha=ridge_alpha,
                pinv_rcond=pinv_rcond,
            )
            for k in range(data.n_folds)
        ]
    return [
        IncrementalSolver.from_active_set(
            data.train_data_for_fold(k),
            active_set,
            tol,
            solver_policy=solver_policy,
            ridge_alpha=ridge_alpha,
            pinv_rcond=pinv_rcond,
        )
        for k in range(data.n_folds)
    ]


def _assert_synced_active_sets(fold_states: list[IncrementalSolver]) -> None:
    """Require identical active sets across all fold states.

    Contract:
    - CV search helpers assume synchronized supports across folds.
    - callers should enforce this via construction/rebuild paths.
    """
    if not fold_states:
        return
    reference = list(fold_states[0].active_set)
    for fold_idx, state in enumerate(fold_states[1:], start=1):
        if list(state.active_set) != reference:
            raise ValueError(
                "CV fold states must share identical active_set across folds; "
                f"fold 0 has {reference}, fold {fold_idx} has {state.active_set}."
            )


def _cv_rss(
    fold_states: list[IncrementalSolver],
    data: CrossValGramData,
    *,
    cv_aggregation: CVAggregation = "sum_rss",
) -> float:
    """Compute summed validation RSS for the shared active set across folds.

    Preconditions:
    - fold states are synchronized (`_assert_synced_active_sets`).

    Postconditions:
    - returns CV objective according to `cv_aggregation`.
    """
    _assert_synced_active_sets(fold_states)
    if not fold_states or not fold_states[0].active_set:
        return _aggregate_cv_rss_vector(
            np.asarray(data.y_norm_folds, dtype=float),
            data.fold_sizes,
            cv_aggregation,
        )
    idx = np.array(fold_states[0].active_set, dtype=int)
    fold_rss = np.empty(len(fold_states), dtype=float)
    for fold_idx, work_state in enumerate(fold_states):
        beta_S = work_state.beta_S[: work_state.k]
        G_val = data.gram_folds[fold_idx]
        c_val = data.cov_folds[fold_idx]
        y_norm_val = data.y_norm_folds[fold_idx]
        G_val_SS = G_val[np.ix_(idx, idx)]
        c_val_S = c_val[idx]
        rss_k = (
            y_norm_val
            - 2.0 * float(beta_S @ c_val_S)
            + float(beta_S @ (G_val_SS @ beta_S))
        )
        fold_rss[fold_idx] = rss_k
    return _aggregate_cv_rss_vector(fold_rss, data.fold_sizes, cv_aggregation)


def _rebuild_states(
    data: CrossValGramData,
    active_set: list[int],
    tol: float,
    *,
    solver_policy: str = "strict",
    ridge_alpha: float = 1e-8,
    pinv_rcond: float = 1e-12,
) -> list[IncrementalSolver]:
    """Rebuild fold states from scratch to limit numerical drift.

    Notes:
    - This is intentionally conservative and favors numerical robustness over
      maximal speed for long CV paths.
    """
    return _build_fold_states(
        data,
        tol=tol,
        active_set=active_set,
        solver_policy=solver_policy,
        ridge_alpha=ridge_alpha,
        pinv_rcond=pinv_rcond,
    )


def _build_cv_state_from_active_set(
    data: CrossValGramData,
    active_set: list[int],
    *,
    state: CrossValSelectionState | None = None,
    solver_policy: str = "strict",
    ridge_alpha: float = 1e-8,
    pinv_rcond: float = 1e-12,
) -> CrossValSelectionState:
    """Materialize `CrossValSelectionState` from a finalized active set.

    Preconditions:
    - `active_set` is valid for `data.p`.
    - `state` is either `None` or bound to `data`.

    Postconditions:
    - per-fold train states are rebuilt on `active_set`.
    - `active_set`, `rss_cv`, and full-data refit `beta` are synchronized.
    """
    cv_state = (
        state
        if state is not None
        else CrossValSelectionState(
            data,
            solver_policy=solver_policy,
            ridge_alpha=ridge_alpha,
            pinv_rcond=pinv_rcond,
        )
    )
    for fold_state in cv_state.train_states:
        fold_state.init_from_active_set(active_set)
    cv_state._sync_active_set()
    cv_state.recompute_oos_rss()
    return cv_state


def _cv_forward_scores(
    fold_states: list[IncrementalSolver],
    data: CrossValGramData,
    tol: float,
    *,
    cv_aggregation: CVAggregation = "sum_rss",
) -> tuple[list[int], np.ndarray] | None:
    """Return common candidates and aggregated (summed) validation RSS.

    Uses Gram-only formulas to score candidates per fold without materializing
    design matrices, then sums fold RSS to match rss_cv scale.

    Preconditions:
    - fold states are synchronized and numerically valid.
    - `tol` is positive and consistent with solver tolerance policy.

    Returns:
    - `(candidates, aggregated_score)` where aggregation is controlled by
      `cv_aggregation`.
    - `None` when no common valid candidate remains across folds.
    """
    _assert_synced_active_sets(fold_states)
    if not fold_states:
        return None

    candidate_lists = []
    for state in fold_states:
        scored = state.candidate_scores()
        if scored is None:
            return None
        candidates, _ = scored
        candidate_lists.append(set(int(c) for c in candidates))

    common = set.intersection(*candidate_lists)
    if not common:
        return None
    candidates = sorted(common)
    num_candidates = len(candidates)
    rss_matrix = np.empty((len(fold_states), num_candidates), dtype=float)

    for fold_idx, state in enumerate(fold_states):
        idx_S = np.array(state.active_set, dtype=int)
        k = state.k
        resid_corr = state.r[candidates]
        resid_var = state.v[candidates]
        beta_j = resid_corr / resid_var

        G_val = data.gram_folds[fold_idx]
        c_val = data.cov_folds[fold_idx]
        y_norm_val = data.y_norm_folds[fold_idx]
        c_val_j = c_val[candidates]
        g_val_jj = G_val[candidates, candidates]

        if k == 0:
            rss = y_norm_val - 2.0 * beta_j * c_val_j + (beta_j**2) * g_val_jj
            rss_matrix[fold_idx] = rss
            continue

        G_val_SS = G_val[np.ix_(idx_S, idx_S)]
        c_val_S = c_val[idx_S]
        g_val_Sc = G_val[np.ix_(idx_S, candidates)]

        g_train_Sc = state.data.gram[np.ix_(idx_S, candidates)]
        proj = state.K[:k, :k] @ g_train_Sc
        beta_S_new = state.beta_S[:k, None] - proj * beta_j[None, :]

        term1 = -2.0 * (beta_S_new * c_val_S[:, None]).sum(axis=0)
        term2 = -2.0 * beta_j * c_val_j
        term3 = (beta_S_new * (G_val_SS @ beta_S_new)).sum(axis=0)
        term4 = 2.0 * beta_j * (g_val_Sc * beta_S_new).sum(axis=0)
        term5 = (beta_j**2) * g_val_jj
        rss_matrix[fold_idx] = y_norm_val + term1 + term2 + term3 + term4 + term5

    aggregated = _aggregate_cv_rss_matrix(
        rss_matrix, data.fold_sizes, cv_aggregation
    )
    return candidates, aggregated


def _cv_backward_scores(
    fold_states: list[IncrementalSolver],
    data: CrossValGramData,
    tol: float,
    *,
    cv_aggregation: CVAggregation = "sum_rss",
) -> np.ndarray | None:
    """Compute aggregated CV backward scores using Gram-only downdates.

    Preconditions:
    - fold states are synchronized and non-empty.

    Returns:
    - fold-summed validation RSS per removable local support index.
    - `None` if no active features exist.
    """
    _assert_synced_active_sets(fold_states)
    if not fold_states or not fold_states[0].active_set:
        return None
    idx_full = np.array(fold_states[0].active_set, dtype=int)
    k = len(idx_full)
    rss_matrix = np.empty((len(fold_states), k), dtype=float)

    for fold_idx, state in enumerate(fold_states):
        Kk = state.K[:k, :k]
        beta_k = state.beta_S[:k]
        G_val = data.gram_folds[fold_idx]
        c_val = data.cov_folds[fold_idx]
        y_norm_val = data.y_norm_folds[fold_idx]

        for local_idx in range(k):
            k_22 = Kk[local_idx, local_idx]
            if not np.isfinite(k_22) or k_22 <= tol:
                rss_matrix[fold_idx, local_idx] = np.inf
                continue
            idx_keep = np.delete(np.arange(k), local_idx)
            beta_removed = beta_k[local_idx]
            k_12 = Kk[idx_keep, local_idx]
            beta_keep = beta_k[idx_keep]
            beta_new = (
                beta_keep - (beta_removed / k_22) * k_12
                if beta_keep.size
                else np.zeros(0)
            )
            if not beta_new.size:
                rss_matrix[fold_idx, local_idx] = y_norm_val
                continue
            idx = idx_full[idx_keep]
            G_val_SS = G_val[np.ix_(idx, idx)]
            c_val_S = c_val[idx]
            rss = (
                y_norm_val
                - 2.0 * float(beta_new @ c_val_S)
                + float(beta_new @ (G_val_SS @ beta_new))
            )
            rss_matrix[fold_idx, local_idx] = rss

    return _aggregate_cv_rss_matrix(rss_matrix, data.fold_sizes, cv_aggregation)


__all__ = [
    "CVAggregation",
    "_normalize_cv_aggregation",
    "_aggregate_cv_rss_vector",
    "_aggregate_cv_rss_matrix",
    "_build_fold_states",
    "_cv_rss",
    "_rebuild_states",
    "_build_cv_state_from_active_set",
    "_cv_forward_scores",
    "_cv_backward_scores",
]
