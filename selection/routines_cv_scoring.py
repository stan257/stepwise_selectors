"""Core CV scoring helpers for selection routines."""

from __future__ import annotations

import numpy as np

from .definitions import CrossValGramData
from .forward_state import ForwardState
from .state import CrossValSelectionState


def _build_fold_states(
    data: CrossValGramData, *, tol: float, active_set: list[int] | None = None
) -> list[ForwardState]:
    """Build per-fold ForwardState objects for a shared active set."""
    if active_set is None:
        return [
            ForwardState.create(data.train_data_for_fold(k), tol)
            for k in range(data.n_folds)
        ]
    return [
        ForwardState.from_active_set(data.train_data_for_fold(k), active_set, tol)
        for k in range(data.n_folds)
    ]


def _assert_synced_active_sets(fold_states: list[ForwardState]) -> None:
    """Require identical active sets across all fold states."""
    if not fold_states:
        return
    reference = list(fold_states[0].active_set)
    for fold_idx, state in enumerate(fold_states[1:], start=1):
        if list(state.active_set) != reference:
            raise ValueError(
                "CV fold states must share identical active_set across folds; "
                f"fold 0 has {reference}, fold {fold_idx} has {state.active_set}."
            )


def _cv_rss(fold_states: list[ForwardState], data: CrossValGramData) -> float:
    """Compute summed validation RSS for the shared active set across folds."""
    _assert_synced_active_sets(fold_states)
    if not fold_states or not fold_states[0].active_set:
        return float(np.sum(data.y_norm_folds))
    idx = np.array(fold_states[0].active_set, dtype=int)
    rss_sum = 0.0
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
        rss_sum += rss_k
    return float(rss_sum)


def _rebuild_states(
    data: CrossValGramData, active_set: list[int], tol: float
) -> list[ForwardState]:
    """Rebuild fold states from scratch to limit numerical drift."""
    return _build_fold_states(data, tol=tol, active_set=active_set)


def _build_cv_state_from_active_set(
    data: CrossValGramData,
    active_set: list[int],
    *,
    state: CrossValSelectionState | None = None,
) -> CrossValSelectionState:
    """Materialize a CrossValSelectionState from an active set."""
    cv_state = state if state is not None else CrossValSelectionState(data)
    for fold_state in cv_state.train_states:
        fold_state.init_from_active_set(active_set)
    cv_state._sync_active_set()
    cv_state.recompute_oos_rss()
    return cv_state


def _cv_forward_scores(
    fold_states: list[ForwardState], data: CrossValGramData, tol: float
) -> tuple[list[int], np.ndarray] | None:
    """Return common candidates and aggregated (summed) validation RSS.

    Uses Gram-only formulas to score candidates per fold without materializing
    design matrices, then sums fold RSS to match rss_cv scale.
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

    # Sum across folds to match rss_cv scale.
    aggregated = np.sum(rss_matrix, axis=0)
    return candidates, aggregated


def _cv_backward_scores(
    fold_states: list[ForwardState], data: CrossValGramData, tol: float
) -> np.ndarray | None:
    """Compute aggregated CV backward scores using Gram-only downdates."""
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

    # Sum across folds to match rss_cv scale.
    return np.sum(rss_matrix, axis=0)


__all__ = [
    "_build_fold_states",
    "_cv_rss",
    "_rebuild_states",
    "_build_cv_state_from_active_set",
    "_cv_forward_scores",
    "_cv_backward_scores",
]
