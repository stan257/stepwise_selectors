from dataclasses import dataclass

import numpy as np

from .state import CrossValSelectionState, ForwardDeltaCache


@dataclass
class CVForwardScores:
    """Per-fold caches and aggregated RSS for a CV forward step."""

    fold_caches: list[ForwardDeltaCache]
    candidate_indices: list[np.ndarray]
    candidates: np.ndarray
    aggregated_rss: np.ndarray


@dataclass
class CVBackwardScores:
    """Per-fold and aggregated RSS for dropping each active feature."""

    rss_matrix: np.ndarray
    aggregated_rss: np.ndarray


def cv_forward_scores(
    cv_state: CrossValSelectionState, tol: float
) -> CVForwardScores | None:
    """Score forward candidates using validation RSS aggregated across folds.

    Intersects candidate sets across folds via boolean masks and records the
    per-fold cache indices needed to apply a chosen candidate consistently.
    """
    fold_caches: list[ForwardDeltaCache] = []
    p = cv_state.p
    common_mask = np.ones(p, dtype=bool)
    temp_mask = np.zeros(p, dtype=bool)
    for train_state in cv_state.train_states:
        cache = train_state.compute_forward_deltas(tol)
        if cache is None or not cache.candidates.size:
            return None
        fold_caches.append(cache)
        temp_mask.fill(False)
        temp_mask[cache.candidates] = True
        common_mask &= temp_mask
        if not np.any(common_mask):
            return None

    candidates = np.flatnonzero(common_mask)
    if not candidates.size:
        return None

    num_candidates = candidates.size
    rss_matrix = np.full((cv_state.n_folds, num_candidates), np.inf, dtype=float)
    candidate_indices: list[np.ndarray] = []
    for fold_idx, cache in enumerate(fold_caches):
        # Map common candidates to local cache indices (cache.candidates is sorted).
        idx_local = np.searchsorted(cache.candidates, candidates)
        if __debug__:
            if not np.array_equal(cache.candidates[idx_local], candidates):
                raise RuntimeError("Candidate mapping mismatch during CV forward scoring.")
        candidate_indices.append(idx_local)
        state_k = cv_state.train_states[fold_idx]
        resid_corr = cache.resid_corr[idx_local]
        resid_var = cache.resid_var[idx_local]
        beta_j = resid_corr / resid_var

        G_val = cv_state.data.gram_folds[fold_idx]
        c_val = cv_state.data.cov_folds[fold_idx]
        y_norm_val = cv_state.data.y_norm_folds[fold_idx]
        c_val_j = c_val[candidates]
        g_val_jj = np.diag(G_val)[candidates]

        if cache.active_rk == 0:
            rss_matrix[fold_idx] = (
                y_norm_val - 2.0 * beta_j * c_val_j + (beta_j**2) * g_val_jj
            )
            continue

        if cache.proj_col is None:
            raise RuntimeError("Missing projection cache for non-empty active set.")
        idx_S = np.array(state_k.active_set, dtype=int)
        c_val_S = c_val[idx_S]
        G_val_SS = G_val[np.ix_(idx_S, idx_S)]
        g_val_Sc = G_val[np.ix_(idx_S, candidates)]

        proj_col = cache.proj_col[:, idx_local]
        beta_S_new = state_k.beta_S[:, None] - proj_col * beta_j[None, :]

        # Gram-only RSS formula for validation fold, vectorized across candidates.
        term1 = -2.0 * (beta_S_new * c_val_S[:, None]).sum(axis=0)
        term2 = -2.0 * beta_j * c_val_j
        term3 = (beta_S_new * (G_val_SS @ beta_S_new)).sum(axis=0)
        term4 = 2.0 * beta_j * (g_val_Sc * beta_S_new).sum(axis=0)
        term5 = (beta_j**2) * g_val_jj
        rss_matrix[fold_idx] = y_norm_val + term1 + term2 + term3 + term4 + term5

    # Use summed CV RSS to keep the scale consistent with rss_cv (sum over folds).
    aggregated = np.sum(rss_matrix, axis=0)
    return CVForwardScores(
        fold_caches=fold_caches,
        candidate_indices=candidate_indices,
        candidates=candidates,
        aggregated_rss=aggregated,
    )


def cv_backward_scores(
    cv_state: CrossValSelectionState, tol: float
) -> CVBackwardScores | None:
    if not cv_state.active_set:
        return None
    num_active = len(cv_state.active_set)
    rss_matrix = np.full((cv_state.n_folds, num_active), np.inf, dtype=float)
    for fold_idx in range(cv_state.n_folds):
        for local_idx in range(num_active):
            rss_matrix[fold_idx, local_idx] = (
                cv_state.validation_rss_for_backward_candidate(
                    fold_idx, local_idx, tol
                )
            )

    # Use summed CV RSS to keep the scale consistent with rss_cv (sum over folds).
    aggregated = np.sum(rss_matrix, axis=0)
    return CVBackwardScores(rss_matrix=rss_matrix, aggregated_rss=aggregated)
