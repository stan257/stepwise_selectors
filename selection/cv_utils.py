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
        for col, cache_idx in enumerate(idx_local):
            rss_matrix[fold_idx, col] = cv_state.validation_rss_for_candidate(
                fold_idx, cache, int(cache_idx)
            )

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
