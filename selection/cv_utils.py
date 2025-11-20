from dataclasses import dataclass
from math import ceil

import numpy as np

from .state import CrossValSelectionState, ForwardDeltaCache


@dataclass
class CVForwardScores:
    """Per-fold caches and aggregated RSS for a CV forward step."""

    fold_caches: list[ForwardDeltaCache]
    candidate_maps: list[dict[int, int]]
    candidates: list[int]
    aggregated_rss: np.ndarray


@dataclass
class CVBackwardScores:
    """Per-fold and aggregated RSS for dropping each active feature."""

    rss_matrix: np.ndarray
    aggregated_rss: np.ndarray


def cv_forward_scores(
    cv_state: CrossValSelectionState, tol: float
) -> CVForwardScores | None:
    fold_caches: list[ForwardDeltaCache] = []
    candidate_maps: list[dict[int, int]] = []
    # Collect per-fold caches and mappings; we will later aggregate over the union.
    for train_state in cv_state.train_states:
        cache = train_state.compute_forward_deltas(tol)
        if cache is None or not cache.candidates.size:
            candidate_maps.append({})
            fold_caches.append(cache if cache is not None else ForwardDeltaCache(
                candidates=np.array([], dtype=int),
                rss_new=np.array([], dtype=float),
                resid_var=np.array([], dtype=float),
                resid_corr=np.array([], dtype=float),
                proj_col=None,
                active_rk=len(train_state.active_set),
            ))
            continue
        fold_caches.append(cache)
        candidate_maps.append({int(c): idx for idx, c in enumerate(cache.candidates)})

    # Build the union of all candidates across folds.
    candidate_set = set()
    for mapping in candidate_maps:
        candidate_set |= set(mapping.keys())
    if not candidate_set:
        return None

    candidates = sorted(candidate_set)
    num_candidates = len(candidates)
    rss_matrix = np.full((cv_state.n_folds, num_candidates), np.nan, dtype=float)
    # Minimum number of folds that must support a candidate to consider it.
    min_support = ceil(cv_state.n_folds / 2)
    col_index = {c: idx for idx, c in enumerate(candidates)}

    for fold_idx, cache in enumerate(fold_caches):
        mapping = candidate_maps[fold_idx]
        for candidate, idx_local in mapping.items():
            col = col_index[candidate]
            rss_matrix[fold_idx, col] = cv_state.validation_rss_for_candidate(
                fold_idx, cache, idx_local
            )

    support_counts = np.sum(~np.isnan(rss_matrix), axis=0)
    support_mask = support_counts >= min_support
    if not np.any(support_mask):
        return None

    aggregated = np.nanmean(rss_matrix[:, support_mask], axis=0)
    supported_candidates = [c for c, keep in zip(candidates, support_mask) if keep]

    return CVForwardScores(
        fold_caches=fold_caches,
        candidate_maps=candidate_maps,
        candidates=supported_candidates,
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

    aggregated = np.mean(rss_matrix, axis=0)
    return CVBackwardScores(rss_matrix=rss_matrix, aggregated_rss=aggregated)
