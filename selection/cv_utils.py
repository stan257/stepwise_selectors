from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .state import CrossValSelectionState, ForwardDeltaCache


@dataclass
class CVForwardScores:
    """Per-fold caches and aggregated RSS for a CV forward step."""

    fold_caches: List[ForwardDeltaCache]
    candidate_maps: List[dict[int, int]]
    candidates: List[int]
    aggregated_rss: np.ndarray


@dataclass
class CVBackwardScores:
    """Per-fold and aggregated RSS for dropping each active feature."""

    rss_matrix: np.ndarray
    aggregated_rss: np.ndarray


def cv_forward_scores(
    cv_state: CrossValSelectionState, tol: float
) -> Optional[CVForwardScores]:
    fold_caches: List[ForwardDeltaCache] = []
    candidate_maps: List[dict[int, int]] = []
    for train_state in cv_state.train_states:
        cache = train_state.compute_forward_deltas(tol)
        if cache is None or not cache.candidates.size:
            return None
        fold_caches.append(cache)
        candidate_maps.append({int(c): idx for idx, c in enumerate(cache.candidates)})

    common = set(candidate_maps[0].keys())
    for mapping in candidate_maps[1:]:
        common &= set(mapping.keys())
    if not common:
        return None
    candidates = sorted(common)
    num_candidates = len(candidates)
    rss_matrix = np.full((cv_state.n_folds, num_candidates), np.inf, dtype=float)
    for fold_idx, cache in enumerate(fold_caches):
        mapping = candidate_maps[fold_idx]
        for col, candidate in enumerate(candidates):
            idx_local = mapping.get(candidate)
            if idx_local is None:
                raise RuntimeError("Candidate mapping missing during CV forward scoring.")
            rss_matrix[fold_idx, col] = cv_state.validation_rss_for_candidate(
                fold_idx, cache, idx_local
            )

    aggregated = np.mean(rss_matrix, axis=0)
    return CVForwardScores(
        fold_caches=fold_caches,
        candidate_maps=candidate_maps,
        candidates=candidates,
        aggregated_rss=aggregated,
    )


def cv_backward_scores(
    cv_state: CrossValSelectionState, tol: float
) -> Optional[CVBackwardScores]:
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
