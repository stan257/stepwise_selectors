"""Top-k selection helpers with deterministic tie-breaking."""

from __future__ import annotations

import numpy as np


def topk_indices(scores: np.ndarray, k: int, *, minimize: bool = True) -> np.ndarray:
    """Return indices of top-k scores ordered by score then index.

    This uses argpartition for O(n) selection and lexsort for deterministic
    tie-breaking (by index) within the top-k.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = scores.size
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(k, n)

    if k == n:
        idx = np.arange(n)
    else:
        if minimize:
            idx = np.argpartition(scores, k - 1)[:k]
        else:
            idx = np.argpartition(-scores, k - 1)[:k]

    if minimize:
        order = np.lexsort((idx, scores[idx]))
    else:
        order = np.lexsort((idx, -scores[idx]))
    return idx[order]


__all__ = ["topk_indices"]
