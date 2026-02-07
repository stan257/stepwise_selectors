"""Branch-and-bound best-subset selection using Gram statistics only."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

from .constants import ABS_TOL
from .criteria import AICCriterion, SelectionCriterion
from .definitions import GramData
from .fast_routines import FastForwardState
from .state import SelectionState


@dataclass
class BranchAndBoundStats:
    nodes: int = 0
    pruned: int = 0


class BranchAndBoundSelection:
    """Exact best-subset search with pruning.

    This is intended for small subset sizes (k) or small p. It explores
    subsets in lexicographic order and prunes using optimistic bounds from
    forward-step improvements. Use bound_strategy="none" for fully exhaustive
    (exact) search without pruning.
    """

    def __init__(
        self,
        *,
        tol: float = ABS_TOL,
        bound_strategy: str = "none",
        criterion_cls=None,
        criterion_kwargs=None,
    ):
        self.tol = float(tol)
        self.bound_strategy = bound_strategy
        self.criterion_cls = criterion_cls or AICCriterion
        self.criterion_kwargs = dict(criterion_kwargs or {})

    def _init_criterion(self, data: GramData) -> SelectionCriterion:
        params = dict(self.criterion_kwargs)
        init_params = inspect.signature(self.criterion_cls.__init__).parameters
        if "n_samples" in init_params and "n_samples" not in params:
            params["n_samples"] = data.n_samples
        return self.criterion_cls(**params)

    def fit(
        self,
        *,
        data: GramData,
        max_subset_size: int,
        exact_k: bool = False,
    ) -> SelectionState:
        if max_subset_size < 0:
            raise ValueError("max_subset_size must be non-negative.")
        if self.bound_strategy not in ("none", "forward_sum", "full"):
            raise ValueError(
                "bound_strategy must be one of: 'none', 'forward_sum', 'full'."
            )

        criterion = self._init_criterion(data)
        stats = BranchAndBoundStats()
        best_score = float("inf") if criterion.minimize else float("-inf")
        best_set: list[int] = []

        rss_floor = max(self.tol, 1e-12)

        def optimistic_bound(
            state: FastForwardState,
            candidates: np.ndarray,
            rss_new: np.ndarray,
        ) -> float | None:
            """Compute optimistic best achievable score from this node."""
            if self.bound_strategy == "none":
                return None

            if self.bound_strategy == "full":
                idx = list(state.active_set) + [int(c) for c in candidates]
                if not idx:
                    return None
                idx_arr = np.array(idx, dtype=int)
                try:
                    G_ss = data.gram[np.ix_(idx_arr, idx_arr)]
                    cov_s = data.cov[idx_arr]
                    beta = np.linalg.solve(G_ss, cov_s)
                    rss = float(data.y_norm - cov_s @ beta)
                except np.linalg.LinAlgError:
                    return None
                rss = max(rss, rss_floor)
                score = float(np.asarray(criterion.evaluate(rss, len(idx))))
                return score

            remain = max_subset_size - state.k
            if remain <= 0:
                return None
            if candidates.size == 0:
                return None

            improvements = state.rss - rss_new
            improvements = np.maximum(improvements, 0.0)
            if improvements.size == 0:
                return None

            order = np.argsort(improvements)[::-1]
            top = improvements[order]
            cum = np.cumsum(top)
            max_add = min(remain, cum.size)

            scores = []
            if not exact_k:
                score0 = float(
                    np.asarray(criterion.evaluate(max(state.rss, rss_floor), state.k))
                )
                scores.append(score0)

            if exact_k:
                if remain > cum.size:
                    return None
                rss_bound = max(state.rss - cum[remain - 1], rss_floor)
                score = float(
                    np.asarray(criterion.evaluate(rss_bound, state.k + remain))
                )
                scores.append(score)
            else:
                for t in range(1, max_add + 1):
                    rss_bound = max(state.rss - cum[t - 1], rss_floor)
                    score = float(
                        np.asarray(criterion.evaluate(rss_bound, state.k + t))
                    )
                    scores.append(score)

            if criterion.minimize:
                return min(scores)
            return max(scores)

        def dfs(state: FastForwardState, start_idx: int) -> None:
            nonlocal best_score, best_set
            stats.nodes += 1

            k = state.k
            if not exact_k or k == max_subset_size:
                score = float(
                    np.asarray(criterion.evaluate(max(state.rss, rss_floor), k))
                )
                if not np.isfinite(best_score) or criterion.is_improvement(
                    score, incumbent=best_score
                ):
                    best_score = score
                    best_set = list(state.active_set)

            if k >= max_subset_size:
                return

            scored = state.candidate_scores()
            if scored is None:
                return
            candidates, rss_new = scored
            mask = candidates >= start_idx
            if not np.any(mask):
                return
            candidates = candidates[mask]
            rss_new = rss_new[mask]

            bound = optimistic_bound(state, candidates, rss_new)
            if (
                bound is not None
                and np.isfinite(best_score)
                and not criterion.is_improvement(bound, incumbent=best_score)
            ):
                stats.pruned += 1
                return

            crit_scores = np.asarray(criterion.evaluate(rss_new, k + 1))
            order = np.argsort(crit_scores)
            if not criterion.minimize:
                order = order[::-1]

            for idx in order:
                feat_idx = int(candidates[idx])
                child = state.clone()
                child.apply_forward(feat_idx)
                dfs(child, feat_idx + 1)

        root = FastForwardState.create(data, self.tol)
        dfs(root, 0)

        result = SelectionState(data)
        result.init_from_active_set(best_set)
        result.search_stats = stats  # type: ignore[attr-defined]
        result.search_score = best_score  # type: ignore[attr-defined]
        return result


__all__ = ["BranchAndBoundSelection", "BranchAndBoundStats"]
