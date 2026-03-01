from dataclasses import dataclass, field

import numpy as np

from .constants import ABS_TOL

# Tuneable block size for forward candidate scoring to reduce peak memory.
FORWARD_BLOCK_SIZE = 4096
from .definitions import GramData
from .index_validation import validate_feature_indices
from .state_ops import (
    backward_rss_scores,
    backward_schur_downdate,
    validate_solver_params,
)
from .solvers import solve_active_system


@dataclass
class ForwardDeltaCache:
    """Snapshot of all forward-candidate statistics for a given state."""

    candidates: np.ndarray  # indices of inactive features considered
    rss_new: np.ndarray  # resulting RSS after adding each candidate
    resid_var: np.ndarray  # conditional variance (denominator) per candidate
    resid_corr: np.ndarray  # residual correlation with response per candidate
    proj_col: np.ndarray | None  # K @ G_Sj projections; None when S is empty
    active_rk: int  # current active-set size k


@dataclass
class GroupedSelectionState:
    """Output state for grouped selection routines."""

    data: GramData
    groups: tuple[tuple[int, ...], ...]
    active_groups: list[int]
    active_set: list[int]
    beta: np.ndarray
    rss: float


def _build_forward_cache(
    state: "SelectionState", tol: float
) -> ForwardDeltaCache | None:
    """Compute forward-candidate deltas using block Gram projections.

    The cache stores residual variances/correlations needed for a rank-one
    update, avoiding recomputation when applying a chosen candidate.
    """
    # Reuse a shared mask buffer to mark inactive candidates.
    mask = state.mask_buf
    # active_mask tracks active features; invert it to get candidate positions.
    np.logical_not(state.active_mask, out=mask)
    candidates = np.where(mask)[0]
    if not candidates.size:
        return None

    k = len(state.active_set)
    if k == 0:
        resid_var = state.gram_diag[candidates]
        valid = resid_var > tol  # multicollinearity check: avoid near-singular updates
        if not np.any(valid):
            return None
        candidates = candidates[valid]
        resid_var = resid_var[valid]
        resid_corr = state.data.cov[candidates]
        rss_new = state.rss - (resid_corr**2) / resid_var
        valid_rss = rss_new > -tol  # valid rss check: forbid negative/invalid RSS
        if not np.any(valid_rss):
            return None
        candidates = candidates[valid_rss]
        resid_var = resid_var[valid_rss]
        resid_corr = resid_corr[valid_rss]
        rss_new = rss_new[valid_rss]
        return ForwardDeltaCache(
            candidates=candidates,
            rss_new=rss_new,
            resid_var=resid_var,
            resid_corr=resid_corr,
            proj_col=None,
            active_rk=0,
        )

    idx_S = state.active_idx
    k = idx_S.size
    num_candidates = candidates.size
    proj_col = np.empty((k, num_candidates), dtype=float)
    resid_var = np.empty(num_candidates, dtype=float)
    resid_corr = np.empty(num_candidates, dtype=float)

    # Block evaluation avoids materializing the full G_Sc matrix at once.
    for start in range(0, num_candidates, FORWARD_BLOCK_SIZE):
        end = min(start + FORWARD_BLOCK_SIZE, num_candidates)
        cand_block = candidates[start:end]
        g_block = state.data.gram[np.ix_(idx_S, cand_block)]
        proj_block = state.K @ g_block
        proj_col[:, start:end] = proj_block
        resid_var[start:end] = state.gram_diag[cand_block] - np.sum(
            g_block * proj_block, axis=0
        )
        resid_corr[start:end] = state.data.cov[cand_block] - state.beta_S @ g_block
        # Delay RSS computation until after filtering to avoid invalid divides.
    valid = resid_var > tol  # multicollinearity check: avoid near-singular updates
    if not np.any(valid):
        return None
    candidates = candidates[valid]
    resid_var = resid_var[valid]
    proj_col = proj_col[:, valid]
    resid_corr = resid_corr[valid]
    rss_new = state.rss - (resid_corr**2) / resid_var
    valid_rss = rss_new > -tol  # valid rss check: forbid negative/invalid RSS
    if not np.any(valid_rss):
        return None
    return ForwardDeltaCache(
        candidates=candidates[valid_rss],
        rss_new=rss_new[valid_rss],
        resid_var=resid_var[valid_rss],
        resid_corr=resid_corr[valid_rss],
        proj_col=proj_col[:, valid_rss],
        active_rk=k,
    )


def _apply_forward_from_cache(
    state: "SelectionState", cache: ForwardDeltaCache, cache_idx: int
) -> int:
    """Apply a cached forward candidate via a rank-one inverse Gram update."""
    state._ensure_matrix_buffers()
    if state.K_buf is None or state.outer_buf is None:
        raise RuntimeError("SelectionState matrix buffers are unavailable.")

    resid_var = cache.resid_var[cache_idx]
    resid_corr = cache.resid_corr[cache_idx]
    rss_new = cache.rss_new[cache_idx]
    feat_idx = int(cache.candidates[cache_idx])
    if cache.active_rk == 0:
        beta_j = resid_corr / resid_var
        K_new = state.K_buf[:1, :1]
        K_new[0, 0] = 1.0 / resid_var
        beta_new = state.beta_buf[:1]
        beta_new[0] = beta_j
    else:
        proj_vec = cache.proj_col[:, cache_idx]
        beta_j = resid_corr / resid_var
        k_new = cache.active_rk + 1
        K_new = state.K_buf[:k_new, :k_new]
        np.copyto(K_new[:-1, :-1], state.K)
        outer = state.outer_buf[: cache.active_rk, : cache.active_rk]
        np.multiply.outer(proj_vec, proj_vec, out=outer)
        # Scale in-place to avoid a temporary during the rank-one update.
        outer *= 1.0 / resid_var
        K_new[:-1, :-1] += outer
        K_new[:-1, -1] = -proj_vec / resid_var
        K_new[-1, :-1] = -proj_vec / resid_var
        K_new[-1, -1] = 1.0 / resid_var

        beta_new = state.beta_buf[:k_new]
        np.copyto(beta_new[:-1], state.beta_S)
        beta_new[:-1] -= proj_vec * beta_j
        beta_new[-1] = beta_j

    # Keep views into buffers to avoid per-step allocations; clone() will deep copy.
    state.beta_S = beta_new.view()
    state.K = K_new.view()
    state.active_set.append(feat_idx)
    state.active_idx_buf[state.active_len] = feat_idx
    state.active_len += 1
    state.active_mask[feat_idx] = True
    idx_array = state.active_idx
    state.beta[idx_array] = state.beta_S
    state.rss = float(rss_new)
    return feat_idx


def _backward_components(
    state: "SelectionState", active_pos: int, tol: float
) -> tuple[int, list[int], np.ndarray, np.ndarray, float] | None:
    """Compute the Schur-complement downdate for removing one active feature."""
    if state.K is None:
        return None
    downdate = backward_schur_downdate(state.K, state.beta_S, active_pos, tol)
    if downdate is None:
        return None
    idx_to_keep, K_new, beta_S_new, rss_delta = downdate
    removed_var = state.active_set[active_pos]
    new_active_set_indices = [state.active_set[i] for i in idx_to_keep]
    rss_new = state.rss + rss_delta
    return removed_var, new_active_set_indices, K_new, beta_S_new, float(rss_new)


@dataclass
class SelectionState:
    # Fixed sufficient statistics
    data: GramData
    solver_policy: str = "strict"
    ridge_alpha: float = 1e-8
    pinv_rcond: float = 1e-12
    p: int = field(init=False)
    gram_diag: np.ndarray = field(init=False)

    # Mutable model state on the current support
    active_set: list[int] = field(init=False, default_factory=list)
    beta: np.ndarray = field(
        init=False
    )  # full-length coefficients with zeros off-support
    beta_S: np.ndarray = field(init=False)  # coefficients on the active set
    K: np.ndarray | None = field(init=False, default=None)  # G_SS^{-1}
    rss: float = field(init=False)  # residual sum of squares for the current model

    # Scratch buffers reused across updates to avoid per-step allocations
    K_buf: np.ndarray | None = field(init=False, default=None)
    beta_buf: np.ndarray = field(init=False)
    outer_buf: np.ndarray | None = field(init=False, default=None)
    mask_buf: np.ndarray = field(init=False)
    active_mask: np.ndarray = field(init=False)
    active_idx_buf: np.ndarray = field(init=False)
    active_len: int = field(init=False, default=0)

    def __post_init__(self):
        policy, ridge, rcond = validate_solver_params(
            self.solver_policy, self.ridge_alpha, self.pinv_rcond
        )
        self.solver_policy = policy
        self.ridge_alpha = ridge
        self.pinv_rcond = rcond
        self.p = self.data.gram.shape[0]
        self.gram_diag = np.diag(self.data.gram)
        self.beta = np.zeros(self.p)
        self.beta_buf = np.empty(self.p)
        self.mask_buf = np.empty(self.p, dtype=bool)
        self.active_mask = np.zeros(self.p, dtype=bool)
        self.active_idx_buf = np.empty(self.p, dtype=int)
        self.active_len = 0
        self.init_empty()

    @property
    def active_idx(self) -> np.ndarray:
        return self.active_idx_buf[: self.active_len]

    def _refresh_active_cache(self) -> None:
        """Synchronize active index/mask caches from active_set."""
        self.active_len = len(self.active_set)
        if self.active_len:
            self.active_idx_buf[: self.active_len] = np.array(self.active_set, dtype=int)
            # Reset and fill only once per sync to keep mask consistent.
            self.active_mask.fill(False)
            self.active_mask[self.active_idx] = True
        else:
            self.active_mask.fill(False)

    def _ensure_matrix_buffers(self) -> None:
        """Allocate O(p^2) buffers lazily to reduce baseline memory usage."""
        if self.K_buf is None:
            self.K_buf = np.empty((self.p, self.p))
        if self.outer_buf is None:
            self.outer_buf = np.empty((self.p, self.p))

    def init_empty(self):
        self.active_set = []
        self._refresh_active_cache()
        self.beta[:] = 0.0
        self.beta_S = np.zeros(0)
        self.K = None
        self.rss = self.data.y_norm
        # Buffers are reused and will be overwritten on next use.

    def init_full(self):
        self.active_set = list(range(self.p))
        self._refresh_active_cache()
        idx_S = self.active_idx
        G_S = self.data.gram[np.ix_(idx_S, idx_S)]
        rhs = self.data.cov[idx_S]
        beta_S_val, K_val = solve_active_system(
            G_S,
            rhs,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
            context="Active Gram matrix during init_full",
        )

        self.beta[:] = 0.0
        self.beta[idx_S] = beta_S_val
        c_S = self.data.cov[idx_S]
        self.rss = self.data.y_norm - c_S @ beta_S_val

        # Store solutions into buffers and set views to avoid future allocations.
        self._ensure_matrix_buffers()
        if self.K_buf is None:
            raise RuntimeError("K buffer allocation failed.")
        k = len(idx_S)
        self.beta_buf[:k] = beta_S_val
        self.beta_S = self.beta_buf[:k]
        self.K_buf[:k, :k] = K_val
        self.K = self.K_buf[:k, :k]

    def init_from_active_set(self, active_set: list[int]) -> None:
        """Initialize state from an explicit active set using a stable solve."""
        self.active_set = validate_feature_indices(
            list(active_set), self.p, context="active_set"
        )
        self._refresh_active_cache()
        if not self.active_set:
            self.init_empty()
            return
        idx_S = self.active_idx
        G_S = self.data.gram[np.ix_(idx_S, idx_S)]
        rhs = self.data.cov[idx_S]
        beta_S_val, K_val = solve_active_system(
            G_S,
            rhs,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
            context="Active Gram matrix",
        )

        self.beta[:] = 0.0
        self.beta[idx_S] = beta_S_val
        self.rss = self.data.y_norm - self.data.cov[idx_S] @ beta_S_val

        self._ensure_matrix_buffers()
        if self.K_buf is None:
            raise RuntimeError("K buffer allocation failed.")
        k = len(idx_S)
        self.beta_buf[:k] = beta_S_val
        self.beta_S = self.beta_buf[:k]
        self.K_buf[:k, :k] = K_val
        self.K = self.K_buf[:k, :k]

    def compute_forward_deltas(
        self, tol: float | None = None
    ) -> ForwardDeltaCache | None:
        tol_value = ABS_TOL if tol is None else float(tol)
        return _build_forward_cache(self, tol_value)

    def clone(self) -> "SelectionState":
        """Clone mutable state while reusing shared read-only data."""
        clone = object.__new__(SelectionState)
        clone.data = self.data
        clone.solver_policy = self.solver_policy
        clone.ridge_alpha = self.ridge_alpha
        clone.pinv_rcond = self.pinv_rcond
        clone.p = self.p
        clone.gram_diag = self.gram_diag
        clone.beta = self.beta.copy()
        # Deep copy live slices so clones do not share scratch buffers.
        clone.beta_buf = self.beta_buf.copy()
        clone.outer_buf = self.outer_buf.copy() if self.outer_buf is not None else None
        clone.K_buf = self.K_buf.copy() if self.K_buf is not None else None
        clone.mask_buf = self.mask_buf.copy()
        clone.active_mask = self.active_mask.copy()
        clone.active_idx_buf = self.active_idx_buf.copy()
        clone.active_len = self.active_len
        if self.beta_S.size:
            k_beta = self.beta_S.shape[0]
            clone.beta_S = clone.beta_buf[:k_beta]
            np.copyto(clone.beta_S, self.beta_S)
        else:
            clone.beta_S = np.zeros(0)
        if self.K is None:
            clone.K = None
        else:
            k = self.K.shape[0]
            clone.K = clone.K_buf[:k, :k]
            np.copyto(clone.K, self.K)
        clone.active_set = list(self.active_set)
        clone.rss = float(self.rss)
        return clone

    def apply_forward_step(self, cache: ForwardDeltaCache, cache_idx: int) -> int:
        if cache_idx < 0 or cache_idx >= cache.candidates.size:
            raise IndexError("Forward delta index out of bounds.")
        return _apply_forward_from_cache(self, cache, cache_idx)

    def compute_backward_scores(self, tol: float | None = None) -> np.ndarray | None:
        k = len(self.active_set)
        if k == 0 or self.K is None:
            return None
        tol_value = ABS_TOL if tol is None else float(tol)
        return backward_rss_scores(self.rss, self.beta_S, self.K, tol_value)

    def apply_backward_step(self, active_pos: int, tol: float | None = None) -> int:
        tol_value = ABS_TOL if tol is None else float(tol)
        update = _backward_components(self, active_pos, tol_value)
        if update is None:
            raise np.linalg.LinAlgError(
                "Backward downdate failed; index invalid or numerically unstable."
            )
        removed_var, new_active_set, K_new, beta_S_new, rss_new = update
        self.beta[removed_var] = 0.0
        if new_active_set:
            self.beta[new_active_set] = beta_S_new
        self.active_set = new_active_set
        self._refresh_active_cache()
        k = len(new_active_set)
        if k > 0:
            self._ensure_matrix_buffers()
            if self.K_buf is None:
                raise RuntimeError("K buffer allocation failed.")
            self.beta_buf[:k] = beta_S_new
            self.beta_S = self.beta_buf[:k]
            self.K_buf[:k, :k] = K_new
            self.K = self.K_buf[:k, :k]
        else:
            self.beta_S = np.zeros(0)
            self.K = None
        self.rss = float(rss_new)
        return removed_var


__all__ = [
    "FORWARD_BLOCK_SIZE",
    "ForwardDeltaCache",
    "GroupedSelectionState",
    "SelectionState",
]
