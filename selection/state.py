from dataclasses import dataclass, field

import numpy as np

from .constants import ABS_TOL

# Tuneable block size for forward candidate scoring to reduce peak memory.
FORWARD_BLOCK_SIZE = 4096
from .definitions import CrossValGramData, GramData


@dataclass
class ForwardDeltaCache:
    """Snapshot of all forward-candidate statistics for a given state."""

    candidates: np.ndarray  # indices of inactive features considered
    rss_new: np.ndarray  # resulting RSS after adding each candidate
    resid_var: np.ndarray  # conditional variance (denominator) per candidate
    resid_corr: np.ndarray  # residual correlation with response per candidate
    proj_col: np.ndarray | None  # K @ G_Sj projections; None when S is empty
    active_rk: int  # current active-set size k


def _build_forward_cache(
    state: "SelectionState", tol: float
) -> ForwardDeltaCache | None:
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
        rss_new = np.clip(rss_new[valid_rss], tol, None)
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
        rss_new=np.clip(rss_new[valid_rss], tol, None),
        resid_var=resid_var[valid_rss],
        resid_corr=resid_corr[valid_rss],
        proj_col=proj_col[:, valid_rss],
        active_rk=k,
    )


def _apply_forward_from_cache(
    state: "SelectionState", cache: ForwardDeltaCache, cache_idx: int
) -> int:
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
    k = len(state.active_set)
    if not 0 <= active_pos < k:
        return None

    K_inv = state.K
    idx_to_keep = np.delete(np.arange(k), active_pos)

    K_11 = K_inv[np.ix_(idx_to_keep, idx_to_keep)]
    k_12 = K_inv[idx_to_keep, active_pos]
    k_22 = K_inv[active_pos, active_pos]

    if abs(k_22) <= tol:
        return None

    K_new = K_11 - np.outer(k_12, k_12) / k_22

    removed_var = state.active_set[active_pos]
    new_active_set_indices = [state.active_set[i] for i in idx_to_keep]
    beta_removed = state.beta_S[active_pos]
    beta_keep = np.delete(state.beta_S, active_pos)
    beta_S_new = (
        beta_keep - (beta_removed / k_22) * k_12 if beta_keep.size else np.zeros(0)
    )
    rss_new = state.rss + (beta_removed**2) / k_22
    return removed_var, new_active_set_indices, K_new, beta_S_new, float(rss_new)


@dataclass
class SelectionState:
    # Fixed sufficient statistics
    data: GramData
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
    K_buf: np.ndarray = field(init=False)
    beta_buf: np.ndarray = field(init=False)
    outer_buf: np.ndarray = field(init=False)
    mask_buf: np.ndarray = field(init=False)
    active_mask: np.ndarray = field(init=False)
    active_idx_buf: np.ndarray = field(init=False)
    active_len: int = field(init=False, default=0)

    def __post_init__(self):
        self.p = self.data.gram.shape[0]
        self.gram_diag = np.diag(self.data.gram)
        self.beta = np.zeros(self.p)
        self.K_buf = np.empty((self.p, self.p))
        self.beta_buf = np.empty(self.p)
        self.outer_buf = np.empty((self.p, self.p))
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
        try:
            L = np.linalg.cholesky(G_S)
            rhs = self.data.cov[idx_S]
            # Solve G_S beta = cov via Cholesky: L L^T beta = rhs.
            beta_S_val = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
            # Invert G_S using the same factorization to keep symmetry.
            K_val = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(idx_S))))
        except np.linalg.LinAlgError as err:
            raise np.linalg.LinAlgError(
                "Active Gram matrix is singular or ill-conditioned during init_full."
            ) from err

        self.beta[:] = 0.0
        self.beta[idx_S] = beta_S_val
        c_S = self.data.cov[idx_S]
        self.rss = self.data.y_norm - c_S @ beta_S_val

        # Store solutions into buffers and set views to avoid future allocations.
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
        clone.p = self.p
        clone.gram_diag = self.gram_diag
        clone.beta = self.beta.copy()
        # Deep copy live slices so clones do not share scratch buffers.
        clone.beta_buf = self.beta_buf.copy()
        clone.outer_buf = self.outer_buf.copy()
        clone.K_buf = self.K_buf.copy()
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
        diag_K = np.diag(self.K)
        rss_new = np.full(k, np.inf, dtype=float)
        tol_value = ABS_TOL if tol is None else float(tol)
        safe = np.abs(diag_K) > tol_value
        if np.any(safe):
            rss_new[safe] = self.rss + self.beta_S[safe] ** 2 / diag_K[safe]
        return rss_new

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
        self.beta_S = beta_S_new
        self.K = K_new if self.active_set else None
        self.rss = float(rss_new)
        return removed_var


@dataclass
class CrossValSelectionState:
    data: CrossValGramData
    p: int = field(init=False)
    n_folds: int = field(init=False)
    train_states: list[SelectionState] = field(init=False)
    active_set: list[int] = field(init=False, default_factory=list)
    beta: np.ndarray = field(init=False)
    oos_rss_folds: np.ndarray = field(init=False)
    rss_cv: float = field(init=False)

    def __post_init__(self):
        self.p = self.data.gram_total.shape[0]
        self.n_folds = self.data.n_folds
        self.beta = np.zeros(self.p, dtype=float)
        self._init_train_states_empty()
        self.oos_rss_folds = np.array(self.data.y_norm_folds, dtype=float)
        self.rss_cv = float(self.oos_rss_folds.sum())

    def _init_train_states_empty(self) -> None:
        self.train_states = []
        for k in range(self.n_folds):
            train_data_k = self.data.train_data_for_fold(k)
            state_k = SelectionState(train_data_k)
            self.train_states.append(state_k)
        self._sync_active_set()

    def init_empty(self) -> None:
        self._init_train_states_empty()
        self.beta[:] = 0.0
        self.oos_rss_folds = np.array(self.data.y_norm_folds, dtype=float)
        self.rss_cv = float(self.oos_rss_folds.sum())

    def init_full(self) -> None:
        self.train_states = []
        for k in range(self.n_folds):
            train_data_k = self.data.train_data_for_fold(k)
            state_k = SelectionState(train_data_k)
            state_k.init_full()
            self.train_states.append(state_k)
        self._sync_active_set()
        self.beta[:] = 0.0
        self.recompute_oos_rss()

    def recompute_oos_rss(self) -> float:
        S = self.active_set
        if not S:
            self.oos_rss_folds = np.array(self.data.y_norm_folds, dtype=float)
            self.rss_cv = float(self.oos_rss_folds.sum())
            return self.rss_cv

        idx_S = np.array(S, dtype=int)
        # Stack fold-specific values to compute RSS across folds in one pass.
        beta_mat = np.stack([state.beta_S for state in self.train_states], axis=0)
        c_val = self.data.cov_folds_arr[:, idx_S]
        y_norm_val = self.data.y_norm_folds_arr
        G_val_SS = np.stack(
            [G[np.ix_(idx_S, idx_S)] for G in self.data.gram_folds], axis=0
        )
        term2 = 2.0 * np.einsum("fk,fk->f", beta_mat, c_val)
        term3 = np.einsum("fk,fkl,fl->f", beta_mat, G_val_SS, beta_mat)
        oos = y_norm_val - term2 + term3

        self.oos_rss_folds = oos
        self.rss_cv = float(oos.sum())
        return self.rss_cv

    def _sync_active_set(self) -> None:
        if self.train_states:
            self.active_set = list(self.train_states[0].active_set)
        else:
            self.active_set = []

    def apply_backward_step(self, idx_local: int, tol: float | None = None) -> None:
        tol_value = ABS_TOL if tol is None else float(tol)
        for fold_state in self.train_states:
            fold_state.apply_backward_step(idx_local, tol_value)
        self._sync_active_set()
        self.recompute_oos_rss()

    def validation_rss_for_candidate(
        self,
        fold_idx: int,
        cache: ForwardDeltaCache,
        cache_idx: int,
    ) -> float:
        state_k = self.train_states[fold_idx]
        feat_idx = int(cache.candidates[cache_idx])
        beta_j = cache.resid_corr[cache_idx] / cache.resid_var[cache_idx]
        if cache.active_rk == 0:
            beta_vec = np.array([beta_j])
            active = [feat_idx]
        else:
            proj_vec = cache.proj_col[:, cache_idx]
            beta_S_new = state_k.beta_S - proj_vec * beta_j
            beta_vec = np.concatenate([beta_S_new, np.array([beta_j])])
            active = state_k.active_set + [feat_idx]

        idx = np.array(active, dtype=int)
        G_val = self.data.gram_folds[fold_idx]
        c_val = self.data.cov_folds[fold_idx]
        y_norm_val = self.data.y_norm_folds[fold_idx]
        G_val_SS = G_val[np.ix_(idx, idx)]
        c_val_S = c_val[idx]
        rss = (
            y_norm_val
            - 2.0 * float(beta_vec @ c_val_S)
            + float(beta_vec @ (G_val_SS @ beta_vec))
        )
        return rss

    def validation_rss_for_backward_candidate(
        self, fold_idx: int, local_idx: int, tol: float
    ) -> float:
        state_k = self.train_states[fold_idx]
        if not state_k.active_set or local_idx >= len(state_k.active_set):
            return np.inf
        update_components = _backward_components(state_k, local_idx, tol)
        if update_components is None:
            return np.inf
        _, new_active_set, _, beta_S_new, _ = update_components
        if not new_active_set:
            return self.data.y_norm_folds[fold_idx]

        idx = np.array(new_active_set, dtype=int)
        G_val = self.data.gram_folds[fold_idx]
        c_val = self.data.cov_folds[fold_idx]
        y_norm_val = self.data.y_norm_folds[fold_idx]
        G_val_SS = G_val[np.ix_(idx, idx)]
        c_val_S = c_val[idx]
        rss = (
            y_norm_val
            - 2.0 * float(beta_S_new @ c_val_S)
            + float(beta_S_new @ (G_val_SS @ beta_S_new))
        )
        return rss

    def clone(self) -> "CrossValSelectionState":
        """Clone mutable per-fold state while reusing shared read-only data."""
        clone = object.__new__(CrossValSelectionState)
        clone.data = self.data
        clone.p = self.p
        clone.n_folds = self.n_folds
        clone.train_states = [s.clone() for s in self.train_states]
        clone.active_set = list(self.active_set)
        clone.beta = self.beta.copy()
        clone.oos_rss_folds = self.oos_rss_folds.copy()
        clone.rss_cv = float(self.rss_cv)
        return clone
