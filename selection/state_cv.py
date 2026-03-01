from dataclasses import dataclass, field

import numpy as np

from .constants import ABS_TOL
from .definitions import CrossValGramData
from .state_single import ForwardDeltaCache, SelectionState, _backward_components
from .solvers import solve_active_system

@dataclass
class CrossValSelectionState:
    data: CrossValGramData
    solver_policy: str = "strict"
    ridge_alpha: float = 1e-8
    pinv_rcond: float = 1e-12
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

    def _assert_synced_active_set(self) -> None:
        """Require identical active_set across all per-fold training states."""
        if not self.train_states:
            return
        reference = list(self.train_states[0].active_set)
        for fold_idx, fold_state in enumerate(self.train_states[1:], start=1):
            if list(fold_state.active_set) != reference:
                raise ValueError(
                    "CV fold states must share identical active_set across folds; "
                    f"fold 0 has {reference}, fold {fold_idx} has {fold_state.active_set}."
                )

    def _refit_full_data_beta(self) -> None:
        """Refit coefficients on full data for the current active set.

        This exposes a single post-selection coefficient vector that is
        directly interpretable for downstream analysis. CV model selection
        still uses held-out RSS (`rss_cv`) from fold-wise training models.
        """
        self.beta[:] = 0.0
        if not self.active_set:
            return
        idx_S = np.array(self.active_set, dtype=int)
        gram_ss = self.data.gram_total[np.ix_(idx_S, idx_S)]
        cov_s = self.data.cov_total[idx_S]
        beta_s, _ = solve_active_system(
            gram_ss,
            cov_s,
            solver_policy=self.solver_policy,
            ridge_alpha=self.ridge_alpha,
            pinv_rcond=self.pinv_rcond,
            context="Selected support on full data",
        )
        self.beta[idx_S] = beta_s

    def _init_train_states_empty(self) -> None:
        self.train_states = []
        for k in range(self.n_folds):
            train_data_k = self.data.train_data_for_fold(k)
            state_k = SelectionState(
                train_data_k,
                solver_policy=self.solver_policy,
                ridge_alpha=self.ridge_alpha,
                pinv_rcond=self.pinv_rcond,
            )
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
            state_k = SelectionState(
                train_data_k,
                solver_policy=self.solver_policy,
                ridge_alpha=self.ridge_alpha,
                pinv_rcond=self.pinv_rcond,
            )
            state_k.init_full()
            self.train_states.append(state_k)
        self._sync_active_set()
        self.beta[:] = 0.0
        self.recompute_oos_rss()

    def recompute_oos_rss(self) -> float:
        """Recompute out-of-sample RSS across folds using Gram-only formulas."""
        self._assert_synced_active_set()
        S = self.active_set
        if not S:
            self.oos_rss_folds = np.array(self.data.y_norm_folds, dtype=float)
            self.rss_cv = float(self.oos_rss_folds.sum())
            self.beta[:] = 0.0
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
        self._refit_full_data_beta()
        return self.rss_cv

    def _sync_active_set(self) -> None:
        self._assert_synced_active_set()
        self.active_set = (
            list(self.train_states[0].active_set) if self.train_states else []
        )

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
        """Compute validation RSS for adding a candidate in one fold."""
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
        clone.solver_policy = self.solver_policy
        clone.ridge_alpha = self.ridge_alpha
        clone.pinv_rcond = self.pinv_rcond
        clone.p = self.p
        clone.n_folds = self.n_folds
        clone.train_states = [s.clone() for s in self.train_states]
        clone.active_set = list(self.active_set)
        clone.beta = self.beta.copy()
        clone.oos_rss_folds = self.oos_rss_folds.copy()
        clone.rss_cv = float(self.rss_cv)
        return clone

__all__ = ["CrossValSelectionState"]
