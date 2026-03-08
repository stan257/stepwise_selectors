# Porting Contract

This document defines the minimal algorithm contract for reimplementing the selection engine in another codebase or language.

## Scope
- The engine operates on sufficient statistics only:
  - `G = X^T X` (`p x p`)
  - `c = X^T y` (`p`)
  - `y_norm = y^T y` (scalar)
- No raw design matrix is required after those statistics are computed.

## Public Behavior To Preserve
- Inputs:
  - single-dataset: `GramData`
  - cross-validation: `CrossValGramData` as a list of per-fold `GramData`
- Outputs:
  - `SelectionState`: `active_set`, `beta`, `rss`
  - `CrossValSelectionState`: `active_set`, `beta`, `rss_cv`
  - `GroupedSelectionState`: `active_groups`, `active_set`, `beta`, `rss`
- Criteria:
  - selectors accept `criterion` as class, instance, factory, or built-in string key
  - built-in keys: `rss`, `aic`, `aicc`, `bic`, `hqic`, `ebic`, `gcv`

## Incremental Update Equations

Notation:
- active support size `k`
- active support `S`
- residual correlation vector `r` and residual variance vector `v`
- basis rows `Z[0:k, :]`, target projections `qy[0:k]`

Forward candidate score for feature `j`:
- valid if `j` is inactive and `v_j > tol`
- `rss_new(j) = rss - r_j^2 / v_j`

Forward step for chosen `j`:
- `denom = sqrt(v_j)`
- `z_col = Z[0:k, j]`
- `proj = z_col @ Z[0:k, :]`
- `qy_proj = z_col @ qy[0:k]`
- `z_new = (G[:, j] - proj) / denom`
- `qy_new = (c_j - qy_proj) / denom`
- residual updates:
  - `r <- r - z_new * qy_new`
  - `v <- v - z_new^2`
  - `rss <- rss - qy_new^2`
- force selected index to zeroed residual terms:
  - `r_j = 0`, `v_j = 0`

Backward scores on active support index `i` (support order):
- with `K = (G_SS)^{-1}` and active coefficients `beta_S`
- valid if `K_ii > tol`
- `rss_drop(i) = rss + beta_S[i]^2 / K_ii`

Backward downdate:
- remove support index `i`
- apply Schur complement update to `K` and corresponding `beta_S`
- update `rss` by adding `beta_removed^2 / K_ii`
- refresh `r`/`v` from current basis (`Z`, `qy`) for consistency

## Tie-Breaking And Stopping
- Candidate feature order is ascending index order.
- Criterion object determines best candidate from score arrays.
- Default criterion implementations use NumPy `argmin`/`argmax`, so exact ties choose the first candidate in order.
- Forward-like selectors are budget-driven by default:
  - forward selectors require `max_steps`
  - mixed selectors require `max_forward_steps`
  - once a valid best candidate exists, it is accepted even if the criterion worsens
- Legacy forward self-stopping remains available with `stop_on_no_improvement=True`; in that mode a step is accepted only if `criterion.is_improvement(...)` passes tolerance checks.
- Backward selectors and mixed backward cleanup remain improvement-driven.

## Cross-Validation Contract
- Fold supports are kept synchronized across folds.
- CV selection objective uses a configurable aggregation over fold validation
  losses:
  - `sum_rss` (default): summed fold RSS
  - `mean_mse`: mean fold MSE
  - `median_mse`: median fold MSE
- Returned `CrossValSelectionState.beta` is a refit on full aggregated data at the selected support.

## Failure Semantics
- Input-shape/type/index errors: `TypeError` / `ValueError`
- Ill-posed linear algebra during factorization or refit: `np.linalg.LinAlgError`
- Cross-validation active-set synchronization violations: `RuntimeError`
- Numerically invalid backward removals are screened (e.g., candidate RSS becomes `inf`) or rejected via `ValueError` in step application paths.

## Compatibility Notes
- Preserve deterministic behavior for equal-score ties.
- Preserve strict validation of feature indices and group indices.
- Preserve the distinction between:
  - model-selection objective (`rss` or `rss_cv`)
  - final refit coefficients (`beta`)
