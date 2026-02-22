# useful codebase – detailed walkthrough

This document explains every module in the `selection` package and the associated tests. For each file you will find a description of its purpose followed by detailed notes on every function, method, and class: what it is for, how it works internally, and any important behaviors or assumptions. The goal is to give a researcher enough context to understand and extend the code without reading the source line-by-line.

---

## `selection/constants.py`
- `ABS_TOL = 1e-10`
  - Global absolute tolerance used throughout the codebase for numerical comparisons (e.g., rejecting near-singular pivots, determining improvement thresholds). Centralizing it keeps behavior consistent across modules.

---

## `selection/criteria.py` – scoring functions for model selection
- `class SelectionCriterion`
  - Purpose: Abstract base for selection criteria. Tracks whether a model update improves the current score.
  - Key attributes: `minimize` (bool), `abs_tol`, `rel_tol`, and `current_value` (initialized to ±∞ depending on `minimize`).
  - Methods:
    - `__init__(minimize=True, abs_tol=ABS_TOL, rel_tol=1e-8)`: Stores config and seeds `current_value`.
    - `evaluate(self, rss, k)`: Abstract; subclasses implement score computation given residual sum of squares (RSS) and active-set size `k`.
    - `is_improvement(self, candidate, incumbent=None)`: Compares a candidate score to `incumbent` (or `current_value`) using absolute + relative tolerance; returns True if strictly better beyond tolerance.
    - `update_current(self, value)`: Records a finite score as the new incumbent; rejects non-finite values.
    - `clone(self)`: Deep-copies the criterion (used for beam branches so each branch tracks its own incumbent).
    - `best_candidate(self, rss, k)`: Evaluates scores for an array of RSS values, picks argmin/argmax depending on `minimize`, returns `(idx, best_score)`. Flattens inputs to handle multidimensional arrays.

- `class AICCriterion(SelectionCriterion)`
  - Purpose: Implements Akaike Information Criterion (AIC) as the selection score.
  - Methods:
    - `__init__(n_samples, **kwargs)`: Requires number of samples; sets `minimize=True`.
    - `evaluate(self, rss, k)`: Computes `n * log(rss / n) + 2 * k`; enforces positive RSS.
    - `best_candidate(self, rss, k)`: Optimized to pick the argmin of RSS directly, then compute the corresponding AIC value for that candidate.

- `class BestRSSCriterion(SelectionCriterion)`
  - Purpose: Greedy RSS minimization without complexity penalty.
  - Methods:
    - `__init__(**kwargs)`: Sets `minimize=True`.
    - `evaluate(self, rss, k)`: Returns RSS array (validates non-negative in debug mode).
    - `best_candidate(self, rss, k)`: Argmin of RSS; returns `(idx, rss[idx])`.

---

## `selection/definitions.py` – immutable problem descriptions
- `@dataclass GramData`
  - Encapsulates sufficient statistics for a dataset: `gram` (XᵀX), `cov` (Xᵀy), `y_norm` (yᵀy), `n_samples`.
  - `__post_init__`: Validates types (ndarrays for gram/cov, real non-negative `y_norm`, integer `n_samples`), shapes (square Gram, matching cov length), and `n_samples > 0`.

- `@dataclass CrossValGramData`
  - Represents per-fold Gram statistics for cross-validation and their aggregate.
  - Derived attributes: `gram_total`, `cov_total`, `y_norm_total` (sums over folds); `n_folds`, `p`, `fold_sizes`, `n_samples_total`; cached `gram_folds`, `cov_folds`, `y_norm_folds`.
  - `__post_init__`: Calls `check_data_validity` to ensure all folds are `GramData` with consistent dimensions; sums Gram/cov/y_norm across folds and fold sizes; sets `n_samples_total`.
  - Methods:
    - `check_data_validity()`: Asserts folds list is non-empty, all elements are `GramData`, and shapes match a common `p`; returns `p`.
    - `val_data_for_fold(k)`: Returns fold `k` GramData (validation portion).
    - `train_data_for_fold(k)`: Constructs training GramData for fold `k` by subtracting fold `k` stats from totals; checks `n_samples_total` is known.
    - `make_full_data()`: Reconstructs full-data GramData from aggregated sums; requires `n_samples_total`.

---

## `selection/state.py` – linear algebra state for selection
- `@dataclass ForwardDeltaCache`
  - Holds forward-lookahead info: `candidates` (inactive indices), `rss_new` after adding each, `resid_var` (conditional variance denominators), `resid_corr` (residual correlations), `proj_col` (K @ G_S,j for active set S), and `active_rk` (current active size).

- `_build_forward_cache(state, tol) -> ForwardDeltaCache | None`
  - Purpose: Compute per-candidate forward deltas for the current `SelectionState`.
  - Behavior: Masks active variables, computes residual variance/correlation; for empty model uses Gram diagonal and cov; for non-empty uses projected columns via current inverse Gram `K`. Filters out numerically unstable candidates (resid_var <= tol or rss_new <= -tol), clips RSS to be positive, returns None if no viable candidates.

- `_apply_forward_from_cache(state, cache, idx) -> int`
  - Purpose: Apply the cached forward update for candidate at position `idx` in `cache`.
  - Behavior: Computes new coefficients and updated inverse Gram using Sherman–Morrison rank-one update; updates `beta_S`, `K`, `active_set`, full `beta`, and `rss`. Returns the added feature index.

- `_backward_components(state, idx_local, tol) -> tuple | None`
  - Purpose: Compute components needed to remove an active predictor using Sherman–Morrison.
  - Behavior: Partitions `K` into blocks, checks pivot magnitude (`k_22`), computes updated inverse, new coefficients, and RSS increase. Returns None if index invalid or numerically unstable.

- `@dataclass SelectionState`
  - Represents mutable model state for a single Gram dataset.
  - Attributes: fixed stats (`data`, `p`, `gram_diag`); support/model state (`active_set`, full `beta`, active-set `beta_S`, inverse Gram `K`, `rss`); scratch buffers (`K_buf`, `beta_buf`, `outer_buf`) reused for rank-one updates.
  - Methods:
    - `__post_init__`: Sets dimensions, initializes `beta`, and calls `init_empty`.
    - `init_empty()`: Resets to empty model (no active variables, zero coefficients, `K=None`, `rss=y_norm`).
    - `init_full()`: Sets active set to all predictors, solves `G_S beta_S = cov_S`, computes `K = G_S^{-1}`, updates `beta` and `rss`; raises if Gram is singular/ill-conditioned. Seeds scratch buffers with the solved values.
    - `compute_forward_deltas(tol=None)`: Builds a `ForwardDeltaCache` using `_build_forward_cache`; optional tolerance override.
    - `apply_forward_step(cache, idx)`: bounds-checks idx and delegates to `_apply_forward_from_cache`.
    - `compute_backward_scores(tol=None)`: Vector of RSS values if each active variable were dropped; uses diagonal of `K` to compute RSS efficiently; returns None if no active set.
    - `apply_backward_step(idx_local, tol=None)`: Uses `_backward_components` to remove an active variable; updates coefficients, `K`, `rss`; raises on instability or invalid index.
    - `clone()`: Deep-copies mutable arrays and lists while sharing immutable data; scratch buffers (for rank-one updates) are state-local and copied per clone.

- `@dataclass CrossValSelectionState`
  - Wraps one `SelectionState` per fold (training-only Gram) plus global state for CV.
  - Attributes: `data`, `p`, `n_folds`, `train_states`, `active_set`, `beta`, per-fold OOS RSS `oos_rss_folds`, and aggregated `rss_cv`.
  - Methods:
    - `__post_init__`: Initializes per-fold training states (empty), sets `beta=0`, and seeds OOS RSS to validation norms (empty model prediction).
    - `_init_train_states_empty()`: Builds new `SelectionState` for each fold’s training Gram and syncs active set.
    - `init_empty()`: Resets all folds to empty model; resets beta and OOS RSS.
    - `init_full()`: Initializes all training states with full active set; resets beta; recomputes OOS RSS.
    - `_sync_active_set()`: Copies active set from the first fold (assumed consistent across folds).
    - `recompute_oos_rss()`: Computes validation RSS on each fold for current `active_set` using per-fold coefficients and validation Gram/cov; updates `oos_rss_folds` and `rss_cv`. Empty set yields validation norms.
    - `apply_backward_step(idx_local, tol=None)`: Applies backward removal to every fold state, syncs active set, recomputes OOS RSS.
    - `validation_rss_for_candidate(fold_idx, cache, cache_idx)`: Hypothetical validation RSS on fold `fold_idx` if a cached forward candidate were added; reconstructs prospective coefficients and evaluates validation Gram.
    - `validation_rss_for_backward_candidate(fold_idx, local_idx, tol)`: Hypothetical validation RSS if an active variable were removed; uses `_backward_components` without mutating state.
    - `clone()`: Deep-copies mutable per-fold states and global arrays.

---

## `selection/beam_search.py` – beam data structures and basic expansion
- `@dataclass Beam`
  - Encapsulates a search branch: `state` (SelectionState), `criterion` (with its current value), `score`, and a cached `_signature` (sorted active set) for deduplication.
  - `__post_init__`: Computes `_signature`.
  - `signature` property: Exposes the cached active-set tuple.

- `@dataclass BeamManager`
  - Maintains current beam frontier and beam-width limit.
  - Methods:
    - `expand(expand_fn)`: Calls `expand_fn` on each beam, accumulates candidates, prunes via `beam_prune` to `num_beams`; returns False if no candidates.
    - `best_state()`: Returns state of the beam with lowest score.
    - `__len__`: Number of beams.

- `beam_forward_children(beam, beam_width, tol=ABS_TOL)`
  - Purpose: Generate child beams for each forward candidate.
  - Process: Computes forward deltas, evaluates criterion scores, orders by score, filters non-improving moves, clones state/criterion per candidate, applies forward step, and returns Beam objects.

- `beam_backward_children(beam, beam_width, tol=ABS_TOL, allow_worse=False)`
  - Purpose: Generate child beams by removing active variables.
  - Process: Computes backward RSS scores, evaluates criterion at reduced model size, orders by score, optionally filters non-improving moves, clones/apply removal, returns Beam children (skips failures).

- `beam_best_backward_child(beam, tol=ABS_TOL)`
  - Returns the first improving backward child (if any) by delegating to `beam_backward_children` with width 1.

- `beam_prune(candidates, beam_limit)`
  - Deduplicates candidate beams by signature, keeps the best-scoring unique beams up to `beam_limit`.

---

## `selection/beam_utils.py` – beam search runners and CV beam helpers
- `run_beam_search(initial_beam, beam_width, max_steps, expand_fn, track_best=True)`
  - Generic beam loop: initializes BeamManager with `initial_beam`, repeatedly expands with `expand_fn` until no children or step budget exhausted, tracks best beam by score, returns best state.

- `run_beam_mixed(initial_beam, beam_width, max_forward_steps, max_total_steps, forward_expand, backward_improve)`
  - Mixed forward/backward beam controller: alternates forward expansions with in-place backward improvements per beam while respecting operation budgets; tracks best beam; returns best state.

- `cv_beam_forward_children(beam, beam_width, tol)`
  - Builds beam children for CV forward steps: uses `cv_forward_scores` to intersect candidates across folds, aggregates validation RSS, ranks by criterion, clones CV state per child (applying forward step on each fold), recomputes OOS RSS, and returns Beam children.

- `cv_beam_backward_children(beam, beam_width, tol, allow_worse)`
  - Builds beam children for CV backward steps: uses `cv_backward_scores` to aggregate validation RSS after dropping each active variable, ranks candidates, optionally filters non-improving moves, clones CV state per child (applying backward step across folds), returns Beam children.

- `cv_beam_best_backward_child(beam, tol)`
  - Convenience: returns the first improving CV backward child if any.

---

## `selection/cv_utils.py` – shared scoring for cross-validation
- `@dataclass CVForwardScores`
  - Holds per-fold forward caches, candidate maps (candidate id → local index), common candidate list, and aggregated validation RSS for all candidates.

- `@dataclass CVBackwardScores`
  - Holds per-fold validation RSS matrix (fold × active index) and aggregated RSS across folds.

- `cv_forward_scores(cv_state, tol) -> CVForwardScores | None`
  - Builds forward caches on each fold’s training state, intersects candidate sets across folds, computes validation RSS for each common candidate on each fold, aggregates by sum. Returns None if no viable candidates.

- `cv_backward_scores(cv_state, tol) -> CVBackwardScores | None`
  - Computes validation RSS on each fold for dropping each active variable (via `validation_rss_for_backward_candidate`), aggregates by sum. Returns None if no active variables.

---

## `selection/routines.py` – public API aliases
- `selection/routines.py` is a thin API layer that re-exports the fast implementations from `selection/fast_routines.py` under the default names:
  - `ForwardSelection`, `BackwardSelection`, `MixedSelection`
  - `BeamForwardSelection`, `BeamBackwardSelection`, `BeamMixedSelection`
  - `CrossValForwardSelection`, `CrossValBackwardSelection`, `CrossValMixedSelection`
  - `BeamCrossValForwardSelection`, `BeamCrossValBackwardSelection`, `BeamCrossValMixedSelection`
- The algorithmic behavior and complexity are therefore defined by `selection/fast_routines.py`.

---

## `selection/beam_utils.py` – beam runners and CV beam helpers
*(Covered above; listed again for completeness)*
- `run_beam_search`, `run_beam_mixed` (generic beam controllers).
- `cv_beam_forward_children`, `cv_beam_backward_children`, `cv_beam_best_backward_child` (CV-specific beam expansion using shared CV scoring).

---

## `tests` – verification and documentation-by-example
- `tests/conftest.py`
  - Prepends the project root to `sys.path` so tests can import `selection.*` modules as top-level.

- `tests/test_criteria.py`
  - Exercises `AICCriterion` and `BestRSSCriterion`: formula correctness, improvement checks with tolerance, tracking `current_value`, and `best_candidate` behavior.

- `tests/test_state_management.py`
  - Validates `SelectionState` shape checks, full initialization, correctness of forward/backward updates vs. direct inverses, round-trip add/remove, and backward score consistency.

- `tests/test_selection_routines.py`
  - Integration tests across greedy and beam routines (single and CV):
    - Small hand-crafted problems (2-variable toy) for forward/backward/mixed correctness.
    - ESL-like synthetic data for support recovery (forward/backward).
    - CV variants matching full-data runs and recovering true support.
    - Beam search behaviors (forward/backward/mixed), deduplication, permutation invariance, deterministic pruning, and CV beam vs. greedy consistency.
    - Rank-deficient active set recovery: backward (direct and beam) drops redundant columns and yields invertible reduced Gram.
    - Diagonal problems to verify BestRSS picks largest coefficients and beam variants stay aligned.

---

## Notes on data flow and usage
- Construct `GramData`/`CrossValGramData` from precomputed statistics; pass to selection routines.
- Greedy routines mutate a `SelectionState` (or `CrossValSelectionState`) internally; beam routines clone and branch using `Beam`/`BeamManager`.
- Criteria are configured via `criterion_cls`/`criterion_kwargs`; each branch/run gets its own criterion instance to track current scores independently.
- CV scoring uses training folds for coefficients and validation folds for RSS aggregation, ensuring out-of-sample evaluation.

This summary is intentionally explicit; reading it alongside the code should give a researcher a clear map of responsibilities, data structures, and algorithmic flow across the repository.
