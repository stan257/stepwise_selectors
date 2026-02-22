# useful codebase – detailed walkthrough

This document summarizes the current `selection` package after migration to a fast-only implementation. It is intended as a technical map for researchers who need to validate assumptions, extend algorithms, or trace behavior in tests.

---

## `selection/constants.py`
- `ABS_TOL = 1e-10`
  - Global numerical tolerance for stability checks and improvement thresholds.

---

## `selection/criteria.py` – model-selection objectives
- `SelectionCriterion`
  - Base class for objective scoring and improvement logic.
  - Tracks `current_value`, supports minimization/maximization, and applies absolute + relative tolerances in `is_improvement`.
  - `best_candidate` evaluates candidate scores and returns `(best_index, best_score)`.

- Implemented criteria include:
  - `BestRSSCriterion`
  - `AICCriterion`
  - `BICCriterion`
  - `AICcCriterion`
  - `HQICCriterion`
  - `EBICCriterion`
  - `GCVCriterion`
- CV routines reject unsupported information criteria where assumptions do not hold across folds.

---

## `selection/definitions.py` – immutable sufficient statistics
- `GramData`
  - Holds `gram = X^T X`, `cov = X^T y`, `y_norm = y^T y`, and `n_samples`.
  - Enforces shape/type/positivity invariants in `__post_init__`.

- `CrossValGramData`
  - Holds per-fold `GramData` and aggregates (`gram_total`, `cov_total`, `y_norm_total`).
  - Provides:
    - `val_data_for_fold(k)`
    - `train_data_for_fold(k)` via complement subtraction
    - `make_full_data()`
  - Validates fold consistency and sample counts.

---

## `selection/state.py` – reference mutable state objects
- `ForwardDeltaCache`
  - Forward-step cache for candidate indices, residual variance/correlation, and post-step RSS.

- `SelectionState`
  - Reference state with active set, full coefficients, active-set coefficients, inverse Gram (`K`), and RSS.
  - Key methods:
    - `init_empty()`, `init_full()`
    - `compute_forward_deltas()`, `apply_forward_step()`
    - `compute_backward_scores()`, `apply_backward_step()`
    - `clone()`
  - Uses rank-one updates/downdates and shared scratch buffers for efficiency.

- `CrossValSelectionState`
  - Wraps one training `SelectionState` per fold and tracks aggregate OOS RSS (`rss_cv`).
  - Key methods:
    - `_sync_active_set()`
    - `recompute_oos_rss()`
    - `apply_backward_step()`
    - `validation_rss_for_candidate()`
    - `validation_rss_for_backward_candidate()`
    - `clone()`

These classes remain the output interface for `.fit(...)` results and some validation/test logic.

---

## `selection/fast_routines.py` – core fast implementation
This module is now the single source of algorithmic behavior for all public selection routines.

- `FastForwardState`
  - Gram-only incremental state:
    - active support (`active_set`, `active_mask`)
    - residual correlation/variance (`r`, `v`)
    - QR-like factors (`Z`, `R`, `qy`)
    - inverse-Gram/cache terms (`K`, `beta_S`)
    - current `rss`
  - Methods:
    - `create(...)`
    - `from_active_set(...)`
    - `candidate_scores()`
    - `apply_forward(...)`
    - `backward_scores()`
    - `apply_backward(...)`

- Fast greedy selectors:
  - `FastForwardSelection`
  - `FastBackwardSelection`
  - `FastMixedSelection`

- Fast beam selectors (single dataset):
  - `FastBeamForwardSelection`
  - `FastBeamBackwardSelection`
  - `FastBeamMixedSelection`
  - Helper node/state types: `FastBeam`, plus internal helpers `_fast_beam_*`.

- Fast CV selectors:
  - `FastCrossValForwardSelection`
  - `FastCrossValBackwardSelection`
  - `FastCrossValMixedSelection`
  - CV scoring helpers:
    - `_fast_cv_rss`
    - `_fast_cv_forward_scores`
    - `_fast_cv_backward_scores`
    - `_rebuild_fast_states`
    - `_build_cv_state_from_active_set`

- Fast CV beam selectors:
  - `FastBeamCrossValForwardSelection`
  - `FastBeamCrossValBackwardSelection`
  - `FastBeamCrossValMixedSelection`
  - Helper node/type: `FastCVBeam`, with internal helpers `_fast_cv_beam_*`.

Important behavior:
- CV candidate scoring uses summed fold validation RSS (same scale as `rss_cv`).
- Beam pruning deduplicates by active-set bitmask signatures.
- Rebuilds from active set are used after accepted moves to limit numerical drift.
- Warm starts are intentionally disallowed for fast beam/CV selectors where state reconstruction assumptions are strict.

---

## `selection/grouped_routines.py` – grouped feature selection
- Provides grouped forward/backward variants on top of Gram statistics.
- Exports:
  - `GroupForwardSelection`
  - `GroupBackwardSelection`
  - `FastGroupForwardSelection`
  - `FastGroupBackwardSelection`

---

## `selection/routines.py` – default public aliases
- Re-exports fast selectors under default names:
  - `ForwardSelection`, `BackwardSelection`, `MixedSelection`
  - `BeamForwardSelection`, `BeamBackwardSelection`, `BeamMixedSelection`
  - `CrossValForwardSelection`, `CrossValBackwardSelection`, `CrossValMixedSelection`
  - `BeamCrossValForwardSelection`, `BeamCrossValBackwardSelection`, `BeamCrossValMixedSelection`

This preserves a stable import surface while keeping implementation fast-only.

---

## `selection/__init__.py` – package API
- Aggregates criteria, data definitions, grouped selectors, fast selectors, and default aliases.
- Both explicit `Fast*` and alias names are exported in `__all__`.

---

## `selection/topk.py`
- Utility helper for selecting top-k candidate indices with deterministic handling for min/max criteria.

---

## Tests (`tests/`)
Major coverage themes:
- Criterion correctness and tolerance logic.
- State update algebra (forward/backward) and rank-deficiency handling.
- Equivalence/consistency across selector families.
- Fast CV scoring correctness vs explicit OOS computations.
- Beam behavior (deduplication, pruning, regression checks, deterministic behavior).
- Golden-output and regression stability checks.

Notable files:
- `tests/test_state_management.py`
- `tests/test_selection_routines.py`
- `tests/test_fast_cv_explicit_oos.py`
- `tests/test_fast_cv_beam_selection.py`
- `tests/test_fast_equivalence_sweeps.py`
- `tests/test_fast_oracle_exhaustive.py`
- `tests/test_golden_outputs.py`

---

## Data-flow summary
1. Build `GramData` / `CrossValGramData`.
2. Run a selector from `selection.routines` (alias) or `selection.fast_routines` (explicit).
3. Internal optimization runs on fast Gram-only states.
4. Final result is materialized as `SelectionState` or `CrossValSelectionState` with `active_set`, `beta`, and RSS metrics.

This is the current architecture baseline: no separate legacy `beam_search.py`, `beam_utils.py`, or `cv_utils.py` modules remain.
