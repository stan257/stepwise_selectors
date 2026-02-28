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
  - Exposes `beta` as a post-selection full-data refit on the selected support.
  - Key methods:
    - `_sync_active_set()`
    - `_refit_full_data_beta()`
    - `recompute_oos_rss()`
    - `apply_backward_step()`
    - `validation_rss_for_candidate()`
    - `validation_rss_for_backward_candidate()`
    - `clone()`

- `GroupedSelectionState`
  - Typed output container for grouped selectors.
  - Stores `active_groups`, derived `active_set` (union of selected groups), `beta`, and `rss`.

These classes remain the output interface for `.fit(...)` results and some validation/test logic.

---

## Core module layout
The implementation is split by responsibility:

- `selection/forward_state.py`
  - `ForwardState` and its QR/Gram rank-one update and downdate machinery.

- `selection/routines_greedy.py`
  - Single-dataset greedy selectors:
    - `ForwardSelection`
    - `BackwardSelection`
    - `MixedSelection`

- `selection/routines_beam.py`
  - Single-dataset beam selectors:
    - `BeamForwardSelection`
    - `BeamBackwardSelection`
    - `BeamMixedSelection`
  - Beam helper type/functions:
    - `Beam`
    - `_beam_*`

- `selection/routines_cv_scoring.py`
  - Shared CV scoring/rebuild helpers:
    - `_cv_rss`
    - `_cv_forward_scores`
    - `_cv_backward_scores`
    - `_rebuild_states`
    - `_build_cv_state_from_active_set`

- `selection/routines_cv.py`
  - CV greedy selectors:
    - `CrossValForwardSelection`
    - `CrossValBackwardSelection`
    - `CrossValMixedSelection`
  - CV beam selectors:
    - `BeamCrossValForwardSelection`
    - `BeamCrossValBackwardSelection`
    - `BeamCrossValMixedSelection`
  - CV beam helper type/functions:
    - `CVBeam`
    - `_cv_beam_*`

- `selection/routines_core.py`
  - Facade that re-exports the core selector surface.
  - Keeps monkeypatchable private helper hooks used by tests.

Important behavior:
- CV candidate scoring uses summed fold validation RSS (same scale as `rss_cv`).
- `BeamCrossValBackwardSelection` is improvement-only by default; `allow_worse=True` enables forced removals.
- Beam pruning deduplicates by active-set bitmask signatures.
- Rebuilds from active set are used after accepted moves to limit numerical drift.
- Warm starts are intentionally disallowed for beam/CV selectors where state reconstruction assumptions are strict.

---

## `selection/grouped_routines.py` – grouped feature selection
- Provides grouped forward/backward variants on top of Gram statistics.
- Exports:
  - `GroupForwardSelection`
  - `GroupBackwardSelection`

---

## `selection/routines.py` – default public aliases
- Re-exports core selectors under default names:
  - `ForwardSelection`, `BackwardSelection`, `MixedSelection`
  - `BeamForwardSelection`, `BeamBackwardSelection`, `BeamMixedSelection`
  - `CrossValForwardSelection`, `CrossValBackwardSelection`, `CrossValMixedSelection`
  - `BeamCrossValForwardSelection`, `BeamCrossValBackwardSelection`, `BeamCrossValMixedSelection`

This preserves a stable import surface while keeping implementation modular.

---

## `selection/__init__.py` – package API
- Aggregates criteria, data definitions, grouped selectors, and default selectors.

---

## `selection/topk.py`
- Utility helper for selecting top-k candidate indices with deterministic handling for min/max criteria.

---

## Tests (`tests/`)
Major coverage themes:
- Criterion correctness and tolerance logic.
- State update algebra (forward/backward) and rank-deficiency handling.
- Equivalence/consistency across selector families.
- CV scoring correctness vs explicit OOS computations.
- Beam behavior (deduplication, pruning, regression checks, deterministic behavior).
- Golden-output and regression stability checks.

Notable files:
- `tests/unit/test_state.py`
- `tests/integration/test_selection_routines.py`
- `tests/regression/test_fast_cv_explicit_oos.py`
- `tests/unit/test_fast_cv_beam_selection.py`
- `tests/property/test_fast_equivalence_sweeps.py`
- `tests/regression/test_fast_oracle_exhaustive.py`
- `tests/regression/test_golden_outputs.py`

---

## Data-flow summary
1. Build `GramData` / `CrossValGramData`.
2. Run a selector from `selection.routines` (alias) or `selection.routines_core` (explicit).
3. Internal optimization runs on Gram-only states.
4. Final result is materialized as `SelectionState` or `CrossValSelectionState` with `active_set` and RSS metrics; CV `beta` is a full-data post-selection refit.

This is the current architecture baseline: no separate legacy `beam_search.py`, `beam_utils.py`, or `cv_utils.py` modules remain.
