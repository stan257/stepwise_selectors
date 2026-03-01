# useful selection toolkit

Lightweight linear-model selection routines built on precomputed Gram statistics. Includes:
- Greedy forward/backward/mixed selection with AIC and RSS criteria.
- Beam-search variants for exploring multiple active sets.
- Cross-validation counterparts over per-fold Grams.
- Optional grouped forward/backward selection.

## Layout
- `selection/`: core code (criteria, state objects, selection routines, grouped variants).
- `docs/PORTING.md`: exact porting contract (data interfaces, equations, failure semantics).
- `docs/CONTRACTS.md`: validation-boundary and precondition contract for contributors.
- `tests/`: pytest suite organized by category:
  - `tests/unit/`
  - `tests/integration/`
  - `tests/property/`
  - `tests/regression/`
  - shared helpers in `tests/helpers/`
- `summary.md`: annotated walkthrough of the selection codebase.

## API overview
- Data containers: `GramData`, `CrossValGramData`
- Output states: `SelectionState`, `CrossValSelectionState`, `GroupedSelectionState`
- Greedy routines: `ForwardSelection`, `BackwardSelection`, `MixedSelection`
- Beam routines: `BeamForwardSelection`, `BeamBackwardSelection`, `BeamMixedSelection`
- Cross-val routines: `CrossValForwardSelection`, `CrossValBackwardSelection`, `CrossValMixedSelection`
- Beam + cross-val: `BeamCrossValForwardSelection`, `BeamCrossValBackwardSelection`, `BeamCrossValMixedSelection`
- Grouped routines: `GroupForwardSelection`, `GroupBackwardSelection`
- Criteria: `AICCriterion`, `BestRSSCriterion`, `CriterionProtocol`

## Public API Boundary
- Stable import surface:
  - `selection.routines`
  - `selection.grouped_routines`
  - `selection.definitions`
  - `selection.criteria`
  - `selection` (package-level re-exports)
- Internal modules in `selection/*` outside those entrypoints are implementation details and may change without compatibility guarantees.

## Selector guide
- Use `ForwardSelection` when you want a fast, deterministic baseline and a single support path.
- Use `BackwardSelection` when starting from full support is meaningful and feature removal cost is acceptable.
- Use `MixedSelection` when greedy forward may over-select and periodic backward cleanup is desirable.
- Use beam variants when local greedy choices are likely brittle and you can spend extra compute.
- Use CV variants when model-size regularization should be driven by held-out error instead of IC penalties.

## Quick start
1) Install deps (Python 3.12+; minimal requirements: `numpy`, `pytest`). In conda:
```bash
conda create -n jax-arm python=3.12 numpy pytest
conda activate jax-arm
```
2) Install the package (editable):
```bash
python -m pip install -e .
```
3) Run tests:
```bash
pytest tests
```
Filter by category marker:
```bash
pytest -m unit
pytest -m integration
pytest -m property
pytest -m regression
```
4) Use selection routines:
```python
from selection.definitions import GramData
from selection.routines import ForwardSelection

data = GramData(gram, cov, y_norm, n_samples)
state = ForwardSelection().fit(data=data)
print(state.active_set, state.beta, state.rss)
```

### Grouped selection
Pass feature groups as lists of indices:
```python
from selection.grouped_routines import GroupForwardSelection

groups = [[0, 1], [2, 3]]  # add/remove as units
gstate = GroupForwardSelection(groups).fit(data=data)
print(gstate.active_groups, gstate.active_set, gstate.beta, gstate.rss)
```


## Notes
- The code operates on Gram statistics (`X.T @ X`, `X.T @ y`, `y.T @ y`) and does not depend on raw design matrices.
- `selection.routines` and `selection.grouped_routines` expose the default implementations.
- For cross-validation, provide per-fold `GramData` via `CrossValGramData`.
- `GramData` accepts array-like inputs for `gram` and `cov` and stores contiguous NumPy arrays internally.
- CV output states expose `beta` as a post-selection refit on full-data Gram statistics at the selected support.
- Single-dataset selectors default to `AICCriterion`; CV selectors default to `BestRSSCriterion`.
- Selector constructors accept `criterion` (string key, class, instance, or factory). Built-in keys: `rss`, `aic`, `aicc`, `bic`, `hqic`, `ebic`, `gcv`.
- CV selectors reject criteria with `cv_compatible=False` to avoid double regularization on top of held-out RSS.
- Backward beam selectors are improvement-only by default; set `allow_worse=True` to force removals under a step budget.
- Selector hyperparameters are validated strictly (fail-fast): no implicit coercion for `beam_width`, `allow_worse`, step budgets, or `tol`.

## Portability And Embedding
- The package is NumPy-only and operates on Gram statistics, so it ports cleanly into most scientific Python codebases.
- Minimal integration contract:
  - construct `GramData` (or `CrossValGramData`) from your preprocessing pipeline;
  - call selectors from `selection.routines`;
  - consume output state fields (`active_set`, `beta`, `rss` or `rss_cv`).
- Recommended adapter boundary:
  - keep feature engineering and scaling outside this package;
  - treat this package as a pure model-selection engine over fixed sufficient statistics.
- For non-Python ports, `selection/routines_*`, `selection/state_single.py`, `selection/state_cv.py`, and `selection/incremental_solver.py` define the core algebraic behavior to mirror.

## Failure Semantics
- Input schema/type violations raise `TypeError`/`ValueError` at construction time where possible.
- Numerically unstable active-set operations raise `np.linalg.LinAlgError` or return `inf` candidate scores (for screened candidates).
- CV routines require synchronized fold supports internally; desync now fails fast with a `ValueError`.

## Assumptions And Limitations (Research Use)
- The package does **not** fit an intercept automatically. If your model needs one, include a constant feature before building Gram statistics, or center `X`/`y` and fit a no-intercept model.
- Input statistics are assumed to be coherent summaries of the same data matrix/target vector. `GramData` validates shape, symmetry, finiteness, and basic positivity constraints, but it does not prove full PSD correctness.
- Active-set solves rely on Cholesky factorization. Singular or ill-conditioned supports may raise `np.linalg.LinAlgError` during refits or state initialization.
- Criterion values (AIC/BIC/AICc/HQIC/EBIC/GCV) are optimization targets. Their statistical interpretation depends on standard linear-model assumptions (e.g., iid noise, comparable sample definitions).
- CV selectors optimize **summed fold validation RSS** and return `CrossValSelectionState.rss_cv` on that same scale. `CrossValSelectionState.beta` is a post-selection refit on full aggregated data, not a fold-averaged coefficient.
- Improvement checks use absolute+relative tolerances. Very small numerical differences can intentionally stop additional steps.
- Grouped routines require disjoint groups with integer feature indices; selected support is the union of complete groups (`GroupedSelectionState.active_set`).

## Reproducibility Guidance
- Keep preprocessing fixed and deterministic before building Gram statistics (including centering, scaling, and feature ordering).
- Use explicit seeds when generating synthetic data for experiments.
- Treat near-tie selections as potentially unstable across tiny floating-point perturbations; report support sets and objective values, not only one metric.
- For critical claims, validate selected supports against explicit OLS recomputation on the same support (as done in the test suite).
