# useful selection toolkit

Lightweight linear-model selection routines built on precomputed Gram statistics. Includes:
- Greedy forward/backward/mixed selection with AIC and RSS criteria.
- Beam-search variants for exploring multiple active sets.
- Cross-validation counterparts over per-fold Grams.
- Optional grouped forward/backward selection.

## Layout
- `selection/`: core code (criteria, state updates, beam utils, grouped variants).
- `tests/`: pytest suite covering selectors, state management, and grouped routines.
- `summary.md`: annotated walkthrough of the selection codebase.

## API overview
- Data containers: `GramData`, `CrossValGramData`
- Greedy routines: `ForwardSelection`, `BackwardSelection`, `MixedSelection`
- Beam routines: `BeamForwardSelection`, `BeamBackwardSelection`, `BeamMixedSelection`
- Cross-val routines: `CrossValForwardSelection`, `CrossValBackwardSelection`, `CrossValMixedSelection`
- Beam + cross-val: `BeamCrossValForwardSelection`, `BeamCrossValBackwardSelection`, `BeamCrossValMixedSelection`
- Grouped routines: `GroupForwardSelection`, `GroupBackwardSelection`
- Criteria: `AICCriterion`, `BestRSSCriterion`
- Reference (slow) implementations live in `selection.legacy_routines` and
  `selection.legacy_grouped_routines`.

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
print(gstate.active_groups, gstate.beta, gstate.rss)
```

## Notes
- The code operates on Gram statistics (`X.T @ X`, `X.T @ y`, `y.T @ y`) and does not depend on raw design matrices.
- `selection.routines` and `selection.grouped_routines` now default to the fast implementations; the reference versions are under `selection.legacy_*`.
- For cross-validation, provide per-fold `GramData` via `CrossValGramData`.
- Beam search and mixed routines rely on the same criterion interface (default AIC; can swap in `BestRSSCriterion`).
