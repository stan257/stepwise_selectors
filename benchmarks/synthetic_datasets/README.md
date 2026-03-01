# Synthetic Datasets

Curated support-recovery scenarios used by the stability benchmark pipeline.

Design goals:
- known true support for recovery metrics,
- fixed support across seeds via `support_seed`,
- progressively harder conditions (noise, collinearity, and `p/n` ratio),
- explicit failure modes (twin decoys and nonlinear misspecification),
- one small-`p` scenario for exact-subset oracle gap checks.
- researcher-facing commentary per scenario:
  - `description`: what the dataset is,
  - `checks`: what behavior the benchmark should validate,
  - `why_hard`: why the scenario is challenging.

Use from Python:

```python
from benchmarks.synthetic_datasets import progressive_support_recovery_scenarios

scenarios = progressive_support_recovery_scenarios()
```

For profile-aware stability runs:

```python
from benchmarks.synthetic_datasets import stability_scenarios_for_profile

scenarios = stability_scenarios_for_profile("quick")
```
