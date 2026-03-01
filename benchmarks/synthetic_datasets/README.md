# Synthetic Datasets

Curated support-recovery scenarios used by the stability benchmark pipeline.

Design goals:
- known true support for recovery metrics,
- fixed support across seeds via `support_seed`,
- progressively harder conditions (noise, collinearity, and `p/n` ratio).

Use from Python:

```python
from benchmarks.synthetic_datasets import progressive_support_recovery_scenarios

scenarios = progressive_support_recovery_scenarios()
```
