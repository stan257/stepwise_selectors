# Benchmarks

Deterministic benchmark harness for evaluating selector behavior across fixed specs.

## Quick Start

Run all specs under `benchmarks/specs`:

```bash
python benchmarks/runner.py
```

Run a specific spec:

```bash
python benchmarks/runner.py --spec benchmarks/specs/smoke_forward.json
```

Append to an existing output file:

```bash
python benchmarks/runner.py --spec benchmarks/specs/smoke_forward.json --output benchmarks/results/history.jsonl --append
```

## Spec Format

Specs are JSON objects with these keys:

- `name`: unique benchmark name.
- `dataset`: dataset config.
- `methods`: list of method configs.

### Dataset config

Currently supported:

- `kind`: `"synthetic_linear"`
- `seed`: integer random seed.
- `n_samples`: total sample count.
- `n_features`: feature count.
- `support_size`: number of true nonzero coefficients.
- `noise_std`: Gaussian noise std.
- `signal_scale`: scale for true coefficients.
- `train_fraction`: split fraction for train.
- `val_fraction`: split fraction for validation.

Test fraction is `1 - train_fraction - val_fraction`.

### Method config

- `name`: method label in output rows.
- exactly one of:
  - `selector`: selector class name from `selection.routines`
  - `baseline`: baseline name from `benchmarks.baselines`

For selectors:

- `selector_params`: constructor kwargs.
- `fit_params`: kwargs passed to `.fit(...)`.
- `cv_folds`: optional, used only for CV selectors.
- `cv_seed`: optional, used only for CV selectors.

For baselines:

- `baseline_params`: constructor kwargs.

For criteria, pass class names as strings in `selector_params`:

- `criterion_cls`: e.g. `"AICCriterion"`, `"BestRSSCriterion"`
- `criterion`: also accepted

Current built-in baseline:

- `TopKAbsCovBaseline`: selects top-k features by absolute train covariance (`|X^T y|`) and refits OLS on that support.

## Output Format

Each method execution writes one JSON line with:

- run metadata (`run_id`, timestamp, `git_sha`, versions)
- spec and dataset metadata
- method config and selector
- selected support (`active_set`)
- metrics (`train/val/test` RSS+MSE, support precision/recall/F1, elapsed ms)
- error information if the method fails

Generated outputs are written to `benchmarks/results/` by default.
