# Benchmarks

Deterministic benchmark harness for evaluating selector behavior across fixed specs.

## Quick Start

Run smoke benchmark + threshold checks + report in one command:

```bash
python3 benchmarks/smoke.py
```

Run multi-seed stability benchmark suite + summary artifacts:

```bash
python3 benchmarks/stability.py
```

Re-render stability markdown report from saved summary JSON:

```bash
python3 benchmarks/stability_report.py --summary benchmarks/results/stability_summary.json --output benchmarks/results/stability_report.md
```

Equivalent manual steps are still available:

Run all specs under `benchmarks/specs`:

```bash
python benchmarks/runner.py
```

Run a specific spec:

```bash
python benchmarks/runner.py --spec benchmarks/specs/smoke_forward.json
```

Validate output against committed thresholds:

```bash
python benchmarks/check_regression.py --results benchmarks/results/smoke_run.jsonl --baseline benchmarks/baseline.json
```

Render a markdown report:

```bash
python benchmarks/report.py --results benchmarks/results/smoke_run.jsonl --output benchmarks/results/smoke_report.md
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
- `kind`: `"synthetic_support_recovery"` (correlated synthetic suite with known support)
- `seed`: integer random seed.
- `n_samples`: total sample count.
- `n_features`: feature count.
- `support_size`: number of true nonzero coefficients.
- `noise_std`: Gaussian noise std.
- `signal_scale`: scale for true coefficients.
- `train_fraction`: split fraction for train.
- `val_fraction`: split fraction for validation.

Test fraction is `1 - train_fraction - val_fraction`.

Additional support-recovery options:

- `correlation`: Toeplitz feature-correlation parameter in `[0, 1)`.
- `clustered_support`: if `true`, support indices are contiguous.
- `support_seed`: optional fixed seed for true-support identity across runs.
- `min_signal_abs`: optional lower bound on absolute true coefficient magnitudes.

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

- `criterion`: e.g. `"AICCriterion"`, `"BestRSSCriterion"`

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

Stability pipeline outputs:

- `stability_rows.jsonl`: per-seed run rows.
- `stability_summary.json`: aggregated stability/effectiveness metrics.
- `stability_report.md`: scenario-wise markdown summary table.

## Synthetic Dataset Suite

`benchmarks/synthetic_datasets/` provides a curated set of progressively harder
support-recovery scenarios (easy -> very_hard) with known true support.

## CI Smoke Gate

`benchmark-smoke` workflow runs the smoke spec on pushes/PRs, checks threshold guards from `benchmarks/baseline.json`, and uploads JSONL + markdown artifacts.
