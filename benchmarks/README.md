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

Run higher-confidence full profile (more seeds on hard scenarios):

```bash
python3 benchmarks/stability.py --profile full
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
- `twin_decoys_per_signal`: number of near-duplicate decoys per true feature.
- `twin_strength`: linear coupling of twin decoys to source feature in `[0, 1]`.
- `twin_noise_std`: additive noise scale for twin decoys.
- `nonlinear_strength`: magnitude of nonlinear target component (misspecification stress).

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

Stability summary now includes:

- mean test/support metrics with 95% CI,
- paired-seed win rates vs `forward_bic` and `topk_abs_cov`,
- optional oracle gap vs exact train-RSS best subset (when `p` and `C(p,k)` are under configured limits).

## Synthetic Dataset Suite

`benchmarks/synthetic_datasets/` provides a curated set of progressively harder
support-recovery scenarios, plus explicit failure modes:

- ultra-collinear twin decoys,
- misspecified nonlinear targets,
- small-`p` oracle-check scenario for exact subset gaps.

## CI Smoke Gate

`benchmark-smoke` workflow runs the smoke spec on pushes/PRs, checks threshold guards from `benchmarks/baseline.json`, and uploads JSONL + markdown artifacts.
