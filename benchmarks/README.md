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

Run full profile without external (scikit-learn) baselines:

```bash
python3 benchmarks/stability.py --profile full --no-include-external-baselines
```

Run a reproducible "evidence" job with explicit artifact names:

```bash
python3 benchmarks/stability.py --profile full --seed-start 202600 --rows-output benchmarks/results/stability_rows_full.jsonl --summary-output benchmarks/results/stability_summary_full.json --report-output benchmarks/results/stability_report_full.md
```

Re-render stability markdown report from saved summary JSON:

```bash
python3 benchmarks/stability_report.py --summary benchmarks/results/stability_summary.json --output benchmarks/results/stability_report.md
```

## Validating Repo Claims

To validate method-strength claims in a reproducible way:

1. Run the full-profile stability benchmark:

```bash
python3 benchmarks/stability.py --profile full --seed-start 202600 --rows-output benchmarks/results/stability_rows_full.jsonl --summary-output benchmarks/results/stability_summary_full.json --report-output benchmarks/results/stability_report_full.md
```

2. Read `benchmarks/results/stability_report_full.md` for scenario-wise comparison.
3. Use `benchmarks/results/stability_summary_full.json` for programmatic checks.

Recommended interpretation rules:

- Effectiveness: selector methods should outperform `topk_abs_cov` on `mean_test_mse` in most hard scenarios (`delta_test_mse_vs_topk < 0`).
- Consistency: top methods should have reasonably tight `test_mse_95ci` and higher `mean_pairwise_jaccard` than weak baselines on correlated scenarios.
- Search quality: in `oracle_small_p`, strong methods should have near-zero `mean_oracle_gap_test_mse`.
- Pairwise robustness: `win_rate_vs_topk` should usually be above `50%` for competitive selectors.

Optional fast sanity run:

```bash
python3 benchmarks/stability.py --profile quick
```

Build a compact dominance dashboard from the produced summary:

```bash
python3 benchmarks/dominance_report.py --summary benchmarks/results/stability_summary.json --output benchmarks/results/dominance_report.md
```

## Latest Claim Snapshot

From the latest full-profile run (`--profile full --seed-start 202600`):

- Best selector beat `topk_abs_cov` on test MSE in all scenarios.
- Winner relative test-MSE gain vs `topk_abs_cov`: ~`61.8%` to `95.3%` (scenario-dependent).
- `LassoCVBaseline` and `AdaptiveLassoBaseline` did not beat `forward_bic` on mean test MSE in any scenario in this run.
- In `hard_ultra_collinear_twins`, beam search outperformed greedy forward (`beam_forward_bic_w4` vs `forward_bic`) on mean test MSE.
- In `oracle_small_p`, top selector methods had near-zero oracle gap, while `topk_abs_cov` had a large gap.

Treat this as a reproducible snapshot, not a fixed theorem: rerun with the command above when algorithm or dataset settings change.

Suggested report paragraph:
"On a seven-scenario synthetic benchmark suite (easy through very-hard regimes, including collinear decoys, misspecification, and oracle-check cases), Gram-based selector methods consistently outperformed naive covariance ranking, with winner-level test-MSE improvements of approximately 62% to 95% over `TopKAbsCovBaseline`. Relative to external sparse baselines (`LassoCV` and adaptive lasso), the selector family achieved better predictive accuracy across scenarios, while maintaining near-oracle performance in the tractable small-`p` setting. Beam search provided additional gains in decoy-heavy collinearity regimes, indicating practical value beyond single-path greedy selection."

## Artifact Versioning Policy

- Do not commit raw `benchmarks/results/stability_rows*.jsonl` files (large and run-specific).
- Prefer committing compact evidence artifacts only when needed:
  - a summary JSON snapshot,
  - a markdown report,
  - and optionally a dominance report.
- If you want historical snapshots in git, store them under a docs path (for example `docs/benchmarks/`) rather than `benchmarks/results/`.

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
  - `selector`: selector class name from `selection`
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
- `LassoCVBaseline`: `LassoCV` with standardized features; reports coefficients in original feature scale.
- `AdaptiveLassoBaseline`: two-stage adaptive lasso (ridge weights + weighted `LassoCV`).

For `LassoCVBaseline` and `AdaptiveLassoBaseline`, install `scikit-learn`.

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

By default:

- `--profile quick` runs internal selectors + `TopKAbsCovBaseline`.
- `--profile full` also includes `LassoCVBaseline` and `AdaptiveLassoBaseline` when scikit-learn is available.

## Synthetic Dataset Suite

`benchmarks/synthetic_datasets/` provides a curated set of progressively harder
support-recovery scenarios, plus explicit failure modes:

- ultra-collinear twin decoys,
- misspecified nonlinear targets,
- small-`p` oracle-check scenario for exact subset gaps.

## CI Smoke Gate

`benchmark-smoke` workflow runs the smoke spec on pushes/PRs, checks threshold guards from `benchmarks/baseline.json`, and uploads JSONL + markdown artifacts.
