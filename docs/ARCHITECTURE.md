# Architecture Map

This document is a fast orientation guide for contributors and downstream users.
It describes how the repository is split, how data moves through the system, and
where to extend behavior safely.

## Repository Boundaries

| Area | Purpose | Notes |
|---|---|---|
| `selection/` | Core model-selection engine over Gram statistics | Main reusable library surface |
| `benchmarks/` | Reproducible evaluation harness, synthetic suites, reports | Research tooling, not required for library embedding |
| `tests/` | Behavior, regression, property, and contract checks | Mirrors both library and benchmark expectations |
| `docs/` | Porting/contract/architecture guidance | Contributor-facing documentation |

## Core Data Flow (`selection/`)

1. Build `GramData` (or `CrossValGramData`) from sufficient statistics.
2. Instantiate selector (`ForwardSelection`, `BeamForwardSelection`, CV variants, grouped variants).
3. Run `fit(...)` to produce state (`SelectionState`, `CrossValSelectionState`, `GroupedSelectionState`).
4. Consume state fields (`active_set`, `beta`, `rss`/`rss_cv`) in downstream pipelines.

Selector control-flow note:
- Forward-like selectors are budget-driven by default and require explicit forward budgets.
- `stop_on_no_improvement=True` restores the legacy forward self-stop behavior.
- Backward selectors and mixed backward cleanup remain improvement-driven.

Key design principle:
- Boundary components validate aggressively.
- Internal kernels assume validated inputs and focus on numerical updates.

See also:
- `docs/CONTRACTS.md`
- `docs/PORTING.md`

## Benchmark Data Flow (`benchmarks/`)

1. Generate datasets with known support (`benchmarks/datasets.py` and `benchmarks/synthetic_datasets/`).
2. Execute methods and baselines (`benchmarks/methods.py`, `benchmarks/baselines.py`).
3. Compute metrics (`benchmarks/metrics.py`).
4. Aggregate stability/effectiveness summaries (`benchmarks/stability_utils.py`).
5. Render reports (`benchmarks/stability_report.py`, `benchmarks/dominance_report.py`).

Primary entrypoint:
- `python3 benchmarks/stability.py --profile full --seed-start 202600`

## Validation And Failure Semantics

Validation layers:
1. `selection.GramData` / `selection.CrossValGramData` and selector interfaces reject schema/type/range misuse.
2. Selector validation utilities normalize and guard runtime parameters.
3. Numerical kernels (`selection.core.solvers`, incremental updates) retain targeted safety checks for unstable linear algebra paths.

Expected failures:
- Boundary misuse: `TypeError` / `ValueError`
- Ill-conditioned strict solve paths: `np.linalg.LinAlgError`
- Invalid screened candidates: `None` / `inf` scoring behavior (algorithm-specific)

## Public API Vs Internal Details

Stable public import surfaces:
- `selection`
- `selection.criteria`

Everything else under `selection/` is internal and may change without compatibility guarantees.

## Extension Paths

Add a new criterion:
1. Implement in `selection/criteria.py`.
2. Register mapping in benchmark method resolution (`benchmarks/methods.py`) if benchmark usage is needed.
3. Add unit tests for score behavior and selector integration.

Add a new selector:
1. Implement in `selection/selectors/routines*.py` (or grouped counterpart).
2. Expose through `selection/__init__.py`.
3. Add contract tests for state fields, stopping behavior, and failure semantics.
4. Optionally add benchmark method config for comparative runs.

Add a new benchmark baseline:
1. Implement in `benchmarks/baselines.py`.
2. Register in `BASELINE_MAP`.
3. Add focused unit tests and include in stability profiles if desired.

Add a new synthetic scenario:
1. Define scenario metadata in `benchmarks/synthetic_datasets/catalog.py`.
2. Ensure `description`, `checks`, and `why_hard` are explicit.
3. Confirm scenario appears in stability reports and dominance summaries.

## Reproducibility Notes

- Stability profiles are deterministic with explicit seed schedules.
- Prefer compact summary artifacts for version control (summary/report/dominance), not raw row dumps.
- Treat benchmark outputs as reproducible snapshots tied to code + config, not permanent truths.
