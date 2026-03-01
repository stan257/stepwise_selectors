#!/usr/bin/env python3
"""Run reproducible multi-seed stability benchmarks on synthetic suites."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.datasets import build_dataset
from benchmarks.methods import run_method
from benchmarks.metrics import collect_metrics
from benchmarks.oracle import exact_best_subset_train_rss
from benchmarks.stability_utils import render_stability_markdown, summarize_stability_rows
from benchmarks.synthetic_datasets import stability_scenarios_for_profile


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_output_paths() -> tuple[Path, Path, Path]:
    out_dir = ROOT / "benchmarks" / "results"
    return (
        out_dir / "stability_rows.jsonl",
        out_dir / "stability_summary.json",
        out_dir / "stability_report.md",
    )


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def _default_methods(
    *, support_size: int, include_external_baselines: bool
) -> list[dict]:
    # In this synthetic suite we know true support size and use it as the
    # selection budget to evaluate search quality (not size-penalty tuning).
    max_steps = int(support_size)
    methods = [
        {
            "name": "forward_bic",
            "selector": "ForwardSelection",
            "selector_params": {"criterion": "BICCriterion"},
            "fit_params": {"max_steps": max_steps},
        },
        {
            "name": "beam_forward_bic_w4",
            "selector": "BeamForwardSelection",
            "selector_params": {"criterion": "BICCriterion", "beam_width": 4},
            "fit_params": {"max_steps": max_steps},
        },
        {
            "name": "mixed_bic",
            "selector": "MixedSelection",
            "selector_params": {"criterion": "BICCriterion"},
            "fit_params": {
                "max_forward_steps": max_steps,
                "max_total_steps": max_steps * 2,
            },
        },
        {
            "name": "cv_forward_sum_rss",
            "selector": "CrossValForwardSelection",
            "selector_params": {
                "criterion": "BestRSSCriterion",
                "cv_aggregation": "sum_rss",
            },
            "cv_folds": 5,
            "fit_params": {"max_steps": max_steps},
        },
        {
            "name": "cv_forward_mean_mse",
            "selector": "CrossValForwardSelection",
            "selector_params": {
                "criterion": "BestRSSCriterion",
                "cv_aggregation": "mean_mse",
            },
            "cv_folds": 5,
            "fit_params": {"max_steps": max_steps},
        },
        {
            "name": "topk_abs_cov",
            "baseline": "TopKAbsCovBaseline",
            "baseline_params": {"k": int(support_size)},
        },
    ]
    if include_external_baselines:
        methods.extend(
            [
                {
                    "name": "lasso_cv",
                    "baseline": "LassoCVBaseline",
                    "baseline_params": {"cv_folds": 5, "random_state": 0},
                },
                {
                    "name": "adaptive_lasso_cv",
                    "baseline": "AdaptiveLassoBaseline",
                    "baseline_params": {
                        "cv_folds": 5,
                        "random_state": 0,
                        "gamma": 1.0,
                    },
                },
            ]
        )
    return methods


def _profile_defaults(profile: Literal["quick", "full"]) -> dict:
    match profile:
        case "quick":
            return {
                "oracle_max_features": 18,
                "oracle_max_combinations": 60000,
            }
        case "full":
            return {
                "oracle_max_features": 22,
                "oracle_max_combinations": 220000,
            }
        case _:
            raise ValueError(f"Unsupported profile: {profile!r}")


def _scenario_seed_count(
    *,
    profile: Literal["quick", "full"],
    scenario,
    seeds_override: int | None,
) -> int:
    if seeds_override is not None:
        return int(seeds_override)
    match profile:
        case "quick":
            return int(scenario.quick_seeds)
        case "full":
            return int(scenario.full_seeds)
        case _:
            raise ValueError(f"Unsupported profile: {profile!r}")


def _scenario_seed_value(seed_start: int, scenario_idx: int, seed_offset: int) -> int:
    # Keep deterministic seed partitions per scenario so profile changes do not
    # silently reshuffle seeds for existing scenarios.
    return int(seed_start + scenario_idx * 1000 + seed_offset)


def _oracle_payload(oracle_result) -> dict:
    payload = asdict(oracle_result)
    payload["objective"] = "exact_train_rss_fixed_k"
    return payload


def run_stability_benchmark(
    *,
    profile: Literal["quick", "full"],
    seeds_override: int | None,
    seed_start: int,
    rows_output_path: Path,
    strict: bool,
    oracle_max_features: int,
    oracle_max_combinations: int,
    include_external_baselines: bool,
) -> tuple[list[dict], list[dict]]:
    if seeds_override is not None and seeds_override <= 0:
        raise ValueError("seeds_override must be > 0 when provided.")
    if oracle_max_features <= 0:
        raise ValueError("oracle_max_features must be > 0.")
    if oracle_max_combinations <= 0:
        raise ValueError("oracle_max_combinations must be > 0.")

    scenarios = stability_scenarios_for_profile(profile)
    rows: list[dict] = []
    scenario_plan: list[dict] = []
    git_sha = _git_sha()
    python_version = platform.python_version()
    numpy_version = np.__version__

    for scenario_idx, scenario in enumerate(scenarios):
        n_scenario_seeds = _scenario_seed_count(
            profile=profile,
            scenario=scenario,
            seeds_override=seeds_override,
        )
        if n_scenario_seeds <= 0:
            raise ValueError(
                f"Scenario {scenario.name!r} resolved to non-positive seeds: {n_scenario_seeds}."
            )
        scenario_plan.append(
            {
                "name": scenario.name,
                "difficulty": scenario.difficulty,
                "checks": scenario.checks,
                "why_hard": scenario.why_hard,
                "seeds": n_scenario_seeds,
            }
        )
        for seed_offset in range(n_scenario_seeds):
            run_seed = _scenario_seed_value(seed_start, scenario_idx, seed_offset)
            dataset_cfg = dict(scenario.dataset)
            dataset_cfg["seed"] = run_seed
            dataset = build_dataset(dataset_cfg)

            support_size = int(dataset_cfg["support_size"])
            methods = _default_methods(
                support_size=support_size,
                include_external_baselines=include_external_baselines,
            )
            oracle_result = exact_best_subset_train_rss(
                dataset,
                k=support_size,
                max_features=oracle_max_features,
                max_combinations=oracle_max_combinations,
            )

            for method_cfg in methods:
                row = {
                    "created_at_utc": _utc_now(),
                    "git_sha": git_sha,
                    "python": python_version,
                    "numpy": numpy_version,
                    "profile": profile,
                    "scenario_name": scenario.name,
                    "difficulty": scenario.difficulty,
                    "scenario_description": scenario.description,
                    "scenario_checks": scenario.checks,
                    "scenario_why_hard": scenario.why_hard,
                    "dataset_seed": int(run_seed),
                    "dataset_config": dataset_cfg,
                    "method_config": method_cfg,
                    "method_name": str(method_cfg["name"]),
                    "status": "ok",
                }
                if oracle_result is None:
                    row["oracle_status"] = "skipped"
                else:
                    row["oracle_status"] = "available"
                    row["oracle"] = _oracle_payload(oracle_result)

                started = time.perf_counter()
                try:
                    result = run_method(method_cfg, dataset)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    row["selector"] = result.selector_name
                    row["active_set"] = result.active_set
                    row["true_support"] = [int(i) for i in dataset.true_support.tolist()]
                    row["metrics"] = collect_metrics(result.state, dataset, elapsed_ms)
                    if oracle_result is not None:
                        row["oracle_gap_train_rss"] = float(
                            row["metrics"]["train_rss"] - oracle_result.train_rss
                        )
                        row["oracle_gap_test_mse"] = float(
                            row["metrics"]["test_mse"] - oracle_result.test_mse
                        )
                except Exception as exc:
                    row["status"] = "error"
                    row["selector"] = str(method_cfg.get("selector", method_cfg.get("baseline", "unknown")))
                    row["error_type"] = type(exc).__name__
                    row["error_message"] = str(exc)
                    row["traceback"] = traceback.format_exc()
                    if strict:
                        rows.append(row)
                        rows_output_path.parent.mkdir(parents=True, exist_ok=True)
                        with rows_output_path.open("w", encoding="utf-8") as out:
                            for item in rows:
                                out.write(json.dumps(item, sort_keys=True) + "\n")
                        raise
                rows.append(row)

    rows_output_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + "\n")
    return rows, scenario_plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run synthetic stability benchmark suite and emit summary artifacts."
    )
    rows_default, summary_default, report_default = _default_output_paths()
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
        help="Scenario/seed profile for runtime-vs-confidence tradeoff.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Optional override for seeds per scenario; otherwise profile defaults.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=202600,
        help="Base seed used for scenario runs (inclusive).",
    )
    parser.add_argument(
        "--rows-output",
        default=str(rows_default),
        help="JSONL output path for per-run rows.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(summary_default),
        help="JSON output path for aggregated summary.",
    )
    parser.add_argument(
        "--report-output",
        default=str(report_default),
        help="Markdown output path for human-readable summary.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort immediately on first method error.",
    )
    parser.add_argument(
        "--oracle-max-features",
        type=int,
        default=None,
        help="Optional upper bound on p for exact-subset oracle.",
    )
    parser.add_argument(
        "--oracle-max-combinations",
        type=int,
        default=None,
        help="Optional upper bound on C(p, k) for exact-subset oracle.",
    )
    parser.add_argument(
        "--include-external-baselines",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include external baselines (LassoCV, AdaptiveLasso). "
            "Defaults to enabled for full profile and disabled for quick."
        ),
    )
    args = parser.parse_args()

    profile = str(args.profile)
    defaults = _profile_defaults(profile)
    oracle_max_features = (
        int(args.oracle_max_features)
        if args.oracle_max_features is not None
        else int(defaults["oracle_max_features"])
    )
    oracle_max_combinations = (
        int(args.oracle_max_combinations)
        if args.oracle_max_combinations is not None
        else int(defaults["oracle_max_combinations"])
    )
    if args.include_external_baselines is None:
        include_external_baselines = profile == "full"
    else:
        include_external_baselines = bool(args.include_external_baselines)
    if include_external_baselines and not _sklearn_available():
        print(
            "Warning: scikit-learn not available; disabling external baselines.",
            file=sys.stderr,
        )
        include_external_baselines = False

    rows_output_path = Path(args.rows_output).resolve()
    summary_output_path = Path(args.summary_output).resolve()
    report_output_path = Path(args.report_output).resolve()

    rows, scenario_plan = run_stability_benchmark(
        profile=profile,
        seeds_override=args.seeds,
        seed_start=int(args.seed_start),
        rows_output_path=rows_output_path,
        strict=bool(args.strict),
        oracle_max_features=oracle_max_features,
        oracle_max_combinations=oracle_max_combinations,
        include_external_baselines=include_external_baselines,
    )

    summary_payload = summarize_stability_rows(rows)
    summary_payload["config"] = {
        "profile": profile,
        "seeds_override": None if args.seeds is None else int(args.seeds),
        "seed_start": int(args.seed_start),
        "oracle_max_features": oracle_max_features,
        "oracle_max_combinations": oracle_max_combinations,
        "include_external_baselines": include_external_baselines,
        "scenario_plan": scenario_plan,
    }

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    report = render_stability_markdown(summary_payload)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(report, encoding="utf-8")

    print(f"Wrote rows: {rows_output_path}")
    print(f"Wrote summary: {summary_output_path}")
    print(f"Wrote report: {report_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
