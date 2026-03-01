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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.datasets import build_dataset
from benchmarks.methods import run_method
from benchmarks.metrics import collect_metrics
from benchmarks.stability_utils import render_stability_markdown, summarize_stability_rows
from benchmarks.synthetic_datasets import progressive_support_recovery_scenarios


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


def _default_methods(*, support_size: int) -> list[dict]:
    # In this synthetic suite we know true support size and use it as the
    # selection budget to evaluate search quality (not size-penalty tuning).
    max_steps = int(support_size)
    return [
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


def run_stability_benchmark(
    *,
    n_seeds: int,
    seed_start: int,
    rows_output_path: Path,
    strict: bool,
) -> list[dict]:
    if n_seeds <= 0:
        raise ValueError("n_seeds must be > 0.")

    scenarios = progressive_support_recovery_scenarios()
    rows: list[dict] = []
    git_sha = _git_sha()
    python_version = platform.python_version()
    numpy_version = np.__version__

    for scenario in scenarios:
        for seed_offset in range(n_seeds):
            run_seed = seed_start + seed_offset
            dataset_cfg = dict(scenario.dataset)
            dataset_cfg["seed"] = run_seed
            dataset = build_dataset(dataset_cfg)

            methods = _default_methods(support_size=int(dataset_cfg["support_size"]))

            for method_cfg in methods:
                row = {
                    "created_at_utc": _utc_now(),
                    "git_sha": git_sha,
                    "python": python_version,
                    "numpy": numpy_version,
                    "scenario_name": scenario.name,
                    "difficulty": scenario.difficulty,
                    "scenario_description": scenario.description,
                    "dataset_seed": int(run_seed),
                    "dataset_config": dataset_cfg,
                    "method_config": method_cfg,
                    "method_name": str(method_cfg["name"]),
                    "status": "ok",
                }

                started = time.perf_counter()
                try:
                    result = run_method(method_cfg, dataset)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    row["selector"] = result.selector_name
                    row["active_set"] = result.active_set
                    row["true_support"] = [int(i) for i in dataset.true_support.tolist()]
                    row["metrics"] = collect_metrics(result.state, dataset, elapsed_ms)
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
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run synthetic stability benchmark suite and emit summary artifacts."
    )
    rows_default, summary_default, report_default = _default_output_paths()
    parser.add_argument("--seeds", type=int, default=12, help="Seeds per scenario.")
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
    args = parser.parse_args()

    rows_output_path = Path(args.rows_output).resolve()
    summary_output_path = Path(args.summary_output).resolve()
    report_output_path = Path(args.report_output).resolve()

    rows = run_stability_benchmark(
        n_seeds=int(args.seeds),
        seed_start=int(args.seed_start),
        rows_output_path=rows_output_path,
        strict=bool(args.strict),
    )

    summary_payload = summarize_stability_rows(rows)
    summary_payload["config"] = {
        "seeds": int(args.seeds),
        "seed_start": int(args.seed_start),
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
