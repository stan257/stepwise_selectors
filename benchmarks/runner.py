#!/usr/bin/env python3
"""Run deterministic benchmark specs and emit JSONL results."""

from __future__ import annotations

import argparse
import json
import os
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
from benchmarks.metrics import collect_metrics
from benchmarks.methods import run_method


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def _load_spec(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Spec at {path} must be a JSON object.")
    if "name" not in cfg:
        raise ValueError(f"Spec at {path} is missing required key: 'name'.")
    if "dataset" not in cfg:
        raise ValueError(f"Spec at {path} is missing required key: 'dataset'.")
    if "methods" not in cfg:
        raise ValueError(f"Spec at {path} is missing required key: 'methods'.")
    if not isinstance(cfg["methods"], list) or not cfg["methods"]:
        raise ValueError(f"Spec at {path} has empty or invalid 'methods'.")
    return cfg


def _default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "benchmarks" / "results" / f"run_{stamp}.jsonl"


def _iter_spec_paths(spec_args: list[str], spec_dir: Path) -> list[Path]:
    if spec_args:
        return [Path(p).resolve() for p in spec_args]

    if not spec_dir.exists():
        raise FileNotFoundError(f"Spec directory not found: {spec_dir}")

    specs = sorted(spec_dir.glob("*.json"))
    if not specs:
        raise FileNotFoundError(f"No JSON specs found in: {spec_dir}")
    return specs


def _record_base(run_id: str, spec_name: str, spec_path: Path) -> dict:
    return {
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "spec_name": spec_name,
        "spec_path": str(spec_path),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _ensure_output_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark specs and emit JSONL rows.")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Path to a JSON spec file. Repeat to run multiple. If omitted, all specs in --spec-dir are run.",
    )
    parser.add_argument(
        "--spec-dir",
        default=str(ROOT / "benchmarks" / "specs"),
        help="Directory used when --spec is omitted (default: benchmarks/specs).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: benchmarks/results/run_<utc>.jsonl).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output if it already exists.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first method failure.",
    )
    args = parser.parse_args()

    spec_dir = Path(args.spec_dir).resolve()
    spec_paths = _iter_spec_paths(args.spec, spec_dir)

    output_path = Path(args.output).resolve() if args.output else _default_output_path()
    _ensure_output_parent(output_path)

    mode = "a" if args.append else "w"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    failures = 0
    rows_written = 0
    with output_path.open(mode, encoding="utf-8") as out:
        for spec_path in spec_paths:
            spec_cfg = _load_spec(spec_path)
            dataset = build_dataset(spec_cfg["dataset"])
            base = _record_base(run_id, str(spec_cfg["name"]), spec_path)
            base.update(
                {
                    "dataset_name": dataset.name,
                    "dataset_seed": int(dataset.seed),
                    "n_train": int(dataset.X_train.shape[0]),
                    "n_val": int(dataset.X_val.shape[0]),
                    "n_test": int(dataset.X_test.shape[0]),
                    "n_features": int(dataset.X_train.shape[1]),
                    "true_support": [int(i) for i in dataset.true_support.tolist()],
                }
            )

            for method_cfg in spec_cfg["methods"]:
                row = dict(base)
                row["method_config"] = method_cfg
                row["status"] = "ok"

                started = time.perf_counter()
                try:
                    result = run_method(method_cfg, dataset)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    row["method_name"] = result.method_name
                    row["selector"] = result.selector_name
                    row["active_set"] = result.active_set
                    row["metrics"] = collect_metrics(result.state, dataset, elapsed_ms)
                except Exception as exc:
                    failures += 1
                    row["status"] = "error"
                    row["method_name"] = str(method_cfg.get("name", "unknown"))
                    row["selector"] = str(method_cfg.get("selector", "unknown"))
                    row["error_type"] = type(exc).__name__
                    row["error_message"] = str(exc)
                    row["traceback"] = traceback.format_exc()
                    if args.strict:
                        out.write(json.dumps(row, sort_keys=True) + "\n")
                        rows_written += 1
                        print(f"Wrote {rows_written} row(s) to {output_path}")
                        print("Aborting due to --strict and method failure.")
                        return 1

                out.write(json.dumps(row, sort_keys=True) + "\n")
                rows_written += 1

    print(f"Wrote {rows_written} row(s) to {output_path}")
    if failures:
        print(f"Completed with {failures} method failure(s).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
