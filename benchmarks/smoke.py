#!/usr/bin/env python3
"""Run the smoke benchmark pipeline in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run smoke benchmark + regression checks + report generation."
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=None,
        help="Benchmark spec path. Repeat to run multiple specs.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "benchmarks" / "results" / "smoke_run.jsonl"),
        help="JSONL output path for benchmark rows.",
    )
    parser.add_argument(
        "--baseline",
        default=str(ROOT / "benchmarks" / "baseline.json"),
        help="Baseline threshold file used by regression check.",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "benchmarks" / "results" / "smoke_report.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to output if it already exists.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on the first benchmark method error.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    spec_paths = args.spec or [str(ROOT / "benchmarks" / "specs" / "smoke_forward.json")]

    runner_cmd = [sys.executable, "benchmarks/runner.py", "--output", str(output_path)]
    for spec in spec_paths:
        runner_cmd.extend(["--spec", str(Path(spec).resolve())])
    if args.append:
        runner_cmd.append("--append")
    if args.strict:
        runner_cmd.append("--strict")

    try:
        _run(runner_cmd)
        _run(
            [
                sys.executable,
                "benchmarks/check_regression.py",
                "--results",
                str(output_path),
                "--baseline",
                str(Path(args.baseline).resolve()),
            ]
        )
        _run(
            [
                sys.executable,
                "benchmarks/report.py",
                "--results",
                str(output_path),
                "--output",
                str(report_path),
            ]
        )
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode)

    print(f"Smoke pipeline completed: {output_path} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
