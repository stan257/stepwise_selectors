#!/usr/bin/env python3
"""Check benchmark JSONL results against threshold baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _load_results(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise TypeError(f"Expected object row at line {line_no} in {path}")
            rows.append(obj)
    if not rows:
        raise ValueError(f"No rows found in results file: {path}")
    return rows


def _index_latest_ok_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        spec_name = str(row.get("spec_name", ""))
        method_name = str(row.get("method_name", ""))
        key = (spec_name, method_name)
        indexed[key] = row
    return indexed


def _check_threshold(metric_name: str, actual: float, expected: float, mode: str) -> bool:
    if mode == "max":
        return actual <= expected
    if mode == "min":
        return actual >= expected
    raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark results against baseline thresholds.")
    parser.add_argument("--results", required=True, help="Path to benchmark JSONL output.")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON thresholds.")
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    baseline_path = Path(args.baseline).resolve()

    rows = _load_results(results_path)
    baseline = _load_json(baseline_path)
    indexed = _index_latest_ok_rows(rows)

    failures: list[str] = []
    checks = 0

    for spec_name, method_thresholds in baseline.items():
        if not isinstance(method_thresholds, dict):
            raise TypeError(f"Expected object for baseline spec {spec_name!r}")
        for method_name, thresholds in method_thresholds.items():
            if not isinstance(thresholds, dict):
                raise TypeError(
                    f"Expected object for baseline spec/method {spec_name!r}/{method_name!r}"
                )
            row = indexed.get((spec_name, method_name))
            if row is None:
                failures.append(f"Missing benchmark row for {spec_name}/{method_name}")
                continue

            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                failures.append(f"Missing metrics for {spec_name}/{method_name}")
                continue

            for key, expected in thresholds.items():
                if key.startswith("max_"):
                    metric_key = key.removeprefix("max_")
                    mode = "max"
                elif key.startswith("min_"):
                    metric_key = key.removeprefix("min_")
                    mode = "min"
                else:
                    raise ValueError(
                        f"Unsupported threshold key {key!r} for {spec_name}/{method_name}. "
                        "Use max_<metric> or min_<metric>."
                    )

                actual = metrics.get(metric_key)
                if actual is None:
                    failures.append(
                        f"Metric {metric_key!r} missing for {spec_name}/{method_name}"
                    )
                    continue

                checks += 1
                ok = _check_threshold(metric_key, float(actual), float(expected), mode)
                if not ok:
                    comparator = "<=" if mode == "max" else ">="
                    failures.append(
                        f"{spec_name}/{method_name}: {metric_key}={actual:.6g} "
                        f"must be {comparator} {float(expected):.6g}"
                    )

    if failures:
        print("Benchmark regression check failed:")
        for failure in failures:
            print(f"- {failure}")
        print(f"Failed {len(failures)} checks out of {checks} executed thresholds.")
        return 1

    print(f"Benchmark regression check passed ({checks} threshold checks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
