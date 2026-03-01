#!/usr/bin/env python3
"""Render benchmark JSONL results into a compact Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected object row at line {line_no}.")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def _group_by_spec(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        spec_name = str(row.get("spec_name", "unknown"))
        grouped.setdefault(spec_name, []).append(row)
    return grouped


def _build_report(rows: list[dict]) -> str:
    grouped = _group_by_spec(rows)
    lines: list[str] = []

    for spec_name in sorted(grouped):
        lines.append(f"## {spec_name}")
        lines.append("")
        lines.append(
            "| method | selector | n_selected | val_mse | test_mse | support_f1 | elapsed_ms |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")

        spec_rows = sorted(grouped[spec_name], key=lambda r: str(r.get("method_name", "")))
        for row in spec_rows:
            metrics = row.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("method_name", "-")),
                        str(row.get("selector", "-")),
                        _fmt(metrics.get("n_selected")),
                        _fmt(metrics.get("val_mse")),
                        _fmt(metrics.get("test_mse")),
                        _fmt(metrics.get("support_f1")),
                        _fmt(metrics.get("elapsed_ms")),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render benchmark JSONL as Markdown.")
    parser.add_argument("--results", required=True, help="Path to benchmark JSONL output.")
    parser.add_argument("--output", default=None, help="Optional output markdown file path.")
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    rows = _load_rows(results_path)
    report = _build_report(rows)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote benchmark report to {out_path}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
