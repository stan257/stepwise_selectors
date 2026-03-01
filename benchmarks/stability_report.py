#!/usr/bin/env python3
"""Render stability-summary JSON into markdown report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.stability_utils import render_stability_markdown


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render benchmarks/stability.py summary JSON to markdown."
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to stability summary JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional markdown output path; prints to stdout if omitted.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    report = render_stability_markdown(payload)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote stability report: {out_path}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
