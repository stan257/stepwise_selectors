#!/usr/bin/env python3
"""Render a compact method-dominance report from stability summary JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Summary payload must be a JSON object.")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("Summary payload must include list field `rows`.")
    return payload


def _group_by_scenario(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        scenario = str(row.get("scenario_name", "unknown"))
        grouped[scenario].append(row)
    return grouped


def _scenario_winner_rows(grouped: dict[str, list[dict]]) -> list[dict]:
    winners: list[dict] = []
    for scenario_name in sorted(grouped):
        rows = sorted(grouped[scenario_name], key=lambda r: float(r["mean_test_mse"]))
        winner = rows[0]
        baseline = next((r for r in rows if r["method_name"] == "topk_abs_cov"), None)
        if baseline is None:
            rel_gain = None
        else:
            denom = float(baseline["mean_test_mse"])
            rel_gain = None if denom <= 0 else (denom - float(winner["mean_test_mse"])) / denom
        winners.append(
            {
                "scenario_name": scenario_name,
                "difficulty": str(winner.get("difficulty", "unknown")),
                "winner": str(winner["method_name"]),
                "winner_test_mse": float(winner["mean_test_mse"]),
                "winner_support_f1": float(winner["mean_support_f1"]),
                "winner_jaccard": float(winner["mean_pairwise_jaccard"]),
                "winner_rel_gain_vs_topk": rel_gain,
            }
        )
    return winners


def _method_scoreboard(grouped: dict[str, list[dict]]) -> list[dict]:
    methods = sorted({str(r["method_name"]) for rows in grouped.values() for r in rows})
    by_method: dict[str, dict] = {
        method: {
            "method": method,
            "scenarios": 0,
            "scenario_wins": 0,
            "top3_finishes": 0,
            "rank_sum": 0.0,
            "rel_gain_vs_topk_values": [],
            "win_rate_vs_topk_values": [],
        }
        for method in methods
    }

    for scenario_name in sorted(grouped):
        rows = sorted(grouped[scenario_name], key=lambda r: float(r["mean_test_mse"]))
        topk = next((r for r in rows if r["method_name"] == "topk_abs_cov"), None)
        topk_mse = None if topk is None else float(topk["mean_test_mse"])

        for rank, row in enumerate(rows, start=1):
            method = str(row["method_name"])
            s = by_method[method]
            s["scenarios"] += 1
            s["rank_sum"] += float(rank)
            if rank == 1:
                s["scenario_wins"] += 1
            if rank <= 3:
                s["top3_finishes"] += 1
            if topk_mse is not None and topk_mse > 0.0:
                rel_gain = (topk_mse - float(row["mean_test_mse"])) / topk_mse
                s["rel_gain_vs_topk_values"].append(float(rel_gain))
            value = row.get("win_rate_test_mse_vs_topk")
            if value is not None:
                s["win_rate_vs_topk_values"].append(float(value))

    scoreboard: list[dict] = []
    for method in methods:
        s = by_method[method]
        scenarios = int(s["scenarios"])
        if scenarios <= 0:
            continue
        rel_gains = s["rel_gain_vs_topk_values"]
        win_rates = s["win_rate_vs_topk_values"]
        scoreboard.append(
            {
                "method": method,
                "scenarios": scenarios,
                "scenario_wins": int(s["scenario_wins"]),
                "top3_finishes": int(s["top3_finishes"]),
                "mean_rank": float(s["rank_sum"]) / scenarios,
                "mean_rel_gain_vs_topk": (
                    None if not rel_gains else float(sum(rel_gains) / len(rel_gains))
                ),
                "mean_win_rate_vs_topk": (
                    None if not win_rates else float(sum(win_rates) / len(win_rates))
                ),
            }
        )
    return sorted(
        scoreboard,
        key=lambda r: (
            -int(r["scenario_wins"]),
            float(r["mean_rank"]),
            -int(r["top3_finishes"]),
        ),
    )


def _pairwise_dominance(grouped: dict[str, list[dict]]) -> tuple[list[str], dict[tuple[str, str], int]]:
    methods = sorted({str(r["method_name"]) for rows in grouped.values() for r in rows})
    matrix: dict[tuple[str, str], int] = {(a, b): 0 for a in methods for b in methods}
    for scenario_name in grouped:
        rows = grouped[scenario_name]
        mse_by_method = {str(r["method_name"]): float(r["mean_test_mse"]) for r in rows}
        for a in methods:
            mse_a = mse_by_method.get(a)
            if mse_a is None:
                continue
            for b in methods:
                if a == b:
                    continue
                mse_b = mse_by_method.get(b)
                if mse_b is None:
                    continue
                if mse_a < mse_b:
                    matrix[(a, b)] += 1
    return methods, matrix


def _fmt_float(value: float | None, ndigits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{ndigits}f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.1f}%"


def build_dominance_markdown(summary_payload: dict) -> str:
    rows = summary_payload["rows"]
    grouped = _group_by_scenario(rows)
    winners = _scenario_winner_rows(grouped)
    scoreboard = _method_scoreboard(grouped)
    methods, matrix = _pairwise_dominance(grouped)

    lines: list[str] = []
    lines.append("# Dominance Summary")
    lines.append("")
    lines.append("## Scenario Winners")
    lines.append("")
    lines.append(
        "| scenario | difficulty | winner | winner_test_mse | winner_support_f1 | winner_jaccard | winner_rel_gain_vs_topk |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in winners:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario_name"]),
                    str(row["difficulty"]),
                    str(row["winner"]),
                    _fmt_float(float(row["winner_test_mse"])),
                    _fmt_float(float(row["winner_support_f1"])),
                    _fmt_float(float(row["winner_jaccard"])),
                    _fmt_pct(row["winner_rel_gain_vs_topk"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Method Scoreboard")
    lines.append("")
    lines.append(
        "| method | scenarios | scenario_wins | top3_finishes | mean_rank | mean_rel_gain_vs_topk | mean_win_rate_vs_topk |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in scoreboard:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(int(row["scenarios"])),
                    str(int(row["scenario_wins"])),
                    str(int(row["top3_finishes"])),
                    _fmt_float(float(row["mean_rank"]), ndigits=3),
                    _fmt_pct(row["mean_rel_gain_vs_topk"]),
                    _fmt_pct(row["mean_win_rate_vs_topk"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Pairwise Dominance Counts")
    lines.append("")
    lines.append(
        "Cell `(A,B)` is the number of scenarios where method `A` has lower mean test MSE than method `B`."
    )
    lines.append("")
    header = "| A \\ B | " + " | ".join(methods) + " |"
    sep = "|---|" + "|".join(["---:"] * len(methods)) + "|"
    lines.append(header)
    lines.append(sep)
    for a in methods:
        vals = []
        for b in methods:
            if a == b:
                vals.append("-")
            else:
                vals.append(str(int(matrix[(a, b)])))
        lines.append("| " + a + " | " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build compact dominance markdown from stability summary JSON."
    )
    parser.add_argument(
        "--summary",
        default="benchmarks/results/stability_summary.json",
        help="Path to stability summary JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output markdown path; stdout if omitted.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    payload = _load_summary(summary_path)
    report = build_dominance_markdown(payload)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote dominance report: {output_path}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
