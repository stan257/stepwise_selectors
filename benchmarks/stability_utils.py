"""Shared stability-analysis helpers for benchmark outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True)
class PairwiseJaccard:
    mean: float
    minimum: float
    maximum: float


def support_jaccard(lhs: list[int], rhs: list[int]) -> float:
    """Compute Jaccard overlap between two supports."""
    lhs_set = set(int(i) for i in lhs)
    rhs_set = set(int(i) for i in rhs)
    union = lhs_set | rhs_set
    if not union:
        return 1.0
    return len(lhs_set & rhs_set) / len(union)


def pairwise_support_jaccard(supports: list[list[int]]) -> PairwiseJaccard:
    """Summarize pairwise support Jaccard scores across runs."""
    if len(supports) < 2:
        return PairwiseJaccard(mean=1.0, minimum=1.0, maximum=1.0)
    scores: list[float] = []
    for i in range(len(supports)):
        for j in range(i + 1, len(supports)):
            scores.append(support_jaccard(supports[i], supports[j]))
    arr = np.asarray(scores, dtype=float)
    return PairwiseJaccard(
        mean=float(np.mean(arr)),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
    )


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_std(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def _safe_median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=float)))


def summarize_stability_rows(rows: list[dict]) -> dict:
    """Aggregate per-run benchmark rows into stability summary metrics."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row.get("scenario_name", "")), str(row.get("method_name", "")))
        grouped[key].append(row)

    summary_rows: list[dict] = []
    scenario_baselines: dict[str, dict] = {}

    for (scenario_name, method_name), group_rows in sorted(grouped.items()):
        metrics_list = [row["metrics"] for row in group_rows if isinstance(row.get("metrics"), dict)]
        supports = [list(row.get("active_set", [])) for row in group_rows]
        truths = [list(row.get("true_support", [])) for row in group_rows]
        exact_match = [int(set(s) == set(t)) for s, t in zip(supports, truths)]

        jaccard_summary = pairwise_support_jaccard(supports)

        mean_test_mse = _safe_mean([float(m["test_mse"]) for m in metrics_list])
        mean_val_mse = _safe_mean([float(m["val_mse"]) for m in metrics_list])
        mean_support_f1 = _safe_mean([float(m["support_f1"]) for m in metrics_list])
        mean_precision = _safe_mean([float(m["support_precision"]) for m in metrics_list])
        mean_recall = _safe_mean([float(m["support_recall"]) for m in metrics_list])
        support_size_mean = _safe_mean([float(m["n_selected"]) for m in metrics_list])
        support_size_std = _safe_std([float(m["n_selected"]) for m in metrics_list])
        elapsed_ms_mean = _safe_mean([float(m["elapsed_ms"]) for m in metrics_list])
        elapsed_ms_median = _safe_median([float(m["elapsed_ms"]) for m in metrics_list])

        scenario_info = group_rows[0]
        summary = {
            "scenario_name": scenario_name,
            "difficulty": str(scenario_info.get("difficulty", "unknown")),
            "method_name": method_name,
            "selector": str(scenario_info.get("selector", "unknown")),
            "runs": int(len(group_rows)),
            "mean_test_mse": mean_test_mse,
            "mean_val_mse": mean_val_mse,
            "mean_support_f1": mean_support_f1,
            "mean_support_precision": mean_precision,
            "mean_support_recall": mean_recall,
            "exact_support_rate": _safe_mean([float(x) for x in exact_match]),
            "mean_pairwise_jaccard": jaccard_summary.mean,
            "min_pairwise_jaccard": jaccard_summary.minimum,
            "max_pairwise_jaccard": jaccard_summary.maximum,
            "support_size_mean": support_size_mean,
            "support_size_std": support_size_std,
            "elapsed_ms_mean": elapsed_ms_mean,
            "elapsed_ms_median": elapsed_ms_median,
        }
        summary_rows.append(summary)

        if method_name == "topk_abs_cov":
            scenario_baselines[scenario_name] = summary

    for summary in summary_rows:
        baseline = scenario_baselines.get(summary["scenario_name"])
        if baseline is None:
            summary["delta_test_mse_vs_topk"] = None
            summary["delta_support_f1_vs_topk"] = None
            continue
        summary["delta_test_mse_vs_topk"] = float(
            summary["mean_test_mse"] - baseline["mean_test_mse"]
        )
        summary["delta_support_f1_vs_topk"] = float(
            summary["mean_support_f1"] - baseline["mean_support_f1"]
        )

    return {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": summary_rows,
    }


def render_stability_markdown(summary_payload: dict) -> str:
    """Render summary payload to a markdown table grouped by scenario."""
    rows = summary_payload.get("rows", [])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_name"])].append(row)

    lines: list[str] = []
    for scenario_name in sorted(grouped):
        scenario_rows = sorted(grouped[scenario_name], key=lambda r: float(r["mean_test_mse"]))
        difficulty = str(scenario_rows[0].get("difficulty", "unknown"))
        lines.append(f"## {scenario_name} ({difficulty})")
        lines.append("")
        lines.append(
            "| method | mean_test_mse | mean_support_f1 | exact_support_rate | mean_pairwise_jaccard | support_size_mean | support_size_std | delta_test_mse_vs_topk |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in scenario_rows:
            delta_mse = row.get("delta_test_mse_vs_topk")
            delta_cell = "-" if delta_mse is None else f"{float(delta_mse):.6f}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["method_name"]),
                        f"{float(row['mean_test_mse']):.6f}",
                        f"{float(row['mean_support_f1']):.6f}",
                        f"{float(row['exact_support_rate']):.6f}",
                        f"{float(row['mean_pairwise_jaccard']):.6f}",
                        f"{float(row['support_size_mean']):.3f}",
                        f"{float(row['support_size_std']):.3f}",
                        delta_cell,
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


__all__ = [
    "PairwiseJaccard",
    "support_jaccard",
    "pairwise_support_jaccard",
    "summarize_stability_rows",
    "render_stability_markdown",
]
