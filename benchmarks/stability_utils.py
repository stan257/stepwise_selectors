"""Shared stability-analysis helpers for benchmark outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt

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


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, mean
    std = float(np.std(arr, ddof=1))
    half_width = 1.96 * std / sqrt(float(arr.size))
    return mean - half_width, mean + half_width


def _win_rate_vs_reference(
    *,
    scenario_name: str,
    method_name: str,
    reference_method: str,
    grouped: dict[tuple[str, str], list[dict]],
    metric_name: str,
    tie_tolerance: float = 1e-12,
) -> tuple[float | None, int]:
    method_rows = grouped.get((scenario_name, method_name), [])
    ref_rows = grouped.get((scenario_name, reference_method), [])
    if not method_rows or not ref_rows:
        return None, 0

    method_by_seed = {
        int(row["dataset_seed"]): float(row["metrics"][metric_name])
        for row in method_rows
        if isinstance(row.get("metrics"), dict) and "dataset_seed" in row
    }
    ref_by_seed = {
        int(row["dataset_seed"]): float(row["metrics"][metric_name])
        for row in ref_rows
        if isinstance(row.get("metrics"), dict) and "dataset_seed" in row
    }

    shared = sorted(set(method_by_seed) & set(ref_by_seed))
    if not shared:
        return None, 0

    score = 0.0
    for seed in shared:
        left = method_by_seed[seed]
        right = ref_by_seed[seed]
        delta = left - right
        if delta < -tie_tolerance:
            score += 1.0
        elif abs(delta) <= tie_tolerance:
            score += 0.5
    return score / len(shared), len(shared)


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
        test_mse_ci95_low, test_mse_ci95_high = _mean_ci95(
            [float(m["test_mse"]) for m in metrics_list]
        )
        support_f1_ci95_low, support_f1_ci95_high = _mean_ci95(
            [float(m["support_f1"]) for m in metrics_list]
        )
        oracle_gap_test_values = [
            float(row["oracle_gap_test_mse"])
            for row in group_rows
            if row.get("oracle_gap_test_mse") is not None
        ]
        oracle_gap_train_values = [
            float(row["oracle_gap_train_rss"])
            for row in group_rows
            if row.get("oracle_gap_train_rss") is not None
        ]
        win_rate_vs_topk, paired_vs_topk = _win_rate_vs_reference(
            scenario_name=scenario_name,
            method_name=method_name,
            reference_method="topk_abs_cov",
            grouped=grouped,
            metric_name="test_mse",
        )
        win_rate_vs_forward, paired_vs_forward = _win_rate_vs_reference(
            scenario_name=scenario_name,
            method_name=method_name,
            reference_method="forward_bic",
            grouped=grouped,
            metric_name="test_mse",
        )

        scenario_info = group_rows[0]
        summary = {
            "scenario_name": scenario_name,
            "difficulty": str(scenario_info.get("difficulty", "unknown")),
            "scenario_description": str(scenario_info.get("scenario_description", "")),
            "scenario_checks": str(scenario_info.get("scenario_checks", "")),
            "scenario_why_hard": str(scenario_info.get("scenario_why_hard", "")),
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
            "test_mse_ci95_low": test_mse_ci95_low,
            "test_mse_ci95_high": test_mse_ci95_high,
            "support_f1_ci95_low": support_f1_ci95_low,
            "support_f1_ci95_high": support_f1_ci95_high,
            "support_size_mean": support_size_mean,
            "support_size_std": support_size_std,
            "elapsed_ms_mean": elapsed_ms_mean,
            "elapsed_ms_median": elapsed_ms_median,
            "mean_oracle_gap_test_mse": (
                None if not oracle_gap_test_values else _safe_mean(oracle_gap_test_values)
            ),
            "mean_oracle_gap_train_rss": (
                None if not oracle_gap_train_values else _safe_mean(oracle_gap_train_values)
            ),
            "oracle_runs": int(len(oracle_gap_test_values)),
            "win_rate_test_mse_vs_topk": win_rate_vs_topk,
            "paired_runs_vs_topk": int(paired_vs_topk),
            "win_rate_test_mse_vs_forward_bic": win_rate_vs_forward,
            "paired_runs_vs_forward_bic": int(paired_vs_forward),
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

    def _fmt_ci(row: dict, low_key: str, high_key: str) -> str:
        low = row.get(low_key)
        high = row.get(high_key)
        if low is None or high is None:
            return "-"
        return f"[{float(low):.6f}, {float(high):.6f}]"

    def _fmt_rate(value) -> str:
        if value is None:
            return "-"
        return f"{100.0 * float(value):.1f}%"

    lines: list[str] = []
    for scenario_name in sorted(grouped):
        scenario_rows = sorted(grouped[scenario_name], key=lambda r: float(r["mean_test_mse"]))
        difficulty = str(scenario_rows[0].get("difficulty", "unknown"))
        scenario_description = str(scenario_rows[0].get("scenario_description", "")).strip()
        scenario_checks = str(scenario_rows[0].get("scenario_checks", "")).strip()
        scenario_why_hard = str(scenario_rows[0].get("scenario_why_hard", "")).strip()
        lines.append(f"## {scenario_name} ({difficulty})")
        if scenario_description:
            lines.append(f"- Description: {scenario_description}")
        if scenario_checks:
            lines.append(f"- Checks: {scenario_checks}")
        if scenario_why_hard:
            lines.append(f"- Why hard: {scenario_why_hard}")
        lines.append("")
        lines.append(
            "| method | mean_test_mse | test_mse_95ci | mean_support_f1 | support_f1_95ci | exact_support_rate | mean_pairwise_jaccard | win_rate_vs_forward_bic | win_rate_vs_topk | mean_oracle_gap_test_mse | delta_test_mse_vs_topk |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in scenario_rows:
            delta_mse = row.get("delta_test_mse_vs_topk")
            delta_cell = "-" if delta_mse is None else f"{float(delta_mse):.6f}"
            oracle_gap = row.get("mean_oracle_gap_test_mse")
            oracle_cell = "-" if oracle_gap is None else f"{float(oracle_gap):.6f}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["method_name"]),
                        f"{float(row['mean_test_mse']):.6f}",
                        _fmt_ci(row, "test_mse_ci95_low", "test_mse_ci95_high"),
                        f"{float(row['mean_support_f1']):.6f}",
                        _fmt_ci(row, "support_f1_ci95_low", "support_f1_ci95_high"),
                        f"{float(row['exact_support_rate']):.6f}",
                        f"{float(row['mean_pairwise_jaccard']):.6f}",
                        _fmt_rate(row.get("win_rate_test_mse_vs_forward_bic")),
                        _fmt_rate(row.get("win_rate_test_mse_vs_topk")),
                        oracle_cell,
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
