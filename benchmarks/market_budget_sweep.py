"""Budget/seed sweeps for market chunk forward-selection experiments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from selection import (
    BeamForwardSelection,
    CrossValForwardSelection,
    CrossValGramData,
    ForwardSelection,
    GramData,
)
from selection.criteria import AICCriterion, BICCriterion

from .market_chunks import (
    MarketChunkConfig,
    generate_market_gram_chunks_known,
    generate_market_gram_chunks_unknown,
)


def _combine_gram_chunks(chunks: list[GramData]) -> GramData:
    if not chunks:
        raise ValueError("chunks must contain at least one GramData object.")
    p = chunks[0].gram.shape[0]
    gram = np.zeros((p, p), dtype=float)
    cov = np.zeros(p, dtype=float)
    y_norm = 0.0
    n_samples = 0
    for chunk in chunks:
        if chunk.gram.shape != (p, p):
            raise ValueError("All chunks must share the same Gram dimensions.")
        if chunk.cov.shape != (p,):
            raise ValueError("All chunks must share the same covariance dimensions.")
        gram += chunk.gram
        cov += chunk.cov
        y_norm += float(chunk.y_norm)
        n_samples += int(chunk.n_samples)
    return GramData(
        gram=np.ascontiguousarray(gram),
        cov=np.ascontiguousarray(cov),
        y_norm=float(y_norm),
        n_samples=int(n_samples),
        warn_if_uncentered=False,
    )


def _rss_for_beta(data: GramData, beta: np.ndarray) -> float:
    beta_arr = np.asarray(beta, dtype=float)
    if beta_arr.shape != data.cov.shape:
        raise ValueError(
            f"beta has shape {beta_arr.shape}; expected {data.cov.shape}."
        )
    rss = float(
        data.y_norm
        - 2.0 * float(beta_arr @ data.cov)
        + float(beta_arr @ (data.gram @ beta_arr))
    )
    return float(max(rss, 0.0))


def _safe_ic(criterion_cls, *, rss: float, k: int, n_samples: int) -> float | None:
    try:
        criterion = criterion_cls(n_samples=int(n_samples))
        value = criterion.evaluate(rss, k)
        return float(np.asarray(value))
    except Exception:
        return None


def _build_row(
    *,
    seed: int,
    max_steps: int,
    method_name: str,
    state,
    elapsed_ms: float,
    train_data: GramData,
    holdout_data: GramData,
    cv_data: CrossValGramData | None,
) -> dict[str, object]:
    beta = np.asarray(state.beta, dtype=float)
    train_rss = _rss_for_beta(train_data, beta)
    holdout_rss = _rss_for_beta(holdout_data, beta)
    train_mse = train_rss / float(train_data.n_samples)
    holdout_mse = holdout_rss / float(holdout_data.n_samples)
    holdout_null_mse = float(holdout_data.y_norm) / float(holdout_data.n_samples)
    holdout_r2 = (
        1.0 - holdout_rss / float(holdout_data.y_norm)
        if holdout_data.y_norm > 0.0
        else 0.0
    )
    row: dict[str, object] = {
        "seed": int(seed),
        "max_steps": int(max_steps),
        "method": method_name,
        "elapsed_ms": float(elapsed_ms),
        "n_selected": int(len(state.active_set)),
        "active_set": [int(i) for i in state.active_set],
        "train_combined_rss": float(train_rss),
        "train_combined_mse": float(train_mse),
        "train_combined_aic": _safe_ic(
            AICCriterion,
            rss=train_rss,
            k=len(state.active_set),
            n_samples=train_data.n_samples,
        ),
        "train_combined_bic": _safe_ic(
            BICCriterion,
            rss=train_rss,
            k=len(state.active_set),
            n_samples=train_data.n_samples,
        ),
        "oos_holdout_rss": float(holdout_rss),
        "oos_holdout_mse": float(holdout_mse),
        "oos_holdout_null_mse": float(holdout_null_mse),
        "oos_holdout_r2": float(holdout_r2),
        "oos_mse_vs_null_ratio": (
            float(holdout_mse / holdout_null_mse)
            if holdout_null_mse > 0.0
            else None
        ),
        "state_rss": (
            float(getattr(state, "rss"))
            if getattr(state, "rss", None) is not None
            else None
        ),
        "state_rss_cv": (
            float(getattr(state, "rss_cv"))
            if getattr(state, "rss_cv", None) is not None
            else None
        ),
    }
    if cv_data is not None and hasattr(state, "oos_rss_folds"):
        fold_sizes = np.asarray(cv_data.fold_sizes, dtype=float)
        fold_rss = np.asarray(state.oos_rss_folds, dtype=float)
        fold_mse = fold_rss / fold_sizes
        row["cv_fold_mean_mse"] = float(np.mean(fold_mse))
        row["cv_fold_median_mse"] = float(np.median(fold_mse))
    else:
        row["cv_fold_mean_mse"] = None
        row["cv_fold_median_mse"] = None
    return row


def _method_rows_for_budget(
    *,
    seed: int,
    max_steps: int,
    train_data: GramData,
    holdout_data: GramData,
    cv_data: CrossValGramData,
    beam_width: int,
    cv_aggregation: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods: list[tuple[str, object, GramData | CrossValGramData, CrossValGramData | None]] = [
        ("forward_aic_in_sample", ForwardSelection(criterion="aic"), train_data, None),
        (
            "cv_forward_oos_mean_mse",
            CrossValForwardSelection(cv_aggregation=cv_aggregation),
            cv_data,
            cv_data,
        ),
        (
            f"beam_forward_in_sample_rss_w{beam_width}",
            BeamForwardSelection(beam_width=beam_width, criterion="rss"),
            train_data,
            None,
        ),
    ]

    for method_name, selector, fit_data, maybe_cv in methods:
        start = time.perf_counter()
        state = selector.fit(data=fit_data, max_steps=max_steps)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        rows.append(
            _build_row(
                seed=seed,
                max_steps=max_steps,
                method_name=method_name,
                state=state,
                elapsed_ms=elapsed_ms,
                train_data=train_data,
                holdout_data=holdout_data,
                cv_data=maybe_cv,
            )
        )

    ranked = sorted(rows, key=lambda row: float(row["oos_holdout_mse"]))
    for rank_idx, row in enumerate(ranked, start=1):
        row["rank_oos_holdout_mse"] = int(rank_idx)
    return ranked


def _aggregate_method_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    holdout = np.asarray([float(row["oos_holdout_mse"]) for row in rows], dtype=float)
    ranks = np.asarray([float(row["rank_oos_holdout_mse"]) for row in rows], dtype=float)
    n_selected = np.asarray([float(row["n_selected"]) for row in rows], dtype=float)
    train = np.asarray([float(row["train_combined_mse"]) for row in rows], dtype=float)
    holdout_r2 = np.asarray([float(row["oos_holdout_r2"]) for row in rows], dtype=float)
    elapsed = np.asarray([float(row["elapsed_ms"]) for row in rows], dtype=float)
    cv_vals = np.asarray(
        [
            float(row["cv_fold_mean_mse"])
            for row in rows
            if row["cv_fold_mean_mse"] is not None
        ],
        dtype=float,
    )
    return {
        "n_runs": int(len(rows)),
        "wins": int(sum(int(row["rank_oos_holdout_mse"]) == 1 for row in rows)),
        "win_rate": float(
            sum(int(row["rank_oos_holdout_mse"]) == 1 for row in rows) / len(rows)
        ),
        "mean_oos_holdout_mse": float(np.mean(holdout)),
        "median_oos_holdout_mse": float(np.median(holdout)),
        "std_oos_holdout_mse": float(np.std(holdout)),
        "mean_rank_oos_holdout_mse": float(np.mean(ranks)),
        "median_rank_oos_holdout_mse": float(np.median(ranks)),
        "mean_n_selected": float(np.mean(n_selected)),
        "median_n_selected": float(np.median(n_selected)),
        "mean_train_combined_mse": float(np.mean(train)),
        "mean_oos_holdout_r2": float(np.mean(holdout_r2)),
        "mean_elapsed_ms": float(np.mean(elapsed)),
        "mean_cv_fold_mean_mse": (
            float(np.mean(cv_vals)) if cv_vals.size > 0 else None
        ),
    }


def _pairwise_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    method_names = sorted({str(row["method"]) for row in rows})
    grouped: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["max_steps"]))
        grouped.setdefault(key, {})[str(row["method"])] = row

    pairs: dict[str, dict[str, object]] = {}
    for left_idx, left in enumerate(method_names):
        for right in method_names[left_idx + 1 :]:
            deltas = np.asarray(
                [
                    float(method_rows[left]["oos_holdout_mse"])
                    - float(method_rows[right]["oos_holdout_mse"])
                    for method_rows in grouped.values()
                    if left in method_rows and right in method_rows
                ],
                dtype=float,
            )
            if deltas.size == 0:
                continue
            key = f"{left}__vs__{right}"
            pairs[key] = {
                "left_method": left,
                "right_method": right,
                "mean_delta_left_minus_right": float(np.mean(deltas)),
                "median_delta_left_minus_right": float(np.median(deltas)),
                "left_wins": int(np.sum(deltas < 0.0)),
                "ties": int(np.sum(deltas == 0.0)),
                "right_wins": int(np.sum(deltas > 0.0)),
            }
    return pairs


def summarize_market_budget_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    by_budget: dict[str, object] = {}
    budgets = sorted({int(row["max_steps"]) for row in rows})
    method_names = sorted({str(row["method"]) for row in rows})

    overall_methods = {
        method: _aggregate_method_rows(
            [row for row in rows if str(row["method"]) == method]
        )
        for method in method_names
    }

    for budget in budgets:
        budget_rows = [row for row in rows if int(row["max_steps"]) == budget]
        by_budget[str(budget)] = {
            "methods": {
                method: _aggregate_method_rows(
                    [row for row in budget_rows if str(row["method"]) == method]
                )
                for method in method_names
            },
            "pairwise": _pairwise_summary(budget_rows),
        }

    best_budget_per_method = {}
    for method in method_names:
        candidates = [
            (budget, by_budget[str(budget)]["methods"][method]["mean_oos_holdout_mse"])
            for budget in budgets
        ]
        best_budget, best_score = min(candidates, key=lambda item: float(item[1]))
        best_budget_per_method[method] = {
            "best_budget": int(best_budget),
            "mean_oos_holdout_mse": float(best_score),
        }

    return {
        "overall": {
            "methods": overall_methods,
            "pairwise": _pairwise_summary(rows),
        },
        "by_budget": by_budget,
        "best_budget_per_method": best_budget_per_method,
    }


def render_market_budget_report(
    *,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    config: dict[str, object],
) -> str:
    budgets = [int(v) for v in config["budgets"]]
    seeds = [int(v) for v in config["seeds"]]
    method_names = sorted(summary["overall"]["methods"].keys())
    lines: list[str] = []
    lines.append("# Market Budget Sweep Report")
    lines.append("")
    lines.append(
        f"Seeds: `{seeds[0]}`..`{seeds[-1]}` ({len(seeds)} total), budgets: "
        + ", ".join(f"`{budget}`" for budget in budgets)
    )
    lines.append(
        f"Train chunks: `0..{int(config['train_chunks']) - 1}`, holdout chunk: "
        f"`{config['holdout_chunk']}`"
    )
    lines.append("")
    lines.append("## Overall Summary")
    lines.append("")
    lines.append(
        "| method | win_rate | mean_holdout_mse | mean_rank | mean_n_selected | best_budget |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for method in method_names:
        overall = summary["overall"]["methods"][method]
        best_budget = summary["best_budget_per_method"][method]["best_budget"]
        lines.append(
            "| "
            + method
            + f" | {overall['win_rate']:.3f}"
            + f" | {overall['mean_oos_holdout_mse']:.6f}"
            + f" | {overall['mean_rank_oos_holdout_mse']:.3f}"
            + f" | {overall['mean_n_selected']:.2f}"
            + f" | {best_budget} |"
        )
    lines.append("")
    lines.append("## By Budget")
    lines.append("")
    lines.append(
        "| budget | best_method | best_mean_holdout_mse | second_best_gap |"
    )
    lines.append("| ---: | --- | ---: | ---: |")
    for budget in budgets:
        methods = summary["by_budget"][str(budget)]["methods"]
        ordered = sorted(
            (
                (method, float(methods[method]["mean_oos_holdout_mse"]))
                for method in method_names
            ),
            key=lambda item: item[1],
        )
        best_method, best_score = ordered[0]
        gap = ordered[1][1] - best_score if len(ordered) > 1 else 0.0
        lines.append(
            f"| {budget} | {best_method} | {best_score:.6f} | {gap:.6f} |"
        )
    lines.append("")
    lines.append("## Overall Pairwise")
    lines.append("")
    lines.append("| pair | mean_delta_left_minus_right | left_wins | right_wins |")
    lines.append("| --- | ---: | ---: | ---: |")
    for pair_name, pair in summary["overall"]["pairwise"].items():
        lines.append(
            f"| {pair_name} | {pair['mean_delta_left_minus_right']:.6f} "
            f"| {pair['left_wins']} | {pair['right_wins']} |"
        )
    lines.append("")
    lines.append(f"Flat rows captured: `{len(rows)}`")
    return "\n".join(lines) + "\n"


def run_market_budget_sweep(
    *,
    base_config: MarketChunkConfig,
    budgets: list[int],
    seeds: list[int],
    train_chunks: int = 10,
    holdout_chunk: int = -1,
    beam_width: int = 5,
    cv_aggregation: str = "mean_mse",
    flavor: str = "unknown",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if flavor not in {"known", "unknown"}:
        raise ValueError("flavor must be 'known' or 'unknown'.")
    if not budgets:
        raise ValueError("budgets must be non-empty.")
    if not seeds:
        raise ValueError("seeds must be non-empty.")

    rows: list[dict[str, object]] = []
    for seed in seeds:
        config = replace(base_config, seed=int(seed))
        if flavor == "known":
            dataset = generate_market_gram_chunks_known(config)
        else:
            dataset = generate_market_gram_chunks_unknown(config)

        chunks = list(dataset.gram_chunks)
        n_chunks_total = len(chunks)
        holdout_idx = holdout_chunk if holdout_chunk >= 0 else (n_chunks_total - 1)
        if train_chunks <= 1:
            raise ValueError("train_chunks must be >= 2 for CV comparison.")
        if train_chunks >= n_chunks_total:
            raise ValueError("train_chunks must be less than total chunk count.")
        if holdout_idx < 0 or holdout_idx >= n_chunks_total:
            raise ValueError("holdout_chunk is out of range.")
        if holdout_idx in range(train_chunks):
            raise ValueError("holdout_chunk must not overlap with training chunks.")

        train_data = _combine_gram_chunks(chunks[:train_chunks])
        holdout_data = chunks[holdout_idx]
        cv_data = CrossValGramData(chunks[:train_chunks])

        for max_steps in budgets:
            rows.extend(
                _method_rows_for_budget(
                    seed=int(seed),
                    max_steps=int(max_steps),
                    train_data=train_data,
                    holdout_data=holdout_data,
                    cv_data=cv_data,
                    beam_width=beam_width,
                    cv_aggregation=cv_aggregation,
                )
            )

    summary = summarize_market_budget_rows(rows)
    return rows, summary


def _parse_csv_ints(value: str) -> list[int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise ValueError("Expected at least one integer value.")
    return [int(item) for item in parts]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a budget sweep over seeds for three market forward-selection "
            "methods: AIC in-sample, CV forward, and beam forward RSS."
        )
    )
    parser.add_argument("--seed-start", type=int, default=20260307)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument(
        "--budgets",
        type=str,
        default="2,4,8,16,32",
        help="Comma-separated max_steps values shared across all methods.",
    )
    parser.add_argument(
        "--flavor",
        choices=("known", "unknown"),
        default="unknown",
    )
    parser.add_argument("--n-chunks", type=int, default=11)
    parser.add_argument("--train-chunks", type=int, default=10)
    parser.add_argument("--holdout-chunk", type=int, default=-1)
    parser.add_argument("--bars-per-chunk", type=int, default=2500)
    parser.add_argument("--warmup-bars", type=int, default=250)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--n-regimes", type=int, default=5)
    parser.add_argument("--support-size", type=int, default=8)
    parser.add_argument("--signal-scale", type=float, default=0.22)
    parser.add_argument("--target-noise-std", type=float, default=0.01)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument(
        "--cv-aggregation",
        choices=("mean_mse", "median_mse", "sum_rss"),
        default="mean_mse",
    )
    parser.add_argument(
        "--rows-output",
        type=str,
        default="benchmarks/results/market_budget_sweep_rows.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="benchmarks/results/market_budget_sweep_summary.json",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="benchmarks/results/market_budget_sweep_report.md",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    budgets = _parse_csv_ints(args.budgets)
    seeds = [int(args.seed_start) + offset for offset in range(int(args.num_seeds))]
    base_config = MarketChunkConfig(
        seed=int(args.seed_start),
        n_chunks=int(args.n_chunks),
        bars_per_chunk=int(args.bars_per_chunk),
        warmup_bars=int(args.warmup_bars),
        n_features=int(args.n_features),
        n_regimes=int(args.n_regimes),
        support_size=int(args.support_size),
        signal_scale=float(args.signal_scale),
        target_noise_std=float(args.target_noise_std),
    )

    rows, summary = run_market_budget_sweep(
        base_config=base_config,
        budgets=budgets,
        seeds=seeds,
        train_chunks=int(args.train_chunks),
        holdout_chunk=int(args.holdout_chunk),
        beam_width=int(args.beam_width),
        cv_aggregation=args.cv_aggregation,
        flavor=args.flavor,
    )

    config = {
        "seed_start": int(args.seed_start),
        "num_seeds": int(args.num_seeds),
        "seeds": seeds,
        "budgets": budgets,
        "flavor": args.flavor,
        "train_chunks": int(args.train_chunks),
        "holdout_chunk": (
            int(args.holdout_chunk)
            if int(args.holdout_chunk) >= 0
            else int(args.n_chunks) - 1
        ),
        "beam_width": int(args.beam_width),
        "cv_aggregation": args.cv_aggregation,
        "dataset_config": {
            "n_chunks": int(args.n_chunks),
            "bars_per_chunk": int(args.bars_per_chunk),
            "warmup_bars": int(args.warmup_bars),
            "n_features": int(args.n_features),
            "n_regimes": int(args.n_regimes),
            "support_size": int(args.support_size),
            "signal_scale": float(args.signal_scale),
            "target_noise_std": float(args.target_noise_std),
        },
    }

    rows_output = Path(args.rows_output)
    summary_output = Path(args.summary_output)
    report_output = Path(args.report_output)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    with rows_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary_payload = {"config": config, "summary": summary}
    summary_output.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    report = render_market_budget_report(rows=rows, summary=summary, config=config)
    report_output.write_text(report, encoding="utf-8")

    overall_methods = summary["overall"]["methods"]
    ordered = sorted(
        (
            (name, float(payload["mean_oos_holdout_mse"]))
            for name, payload in overall_methods.items()
        ),
        key=lambda item: item[1],
    )
    print("Overall mean holdout MSE:")
    for method, score in ordered:
        print(f"  {method}: {score:.6f}")
    print(f"Wrote rows: {rows_output}")
    print(f"Wrote summary: {summary_output}")
    print(f"Wrote report: {report_output}")


if __name__ == "__main__":
    main()
