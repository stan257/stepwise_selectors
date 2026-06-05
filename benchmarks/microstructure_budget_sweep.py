"""Budget/seed sweeps for exact-k microstructure forward-selection experiments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from selection import (
    BeamForwardSelection,
    CrossValForwardSelection,
    CrossValGramData,
    ForwardSelection,
)

from .forward_exact_k import (
    fit_beam_forward_exact_k,
    fit_cv_forward_exact_k,
    fit_forward_exact_k,
)
from .market_budget_sweep import (
    _build_row,
    _combine_gram_chunks,
    _parse_csv_ints,
    summarize_market_budget_rows,
)
from .microstructure_chunks import (
    MicrostructureChunkConfig,
    apply_microstructure_preset,
    generate_microstructure_gram_chunks_known,
    generate_microstructure_gram_chunks_unknown,
    microstructure_preset_names,
)


def _method_rows_for_budget(
    *,
    seed: int,
    max_steps: int,
    train_data,
    holdout_data,
    cv_data: CrossValGramData,
    beam_width: int,
    cv_aggregation: str,
    selection_mode: str,
    solver_policy: str,
    ridge_alpha: float,
    pinv_rcond: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = [
        (
            "forward_aic_in_sample",
            ForwardSelection(
                criterion="aic",
                solver_policy=solver_policy,
                ridge_alpha=ridge_alpha,
                pinv_rcond=pinv_rcond,
            ),
            train_data,
            None,
        ),
        (
            "cv_forward_oos_mean_mse",
            CrossValForwardSelection(
                cv_aggregation=cv_aggregation,
                solver_policy=solver_policy,
                ridge_alpha=ridge_alpha,
                pinv_rcond=pinv_rcond,
            ),
            cv_data,
            cv_data,
        ),
        (
            f"beam_forward_in_sample_rss_w{beam_width}",
            BeamForwardSelection(
                beam_width=beam_width,
                criterion="rss",
                solver_policy=solver_policy,
                ridge_alpha=ridge_alpha,
                pinv_rcond=pinv_rcond,
            ),
            train_data,
            None,
        ),
    ]

    for method_name, selector, fit_data, maybe_cv in methods:
        start = time.perf_counter()
        if selection_mode == "exact_k":
            if method_name == "forward_aic_in_sample":
                state = fit_forward_exact_k(selector, data=fit_data, max_steps=max_steps)
            elif method_name == "cv_forward_oos_mean_mse":
                state = fit_cv_forward_exact_k(selector, data=fit_data, max_steps=max_steps)
            else:
                state = fit_beam_forward_exact_k(selector, data=fit_data, max_steps=max_steps)
        elif selection_mode == "native":
            state = selector.fit(data=fit_data, max_steps=max_steps)
        else:
            raise ValueError("selection_mode must be 'exact_k' or 'native'.")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        row = _build_row(
            seed=seed,
            max_steps=max_steps,
            method_name=method_name,
            state=state,
            elapsed_ms=elapsed_ms,
            train_data=train_data,
            holdout_data=holdout_data,
            cv_data=maybe_cv,
        )
        row["selection_mode"] = selection_mode
        row["solver_policy"] = solver_policy
        rows.append(row)

    ranked = sorted(rows, key=lambda row: float(row["oos_holdout_mse"]))
    for rank_idx, row in enumerate(ranked, start=1):
        row["rank_oos_holdout_mse"] = int(rank_idx)
    return ranked


def run_microstructure_budget_sweep(
    *,
    base_config: MicrostructureChunkConfig,
    budgets: list[int],
    seeds: list[int],
    train_chunks: int = 10,
    holdout_chunk: int = -1,
    beam_width: int = 5,
    cv_aggregation: str = "mean_mse",
    flavor: str = "unknown",
    preset: str = "default",
    selection_mode: str = "exact_k",
    solver_policy: str = "pinv",
    ridge_alpha: float = 1e-8,
    pinv_rcond: float = 1e-12,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if flavor not in {"known", "unknown"}:
        raise ValueError("flavor must be 'known' or 'unknown'.")
    if selection_mode not in {"exact_k", "native"}:
        raise ValueError("selection_mode must be 'exact_k' or 'native'.")
    if not budgets:
        raise ValueError("budgets must be non-empty.")
    if not seeds:
        raise ValueError("seeds must be non-empty.")

    rows: list[dict[str, object]] = []
    for seed in seeds:
        config = apply_microstructure_preset(
            replace(base_config, seed=int(seed)),
            preset=preset,
        )
        if flavor == "known":
            dataset = generate_microstructure_gram_chunks_known(config)
        else:
            dataset = generate_microstructure_gram_chunks_unknown(config)

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
                    selection_mode=selection_mode,
                    solver_policy=solver_policy,
                    ridge_alpha=ridge_alpha,
                    pinv_rcond=pinv_rcond,
                )
            )

    summary = summarize_market_budget_rows(rows)
    return rows, summary


def render_microstructure_budget_report(
    *,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    config: dict[str, object],
) -> str:
    budgets = [int(v) for v in config["budgets"]]
    seeds = [int(v) for v in config["seeds"]]
    method_names = sorted(summary["overall"]["methods"].keys())
    lines: list[str] = []
    title = (
        "# Microstructure Exact-K Budget Sweep Report"
        if str(config["selection_mode"]) == "exact_k"
        else "# Microstructure Budget Sweep Report"
    )
    lines.append(title)
    lines.append("")
    lines.append(
        f"Seeds: `{seeds[0]}`..`{seeds[-1]}` ({len(seeds)} total), budgets: "
        + ", ".join(f"`{budget}`" for budget in budgets)
    )
    lines.append(
        f"Train chunks: `0..{int(config['train_chunks']) - 1}`, holdout chunk: "
        f"`{config['holdout_chunk']}`"
    )
    lines.append(
        f"Selection mode: `{config['selection_mode']}`, flavor: `{config['flavor']}`, "
        f"preset: `{config['preset']}`, solver: `{config['solver_policy']}`"
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
    lines.append("| budget | best_method | best_mean_holdout_mse | second_best_gap |")
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a budget sweep over seeds for three microstructure forward-selection "
            "methods under either native stopping or exact-k matched support sizes."
        )
    )
    parser.add_argument("--seed-start", type=int, default=20260307)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument(
        "--budgets",
        type=str,
        default="2,4,8,16,32",
        help="Comma-separated max_steps / target-k values shared across all methods.",
    )
    parser.add_argument("--train-chunks", type=int, default=10)
    parser.add_argument("--holdout-chunk", type=int, default=-1)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--cv-aggregation", type=str, default="mean_mse")
    parser.add_argument("--flavor", type=str, default="unknown", choices=["known", "unknown"])
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=microstructure_preset_names(),
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="exact_k",
        choices=["exact_k", "native"],
    )
    parser.add_argument(
        "--solver-policy",
        type=str,
        default="pinv",
        choices=["strict", "ridge", "pinv"],
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-8)
    parser.add_argument("--pinv-rcond", type=float, default=1e-12)
    parser.add_argument("--n-chunks", type=int, default=11)
    parser.add_argument("--events-per-chunk", type=int, default=5000)
    parser.add_argument("--warmup-events", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--n-regimes", type=int, default=5)
    parser.add_argument("--target-horizon-events", type=int, default=8)
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("benchmarks/results/microstructure_budget_sweep_exact_k_rows.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmarks/results/microstructure_budget_sweep_exact_k_summary.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("benchmarks/results/microstructure_budget_sweep_exact_k_report.md"),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    budgets = _parse_csv_ints(args.budgets)
    seeds = [args.seed_start + offset for offset in range(args.num_seeds)]
    rows, summary = run_microstructure_budget_sweep(
        base_config=MicrostructureChunkConfig(
            seed=args.seed_start,
            n_chunks=args.n_chunks,
            events_per_chunk=args.events_per_chunk,
            warmup_events=args.warmup_events,
            n_features=args.n_features,
            n_regimes=args.n_regimes,
            target_horizon_events=args.target_horizon_events,
        ),
        budgets=budgets,
        seeds=seeds,
        train_chunks=args.train_chunks,
        holdout_chunk=args.holdout_chunk,
        beam_width=args.beam_width,
        cv_aggregation=args.cv_aggregation,
        flavor=args.flavor,
        preset=args.preset,
        selection_mode=args.selection_mode,
        solver_policy=args.solver_policy,
        ridge_alpha=args.ridge_alpha,
        pinv_rcond=args.pinv_rcond,
    )

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with args.summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report = render_microstructure_budget_report(
        rows=rows,
        summary=summary,
        config={
            "budgets": budgets,
            "seeds": seeds,
            "train_chunks": args.train_chunks,
            "holdout_chunk": args.holdout_chunk
            if args.holdout_chunk >= 0
            else (args.n_chunks - 1),
            "selection_mode": args.selection_mode,
            "solver_policy": args.solver_policy,
            "flavor": args.flavor,
            "preset": args.preset,
        },
    )
    with args.report_output.open("w", encoding="utf-8") as handle:
        handle.write(report)

    print(report)


__all__ = [
    "render_microstructure_budget_report",
    "run_microstructure_budget_sweep",
]


if __name__ == "__main__":
    main()
