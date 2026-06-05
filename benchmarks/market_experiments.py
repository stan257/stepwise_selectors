"""Compare forward-selection strategies on chunked market Gram datasets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from selection import (
    BeamCrossValForwardSelection,
    CrossValForwardSelection,
    CrossValGramData,
    ForwardSelection,
    GramData,
)
from selection.criteria import AICCriterion, BICCriterion

from .market_chunks import load_market_chunk_dataset


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
        data.y_norm - 2.0 * float(beta_arr @ data.cov) + float(beta_arr @ (data.gram @ beta_arr))
    )
    return float(max(rss, 0.0))


def _safe_ic(criterion_cls, *, rss: float, k: int, n_samples: int) -> float | None:
    try:
        criterion = criterion_cls(n_samples=int(n_samples))
        value = criterion.evaluate(rss, k)
        return float(np.asarray(value))
    except Exception:
        return None


def _build_result_row(
    *,
    method_name: str,
    state,
    elapsed_ms: float,
    train_data: GramData,
    holdout_data: GramData,
    train_chunks: list[GramData],
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
    train_chunk_mse = [
        _rss_for_beta(chunk, beta) / float(chunk.n_samples) for chunk in train_chunks
    ]
    k = len(state.active_set)
    row: dict[str, object] = {
        "method": method_name,
        "elapsed_ms": float(elapsed_ms),
        "n_selected": int(k),
        "active_set": [int(i) for i in state.active_set],
        "train_combined_rss": float(train_rss),
        "train_combined_mse": float(train_mse),
        "train_combined_aic": _safe_ic(
            AICCriterion, rss=train_rss, k=k, n_samples=train_data.n_samples
        ),
        "train_combined_bic": _safe_ic(
            BICCriterion, rss=train_rss, k=k, n_samples=train_data.n_samples
        ),
        "train_chunk_mean_mse": float(np.mean(np.asarray(train_chunk_mse, dtype=float))),
        "train_chunk_std_mse": float(np.std(np.asarray(train_chunk_mse, dtype=float))),
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


def run_market_chunk_experiment(
    *,
    dataset_path: str | Path,
    train_chunks: int = 10,
    holdout_chunk: int = -1,
    max_steps: int | None = None,
    beam_width: int = 5,
    cv_aggregation: str = "mean_mse",
) -> dict[str, object]:
    dataset = load_market_chunk_dataset(dataset_path)
    chunks = list(dataset.gram_chunks)
    n_chunks_total = len(chunks)
    if n_chunks_total < 3:
        raise ValueError("Need at least 3 chunks for train/CV and holdout comparison.")
    if train_chunks <= 1:
        raise ValueError("train_chunks must be >= 2 for CV-based methods.")
    if train_chunks >= n_chunks_total:
        raise ValueError("train_chunks must be less than total number of chunks.")
    holdout_idx = holdout_chunk if holdout_chunk >= 0 else (n_chunks_total - 1)
    if holdout_idx < 0 or holdout_idx >= n_chunks_total:
        raise ValueError("holdout_chunk index is out of range.")
    train_indices = list(range(train_chunks))
    if holdout_idx in train_indices:
        raise ValueError("holdout_chunk must not overlap with training chunk indices.")

    train_chunk_data = [chunks[idx] for idx in train_indices]
    holdout_data = chunks[holdout_idx]
    combined_train = _combine_gram_chunks(train_chunk_data)
    cv_data = CrossValGramData(train_chunk_data)

    results: list[dict[str, object]] = []

    runs: list[tuple[str, object]] = [
        ("forward_aic_in_sample", ForwardSelection(criterion="aic")),
        (
            "cv_forward_oos_mean_mse",
            CrossValForwardSelection(cv_aggregation=cv_aggregation),
        ),
        (
            f"cv_beam_forward_oos_mean_mse_w{beam_width}",
            BeamCrossValForwardSelection(
                beam_width=beam_width,
                cv_aggregation=cv_aggregation,
            ),
        ),
    ]

    for method_name, selector in runs:
        start = time.perf_counter()
        if method_name == "forward_aic_in_sample":
            state = selector.fit(data=combined_train, max_steps=max_steps)
            row = _build_result_row(
                method_name=method_name,
                state=state,
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                train_data=combined_train,
                holdout_data=holdout_data,
                train_chunks=train_chunk_data,
                cv_data=None,
            )
        else:
            state = selector.fit(data=cv_data, max_steps=max_steps)
            row = _build_result_row(
                method_name=method_name,
                state=state,
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
                train_data=combined_train,
                holdout_data=holdout_data,
                train_chunks=train_chunk_data,
                cv_data=cv_data,
            )
        results.append(row)

    ranked = sorted(results, key=lambda row: float(row["oos_holdout_mse"]))
    for rank_idx, row in enumerate(ranked, start=1):
        row["rank_oos_holdout_mse"] = rank_idx

    return {
        "dataset_path": str(dataset_path),
        "dataset_flavor": dataset.meta.get("flavor"),
        "dataset_n_chunks": n_chunks_total,
        "dataset_n_features": len(dataset.feature_names),
        "train_chunk_indices": train_indices,
        "holdout_chunk_index": holdout_idx,
        "max_steps": max_steps,
        "beam_width": beam_width,
        "cv_aggregation": cv_aggregation,
        "results": ranked,
    }


def _render_terminal_table(payload: dict[str, object]) -> str:
    rows = payload["results"]
    if not isinstance(rows, list) or not rows:
        return "No results."
    lines = []
    lines.append(
        "method                              k    holdout_mse    holdout_r2    cv_mean_mse    rank"
    )
    lines.append(
        "----------------------------------------------------------------------------------------------"
    )
    for row in rows:
        method = str(row["method"])
        k = int(row["n_selected"])
        holdout_mse = float(row["oos_holdout_mse"])
        holdout_r2 = float(row["oos_holdout_r2"])
        cv_mean = row["cv_fold_mean_mse"]
        cv_str = "n/a" if cv_mean is None else f"{float(cv_mean):.6f}"
        rank = int(row["rank_oos_holdout_mse"])
        lines.append(
            f"{method:<34} {k:>3d}  {holdout_mse:>11.6f}  {holdout_r2:>10.4f}  "
            f"{cv_str:>11}  {rank:>4d}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare three forward-selection strategies on chunked market data: "
            "AIC-in-sample, CV-forward OOS, and CV-beam-forward OOS."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmarks/results/market_chunks_unknown.npz",
        help="Input NPZ produced by scripts/generate_market_chunks.py.",
    )
    parser.add_argument(
        "--train-chunks",
        type=int,
        default=10,
        help="Number of leading chunks used for model selection.",
    )
    parser.add_argument(
        "--holdout-chunk",
        type=int,
        default=-1,
        help="Chunk index used as true holdout; -1 means final chunk.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max steps for each selector.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width for the beam CV forward method.",
    )
    parser.add_argument(
        "--cv-aggregation",
        choices=("mean_mse", "median_mse", "sum_rss"),
        default="mean_mse",
        help="CV aggregation metric for methods 2/3.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/market_experiment_compare.json",
        help="Output JSON report path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_market_chunk_experiment(
        dataset_path=args.dataset,
        train_chunks=args.train_chunks,
        holdout_chunk=args.holdout_chunk,
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        cv_aggregation=args.cv_aggregation,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"Dataset={payload['dataset_path']} train_chunks={payload['train_chunk_indices']} "
        f"holdout_chunk={payload['holdout_chunk_index']}"
    )
    print(_render_terminal_table(payload))
    print(f"Wrote report: {output_path}")


if __name__ == "__main__":
    main()
