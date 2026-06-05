#!/usr/bin/env python
"""Generate and persist microstructure-style GramData chunks for forward experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.microstructure_chunks import (
    MicrostructureChunkConfig,
    apply_microstructure_preset,
    generate_microstructure_gram_chunks_known,
    generate_microstructure_gram_chunks_unknown,
    microstructure_preset_names,
    save_microstructure_chunk_dataset,
)



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an event-time L1 microstructure chunk dataset and save it as NPZ. "
            "Use the last chunk as true out-of-sample."
        )
    )
    parser.add_argument(
        "--flavor",
        choices=("known", "unknown"),
        default="unknown",
        help="Data flavor: known support (diagnostic) or unknown support (OOS-driven).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npz path. Default: benchmarks/results/"
            "microstructure_chunks_<flavor>.npz"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260307)
    parser.add_argument(
        "--preset",
        choices=microstructure_preset_names(),
        default="default",
        help="Named config preset to apply before generation.",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=11,
        help="Total chunks to generate. Recommended: keep final chunk as OOS.",
    )
    parser.add_argument("--events-per-chunk", type=int, default=5000)
    parser.add_argument("--warmup-events", type=int, default=1000)
    parser.add_argument(
        "--n-features",
        type=int,
        default=64,
        help="Fixed public feature library size. Supported values: 64, 128, 192.",
    )
    parser.add_argument("--n-regimes", type=int, default=5)
    parser.add_argument("--target-horizon-events", type=int, default=8)
    parser.add_argument("--support-size", type=int, default=8)
    parser.add_argument("--support-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--signal-scale", type=float, default=0.20)
    parser.add_argument("--target-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--unknown-support-size",
        type=int,
        default=None,
        help="Only used for --flavor unknown. If omitted, uses module default logic.",
    )
    parser.add_argument("--unknown-support-overlap-ratio", type=float, default=0.35)
    parser.add_argument("--unknown-chunk-jitter", type=int, default=2)
    parser.add_argument("--unknown-nonlinear-strength", type=float, default=0.10)
    parser.add_argument(
        "--expose-truth",
        action="store_true",
        help="Only relevant for --flavor unknown. Include hidden support/beta in output.",
    )
    return parser



def main() -> None:
    args = _build_parser().parse_args()
    config = MicrostructureChunkConfig(
        seed=args.seed,
        n_chunks=args.n_chunks,
        events_per_chunk=args.events_per_chunk,
        warmup_events=args.warmup_events,
        n_features=args.n_features,
        n_regimes=args.n_regimes,
        target_horizon_events=args.target_horizon_events,
        support_size=args.support_size,
        support_overlap_ratio=args.support_overlap_ratio,
        signal_scale=args.signal_scale,
        target_noise_std=args.target_noise_std,
        unknown_support_size=args.unknown_support_size,
        unknown_support_overlap_ratio=args.unknown_support_overlap_ratio,
        unknown_chunk_jitter=args.unknown_chunk_jitter,
        unknown_nonlinear_strength=args.unknown_nonlinear_strength,
    )
    config = apply_microstructure_preset(config, preset=args.preset)

    if args.flavor == "known":
        dataset = generate_microstructure_gram_chunks_known(config)
    else:
        dataset = generate_microstructure_gram_chunks_unknown(
            config, expose_truth=bool(args.expose_truth)
        )

    output = (
        Path(args.output)
        if args.output is not None
        else ROOT / "benchmarks" / "results" / f"microstructure_chunks_{args.flavor}.npz"
    )
    save_microstructure_chunk_dataset(dataset, output)

    n_chunks = len(dataset.gram_chunks)
    p = len(dataset.feature_names)
    supports = "available" if dataset.support_by_chunk is not None else "hidden"
    print(f"Saved microstructure chunk dataset to: {output}")
    print(
        f"Flavor={args.flavor} preset={args.preset} chunks={n_chunks} "
        f"features={p} supports={supports}"
    )
    print(
        f"Recommended split: train chunks [0..{n_chunks - 2}], "
        f"OOS chunk [{n_chunks - 1}]"
    )
    print(f"Target horizon: {dataset.meta['target_horizon_events']} events")
    print(
        "Load later via: from benchmarks.microstructure_chunks import "
        "load_microstructure_chunk_dataset"
    )


if __name__ == "__main__":
    main()
