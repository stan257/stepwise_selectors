#!/usr/bin/env python
"""Inspect and export canonical microstructure features from simulated observables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.microstructure_chunks import (
    MicrostructureChunkConfig,
    build_microstructure_feature_table,
    microstructure_feature_registry,
    simulate_microstructure_observables,
)



def _csv_tuple(value: str | None) -> tuple[str, ...] | None:
    if value is None or value.strip() == "":
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate L1 microstructure observables, list canonical features, and "
            "optionally export selected feature subsets."
        )
    )
    parser.add_argument("--seed", type=int, default=20260307)
    parser.add_argument("--n-chunks", type=int, default=11)
    parser.add_argument("--events-per-chunk", type=int, default=5000)
    parser.add_argument("--warmup-events", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--n-regimes", type=int, default=5)
    parser.add_argument("--target-horizon-events", type=int, default=8)
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Print the canonical feature library and exit unless --output is also set.",
    )
    parser.add_argument(
        "--families",
        type=str,
        default=None,
        help="Comma-separated family filter, e.g. imbalance,flow,spread",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated explicit feature names to export or inspect.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional NPZ path for the selected feature matrix.",
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
    )
    names = _csv_tuple(args.features)
    families = _csv_tuple(args.families)

    registry = microstructure_feature_registry(args.n_features)
    if args.list_features:
        for spec in registry.all_specs():
            print(f"{spec.family:12s} {spec.name}")
        if args.output is None:
            return

    observables = simulate_microstructure_observables(config)
    table = build_microstructure_feature_table(
        observables,
        registry=registry,
        names=names,
        families=families,
    )

    print(
        f"Generated {len(table.feature_names)} features across {table.matrix.shape[0]} events."
    )
    print("Families:", ", ".join(sorted(set(table.feature_families))))

    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            matrix=table.matrix,
            feature_names=np.asarray(table.feature_names, dtype=str),
            feature_families=np.asarray(table.feature_families, dtype=str),
            event_index=observables.event_index,
        )
        print(f"Saved feature matrix to: {out}")


if __name__ == "__main__":
    main()
