"""Compatibility facade for the microstructure benchmark pipeline."""

from __future__ import annotations

from pathlib import Path

from .chunk_dataset_io import load_chunk_dataset, save_chunk_dataset
from .microstructure import (
    CANONICAL_MICROSTRUCTURE_FEATURE_NAMES,
    L1MicrostructureSimulator,
    MicrostructureChunkConfig,
    MicrostructureChunkDataset,
    MicrostructureFeatureRegistry,
    MicrostructureFeatureSpec,
    MicrostructureFeatureTable,
    MicrostructureObservables,
    build_microstructure_feature_table,
    build_microstructure_target,
    default_microstructure_feature_registry,
    generate_known_microstructure_dataset,
    generate_unknown_microstructure_dataset,
    microstructure_feature_registry,
    apply_microstructure_preset,
    simulate_microstructure_observables,
    microstructure_preset_names,
    supported_microstructure_feature_counts,
    validate_microstructure_config,
)



def generate_microstructure_gram_chunks_known(
    config: MicrostructureChunkConfig,
) -> MicrostructureChunkDataset:
    """Generate sequential GramData chunks with known rotating microstructure support."""
    return generate_known_microstructure_dataset(config)



def generate_microstructure_gram_chunks_unknown(
    config: MicrostructureChunkConfig,
    *,
    expose_truth: bool = False,
) -> MicrostructureChunkDataset:
    """Generate harder microstructure chunks; reserve the final chunk as true OOS."""
    return generate_unknown_microstructure_dataset(config, expose_truth=expose_truth)



def save_microstructure_chunk_dataset(
    dataset: MicrostructureChunkDataset, path: str | Path
) -> None:
    """Persist a MicrostructureChunkDataset to a compressed NPZ file."""
    save_chunk_dataset(dataset, path)



def load_microstructure_chunk_dataset(path: str | Path) -> MicrostructureChunkDataset:
    """Load a MicrostructureChunkDataset previously stored as NPZ."""
    return load_chunk_dataset(
        path,
        dataset_factory=MicrostructureChunkDataset,
        invalid_version_message="Unsupported microstructure chunk dataset version.",
    )


__all__ = [
    "CANONICAL_MICROSTRUCTURE_FEATURE_NAMES",
    "L1MicrostructureSimulator",
    "apply_microstructure_preset",
    "MicrostructureChunkConfig",
    "MicrostructureChunkDataset",
    "MicrostructureFeatureRegistry",
    "MicrostructureFeatureSpec",
    "MicrostructureFeatureTable",
    "MicrostructureObservables",
    "build_microstructure_feature_table",
    "build_microstructure_target",
    "default_microstructure_feature_registry",
    "generate_microstructure_gram_chunks_known",
    "generate_microstructure_gram_chunks_unknown",
    "load_microstructure_chunk_dataset",
    "microstructure_feature_registry",
    "microstructure_preset_names",
    "save_microstructure_chunk_dataset",
    "simulate_microstructure_observables",
    "supported_microstructure_feature_counts",
    "validate_microstructure_config",
]
