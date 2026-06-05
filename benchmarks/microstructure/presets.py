"""Named configuration presets for the microstructure benchmark."""

from __future__ import annotations

from dataclasses import replace

from .types import MicrostructureChunkConfig


_MICROSTRUCTURE_PRESET_NAMES: tuple[str, ...] = (
    "default",
    "unknown_wide_support",
    "unknown_multiscale_192",
    "unknown_multiscale_192_rapid_rotation",
)


def microstructure_preset_names() -> tuple[str, ...]:
    """Return supported named microstructure preset identifiers."""
    return _MICROSTRUCTURE_PRESET_NAMES


def apply_microstructure_preset(
    config: MicrostructureChunkConfig,
    *,
    preset: str,
) -> MicrostructureChunkConfig:
    """Apply a named benchmark preset to a base microstructure config.

    `unknown_wide_support` makes larger k budgets matter more by spreading
    comparable linear signal across a broader hidden support.

    `unknown_multiscale_192` and `unknown_multiscale_192_rapid_rotation` move to
    a wider multiscale feature bank with more regimes, broader hidden support,
    and faster support rotation.
    """
    if not isinstance(preset, str):
        raise TypeError("preset must be a string.")
    normalized = preset.strip().lower()
    if normalized == "default":
        return config
    if normalized == "unknown_wide_support":
        return replace(
            config,
            unknown_support_size=20,
            unknown_support_overlap_ratio=0.15,
            unknown_chunk_jitter=5,
            signal_scale=0.155,
        )
    if normalized == "unknown_multiscale_192":
        return replace(
            config,
            n_features=192,
            n_regimes=min(config.n_chunks, 7),
            target_horizon_events=10,
            unknown_support_size=24,
            unknown_support_overlap_ratio=0.22,
            unknown_chunk_jitter=6,
            signal_scale=0.145,
            unknown_nonlinear_strength=0.14,
        )
    if normalized == "unknown_multiscale_192_rapid_rotation":
        return replace(
            config,
            n_features=192,
            n_regimes=min(config.n_chunks, 8),
            target_horizon_events=12,
            unknown_support_size=32,
            unknown_support_overlap_ratio=0.10,
            unknown_chunk_jitter=10,
            signal_scale=0.125,
            unknown_nonlinear_strength=0.18,
            target_noise_std=0.012,
        )
    raise ValueError(
        "Unknown microstructure preset. Supported values: default, unknown_wide_support, "
        "unknown_multiscale_192, unknown_multiscale_192_rapid_rotation."
    )


__all__ = [
    "apply_microstructure_preset",
    "microstructure_preset_names",
]
