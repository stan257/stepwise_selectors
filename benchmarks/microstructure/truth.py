"""Support and signal-truth builders for the microstructure benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import MicrostructureFeatureRegistry
from .types import MicrostructureChunkConfig
from .utils import student_unit_variance

_SLOT_GROUP_ORDER: tuple[str, ...] = (
    "microdev",
    "imbalance",
    "flow",
    "spread",
    "volatility",
    "interaction",
    "momentum",
    "flow",
    "imbalance",
    "microdev",
    "spread",
    "volatility",
    "momentum",
    "flow",
    "imbalance",
    "microdev",
    "spread",
    "volatility",
    "interaction",
    "flow",
    "depth",
    "intensity",
    "depth",
    "momentum",
)


@dataclass(frozen=True)
class RegimeSignalSpec:
    regime_by_chunk: tuple[int, ...]
    support_by_chunk: tuple[tuple[int, ...], ...]
    beta_by_chunk: tuple[np.ndarray, ...]
    signal_component: np.ndarray
    nonlinear_component: np.ndarray



def _family_index_map(registry: MicrostructureFeatureRegistry) -> dict[str, tuple[int, ...]]:
    out: dict[str, list[int]] = {}
    for idx, spec in enumerate(registry.all_specs()):
        out.setdefault(spec.family, []).append(idx)
    return {family: tuple(indices) for family, indices in out.items()}



def _sample_feature_for_family(
    family: str,
    *,
    family_indices: dict[str, tuple[int, ...]],
    active: set[int],
    rng: np.random.Generator,
    n_features: int,
) -> int:
    candidates = [idx for idx in family_indices.get(family, ()) if idx not in active]
    if not candidates:
        candidates = [idx for idx in range(n_features) if idx not in active]
    return int(rng.choice(np.asarray(candidates, dtype=int)))



def _draw_feature_coefficient(feature_idx: int, registry: MicrostructureFeatureRegistry, rng: np.random.Generator) -> float:
    family = registry.all_specs()[feature_idx].family
    magnitude = float(rng.uniform(0.35, 0.95))
    if family in {"microdev", "imbalance", "flow", "interaction", "depth"}:
        sign = 1.0
    elif family == "spread":
        sign = -1.0
    elif family == "intensity":
        sign = -1.0 if rng.random() < 0.55 else 1.0
    else:
        sign = -1.0 if rng.random() < 0.35 else 1.0
    return sign * magnitude



def build_regime_truth(
    *,
    registry: MicrostructureFeatureRegistry,
    n_regimes: int,
    support_size: int,
    overlap_ratio: float,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[int, ...], ...], tuple[np.ndarray, ...]]:
    n_features = len(registry.feature_names())
    family_indices = _family_index_map(registry)
    supports: list[tuple[int, ...]] = []
    betas: list[np.ndarray] = []
    slot_groups = list(_SLOT_GROUP_ORDER[:support_size])
    while len(slot_groups) < support_size:
        slot_groups.extend(_SLOT_GROUP_ORDER)
    slot_groups = slot_groups[:support_size]

    prev_slot_features: list[int] | None = None
    for _ in range(n_regimes):
        active: set[int] = set()
        slot_features: list[int] = []
        for slot_idx, family in enumerate(slot_groups):
            keep_prev = (
                prev_slot_features is not None
                and rng.random() < overlap_ratio
                and prev_slot_features[slot_idx] not in active
            )
            if keep_prev:
                chosen = int(prev_slot_features[slot_idx])
            else:
                chosen = _sample_feature_for_family(
                    family,
                    family_indices=family_indices,
                    active=active,
                    rng=rng,
                    n_features=n_features,
                )
            active.add(chosen)
            slot_features.append(chosen)

        support = tuple(sorted(slot_features))
        beta = np.zeros(n_features, dtype=float)
        for feature_idx in support:
            beta[feature_idx] = _draw_feature_coefficient(feature_idx, registry, rng)
        supports.append(support)
        betas.append(beta)
        prev_slot_features = slot_features

    return tuple(supports), tuple(betas)



def build_unknown_chunk_truth(
    *,
    config: MicrostructureChunkConfig,
    regime_by_chunk: tuple[int, ...],
    registry: MicrostructureFeatureRegistry,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[int, ...], ...], tuple[np.ndarray, ...]]:
    support_size = (
        config.unknown_support_size
        if config.unknown_support_size is not None
        else min(
            config.n_features - 1,
            max(config.support_size + 4, int(round(config.support_size * 1.5))),
        )
    )
    if support_size <= 0:
        raise ValueError("Unknown flavor resolved to invalid support size.")

    regime_supports, regime_betas = build_regime_truth(
        registry=registry,
        n_regimes=config.n_regimes,
        support_size=support_size,
        overlap_ratio=config.unknown_support_overlap_ratio,
        rng=rng,
    )

    full_index = np.arange(config.n_features, dtype=int)
    supports_by_chunk: list[tuple[int, ...]] = []
    betas_by_chunk: list[np.ndarray] = []
    for chunk_regime in regime_by_chunk:
        base_support = np.array(regime_supports[chunk_regime], dtype=int)
        active = set(int(v) for v in base_support.tolist())
        jitter = min(config.unknown_chunk_jitter, support_size - 1)
        if jitter > 0:
            drop = rng.choice(np.array(sorted(active), dtype=int), size=jitter, replace=False)
            for idx in drop.tolist():
                active.remove(int(idx))
            pool = np.array([idx for idx in full_index if idx not in active], dtype=int)
            add = rng.choice(pool, size=jitter, replace=False)
            for idx in add.tolist():
                active.add(int(idx))

        support = tuple(sorted(int(i) for i in active))
        beta = np.zeros(config.n_features, dtype=float)
        base_beta = regime_betas[chunk_regime]
        base_support_set = set(int(i) for i in base_support.tolist())
        for feature_idx in support:
            coeff = float(base_beta[feature_idx])
            if feature_idx not in base_support_set:
                coeff = _draw_feature_coefficient(feature_idx, registry, rng)
            coeff *= 1.0 + 0.12 * student_unit_variance(config.student_df, rng)
            if abs(coeff) < 0.05:
                coeff = 0.05 if coeff >= 0.0 else -0.05
            beta[feature_idx] = coeff
        supports_by_chunk.append(support)
        betas_by_chunk.append(beta)

    return tuple(supports_by_chunk), tuple(betas_by_chunk)


__all__ = [
    "RegimeSignalSpec",
    "build_regime_truth",
    "build_unknown_chunk_truth",
]
