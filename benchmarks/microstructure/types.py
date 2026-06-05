"""Public data types for the microstructure benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from selection import GramData


@dataclass(frozen=True)
class MicrostructureChunkConfig:
    """Configuration for chunked microstructure-style synthetic generators."""

    seed: int = 20260307
    n_chunks: int = 11
    events_per_chunk: int = 5000
    warmup_events: int = 1000
    n_features: int = 64
    n_regimes: int = 5
    target_horizon_events: int = 8

    # Known-support flavor controls.
    support_size: int = 8
    support_overlap_ratio: float = 0.5
    signal_scale: float = 0.20
    target_noise_std: float = 0.01

    # Unknown-support flavor controls.
    unknown_support_size: int | None = None
    unknown_support_overlap_ratio: float = 0.35
    unknown_chunk_jitter: int = 2
    unknown_nonlinear_strength: float = 0.10

    # Base microstructure dynamics.
    tick_size: float = 1e-4
    base_half_spread_ticks: float = 1.0
    base_queue_depth: float = 100.0
    queue_mean_reversion: float = 0.92
    activity_persistence: float = 0.97
    sign_persistence: float = 0.85
    impact_strength: float = 0.60
    spread_vol_sensitivity: float = 0.50
    cancel_rate_scale: float = 0.20
    replenish_rate_scale: float = 0.25
    student_df: float = 7.0


@dataclass(frozen=True)
class MicrostructureChunkDataset:
    """Output container for chunked microstructure experiments."""

    gram_chunks: list[GramData]
    feature_names: tuple[str, ...]
    chunk_ranges: tuple[tuple[int, int], ...]
    regime_by_chunk: tuple[int, ...]
    support_by_chunk: tuple[tuple[int, ...], ...] | None
    beta_by_chunk: tuple[np.ndarray, ...] | None
    meta: dict[str, object]


@dataclass(frozen=True)
class MicrostructureObservables:
    """Simulated event-time L1 order-book state and derived observable series."""

    event_index: np.ndarray
    regime_by_event: np.ndarray
    mid_price: np.ndarray
    spread_ticks: np.ndarray
    q_bid: np.ndarray
    q_ask: np.ndarray
    signed_trade: np.ndarray
    signed_volume: np.ndarray
    price_changed: np.ndarray
    spread_changed: np.ndarray
    depletion_dir: np.ndarray
    ofi: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    microprice: np.ndarray
    microdev: np.ndarray
    imbalance: np.ndarray
    rel_spread: np.ndarray
    mid_log: np.ndarray
    micro_log: np.ndarray
    mid_ret: np.ndarray
    micro_ret: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def series(self, name: str) -> np.ndarray:
        if hasattr(self, name):
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                return value
        if name in self.extras:
            return self.extras[name]
        raise KeyError(f"Unknown observable series: {name!r}")

    def as_dict(self) -> dict[str, np.ndarray]:
        out = {
            "event_index": self.event_index,
            "regime_by_event": self.regime_by_event,
            "mid_price": self.mid_price,
            "spread_ticks": self.spread_ticks,
            "q_bid": self.q_bid,
            "q_ask": self.q_ask,
            "signed_trade": self.signed_trade,
            "signed_volume": self.signed_volume,
            "price_changed": self.price_changed,
            "spread_changed": self.spread_changed,
            "depletion_dir": self.depletion_dir,
            "ofi": self.ofi,
            "bid": self.bid,
            "ask": self.ask,
            "microprice": self.microprice,
            "microdev": self.microdev,
            "imbalance": self.imbalance,
            "rel_spread": self.rel_spread,
            "mid_log": self.mid_log,
            "micro_log": self.micro_log,
            "mid_ret": self.mid_ret,
            "micro_ret": self.micro_ret,
        }
        out.update(self.extras)
        return out


@dataclass(frozen=True)
class MicrostructureFeatureSpec:
    """Specification for one named feature derived from observables."""

    name: str
    family: str
    lookback: int
    required_series: tuple[str, ...]
    description: str
    builder: Callable[[MicrostructureObservables], np.ndarray]

    def compute(self, observables: MicrostructureObservables) -> np.ndarray:
        return np.asarray(self.builder(observables), dtype=float)


@dataclass(frozen=True)
class MicrostructureFeatureTable:
    """Inspectable generated microstructure feature matrix."""

    matrix: np.ndarray
    feature_names: tuple[str, ...]
    feature_families: tuple[str, ...]
    feature_descriptions: tuple[str, ...]
    name_to_index: dict[str, int]

    def column(self, name: str) -> np.ndarray:
        if name not in self.name_to_index:
            raise KeyError(f"Unknown feature name: {name!r}")
        return np.asarray(self.matrix[:, self.name_to_index[name]], dtype=float)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: self.column(name) for name in self.feature_names}

    def select(
        self,
        *,
        names: tuple[str, ...] | None = None,
        families: tuple[str, ...] | None = None,
    ) -> "MicrostructureFeatureTable":
        keep: list[int] = []
        name_filter = None if names is None else tuple(str(name) for name in names)
        family_filter = None if families is None else set(str(family) for family in families)
        if name_filter is None:
            for idx, (_, family) in enumerate(
                zip(self.feature_names, self.feature_families)
            ):
                if family_filter is not None and family not in family_filter:
                    continue
                keep.append(idx)
        else:
            for name in name_filter:
                if name not in self.name_to_index:
                    raise KeyError(f"Unknown feature name: {name!r}")
                idx = self.name_to_index[name]
                if family_filter is not None and self.feature_families[idx] not in family_filter:
                    continue
                keep.append(idx)

        matrix = np.ascontiguousarray(self.matrix[:, keep]) if keep else np.zeros((self.matrix.shape[0], 0), dtype=float)
        feature_names = tuple(self.feature_names[idx] for idx in keep)
        feature_families = tuple(self.feature_families[idx] for idx in keep)
        feature_descriptions = tuple(self.feature_descriptions[idx] for idx in keep)
        name_to_index = {name: idx for idx, name in enumerate(feature_names)}
        return MicrostructureFeatureTable(
            matrix=matrix,
            feature_names=feature_names,
            feature_families=feature_families,
            feature_descriptions=feature_descriptions,
            name_to_index=name_to_index,
        )


__all__ = [
    "MicrostructureChunkConfig",
    "MicrostructureChunkDataset",
    "MicrostructureFeatureSpec",
    "MicrostructureFeatureTable",
    "MicrostructureObservables",
]
