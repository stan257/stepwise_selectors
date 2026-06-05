"""End-to-end microstructure benchmark dataset pipeline."""

from __future__ import annotations

import numpy as np

from selection import GramData

from .features import (
    MicrostructureFeatureRegistry,
    build_microstructure_feature_table,
    microstructure_feature_registry,
)
from .simulator import L1MicrostructureSimulator, simulate_microstructure_observables
from .targets import build_microstructure_target
from .truth import RegimeSignalSpec, build_regime_truth, build_unknown_chunk_truth
from .types import MicrostructureChunkConfig, MicrostructureChunkDataset, MicrostructureObservables
from .utils import (
    assign_regimes,
    build_event_regimes,
    safe_corr,
    trade_sign_autocorr,
    validate_microstructure_config,
)



def _usable_window(
    config: MicrostructureChunkConfig,
    observables: MicrostructureObservables,
    *,
    lookback: int,
) -> tuple[int, int, int]:
    valid_stop = observables.mid_price.shape[0] - config.target_horizon_events
    usable_start = lookback + config.warmup_events
    usable_end = usable_start + config.n_chunks * config.events_per_chunk
    return usable_start, usable_end, valid_stop



def _build_diagnostics(
    observables: MicrostructureObservables,
    *,
    usable_start: int,
    usable_end: int,
    valid_stop: int,
) -> dict[str, np.ndarray]:
    return {
        "spread_ticks": np.asarray(
            observables.spread_ticks[:valid_stop][usable_start:usable_end], dtype=float
        ),
        "imbalance": np.asarray(
            observables.imbalance[:valid_stop][usable_start:usable_end], dtype=float
        ),
        "signed_trade": np.asarray(
            observables.signed_trade[:valid_stop][usable_start:usable_end], dtype=float
        ),
        "ofi": np.asarray(observables.ofi[:valid_stop][usable_start:usable_end], dtype=float),
        "microdev": np.asarray(
            observables.microdev[:valid_stop][usable_start:usable_end], dtype=float
        ),
        "price_changed": np.asarray(
            observables.price_changed[:valid_stop][usable_start:usable_end], dtype=float
        ),
    }



def _slice_feature_table(
    table,
    *,
    usable_start: int,
    usable_end: int,
    valid_stop: int,
):
    return np.ascontiguousarray(
        table.matrix[:valid_stop][usable_start:usable_end], dtype=float
    )



def _make_signal_spec_known(
    config: MicrostructureChunkConfig,
    *,
    X: np.ndarray,
    registry: MicrostructureFeatureRegistry,
) -> RegimeSignalSpec:
    regime_by_chunk = assign_regimes(config.n_chunks, config.n_regimes)
    regime_supports, regime_betas = build_regime_truth(
        registry=registry,
        n_regimes=config.n_regimes,
        support_size=config.support_size,
        overlap_ratio=config.support_overlap_ratio,
        rng=np.random.default_rng(config.seed + 271),
    )
    support_by_chunk = tuple(regime_supports[regime] for regime in regime_by_chunk)
    beta_by_chunk = tuple(regime_betas[regime].copy() for regime in regime_by_chunk)
    row_regimes = np.repeat(np.asarray(regime_by_chunk, dtype=int), config.events_per_chunk)
    beta_rows = np.stack([regime_betas[int(regime)] for regime in row_regimes], axis=0)
    X_scaled = (X - np.mean(X, axis=0, keepdims=True)) / (
        np.std(X, axis=0, keepdims=True) + 1e-12
    )
    signal = np.einsum("ij,ij->i", X_scaled, beta_rows)
    return RegimeSignalSpec(
        regime_by_chunk=regime_by_chunk,
        support_by_chunk=support_by_chunk,
        beta_by_chunk=beta_by_chunk,
        signal_component=config.signal_scale * signal,
        nonlinear_component=np.zeros(X.shape[0], dtype=float),
    )



def _make_signal_spec_unknown(
    config: MicrostructureChunkConfig,
    *,
    X: np.ndarray,
    registry: MicrostructureFeatureRegistry,
) -> RegimeSignalSpec:
    regime_by_chunk = assign_regimes(config.n_chunks, config.n_regimes)
    support_by_chunk, beta_by_chunk = build_unknown_chunk_truth(
        config=config,
        regime_by_chunk=regime_by_chunk,
        registry=registry,
        rng=np.random.default_rng(config.seed + 313),
    )
    X_scaled = (X - np.mean(X, axis=0, keepdims=True)) / (
        np.std(X, axis=0, keepdims=True) + 1e-12
    )
    beta_rows = np.repeat(np.stack(beta_by_chunk, axis=0), config.events_per_chunk, axis=0)
    signal = np.einsum("ij,ij->i", X_scaled, beta_rows)
    nonlinear = config.unknown_nonlinear_strength * (
        0.55 * (X_scaled[:, 13] ** 2 - 1.0)
        + 0.30 * (X_scaled[:, 62] * X_scaled[:, 49])
        + 0.15 * (X_scaled[:, 33] * X_scaled[:, 43])
    )
    return RegimeSignalSpec(
        regime_by_chunk=regime_by_chunk,
        support_by_chunk=support_by_chunk,
        beta_by_chunk=beta_by_chunk,
        signal_component=config.signal_scale * signal,
        nonlinear_component=nonlinear,
    )



def _build_chunked_dataset(
    *,
    X: np.ndarray,
    y: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    regime_by_chunk: tuple[int, ...],
    support_by_chunk: tuple[tuple[int, ...], ...] | None,
    beta_by_chunk: tuple[np.ndarray, ...] | None,
    feature_names: tuple[str, ...],
    config: MicrostructureChunkConfig,
    flavor: str,
    signal_component: np.ndarray,
) -> MicrostructureChunkDataset:
    gram_chunks: list[GramData] = []
    chunk_ranges: list[tuple[int, int]] = []
    feature_mean_max: list[float] = []
    target_mean_abs: list[float] = []
    chunk_target_std: list[float] = []
    chunk_mean_spread: list[float] = []
    chunk_mean_abs_imbalance: list[float] = []
    chunk_trade_sign_autocorr: list[float] = []
    chunk_ofi_target_corr: list[float] = []
    chunk_microdev_target_corr: list[float] = []
    chunk_price_change_rate: list[float] = []

    for chunk_idx in range(config.n_chunks):
        start = chunk_idx * config.events_per_chunk
        end = start + config.events_per_chunk
        chunk_ranges.append((start, end))
        X_chunk = X[start:end]
        y_chunk = y[start:end]
        X_center = X_chunk - np.mean(X_chunk, axis=0, keepdims=True)
        y_center = y_chunk - float(np.mean(y_chunk))

        feature_mean_max.append(float(np.max(np.abs(np.mean(X_center, axis=0)))))
        target_mean_abs.append(float(abs(float(np.mean(y_center)))))
        chunk_target_std.append(float(np.std(y_chunk)))
        gram_chunks.append(
            GramData(
                gram=np.ascontiguousarray(X_center.T @ X_center),
                cov=np.ascontiguousarray(X_center.T @ y_center),
                y_norm=float(y_center @ y_center),
                n_samples=int(config.events_per_chunk),
                warn_if_uncentered=False,
            )
        )

        spread_chunk = diagnostics["spread_ticks"][start:end]
        imbalance_chunk = diagnostics["imbalance"][start:end]
        sign_chunk = diagnostics["signed_trade"][start:end]
        ofi_chunk = diagnostics["ofi"][start:end]
        microdev_chunk = diagnostics["microdev"][start:end]
        price_change_chunk = diagnostics["price_changed"][start:end]

        chunk_mean_spread.append(float(np.mean(spread_chunk)))
        chunk_mean_abs_imbalance.append(float(np.mean(np.abs(imbalance_chunk))))
        chunk_trade_sign_autocorr.append(trade_sign_autocorr(sign_chunk))
        chunk_ofi_target_corr.append(safe_corr(ofi_chunk, y_chunk))
        chunk_microdev_target_corr.append(safe_corr(microdev_chunk, y_chunk))
        chunk_price_change_rate.append(float(np.mean(price_change_chunk)))

    y_centered = y - float(np.mean(y))
    y_var = float(np.var(y_centered))
    signal_var = float(np.var(signal_component))
    denom = max(y_var - signal_var, 1e-12)
    variance = float(np.mean(y_centered**2))
    kurtosis = 0.0 if variance <= 0.0 else float(np.mean(y_centered**4) / (variance**2))

    meta: dict[str, object] = {
        "flavor": flavor,
        "n_chunks": config.n_chunks,
        "events_per_chunk": config.events_per_chunk,
        "n_features": config.n_features,
        "n_regimes": config.n_regimes,
        "target_horizon_events": config.target_horizon_events,
        "train_chunk_count_recommended": config.n_chunks - 1,
        "oos_chunk_index_recommended": config.n_chunks - 1,
        "signal_to_noise_ratio_est": signal_var / denom,
        "target_kurtosis": kurtosis,
        "chunk_target_std": chunk_target_std,
        "chunk_mean_spread": chunk_mean_spread,
        "chunk_mean_abs_imbalance": chunk_mean_abs_imbalance,
        "chunk_trade_sign_autocorr_lag1": chunk_trade_sign_autocorr,
        "chunk_ofi_target_corr": chunk_ofi_target_corr,
        "chunk_microdev_target_corr": chunk_microdev_target_corr,
        "chunk_price_change_rate": chunk_price_change_rate,
        "max_abs_centered_feature_mean_per_chunk": feature_mean_max,
        "max_abs_centered_target_mean_per_chunk": target_mean_abs,
    }

    return MicrostructureChunkDataset(
        gram_chunks=gram_chunks,
        feature_names=feature_names,
        chunk_ranges=tuple(chunk_ranges),
        regime_by_chunk=regime_by_chunk,
        support_by_chunk=support_by_chunk,
        beta_by_chunk=beta_by_chunk,
        meta=meta,
    )



def _simulate_and_build_components(
    config: MicrostructureChunkConfig,
    *,
    registry: MicrostructureFeatureRegistry,
) -> tuple[MicrostructureObservables, np.ndarray, np.ndarray, dict[str, np.ndarray], tuple[str, ...]]:
    _, regime_by_event, lookback, _, total_events = build_event_regimes(config)
    simulator = L1MicrostructureSimulator()
    observables = simulator.simulate(
        config,
        rng=np.random.default_rng(config.seed),
        regime_by_event=regime_by_event,
        n_events_total=total_events,
    )
    table = build_microstructure_feature_table(observables, registry=registry)
    y_base_full = build_microstructure_target(
        observables, horizon_events=config.target_horizon_events
    )
    usable_start, usable_end, valid_stop = _usable_window(
        config, observables, lookback=lookback
    )
    X = _slice_feature_table(
        table,
        usable_start=usable_start,
        usable_end=usable_end,
        valid_stop=valid_stop,
    )
    y_base = np.asarray(y_base_full[usable_start:usable_end], dtype=float)
    diagnostics = _build_diagnostics(
        observables,
        usable_start=usable_start,
        usable_end=usable_end,
        valid_stop=valid_stop,
    )
    return observables, X, y_base, diagnostics, table.feature_names



def generate_known_microstructure_dataset(
    config: MicrostructureChunkConfig,
) -> MicrostructureChunkDataset:
    validated = validate_microstructure_config(config)
    registry = microstructure_feature_registry(validated.n_features)
    _, X, y_base, diagnostics, feature_names = _simulate_and_build_components(
        validated, registry=registry
    )
    signal_spec = _make_signal_spec_known(validated, X=X, registry=registry)
    noise_rng = np.random.default_rng(validated.seed + 401)
    y = (
        y_base
        + signal_spec.signal_component
        + validated.target_noise_std * noise_rng.standard_normal(y_base.shape[0])
    )
    return _build_chunked_dataset(
        X=X,
        y=y,
        diagnostics=diagnostics,
        regime_by_chunk=signal_spec.regime_by_chunk,
        support_by_chunk=signal_spec.support_by_chunk,
        beta_by_chunk=signal_spec.beta_by_chunk,
        feature_names=feature_names,
        config=validated,
        flavor="known_support",
        signal_component=signal_spec.signal_component,
    )



def generate_unknown_microstructure_dataset(
    config: MicrostructureChunkConfig,
    *,
    expose_truth: bool = False,
) -> MicrostructureChunkDataset:
    validated = validate_microstructure_config(config)
    registry = microstructure_feature_registry(validated.n_features)
    _, X, y_base, diagnostics, feature_names = _simulate_and_build_components(
        validated, registry=registry
    )
    signal_spec = _make_signal_spec_unknown(validated, X=X, registry=registry)
    noise_rng = np.random.default_rng(validated.seed + 443)
    y = (
        y_base
        + signal_spec.signal_component
        + signal_spec.nonlinear_component
        + (1.5 * validated.target_noise_std)
        * noise_rng.standard_normal(y_base.shape[0])
    )
    support_out = signal_spec.support_by_chunk if expose_truth else None
    beta_out = signal_spec.beta_by_chunk if expose_truth else None
    return _build_chunked_dataset(
        X=X,
        y=y,
        diagnostics=diagnostics,
        regime_by_chunk=signal_spec.regime_by_chunk,
        support_by_chunk=support_out,
        beta_by_chunk=beta_out,
        feature_names=feature_names,
        config=validated,
        flavor="unknown_support",
        signal_component=signal_spec.signal_component + signal_spec.nonlinear_component,
    )


__all__ = [
    "generate_known_microstructure_dataset",
    "generate_unknown_microstructure_dataset",
    "simulate_microstructure_observables",
    "validate_microstructure_config",
]
