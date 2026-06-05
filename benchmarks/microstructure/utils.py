"""Utility helpers for the microstructure benchmark pipeline."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from .types import MicrostructureChunkConfig


CANONICAL_MICROSTRUCTURE_FEATURE_COUNT = 64
SUPPORTED_MICROSTRUCTURE_FEATURE_COUNTS: tuple[int, ...] = (64, 128, 192)


class ValidationError(ValueError):
    """Validation error for microstructure benchmark inputs."""


def supported_microstructure_feature_counts() -> tuple[int, ...]:
    return SUPPORTED_MICROSTRUCTURE_FEATURE_COUNTS


def validate_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"{name} must be finite.")
    return value_f



def validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value_i = int(value)
    if value_i <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value_i



def validate_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value_i = int(value)
    if value_i < 0:
        raise ValueError(f"{name} must be >= 0.")
    return value_i



def validate_microstructure_config(
    config: MicrostructureChunkConfig,
) -> MicrostructureChunkConfig:
    seed = validate_positive_int(config.seed, name="seed")
    n_chunks = validate_positive_int(config.n_chunks, name="n_chunks")
    events_per_chunk = validate_positive_int(
        config.events_per_chunk, name="events_per_chunk"
    )
    warmup_events = validate_non_negative_int(
        config.warmup_events, name="warmup_events"
    )
    n_features = validate_positive_int(config.n_features, name="n_features")
    n_regimes = validate_positive_int(config.n_regimes, name="n_regimes")
    target_horizon_events = validate_positive_int(
        config.target_horizon_events, name="target_horizon_events"
    )
    support_size = validate_positive_int(config.support_size, name="support_size")
    support_overlap_ratio = validate_real(
        config.support_overlap_ratio, name="support_overlap_ratio"
    )
    signal_scale = validate_real(config.signal_scale, name="signal_scale")
    target_noise_std = validate_real(config.target_noise_std, name="target_noise_std")
    unknown_chunk_jitter = validate_non_negative_int(
        config.unknown_chunk_jitter, name="unknown_chunk_jitter"
    )
    unknown_nonlinear_strength = validate_real(
        config.unknown_nonlinear_strength, name="unknown_nonlinear_strength"
    )
    tick_size = validate_real(config.tick_size, name="tick_size")
    base_half_spread_ticks = validate_real(
        config.base_half_spread_ticks, name="base_half_spread_ticks"
    )
    base_queue_depth = validate_real(config.base_queue_depth, name="base_queue_depth")
    queue_mean_reversion = validate_real(
        config.queue_mean_reversion, name="queue_mean_reversion"
    )
    activity_persistence = validate_real(
        config.activity_persistence, name="activity_persistence"
    )
    sign_persistence = validate_real(config.sign_persistence, name="sign_persistence")
    impact_strength = validate_real(config.impact_strength, name="impact_strength")
    spread_vol_sensitivity = validate_real(
        config.spread_vol_sensitivity, name="spread_vol_sensitivity"
    )
    cancel_rate_scale = validate_real(
        config.cancel_rate_scale, name="cancel_rate_scale"
    )
    replenish_rate_scale = validate_real(
        config.replenish_rate_scale, name="replenish_rate_scale"
    )
    student_df = validate_real(config.student_df, name="student_df")
    unknown_support_overlap_ratio = validate_real(
        config.unknown_support_overlap_ratio, name="unknown_support_overlap_ratio"
    )

    if n_chunks < 2:
        raise ValueError(
            "n_chunks must be >= 2 to reserve the final chunk for true out-of-sample."
        )
    if events_per_chunk <= target_horizon_events:
        raise ValueError("events_per_chunk must be > target_horizon_events.")
    if n_features not in SUPPORTED_MICROSTRUCTURE_FEATURE_COUNTS:
        supported = ", ".join(str(value) for value in SUPPORTED_MICROSTRUCTURE_FEATURE_COUNTS)
        raise ValueError(
            f"n_features must be one of {supported} for the fixed microstructure feature library."
        )
    if n_regimes > n_chunks:
        raise ValueError("n_regimes must be <= n_chunks.")
    if support_size >= n_features:
        raise ValueError("support_size must be < n_features.")
    if not (0.0 <= support_overlap_ratio <= 1.0):
        raise ValueError("support_overlap_ratio must be in [0, 1].")
    if signal_scale <= 0.0:
        raise ValueError("signal_scale must be > 0.")
    if target_noise_std < 0.0:
        raise ValueError("target_noise_std must be >= 0.")
    if unknown_nonlinear_strength < 0.0:
        raise ValueError("unknown_nonlinear_strength must be >= 0.")
    if tick_size <= 0.0:
        raise ValueError("tick_size must be > 0.")
    if base_half_spread_ticks <= 0.0:
        raise ValueError("base_half_spread_ticks must be > 0.")
    if base_queue_depth <= 0.0:
        raise ValueError("base_queue_depth must be > 0.")
    if not (0.0 <= queue_mean_reversion < 1.0):
        raise ValueError("queue_mean_reversion must be in [0, 1).")
    if not (0.0 <= activity_persistence < 1.0):
        raise ValueError("activity_persistence must be in [0, 1).")
    if not (0.0 <= sign_persistence < 1.0):
        raise ValueError("sign_persistence must be in [0, 1).")
    if impact_strength < 0.0:
        raise ValueError("impact_strength must be >= 0.")
    if spread_vol_sensitivity < 0.0:
        raise ValueError("spread_vol_sensitivity must be >= 0.")
    if cancel_rate_scale < 0.0:
        raise ValueError("cancel_rate_scale must be >= 0.")
    if replenish_rate_scale < 0.0:
        raise ValueError("replenish_rate_scale must be >= 0.")
    if student_df <= 2.0:
        raise ValueError("student_df must be > 2 for finite innovation variance.")
    if not (0.0 <= unknown_support_overlap_ratio <= 1.0):
        raise ValueError("unknown_support_overlap_ratio must be in [0, 1].")

    unknown_support_size = (
        None
        if config.unknown_support_size is None
        else validate_positive_int(
            config.unknown_support_size, name="unknown_support_size"
        )
    )
    if unknown_support_size is not None and unknown_support_size >= n_features:
        raise ValueError("unknown_support_size must be < n_features.")
    if unknown_support_size is not None and unknown_chunk_jitter >= unknown_support_size:
        raise ValueError("unknown_chunk_jitter must be < unknown_support_size.")

    return MicrostructureChunkConfig(
        seed=seed,
        n_chunks=n_chunks,
        events_per_chunk=events_per_chunk,
        warmup_events=warmup_events,
        n_features=n_features,
        n_regimes=n_regimes,
        target_horizon_events=target_horizon_events,
        support_size=support_size,
        support_overlap_ratio=support_overlap_ratio,
        signal_scale=signal_scale,
        target_noise_std=target_noise_std,
        unknown_support_size=unknown_support_size,
        unknown_support_overlap_ratio=unknown_support_overlap_ratio,
        unknown_chunk_jitter=unknown_chunk_jitter,
        unknown_nonlinear_strength=unknown_nonlinear_strength,
        tick_size=tick_size,
        base_half_spread_ticks=base_half_spread_ticks,
        base_queue_depth=base_queue_depth,
        queue_mean_reversion=queue_mean_reversion,
        activity_persistence=activity_persistence,
        sign_persistence=sign_persistence,
        impact_strength=impact_strength,
        spread_vol_sensitivity=spread_vol_sensitivity,
        cancel_rate_scale=cancel_rate_scale,
        replenish_rate_scale=replenish_rate_scale,
        student_df=student_df,
    )



def assign_regimes(n_chunks: int, n_regimes: int) -> tuple[int, ...]:
    bounds = np.linspace(0, n_chunks, num=n_regimes + 1, dtype=int)
    regime_by_chunk: list[int] = []
    for regime in range(n_regimes):
        count = int(bounds[regime + 1] - bounds[regime])
        regime_by_chunk.extend([regime] * count)
    return tuple(regime_by_chunk)



def build_event_regimes(
    config: MicrostructureChunkConfig,
) -> tuple[tuple[int, ...], np.ndarray, int, int, int]:
    lookback = 40
    usable_rows = config.n_chunks * config.events_per_chunk
    total_events = (
        lookback
        + config.warmup_events
        + usable_rows
        + config.target_horizon_events
    )
    regime_by_chunk = assign_regimes(config.n_chunks, config.n_regimes)
    chunk_rows = np.repeat(np.asarray(regime_by_chunk, dtype=int), config.events_per_chunk)
    prefix = np.full(lookback + config.warmup_events, regime_by_chunk[0], dtype=int)
    suffix = np.full(config.target_horizon_events, regime_by_chunk[-1], dtype=int)
    regime_by_event = np.concatenate((prefix, chunk_rows, suffix))
    return regime_by_chunk, regime_by_event, lookback, usable_rows, total_events



def student_unit_variance(df: float, rng: np.random.Generator) -> float:
    sample = float(rng.standard_t(df=df))
    return sample * float(np.sqrt((df - 2.0) / df))



def sigmoid(x: float) -> float:
    x_clip = float(np.clip(x, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-x_clip)))



def lag(arr: np.ndarray, steps: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=float)
    if steps <= 0:
        out[:] = np.asarray(arr, dtype=float)
        return out
    out[steps:] = np.asarray(arr[:-steps], dtype=float)
    return out



def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if window <= 0:
        raise ValueError("window must be > 0.")
    n = arr.shape[0]
    if n == 0:
        return np.asarray([], dtype=float)
    csum = np.concatenate(([0.0], np.cumsum(arr, dtype=float)))
    full = (csum[window:] - csum[:-window]) / float(window)
    if window == 1:
        return full
    prefix = csum[1:window] / np.arange(1, window, dtype=float)
    return np.concatenate((prefix, full))



def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    mean = rolling_mean(arr, window)
    mean_sq = rolling_mean(np.asarray(arr, dtype=float) ** 2, window)
    var = np.maximum(mean_sq - mean**2, 0.0)
    return np.sqrt(var)



def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape[0] != right.shape[0] or left.shape[0] < 2:
        return 0.0
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])



def trade_sign_autocorr(signs: np.ndarray) -> float:
    trade_signs = np.asarray(signs, dtype=float)
    trade_signs = trade_signs[np.abs(trade_signs) > 0.0]
    if trade_signs.shape[0] < 2:
        return 0.0
    if np.all(trade_signs == trade_signs[0]):
        return 1.0
    return safe_corr(trade_signs[1:], trade_signs[:-1])


__all__ = [
    "CANONICAL_MICROSTRUCTURE_FEATURE_COUNT",
    "SUPPORTED_MICROSTRUCTURE_FEATURE_COUNTS",
    "assign_regimes",
    "build_event_regimes",
    "lag",
    "rolling_mean",
    "rolling_std",
    "safe_corr",
    "sigmoid",
    "supported_microstructure_feature_counts",
    "student_unit_variance",
    "trade_sign_autocorr",
    "validate_microstructure_config",
]
