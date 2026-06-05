"""Market-like nonstationary generators that emit sequential GramData chunks."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from selection import GramData

from .chunk_dataset_io import load_chunk_dataset, save_chunk_dataset


@dataclass(frozen=True)
class MarketChunkConfig:
    """Configuration for chunked market-style synthetic generators."""

    seed: int = 20260307
    n_chunks: int = 11
    bars_per_chunk: int = 2500
    warmup_bars: int = 250
    n_features: int = 64
    n_regimes: int = 5

    # Known-support flavor controls.
    support_size: int = 8
    support_overlap_ratio: float = 0.5
    signal_scale: float = 0.22
    target_noise_std: float = 0.01

    # Unknown-support flavor controls.
    unknown_support_size: int | None = None
    unknown_support_overlap_ratio: float = 0.35
    unknown_chunk_jitter: int = 2
    unknown_nonlinear_strength: float = 0.12

    # Base market dynamics.
    student_df: float = 7.0
    base_vol: float = 0.006
    garch_omega: float = 1e-6
    garch_alpha: float = 0.06
    garch_beta: float = 0.92
    mean_reversion: float = 0.97


@dataclass(frozen=True)
class MarketChunkDataset:
    """Output container for chunked market experiments."""

    gram_chunks: list[GramData]
    feature_names: tuple[str, ...]
    chunk_ranges: tuple[tuple[int, int], ...]
    regime_by_chunk: tuple[int, ...]
    support_by_chunk: tuple[tuple[int, ...], ...] | None
    beta_by_chunk: tuple[np.ndarray, ...] | None
    meta: dict[str, object]


def _validate_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"{name} must be finite.")
    return value_f


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value_i = int(value)
    if value_i <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value_i


def _validate_non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value_i = int(value)
    if value_i < 0:
        raise ValueError(f"{name} must be >= 0.")
    return value_i


def _validate_config(config: MarketChunkConfig) -> MarketChunkConfig:
    seed = _validate_positive_int(config.seed, name="seed")
    n_chunks = _validate_positive_int(config.n_chunks, name="n_chunks")
    bars_per_chunk = _validate_positive_int(
        config.bars_per_chunk, name="bars_per_chunk"
    )
    warmup_bars = _validate_non_negative_int(config.warmup_bars, name="warmup_bars")
    n_features = _validate_positive_int(config.n_features, name="n_features")
    n_regimes = _validate_positive_int(config.n_regimes, name="n_regimes")
    support_size = _validate_positive_int(config.support_size, name="support_size")
    support_overlap_ratio = _validate_real(
        config.support_overlap_ratio, name="support_overlap_ratio"
    )
    signal_scale = _validate_real(config.signal_scale, name="signal_scale")
    target_noise_std = _validate_real(config.target_noise_std, name="target_noise_std")
    unknown_chunk_jitter = _validate_non_negative_int(
        config.unknown_chunk_jitter, name="unknown_chunk_jitter"
    )
    unknown_nonlinear_strength = _validate_real(
        config.unknown_nonlinear_strength, name="unknown_nonlinear_strength"
    )

    student_df = _validate_real(config.student_df, name="student_df")
    base_vol = _validate_real(config.base_vol, name="base_vol")
    garch_omega = _validate_real(config.garch_omega, name="garch_omega")
    garch_alpha = _validate_real(config.garch_alpha, name="garch_alpha")
    garch_beta = _validate_real(config.garch_beta, name="garch_beta")
    mean_reversion = _validate_real(config.mean_reversion, name="mean_reversion")
    unknown_support_overlap_ratio = _validate_real(
        config.unknown_support_overlap_ratio, name="unknown_support_overlap_ratio"
    )

    if n_chunks < 2:
        raise ValueError(
            "n_chunks must be >= 2 to reserve the final chunk for true out-of-sample."
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
    if student_df <= 2.0:
        raise ValueError("student_df must be > 2 for finite innovation variance.")
    if base_vol <= 0.0:
        raise ValueError("base_vol must be > 0.")
    if garch_omega <= 0.0:
        raise ValueError("garch_omega must be > 0.")
    if garch_alpha < 0.0 or garch_beta < 0.0:
        raise ValueError("garch_alpha and garch_beta must be >= 0.")
    if garch_alpha + garch_beta >= 1.0:
        raise ValueError("garch_alpha + garch_beta must be < 1 for stability.")
    if not (-1.0 < mean_reversion < 1.0):
        raise ValueError("mean_reversion must be in (-1, 1).")
    if not (0.0 <= unknown_support_overlap_ratio <= 1.0):
        raise ValueError("unknown_support_overlap_ratio must be in [0, 1].")

    unknown_support_size = (
        None
        if config.unknown_support_size is None
        else _validate_positive_int(
            config.unknown_support_size, name="unknown_support_size"
        )
    )
    if unknown_support_size is not None and unknown_support_size >= n_features:
        raise ValueError("unknown_support_size must be < n_features.")
    if unknown_support_size is not None and unknown_chunk_jitter >= unknown_support_size:
        raise ValueError("unknown_chunk_jitter must be < unknown_support_size.")

    return MarketChunkConfig(
        seed=seed,
        n_chunks=n_chunks,
        bars_per_chunk=bars_per_chunk,
        warmup_bars=warmup_bars,
        n_features=n_features,
        n_regimes=n_regimes,
        support_size=support_size,
        support_overlap_ratio=support_overlap_ratio,
        signal_scale=signal_scale,
        target_noise_std=target_noise_std,
        unknown_support_size=unknown_support_size,
        unknown_support_overlap_ratio=unknown_support_overlap_ratio,
        unknown_chunk_jitter=unknown_chunk_jitter,
        unknown_nonlinear_strength=unknown_nonlinear_strength,
        student_df=student_df,
        base_vol=base_vol,
        garch_omega=garch_omega,
        garch_alpha=garch_alpha,
        garch_beta=garch_beta,
        mean_reversion=mean_reversion,
    )


def _simulate_market_paths(
    config: MarketChunkConfig, *, n_bars_total: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Heavy-tailed innovations normalized to unit variance.
    innovations = rng.standard_t(df=config.student_df, size=n_bars_total)
    innovations *= np.sqrt((config.student_df - 2.0) / config.student_df)

    returns = np.zeros(n_bars_total, dtype=float)
    factor_fast = np.zeros(n_bars_total, dtype=float)
    factor_slow = np.zeros(n_bars_total, dtype=float)
    log_volume = np.zeros(n_bars_total, dtype=float)
    eps = np.zeros(n_bars_total, dtype=float)
    sigma2 = np.full(n_bars_total, config.base_vol**2, dtype=float)
    drift = np.zeros(n_bars_total, dtype=float)

    for t in range(1, n_bars_total):
        sigma2[t] = (
            config.garch_omega
            + config.garch_alpha * (eps[t - 1] ** 2)
            + config.garch_beta * sigma2[t - 1]
        )
        sigma = float(np.sqrt(max(sigma2[t], 1e-12)))
        factor_fast[t] = 0.85 * factor_fast[t - 1] + 0.015 * rng.standard_normal()
        factor_slow[t] = 0.97 * factor_slow[t - 1] + 0.008 * rng.standard_normal()
        drift[t] = config.mean_reversion * drift[t - 1] + 0.0015 * rng.standard_normal()
        eps[t] = sigma * innovations[t]
        returns[t] = drift[t] + 0.08 * factor_fast[t] + 0.05 * factor_slow[t] + eps[t]
        log_volume[t] = (
            0.92 * log_volume[t - 1]
            + 0.35 * abs(returns[t - 1])
            + 0.12 * rng.standard_normal()
        )

    volume = np.exp(log_volume - np.mean(log_volume))
    return returns, factor_fast, factor_slow, volume


def _rolling_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr))


def _rolling_std(arr: np.ndarray) -> float:
    return float(np.std(arr))


def _build_feature_matrix(
    config: MarketChunkConfig,
    *,
    returns: np.ndarray,
    factor_fast: np.ndarray,
    factor_slow: np.ndarray,
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    lag_features = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 40, 48]
    mean_windows = [3, 5, 8, 10, 16, 20, 32, 40]
    vol_windows = [5, 8, 10, 16, 20, 32, 40]
    spread_pairs = [(3, 10), (5, 20), (8, 32), (10, 40), (16, 40)]
    abs_windows = [5, 10, 20, 40]
    volume_lags = [1, 2, 3, 5, 8, 13]
    zscore_windows = [10, 20, 40]
    factor_lags = [1, 2, 3, 5, 8, 13]

    lookback = max(
        max(lag_features),
        max(mean_windows),
        max(vol_windows),
        max(abs_windows),
        max(volume_lags),
        max(zscore_windows),
        max(factor_lags),
    )
    valid_start = lookback
    valid_stop = returns.shape[0] - 1  # y(t) = r(t + 1)
    n_rows = valid_stop - valid_start
    if n_rows < config.warmup_bars + config.n_chunks * config.bars_per_chunk:
        raise ValueError("Insufficient bars to satisfy warmup and chunk lengths.")

    names: list[str] = []
    names.extend([f"ret_lag_{lag}" for lag in lag_features])
    names.extend([f"ret_mean_{w}" for w in mean_windows])
    names.extend([f"ret_spread_{s}_{l}" for s, l in spread_pairs])
    names.extend([f"ret_vol_{w}" for w in vol_windows])
    names.extend([f"ret_abs_mean_{w}" for w in abs_windows])
    names.extend([f"ret_updown_{w}" for w in abs_windows])
    names.extend([f"volume_lag_{lag}" for lag in volume_lags])
    names.extend([f"volume_z_{w}" for w in zscore_windows])
    names.extend([f"fast_lag_{lag}" for lag in factor_lags])
    names.extend([f"slow_lag_{lag}" for lag in factor_lags])
    names.extend(
        [
            "int_ret1_x_volume1",
            "int_ret1_x_fast1",
            "int_ret1_x_slow1",
            "int_fast1_x_slow1",
            "int_mom5_x_vol20",
            "int_vol10_x_volume1",
            "int_abs20_x_fast1",
            "shape_signed_sqrt_ret1",
        ]
    )

    if config.n_features > len(names):
        raise ValueError(
            f"Requested n_features={config.n_features} exceeds available "
            f"candidate features={len(names)}."
        )

    X_full = np.empty((n_rows, len(names)), dtype=float)
    y_next = np.empty(n_rows, dtype=float)
    for row_idx, t in enumerate(range(valid_start, valid_stop)):
        cursor = 0
        for lag in lag_features:
            X_full[row_idx, cursor] = returns[t - lag]
            cursor += 1

        for window in mean_windows:
            X_full[row_idx, cursor] = _rolling_mean(returns[t - window + 1 : t + 1])
            cursor += 1

        for short_w, long_w in spread_pairs:
            short_mean = _rolling_mean(returns[t - short_w + 1 : t + 1])
            long_mean = _rolling_mean(returns[t - long_w + 1 : t + 1])
            X_full[row_idx, cursor] = short_mean - long_mean
            cursor += 1

        for window in vol_windows:
            X_full[row_idx, cursor] = _rolling_std(returns[t - window + 1 : t + 1])
            cursor += 1

        for window in abs_windows:
            X_full[row_idx, cursor] = _rolling_mean(
                np.abs(returns[t - window + 1 : t + 1])
            )
            cursor += 1

        for window in abs_windows:
            wins = returns[t - window + 1 : t + 1]
            up = _rolling_mean(np.clip(wins, 0.0, None))
            down = _rolling_mean(np.clip(-wins, 0.0, None))
            X_full[row_idx, cursor] = up - down
            cursor += 1

        for lag in volume_lags:
            X_full[row_idx, cursor] = volume[t - lag]
            cursor += 1

        for window in zscore_windows:
            wins = volume[t - window + 1 : t + 1]
            X_full[row_idx, cursor] = (volume[t] - _rolling_mean(wins)) / (
                _rolling_std(wins) + 1e-12
            )
            cursor += 1

        for lag in factor_lags:
            X_full[row_idx, cursor] = factor_fast[t - lag]
            cursor += 1

        for lag in factor_lags:
            X_full[row_idx, cursor] = factor_slow[t - lag]
            cursor += 1

        ret1 = returns[t - 1]
        mom5 = _rolling_mean(returns[t - 5 + 1 : t + 1])
        vol20 = _rolling_std(returns[t - 20 + 1 : t + 1])
        vol10 = _rolling_std(returns[t - 10 + 1 : t + 1])
        abs20 = _rolling_mean(np.abs(returns[t - 20 + 1 : t + 1]))
        fast1 = factor_fast[t - 1]
        slow1 = factor_slow[t - 1]
        vol1 = volume[t - 1]
        X_full[row_idx, cursor] = ret1 * vol1
        cursor += 1
        X_full[row_idx, cursor] = ret1 * fast1
        cursor += 1
        X_full[row_idx, cursor] = ret1 * slow1
        cursor += 1
        X_full[row_idx, cursor] = fast1 * slow1
        cursor += 1
        X_full[row_idx, cursor] = mom5 * vol20
        cursor += 1
        X_full[row_idx, cursor] = vol10 * vol1
        cursor += 1
        X_full[row_idx, cursor] = abs20 * fast1
        cursor += 1
        X_full[row_idx, cursor] = np.sign(ret1) * np.sqrt(abs(ret1) + 1e-12)
        cursor += 1
        if cursor != len(names):
            raise RuntimeError("Feature row construction produced inconsistent length.")

        y_next[row_idx] = returns[t + 1]

    keep_start = config.warmup_bars
    keep_stop = keep_start + config.n_chunks * config.bars_per_chunk
    X = X_full[keep_start:keep_stop, : config.n_features].copy()
    y = y_next[keep_start:keep_stop].copy()
    selected_names = tuple(names[: config.n_features])
    return X, y, selected_names


def _assign_regimes(n_chunks: int, n_regimes: int) -> tuple[int, ...]:
    # Deterministic contiguous regime blocks.
    mapped = np.floor(np.arange(n_chunks) * (n_regimes / n_chunks)).astype(int)
    return tuple(int(v) for v in mapped.tolist())


def _build_regime_truth(
    *,
    n_features: int,
    n_regimes: int,
    support_size: int,
    overlap_ratio: float,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[int, ...], ...], tuple[np.ndarray, ...]]:
    supports: list[tuple[int, ...]] = []
    betas: list[np.ndarray] = []
    universe = np.arange(n_features, dtype=int)

    for regime_idx in range(n_regimes):
        if regime_idx == 0:
            support = np.sort(rng.choice(universe, size=support_size, replace=False))
        else:
            prev = np.array(supports[-1], dtype=int)
            keep = int(round(overlap_ratio * support_size))
            keep = min(max(0, keep), support_size)
            kept = (
                np.sort(rng.choice(prev, size=keep, replace=False))
                if keep > 0
                else np.array([], dtype=int)
            )
            inactive = np.array([idx for idx in universe if idx not in set(kept)], dtype=int)
            add_count = support_size - kept.size
            added = np.sort(rng.choice(inactive, size=add_count, replace=False))
            support = np.sort(np.concatenate([kept, added]))

        coeff = rng.standard_normal(support_size)
        coeff = np.sign(coeff) * np.maximum(np.abs(coeff), 0.35)
        beta = np.zeros(n_features, dtype=float)
        beta[support] = coeff
        supports.append(tuple(int(i) for i in support.tolist()))
        betas.append(beta)

    return tuple(supports), tuple(betas)


def _build_unknown_chunk_truth(
    *,
    config: MarketChunkConfig,
    regime_by_chunk: tuple[int, ...],
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

    regime_supports, regime_betas = _build_regime_truth(
        n_features=config.n_features,
        n_regimes=config.n_regimes,
        support_size=support_size,
        overlap_ratio=config.unknown_support_overlap_ratio,
        rng=rng,
    )

    supports_by_chunk: list[tuple[int, ...]] = []
    betas_by_chunk: list[np.ndarray] = []
    full_index = np.arange(config.n_features, dtype=int)
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
        for feat_idx in support:
            coeff = float(base_beta[feat_idx])
            if feat_idx not in base_support_set:
                coeff = 0.9 * rng.standard_normal()
            coeff *= 1.0 + 0.15 * rng.standard_normal()
            if abs(coeff) < 0.05:
                coeff = 0.05 if coeff >= 0.0 else -0.05
            beta[feat_idx] = coeff
        supports_by_chunk.append(support)
        betas_by_chunk.append(beta)

    return tuple(supports_by_chunk), tuple(betas_by_chunk)


def _build_chunked_gram_dataset(
    *,
    X: np.ndarray,
    y: np.ndarray,
    regime_by_chunk: tuple[int, ...],
    support_by_chunk: tuple[tuple[int, ...], ...] | None,
    beta_by_chunk: tuple[np.ndarray, ...] | None,
    feature_names: tuple[str, ...],
    config: MarketChunkConfig,
    flavor: str,
    signal_component: np.ndarray,
) -> MarketChunkDataset:
    gram_chunks: list[GramData] = []
    chunk_ranges: list[tuple[int, int]] = []
    feature_mean_max: list[float] = []
    target_mean_abs: list[float] = []
    chunk_target_std: list[float] = []

    for chunk_idx in range(config.n_chunks):
        start = chunk_idx * config.bars_per_chunk
        end = start + config.bars_per_chunk
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
                n_samples=int(config.bars_per_chunk),
                warn_if_uncentered=False,
            )
        )

    y_centered = y - float(np.mean(y))
    y_var = float(np.var(y_centered))
    signal_var = float(np.var(signal_component))
    denom = max(y_var - signal_var, 1e-12)
    abs_returns = np.abs(y_centered)
    if abs_returns.size >= 2 and np.std(abs_returns[:-1]) > 0 and np.std(abs_returns[1:]) > 0:
        abs_ret_autocorr = float(np.corrcoef(abs_returns[1:], abs_returns[:-1])[0, 1])
    else:
        abs_ret_autocorr = 0.0
    variance = float(np.mean(y_centered**2))
    if variance <= 0.0:
        kurtosis = 0.0
    else:
        kurtosis = float(np.mean(y_centered**4) / (variance**2))

    meta: dict[str, object] = {
        "flavor": flavor,
        "n_chunks": config.n_chunks,
        "bars_per_chunk": config.bars_per_chunk,
        "n_features": config.n_features,
        "n_regimes": config.n_regimes,
        "train_chunk_count_recommended": config.n_chunks - 1,
        "oos_chunk_index_recommended": config.n_chunks - 1,
        "signal_to_noise_ratio_est": signal_var / denom,
        "target_kurtosis": kurtosis,
        "abs_target_autocorr_lag1": abs_ret_autocorr,
        "chunk_target_std": chunk_target_std,
        "max_abs_centered_feature_mean_per_chunk": feature_mean_max,
        "max_abs_centered_target_mean_per_chunk": target_mean_abs,
    }

    return MarketChunkDataset(
        gram_chunks=gram_chunks,
        feature_names=feature_names,
        chunk_ranges=tuple(chunk_ranges),
        regime_by_chunk=regime_by_chunk,
        support_by_chunk=support_by_chunk,
        beta_by_chunk=beta_by_chunk,
        meta=meta,
    )


def generate_market_gram_chunks_known(config: MarketChunkConfig) -> MarketChunkDataset:
    """Generate sequential GramData chunks with known rotating support."""
    config = _validate_config(config)
    rng = np.random.default_rng(config.seed)
    lookback = 48
    n_rows_needed = config.warmup_bars + config.n_chunks * config.bars_per_chunk
    n_bars_total = lookback + n_rows_needed + 1

    returns, factor_fast, factor_slow, volume = _simulate_market_paths(
        config, n_bars_total=n_bars_total, rng=rng
    )
    X, y_base, feature_names = _build_feature_matrix(
        config,
        returns=returns,
        factor_fast=factor_fast,
        factor_slow=factor_slow,
        volume=volume,
    )
    X_scaled = (X - np.mean(X, axis=0, keepdims=True)) / (
        np.std(X, axis=0, keepdims=True) + 1e-12
    )

    regime_by_chunk = _assign_regimes(config.n_chunks, config.n_regimes)
    regime_supports, regime_betas = _build_regime_truth(
        n_features=config.n_features,
        n_regimes=config.n_regimes,
        support_size=config.support_size,
        overlap_ratio=config.support_overlap_ratio,
        rng=np.random.default_rng(config.seed + 41),
    )
    support_by_chunk = tuple(regime_supports[reg] for reg in regime_by_chunk)
    beta_by_chunk = tuple(regime_betas[reg].copy() for reg in regime_by_chunk)

    row_regimes = np.repeat(np.array(regime_by_chunk, dtype=int), config.bars_per_chunk)
    beta_rows = np.stack([regime_betas[int(reg)] for reg in row_regimes], axis=0)
    signal = np.einsum("ij,ij->i", X_scaled, beta_rows)
    y = y_base + config.signal_scale * signal + config.target_noise_std * rng.standard_normal(
        y_base.shape[0]
    )

    return _build_chunked_gram_dataset(
        X=X,
        y=y,
        regime_by_chunk=regime_by_chunk,
        support_by_chunk=support_by_chunk,
        beta_by_chunk=beta_by_chunk,
        feature_names=feature_names,
        config=config,
        flavor="known_support",
        signal_component=config.signal_scale * signal,
    )


def generate_market_gram_chunks_unknown(
    config: MarketChunkConfig,
    *,
    expose_truth: bool = False,
) -> MarketChunkDataset:
    """Generate harder chunks; reserve the final chunk for true out-of-sample."""
    config = _validate_config(config)
    rng = np.random.default_rng(config.seed)
    lookback = 48
    n_rows_needed = config.warmup_bars + config.n_chunks * config.bars_per_chunk
    n_bars_total = lookback + n_rows_needed + 1

    returns, factor_fast, factor_slow, volume = _simulate_market_paths(
        config, n_bars_total=n_bars_total, rng=rng
    )
    X, y_base, feature_names = _build_feature_matrix(
        config,
        returns=returns,
        factor_fast=factor_fast,
        factor_slow=factor_slow,
        volume=volume,
    )
    X_scaled = (X - np.mean(X, axis=0, keepdims=True)) / (
        np.std(X, axis=0, keepdims=True) + 1e-12
    )

    regime_by_chunk = _assign_regimes(config.n_chunks, config.n_regimes)
    support_by_chunk, beta_by_chunk = _build_unknown_chunk_truth(
        config=config,
        regime_by_chunk=regime_by_chunk,
        rng=np.random.default_rng(config.seed + 83),
    )

    beta_rows = np.repeat(np.stack(beta_by_chunk, axis=0), config.bars_per_chunk, axis=0)
    signal = np.einsum("ij,ij->i", X_scaled, beta_rows)

    interaction_seed = np.random.default_rng(config.seed + 109)
    interaction_idx = interaction_seed.choice(
        np.arange(config.n_features, dtype=int), size=3, replace=False
    )
    i0, i1, i2 = (int(v) for v in interaction_idx.tolist())
    nonlinear = config.unknown_nonlinear_strength * (
        0.6 * (X_scaled[:, i0] ** 2 - 1.0) + 0.4 * (X_scaled[:, i1] * X_scaled[:, i2])
    )
    y = (
        y_base
        + config.signal_scale * signal
        + nonlinear
        + (1.5 * config.target_noise_std) * rng.standard_normal(y_base.shape[0])
    )

    if expose_truth:
        support_out: tuple[tuple[int, ...], ...] | None = support_by_chunk
        beta_out: tuple[np.ndarray, ...] | None = beta_by_chunk
    else:
        support_out = None
        beta_out = None

    return _build_chunked_gram_dataset(
        X=X,
        y=y,
        regime_by_chunk=regime_by_chunk,
        support_by_chunk=support_out,
        beta_by_chunk=beta_out,
        feature_names=feature_names,
        config=config,
        flavor="unknown_support",
        signal_component=config.signal_scale * signal + nonlinear,
    )


def save_market_chunk_dataset(dataset: MarketChunkDataset, path: str | Path) -> None:
    """Persist a MarketChunkDataset to a compressed NPZ file."""
    save_chunk_dataset(dataset, path)


def load_market_chunk_dataset(path: str | Path) -> MarketChunkDataset:
    """Load a MarketChunkDataset previously stored with save_market_chunk_dataset."""
    return load_chunk_dataset(
        path,
        dataset_factory=MarketChunkDataset,
        invalid_version_message="Unsupported market chunk dataset version.",
    )


__all__ = [
    "MarketChunkConfig",
    "MarketChunkDataset",
    "generate_market_gram_chunks_known",
    "generate_market_gram_chunks_unknown",
    "save_market_chunk_dataset",
    "load_market_chunk_dataset",
]
