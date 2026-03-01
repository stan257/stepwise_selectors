"""Deterministic dataset builders for benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from selection.definitions import GramData


@dataclass(frozen=True)
class BenchmarkDataset:
    """Container for benchmark splits and generation metadata."""

    name: str
    seed: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    true_support: np.ndarray
    train_data: GramData


def _to_gram_data(X: np.ndarray, y: np.ndarray) -> GramData:
    return GramData(
        gram=np.ascontiguousarray(X.T @ X),
        cov=np.ascontiguousarray(X.T @ y),
        y_norm=float(y @ y),
        n_samples=int(X.shape[0]),
    )


def _validate_fraction(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float in (0, 1).")
    value = float(value)
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be in (0, 1).")
    return value


def _validate_optional_seed(value, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None.")
    return int(value)


def _validate_nonnegative_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")
    return value


def _split_indices(
    n_samples: int, train_fraction: float, val_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(np.floor(train_fraction * n_samples))
    n_val = int(np.floor(val_fraction * n_samples))
    n_test = n_samples - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            "Invalid split: train/val/test must each contain at least one sample."
        )
    train = np.arange(0, n_train, dtype=int)
    val = np.arange(n_train, n_train + n_val, dtype=int)
    test = np.arange(n_train + n_val, n_samples, dtype=int)
    return train, val, test


def _toeplitz_cholesky(n_features: int, correlation: float) -> np.ndarray:
    if not (0.0 <= correlation < 1.0):
        raise ValueError("correlation must be in [0, 1).")
    if correlation == 0.0:
        return np.eye(n_features)
    offsets = np.abs(
        np.subtract.outer(np.arange(n_features), np.arange(n_features))
    )
    cov = correlation ** offsets
    cov += 1e-12 * np.eye(n_features)
    return np.linalg.cholesky(cov)


def _sample_true_support(
    *,
    n_features: int,
    support_size: int,
    seed: int,
    support_seed: int | None,
    clustered_support: bool,
) -> np.ndarray:
    base_seed = seed if support_seed is None else support_seed
    rng = np.random.default_rng(base_seed)
    if clustered_support:
        start = int(rng.integers(0, n_features - support_size + 1))
        return np.arange(start, start + support_size, dtype=int)
    return np.sort(rng.choice(n_features, size=support_size, replace=False))


def _enforce_min_abs(values: np.ndarray, min_abs: float) -> np.ndarray:
    if min_abs <= 0.0:
        return values
    signs = np.sign(values)
    signs[signs == 0.0] = 1.0
    magnitudes = np.maximum(np.abs(values), min_abs)
    return signs * magnitudes


def _apply_twin_decoys(
    *,
    X: np.ndarray,
    true_support: np.ndarray,
    rng: np.random.Generator,
    twin_decoys_per_signal: int,
    twin_strength: float,
    twin_noise_std: float,
) -> None:
    if twin_decoys_per_signal <= 0:
        return

    n_features = X.shape[1]
    support_set = set(int(i) for i in true_support.tolist())
    available = [
        idx
        for idx in range(n_features)
        if idx not in support_set
    ]
    needed = int(twin_decoys_per_signal) * int(true_support.size)
    if needed > len(available):
        raise ValueError(
            "Not enough non-support features to allocate requested twin decoys."
        )

    cursor = 0
    n_samples = X.shape[0]
    for support_idx in true_support.tolist():
        source = X[:, int(support_idx)]
        for _ in range(twin_decoys_per_signal):
            dst = available[cursor]
            cursor += 1
            X[:, dst] = (
                twin_strength * source + twin_noise_std * rng.standard_normal(n_samples)
            )


def _nonlinear_component(X: np.ndarray, support: np.ndarray) -> np.ndarray:
    if support.size == 0:
        return np.zeros(X.shape[0], dtype=float)
    terms = np.zeros(X.shape[0], dtype=float)
    s0 = int(support[0])
    terms += 0.5 * (X[:, s0] ** 2 - 1.0)
    if support.size >= 2:
        s1 = int(support[1])
        terms += X[:, s0] * X[:, s1]
    if support.size >= 3:
        s2 = int(support[2])
        terms += 0.5 * (X[:, s2] ** 2 - 1.0)
    return terms


def _build_synthetic_dataset(config: dict, *, correlation: float) -> BenchmarkDataset:
    seed = int(config["seed"])
    n_samples = int(config["n_samples"])
    n_features = int(config["n_features"])
    support_size = int(config.get("support_size", min(8, n_features)))
    noise_std = float(config.get("noise_std", 0.2))
    signal_scale = float(config.get("signal_scale", 2.0))
    min_signal_abs = float(config.get("min_signal_abs", 0.0))
    clustered_support = bool(config.get("clustered_support", False))
    support_seed = _validate_optional_seed(config.get("support_seed"), name="support_seed")
    twin_decoys_per_signal = _validate_nonnegative_int(
        config.get("twin_decoys_per_signal", 0), name="twin_decoys_per_signal"
    )
    twin_strength = float(config.get("twin_strength", 0.98))
    twin_noise_std = float(config.get("twin_noise_std", 0.1))
    nonlinear_strength = float(config.get("nonlinear_strength", 0.0))

    train_fraction = _validate_fraction(
        config.get("train_fraction", 0.6), name="train_fraction"
    )
    val_fraction = _validate_fraction(config.get("val_fraction", 0.2), name="val_fraction")

    if n_samples < 3:
        raise ValueError("n_samples must be >= 3.")
    if n_features <= 0:
        raise ValueError("n_features must be > 0.")
    if support_size <= 0 or support_size > n_features:
        raise ValueError("support_size must be in [1, n_features].")
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0.")
    if min_signal_abs < 0:
        raise ValueError("min_signal_abs must be >= 0.")
    if not (0.0 <= twin_strength <= 1.0):
        raise ValueError("twin_strength must be in [0, 1].")
    if twin_noise_std < 0:
        raise ValueError("twin_noise_std must be >= 0.")
    if nonlinear_strength < 0:
        raise ValueError("nonlinear_strength must be >= 0.")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_samples, n_features))
    transform = _toeplitz_cholesky(n_features, correlation)
    X = z @ transform.T

    true_support = _sample_true_support(
        n_features=n_features,
        support_size=support_size,
        seed=seed,
        support_seed=support_seed,
        clustered_support=clustered_support,
    )

    beta_true = np.zeros(n_features, dtype=float)
    support_rng = np.random.default_rng(seed + 13)
    beta_support = signal_scale * support_rng.standard_normal(support_size)
    beta_support = _enforce_min_abs(beta_support, min_signal_abs)
    beta_true[true_support] = beta_support

    _apply_twin_decoys(
        X=X,
        true_support=true_support,
        rng=np.random.default_rng(seed + 29),
        twin_decoys_per_signal=twin_decoys_per_signal,
        twin_strength=twin_strength,
        twin_noise_std=twin_noise_std,
    )

    y = X @ beta_true + noise_std * rng.standard_normal(n_samples)
    if nonlinear_strength > 0.0:
        y = y + nonlinear_strength * _nonlinear_component(X, true_support)

    split_rng = np.random.default_rng(seed + 1)
    permutation = split_rng.permutation(n_samples)
    X = X[permutation]
    y = y[permutation]

    train_idx, val_idx, test_idx = _split_indices(n_samples, train_fraction, val_fraction)
    X_train = np.ascontiguousarray(X[train_idx])
    y_train = np.ascontiguousarray(y[train_idx])
    X_val = np.ascontiguousarray(X[val_idx])
    y_val = np.ascontiguousarray(y[val_idx])
    X_test = np.ascontiguousarray(X[test_idx])
    y_test = np.ascontiguousarray(y[test_idx])

    return BenchmarkDataset(
        name=str(config.get("name", "synthetic_linear")),
        seed=seed,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        true_support=true_support,
        train_data=_to_gram_data(X_train, y_train),
    )


def build_synthetic_linear_dataset(config: dict) -> BenchmarkDataset:
    """Build a synthetic linear regression dataset with i.i.d. features."""
    return _build_synthetic_dataset(config, correlation=0.0)


def build_synthetic_support_recovery_dataset(config: dict) -> BenchmarkDataset:
    """Build correlated synthetic data intended for support-recovery stress tests."""
    correlation = float(config.get("correlation", 0.7))
    return _build_synthetic_dataset(config, correlation=correlation)


def build_dataset(config: dict) -> BenchmarkDataset:
    """Dispatch benchmark dataset construction from config."""
    kind = str(config.get("kind", ""))
    match kind:
        case "synthetic_linear":
            return build_synthetic_linear_dataset(config)
        case "synthetic_support_recovery":
            return build_synthetic_support_recovery_dataset(config)
        case _:
            raise ValueError(f"Unsupported dataset kind: {kind!r}")
