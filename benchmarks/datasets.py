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


def _split_indices(n_samples: int, train_fraction: float, val_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def build_synthetic_linear_dataset(config: dict) -> BenchmarkDataset:
    """Build a synthetic linear regression dataset with fixed train/val/test splits."""
    seed = int(config["seed"])
    n_samples = int(config["n_samples"])
    n_features = int(config["n_features"])
    support_size = int(config.get("support_size", min(8, n_features)))
    noise_std = float(config.get("noise_std", 0.2))
    signal_scale = float(config.get("signal_scale", 2.0))
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

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))

    beta_true = np.zeros(n_features, dtype=float)
    true_support = np.sort(rng.choice(n_features, size=support_size, replace=False))
    beta_true[true_support] = signal_scale * rng.standard_normal(support_size)

    y = X @ beta_true + noise_std * rng.standard_normal(n_samples)

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


def build_dataset(config: dict) -> BenchmarkDataset:
    """Dispatch benchmark dataset construction from spec config."""
    kind = str(config.get("kind", ""))
    if kind == "synthetic_linear":
        return build_synthetic_linear_dataset(config)
    raise ValueError(f"Unsupported dataset kind: {kind!r}")
