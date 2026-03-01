"""Curated synthetic benchmark scenarios for support recovery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticScenario:
    """One reproducible synthetic scenario configuration."""

    name: str
    difficulty: str
    description: str
    dataset: dict


def progressive_support_recovery_scenarios() -> list[SyntheticScenario]:
    """Return progressively harder support-recovery scenarios.

    Scenarios keep true support fixed via `support_seed` while varying sample
    noise and feature draws across run seeds.
    """
    return [
        SyntheticScenario(
            name="easy_independent",
            difficulty="easy",
            description="Well-specified linear model with low noise and independent features.",
            dataset={
                "kind": "synthetic_support_recovery",
                "name": "easy_independent",
                "n_samples": 480,
                "n_features": 48,
                "support_size": 6,
                "noise_std": 0.25,
                "signal_scale": 3.0,
                "min_signal_abs": 0.8,
                "correlation": 0.0,
                "clustered_support": False,
                "support_seed": 31415,
                "train_fraction": 0.6,
                "val_fraction": 0.2,
            },
        ),
        SyntheticScenario(
            name="medium_correlated",
            difficulty="medium",
            description="Moderate Toeplitz correlation with moderate noise.",
            dataset={
                "kind": "synthetic_support_recovery",
                "name": "medium_correlated",
                "n_samples": 320,
                "n_features": 64,
                "support_size": 8,
                "noise_std": 0.5,
                "signal_scale": 2.5,
                "min_signal_abs": 0.7,
                "correlation": 0.55,
                "clustered_support": True,
                "support_seed": 27182,
                "train_fraction": 0.6,
                "val_fraction": 0.2,
            },
        ),
        SyntheticScenario(
            name="hard_high_collinearity",
            difficulty="hard",
            description="High collinearity with clustered support and lower SNR.",
            dataset={
                "kind": "synthetic_support_recovery",
                "name": "hard_high_collinearity",
                "n_samples": 220,
                "n_features": 96,
                "support_size": 10,
                "noise_std": 0.9,
                "signal_scale": 2.0,
                "min_signal_abs": 0.6,
                "correlation": 0.82,
                "clustered_support": True,
                "support_seed": 16180,
                "train_fraction": 0.6,
                "val_fraction": 0.2,
            },
        ),
        SyntheticScenario(
            name="very_hard_p_gt_n",
            difficulty="very_hard",
            description="p > n with strong collinearity and weak-to-moderate signals.",
            dataset={
                "kind": "synthetic_support_recovery",
                "name": "very_hard_p_gt_n",
                "n_samples": 160,
                "n_features": 160,
                "support_size": 12,
                "noise_std": 1.1,
                "signal_scale": 1.9,
                "min_signal_abs": 0.55,
                "correlation": 0.88,
                "clustered_support": True,
                "support_seed": 14142,
                "train_fraction": 0.6,
                "val_fraction": 0.2,
            },
        ),
    ]


__all__ = [
    "SyntheticScenario",
    "progressive_support_recovery_scenarios",
]
