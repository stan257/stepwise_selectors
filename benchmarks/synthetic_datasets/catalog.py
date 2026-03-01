"""Curated synthetic benchmark scenarios for support recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SyntheticScenario:
    """One reproducible synthetic scenario configuration."""

    name: str
    difficulty: str
    description: str
    checks: str
    why_hard: str
    dataset: dict
    quick_seeds: int
    full_seeds: int


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
            checks="Sanity check: selectors should recover support exactly and generalize cleanly.",
            why_hard="Not hard; this guards against implementation bugs in otherwise friendly conditions.",
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
            quick_seeds=12,
            full_seeds=20,
        ),
        SyntheticScenario(
            name="medium_correlated",
            difficulty="medium",
            description="Moderate Toeplitz correlation with moderate noise.",
            checks=(
                "Correlation robustness: can methods keep high support recovery when features overlap?"
            ),
            why_hard=(
                "Predictors are no longer independent, so local search can confuse correlated alternatives."
            ),
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
            quick_seeds=12,
            full_seeds=28,
        ),
        SyntheticScenario(
            name="hard_high_collinearity",
            difficulty="hard",
            description="High collinearity with clustered support and lower SNR.",
            checks=(
                "Stability under strong collinearity: support overlap, exact-recovery rate, and variance across seeds."
            ),
            why_hard=(
                "Many near-equivalent supports exist; small perturbations can flip chosen subsets."
            ),
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
            quick_seeds=12,
            full_seeds=48,
        ),
        SyntheticScenario(
            name="very_hard_p_gt_n",
            difficulty="very_hard",
            description="p > n with strong collinearity and weak-to-moderate signals.",
            checks=(
                "High-dimensional stress: compare predictive error vs support metrics when exact recovery is unlikely."
            ),
            why_hard=(
                "Combinatorial ambiguity is high, many supports interpolate similarly, and variance inflates."
            ),
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
            quick_seeds=12,
            full_seeds=64,
        ),
    ]


def challenging_support_recovery_scenarios() -> list[SyntheticScenario]:
    """Return failure-mode stress scenarios beyond the progressive core."""
    return [
        SyntheticScenario(
            name="hard_ultra_collinear_twins",
            difficulty="hard",
            description=(
                "Each true signal has highly correlated decoy twins, stressing greedy ranking."
            ),
            checks=(
                "Decoy resistance: can methods avoid selecting look-alike proxy features over true signals?"
            ),
            why_hard=(
                "Twin features are almost indistinguishable by marginal relevance and can trap greedy heuristics."
            ),
            dataset={
                "kind": "synthetic_support_recovery",
                "name": "hard_ultra_collinear_twins",
                "n_samples": 220,
                "n_features": 120,
                "support_size": 10,
                "noise_std": 0.95,
                "signal_scale": 2.1,
                "min_signal_abs": 0.6,
                "correlation": 0.82,
                "clustered_support": True,
                "support_seed": 17320,
                "twin_decoys_per_signal": 2,
                "twin_strength": 0.985,
                "twin_noise_std": 0.12,
                "train_fraction": 0.6,
                "val_fraction": 0.2,
            },
            quick_seeds=12,
            full_seeds=72,
        ),
        SyntheticScenario(
            name="hard_misspecified_nonlinear",
            difficulty="hard",
            description=(
                "Target includes nonlinear signal components while selectors remain linear."
            ),
            checks=(
                "Misspecification behavior: quantify graceful degradation in linear subset selection."
            ),
            why_hard=(
                "The model class is wrong by design; no linear subset can perfectly represent the target."
            ),
            dataset={
                "kind": "synthetic_support_recovery",
                "name": "hard_misspecified_nonlinear",
                "n_samples": 260,
                "n_features": 72,
                "support_size": 8,
                "noise_std": 0.8,
                "signal_scale": 2.3,
                "min_signal_abs": 0.65,
                "correlation": 0.55,
                "clustered_support": False,
                "support_seed": 27181,
                "nonlinear_strength": 0.9,
                "train_fraction": 0.6,
                "val_fraction": 0.2,
            },
            quick_seeds=12,
            full_seeds=56,
        ),
    ]


def oracle_support_recovery_scenarios() -> list[SyntheticScenario]:
    """Return small-p scenario where exact subset oracle is tractable."""
    return [
        SyntheticScenario(
            name="oracle_small_p",
            difficulty="medium",
            description=(
                "Small feature space for exact best-subset oracle gap measurement."
            ),
            checks=(
                "Approximation quality: measure gap between heuristic search and exact train-RSS optimum."
            ),
            why_hard=(
                "Not intrinsically hardest; it is designed for tractable exhaustive search and quality auditing."
            ),
            dataset={
                "kind": "synthetic_support_recovery",
                "name": "oracle_small_p",
                "n_samples": 260,
                "n_features": 18,
                "support_size": 6,
                "noise_std": 0.65,
                "signal_scale": 2.2,
                "min_signal_abs": 0.7,
                "correlation": 0.68,
                "clustered_support": False,
                "support_seed": 31416,
                "train_fraction": 0.6,
                "val_fraction": 0.2,
            },
            quick_seeds=10,
            full_seeds=40,
        )
    ]


def stability_scenarios_for_profile(
    profile: Literal["quick", "full"],
) -> list[SyntheticScenario]:
    """Return scenario list for a named stability profile."""
    match profile:
        case "quick" | "full":
            return (
                progressive_support_recovery_scenarios()
                + challenging_support_recovery_scenarios()
                + oracle_support_recovery_scenarios()
            )
        case _:
            raise ValueError(f"Unsupported profile: {profile!r}")


__all__ = [
    "SyntheticScenario",
    "progressive_support_recovery_scenarios",
    "challenging_support_recovery_scenarios",
    "oracle_support_recovery_scenarios",
    "stability_scenarios_for_profile",
]
