"""Synthetic benchmark scenario catalog."""

from .catalog import (
    SyntheticScenario,
    challenging_support_recovery_scenarios,
    oracle_support_recovery_scenarios,
    progressive_support_recovery_scenarios,
    stability_scenarios_for_profile,
)

__all__ = [
    "SyntheticScenario",
    "progressive_support_recovery_scenarios",
    "challenging_support_recovery_scenarios",
    "oracle_support_recovery_scenarios",
    "stability_scenarios_for_profile",
]
