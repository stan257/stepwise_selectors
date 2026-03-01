import pytest

from benchmarks.datasets import build_dataset
from benchmarks.oracle import exact_best_subset_train_rss
from benchmarks.synthetic_datasets import stability_scenarios_for_profile


def test_oracle_returns_result_for_small_problem():
    cfg = {
        "kind": "synthetic_support_recovery",
        "name": "oracle_unit_small",
        "seed": 123,
        "n_samples": 140,
        "n_features": 8,
        "support_size": 3,
        "noise_std": 0.3,
        "signal_scale": 2.0,
        "correlation": 0.3,
        "train_fraction": 0.6,
        "val_fraction": 0.2,
    }
    dataset = build_dataset(cfg)
    oracle = exact_best_subset_train_rss(
        dataset,
        k=3,
        max_features=12,
        max_combinations=1000,
    )
    assert oracle is not None
    assert len(oracle.active_set) == 3
    assert oracle.n_features == 8
    assert oracle.n_combinations > 0
    assert oracle.train_rss >= 0.0


def test_oracle_skips_when_problem_exceeds_budget():
    cfg = {
        "kind": "synthetic_support_recovery",
        "name": "oracle_unit_large",
        "seed": 123,
        "n_samples": 140,
        "n_features": 24,
        "support_size": 4,
        "noise_std": 0.3,
        "signal_scale": 2.0,
        "correlation": 0.3,
        "train_fraction": 0.6,
        "val_fraction": 0.2,
    }
    dataset = build_dataset(cfg)
    oracle = exact_best_subset_train_rss(
        dataset,
        k=4,
        max_features=18,
        max_combinations=100000,
    )
    assert oracle is None


def test_profile_scenarios_have_researcher_commentary():
    scenarios = stability_scenarios_for_profile("quick")
    assert scenarios
    for scenario in scenarios:
        assert scenario.description
        assert scenario.checks
        assert scenario.why_hard
        assert scenario.quick_seeds > 0
        assert scenario.full_seeds > 0


def test_invalid_profile_raises():
    with pytest.raises(ValueError):
        stability_scenarios_for_profile("invalid")  # type: ignore[arg-type]
