from __future__ import annotations

from benchmarks.microstructure.types import MicrostructureChunkConfig
from benchmarks.microstructure_budget_sweep import (
    render_microstructure_budget_report,
    run_microstructure_budget_sweep,
)


def test_microstructure_budget_sweep_exact_k_runs_and_respects_budget():
    rows, summary = run_microstructure_budget_sweep(
        base_config=MicrostructureChunkConfig(
            seed=7,
            n_chunks=5,
            events_per_chunk=120,
            warmup_events=60,
            n_features=64,
            n_regimes=3,
            target_horizon_events=6,
        ),
        budgets=[2, 4],
        seeds=[7, 8],
        train_chunks=4,
        holdout_chunk=4,
        beam_width=3,
        cv_aggregation="mean_mse",
        flavor="unknown",
        selection_mode="exact_k",
        solver_policy="pinv",
    )

    assert len(rows) == 2 * 2 * 3
    assert set(summary["by_budget"].keys()) == {"2", "4"}
    assert set(summary["overall"]["methods"].keys()) == {
        "beam_forward_in_sample_rss_w3",
        "cv_forward_oos_mean_mse",
        "forward_aic_in_sample",
    }
    for row in rows:
        assert int(row["rank_oos_holdout_mse"]) in {1, 2, 3}
        assert int(row["max_steps"]) in {2, 4}
        assert int(row["n_selected"]) == int(row["max_steps"])
        assert row["selection_mode"] == "exact_k"
        assert row["solver_policy"] == "pinv"

    report = render_microstructure_budget_report(
        rows=rows,
        summary=summary,
        config={
            "budgets": [2, 4],
            "seeds": [7, 8],
            "train_chunks": 4,
            "holdout_chunk": 4,
            "preset": "default",
            "selection_mode": "exact_k",
            "solver_policy": "pinv",
            "flavor": "unknown",
        },
    )
    assert "# Microstructure Exact-K Budget Sweep Report" in report
    assert "Selection mode: `exact_k`" in report


def test_microstructure_budget_sweep_supports_multiscale_192_preset():
    rows, summary = run_microstructure_budget_sweep(
        base_config=MicrostructureChunkConfig(
            seed=11,
            n_chunks=5,
            events_per_chunk=120,
            warmup_events=60,
            n_features=64,
            n_regimes=3,
            target_horizon_events=6,
        ),
        budgets=[4],
        seeds=[11],
        train_chunks=4,
        holdout_chunk=4,
        beam_width=3,
        cv_aggregation="mean_mse",
        flavor="unknown",
        preset="unknown_multiscale_192",
        selection_mode="exact_k",
        solver_policy="pinv",
    )

    assert len(rows) == 3
    assert set(summary["overall"]["methods"].keys()) == {
        "beam_forward_in_sample_rss_w3",
        "cv_forward_oos_mean_mse",
        "forward_aic_in_sample",
    }
    assert all(len(row["active_set"]) == 4 for row in rows)
