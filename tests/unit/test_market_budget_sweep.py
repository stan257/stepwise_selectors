from __future__ import annotations

from benchmarks.market_budget_sweep import (
    render_market_budget_report,
    run_market_budget_sweep,
)
from benchmarks.market_chunks import MarketChunkConfig


def test_market_budget_sweep_runs_and_summarizes():
    rows, summary = run_market_budget_sweep(
        base_config=MarketChunkConfig(
            seed=7,
            n_chunks=5,
            bars_per_chunk=120,
            warmup_bars=60,
            n_features=24,
            n_regimes=3,
            support_size=5,
        ),
        budgets=[2, 4],
        seeds=[7, 8],
        train_chunks=4,
        holdout_chunk=4,
        beam_width=3,
        cv_aggregation="mean_mse",
        flavor="unknown",
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

    report = render_market_budget_report(
        rows=rows,
        summary=summary,
        config={
            "budgets": [2, 4],
            "seeds": [7, 8],
            "train_chunks": 4,
            "holdout_chunk": 4,
        },
    )
    assert "# Market Budget Sweep Report" in report
    assert "## Overall Summary" in report

