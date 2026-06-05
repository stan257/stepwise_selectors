from __future__ import annotations

import math

import pytest

from benchmarks.market_chunks import (
    MarketChunkConfig,
    generate_market_gram_chunks_unknown,
    save_market_chunk_dataset,
)
from benchmarks.market_experiments import run_market_chunk_experiment


def _tiny_dataset_path(tmp_path) -> str:
    cfg = MarketChunkConfig(
        seed=17,
        n_chunks=5,
        bars_per_chunk=140,
        warmup_bars=60,
        n_features=28,
        n_regimes=3,
        support_size=6,
    )
    dataset = generate_market_gram_chunks_unknown(cfg)
    path = tmp_path / "market_chunks_tiny.npz"
    save_market_chunk_dataset(dataset, path)
    return str(path)


def test_market_experiment_runs_three_methods_and_ranks(tmp_path):
    dataset_path = _tiny_dataset_path(tmp_path)
    payload = run_market_chunk_experiment(
        dataset_path=dataset_path,
        train_chunks=4,
        holdout_chunk=4,
        max_steps=8,
        beam_width=3,
        cv_aggregation="mean_mse",
    )

    assert payload["dataset_n_chunks"] == 5
    assert payload["train_chunk_indices"] == [0, 1, 2, 3]
    assert payload["holdout_chunk_index"] == 4

    results = payload["results"]
    assert isinstance(results, list)
    assert len(results) == 3
    methods = {row["method"] for row in results}
    assert methods == {
        "forward_aic_in_sample",
        "cv_forward_oos_mean_mse",
        "cv_beam_forward_oos_mean_mse_w3",
    }
    ranks = sorted(int(row["rank_oos_holdout_mse"]) for row in results)
    assert ranks == [1, 2, 3]

    for row in results:
        assert int(row["n_selected"]) >= 0
        assert math.isfinite(float(row["oos_holdout_mse"]))
        assert math.isfinite(float(row["train_combined_mse"]))
        assert -10.0 <= float(row["oos_holdout_r2"]) <= 1.0


def test_market_experiment_rejects_holdout_overlap(tmp_path):
    dataset_path = _tiny_dataset_path(tmp_path)
    with pytest.raises(ValueError, match="must not overlap"):
        run_market_chunk_experiment(
            dataset_path=dataset_path,
            train_chunks=4,
            holdout_chunk=2,
        )
