import pytest

from benchmarks.stability_utils import (
    pairwise_support_jaccard,
    summarize_stability_rows,
    support_jaccard,
)


def test_support_jaccard_handles_empty_sets():
    assert support_jaccard([], []) == 1.0
    assert support_jaccard([1, 2], []) == 0.0


def test_pairwise_support_jaccard_summary():
    summary = pairwise_support_jaccard([[0, 1], [0, 1], [1, 2]])
    assert summary.mean == pytest.approx((1.0 + (1.0 / 3.0) + (1.0 / 3.0)) / 3.0)
    assert summary.minimum == pytest.approx(1.0 / 3.0)
    assert summary.maximum == pytest.approx(1.0)


def test_summarize_stability_rows_adds_baseline_deltas():
    rows = [
        {
            "status": "ok",
            "scenario_name": "scenario_a",
            "difficulty": "easy",
            "dataset_seed": 1,
            "method_name": "topk_abs_cov",
            "selector": "TopKAbsCovBaseline",
            "active_set": [0, 2],
            "true_support": [0, 2],
            "metrics": {
                "test_mse": 1.0,
                "val_mse": 1.1,
                "support_f1": 0.8,
                "support_precision": 0.8,
                "support_recall": 0.8,
                "n_selected": 2,
                "elapsed_ms": 1.0,
            },
        },
        {
            "status": "ok",
            "scenario_name": "scenario_a",
            "difficulty": "easy",
            "dataset_seed": 1,
            "method_name": "forward_aic",
            "selector": "ForwardSelection",
            "active_set": [0, 2],
            "true_support": [0, 2],
            "metrics": {
                "test_mse": 0.8,
                "val_mse": 0.9,
                "support_f1": 0.9,
                "support_precision": 0.9,
                "support_recall": 0.9,
                "n_selected": 2,
                "elapsed_ms": 2.0,
            },
        },
    ]

    payload = summarize_stability_rows(rows)
    summary_rows = payload["rows"]
    forward = next(r for r in summary_rows if r["method_name"] == "forward_aic")
    baseline = next(r for r in summary_rows if r["method_name"] == "topk_abs_cov")

    assert baseline["delta_test_mse_vs_topk"] == pytest.approx(0.0)
    assert baseline["delta_support_f1_vs_topk"] == pytest.approx(0.0)
    assert forward["delta_test_mse_vs_topk"] == pytest.approx(-0.2)
    assert forward["delta_support_f1_vs_topk"] == pytest.approx(0.1)


def test_summarize_stability_rows_computes_win_rates_and_oracle_gap():
    rows = [
        {
            "status": "ok",
            "scenario_name": "scenario_b",
            "difficulty": "hard",
            "dataset_seed": 101,
            "method_name": "forward_bic",
            "selector": "ForwardSelection",
            "active_set": [0, 1],
            "true_support": [0, 1],
            "oracle_gap_test_mse": 0.10,
            "oracle_gap_train_rss": 0.30,
            "metrics": {
                "test_mse": 1.0,
                "val_mse": 1.0,
                "support_f1": 0.8,
                "support_precision": 0.8,
                "support_recall": 0.8,
                "n_selected": 2,
                "elapsed_ms": 1.0,
            },
        },
        {
            "status": "ok",
            "scenario_name": "scenario_b",
            "difficulty": "hard",
            "dataset_seed": 102,
            "method_name": "forward_bic",
            "selector": "ForwardSelection",
            "active_set": [0, 1],
            "true_support": [0, 1],
            "oracle_gap_test_mse": 0.20,
            "oracle_gap_train_rss": 0.40,
            "metrics": {
                "test_mse": 1.2,
                "val_mse": 1.2,
                "support_f1": 0.7,
                "support_precision": 0.7,
                "support_recall": 0.7,
                "n_selected": 2,
                "elapsed_ms": 1.0,
            },
        },
        {
            "status": "ok",
            "scenario_name": "scenario_b",
            "difficulty": "hard",
            "dataset_seed": 101,
            "method_name": "beam_forward_bic_w4",
            "selector": "BeamForwardSelection",
            "active_set": [0, 1],
            "true_support": [0, 1],
            "oracle_gap_test_mse": 0.05,
            "oracle_gap_train_rss": 0.20,
            "metrics": {
                "test_mse": 0.8,
                "val_mse": 0.8,
                "support_f1": 0.9,
                "support_precision": 0.9,
                "support_recall": 0.9,
                "n_selected": 2,
                "elapsed_ms": 1.0,
            },
        },
        {
            "status": "ok",
            "scenario_name": "scenario_b",
            "difficulty": "hard",
            "dataset_seed": 102,
            "method_name": "beam_forward_bic_w4",
            "selector": "BeamForwardSelection",
            "active_set": [0, 1],
            "true_support": [0, 1],
            "oracle_gap_test_mse": 0.15,
            "oracle_gap_train_rss": 0.35,
            "metrics": {
                "test_mse": 1.1,
                "val_mse": 1.1,
                "support_f1": 0.85,
                "support_precision": 0.85,
                "support_recall": 0.85,
                "n_selected": 2,
                "elapsed_ms": 1.0,
            },
        },
        {
            "status": "ok",
            "scenario_name": "scenario_b",
            "difficulty": "hard",
            "dataset_seed": 101,
            "method_name": "topk_abs_cov",
            "selector": "TopKAbsCovBaseline",
            "active_set": [0, 1],
            "true_support": [0, 1],
            "oracle_gap_test_mse": 0.30,
            "oracle_gap_train_rss": 0.60,
            "metrics": {
                "test_mse": 1.4,
                "val_mse": 1.4,
                "support_f1": 0.6,
                "support_precision": 0.6,
                "support_recall": 0.6,
                "n_selected": 2,
                "elapsed_ms": 1.0,
            },
        },
        {
            "status": "ok",
            "scenario_name": "scenario_b",
            "difficulty": "hard",
            "dataset_seed": 102,
            "method_name": "topk_abs_cov",
            "selector": "TopKAbsCovBaseline",
            "active_set": [0, 1],
            "true_support": [0, 1],
            "oracle_gap_test_mse": 0.40,
            "oracle_gap_train_rss": 0.80,
            "metrics": {
                "test_mse": 1.5,
                "val_mse": 1.5,
                "support_f1": 0.55,
                "support_precision": 0.55,
                "support_recall": 0.55,
                "n_selected": 2,
                "elapsed_ms": 1.0,
            },
        },
    ]

    payload = summarize_stability_rows(rows)
    summary_rows = payload["rows"]
    beam = next(r for r in summary_rows if r["method_name"] == "beam_forward_bic_w4")

    assert beam["paired_runs_vs_forward_bic"] == 2
    assert beam["paired_runs_vs_topk"] == 2
    assert beam["win_rate_test_mse_vs_forward_bic"] == pytest.approx(1.0)
    assert beam["win_rate_test_mse_vs_topk"] == pytest.approx(1.0)
    assert beam["mean_oracle_gap_test_mse"] == pytest.approx(0.10)
    assert beam["mean_oracle_gap_train_rss"] == pytest.approx(0.275)
    assert beam["test_mse_ci95_low"] < beam["mean_test_mse"] < beam["test_mse_ci95_high"]
