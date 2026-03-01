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
