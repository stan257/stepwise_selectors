from benchmarks.dominance_report import build_dominance_markdown


def test_build_dominance_markdown_contains_sections():
    payload = {
        "rows": [
            {
                "scenario_name": "s1",
                "difficulty": "easy",
                "method_name": "forward_bic",
                "mean_test_mse": 1.0,
                "mean_support_f1": 0.9,
                "mean_pairwise_jaccard": 0.8,
                "win_rate_test_mse_vs_topk": 1.0,
            },
            {
                "scenario_name": "s1",
                "difficulty": "easy",
                "method_name": "topk_abs_cov",
                "mean_test_mse": 2.0,
                "mean_support_f1": 0.4,
                "mean_pairwise_jaccard": 0.3,
                "win_rate_test_mse_vs_topk": 0.5,
            },
            {
                "scenario_name": "s2",
                "difficulty": "hard",
                "method_name": "forward_bic",
                "mean_test_mse": 3.0,
                "mean_support_f1": 0.7,
                "mean_pairwise_jaccard": 0.5,
                "win_rate_test_mse_vs_topk": 0.5,
            },
            {
                "scenario_name": "s2",
                "difficulty": "hard",
                "method_name": "topk_abs_cov",
                "mean_test_mse": 4.0,
                "mean_support_f1": 0.6,
                "mean_pairwise_jaccard": 0.4,
                "win_rate_test_mse_vs_topk": 0.5,
            },
        ]
    }
    md = build_dominance_markdown(payload)
    assert "# Dominance Summary" in md
    assert "## Scenario Winners" in md
    assert "## Method Scoreboard" in md
    assert "## Pairwise Dominance Counts" in md
    assert "forward_bic" in md
    assert "topk_abs_cov" in md
