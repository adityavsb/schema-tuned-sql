"""Unit tests for bootstrap confidence intervals and McNemar significance tests."""

from __future__ import annotations

from src.eval.metrics import aggregate_system_metrics, compute_bootstrap_ci, compute_mcnemar_test


def test_bootstrap_ci_coverage():
    # Symmetric 50% distribution
    vals = [1.0] * 50 + [0.0] * 50
    low, high = compute_bootstrap_ci(vals, n_bootstrap=1000, ci=0.95, seed=42)
    assert 0.35 <= low <= 0.45
    assert 0.55 <= high <= 0.65
    assert low < high


def test_bootstrap_ci_constant_values():
    vals = [1.0] * 20
    low, high = compute_bootstrap_ci(vals, n_bootstrap=500, ci=0.95)
    assert low == 1.0
    assert high == 1.0


def test_mcnemar_identical_models():
    preds = [True, False, True, True, False]
    res = compute_mcnemar_test(preds, preds)
    assert res["discordant_pairs"] == 0
    assert res["b_a_only"] == 0
    assert res["c_b_only"] == 0
    assert res["p_value"] == 1.0
    assert res["is_significant_05"] is False


def test_mcnemar_significant_difference():
    # Model A is much better than Model B
    # 25 discordant pairs where Model A is right and Model B is wrong, 1 where B is right
    preds_a = [True] * 25 + [False] * 1 + [True] * 50
    preds_b = [False] * 25 + [True] * 1 + [True] * 50
    res = compute_mcnemar_test(preds_a, preds_b)
    assert res["b_a_only"] == 25
    assert res["c_b_only"] == 1
    assert res["discordant_pairs"] == 26
    assert res["p_value"] < 0.001
    assert res["is_significant_05"] is True


def test_aggregate_system_metrics():
    records = [
        {"is_match": True, "is_valid_sql": True, "difficulty": "easy", "is_negative_query": False},
        {"is_match": True, "is_valid_sql": True, "difficulty": "easy", "is_negative_query": False},
        {
            "is_match": False,
            "is_valid_sql": True,
            "difficulty": "medium",
            "is_negative_query": False,
        },
        {"is_match": True, "is_valid_sql": True, "difficulty": "hard", "is_negative_query": False},
        # Negative query
        {
            "is_match": False,
            "is_valid_sql": False,
            "is_negative_query": True,
            "refusal_detected": True,
        },
    ]
    summary = aggregate_system_metrics(records, n_bootstrap=500)
    assert summary["total_samples"] == 5
    assert summary["answerable_samples"] == 4
    assert summary["negative_samples"] == 1
    assert summary["execution_accuracy"] == 0.75  # 3 / 4
    assert summary["valid_sql_rate"] == 0.8  # 4 / 5
    assert summary["refusal_accuracy"] == 1.0  # 1 / 1
    assert "easy" in summary["by_difficulty"]
    assert summary["by_difficulty"]["easy"]["execution_accuracy"] == 1.0
