"""Unit tests for SQL execution equivalence and result set comparator."""

from __future__ import annotations

import duckdb

from src.eval.execution import compare_result_sets, evaluate_execution, has_order_by


def test_has_order_by_detection():
    assert has_order_by("SELECT * FROM accounts ORDER BY created_at DESC") is True
    assert has_order_by("SELECT account_id, name FROM accounts") is False
    assert (
        has_order_by(
            "WITH ordered_acc AS (SELECT * FROM accounts ORDER BY name) SELECT * FROM ordered_acc"
        )
        is True
    )


def test_compare_result_sets_unordered_multiset():
    gold = [(1, "Alice", 100.0), (2, "Bob", 200.0)]
    pred = [(2, "Bob", 200.0), (1, "Alice", 100.0)]
    assert compare_result_sets(gold, pred, order_matters=False) is True
    assert compare_result_sets(gold, pred, order_matters=True) is False


def test_compare_result_sets_float_tolerance():
    gold = [(1, 99.99999)]
    pred = [(1, 100.00001)]
    assert compare_result_sets(gold, pred, float_tol=1e-4) is True

    pred_far = [(1, 100.5)]
    assert compare_result_sets(gold, pred_far, float_tol=1e-4) is False


def test_compare_result_sets_duplicate_counts_matter():
    gold = [(1, "A"), (1, "A"), (2, "B")]
    pred = [(1, "A"), (2, "B"), (2, "B")]
    assert compare_result_sets(gold, pred, order_matters=False) is False


def test_compare_result_sets_null_handling():
    gold = [(1, None), (2, "Valid")]
    pred = [(2, "Valid"), (1, None)]
    assert compare_result_sets(gold, pred, order_matters=False) is True

    pred_wrong = [(1, "Not None"), (2, "Valid")]
    assert compare_result_sets(gold, pred_wrong, order_matters=False) is False


def test_evaluate_execution_live(memory_db: duckdb.DuckDBPyConnection):
    gold = "SELECT account_id, plan_tier FROM accounts WHERE status = 'active'"
    # Equivalent query with different alias or format
    pred_equiv = "SELECT a.account_id, a.plan_tier FROM accounts a WHERE a.status = 'active'"
    is_match, msg, _ = evaluate_execution(gold, pred_equiv, memory_db)
    assert is_match is True
    assert msg == "Match"

    pred_wrong = "SELECT account_id, plan_tier FROM accounts WHERE status = 'churned'"
    is_match2, msg2, _ = evaluate_execution(gold, pred_wrong, memory_db)
    assert is_match2 is False
    assert msg2 == "Mismatch"
