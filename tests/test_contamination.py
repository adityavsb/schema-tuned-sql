"""Unit tests for execution verification filter, AST/similarity deduplication, and general-SQL holdout."""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.datagen.dedupe import ContaminationFilter
from src.datagen.verifier import SQLVerifier
from src.eval.general_holdout import evaluate_general_holdout

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_sql_verifier_filters_invalid_and_empty(memory_db: duckdb.DuckDBPyConnection):
    verifier = SQLVerifier(memory_db)

    # 1. Valid positive query
    ok, reason, clean_sql, rows = verifier.verify_pair(
        question="How many active accounts exist?",
        raw_sql_or_completion="```sql\nSELECT COUNT(*) FROM accounts WHERE status = 'active';\n```",
    )
    assert ok is True
    assert reason == "accepted"
    assert len(rows) == 1

    # 2. Hallucinated column rejected
    ok2, reason2, _, _ = verifier.verify_pair(
        question="Bad column query",
        raw_sql_or_completion="SELECT non_existent_column FROM accounts",
    )
    assert ok2 is False
    assert "execution_error" in reason2

    # 3. Guardrail violation rejected
    ok3, reason3, _, _ = verifier.verify_pair(
        question="Malicious query",
        raw_sql_or_completion="DROP TABLE accounts",
    )
    assert ok3 is False
    assert reason3 == "guardrail_rejected"

    # 4. Empty result set rejected
    ok4, reason4, _, _ = verifier.verify_pair(
        question="Impossible filter",
        raw_sql_or_completion="SELECT * FROM accounts WHERE country = 'MARS_COLONY'",
    )
    assert ok4 is False
    assert reason4 == "empty_result_set"

    # 5. Negative query must be refused
    ok5, reason5, _, _ = verifier.verify_pair(
        question="What is the CEO's credit score?",
        raw_sql_or_completion="REFUSAL: UNANSWERABLE_FROM_SCHEMA",
        is_negative_query=True,
    )
    assert ok5 is True
    assert reason5 == "accepted_refusal"


def test_contamination_filter_blocks_test_and_dev_leaks():
    cf = ContaminationFilter(
        test_path=REPO_ROOT / "data" / "test_gold.jsonl",
        dev_path=REPO_ROOT / "data" / "dev.jsonl",
        similarity_threshold=0.85,
    )

    # Exact AST match against test_easy_001
    is_clean, reason, _ = cf.check_candidate(
        question="A completely rephrased question asking for active account count",
        sql="SELECT count(*) FROM accounts WHERE status = 'active'",
    )
    assert is_clean is False
    assert reason == "contaminated_exact_ast"

    # Near-duplicate question text against test_easy_001
    is_clean2, reason2, sim2 = cf.check_candidate(
        question="How many accounts are in active status right now?",
        sql="SELECT COUNT(account_id) FROM accounts WHERE status = 'active'",
    )
    assert is_clean2 is False
    assert reason2 in ("contaminated_question_similarity", "contaminated_exact_ast")
    assert sim2 >= 0.85

    # Clean novel query passes once, then is blocked as internal duplicate
    novel_q = "Compute the median mrr_amount for enterprise accounts created after 2023-04-15."
    novel_sql = "SELECT MEDIAN(s.mrr_amount) FROM subscriptions s JOIN accounts a ON s.account_id = a.account_id WHERE a.plan_tier = 'enterprise'"
    ok_first, reason_first, _ = cf.check_candidate(novel_q, novel_sql)
    assert ok_first is True
    assert reason_first == "clean"

    ok_dup, reason_dup, _ = cf.check_candidate(novel_q, novel_sql)
    assert ok_dup is False
    assert reason_dup == "internal_duplicate"


def test_general_sql_holdout_execution():
    holdout_path = REPO_ROOT / "data" / "general_sql_holdout.jsonl"
    assert holdout_path.is_file()
    import json

    items = [
        json.loads(line)
        for line in holdout_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(items) == 200

    # Oracle predictions should achieve 100% EX on the 4-domain general-SQL database
    oracle_preds = {item["id"]: item["gold_sql"] for item in items}
    res = evaluate_general_holdout(oracle_preds, holdout_path=holdout_path)
    assert res["total_items"] == 200
    assert res["execution_accuracy"] == 1.0
