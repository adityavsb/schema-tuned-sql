"""Unit tests for AST-enforced Read-Only SQL Guardrail."""

from __future__ import annotations

import pytest

from src.utils.db import ReadOnlySQLGuardrail, SecurityViolationError


@pytest.fixture
def guardrail() -> ReadOnlySQLGuardrail:
    return ReadOnlySQLGuardrail()


def test_valid_simple_select(guardrail: ReadOnlySQLGuardrail):
    query = "SELECT account_id, name, plan_tier FROM accounts WHERE status = 'active'"
    assert guardrail.is_safe(query) is True
    clean = guardrail.validate_and_sanitize(query)
    assert "SELECT" in clean
    assert "accounts" in clean


def test_valid_cte_select(guardrail: ReadOnlySQLGuardrail):
    query = """
    WITH active_mrr AS (
        SELECT account_id, SUM(mrr_amount) as total_mrr
        FROM subscriptions
        WHERE status = 'active'
        GROUP BY account_id
    )
    SELECT a.name, m.total_mrr
    FROM accounts a
    JOIN active_mrr m ON a.account_id = m.account_id
    WHERE m.total_mrr > 500
    """
    assert guardrail.is_safe(query) is True
    clean = guardrail.validate_and_sanitize(query)
    assert "WITH" in clean


def test_valid_window_function(guardrail: ReadOnlySQLGuardrail):
    query = """
    SELECT user_id, event_type, timestamp,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp DESC) as seq
    FROM events
    WHERE event_type = 'query_run'
    """
    assert guardrail.is_safe(query) is True


def test_valid_union_query(guardrail: ReadOnlySQLGuardrail):
    query = """
    SELECT account_id, name FROM accounts WHERE plan_tier = 'growth'
    UNION ALL
    SELECT account_id, name FROM accounts WHERE plan_tier = 'enterprise'
    """
    assert guardrail.is_safe(query) is True


def test_block_empty_or_whitespace_query(guardrail: ReadOnlySQLGuardrail):
    assert guardrail.is_safe("") is False
    assert guardrail.is_safe("   \n\t  ") is False
    with pytest.raises(SecurityViolationError, match="Empty SQL query"):
        guardrail.validate_and_sanitize("   ")


def test_block_drop_table(guardrail: ReadOnlySQLGuardrail):
    query = "DROP TABLE accounts"
    assert guardrail.is_safe(query) is False
    with pytest.raises(SecurityViolationError, match="Only read-only SELECT or UNION"):
        guardrail.validate_and_sanitize(query)


def test_block_delete_statement(guardrail: ReadOnlySQLGuardrail):
    query = "DELETE FROM users WHERE user_id = 'usr_00001'"
    assert guardrail.is_safe(query) is False
    with pytest.raises(SecurityViolationError):
        guardrail.validate_and_sanitize(query)


def test_block_insert_statement(guardrail: ReadOnlySQLGuardrail):
    query = "INSERT INTO accounts (account_id, name) VALUES ('acc_9999', 'Evil Corp')"
    assert guardrail.is_safe(query) is False
    with pytest.raises(SecurityViolationError):
        guardrail.validate_and_sanitize(query)


def test_block_update_statement(guardrail: ReadOnlySQLGuardrail):
    query = "UPDATE subscriptions SET mrr_amount = 0 WHERE status = 'active'"
    assert guardrail.is_safe(query) is False
    with pytest.raises(SecurityViolationError):
        guardrail.validate_and_sanitize(query)


def test_block_multi_statement_injection(guardrail: ReadOnlySQLGuardrail):
    query = "SELECT * FROM accounts; DROP TABLE subscriptions;"
    assert guardrail.is_safe(query) is False
    with pytest.raises(SecurityViolationError, match="Multi-statement execution prohibited"):
        guardrail.validate_and_sanitize(query)


def test_block_prohibited_filesystem_functions(guardrail: ReadOnlySQLGuardrail):
    dangerous_queries = [
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM read_parquet('s3://prod-lake/events.parquet')",
        "SELECT * FROM read_json('/root/.aws/credentials')",
        "SELECT * FROM sniff_csv('warehouse/schema.sql')",
        "SELECT * FROM glob('/*')",
        "SELECT duckdb_settings()",
    ]
    for q in dangerous_queries:
        assert guardrail.is_safe(q) is False, f"Expected {q} to be blocked"
        with pytest.raises(SecurityViolationError):
            guardrail.validate_and_sanitize(q)


def test_block_unauthorized_table_access(guardrail: ReadOnlySQLGuardrail):
    unauthorized_queries = [
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM duckdb_tables",
        "SELECT * FROM secret_passwords",
        "SELECT * FROM system_audit_log",
    ]
    for q in unauthorized_queries:
        assert guardrail.is_safe(q) is False, f"Expected {q} to be blocked"
        with pytest.raises(
            SecurityViolationError, match="(Unauthorized table reference|Schema-qualified)"
        ):
            guardrail.validate_and_sanitize(q)
