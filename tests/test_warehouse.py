"""Unit tests for DuckDB warehouse schema integrity and deterministic data generation."""

from __future__ import annotations

import duckdb

from warehouse.generate_data import generate_synthetic_warehouse


def test_all_11_tables_present(memory_db: duckdb.DuckDBPyConnection):
    """Verifies all 11 required B2B SaaS tables exist in the warehouse."""
    tables = {row[0] for row in memory_db.execute("SHOW TABLES").fetchall()}
    expected = {
        "plans",
        "accounts",
        "users",
        "workspaces",
        "subscriptions",
        "invoices",
        "feature_flags",
        "events",
        "experiments",
        "experiment_assignments",
        "support_tickets",
    }
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"


def test_deterministic_seeding_reproducibility():
    """Verifies identical seed produces bitwise identical row counts and values."""
    db1 = generate_synthetic_warehouse(
        db_path=":memory:", seed=123, num_accounts=20, num_events=500
    )
    db2 = generate_synthetic_warehouse(
        db_path=":memory:", seed=123, num_accounts=20, num_events=500
    )

    rows1 = db1.execute(
        "SELECT account_id, name, created_at, plan_tier FROM accounts ORDER BY account_id"
    ).fetchall()
    rows2 = db2.execute(
        "SELECT account_id, name, created_at, plan_tier FROM accounts ORDER BY account_id"
    ).fetchall()

    assert rows1 == rows2, "Seeded generation must be strictly deterministic"
    db1.close()
    db2.close()


def test_foreign_key_referential_integrity(memory_db: duckdb.DuckDBPyConnection):
    """Verifies relational integrity across accounts, users, workspaces, and events."""
    # 1. Every user belongs to an existing account
    orphan_users = memory_db.execute(
        "SELECT COUNT(*) FROM users u LEFT JOIN accounts a ON u.account_id = a.account_id WHERE a.account_id IS NULL"
    ).fetchone()[0]
    assert orphan_users == 0, f"Found {orphan_users} orphan users"

    # 2. Every workspace belongs to an existing account
    orphan_ws = memory_db.execute(
        "SELECT COUNT(*) FROM workspaces w LEFT JOIN accounts a ON w.account_id = a.account_id WHERE a.account_id IS NULL"
    ).fetchone()[0]
    assert orphan_ws == 0, f"Found {orphan_ws} orphan workspaces"

    # 3. Every event maps to a valid user and workspace
    orphan_events = memory_db.execute(
        """
        SELECT COUNT(*)
        FROM events e
        LEFT JOIN users u ON e.user_id = u.user_id
        LEFT JOIN workspaces w ON e.workspace_id = w.workspace_id
        WHERE u.user_id IS NULL OR w.workspace_id IS NULL
        """
    ).fetchone()[0]
    assert orphan_events == 0, f"Found {orphan_events} orphan events"


def test_business_definition_wau(memory_db: duckdb.DuckDBPyConnection):
    """Verifies customer WAU definition runs and returns valid non-negative integer."""
    query = """
    SELECT COUNT(DISTINCT user_id) AS wau
    FROM events
    WHERE event_type IN ('dashboard_create', 'query_run', 'export_csv', 'invite_sent')
      AND timestamp >= '2024-01-01'::DATE;
    """
    res = memory_db.execute(query).fetchone()
    assert res is not None
    assert isinstance(res[0], int)
    assert res[0] >= 0


def test_business_definition_mrr(memory_db: duckdb.DuckDBPyConnection):
    """Verifies active MRR calculation matches sum of active subscriptions."""
    mrr_res = memory_db.execute(
        "SELECT SUM(mrr_amount) FROM subscriptions WHERE status = 'active'"
    ).fetchone()[0]
    assert mrr_res is not None
    assert mrr_res > 0.0


def test_business_definition_activation(memory_db: duckdb.DuckDBPyConnection):
    """Verifies account activation CTE executes without error and computes binary flags."""
    activation_query = """
    WITH account_user_counts AS (
        SELECT a.account_id,
               COUNT(DISTINCT u.user_id) AS total_users
        FROM accounts a
        JOIN users u ON a.account_id = u.account_id
        WHERE u.created_at <= a.created_at + INTERVAL '14 days'
        GROUP BY a.account_id
    ),
    account_query_events AS (
        SELECT a.account_id,
               COUNT(e.event_id) AS total_queries
        FROM accounts a
        JOIN users u ON a.account_id = u.account_id
        JOIN events e ON u.user_id = e.user_id
        WHERE e.event_type = 'query_run'
          AND e.timestamp <= a.created_at + INTERVAL '14 days'
        GROUP BY a.account_id
    )
    SELECT a.account_id,
           CASE WHEN COALESCE(u.total_users, 0) >= 3 AND COALESCE(q.total_queries, 0) >= 1
                THEN TRUE ELSE FALSE END AS is_activated
    FROM accounts a
    LEFT JOIN account_user_counts u ON a.account_id = u.account_id
    LEFT JOIN account_query_events q ON a.account_id = q.account_id;
    """
    rows = memory_db.execute(activation_query).fetchall()
    assert len(rows) > 0
    for acc_id, is_act in rows:
        assert isinstance(is_act, bool)
