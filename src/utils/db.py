"""DuckDB database access manager with AST-enforced read-only security guardrails."""

from __future__ import annotations

from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

ALLOWED_TABLES: set[str] = {
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

PROHIBITED_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_json_auto",
    "read_text",
    "read_blob",
    "sniff_csv",
    "glob",
    "duckdb_settings",
    "duckdb_memory",
    "current_setting",
    "set_variable",
    "get_variable",
    "range",
    "generate_series",
    "system",
}


class SecurityViolationError(Exception):
    """Raised when a SQL query violates read-only or isolation security guardrails."""

    pass


class ReadOnlySQLGuardrail:
    """Validates and sanitizes SQL queries using AST inspection via sqlglot."""

    PROHIBITED_EXP_TYPES: set[type[exp.Expression]] = {
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.Command,
        exp.Pragma,
        exp.Set,
        exp.Transaction,
        exp.Commit,
        exp.Rollback,
        exp.Kill,
        exp.Merge,
        exp.TruncateTable,
        exp.Copy,
        exp.CurrentVersion,
    }

    def __init__(self, allowed_tables: set[str] | None = None, dialect: str = "duckdb"):
        self.allowed_tables = {t.lower() for t in (allowed_tables or ALLOWED_TABLES)}
        self.dialect = dialect

    def is_safe(self, sql_query: str) -> bool:
        """Returns True if the query passes all security checks, False otherwise."""
        try:
            self.validate_and_sanitize(sql_query)
            return True
        except SecurityViolationError:
            return False

    def validate_and_sanitize(self, sql_query: str) -> str:
        """Parses and verifies read-only AST safety, returning normalized SQL.

        Raises SecurityViolationError if any guardrail check fails.
        """
        if not sql_query or not sql_query.strip():
            raise SecurityViolationError("Empty SQL query is prohibited.")

        clean_sql = sql_query.strip()
        if clean_sql.endswith(";"):
            clean_sql = clean_sql[:-1].strip()

        # Multi-statement injection guard
        try:
            statements = sqlglot.parse(clean_sql, read=self.dialect)
        except Exception as e:
            raise SecurityViolationError(f"SQL Syntax parsing failed: {e}")

        if len(statements) != 1 or statements[0] is None:
            raise SecurityViolationError(
                f"Multi-statement execution prohibited. Detected {len(statements)} statements."
            )

        root = statements[0]

        # Read-only statement guard
        if not isinstance(root, (exp.Select, exp.Union)):
            raise SecurityViolationError(
                f"Only read-only SELECT or UNION queries permitted; received {type(root).__name__}."
            )

        # Block prohibited statement types
        for prohibited_type in self.PROHIBITED_EXP_TYPES:
            if root.find(prohibited_type):
                raise SecurityViolationError(
                    f"Prohibited SQL expression detected: {prohibited_type.__name__}."
                )

        # Collect CTE names
        cte_names = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}

        # Strict table allowlist and prohibited function / table literal verification
        for table in root.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                raise SecurityViolationError(
                    "Table functions, file paths, and subquery literals are prohibited in FROM."
                )
            if table.args.get("db") or table.args.get("catalog"):
                raise SecurityViolationError("Schema-qualified table references are prohibited.")
            t_name = table.name.lower()
            if t_name not in self.allowed_tables and t_name not in cte_names:
                raise SecurityViolationError(
                    f"Unauthorized table reference: '{t_name}'. Allowed: {sorted(self.allowed_tables)}"
                )

        # Block prohibited function calls
        for node in root.walk():
            fn_name = ""
            if isinstance(node, exp.Anonymous):
                fn_name = node.name.lower()
            elif isinstance(node, exp.Func):
                fn_name = (
                    getattr(node, "name", "").lower()
                    or getattr(node, "sql_name", lambda: "")().lower()
                )

            if fn_name and fn_name in PROHIBITED_FUNCTIONS:
                raise SecurityViolationError(
                    f"Prohibited system or filesystem function detected: '{fn_name}'."
                )

        return root.sql(dialect=self.dialect)


class WarehouseDatabase:
    """Manages an embedded DuckDB instance with locked security configurations."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        config = {
            "enable_external_access": "false",
        }
        self.conn = duckdb.connect(database=db_path, config=config)
        self.conn.execute("SET lock_configuration=true;")
        self.guardrail = ReadOnlySQLGuardrail()

    def execute_read_only(self, sql_query: str) -> list[tuple[Any, ...]]:
        """Validates AST safety and executes read-only query against DuckDB."""
        sanitized_sql = self.guardrail.validate_and_sanitize(sql_query)
        return self.conn.execute(sanitized_sql).fetchall()

    def close(self) -> None:
        """Closes the underlying DuckDB connection."""
        self.conn.close()
