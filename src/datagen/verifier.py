"""Execution verification filter for teacher-distilled SQL queries."""

from __future__ import annotations

from typing import Any

import duckdb
import sqlglot

from src.eval.execution import compare_result_sets
from src.eval.runner import extract_sql, is_refusal_response
from src.utils.db import ReadOnlySQLGuardrail


class SQLVerifier:
    """Validates candidate SQL syntax, read-only AST safety, execution, and optional dual-teacher agreement."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self.guardrail = ReadOnlySQLGuardrail()

    def verify_pair(
        self,
        question: str,
        raw_sql_or_completion: str,
        is_negative_query: bool = False,
        secondary_completion: str | None = None,
    ) -> tuple[bool, str, str, list[tuple[Any, ...]]]:
        """Verifies a candidate (question, completion) pair.

        Returns (accepted, reason, clean_target_completion, result_rows).
        """
        if is_negative_query:
            if is_refusal_response(raw_sql_or_completion):
                return True, "accepted_refusal", "REFUSAL: UNANSWERABLE_FROM_SCHEMA", []
            return False, "negative_query_not_refused", "", []

        if is_refusal_response(raw_sql_or_completion):
            return False, "unexpected_refusal_on_positive", "", []

        sql = extract_sql(raw_sql_or_completion)
        if not sql:
            return False, "empty_sql", "", []

        # 1. Parse check in DuckDB dialect
        try:
            parsed = sqlglot.parse_one(sql, read="duckdb")
            if parsed is None:
                return False, "sqlglot_parse_none", "", []
        except Exception as e:
            return False, f"sqlglot_parse_error: {type(e).__name__}", "", []

        # 2. Read-only AST guardrail check
        if not self.guardrail.is_safe(sql):
            return False, "guardrail_rejected", "", []

        # 3. Execute in DuckDB and verify non-empty result
        try:
            rows = self.conn.execute(sql).fetchall()
        except Exception as e:
            return False, f"execution_error: {type(e).__name__}", "", []

        if not rows:
            return False, "empty_result_set", "", []

        # Reject single-cell NULL results (e.g. SUM over empty filter)
        if len(rows) == 1 and len(rows[0]) == 1 and rows[0][0] is None:
            return False, "null_scalar_result", "", []

        # 4. Optional dual-teacher execution agreement
        if secondary_completion is not None:
            sec_sql = extract_sql(secondary_completion)
            if not sec_sql or not self.guardrail.is_safe(sec_sql):
                return False, "secondary_teacher_invalid", "", []
            try:
                sec_rows = self.conn.execute(sec_sql).fetchall()
            except Exception:
                return False, "secondary_teacher_exec_error", "", []
            if not compare_result_sets(rows, sec_rows, order_matters=False):
                return False, "dual_teacher_disagreement", "", []

        normalized_sql = parsed.sql(dialect="duckdb")
        return True, "accepted", normalized_sql, rows
