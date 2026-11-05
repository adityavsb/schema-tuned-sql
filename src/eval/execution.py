"""Execution evaluation engine comparing predicted SQL query results against gold truth in DuckDB."""

from __future__ import annotations

import collections
import datetime
import decimal
import math
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp


def normalize_value(val: Any, float_tol: float = 1e-4) -> Any:
    """Normalizes cell values across database types (floats, decimals, dates, nulls)."""
    if val is None:
        return None
    if isinstance(val, (float, decimal.Decimal)):
        # Round float to precision bucket for hashable Counter multiset comparison
        f_val = float(val)
        if math.isnan(f_val):
            return "NaN"
        if math.isinf(f_val):
            return "Infinity" if f_val > 0 else "-Infinity"
        return round(f_val, 4)
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.isoformat()
    if isinstance(val, str):
        return val.strip()
    return val


def normalize_row(row: tuple[Any, ...], float_tol: float = 1e-4) -> tuple[Any, ...]:
    """Normalizes all column values in a database result row."""
    return tuple(normalize_value(v, float_tol) for v in row)


def rows_are_close(row1: tuple[Any, ...], row2: tuple[Any, ...], float_tol: float = 1e-4) -> bool:
    """Checks whether two individual rows are value-equivalent with float tolerance."""
    if len(row1) != len(row2):
        return False
    for v1, v2 in zip(row1, row2):
        if v1 is None and v2 is None:
            continue
        if v1 is None or v2 is None:
            return False
        if isinstance(v1, (float, int, decimal.Decimal)) and isinstance(
            v2, (float, int, decimal.Decimal)
        ):
            f1, f2 = float(v1), float(v2)
            if math.isnan(f1) and math.isnan(f2):
                continue
            if not math.isclose(f1, f2, rel_tol=float_tol, abs_tol=1e-5):
                return False
        elif str(v1).strip() != str(v2).strip():
            return False
    return True


def compare_result_sets(
    gold_rows: list[tuple[Any, ...]],
    pred_rows: list[tuple[Any, ...]],
    order_matters: bool = False,
    float_tol: float = 1e-4,
) -> bool:
    """Compares database result sets.

    If order_matters is False, compares as unordered multisets (Counter).
    If order_matters is True, compares in exact sequential order.
    """
    if len(gold_rows) != len(pred_rows):
        return False

    if order_matters:
        return all(rows_are_close(g, p, float_tol) for g, p in zip(gold_rows, pred_rows))

    # Fast path: normalized tuples Counter multiset equality
    norm_gold = [normalize_row(r, float_tol) for r in gold_rows]
    norm_pred = [normalize_row(r, float_tol) for r in pred_rows]
    if collections.Counter(norm_gold) == collections.Counter(norm_pred):
        return True

    # Fallback path for near-boundary float comparisons
    gold_unmatched = list(gold_rows)
    for p_row in pred_rows:
        matched_idx = -1
        for i, g_row in enumerate(gold_unmatched):
            if rows_are_close(g_row, p_row, float_tol):
                matched_idx = i
                break
        if matched_idx >= 0:
            gold_unmatched.pop(matched_idx)
        else:
            return False

    return len(gold_unmatched) == 0


def has_order_by(sql_query: str, dialect: str = "duckdb") -> bool:
    """Inspects AST to check if query has an explicit top-level ORDER BY clause."""
    try:
        parsed = sqlglot.parse_one(sql_query, read=dialect)
        return bool(parsed.find(exp.Order))
    except Exception:
        return "order by" in sql_query.lower()


def evaluate_execution(
    gold_sql: str,
    pred_sql: str,
    db_conn: duckdb.DuckDBPyConnection,
    float_tol: float = 1e-4,
) -> tuple[bool, str, list[tuple[Any, ...]] | None]:
    """Executes gold and predicted SQL queries against DuckDB and compares outputs.

    Returns (is_match, message, predicted_rows).
    """
    # 1. Execute Gold SQL
    try:
        gold_rows = db_conn.execute(gold_sql).fetchall()
    except Exception as e:
        return False, f"Gold SQL Execution Error: {e}", None

    # 2. Execute Predicted SQL
    try:
        pred_rows = db_conn.execute(pred_sql).fetchall()
    except Exception as e:
        return False, f"Predicted SQL Execution Error: {e}", None

    # 3. Determine if ordering is required by gold query
    order_matters = has_order_by(gold_sql)

    # 4. Compare result sets
    is_match = compare_result_sets(
        gold_rows, pred_rows, order_matters=order_matters, float_tol=float_tol
    )
    status_msg = "Match" if is_match else "Mismatch"
    return is_match, status_msg, pred_rows
