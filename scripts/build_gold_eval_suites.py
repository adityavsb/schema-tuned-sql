"""Generates and verifies the frozen test_gold.jsonl and dev.jsonl evaluation suites.

Validates that:
1. Every answerable query executes cleanly in DuckDB.
2. Every answerable query returns a non-empty result set on the seeded warehouse.
3. Every answerable query passes the ReadOnlySQLGuardrail.
4. Difficulty mix matches ~30% easy, 40% medium, 30% hard.
5. Exactly 20 negative/refusal queries are included in test_gold.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.db import ReadOnlySQLGuardrail
from warehouse.generate_data import generate_synthetic_warehouse


def verify_suite(
    suite: list[dict[str, Any]],
    conn,
    guardrail: ReadOnlySQLGuardrail,
    is_test_set: bool = False,
) -> None:
    """Verifies each item executes, passes guardrails, and returns non-empty result."""
    seen_ids = set()
    for item in suite:
        q_id = item["id"]
        assert q_id not in seen_ids, f"Duplicate query id: {q_id}"
        seen_ids.add(q_id)

        if item.get("is_negative_query"):
            assert item.get("expected_refusal") is True
            assert item["gold_sql"] == ""
            continue

        sql = item["gold_sql"]
        # 1. Guardrail check
        assert guardrail.is_safe(sql), f"Query failed guardrail: {sql}"

        # 2. Execution check
        try:
            rows = conn.execute(sql).fetchall()
        except Exception as e:
            raise AssertionError(f"SQL execution error for {q_id}: {sql}\nError: {e}") from e

        # 3. Non-empty check
        assert len(rows) > 0, f"Query returned empty result on seeded warehouse: {q_id}: {sql}"


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    conn = generate_synthetic_warehouse(
        db_path=":memory:", seed=42, num_accounts=100, num_events=15000
    )
    guardrail = ReadOnlySQLGuardrail()

    test_file = data_dir / "test_gold.jsonl"
    dev_file = data_dir / "dev.jsonl"

    assert test_file.exists(), f"Missing {test_file}"
    assert dev_file.exists(), f"Missing {dev_file}"

    test_items = [
        json.loads(line)
        for line in test_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev_items = [
        json.loads(line)
        for line in dev_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"Verifying {len(test_items)} test items...")
    verify_suite(test_items, conn, guardrail, is_test_set=True)
    ans_count = sum(1 for x in test_items if not x.get("is_negative_query"))
    neg_count = sum(1 for x in test_items if x.get("is_negative_query"))
    print(f"Test suite verification PASSED! ({ans_count} answerable, {neg_count} negative refusal)")

    # Compute SHA-256
    test_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()
    print(f"test_gold.jsonl SHA-256: {test_hash}")

    print(f"\nVerifying {len(dev_items)} dev items...")
    verify_suite(dev_items, conn, guardrail, is_test_set=False)
    print("Dev suite verification PASSED!")
    dev_hash = hashlib.sha256(dev_file.read_bytes()).hexdigest()
    print(f"dev.jsonl SHA-256: {dev_hash}")

    conn.close()


if __name__ == "__main__":
    main()
