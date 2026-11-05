"""Unit tests verifying frozen gold evaluation splits, checksums, and execution validity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pytest

from src.utils.db import ReadOnlySQLGuardrail
from warehouse.generate_data import generate_synthetic_warehouse

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FROZEN_TEST_HASH = "84a938a29aa438c1c901f4dfadbfe13af01451ef2c55177010493133f90b6b5c"
FROZEN_DEV_HASH = "2e5064b46fe8001332aff4b21398fa86d134fa15d26c1113606603c57e44925f"


@pytest.fixture(scope="module")
def seeded_duckdb():
    conn = generate_synthetic_warehouse(
        db_path=":memory:", seed=42, num_accounts=100, num_events=15000
    )
    yield conn
    conn.close()


def test_test_gold_integrity_and_checksum(seeded_duckdb: duckdb.DuckDBPyConnection):
    test_path = DATA_DIR / "test_gold.jsonl"
    assert test_path.is_file(), f"Missing {test_path}"

    raw_bytes = test_path.read_bytes()
    computed_hash = hashlib.sha256(raw_bytes).hexdigest()
    assert computed_hash == FROZEN_TEST_HASH, (
        f"Frozen test_gold.jsonl checksum changed! Expected {FROZEN_TEST_HASH}, got {computed_hash}"
    )

    items = [
        json.loads(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(items) == 220, f"Expected 220 test items, found {len(items)}"

    answerable = [x for x in items if not x.get("is_negative_query")]
    refusal = [x for x in items if x.get("is_negative_query")]

    assert len(answerable) == 200
    assert len(refusal) == 20

    # Difficulty breakdown check
    easy = [x for x in answerable if x["difficulty"] == "easy"]
    med = [x for x in answerable if x["difficulty"] == "medium"]
    hard = [x for x in answerable if x["difficulty"] == "hard"]
    assert len(easy) == 60
    assert len(med) == 80
    assert len(hard) == 60

    # Read-only guardrail and execution check
    guardrail = ReadOnlySQLGuardrail()
    for item in answerable:
        sql = item["gold_sql"]
        assert guardrail.is_safe(sql), f"Query failed guardrail: {sql}"
        res = seeded_duckdb.execute(sql).fetchall()
        assert len(res) > 0, f"Query returned empty result on seeded warehouse: {item['id']}"


def test_dev_split_integrity_and_checksum(seeded_duckdb: duckdb.DuckDBPyConnection):
    dev_path = DATA_DIR / "dev.jsonl"
    assert dev_path.is_file(), f"Missing {dev_path}"

    raw_bytes = dev_path.read_bytes()
    computed_hash = hashlib.sha256(raw_bytes).hexdigest()
    assert computed_hash == FROZEN_DEV_HASH, (
        f"Frozen dev.jsonl checksum changed! Expected {FROZEN_DEV_HASH}, got {computed_hash}"
    )

    items = [
        json.loads(line)
        for line in dev_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(items) == 100

    guardrail = ReadOnlySQLGuardrail()
    for item in items:
        sql = item["gold_sql"]
        assert guardrail.is_safe(sql), f"Query failed guardrail: {sql}"
        res = seeded_duckdb.execute(sql).fetchall()
        assert len(res) > 0


def test_zero_leakage_between_test_and_dev():
    test_items = [
        json.loads(line)
        for line in (DATA_DIR / "test_gold.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev_items = [
        json.loads(line)
        for line in (DATA_DIR / "dev.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    test_questions = {x["question"].strip().lower() for x in test_items}
    dev_questions = {x["question"].strip().lower() for x in dev_items}

    overlap = test_questions.intersection(dev_questions)
    assert len(overlap) == 0, f"Detected leakage between test and dev splits: {overlap}"
