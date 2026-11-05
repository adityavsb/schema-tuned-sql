"""Pytest fixtures for schema-tuned-sql."""

import pytest

from src.utils.db import WarehouseDatabase
from warehouse.generate_data import generate_synthetic_warehouse


@pytest.fixture
def memory_db():
    """Provides a fresh in-memory synthetic DuckDB warehouse seeded deterministically."""
    conn = generate_synthetic_warehouse(
        db_path=":memory:", seed=42, num_accounts=50, num_events=2000
    )
    yield conn
    conn.close()


@pytest.fixture
def warehouse_db(memory_db):
    """Provides a WarehouseDatabase wrapper around the in-memory test warehouse."""
    db = WarehouseDatabase(db_path=":memory:")
    # Populate the locked database with our test fixtures
    from warehouse.generate_data import init_schema

    init_schema(db.conn)
    # Copy tables from memory_db or re-seed directly
    db.conn.close()

    # Re-initialize with seeded data
    seeded_conn = generate_synthetic_warehouse(
        db_path=":memory:", seed=42, num_accounts=50, num_events=2000
    )
    wrapper = WarehouseDatabase.__new__(WarehouseDatabase)
    wrapper.db_path = ":memory:"
    wrapper.conn = seeded_conn
    from src.utils.db import ReadOnlySQLGuardrail

    wrapper.guardrail = ReadOnlySQLGuardrail()
    yield wrapper
    wrapper.conn.close()
