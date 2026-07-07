"""Shared fixtures for tests/data/cards/storage/."""

import duckdb
import pytest


@pytest.fixture
def memory_con():
    """In-memory DuckDB connection, closed after each test."""
    con = duckdb.connect(":memory:")
    yield con
    con.close()
