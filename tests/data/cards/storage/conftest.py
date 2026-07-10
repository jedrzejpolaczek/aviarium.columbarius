"""Shared fixtures for tests/data/cards/storage/.

`memory_con` is piloted here for test_base.py only (see Task 13 of the
maintainability remediation plan). The other test files in this directory
(test_bronze.py, test_silver.py, test_gold.py, test_health.py, etc.) still
use their own inline `duckdb.connect(":memory:")` calls; migrating them is
a separate, not-yet-scheduled follow-up.
"""

import duckdb
import pytest


@pytest.fixture
def memory_con():
    """In-memory DuckDB connection, closed after each test."""
    con = duckdb.connect(":memory:")
    yield con
    con.close()
