"""Shared fixtures for the top-level test suite.

Fixtures here are visible to every test package (unlike the
directory-scoped conftest.py files under tests/app/ and
tests/data/cards/storage/), so this file is reserved for fixtures needed
across multiple, otherwise-unrelated test directories.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from src.data.cards.storage.schema import find_mistyped_date_columns

_DUCKDB_CONNECTIONS = pytest.StashKey[list[tuple[duckdb.DuckDBPyConnection, str]]]()
_REAL_DUCKDB_CONNECT = duckdb.connect


@pytest.fixture(autouse=True)
def _track_duckdb_connections(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record every DuckDB connection opened during a test, by fixtures or code.

    Feeds the schema guard in :func:`pytest_runtest_call` below. Production
    code and tests both call ``duckdb.connect`` through the module attribute,
    so replacing it here sees every connection without changing either.
    """
    connections: list[tuple[duckdb.DuckDBPyConnection, str]] = []

    def tracking_connect(*args: object, **kwargs: object) -> duckdb.DuckDBPyConnection:
        con = _REAL_DUCKDB_CONNECT(*args, **kwargs)  # type: ignore[arg-type]
        database = args[0] if args else kwargs.get("database", ":memory:")
        connections.append((con, str(database)))
        return con

    monkeypatch.setattr(duckdb, "connect", tracking_connect)
    request.node.stash[_DUCKDB_CONNECTIONS] = connections


def _mistyped_date_columns(
    con: duckdb.DuckDBPyConnection, database: str
) -> list[tuple[str, str, str]]:
    """Inspect one tracked connection, reopening its file if it was closed."""
    try:
        return find_mistyped_date_columns(con, include_attached=True)
    except duckdb.Error:
        pass  # closed by the test — fall back to the file, if there is one
    if database in ("", ":memory:") or not Path(database).is_file():
        return []
    try:
        reopened = _REAL_DUCKDB_CONNECT(database, read_only=True)
    except duckdb.Error:
        return []  # still held open elsewhere; that connection was inspected
    try:
        return [
            (f"{Path(database).name}:{t}", c, d)
            for t, c, d in find_mistyped_date_columns(reopened)
        ]
    finally:
        reopened.close()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> object:
    """Fail a passing test whose DuckDB tables break the production date schema.

    Runs right after the test body. Connections still open are inspected
    directly; file databases whose connection the test already closed (a
    ``with GoldStorage(...)`` block, a seeding helper) are reopened read-only.
    Every ``snapshot_date`` / ``tournament_date`` column in any table the
    test (or the code under test) created must be DATE — the invariant in
    ``src/data/cards/storage/schema.py``.

    Why a guard and not a convention: on 2026-09-28 the Silver tests declared
    ``snapshot_date VARCHAR`` long after production had moved to DATE, so they
    kept passing against a schema that no longer existed while the real run
    died on ``TRIM(DATE)``. The same check catches production code that writes
    the wrong type, as the 2026-09-29 Gold rebuild did with TIMESTAMP.

    Tests that need a legacy schema on purpose (e.g. tolerating a pre-0.2.0
    database) opt out with ``@pytest.mark.legacy_date_schema``.
    """
    result = yield
    if item.get_closest_marker("legacy_date_schema"):
        return result
    problems: set[tuple[str, str, str]] = set()
    for con, database in item.stash.get(_DUCKDB_CONNECTIONS, []):
        problems.update(_mistyped_date_columns(con, database))
    if problems:
        listed = ", ".join(f"{t}.{c} is {d}" for t, c, d in sorted(problems))
        pytest.fail(
            f"Date columns must be DATE as in production: {listed}. "
            "Build fixture tables with DuckDBWriter or cast with ::DATE; mark "
            "@pytest.mark.legacy_date_schema only for deliberate legacy schemas.",
            pytrace=False,
        )
    return result


@pytest.fixture(autouse=True)
def _no_real_log_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop any script's main() from creating real logs/pipeline_*.log files.

    All six scripts bind setup_logging into their own namespace with a
    module-level ``from src.logger import setup_logging``, so patching the
    source has no effect — each binding has to be replaced. Rather than listing
    them (the list went stale the moment a seventh script appeared), this walks
    the already-imported ``scripts.*`` modules and patches whatever it finds.

    Why it matters beyond tidiness: setup_logging prunes ``log_dir`` to
    ``keep_last_logs=90`` timestamped groups. Test-created logs therefore do not
    grow the directory — they *evict real pipeline logs* to stay under the cap.
    A stable file count is exactly what that looks like, which is how this went
    unnoticed once already.
    """
    for name, module in list(sys.modules.items()):
        if name.startswith("scripts") and hasattr(module, "setup_logging"):
            monkeypatch.setattr(module, "setup_logging", MagicMock(return_value=None))


@pytest.fixture(autouse=True)
def _isolate_alert_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test's send_alert() calls out of the real logs/alerts.jsonl.

    send_alert resolves its target from the module-level ALERTS_LOG_PATH at
    call time, so redirecting that constant is enough to cover callers that
    do not pass alerts_log_path explicitly.

    Without this, the production alert log is where test noise goes: all 190
    records it held on 2026-09-25 were written by pytest (124 of them
    "Monitor: Gold DB missing" naming pytest tmp paths). The log is documented
    as the one channel that always succeeds and is meant to be replayable —
    it cannot be that and a test scratch file at the same time.
    """
    from src.monitoring import alerts

    monkeypatch.setattr(alerts, "ALERTS_LOG_PATH", tmp_path / "alerts.jsonl")


@pytest.fixture
def tiny_gold_conn():
    """In-memory DuckDB connection pre-populated with a 2-card/2-snapshot
    gold_price_features + gold_card_features dataset.

    Deliberately tiny: it trips retrain()'s InsufficientDataError fallback
    instead of running a full walk-forward CV, which is what makes it
    usable for real-MLflow integration tests (tests/monitoring/
    test_retrain_integration.py, tests/scripts/test_check_and_retrain.py)
    without needing 50+ days of synthetic price history.
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE gold_price_features AS
        SELECT * FROM (VALUES
            ('uuid-1', DATE '2026-06-01', 1.5, 100.0, NULL),
            ('uuid-1', DATE '2026-06-08', 1.8, 100.0, NULL),
            ('uuid-2', DATE '2026-06-01', 0.3, 200.0, NULL),
            ('uuid-2', DATE '2026-06-08', 0.4, 200.0, NULL)
        ) AS t(uuid, snapshot_date, eur, edhrec_rank, foil_premium)
    """)
    # edhrec_saltiness is required here (not in gold_price_features) because
    # IMPUTE_MEDIAN_COLS in src/ml/features/pipeline.py expects it, and in
    # production it is sourced from gold_card_features (see
    # GoldFeatureBuilders.build_card_features in
    # src/data/cards/storage/gold/features.py) — build_inference_features()
    # merges lag_df and card_df on uuid, so it must be present post-merge.
    con.execute("""
        CREATE TABLE gold_card_features AS
        SELECT * FROM (VALUES
            ('uuid-1', 'common', 3, 2.0, 1, false, false, true, NULL),
            ('uuid-2', 'rare',   1, 1.0, 1, false, false, true, NULL)
        ) AS t(uuid, rarity, print_count, mana_value, format_count,
                is_reserved, is_legendary, is_commander_legal, edhrec_saltiness)
    """)
    yield con
    con.close()
