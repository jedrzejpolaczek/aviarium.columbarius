"""Schema invariants shared by every storage tier — and by the tests.

Today there is one: the date columns below are ``DATE`` in every table that has
them. It has been broken three ways so far, each time without a failing test:

- all three tiers stored them as ``VARCHAR`` until the 2026-09-26 migration;
- the Gold rebuild of 2026-09-29 recreated them as ``TIMESTAMP`` (pandas hands
  a DATE column back as ``datetime64``);
- Silver's test fixtures kept declaring ``VARCHAR`` after production had moved
  to ``DATE``, so the suite stayed green while the 2026-09-28 run died on
  ``TRIM(DATE)``.

:class:`~src.data.cards.storage.base.writers.DuckDBWriter` enforces the rule on
write, ``scripts/migrate_snapshot_date_to_date.py`` repairs existing files, and
``tests/conftest.py`` uses :func:`find_mistyped_date_columns` to keep test
fixtures on the production schema.
"""

import duckdb

DATE_COLUMNS: tuple[str, ...] = ("snapshot_date", "tournament_date")
"""Columns that must be DATE wherever they appear."""


def find_mistyped_date_columns(
    con: duckdb.DuckDBPyConnection,
    include_attached: bool = False,
) -> list[tuple[str, str, str]]:
    """Return ``(table, column, current_type)`` for every date column that is not DATE.

    Covers base tables only — not views or registered DataFrames. By default
    only the connection's own database and schema are inspected, and table
    names come back unqualified, ready for ``ALTER TABLE``. With
    ``include_attached`` every ATTACHed database is inspected as well and names
    are qualified ``database.schema.table``; the test-suite guard needs that
    because Silver and Gold read their input tier through ATTACH.

    Anything other than DATE is reported: VARCHAR and TIMESTAMP are the two
    routes seen so far, and a third would be caught too rather than silently
    skipped.
    """
    placeholders = ", ".join("?" for _ in DATE_COLUMNS)
    scope = (
        "c.database_name NOT IN ('system', 'temp')"
        if include_attached
        else "c.database_name = current_database() AND c.schema_name = current_schema()"
    )
    rows = con.execute(
        f"""
        SELECT c.database_name, c.schema_name, c.table_name, c.column_name, c.data_type
        FROM duckdb_columns() c
        JOIN duckdb_tables() t ON c.table_oid = t.table_oid
        WHERE {scope}
          AND c.column_name IN ({placeholders})
          AND c.data_type <> 'DATE'
        ORDER BY c.database_name, c.schema_name, c.table_name, c.column_name
        """,
        list(DATE_COLUMNS),
    ).fetchall()
    return [
        (
            f"{db}.{schema}.{table}" if include_attached else str(table),
            str(col),
            str(dtype),
        )
        for db, schema, table, col, dtype in rows
    ]
