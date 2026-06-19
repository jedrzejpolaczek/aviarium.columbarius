"""Event-triggered retraining check based on format bans and unbans.

Why event-triggered retrain:
    A ban in a competitive format (e.g. Modern, Legacy) can drop a card's
    price by 50–90% within 24 hours.  Waiting for three consecutive days of
    high MAPE before retraining wastes two days during which the model gives
    materially wrong predictions.  Checking for ban events first lets the
    system react immediately.

Data source:
    ``gold_events`` is populated by ``GoldSignalBuilders.build_events()`` in
    ``src/data/cards/storage/gold/signals.py``.  The table stores one row per
    ban/unban announcement with the format and card name affected.

Schema of ``gold_events``:
    event_date  DATE     -- date the ban/unban took effect
    format      VARCHAR  -- format name (e.g. 'modern', 'legacy', 'standard')
    event_type  VARCHAR  -- 'ban' or 'unban'
    card_name   VARCHAR  -- name of the affected card
"""

from datetime import date
from typing import cast

import duckdb


def has_ban_event_today(
    conn: duckdb.DuckDBPyConnection,
    check_date: date | None = None,
) -> bool:
    """Return ``True`` if any ban or unban event occurred on ``check_date``.

    Queries ``gold_events`` for all rows matching the given date.  A non-zero
    count means at least one format change was announced; the caller should
    trigger an immediate retrain rather than waiting for the MAPE threshold.

    Args:
        conn:       Open DuckDB connection with ``gold_events`` in scope.
        check_date: Date to check. Defaults to ``date.today()`` when ``None``.

    Returns:
        ``True`` if at least one event exists on ``check_date``, ``False``
        otherwise (including when ``gold_events`` is empty).
    """
    today = check_date or date.today()
    result = conn.execute(
        "SELECT COUNT(*) FROM gold_events WHERE event_date = ?",
        [today],
    ).fetchone()
    return result is not None and result[0] > 0


def get_todays_events(
    conn: duckdb.DuckDBPyConnection,
    check_date: date | None = None,
) -> list[dict[str, object]]:
    """Return all events that occurred on ``check_date`` as a list of dicts.

    Used for structured logging before a triggered retrain so operators know
    exactly which ban/unban caused the retraining.

    Args:
        conn:       Open DuckDB connection with ``gold_events`` in scope.
        check_date: Date to query. Defaults to ``date.today()`` when ``None``.

    Returns:
        List of dicts with keys ``event_date``, ``format``, ``event_type``,
        ``card_name``.  Empty list when no events exist for the date.
    """
    today = check_date or date.today()
    records = (
        conn.execute(
            """
        SELECT event_date, format, event_type, card_name
        FROM gold_events
        WHERE event_date = ?
        """,
            [today],
        )
        .df()
        .to_dict("records")
    )
    return cast(list[dict[str, object]], records)
