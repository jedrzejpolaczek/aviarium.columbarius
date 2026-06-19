"""Unit tests for src/monitoring/event_trigger.py."""

from datetime import date

import duckdb
import pytest

from src.monitoring.event_trigger import get_todays_events, has_ban_event_today


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn_empty():
    """In-memory DuckDB with empty gold_events table."""
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE gold_events (
            event_date  DATE,
            format      VARCHAR,
            event_type  VARCHAR,
            card_name   VARCHAR
        )
    """)
    yield con
    con.close()


@pytest.fixture
def conn_with_events(conn_empty):
    """gold_events with two events on 2026-06-12 and one on 2026-06-10."""
    conn_empty.executemany(
        "INSERT INTO gold_events VALUES (?, ?, ?, ?)",
        [
            ("2026-06-12", "modern", "ban", "Bridge from Below"),
            ("2026-06-12", "legacy", "unban", "Skullclamp"),
            ("2026-06-10", "modern", "ban", "Hogaak, Arisen Necropolis"),
        ],
    )
    return conn_empty


# ---------------------------------------------------------------------------
# has_ban_event_today
# ---------------------------------------------------------------------------


def test_has_ban_event_returns_true_when_event_on_date(conn_with_events):
    assert has_ban_event_today(conn_with_events, check_date=date(2026, 6, 12)) is True


def test_has_ban_event_returns_false_when_no_event_on_date(conn_with_events):
    assert has_ban_event_today(conn_with_events, check_date=date(2026, 6, 11)) is False


def test_has_ban_event_returns_false_on_empty_table(conn_empty):
    assert has_ban_event_today(conn_empty, check_date=date(2026, 6, 12)) is False


def test_has_ban_event_works_with_single_event(conn_with_events):
    assert has_ban_event_today(conn_with_events, check_date=date(2026, 6, 10)) is True


def test_has_ban_event_returns_bool(conn_with_events):
    result = has_ban_event_today(conn_with_events, check_date=date(2026, 6, 12))
    assert isinstance(result, bool)


def test_has_ban_event_does_not_match_other_dates(conn_with_events):
    assert has_ban_event_today(conn_with_events, check_date=date(2025, 6, 12)) is False


# ---------------------------------------------------------------------------
# get_todays_events
# ---------------------------------------------------------------------------


def test_get_todays_events_returns_list(conn_with_events):
    result = get_todays_events(conn_with_events, check_date=date(2026, 6, 12))
    assert isinstance(result, list)


def test_get_todays_events_returns_correct_count(conn_with_events):
    result = get_todays_events(conn_with_events, check_date=date(2026, 6, 12))
    assert len(result) == 2


def test_get_todays_events_returns_dicts(conn_with_events):
    result = get_todays_events(conn_with_events, check_date=date(2026, 6, 12))
    for item in result:
        assert isinstance(item, dict)


def test_get_todays_events_has_event_type_key(conn_with_events):
    result = get_todays_events(conn_with_events, check_date=date(2026, 6, 12))
    for item in result:
        assert "event_type" in item


def test_get_todays_events_has_card_name_key(conn_with_events):
    result = get_todays_events(conn_with_events, check_date=date(2026, 6, 12))
    for item in result:
        assert "card_name" in item


def test_get_todays_events_has_format_key(conn_with_events):
    result = get_todays_events(conn_with_events, check_date=date(2026, 6, 12))
    for item in result:
        assert "format" in item


def test_get_todays_events_returns_empty_on_no_match(conn_with_events):
    result = get_todays_events(conn_with_events, check_date=date(2026, 6, 11))
    assert result == []


def test_get_todays_events_returns_empty_list_on_empty_table(conn_empty):
    result = get_todays_events(conn_empty, check_date=date(2026, 6, 12))
    assert result == []


def test_get_todays_events_card_names_correct(conn_with_events):
    result = get_todays_events(conn_with_events, check_date=date(2026, 6, 12))
    names = {r["card_name"] for r in result}
    assert names == {"Bridge from Below", "Skullclamp"}


def test_get_todays_events_single_event_correct_format(conn_with_events):
    result = get_todays_events(conn_with_events, check_date=date(2026, 6, 10))
    assert len(result) == 1
    assert result[0]["format"] == "modern"
    assert result[0]["event_type"] == "ban"
