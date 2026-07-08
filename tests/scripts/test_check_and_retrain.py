"""Unit tests for scripts/check_and_retrain.py."""

import json
from unittest.mock import MagicMock

import duckdb
import mlflow
import pytest

from scripts import check_and_retrain


def _make_fake_conn_with_snapshot(snapshot_date: str) -> MagicMock:
    """Build a fake DuckDB connection whose gold_price_features MAX(snapshot_date)
    resolves to `snapshot_date`.

    get_latest_gold_snapshot_date() first checks table presence via a
    ``SHOW TABLES`` query (get_tables) before running the MAX query, so the
    fake execute() must respond to both statements rather than always
    returning the same canned fetchone() result.
    """

    def _execute(sql, *args, **kwargs):
        result = MagicMock()
        if "SHOW TABLES" in sql:
            result.fetchall.return_value = [("gold_price_features",)]
        else:
            result.fetchone.return_value = (snapshot_date,)
        return result

    fake_conn = MagicMock()
    fake_conn.execute.side_effect = _execute
    return fake_conn


def test_main_returns_1_when_gold_db_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        check_and_retrain, "GOLD_DB_PATH", str(tmp_path / "missing.duckdb")
    )
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")

    exit_code = check_and_retrain.main()

    assert exit_code == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["result"] == "error"
    assert status["reason"] == "gold_db_missing"


def test_main_skips_retrain_when_no_trigger(tmp_path, monkeypatch):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(
        check_and_retrain.duckdb, "connect", lambda *a, **k: MagicMock()
    )
    monkeypatch.setattr(
        check_and_retrain, "should_retrain", lambda conn: (False, "no_trigger")
    )
    mock_retrain = MagicMock()
    monkeypatch.setattr(check_and_retrain, "retrain", mock_retrain)

    exit_code = check_and_retrain.main()

    assert exit_code == 0
    mock_retrain.assert_not_called()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["result"] == "no_retrain"
    assert status["reason"] == "no_trigger"


def test_main_retrains_when_triggered(tmp_path, monkeypatch):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")

    fake_conn = _make_fake_conn_with_snapshot("2026-07-01")
    monkeypatch.setattr(check_and_retrain.duckdb, "connect", lambda *a, **k: fake_conn)
    monkeypatch.setattr(
        check_and_retrain, "should_retrain", lambda conn: (True, "mape_threshold")
    )
    monkeypatch.setattr(
        check_and_retrain, "retrain", lambda conn, snapshot_date: "abc123"
    )

    exit_code = check_and_retrain.main()

    assert exit_code == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["result"] == "retrained"
    assert status["reason"] == "mape_threshold"
    assert status["run_id"] == "abc123"


def test_main_writes_error_status_when_retrain_raises(tmp_path, monkeypatch):
    db_path = tmp_path / "cards.duckdb"
    db_path.touch()
    monkeypatch.setattr(check_and_retrain, "GOLD_DB_PATH", str(db_path))
    monkeypatch.setattr(check_and_retrain, "STATUS_PATH", tmp_path / "status.json")

    fake_conn = _make_fake_conn_with_snapshot("2026-07-01")
    monkeypatch.setattr(check_and_retrain.duckdb, "connect", lambda *a, **k: fake_conn)
    monkeypatch.setattr(
        check_and_retrain, "should_retrain", lambda conn: (True, "mape_threshold")
    )

    def _raise(conn, snapshot_date):
        raise RuntimeError("mlflow boom")

    monkeypatch.setattr(check_and_retrain, "retrain", _raise)

    exit_code = check_and_retrain.main()

    assert exit_code == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["result"] == "error"
    assert status["reason"] == "retrain_failed"
    assert "mlflow boom" in status["error"]


@pytest.fixture(autouse=True)
def mlflow_tmp_for_real_retrain(tmp_path, monkeypatch):
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    yield
    if mlflow.active_run():
        mlflow.end_run()


def test_do_retrain_calls_real_retrain_and_writes_run_id(tmp_path):
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE gold_price_features AS
        SELECT * FROM (VALUES
            ('uuid-1', '2026-06-01', 1.5, 100.0, NULL),
            ('uuid-1', '2026-06-08', 1.8, 100.0, NULL),
            ('uuid-2', '2026-06-01', 0.3, 200.0, NULL),
            ('uuid-2', '2026-06-08', 0.4, 200.0, NULL)
        ) AS t(uuid, snapshot_date, eur, edhrec_rank, foil_premium)
    """)
    con.execute("""
        CREATE TABLE gold_card_features AS
        SELECT * FROM (VALUES
            ('uuid-1', 'common', 3, 2.0, 1, false, false, true, NULL),
            ('uuid-2', 'rare',   1, 1.0, 1, false, false, true, NULL)
        ) AS t(uuid, rarity, print_count, mana_value, format_count,
                is_reserved, is_legendary, is_commander_legal, edhrec_saltiness)
    """)

    ok, status = check_and_retrain._do_retrain(con, "2026-06-01", "mape_threshold")

    assert ok is True
    assert status["result"] == "retrained"
    assert status["reason"] == "mape_threshold"
    assert "run_id" in status and status["run_id"]
    con.close()
