"""Unit tests for scripts/train_model.py.

train_model.py's own logic is just two precondition checks (missing DB
file, empty gold_price_features) plus a call into
:func:`src.monitoring.retraining.retrain`. The heavier retrain machinery
itself is already covered by the real-MLflow integration tests in
tests/monitoring/test_retrain_integration.py and
tests/scripts/test_check_and_retrain.py, so here ``retrain`` (and
``setup_experiment``, which would otherwise point MLflow at the real
project-root mlflow.db) are mocked to isolate train_model.main()'s own
branching.
"""

import sys
from unittest.mock import MagicMock

import duckdb
import pytest

from scripts import train_model


def test_main_exits_1_when_db_missing(tmp_path, monkeypatch):
    missing_path = str(tmp_path / "does_not_exist.duckdb")
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--db-path", missing_path])

    with pytest.raises(SystemExit) as exc_info:
        train_model.main()

    assert exc_info.value.code == 1


def test_main_exits_1_when_no_snapshot(tmp_path, monkeypatch):
    db_path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db_path))
    con.close()
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--db-path", str(db_path)])

    with pytest.raises(SystemExit) as exc_info:
        train_model.main()

    assert exc_info.value.code == 1


def test_main_calls_retrain_with_latest_snapshot(tmp_path, monkeypatch):
    db_path = tmp_path / "gold.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE gold_price_features AS
        SELECT * FROM (VALUES
            ('uuid-1', '2026-06-01', 1.5),
            ('uuid-1', '2026-06-08', 1.8)
        ) AS t(uuid, snapshot_date, eur)
    """)
    con.close()
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--db-path", str(db_path)])

    # setup_experiment() is bound directly into scripts.train_model's
    # namespace via `from ... import setup_experiment`, so it must be
    # patched there; left unmocked it would point MLflow at the real
    # project-root mlflow.db as a side effect of this test.
    monkeypatch.setattr(train_model, "setup_experiment", MagicMock())

    # retrain is imported with a *deferred* `from src.monitoring.retraining
    # import retrain` inside main(), so Python resolves the name from
    # src.monitoring.retraining's namespace at call time — patching
    # scripts.train_model.retrain would have no effect here.
    mock_retrain = MagicMock(return_value="run-abc")
    monkeypatch.setattr("src.monitoring.retraining.retrain", mock_retrain)

    train_model.main()

    mock_retrain.assert_called_once()
    call_args = mock_retrain.call_args
    assert call_args.args[1] == "2026-06-08"
