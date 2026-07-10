"""Unit tests for scripts/backup_data.py."""

from unittest.mock import MagicMock

import pytest

from scripts import backup_data


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    """Build a fake project tree with Bronze/Silver/Gold DuckDB stubs,
    mlflow.db, and mlruns/, plus a matching data_sources.yaml config.
    """
    (tmp_path / "data" / "bronze").mkdir(parents=True)
    (tmp_path / "data" / "silver").mkdir(parents=True)
    (tmp_path / "data" / "gold").mkdir(parents=True)
    (tmp_path / "data" / "bronze" / "cards.duckdb").write_text("bronze")
    (tmp_path / "data" / "silver" / "cards.duckdb").write_text("silver")
    (tmp_path / "data" / "gold" / "cards.duckdb").write_text("gold")
    (tmp_path / "mlflow.db").write_text("mlflow")
    mlruns = tmp_path / "mlruns" / "0"
    mlruns.mkdir(parents=True)
    (mlruns / "meta.yaml").write_text("run: 1")

    config_path = tmp_path / "data_sources.yaml"
    config_path.write_text(
        "storage:\n"
        f'  bronze_duckdb_path: "{(tmp_path / "data/bronze/cards.duckdb").as_posix()}"\n'
        f'  silver_duckdb_path: "{(tmp_path / "data/silver/cards.duckdb").as_posix()}"\n'
        f'  gold_duckdb_path: "{(tmp_path / "data/gold/cards.duckdb").as_posix()}"\n'
    )

    monkeypatch.setattr(backup_data, "MLFLOW_DB_PATH", tmp_path / "mlflow.db")
    monkeypatch.setattr(backup_data, "MLRUNS_DIR", tmp_path / "mlruns")
    return tmp_path, config_path


def test_run_backup_returns_timestamped_snapshot_dir_under_backup_dir(fake_project):
    tmp_path, config_path = fake_project
    backup_dir = tmp_path / "backups"

    snapshot_dir = backup_data.run_backup(
        backup_dir=backup_dir, keep_last=7, config_path=str(config_path)
    )

    assert snapshot_dir.parent == backup_dir
    assert snapshot_dir.exists()


def test_run_backup_copies_bronze_silver_gold_mlflow_and_mlruns(fake_project):
    tmp_path, config_path = fake_project
    backup_dir = tmp_path / "backups"

    snapshot_dir = backup_data.run_backup(
        backup_dir=backup_dir, keep_last=7, config_path=str(config_path)
    )

    # Three DuckDB files all happen to share the basename "cards.duckdb" in
    # this project's layout, so backup_data must namespace them by source
    # tier rather than flatten to basenames — verify no data was clobbered.
    contents = {p.read_text() for p in snapshot_dir.rglob("*.duckdb")}
    assert contents == {"bronze", "silver", "gold"}
    assert (snapshot_dir / "mlflow.db").read_text() == "mlflow"
    assert (snapshot_dir / "mlruns" / "0" / "meta.yaml").exists()


def test_run_backup_raises_when_nothing_to_back_up(tmp_path):
    config_path = tmp_path / "data_sources.yaml"
    config_path.write_text(
        "storage:\n"
        '  bronze_duckdb_path: "nope/bronze.duckdb"\n'
        '  silver_duckdb_path: "nope/silver.duckdb"\n'
        '  gold_duckdb_path: "nope/gold.duckdb"\n'
    )

    with pytest.raises(FileNotFoundError):
        backup_data.run_backup(
            backup_dir=tmp_path / "backups",
            keep_last=7,
            config_path=str(config_path),
        )


def test_prune_old_snapshots_keeps_only_the_last_n(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    for name in [
        "2026-01-01_00-00-00",
        "2026-01-02_00-00-00",
        "2026-01-03_00-00-00",
        "2026-01-04_00-00-00",
    ]:
        (backup_dir / name).mkdir()

    backup_data._prune_old_snapshots(backup_dir, keep_last=2)

    remaining = sorted(p.name for p in backup_dir.iterdir())
    assert remaining == ["2026-01-03_00-00-00", "2026-01-04_00-00-00"]


def test_main_returns_1_and_alerts_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(backup_data, "setup_logging", MagicMock())
    mock_send_alert = MagicMock()
    monkeypatch.setattr(backup_data, "send_alert", mock_send_alert)
    monkeypatch.setattr(
        backup_data,
        "run_backup",
        MagicMock(side_effect=FileNotFoundError("nothing to back up")),
    )

    exit_code = backup_data.main()

    assert exit_code == 1
    mock_send_alert.assert_called_once()


def test_main_returns_0_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(backup_data, "setup_logging", MagicMock())
    monkeypatch.setattr(
        backup_data, "run_backup", MagicMock(return_value=tmp_path / "backups/x")
    )
    mock_send_alert = MagicMock()
    monkeypatch.setattr(backup_data, "send_alert", mock_send_alert)

    exit_code = backup_data.main()

    assert exit_code == 0
    mock_send_alert.assert_not_called()
