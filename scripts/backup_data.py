"""Local backup of DuckDB data files and MLflow tracking DB/artefacts.

Copies the Bronze/Silver/Gold DuckDB files, ``mlflow.db``, and ``mlruns/``
into a timestamped snapshot directory under ``--backup-dir`` (default
``backups/``), then deletes snapshots beyond the last ``--keep-last`` (
default 7). Point ``--backup-dir`` at an external drive or a cloud-synced
folder to get off-host protection — this script itself has no cloud
dependency, it only knows how to copy files into a directory you choose.

Run this AFTER the daily pipeline and monitor step (``make pipeline &&
make monitor``) so no DuckDB/SQLite file is mid-write when copied — see
README "Monitoring & Scheduled Retraining" for the existing schedule this
slots into.

Usage:
    python -m scripts.backup_data
    python -m scripts.backup_data --backup-dir D:/mtg-backups --keep-last 14
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import shutil as shutil  # explicit re-export: tests patch backup_data.shutil.copy2,
# which requires this module to explicitly re-export the name under mypy --strict
# (no_implicit_reexport) — a plain `import shutil` makes the attribute invisible
# to importers even though it works fine at runtime.

from src.data.cards.pipelines import load_config
from src.logger import get_logger, setup_logging
from src.monitoring.alerts import send_alert

logger = get_logger(__name__)

DEFAULT_BACKUP_DIR = Path("backups")
DEFAULT_KEEP_LAST = 7
MLFLOW_DB_PATH = Path("mlflow.db")
MLRUNS_DIR = Path("mlruns")


def _tiered_duckdb_sources(config_path: str) -> list[tuple[str, Path]]:
    """Return (tier_name, path) pairs for Bronze/Silver/Gold DuckDB files.

    Named by tier (not basename) because all three files are conventionally
    called "cards.duckdb" in this project — copying by basename alone would
    silently overwrite one tier's backup with another's.
    """
    config = load_config(config_path)
    storage = config["storage"]
    return [
        ("bronze", Path(storage["bronze_duckdb_path"])),
        ("silver", Path(storage["silver_duckdb_path"])),
        ("gold", Path(storage["gold_duckdb_path"])),
    ]


def _copy_duckdb(tier: str, src: Path, dest_dir: Path) -> bool:
    if not src.exists():
        logger.warning("Backup source not found, skipping: %s", src)
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{tier}.duckdb"
    shutil.copy2(src, dest)
    logger.info("Backed up %s -> %s", src, dest)
    return True


def _copy_if_exists(src: Path, dest_dir: Path) -> bool:
    """Copy *src* (file or directory) into *dest_dir*, keeping its own name."""
    if not src.exists():
        logger.warning("Backup source not found, skipping: %s", src)
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    logger.info("Backed up %s -> %s", src, dest)
    return True


def run_backup(
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    keep_last: int = DEFAULT_KEEP_LAST,
    config_path: str = "configs/data_sources.yaml",
) -> Path:
    """Copy all data/model artefacts into a new timestamped snapshot dir.

    Returns the created snapshot directory.

    Raises:
        FileNotFoundError: If none of the backup sources (Bronze/Silver/Gold
            DuckDB, mlflow.db, mlruns/) exist — nothing was backed up.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_dir = backup_dir / timestamp

    copied_any = False
    try:
        for tier, src in _tiered_duckdb_sources(config_path):
            if _copy_duckdb(tier, src, snapshot_dir):
                copied_any = True
        for src in (MLFLOW_DB_PATH, MLRUNS_DIR):
            if _copy_if_exists(src, snapshot_dir):
                copied_any = True
    except OSError:
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        raise

    if not copied_any:
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        raise FileNotFoundError(
            "No backup sources found (Bronze/Silver/Gold DuckDB, mlflow.db, "
            "mlruns/ all missing) — nothing to back up."
        )

    _prune_old_snapshots(backup_dir, keep_last)
    return snapshot_dir


_SNAPSHOT_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def _prune_old_snapshots(backup_dir: Path, keep_last: int) -> None:
    """Delete all but the *keep_last* most recent timestamped snapshot dirs.

    Snapshot directory names are ``YYYY-MM-DD_HH-MM-SS`` (see run_backup),
    which sorts lexicographically in the same order as chronologically.
    Only directories matching that exact pattern are considered prunable —
    a misconfigured ``--backup-dir`` (pointed at an unrelated existing
    directory) or a manually-added folder under ``backups/`` must never be
    deleted by this function.
    """
    if not backup_dir.exists() or keep_last <= 0:
        return
    snapshots = sorted(
        p
        for p in backup_dir.iterdir()
        if p.is_dir() and _SNAPSHOT_NAME_RE.match(p.name)
    )
    for old in snapshots[:-keep_last]:
        shutil.rmtree(old)
        logger.info("Pruned old backup: %s", old)


def main(argv: list[str] | None = None) -> int:
    """Run the backup CLI.

    ``argv`` defaults to an empty argument list (not ``sys.argv``) so this
    function is safe to call directly — e.g. from tests — without picking up
    the caller's own command-line arguments (such as pytest's). Real CLI
    invocation goes through the ``__main__`` block below, which explicitly
    passes ``sys.argv[1:]``.
    """
    setup_logging(log_dir=Path("logs"))
    parser = argparse.ArgumentParser(
        description="Back up DuckDB data and MLflow artefacts to a local (or "
        "mounted external/cloud-synced) directory."
    )
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--keep-last", type=int, default=DEFAULT_KEEP_LAST)
    args = parser.parse_args(argv if argv is not None else [])

    try:
        snapshot_dir = run_backup(Path(args.backup_dir), args.keep_last)
    except Exception as exc:
        logger.error("Backup failed: %s", exc)
        send_alert("Backup failed", str(exc))
        return 1

    logger.info("Backup complete: %s", snapshot_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
