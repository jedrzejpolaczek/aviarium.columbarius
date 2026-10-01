"""Daily model monitoring with conditional retraining.

Run on a daily schedule (cron / Windows Task Scheduler) after the ETL
pipeline (`make pipeline`). Each run (ADR-034):

1. records the served model's predictions for today's Gold snapshot
   (``src.monitoring.prediction_tracker``), so they can be scored a week later;
2. scores every matured prediction date against the naive "no change"
   forecast, and alerts when the model has been worse for 3 dates in a row;
3. checks the distribution of 7-day price returns for drift
   (``src.monitoring.drift``) and alerts on a shift — an alert only;
4. retrains through :func:`should_retrain` / :func:`retrain` only when a real
   trigger fires (ban/unban event, or step 2's degradation), instead of
   retraining unconditionally like ``scripts/train_model.py`` does.

A failure in steps 1–3 is alerted and recorded, never fatal: the retrain
decision and the heartbeat still happen.

Writes a JSON status file to ``logs/last_check_status.json`` on every run
so an operator (or a future alerting tool) can check the outcome without
reading log files. See docs/runbooks/model-incidents.md for what to do
with each result.

Usage:
    python -m scripts.check_and_retrain
"""

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

import duckdb as duckdb  # explicit re-export: tests patch check_and_retrain.duckdb.connect,
# which requires this module to explicitly re-export the name under mypy --strict
# (no_implicit_reexport) — a plain `import duckdb` makes the attribute invisible
# to importers even though it works fine at runtime.

import httpx as httpx  # explicit re-export: tests patch check_and_retrain.httpx.get,
# same mypy --strict re-export requirement as the duckdb import above.

from scripts._common import gold_db_exists
from src.data.cards.storage.gold.storage import (
    get_latest_gold_snapshot_date,
    get_latest_trainable_snapshot_date,
)
from src.data.repository import GOLD_DB_PATH, open_repository
from src.logger import get_logger, setup_logging
from src.monitoring.alerts import send_alert
from src.monitoring.drift import DriftResult, return_drift
from src.monitoring.prediction_tracker import (
    MONITORING_DB_PATH,
    SKILL_THRESHOLD,
    SkillStatus,
    daily_skill,
    record_daily_predictions,
    skill_status,
)
from src.monitoring.retraining import MODEL_REGISTRY_NAME, retrain, should_retrain
from src.monitoring.serving_check import log_serving_alias_check

STATUS_PATH = Path("logs/last_check_status.json")

logger = get_logger(__name__)


def _write_status(status: dict[str, object]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))


def _ping_heartbeat(ok: bool) -> None:
    """Best-effort GET to HEARTBEAT_URL (success) or HEARTBEAT_URL/fail
    (failure) — a dead-man's-switch: if this stops arriving at all
    (machine off, task deleted), the external monitoring service pages
    someone instead of the failure going unnoticed. Skipped entirely if
    HEARTBEAT_URL is unset. Never raises.
    """
    heartbeat_url = os.getenv("HEARTBEAT_URL", "")
    if not heartbeat_url:
        return
    url = heartbeat_url if ok else f"{heartbeat_url}/fail"
    try:
        httpx.get(url, timeout=5.0)
    except httpx.HTTPError as exc:
        logger.warning("Heartbeat ping failed (non-fatal): %s", exc)


@dataclass
class MonitoringResult:
    """What steps 1–3 found; ``summary()`` goes into the status file."""

    served_run_id: str | None = None
    predictions_recorded: int = 0
    skill_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    skill: SkillStatus = field(
        default_factory=lambda: SkillStatus(0, None, None, False)
    )
    drift: DriftResult | None = None
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "served_run_id": self.served_run_id,
            "predictions_recorded": self.predictions_recorded,
            "skill_evaluated_dates": self.skill.evaluated_dates,
            "skill_latest_date": self.skill.latest_date,
            "skill_latest": self.skill.latest_skill,
            "model_worse_than_naive": self.skill.degraded,
            "drift_score": self.drift.score if self.drift else None,
            "drift_detected": self.drift.drifted if self.drift else None,
            "errors": self.errors,
        }


def _served_run_id() -> str | None:
    """The run the API serves: ``MODEL_RUN_ID`` if set, else the ``production`` alias.

    This job runs on the host, where ``MODEL_RUN_ID`` usually lives only in
    ``docker/.env``; the alias is the fallback, and serving_check alerts when
    the two disagree.
    """
    run_id = os.getenv("MODEL_RUN_ID", "")
    if run_id:
        return run_id
    import mlflow  # deferred: only needed for the fallback

    try:
        version = mlflow.tracking.MlflowClient().get_model_version_by_alias(
            MODEL_REGISTRY_NAME, "production"
        )
    except mlflow.exceptions.MlflowException as exc:
        logger.warning("No served model found (no MODEL_RUN_ID, no alias): %s", exc)
        return None
    return str(version.run_id) if version.run_id else None


@contextmanager
def _monitoring_step(result: MonitoringResult, name: str) -> Iterator[None]:
    """Log, alert and record a failed monitoring step, then carry on."""
    try:
        yield
    except Exception as exc:
        logger.error("Monitoring step %r failed: %s", name, exc, exc_info=True)
        send_alert(f"Monitoring step failed: {name}", str(exc), severity="warning")
        result.errors.append(f"{name}: {exc}")


def run_monitoring(conn: duckdb.DuckDBPyConnection) -> MonitoringResult:
    """Steps 1–3 of the module docstring. Never raises."""
    result = MonitoringResult()
    latest = get_latest_gold_snapshot_date(conn)
    if latest is None:
        result.errors.append("gold_price_features is empty")
        return result

    with _monitoring_step(result, "record predictions"):
        from src.ml.training.tracking import setup_experiment

        setup_experiment()
        result.served_run_id = _served_run_id()
        Path(MONITORING_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        mon = duckdb.connect(MONITORING_DB_PATH)
        try:
            if result.served_run_id:
                result.predictions_recorded = record_daily_predictions(
                    conn, mon, result.served_run_id, latest
                )
            result.skill_df = daily_skill(mon, conn)
        finally:
            mon.close()
        result.skill = skill_status(result.skill_df)
        if result.skill.degraded:
            send_alert(
                "Model worse than naive",
                f"The served model scored below {SKILL_THRESHOLD:+.0%} vs the naive "
                f"forecast on the last evaluated dates (latest "
                f"{result.skill.latest_date}: {result.skill.latest_skill:+.1%}). "
                "A retrain will be attempted.",
                severity="warning",
            )

    with _monitoring_step(result, "return drift"):
        result.drift = return_drift(conn, date.fromisoformat(str(latest)))
        if result.drift.drifted:
            send_alert(
                "Price-return drift",
                f"7-day returns starting {result.drift.current_start}.."
                f"{result.drift.current_end} differ from the 4 weeks before "
                f"(score {result.drift.score:.3f}). Check the price feed before "
                "trusting new predictions.",
                severity="warning",
            )
    return result


def _check_preconditions(
    conn: duckdb.DuckDBPyConnection,
    skill_df: pd.DataFrame | None = None,
) -> tuple[bool, str, str | None]:
    """Check whether a retrain should run.

    Returns (triggered, reason, snapshot_date). ``snapshot_date`` is None
    when not applicable (no trigger fired, or no gold snapshot exists).
    """
    triggered, reason = should_retrain(conn, skill_df)
    if not triggered:
        return False, reason, None
    snapshot_date = get_latest_trainable_snapshot_date(conn)
    return True, reason, snapshot_date


def _do_retrain(
    conn: duckdb.DuckDBPyConnection, snapshot_date: str, reason: str
) -> tuple[bool, dict[str, object]]:
    logger.warning(
        "Retrain triggered (reason=%s) — retraining on snapshot %s.",
        reason,
        snapshot_date,
    )
    try:
        run_id = retrain(conn, snapshot_date)
    except Exception as exc:
        logger.error("Retrain failed: %s", exc)
        send_alert("Retrain failed", str(exc))
        return False, {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "result": "error",
            "reason": "retrain_failed",
            "error": str(exc),
        }

    logger.info("Retrain complete. New run_id: %s", run_id)
    return True, {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "result": "retrained",
        "reason": reason,
        "run_id": run_id,
    }


def _warn_on_serving_alias_mismatch() -> None:
    """Alert if the served model and the registry's production alias disagree.

    Checked here because this is the job that acts on the alias: should_retrain
    → retrain → _compare_and_promote all reason about "production" via the
    alias, so a mismatch silently makes those comparisons about a model that is
    not being served. Never fatal — the divergence needs an operator decision,
    not an aborted monitoring run.
    """
    ok, detail = log_serving_alias_check()
    if not ok:
        send_alert("Serving/registry mismatch", detail, severity="warning")


def main() -> int:
    setup_logging(log_dir=Path("logs"))
    _warn_on_serving_alias_mismatch()

    if not gold_db_exists(GOLD_DB_PATH):
        send_alert(
            "Monitor: Gold DB missing",
            f"{GOLD_DB_PATH} not found — run the ETL pipeline first.",
        )
        _write_status(
            {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "result": "error",
                "reason": "gold_db_missing",
            }
        )
        _ping_heartbeat(ok=False)
        return 1

    repo = open_repository(GOLD_DB_PATH, read_only=True)
    conn = repo.connection
    try:
        monitoring = run_monitoring(conn)
        triggered, reason, snapshot_date = _check_preconditions(
            conn, monitoring.skill_df
        )

        if not triggered:
            logger.info("No retrain trigger fired (reason=%s).", reason)
            _write_status(
                {
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "result": "no_retrain",
                    "reason": reason,
                    "monitoring": monitoring.summary(),
                }
            )
            _ping_heartbeat(ok=True)
            return 0

        if snapshot_date is None:
            logger.error(
                "No trainable snapshot available — gold_price_features is "
                "either empty or has no snapshot with a t+7 counterpart yet."
            )
            send_alert(
                "Monitor: no trainable snapshot",
                "No trainable snapshot available — gold_price_features is "
                "either empty or has no snapshot with a t+7 counterpart yet.",
            )
            _write_status(
                {
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "result": "error",
                    "reason": "no_snapshot",
                    "monitoring": monitoring.summary(),
                }
            )
            _ping_heartbeat(ok=False)
            return 1

        ok, status = _do_retrain(conn, snapshot_date, reason)
        status["monitoring"] = monitoring.summary()
        _write_status(status)
        _ping_heartbeat(ok=ok)
        return 0 if ok else 1
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())
