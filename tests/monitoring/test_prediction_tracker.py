"""Tests for src/monitoring/prediction_tracker.py (ADR-034)."""

import math
from datetime import date, timedelta
from unittest.mock import MagicMock

import duckdb
import numpy as np
import pandas as pd
import pytest

import src.ml.features.pipeline as pipeline_module
import src.ml.training.tracking as tracking_module
from src.monitoring.prediction_tracker import (
    daily_skill,
    is_degraded,
    record_daily_predictions,
    save_predictions,
    skill_status,
)

D0 = date(2026, 10, 1)


@pytest.fixture
def mon():
    con = duckdb.connect()
    yield con
    con.close()


@pytest.fixture
def gold():
    """Two cheap cards and one Tier 3 card, priced on D0 and D0+7 only."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE gold_price_features (uuid VARCHAR, snapshot_date DATE, eur DOUBLE)"
    )
    later = D0 + timedelta(days=7)
    rows = [
        ("a", D0, 1.0), ("a", later, 1.0),      # did not move
        ("b", D0, 2.0), ("b", later, 3.0),      # rose
        ("x", D0, 5000.0), ("x", later, 1.0),   # Tier 3 crash — must be ignored
    ]  # fmt: skip
    con.executemany("INSERT INTO gold_price_features VALUES (?, ?, ?)", rows)
    yield con
    con.close()


PRICE_ON_D0 = {"a": 1.0, "b": 2.0, "x": 5000.0}


def _preds(**by_uuid):
    """Predictions carry the card's price on the prediction date, as recorded."""
    return pd.DataFrame(
        {
            "uuid": list(by_uuid),
            "eur": [PRICE_ON_D0[u] for u in by_uuid],
            "predicted_log_return": list(by_uuid.values()),
        }
    )


def test_save_predictions_replaces_the_same_day_and_model(mon):
    save_predictions(mon, _preds(a=0.1, b=0.2), "2026-10-01", "run1")
    save_predictions(mon, _preds(a=0.5), "2026-10-01", "run1")
    save_predictions(mon, _preds(a=0.9), "2026-10-01", "run2")

    rows = mon.execute(
        "SELECT model_run_id, uuid, predicted_log_return FROM predictions ORDER BY 1, 2"
    ).fetchall()
    assert rows == [("run1", "a", 0.5), ("run2", "a", 0.9)]


def test_save_predictions_keeps_old_rows_when_the_insert_fails(mon):
    save_predictions(mon, _preds(a=0.1), "2026-10-01", "run1")
    bad = pd.DataFrame({"uuid": ["a"], "eur": [1.0], "predicted_log_return": [None]})

    with pytest.raises(duckdb.Error):
        save_predictions(mon, bad, "2026-10-01", "run1")

    assert mon.execute("SELECT predicted_log_return FROM predictions").fetchall() == [
        (0.1,)
    ]


def test_daily_skill_compares_model_with_naive_on_tier1_cards(mon, gold):
    actual_b = math.log1p(3.0) - math.log1p(2.0)
    # Model predicts b's rise exactly and says "flat" for a: perfect.
    save_predictions(mon, _preds(a=0.0, b=actual_b, x=0.0), str(D0), "run1")

    skill = daily_skill(mon, gold)

    row = skill.iloc[0]
    assert row["n"] == 2  # Tier 3 card excluded
    assert row["model_mae"] == pytest.approx(0.0)
    assert row["naive_mae"] == pytest.approx(actual_b / 2)
    assert row["skill"] == pytest.approx(1.0)


def test_daily_skill_is_negative_when_model_is_worse_than_no_change(mon, gold):
    save_predictions(mon, _preds(a=0.5, b=-0.5), str(D0), "run1")

    assert daily_skill(mon, gold).iloc[0]["skill"] < 0


def test_daily_skill_skips_dates_whose_t_plus_7_has_not_arrived(mon, gold):
    save_predictions(mon, _preds(a=0.0), str(D0 + timedelta(days=7)), "run1")

    assert daily_skill(mon, gold).empty


def test_daily_skill_on_an_empty_database_is_empty(mon, gold):
    assert daily_skill(mon, gold).empty


def _skill_frame(*values):
    return pd.DataFrame(
        {
            "snapshot_date": pd.date_range("2026-10-01", periods=len(values)),
            "skill": list(values),
        }
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((-0.2, -0.1, -0.06), True),
        ((-0.2, -0.1, -0.01), False),  # latest date recovered
        ((-0.01, -0.2, -0.2, -0.2), True),  # only the last three count
        ((-0.2, -0.2), False),  # not enough evaluated dates yet
        ((-0.2, float("nan"), -0.2), False),  # no-movement week never breaches
    ],
)
def test_is_degraded(values, expected):
    assert is_degraded(_skill_frame(*values)) is expected


def test_skill_status_reports_the_latest_date():
    status = skill_status(_skill_frame(0.01, -0.02))

    assert status.evaluated_dates == 2
    assert status.latest_date is not None
    assert status.latest_date.startswith("2026-10-02")
    assert status.latest_skill == pytest.approx(-0.02)
    assert status.degraded is False


def test_skill_status_of_nothing():
    assert skill_status(pd.DataFrame()).evaluated_dates == 0


def test_record_daily_predictions_uses_the_served_model_and_stores_every_card(
    mon, monkeypatch
):
    X_raw = pd.DataFrame({"uuid": ["a", "b"], "eur": [1.0, 2.0], "f": [0.0, 1.0]})
    model = MagicMock()
    model.predict.return_value = np.array([0.1, -0.2])
    monkeypatch.setattr(tracking_module, "load_model_from_mlflow", lambda run_id: model)
    monkeypatch.setattr(pipeline_module, "build_inference_features", lambda c, d: X_raw)
    monkeypatch.setattr(
        pipeline_module, "fit_transform_features", lambda X: (X[["f"]], None, ["f"])
    )

    n = record_daily_predictions(MagicMock(), mon, "run1", "2026-10-01")

    assert n == 2
    rows = mon.execute(
        "SELECT uuid, snapshot_date, predicted_log_return, model_run_id "
        "FROM predictions ORDER BY uuid"
    ).fetchall()
    assert rows == [
        ("a", date(2026, 10, 1), 0.1, "run1"),
        ("b", date(2026, 10, 1), -0.2, "run1"),
    ]
