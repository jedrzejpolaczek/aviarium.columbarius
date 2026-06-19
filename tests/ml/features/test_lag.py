import math

import duckdb
import pandas as pd
import pytest

from src.ml.features.lag import build_lag_features, build_target


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    """In-memory DuckDB with gold_price_features populated for two cards.

    card_a: 10 consecutive daily snapshots starting 2026-01-01.
            Prices increase linearly: 1.0, 1.1, ..., 1.9.
            On day 8 (2026-01-08) lag_1d and lag_7d are both available.

    card_b: only 2 snapshots (2026-01-01 and 2026-01-08).
            Used to verify that early-history NaNs are handled correctly
            and that build_target works for a card missing intermediate days.
    """
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE gold_price_features (
            uuid          VARCHAR,
            snapshot_date DATE,
            eur           DOUBLE,
            edhrec_rank   DOUBLE,
            foil_premium  DOUBLE
        )
    """)

    rows: list[tuple[str, str, float, float | None, float | None]] = []
    for i in range(10):
        date = f"2026-01-{i + 1:02d}"
        rows.append(("card_a", date, round(1.0 + i * 0.1, 2), None, 1.5))

    # card_b has only the first and eighth snapshot
    rows.append(("card_b", "2026-01-01", 5.0, 100.0, None))
    rows.append(("card_b", "2026-01-08", 6.0, 95.0, None))

    con.executemany("INSERT INTO gold_price_features VALUES (?, ?, ?, ?, ?)", rows)
    yield con
    con.close()


# ---------------------------------------------------------------------------
# build_lag_features()
# ---------------------------------------------------------------------------


def test_build_lag_features_returns_dataframe(conn):
    result = build_lag_features(conn, "2026-01-08")
    assert isinstance(result, pd.DataFrame)


def test_build_lag_features_expected_columns(conn):
    result = build_lag_features(conn, "2026-01-08")
    expected = {
        "uuid",
        "snapshot_date",
        "eur",
        "edhrec_rank",
        "foil_premium",
        "lag_1d",
        "lag_7d",
        "lag_14d",
        "lag_30d",
        "rolling_mean_7d",
        "rolling_std_14d",
        "rolling_min_30d",
        "rolling_max_30d",
        "momentum_7d",
    }
    assert expected.issubset(set(result.columns))


def test_build_lag_features_filters_to_snapshot_date(conn):
    # Only rows for the requested date should be returned.
    result = build_lag_features(conn, "2026-01-08")
    assert (result["snapshot_date"] == pd.Timestamp("2026-01-08")).all()


def test_build_lag_features_lag_1d_correct(conn):
    # card_a on 2026-01-08 (day index 7): eur=1.7, lag_1d should be 1.6 (day 6)
    result = build_lag_features(conn, "2026-01-08")
    row = result[result["uuid"] == "card_a"].iloc[0]
    assert abs(row["lag_1d"] - 1.6) < 1e-9


def test_build_lag_features_lag_7d_correct(conn):
    # card_a on 2026-01-08 (day index 7): lag_7d should be 1.0 (day 0)
    result = build_lag_features(conn, "2026-01-08")
    row = result[result["uuid"] == "card_a"].iloc[0]
    assert abs(row["lag_7d"] - 1.0) < 1e-9


def test_build_lag_features_lag_14d_is_nan_when_history_too_short(conn):
    # card_a only has 8 rows on 2026-01-08, so lag_14d cannot be filled.
    result = build_lag_features(conn, "2026-01-08")
    row = result[result["uuid"] == "card_a"].iloc[0]
    assert pd.isna(row["lag_14d"])


def test_build_lag_features_momentum_7d_correct(conn):
    # card_a on 2026-01-08: eur=1.7, lag_7d=1.0
    # momentum_7d = (1.7 - 1.0) / 1.0 = 0.7
    result = build_lag_features(conn, "2026-01-08")
    row = result[result["uuid"] == "card_a"].iloc[0]
    assert abs(row["momentum_7d"] - 0.7) < 1e-6


def test_build_lag_features_rolling_mean_7d_correct(conn):
    # card_a on 2026-01-08 (day index 7): 7-day window covers days 1–7
    # prices: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 → mean = 1.4
    result = build_lag_features(conn, "2026-01-08")
    row = result[result["uuid"] == "card_a"].iloc[0]
    assert abs(row["rolling_mean_7d"] - 1.4) < 1e-6


def test_build_lag_features_card_with_no_lag_7d_returns_nan_momentum(conn):
    # card_b only has 2 snapshots — on 2026-01-01 lag_7d is NULL.
    # momentum_7d = (eur - NULL) / NULL → should be NaN, not an error.
    result = build_lag_features(conn, "2026-01-01")
    row = result[result["uuid"] == "card_b"].iloc[0]
    assert pd.isna(row["momentum_7d"])


def test_build_lag_features_empty_when_no_data_on_date(conn):
    result = build_lag_features(conn, "2099-01-01")
    assert len(result) == 0


# ---------------------------------------------------------------------------
# build_target()
# ---------------------------------------------------------------------------


def test_build_target_returns_dataframe(conn):
    result = build_target(conn, "2026-01-01")
    assert isinstance(result, pd.DataFrame)


def test_build_target_columns(conn):
    result = build_target(conn, "2026-01-01")
    assert list(result.columns) == ["uuid", "log_return_7d"]


def test_build_target_log_return_correct(conn):
    # card_b: eur_t0=5.0 on 2026-01-01, eur_t7=6.0 on 2026-01-08
    # log_return_7d = ln(1+6) - ln(1+5) = ln(7) - ln(6)
    result = build_target(conn, "2026-01-01")
    row = result[result["uuid"] == "card_b"].iloc[0]
    expected = math.log(7) - math.log(6)
    assert abs(row["log_return_7d"] - expected) < 1e-9


def test_build_target_excludes_cards_missing_future_snapshot(conn):
    # Only card_b has a snapshot on both 2026-01-01 and 2026-01-08.
    # card_a has data on 2026-01-01 but no snapshot on 2026-01-08 from card_b's angle —
    # actually card_a DOES have both, so we check on a date where card_b has no t+7.
    # On 2026-01-08, card_b has no snapshot on 2026-01-15 → should be excluded.
    result = build_target(conn, "2026-01-08")
    assert "card_b" not in result["uuid"].values


def test_build_target_positive_return_when_price_rises(conn):
    # card_b: 5.0 → 6.0, so log_return_7d should be positive
    result = build_target(conn, "2026-01-01")
    row = result[result["uuid"] == "card_b"].iloc[0]
    assert row["log_return_7d"] > 0


def test_build_target_empty_when_no_future_snapshot(conn):
    # No card has data on 2099-01-08 (t+7 of 2099-01-01).
    result = build_target(conn, "2099-01-01")
    assert len(result) == 0
