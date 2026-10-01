"""Tests for src/monitoring/drift.py — 7-day return drift (ADR-034)."""

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy.stats import wasserstein_distance  # type: ignore[import-untyped]

from src.monitoring.drift import (
    DRIFT_THRESHOLD,
    drift_score,
    fetch_returns,
    return_drift,
    wasserstein_1d,
)

LATEST = date(2026, 10, 30)


def _gold(weekly_return_for_start):
    """Gold with 60 cards priced daily for 50 days.

    ``weekly_return_for_start(start_date, card)`` sets each card's 7-day log
    return from that start date, so tests control the return distribution.
    """
    rng = np.random.default_rng(0)
    first = LATEST - timedelta(days=49)
    rows = []
    for card in range(60):
        log_price = np.log1p(5.0 + card)
        prices: dict[date, float] = {}
        for day in range(50):
            d = first + timedelta(days=day)
            if day >= 7:
                start = d - timedelta(days=7)
                log_price = np.log1p(prices[start]) + weekly_return_for_start(
                    start, card, rng
                )
            prices[d] = float(np.expm1(log_price))
            rows.append((f"c{card}", d, prices[d]))
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE gold_price_features (uuid VARCHAR, snapshot_date DATE, eur DOUBLE)"
    )
    con.executemany("INSERT INTO gold_price_features VALUES (?, ?, ?)", rows)
    return con


def test_wasserstein_1d_matches_scipy():
    rng = np.random.default_rng(1)
    u, v = rng.normal(0, 1, 500), rng.normal(0.3, 2, 300)

    assert wasserstein_1d(u, v) == pytest.approx(wasserstein_distance(u, v))


def test_drift_score_is_nan_when_undefined():
    assert np.isnan(drift_score(pd.Series([], dtype=float), pd.Series([0.1])))
    # Frozen feed: every reference return is zero, so there is no spread.
    assert np.isnan(drift_score(pd.Series([0.0, 0.0]), pd.Series([0.1])))


def test_stable_returns_do_not_drift():
    con = _gold(lambda start, card, rng: rng.normal(0, 0.05))

    result = return_drift(con, LATEST)

    assert result.drifted is False
    assert result.score < DRIFT_THRESHOLD
    con.close()


def test_a_market_wide_jump_in_the_current_week_drifts():
    current_start = LATEST - timedelta(days=13)

    def returns(start, card, rng):
        jump = 0.4 if start >= current_start else 0.0
        return jump + rng.normal(0, 0.05)

    con = _gold(returns)

    result = return_drift(con, LATEST)

    assert result.drifted is True
    assert result.score > DRIFT_THRESHOLD
    con.close()


def test_windows_end_where_returns_can_exist():
    con = _gold(lambda start, card, rng: 0.0)

    result = return_drift(con, LATEST)

    # A return starting after LATEST-7 has no end price yet.
    assert result.current_end == str(LATEST - timedelta(days=7))
    assert result.current_start == str(LATEST - timedelta(days=13))
    assert result.current_n == 7 * 60
    assert result.reference_n == 28 * 60
    con.close()


def test_fetch_returns_skips_cards_without_both_prices():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE gold_price_features (uuid VARCHAR, snapshot_date DATE, eur DOUBLE)"
    )
    d = date(2026, 10, 1)
    con.executemany(
        "INSERT INTO gold_price_features VALUES (?, ?, ?)",
        [("a", d, 1.0), ("a", d + timedelta(days=7), 3.0), ("b", d, 1.0)],
    )

    r = fetch_returns(con, d, d)

    assert r.tolist() == pytest.approx([np.log(4.0) - np.log(2.0)])
    con.close()
