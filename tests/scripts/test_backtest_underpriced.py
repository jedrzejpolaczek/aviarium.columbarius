"""Tests for scripts/backtest_underpriced.py.

The model-training path (flag_on) is exercised end to end against real data by
running the script; these tests pin the parts that decide *which* data it may
look at and how results are pooled — the places a lookahead or a mis-weighted
average would hide.
"""

from datetime import date

import duckdb
import pytest

from scripts.backtest_underpriced import (
    DayResult,
    _price_changes,
    evaluation_dates,
    summarise,
)


def test_evaluation_dates_require_snapshots_on_both_sides():
    snaps = [date(2026, 7, d) for d in (1, 8, 9, 15, 20)]

    triples = evaluation_dates(snaps)

    # 07-08: 07-01 and 07-15 exist. 07-09: no 07-16. 07-15: no 07-22.
    assert triples == [(date(2026, 7, 1), date(2026, 7, 8), date(2026, 7, 15))]


def test_training_snapshot_target_ends_on_the_evaluation_date():
    """No lookahead: the train target (train → train+7) is known on eval day."""
    snaps = [date(2026, 9, d) for d in range(1, 30)]

    for train, eval_date, check in evaluation_dates(snaps):
        assert (eval_date - train).days == 7
        assert (check - eval_date).days == 7


def _day(flagged, flagged_hits, eligible, base_hits, cheap=(0, 0), flagged_cheap=0):
    return DayResult(
        eval_date="d",
        train_snapshot="t",
        check_date="c",
        eligible=eligible,
        base_hits=base_hits,
        flagged=flagged,
        flagged_hits=flagged_hits,
        flagged_mean_change=0.0,
        eligible_mean_change=0.0,
        eligible_below_1eur=cheap[0],
        base_hits_below_1eur=cheap[1],
        flagged_below_1eur=flagged_cheap,
        flagged_eur_gain=0.0,
    )


def test_summarise_pools_card_days_rather_than_averaging_daily_rates():
    results = [_day(1, 1, 100, 10), _day(9, 0, 100, 10)]

    s = summarise(results)

    # Pooled 1/10, not the mean of daily rates (1.0 + 0.0) / 2.
    assert s["flag_hit_rate"] == pytest.approx(0.1)
    assert s["base_hit_rate"] == pytest.approx(0.1)
    assert s["dates_with_flags"] == 2


def test_summarise_reports_nan_when_nothing_was_flagged():
    s = summarise([_day(0, 0, 50, 5)])

    assert s["flag_hit_rate"] != s["flag_hit_rate"]  # NaN
    assert s["base_hit_rate"] == pytest.approx(0.1)


def test_price_changes_uses_both_ends_and_skips_missing_prices():
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE gold_price_features AS SELECT * FROM (VALUES
            ('a', DATE '2026-07-01', 10.0), ('a', DATE '2026-07-08', 12.0),
            ('b', DATE '2026-07-01', 5.0),  ('b', DATE '2026-07-08', NULL),
            ('c', DATE '2026-07-08', 3.0)
        ) AS t(uuid, snapshot_date, eur)
        """
    )

    changes = _price_changes(con, date(2026, 7, 1), date(2026, 7, 8))

    assert changes["uuid"].tolist() == ["a"]
    assert changes["change"].iloc[0] == pytest.approx(0.2)
    con.close()


def test_summarise_skips_frozen_feed_dates_that_would_dilute_the_base_rate():
    frozen = _day(0, 0, 1000, 0)
    live = _day(4, 2, 100, 10, cheap=(50, 10), flagged_cheap=4)

    s = summarise([frozen, live])

    assert s["frozen_dates_skipped"] == 1
    assert s["base_hit_rate"] == pytest.approx(0.1)
    assert s["base_hit_rate_below_1eur"] == pytest.approx(0.2)
    assert s["flagged_share_below_1eur"] == pytest.approx(1.0)
