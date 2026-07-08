import json
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.ml.features.pipeline import enrich_card_df, enrich_lag_df
from src.ml.training.trainer import (
    CVFold,
    InsufficientDataError,
    generate_folds,
    get_available_snapshots,
    load_validation_config,
    walk_forward_cv,
)


# ---------------------------------------------------------------------------
# Helpers / minimal model for walk_forward_cv tests
# ---------------------------------------------------------------------------


class _ZeroModel:
    """Always predicts zero — lets walk_forward_cv tests focus on orchestration."""

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


# ---------------------------------------------------------------------------
# load_validation_config()
# ---------------------------------------------------------------------------


def test_load_validation_config_returns_dict(tmp_path):
    cfg = {"min_train_days": 30, "val_days": 7, "step_days": 7}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    result = load_validation_config(tmp_path / "config.json")
    assert isinstance(result, dict)


def test_load_validation_config_values_correct(tmp_path):
    cfg = {"min_train_days": 30, "val_days": 7, "step_days": 7}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    result = load_validation_config(path)
    assert result["min_train_days"] == 30
    assert result["val_days"] == 7
    assert result["step_days"] == 7


def test_load_validation_config_preserves_extra_keys(tmp_path):
    # Notebook may write additional analysis results to the same file.
    cfg = {"min_train_days": 30, "power": 0.75, "n_tier2": 42}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    result = load_validation_config(path)
    assert result["power"] == 0.75


# ---------------------------------------------------------------------------
# get_available_snapshots()
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_conn():
    """Minimal in-memory DuckDB with a few known snapshot dates."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE gold_price_features (
            uuid VARCHAR,
            snapshot_date DATE,
            eur DOUBLE
        )
    """)
    for d in ["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-05"]:
        conn.execute(
            "INSERT INTO gold_price_features VALUES (?, ?, ?)", ["card_a", d, 5.0]
        )
    yield conn
    conn.close()


def test_get_available_snapshots_returns_list(snapshot_conn):
    result = get_available_snapshots(snapshot_conn)
    assert isinstance(result, list)


def test_get_available_snapshots_sorted_ascending(snapshot_conn):
    result = get_available_snapshots(snapshot_conn)
    assert result == sorted(result)


def test_get_available_snapshots_correct_values(snapshot_conn):
    result = get_available_snapshots(snapshot_conn)
    assert "2026-01-01" in result
    assert "2026-01-05" in result


def test_get_available_snapshots_no_duplicates(snapshot_conn):
    # Insert a duplicate row for the same date — must be deduplicated by DISTINCT.
    snapshot_conn.execute(
        "INSERT INTO gold_price_features VALUES (?, ?, ?)",
        ["card_b", "2026-01-01", 3.0],
    )
    result = get_available_snapshots(snapshot_conn)
    assert result.count("2026-01-01") == 1


# ---------------------------------------------------------------------------
# generate_folds()
# ---------------------------------------------------------------------------


def _make_dates(start_iso: str, n_days: int) -> list[str]:
    """Return n_days consecutive ISO date strings starting from start_iso."""
    start = date.fromisoformat(start_iso)
    return [(start + timedelta(days=i)).isoformat() for i in range(n_days)]


def test_generate_folds_returns_list():
    # 55 daily dates → enough for 3 folds with defaults (threshold = 51 days).
    dates = _make_dates("2026-01-01", 55)
    result = generate_folds(dates)
    assert isinstance(result, list)


def test_generate_folds_returns_cvfold_objects():
    dates = _make_dates("2026-01-01", 55)
    folds = generate_folds(dates)
    assert all(isinstance(f, CVFold) for f in folds)


def test_generate_folds_at_least_3_folds():
    dates = _make_dates("2026-01-01", 55)
    folds = generate_folds(dates)
    assert len(folds) >= 3


def test_generate_folds_fold_idx_sequential():
    dates = _make_dates("2026-01-01", 55)
    folds = generate_folds(dates)
    assert [f.fold_idx for f in folds] == list(range(len(folds)))


def test_generate_folds_train_start_constant():
    # All folds must share the same train_start (expanding window).
    dates = _make_dates("2026-01-01", 55)
    folds = generate_folds(dates)
    assert all(f.train_start == folds[0].train_start for f in folds)


def test_generate_folds_train_end_advances_by_step_days():
    dates = _make_dates("2026-01-01", 55)
    folds = generate_folds(dates, step_days=7)
    diffs = [
        (
            date.fromisoformat(folds[i + 1].train_end)
            - date.fromisoformat(folds[i].train_end)
        ).days
        for i in range(len(folds) - 1)
    ]
    assert all(d == 7 for d in diffs)


def test_generate_folds_val_follows_train():
    # val_start must be exactly 1 day after train_end.
    dates = _make_dates("2026-01-01", 55)
    folds = generate_folds(dates)
    for f in folds:
        train_end = date.fromisoformat(f.train_end)
        val_start = date.fromisoformat(f.val_start)
        assert (val_start - train_end).days == 1


def test_generate_folds_val_window_correct_length():
    # val_end - val_start + 1 == val_days (inclusive calendar days).
    dates = _make_dates("2026-01-01", 55)
    folds = generate_folds(dates, val_days=7)
    for f in folds:
        span = (
            date.fromisoformat(f.val_end) - date.fromisoformat(f.val_start)
        ).days + 1
        assert span == 7


def test_generate_folds_first_train_window_at_least_min_train_days():
    dates = _make_dates("2026-01-01", 55)
    folds = generate_folds(dates, min_train_days=30)
    f0 = folds[0]
    span = (
        date.fromisoformat(f0.train_end) - date.fromisoformat(f0.train_start)
    ).days + 1
    assert span >= 30


def test_generate_folds_raises_insufficient_data_error_when_too_few_folds():
    # Only 36 days → first fold needs 30+7=37 → can't form even 1 fold, let alone 3.
    dates = _make_dates("2026-01-01", 36)
    with pytest.raises(InsufficientDataError):
        generate_folds(dates, min_train_days=30, val_days=7, step_days=7)


def test_generate_folds_error_message_contains_unlock_date():
    dates = _make_dates("2026-01-01", 36)
    with pytest.raises(InsufficientDataError, match="2026"):
        generate_folds(dates)


def test_generate_folds_custom_params_create_more_folds():
    # With smaller windows, more folds fit into the same date range.
    dates = _make_dates("2026-01-01", 55)
    folds_default = generate_folds(dates, min_train_days=30, val_days=7, step_days=7)
    folds_small = generate_folds(dates, min_train_days=10, val_days=3, step_days=2)
    assert len(folds_small) > len(folds_default)


# ---------------------------------------------------------------------------
# walk_forward_cv() — uses monkeypatching to isolate orchestration logic
# ---------------------------------------------------------------------------


@pytest.fixture
def wfcv_conn():
    """Minimal DuckDB supporting the snapshot queries in walk_forward_cv."""
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE gold_price_features (
            uuid VARCHAR, snapshot_date DATE, eur DOUBLE
        )
    """)
    for i in range(20):
        d = (date(2026, 1, 1) + timedelta(days=i)).isoformat()
        for uid in ["A", "B", "C", "D", "E"]:
            conn.execute(
                "INSERT INTO gold_price_features VALUES (?, ?, ?)", [uid, d, 5.0]
            )
    conn.execute("CREATE TABLE gold_card_features (uuid VARCHAR)")
    for uid in ["A", "B", "C", "D", "E"]:
        conn.execute("INSERT INTO gold_card_features VALUES (?)", [uid])
    yield conn
    conn.close()


@pytest.fixture
def simple_folds():
    return [
        CVFold(0, "2026-01-01", "2026-01-05", "2026-01-06", "2026-01-07"),
        CVFold(1, "2026-01-01", "2026-01-07", "2026-01-08", "2026-01-09"),
        CVFold(2, "2026-01-01", "2026-01-09", "2026-01-10", "2026-01-11"),
    ]


def _mock_prepare(lag_df, card_df, target_df):
    """Returns a 5-row feature matrix with 'f1' and 'eur' columns."""
    n = 5
    X = pd.DataFrame({"f1": np.zeros(n), "eur": np.full(n, 5.0)})
    y = pd.Series(np.full(n, 0.05))
    return X, y


def _mock_pipeline_factory():
    """Returns a pipeline that passes through f1 and drops everything else."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline([("features", FunctionTransformer(lambda X: X[["f1"]].values))])


def _mock_feature_names(pipeline):
    return ["f1"]


def test_walk_forward_cv_returns_dataframe(wfcv_conn, simple_folds, monkeypatch):
    import src.ml.training.trainer as t

    monkeypatch.setattr(t, "build_lag_features", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "build_target", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "prepare_training_data", _mock_prepare)
    monkeypatch.setattr(t, "build_feature_pipeline", _mock_pipeline_factory)
    monkeypatch.setattr(t, "get_feature_names", _mock_feature_names)

    result = walk_forward_cv(wfcv_conn, _ZeroModel(), simple_folds)
    assert isinstance(result, pd.DataFrame)


def test_walk_forward_cv_columns(wfcv_conn, simple_folds, monkeypatch):
    import src.ml.training.trainer as t

    monkeypatch.setattr(t, "build_lag_features", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "build_target", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "prepare_training_data", _mock_prepare)
    monkeypatch.setattr(t, "build_feature_pipeline", _mock_pipeline_factory)
    monkeypatch.setattr(t, "get_feature_names", _mock_feature_names)

    result = walk_forward_cv(wfcv_conn, _ZeroModel(), simple_folds)
    assert "fold_idx" in result.columns
    assert "tier" in result.columns
    assert "mae" in result.columns
    assert "mape" in result.columns


def test_walk_forward_cv_all_folds_present(wfcv_conn, simple_folds, monkeypatch):
    import src.ml.training.trainer as t

    monkeypatch.setattr(t, "build_lag_features", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "build_target", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "prepare_training_data", _mock_prepare)
    monkeypatch.setattr(t, "build_feature_pipeline", _mock_pipeline_factory)
    monkeypatch.setattr(t, "get_feature_names", _mock_feature_names)

    result = walk_forward_cv(wfcv_conn, _ZeroModel(), simple_folds)
    assert set(result["fold_idx"]) == {0, 1, 2}


def test_walk_forward_cv_empty_folds_returns_empty_df(wfcv_conn, monkeypatch):
    import src.ml.training.trainer as t

    monkeypatch.setattr(t, "build_lag_features", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "build_target", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "prepare_training_data", _mock_prepare)
    monkeypatch.setattr(t, "build_feature_pipeline", _mock_pipeline_factory)
    monkeypatch.setattr(t, "get_feature_names", _mock_feature_names)

    result = walk_forward_cv(wfcv_conn, _ZeroModel(), folds=[])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_walk_forward_cv_skips_fold_when_no_val_snapshot(monkeypatch):
    """A fold whose val window falls outside the available dates must be skipped."""
    import src.ml.training.trainer as t

    monkeypatch.setattr(t, "build_lag_features", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "build_target", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "prepare_training_data", _mock_prepare)
    monkeypatch.setattr(t, "build_feature_pipeline", _mock_pipeline_factory)
    monkeypatch.setattr(t, "get_feature_names", _mock_feature_names)

    # DuckDB has no rows → MAX returns NULL → fold is skipped
    empty_conn = duckdb.connect()
    empty_conn.execute("""
        CREATE TABLE gold_price_features (uuid VARCHAR, snapshot_date DATE, eur DOUBLE)
    """)
    empty_conn.execute("CREATE TABLE gold_card_features (uuid VARCHAR)")

    folds = [CVFold(0, "2026-01-01", "2026-01-05", "2026-01-06", "2026-01-07")]
    result = walk_forward_cv(empty_conn, _ZeroModel(), folds)
    assert len(result) == 0
    empty_conn.close()


# ---------------------------------------------------------------------------
# Alignment: walk_forward_cv uses enrich_card_df / enrich_lag_df helpers
# ---------------------------------------------------------------------------


def test_walk_forward_cv_calls_enrich_card_df(wfcv_conn, simple_folds, monkeypatch):
    """walk_forward_cv must pass card_df through enrich_card_df before fold loop."""
    import src.ml.training.trainer as t

    enriched_calls: list[str] = []

    def _spy_enrich_card(df):
        enriched_calls.append("card")
        return enrich_card_df(df)

    monkeypatch.setattr(t, "enrich_card_df", _spy_enrich_card)
    monkeypatch.setattr(t, "build_lag_features", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "build_target", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "prepare_training_data", _mock_prepare)
    monkeypatch.setattr(t, "build_feature_pipeline", _mock_pipeline_factory)
    monkeypatch.setattr(t, "get_feature_names", _mock_feature_names)

    walk_forward_cv(wfcv_conn, _ZeroModel(), simple_folds)
    # Called exactly once (before the fold loop, not once per fold).
    assert enriched_calls.count("card") == 1


def test_walk_forward_cv_calls_enrich_lag_df_per_fold(
    wfcv_conn, simple_folds, monkeypatch
):
    """walk_forward_cv must call enrich_lag_df for each lag_train and lag_val."""
    import src.ml.training.trainer as t

    enriched_calls: list[str] = []

    def _spy_enrich_lag(df):
        enriched_calls.append("lag")
        return enrich_lag_df(df)

    monkeypatch.setattr(t, "enrich_lag_df", _spy_enrich_lag)
    monkeypatch.setattr(t, "build_lag_features", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "build_target", lambda conn, snap: pd.DataFrame())
    monkeypatch.setattr(t, "prepare_training_data", _mock_prepare)
    monkeypatch.setattr(t, "build_feature_pipeline", _mock_pipeline_factory)
    monkeypatch.setattr(t, "get_feature_names", _mock_feature_names)

    walk_forward_cv(wfcv_conn, _ZeroModel(), simple_folds)
    # 3 folds × 2 calls each (lag_train + lag_val) = 6 total.
    assert enriched_calls.count("lag") == len(simple_folds) * 2
