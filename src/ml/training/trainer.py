"""
Trains models using walk-forward cross-validation.

WHY WALK-FORWARD (important!):
Price data is a time series. A random train/test split would "jump in time" —
the model would learn from future data to predict the past. Results would look
great in evaluation and be useless in production.

Walk-forward guarantees the validation set is ALWAYS later than training:
  Fold 0: train 2026-05-26..2026-06-24, val 2026-06-25..2026-07-01
  Fold 1: train 2026-05-26..2026-07-01, val 2026-07-02..2026-07-08
  ...

The training window grows (more historical data each fold); the validation
window advances at the same rate (step_days = 7 by default).

PARAMETERS (from model_preparation/validation_config.json):
  min_train_days = 30  (minimum calendar days in the training window)
  val_days       = 7   (calendar days covered by each validation window)
  step_days      = 7   (days the split point advances between folds)

DATA GATE:
Walk-forward CV needs >= 3 *usable* folds. A fold spanning the right calendar
range is not necessarily usable: walk_forward_cv skips one whose validation
window contains no snapshot, and one whose chosen snapshot has no snapshot
exactly TARGET_HORIZON_DAYS later, because build_target joins t to t+7 and
returns nothing without both. generate_folds applies the same test and returns
only the folds that will actually run.

With the default parameters the first three usable folds appear after roughly
57 days of daily snapshots (start + 29 train days + 2×7 step days + 7 val days
+ 7 days for the last fold's target horizon).

This used to be checked on calendar span alone, which let generate_folds
report 13 folds on a Gold layer where walk_forward_cv silently executed 2 —
and nothing in the output distinguished the two numbers.
"""

import json
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.ml.evaluation.metrics import evaluate_per_tier
from src.ml.features.lag import build_lag_features, build_target
from src.ml.features.pipeline import (
    build_feature_pipeline,
    enrich_card_df,
    enrich_lag_df,
    get_feature_names,
    prepare_training_data,
)
from src.ml.models.tiered import assign_tier


class InsufficientDataError(Exception):
    """Raised when too few snapshots yield usable walk-forward folds.

    "Usable" is the operative word: see :func:`fold_is_usable`. A fold that spans
    the right calendar range but would be skipped at run time does not count.
    """


TARGET_HORIZON_DAYS = 7
"""Days ahead the target looks. Must match the INTERVAL in features/sql/target.sql:
build_target inner-joins a snapshot to the one exactly this many days later, so a
fold whose snapshot has no such partner yields an empty training or validation
set and is skipped by walk_forward_cv.
"""


MIN_USABLE_FOLDS = 3
"""Credibility threshold from the MP-03 power analysis. Counted over usable folds,
not folds that merely span the right calendar range.
"""


@dataclass
class CVFold:
    """Date ranges for one walk-forward fold."""

    fold_idx: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str


def load_validation_config(
    path: Path = Path("notebooks/model_preparation/validation_config.json"),
) -> dict[str, Any]:
    """Read walk-forward CV configuration written by notebook MP-03.

    The config file captures the statistical power analysis results and the
    chosen train/val window sizes, so the same settings are used in both the
    notebook and the production training script.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        Dict with keys: min_train_days, val_days, step_days, and any other
        values recorded by the notebook.
    """
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def get_available_snapshots(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Return all distinct snapshot dates from gold_price_features, sorted ascending.

    Args:
        conn: Open DuckDB connection with gold_price_features in scope.

    Returns:
        Sorted list of ISO date strings, e.g. ['2026-05-26', '2026-05-27', ...].
    """
    result = conn.execute(
        "SELECT DISTINCT snapshot_date FROM gold_price_features ORDER BY snapshot_date"
    ).df()
    return result["snapshot_date"].astype(str).tolist()


def fold_is_usable(fold: CVFold, available: set[date]) -> bool:
    """Will walk_forward_cv actually run this fold, or skip it?

    Mirrors the loop's two ``continue`` branches, using only the set of
    snapshot dates — no database access needed, because both conditions are
    properties of the calendar:

    1. the validation window must contain at least one snapshot (the loop's
       ``val_snap is None`` check);
    2. the snapshot chosen at each end must have a snapshot exactly
       TARGET_HORIZON_DAYS later, or build_target returns no rows and the
       feature matrix comes back empty (the loop's ``X_train_raw.empty or
       X_val_raw.empty`` check).

    ``train_snap`` is never None in practice — the training window starts at
    the first snapshot — but it is checked here for the same reason the loop
    checks it.

    Args:
        fold:      The candidate fold.
        available: Every snapshot date present in gold_price_features.

    Returns:
        True if the fold contributes metrics, False if it would be skipped.
    """
    horizon = timedelta(days=TARGET_HORIZON_DAYS)

    train_end = date.fromisoformat(fold.train_end)
    train_snap = max((d for d in available if d <= train_end), default=None)
    if train_snap is None or train_snap + horizon not in available:
        return False

    val_start = date.fromisoformat(fold.val_start)
    val_end = date.fromisoformat(fold.val_end)
    val_snaps = [d for d in available if val_start <= d <= val_end]
    if not val_snaps:
        return False

    return max(val_snaps) + horizon in available


def generate_folds(
    snapshot_dates: list[str],
    min_train_days: int = 30,
    val_days: int = 7,
    step_days: int = 7,
) -> list[CVFold]:
    """Generate walk-forward CV folds from a sorted list of snapshot dates.

    The training window starts at snapshot_dates[0] and ends at a split point
    that advances by step_days per fold. The validation window immediately
    follows the split point.

    Example with min_train_days=30, val_days=7, step_days=7:
      Fold 0: train 2026-05-26..2026-06-24, val 2026-06-25..2026-07-01
      Fold 1: train 2026-05-26..2026-07-01, val 2026-07-02..2026-07-08

    Args:
        snapshot_dates: Sorted list of available snapshot dates (ISO strings).
        min_train_days: Minimum number of calendar days in the training window.
        val_days:       Calendar days covered by each validation window.
        step_days:      Days by which the split point advances per fold.

    Returns:
        List of CVFold objects, containing only the folds walk_forward_cv will
        actually run (see :func:`fold_is_usable`), re-indexed from 0.

    Raises:
        InsufficientDataError: Fewer than MIN_USABLE_FOLDS *usable* folds. The
            message reports both counts and the approximate unlock date, so
            "2 usable out of 13 generated" can never again be reported as 13.
    """
    dates = sorted(snapshot_dates)
    if not dates:
        raise InsufficientDataError("No snapshot dates available.")

    available = {date.fromisoformat(d) for d in dates}
    start = date.fromisoformat(dates[0])
    end = date.fromisoformat(dates[-1])

    folds: list[CVFold] = []
    # First valid train_end: start + (min_train_days-1) days so the window
    # spans exactly min_train_days inclusive calendar days.
    train_end = start + timedelta(days=min_train_days - 1)

    while True:
        val_end = train_end + timedelta(days=val_days)
        if val_end > end:
            break
        val_start = train_end + timedelta(days=1)
        folds.append(
            CVFold(
                fold_idx=len(folds),
                train_start=str(start),
                train_end=str(train_end),
                val_start=str(val_start),
                val_end=str(val_end),
            )
        )
        train_end += timedelta(days=step_days)

    usable = [f for f in folds if fold_is_usable(f, available)]

    if len(usable) < MIN_USABLE_FOLDS:
        # Earliest date allowing MIN_USABLE_FOLDS usable folds:
        # start + (min_train_days-1) + 2*step_days + val_days, plus the target
        # horizon the last fold's validation snapshot still needs a partner for.
        unlock = start + timedelta(
            days=min_train_days - 1 + 2 * step_days + val_days + TARGET_HORIZON_DAYS
        )
        raise InsufficientDataError(
            f"Only {len(usable)} usable fold(s) out of {len(folds)} generated "
            f"(minimum {MIN_USABLE_FOLDS} required). A fold is usable only when "
            f"its training and validation snapshots each have a snapshot exactly "
            f"{TARGET_HORIZON_DAYS} days later, which build_target needs. "
            f"Walk-forward CV unlocks at approximately {unlock.isoformat()}. "
            f"Current data spans {str(start)} to {str(end)} "
            f"({len(available)} snapshots)."
        )

    return [replace(f, fold_idx=i) for i, f in enumerate(usable)]


def walk_forward_cv(
    conn: duckdb.DuckDBPyConnection,
    model: Any,
    folds: list[CVFold] | None = None,
) -> pd.DataFrame:
    """Run walk-forward CV and return per-fold per-tier metrics.

    For each fold the function:
      1. Finds the last available snapshot in the train and val date windows.
      2. Builds lag features via lag.py, then enriches them with enrich_lag_df().
      3. Joins static card features from gold_card_features enriched with
         enrich_card_df() — same enrichments as build_inference_features(),
         eliminating training/serving skew.
      4. Builds log_return_7d targets via lag.py.
      5. Fits a fresh sklearn feature pipeline on train data.
      6. Fits the model on the transformed train features.
      7. Evaluates predictions per price tier using evaluate_per_tier().

    enrich_card_df() is called once before the fold loop (card attributes are
    static); enrich_lag_df() is called per fold (lag features vary by snapshot).

    Args:
        conn:  Open DuckDB connection with gold_price_features and
               gold_card_features tables in scope.
        model: Any model with fit(X_train, y_train, X_val, y_val) and
               predict(X) → np.ndarray. Typical choice: LightGBMPriceModel.
        folds: Walk-forward folds generated by generate_folds(). When None,
               folds are generated automatically from the available snapshots
               using default parameters (min_train_days=30, val_days=7,
               step_days=7), and only usable ones are returned — see
               generate_folds(). Raises InsufficientDataError if fewer than
               MIN_USABLE_FOLDS are usable. Folds passed in explicitly are run
               as given; the skip branches below still guard them.

    Returns:
        DataFrame with columns: fold_idx, val_snapshot, model, tier, n_cards,
        mae, mape. Empty DataFrame (same schema) if no fold produces any data.
        val_snapshot is the ISO date of the validation snapshot used in each fold.
    """
    if folds is None:
        folds = generate_folds(get_available_snapshots(conn))

    card_df = enrich_card_df(conn.execute("SELECT * FROM gold_card_features").df())
    all_results: list[pd.DataFrame] = []

    for fold in folds:
        # Last available snapshot within each date window
        _train_row = conn.execute(
            "SELECT MAX(snapshot_date) FROM gold_price_features WHERE snapshot_date <= ?",
            [fold.train_end],
        ).fetchone()
        _val_row = conn.execute(
            "SELECT MAX(snapshot_date) FROM gold_price_features "
            "WHERE snapshot_date >= ? AND snapshot_date <= ?",
            [fold.val_start, fold.val_end],
        ).fetchone()
        train_snap = _train_row[0] if _train_row else None
        val_snap = _val_row[0] if _val_row else None

        if train_snap is None or val_snap is None:
            continue

        train_snap = str(train_snap)
        val_snap = str(val_snap)

        lag_train = enrich_lag_df(build_lag_features(conn, train_snap))
        target_train = build_target(conn, train_snap)
        X_train_raw, y_train = prepare_training_data(lag_train, card_df, target_train)

        lag_val = enrich_lag_df(build_lag_features(conn, val_snap))
        target_val = build_target(conn, val_snap)
        X_val_raw, y_val = prepare_training_data(lag_val, card_df, target_val)

        if X_train_raw.empty or X_val_raw.empty:
            continue

        # Save eur for tier assignment before the pipeline drops it
        val_eur = (
            X_val_raw["eur"].reset_index(drop=True)
            if "eur" in X_val_raw.columns
            else pd.Series(np.zeros(len(X_val_raw)))
        )

        pipeline = build_feature_pipeline().fit(X_train_raw)
        feature_names = get_feature_names(pipeline)

        X_train = pd.DataFrame(pipeline.transform(X_train_raw), columns=feature_names)
        X_val = pd.DataFrame(pipeline.transform(X_val_raw), columns=feature_names)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        model.fit(X_train, y_train, X_val, y_val)
        y_pred = pd.Series(model.predict(X_val), name="predicted")

        tiers = val_eur.apply(assign_tier)
        metrics_df = evaluate_per_tier(y_val, {"model": y_pred}, tiers)
        metrics_df["fold_idx"] = fold.fold_idx
        metrics_df["val_snapshot"] = val_snap
        all_results.append(metrics_df)

    if not all_results:
        return pd.DataFrame(
            columns=[
                "fold_idx",
                "val_snapshot",
                "model",
                "tier",
                "n_cards",
                "mae",
                "mape",
            ]
        )

    return pd.concat(all_results, ignore_index=True)
