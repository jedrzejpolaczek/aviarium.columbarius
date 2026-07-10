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
Walk-forward CV needs >= 3 folds. With the default parameters, the first
three usable folds appear after approximately 50 days of daily snapshots
(start + 29 train days + 2×7 step days + 7 val days).
Raises InsufficientDataError if fewer than 3 folds can be generated.
"""

import json
from dataclasses import dataclass
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
    """Raised when there are too few snapshots to run walk-forward CV (need >= 3 folds)."""


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
        List of CVFold objects.

    Raises:
        InsufficientDataError: Fewer than 3 folds can be generated. The error
            message includes the approximate date when CV will become possible.
    """
    dates = sorted(snapshot_dates)
    if not dates:
        raise InsufficientDataError("No snapshot dates available.")

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

    if len(folds) < 3:
        # Earliest date that would allow 3 folds:
        # start + (min_train_days-1) + 2*step_days + val_days
        unlock = start + timedelta(days=min_train_days - 1 + 2 * step_days + val_days)
        raise InsufficientDataError(
            f"Only {len(folds)} fold(s) generated (minimum 3 required). "
            f"Walk-forward CV unlocks at approximately {unlock.isoformat()}. "
            f"Current data spans {str(start)} to {str(end)}."
        )

    return folds


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
               step_days=7). Raises InsufficientDataError if fewer than 3
               folds can be generated.

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
