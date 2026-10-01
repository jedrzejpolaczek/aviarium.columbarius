"""Walk-forward backtest of the /underpriced flag against real price history.

Question answered: of the cards the API would have flagged as underpriced on
day d, what share actually rose more than 10% by d+7 — and how does that
compare with the share of *all* eligible cards that did?

The comparison with the base rate is the point. A hit rate means nothing on its
own: if 12% of the whole catalogue rises >10% in a given week, a flag with a
12% hit rate has learned nothing.

No lookahead. For each evaluation date d the model is trained on snapshot d-7,
whose target (the d-7 → d return) is already known on day d — exactly the data
a model retrained on d would have. It then scores every card on d through the
same feature pipeline, turns predicted log-returns into EUR prices, applies
``flag_underpriced`` (the rule behind ``GET /underpriced``), and the outcome is
read from snapshot d+7 by ``backtest_underpriced``.

The horizon is 7 days because that is what the model predicts (the flag's own
reason text reads "ML predicts +X% in 7d"). An evaluation date is used only if
snapshots d-7 and d+7 both exist; gaps in the Gold history skip dates.

Usage:
    python -m scripts.backtest_underpriced
    python -m scripts.backtest_underpriced --out notebooks/ml_models/underpriced_backtest.csv
"""

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.data.repository import GOLD_DB_PATH, open_repository
from src.logger import get_logger, setup_logging
from src.ml.features.lag import build_lag_features, build_target
from src.ml.features.pipeline import (
    build_feature_pipeline,
    build_inference_features,
    enrich_card_df,
    enrich_lag_df,
    get_feature_names,
    prepare_training_data,
)
from src.ml.models.lightgbm_model import LightGBMPriceModel
from src.ml.recommendation.underpriced import backtest_underpriced, flag_underpriced

logger = get_logger(__name__)

HORIZON_DAYS = 7
APPRECIATION_THRESHOLD = 0.10
DEFAULT_OUT = Path("notebooks/ml_models/underpriced_backtest.csv")


@dataclass(frozen=True)
class DayResult:
    eval_date: str
    train_snapshot: str
    check_date: str
    eligible: int
    base_hits: int
    flagged: int
    flagged_hits: int
    flagged_mean_change: float
    eligible_mean_change: float
    eligible_below_1eur: int
    base_hits_below_1eur: int
    flagged_below_1eur: int
    flagged_eur_gain: float
    """Sum over flagged cards of (price on d+7 - price on d), in EUR: what buying
    one copy of every flag and selling a week later would have made, before fees."""


def evaluation_dates(snapshots: list[date]) -> list[tuple[date, date, date]]:
    """Return (train, eval, check) triples where all three snapshots exist."""
    available = set(snapshots)
    step = timedelta(days=HORIZON_DAYS)
    return [
        (d - step, d, d + step)
        for d in sorted(available)
        if d - step in available and d + step in available
    ]


def _price_changes(
    conn: duckdb.DuckDBPyConnection, start: date, end: date
) -> pd.DataFrame:
    """uuid, eur on *start*, and relative change to *end* (both prices present)."""
    return conn.execute(
        """
        SELECT a.uuid, a.eur, (b.eur - a.eur) / GREATEST(a.eur, 0.01) AS change
        FROM gold_price_features a
        JOIN gold_price_features b
          ON a.uuid = b.uuid AND b.snapshot_date = ?
        WHERE a.snapshot_date = ? AND a.eur IS NOT NULL AND b.eur IS NOT NULL
        """,
        [end, start],
    ).df()


def flag_on(
    conn: duckdb.DuckDBPyConnection,
    card_df: pd.DataFrame,
    train_snapshot: date,
    eval_date: date,
) -> pd.DataFrame:
    """Train on *train_snapshot*, score *eval_date*, return flag_underpriced output."""
    lag_train = enrich_lag_df(build_lag_features(conn, str(train_snapshot)))
    target_train = build_target(conn, str(train_snapshot))
    X_train_raw, y_train = prepare_training_data(lag_train, card_df, target_train)

    pipeline = build_feature_pipeline().fit(X_train_raw)
    names = get_feature_names(pipeline)
    X_train = pd.DataFrame(pipeline.transform(X_train_raw), columns=names)
    model = LightGBMPriceModel()
    model.fit(X_train, y_train.reset_index(drop=True))

    X_eval_raw = build_inference_features(conn, str(eval_date))
    X_eval_raw = X_eval_raw[X_eval_raw["eur"].notna()].reset_index(drop=True)
    X_eval = pd.DataFrame(pipeline.transform(X_eval_raw), columns=names)
    log_return = model.predict(X_eval)

    eur = X_eval_raw["eur"].to_numpy(dtype=float)
    scored = pd.DataFrame(
        {
            "uuid": X_eval_raw["uuid"],
            "eur": eur,
            "predicted_eur": np.expm1(np.log1p(eur) + log_return),
        }
    )
    return flag_underpriced(scored)


def run_backtest(conn: duckdb.DuckDBPyConnection) -> list[DayResult]:
    snapshots = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT snapshot_date FROM gold_price_features ORDER BY 1"
        ).fetchall()
    ]
    card_df = enrich_card_df(conn.execute("SELECT * FROM gold_card_features").df())
    results: list[DayResult] = []
    for train_snap, eval_date, check_date in evaluation_dates(snapshots):
        flags = flag_on(conn, card_df, train_snap, eval_date)
        eligible = flags[flags["tier"].isin([1, 2])]
        flagged_uuids = flags.loc[flags["is_underpriced"], "uuid"].tolist()

        outcome = backtest_underpriced(
            conn,
            str(eval_date),
            str(check_date),
            flagged_uuids,
            appreciation_threshold=APPRECIATION_THRESHOLD,
        )
        changes = _price_changes(conn, eval_date, check_date)
        base = changes[changes["uuid"].isin(eligible["uuid"])]
        cheap = base[base["eur"] < 1.0]
        flagged_changes = base[base["uuid"].isin(flagged_uuids)]

        day = DayResult(
            eval_date=str(eval_date),
            train_snapshot=str(train_snap),
            check_date=str(check_date),
            eligible=len(base),
            base_hits=int((base["change"] > APPRECIATION_THRESHOLD).sum()),
            flagged=int(outcome["total_flagged"]),
            flagged_hits=int(outcome["appreciated"]),
            flagged_mean_change=float(outcome["avg_appreciation"]),
            eligible_mean_change=float(base["change"].mean()) if len(base) else 0.0,
            eligible_below_1eur=len(cheap),
            base_hits_below_1eur=int((cheap["change"] > APPRECIATION_THRESHOLD).sum()),
            flagged_below_1eur=int((flagged_changes["eur"] < 1.0).sum()),
            flagged_eur_gain=float(
                (flagged_changes["eur"] * flagged_changes["change"]).sum()
            ),
        )
        logger.info(
            "%s: flagged %d (hits %d), eligible %d (hits %d)",
            day.eval_date,
            day.flagged,
            day.flagged_hits,
            day.eligible,
            day.base_hits,
        )
        results.append(day)
    return results


def summarise(all_results: list[DayResult]) -> dict[str, float]:
    """Pooled hit rate of the flag vs the base rate over the same card-days.

    Dates on which no eligible card moved at all are left out: they come from
    the period when the price feed was frozen (up to early July 2026), carry no
    flags, and would only drag the base rate towards zero.

    The flag only ever fires on sub-€1 cards in practice, so the base rate within
    that band is reported too — it is the fair comparison.
    """
    results = [r for r in all_results if r.base_hits or r.flagged]
    flagged = sum(r.flagged for r in results)
    eligible = sum(r.eligible for r in results)
    flagged_hits = sum(r.flagged_hits for r in results)
    base_hits = sum(r.base_hits for r in results)
    return {
        "evaluation_dates": len(results),
        "dates_with_flags": sum(1 for r in results if r.flagged),
        "flagged_card_days": flagged,
        "flag_hit_rate": flagged_hits / flagged if flagged else float("nan"),
        "eligible_card_days": eligible,
        "base_hit_rate": base_hits / eligible if eligible else float("nan"),
        "frozen_dates_skipped": len(all_results) - len(results),
        "flagged_share_below_1eur": (
            sum(r.flagged_below_1eur for r in results) / flagged
            if flagged
            else float("nan")
        ),
        "base_hit_rate_below_1eur": _ratio(
            sum(r.base_hits_below_1eur for r in results),
            sum(r.eligible_below_1eur for r in results),
        ),
        "flagged_eur_gain_total": sum(r.flagged_eur_gain for r in results),
    }


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def main(argv: list[str] | None = None) -> int:
    setup_logging(log_dir=Path("logs"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", default=GOLD_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv if argv is not None else [])

    repo = open_repository(args.db_path, read_only=True)
    try:
        results = run_backtest(repo.connection)
    finally:
        repo.close()

    if not results:
        logger.error("No evaluation date has snapshots on both d-7 and d+7.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in results]).to_csv(args.out, index=False)
    for key, value in summarise(results).items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"Per-date results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
