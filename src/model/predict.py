"""Generate predictions for upcoming MLB games.

Reads:
  data/processed/train.parquet  — all games + features, including future ones
                                 (their home_win/total_runs are NaN)
  data/models/lgb_cls.pkl, lgb_reg.pkl

For every row with future game_date and non-null features, output:
  game_pk, game_date, away, home, p_home_win, predicted_total_runs,
  implied_home_ml (American odds equivalent), suggested_pick

Usage:
  python -m src.model.predict                          # all future games
  python -m src.model.predict --date 2026-06-22        # specific day
  python -m src.model.predict --days 7                 # next 7 days
"""
from __future__ import annotations

import argparse
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED
from .dataset import EXCLUDE_COLS

MODELS = PROCESSED.parent / "models"


def _to_american_odds(p: float) -> int:
    if not np.isfinite(p) or p <= 0 or p >= 1:
        return 0
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def load_models():
    # Prefer the new ensemble (market blend) if available.
    ens_path = MODELS / "ensemble.pkl"
    if ens_path.exists():
        with open(ens_path, "rb") as f:
            cls_bundle = pickle.load(f)
        cls_bundle["__ensemble__"] = True
    else:
        with open(MODELS / "lgb_cls.pkl", "rb") as f:
            cls_bundle = pickle.load(f)
        cls_bundle["__ensemble__"] = False
    with open(MODELS / "lgb_reg.pkl", "rb") as f:
        reg_bundle = pickle.load(f)
    return cls_bundle, reg_bundle


def predict(start: date | None = None, end: date | None = None) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    # Future = no recorded score yet. total_runs is NaN for unplayed games.
    future = df[df["total_runs"].isna()].copy()
    if start:
        future = future[future["game_date"] >= start]
    if end:
        future = future[future["game_date"] <= end]

    cls_b, reg_b = load_models()
    feats = cls_b["feature_names"]
    X = future[feats]

    # Drop rows where all features are missing (probable starter unknown etc).
    keep = X.notna().any(axis=1)
    future = future.loc[keep].copy()
    X = X.loc[keep]

    if cls_b.get("__ensemble__"):
        # ensemble: blend market with model
        p_model = cls_b["model"].predict_proba(X)[:, 1]
        p_market = future["market_p_home"].clip(1e-6, 1 - 1e-6).values
        w = cls_b["blend_weight_market"]
        # When market is missing, fall back to model only.
        p_home = np.where(np.isnan(p_market),
                          p_model,
                          w * p_market + (1 - w) * p_model)
    else:
        p_home = cls_b["calibrator"].predict_proba(X)[:, 1]
    pred_runs = reg_b["model"].predict(X)

    out = future[["game_pk", "game_date", "away_team_abbrev", "home_team_abbrev",
                  "probable_away_pitcher_id", "probable_home_pitcher_id",
                  "venue_name", "weather_temp_f"]].copy()
    out["p_home_win"] = p_home.round(4)
    out["p_away_win"] = (1 - p_home).round(4)
    out["pred_total_runs"] = pred_runs.round(2)
    out["fair_home_ml"] = [_to_american_odds(p) for p in p_home]
    out["fair_away_ml"] = [_to_american_odds(1 - p) for p in p_home]
    out["pick"] = np.where(p_home > 0.55, "HOME",
                  np.where(p_home < 0.45, "AWAY", "—"))
    out = out.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=date.fromisoformat, help="Single date")
    p.add_argument("--start", type=date.fromisoformat)
    p.add_argument("--end", type=date.fromisoformat)
    p.add_argument("--days", type=int, help="Next N days from today")
    args = p.parse_args()

    if args.date:
        start = end = args.date
    elif args.days:
        start = date.today()
        end = start + timedelta(days=args.days)
    else:
        start, end = args.start, args.end

    df = predict(start, end)
    if df.empty:
        print("No future games match.")
        return
    print(df.to_string(index=False))
    out = PROCESSED / "predictions.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
