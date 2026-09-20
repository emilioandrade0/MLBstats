"""Walk-forward backtest — strict, no leakage.

For each target season T in [start..end]:
  1. Train on all games where season < T.
  2. Predict week-by-week for season T; after every week completes,
     REFIT with the week just observed added to the training set.
  3. Emit calibrated probabilities: home_win, cover_home, total_over.

Also predicts UPCOMING games (status != final) using a model trained on
everything up to now.

Outputs:
  data/processed/walkforward_preds.parquet
      One row per game with model_home_win_prob, model_cover_home_prob,
      model_total_over_prob, and CLV-ready fields.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from ..paths import PROCESSED

FEATURE_COLS = [
    "elo_diff", "home_elo_pre", "away_elo_pre",
    "epa_off_diff_l4", "epa_def_diff_l4",
    "home_epa_off_per_play_l4", "away_epa_off_per_play_l4",
    "home_epa_def_per_play_l4", "away_epa_def_per_play_l4",
    "home_epa_off_per_play_l8", "away_epa_off_per_play_l8",
    "home_success_rate_off_l4", "away_success_rate_off_l4",
    "spread_line", "total_line",
    "home_rest", "away_rest", "rest_diff",
    "is_dome", "temp", "wind",
]

TARGETS = ["home_win", "cover_home", "total_over"]


@dataclass
class WFConfig:
    start_season: int = 2015
    end_season: int = 2025
    min_train_games: int = 500
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 5
    calibration_method: str = "isotonic"
    seed: int = 42


def _prep(df: pd.DataFrame, cfg: WFConfig) -> pd.DataFrame:
    df = df.copy()
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    return df


def _fit_calibrated(X: pd.DataFrame, y: pd.Series, cfg: WFConfig) -> CalibratedClassifierCV:
    base = LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        num_leaves=31,
        min_child_samples=20,
        random_state=cfg.seed,
        verbose=-1,
    )
    n_splits = min(5, max(2, len(y) // 200))
    return CalibratedClassifierCV(base, method=cfg.calibration_method, cv=n_splits).fit(X, y)


def walkforward(df: pd.DataFrame, cfg: WFConfig) -> pd.DataFrame:
    df = _prep(df, cfg).sort_values("kickoff_utc").reset_index(drop=True)
    predictions: list[pd.DataFrame] = []

    seasons = sorted(df["season"].unique())
    seasons = [s for s in seasons if cfg.start_season <= s <= cfg.end_season]

    for season in seasons:
        past = df[df["season"] < season]
        current = df[df["season"] == season].sort_values(["week", "kickoff_utc"])
        weeks = sorted(current["week"].dropna().unique().tolist())

        train_pool = past.copy()
        print(f"\n-- Season {season}  (train pool: {len(train_pool):,} games)")

        for wk in weeks:
            wk_games = current[current["week"] == wk]
            preds_wk: dict[str, np.ndarray] = {}

            for target in TARGETS:
                y = train_pool[target].dropna()
                if len(y) < cfg.min_train_games:
                    preds_wk[target] = np.full(len(wk_games), np.nan)
                    continue
                X = train_pool.loc[y.index, FEATURE_COLS].fillna(-999)
                model = _fit_calibrated(X, y.astype(int), cfg)
                Xp = wk_games[FEATURE_COLS].fillna(-999)
                preds_wk[target] = model.predict_proba(Xp)[:, 1]

            preds = wk_games[["game_id", "season", "week", "status"]].copy()
            preds["model_home_win_prob"] = preds_wk["home_win"]
            preds["model_cover_home_prob"] = preds_wk["cover_home"]
            preds["model_total_over_prob"] = preds_wk["total_over"]
            predictions.append(preds)

            # After the week is played, fold into the training pool for the next week
            finals = wk_games[wk_games["status"] == "final"]
            if not finals.empty:
                train_pool = pd.concat([train_pool, finals], ignore_index=True)

            print(f"  wk {int(wk):>2}: predicted {len(wk_games):>2} games  "
                  f"(pool now {len(train_pool):,})")

    out = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    path = PROCESSED / "walkforward_preds.parquet"
    out.to_parquet(path, index=False)
    print(f"\nwalkforward_preds.parquet: {len(out):,} predictions -> {path}")
    return out


def evaluate(preds: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Merge predictions with truth + closing lines; compute Brier, log-loss, CLV."""
    truth = df[["game_id", "home_win", "cover_home", "total_over",
                "spread_line", "total_line"]]
    m = preds.merge(truth, on="game_id", how="left")
    finals = m[m["home_win"].notna()].copy()
    if finals.empty:
        print("No finals to evaluate yet.")
        return finals

    def brier(p, y): return float(((p - y) ** 2).mean())
    def logloss(p, y):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    print("\n-- Evaluation on completed games --")
    for tgt, col in [("home_win", "model_home_win_prob"),
                     ("cover_home", "model_cover_home_prob"),
                     ("total_over", "model_total_over_prob")]:
        sub = finals.dropna(subset=[col, tgt])
        if sub.empty:
            continue
        print(f"  {tgt:<12} n={len(sub):>5}  brier={brier(sub[col].values, sub[tgt].values):.4f}  "
              f"logloss={logloss(sub[col].values, sub[tgt].values):.4f}")
    return finals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / "train.parquet")
    cfg = WFConfig(start_season=args.start, end_season=args.end)
    preds = walkforward(df, cfg)
    evaluate(preds, df)


if __name__ == "__main__":
    main()
