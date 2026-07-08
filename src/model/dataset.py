"""Load train.parquet, do time-based train/val/test split, return X/y arrays.

Time-based split is non-negotiable: training on the future to predict the past
leaks. We split by game_date.

Defaults:
  train:  2023-01-01 to 2024-12-31
  val:    2025-01-01 to 2025-07-31
  test:   2025-08-01 to 2025-12-31
  pred:   2026 (no target — for inference)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED

EXCLUDE_COLS = {
    "game_pk", "game_date", "season", "game_type", "status",
    "home_team_abbrev", "away_team_abbrev",
    # NOTE: home_team_id and away_team_id ARE included as categorical features.
    # Walk-forward showed +1.17pp acc and +3.00pp on high-conf band when paired
    # with stronger regularization (reg=1.0, leaves=16) in train.py.
    "venue_id", "venue_name", "roof_type",
    "probable_home_pitcher_id", "probable_away_pitcher_id",
    "home_score", "away_score",
    "home_win", "total_runs",
    # Raw market probability — we expose the logit form to the model;
    # leaving p_home itself in would double-count.
    "market_p_home",
    # F5 targets — never features for full-game model and vice versa
    "f5_home_score", "f5_away_score", "f5_home_won", "f5_tied",
    "f5_total_runs", "f5_run_diff",
    # Context-only Pythagorean regression features: available for post-model
    # adjustments, intentionally excluded from base model training.
    # Tried exposing pyth_wpct_l30/l10 and pyth_minus_actual_l30 to the base
    # model (walk-forward Apr-Jun 2026, n=1205): acc 0.5394 vs baseline 0.5809
    # (-4.15pp), AUC 0.5583 vs 0.5951 (-3.68pp), log_loss 0.6884 vs 0.6746.
    # Features landed in top-15 by importance but hurt OOS — likely overfit
    # against the win_pct_l30 / run_diff_l10 features that carry the same
    # signal. Post-process nudge in api.py is enough.
    "pyth_wpct_l30_h", "pyth_wpct_l30_a", "pyth_wpct_l30_diff",
    "pyth_wpct_l10_h", "pyth_wpct_l10_a", "pyth_wpct_l10_diff",
    "pyth_minus_actual_l30_h", "pyth_minus_actual_l30_a", "pyth_minus_actual_l30_diff",
    "pyth_run_diff_pg_l30_h", "pyth_run_diff_pg_l30_a", "pyth_run_diff_pg_l30_diff",
}

# Categorical features — LightGBM handles these natively. Must be int.
CATEGORICAL_FEATURES = ["home_team_id", "away_team_id"]


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train_cls: pd.Series
    y_train_reg: pd.Series
    X_val: pd.DataFrame
    y_val_cls: pd.Series
    y_val_reg: pd.Series
    X_test: pd.DataFrame
    y_test_cls: pd.Series
    y_test_reg: pd.Series
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame
    feature_names: list[str]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def load_split(
    train_end: str = "2024-12-31",
    val_end: str = "2025-07-31",
    test_end: str = "2025-12-31",
) -> Split:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.dropna(subset=["home_win", "total_runs"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    fcols = feature_columns(df)
    # Drop features that are entirely NaN (no signal).
    nan_frac = df[fcols].isna().mean()
    fcols = [c for c in fcols if nan_frac[c] < 0.95]

    train_end_d = pd.Timestamp(train_end)
    val_end_d = pd.Timestamp(val_end)
    test_end_d = pd.Timestamp(test_end)

    train = df[df["game_date"] <= train_end_d]
    val = df[(df["game_date"] > train_end_d) & (df["game_date"] <= val_end_d)]
    test = df[(df["game_date"] > val_end_d) & (df["game_date"] <= test_end_d)]

    meta_cols = ["game_pk", "game_date", "season", "home_team_abbrev",
                 "away_team_abbrev", "home_score", "away_score"]

    def _y(d: pd.DataFrame):
        X = d[fcols].copy()
        # Cast categorical features to int32 so LightGBM treats them correctly.
        for c in CATEGORICAL_FEATURES:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-1).astype("int32")
        return X, d["home_win"].astype(int), d["total_runs"].astype(float), d[meta_cols]

    Xt, yt_c, yt_r, mt = _y(train)
    Xv, yv_c, yv_r, mv = _y(val)
    Xs, ys_c, ys_r, ms = _y(test)

    return Split(
        X_train=Xt, y_train_cls=yt_c, y_train_reg=yt_r,
        X_val=Xv, y_val_cls=yv_c, y_val_reg=yv_r,
        X_test=Xs, y_test_cls=ys_c, y_test_reg=ys_r,
        meta_train=mt, meta_val=mv, meta_test=ms,
        feature_names=fcols,
    )
