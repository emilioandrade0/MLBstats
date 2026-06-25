"""Stacked model: use market_logit as offset, train LightGBM to predict only the residual.

This forces the model to start FROM the market line. It can only nudge that line
based on signals not already in the market. We regularize hard so the nudge stays
small.

Architecture:
  init_score = logit(market_p_home)      # forced prior, no parameters
  residual   = lgb.train(...)            # learns small adjustment from Statcast etc
  p_final    = sigmoid(init_score + residual)

If even with hard regularization the residual model can't improve on log_loss vs
"market alone", that means our features genuinely don't have additional info
beyond what the market already prices in — useful finding either way.
"""
from __future__ import annotations

import argparse
import pickle
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from ..normalize.paths import PROCESSED
from .dataset import EXCLUDE_COLS

MODELS = PROCESSED.parent / "models"
MODELS.mkdir(parents=True, exist_ok=True)

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# These columns describe the market itself; we either use them as offset
# (market_logit_p_home) or as auxiliary features (std, n_providers, totals).
MARKET_OFFSET = "market_logit_p_home"
KEEP_AUX_MARKET = {"market_p_home_std", "market_n_providers", "market_over_under", "market_spread"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    # Drop targets/meta and the prior we use as offset; drop raw market_p too
    # since it's redundant with the offset.
    exclude = EXCLUDE_COLS | {MARKET_OFFSET, "market_p_home"}
    feats = [c for c in df.columns if c not in exclude]
    return [c for c in feats if df[c].notna().any()]


def _split(df: pd.DataFrame, train_end: str, val_end: str, test_end: str):
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    te = pd.Timestamp(train_end); ve = pd.Timestamp(val_end); se = pd.Timestamp(test_end)
    tr = df[df["game_date"] <= te]
    vl = df[(df["game_date"] > te) & (df["game_date"] <= ve)]
    ts = df[(df["game_date"] > ve) & (df["game_date"] <= se)]
    return tr, vl, ts


def _train_lgb_residual(Xtr, ytr, offset_tr, Xvl, yvl, offset_vl, params: dict | None = None) -> lgb.Booster:
    """LightGBM with init_score — model predicts log-odds residual on top of offset."""
    base = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.02,
        "num_leaves": 8,           # SMALL — heavy reg
        "min_child_samples": 100,  # SMALL leaves only with enough data
        "reg_lambda": 5.0,
        "reg_alpha": 0.5,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": 42,
    }
    if params:
        base.update(params)
    dtr = lgb.Dataset(Xtr, label=ytr, init_score=offset_tr)
    dvl = lgb.Dataset(Xvl, label=yvl, init_score=offset_vl, reference=dtr)
    return lgb.train(
        base, dtr, num_boost_round=2000, valid_sets=[dvl],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def _eval_block(name: str, y, p_model, p_market):
    print(f"\n=== {name} (n={len(y):,}) ===")
    print(f"  base rate:           {y.mean():.4f}")
    print(f"  market log_loss:     {log_loss(y, p_market):.4f}")
    print(f"  stacked log_loss:    {log_loss(y, p_model):.4f}   delta={log_loss(y, p_model)-log_loss(y, p_market):+.4f}")
    print(f"  market AUC:          {roc_auc_score(y, p_market):.4f}")
    print(f"  stacked AUC:         {roc_auc_score(y, p_model):.4f}")
    print(f"  market accuracy:     {((p_market>0.5)==y).mean():.4f}")
    print(f"  stacked accuracy:    {((p_model>0.5)==y).mean():.4f}")
    print(f"  stacked Brier:       {brier_score_loss(y, p_model):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--val-end", default="2025-07-31")
    parser.add_argument("--test-end", default="2025-12-31")
    args = parser.parse_args()

    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.dropna(subset=["home_win", "total_runs", MARKET_OFFSET]).copy()
    feats = feature_columns(df)
    print(f"using {len(feats)} non-market features over {len(df):,} games")

    tr, vl, ts = _split(df, args.train_end, args.val_end, args.test_end)
    print(f"train: {len(tr):,}  val: {len(vl):,}  test: {len(ts):,}")

    # Offsets (log-odds of market prior)
    o_tr = tr[MARKET_OFFSET].values
    o_vl = vl[MARKET_OFFSET].values
    o_ts = ts[MARKET_OFFSET].values

    # Targets
    y_tr = tr["home_win"].astype(int).values
    y_vl = vl["home_win"].astype(int).values
    y_ts = ts["home_win"].astype(int).values

    # Market baseline probabilities (from the offset itself)
    p_market_vl = _sigmoid(o_vl)
    p_market_ts = _sigmoid(o_ts)

    # ---- Stacked binary classifier ----
    print("\n[binary] training stacked LightGBM with init_score=logit(market)…")
    model_cls = _train_lgb_residual(
        tr[feats], y_tr, o_tr, vl[feats], y_vl, o_vl,
    )

    # Predictions: model returns logit residual (raw_score), final = sigmoid(offset + raw)
    # In LightGBM, predict() with raw_score=True gives the raw model output WITHOUT init_score.
    # We add init_score ourselves.
    raw_vl = model_cls.predict(vl[feats], raw_score=True)
    raw_ts = model_cls.predict(ts[feats], raw_score=True)
    p_stk_vl = _sigmoid(o_vl + raw_vl)
    p_stk_ts = _sigmoid(o_ts + raw_ts)

    _eval_block("VAL", y_vl, p_stk_vl, p_market_vl)
    _eval_block("TEST", y_ts, p_stk_ts, p_market_ts)

    # How much did the model nudge the market on average?
    delta_logit_vl = raw_vl
    delta_p_vl = p_stk_vl - p_market_vl
    print(f"\n  mean |delta logit|: {np.mean(np.abs(delta_logit_vl)):.3f}")
    print(f"  mean |delta prob|:  {np.mean(np.abs(delta_p_vl)):.3f}")
    print(f"  max  |delta prob|:  {np.max(np.abs(delta_p_vl)):.3f}")
    print(f"  best_iter:          {model_cls.best_iteration}")

    # Top features
    fi = pd.Series(model_cls.feature_importance(importance_type="gain"), index=feats)
    fi = fi.sort_values(ascending=False)
    print("\n  Top 15 features by gain:")
    print(fi.head(15).to_string())

    # ---- Stacked total-runs regressor ----
    # For totals we just use the market line as offset on top of standard regression.
    print("\n[totals] training LightGBM total-runs (market total as feature)…")
    reg = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=15,
        min_child_samples=80, reg_lambda=2.0, subsample=0.85,
        colsample_bytree=0.8, random_state=42, verbosity=-1,
        objective="regression",
    )
    reg.fit(
        tr[feats], tr["total_runs"].astype(float),
        eval_set=[(vl[feats], vl["total_runs"].astype(float))],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_total_vl = reg.predict(vl[feats])
    p_total_ts = reg.predict(ts[feats])
    print(f"  val MAE:  {mean_absolute_error(vl['total_runs'], p_total_vl):.3f}")
    print(f"  test MAE: {mean_absolute_error(ts['total_runs'], p_total_ts):.3f}")
    # Compare to market_over_under as predictor
    mou_ts = ts["market_over_under"].dropna()
    if len(mou_ts):
        ts_with_mou = ts[ts["market_over_under"].notna()]
        print(f"  market total as predictor — test MAE: "
              f"{mean_absolute_error(ts_with_mou['total_runs'], ts_with_mou['market_over_under']):.3f}")

    # Save artifacts
    with open(MODELS / "lgb_stacked_cls.pkl", "wb") as f:
        pickle.dump({"model": model_cls, "feature_names": feats,
                     "offset_col": MARKET_OFFSET}, f)
    with open(MODELS / "lgb_stacked_reg.pkl", "wb") as f:
        pickle.dump({"model": reg, "feature_names": feats}, f)
    print(f"\nartifacts saved to {MODELS}")


if __name__ == "__main__":
    main()
