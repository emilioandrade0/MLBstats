"""Walk-forward backtest — the only honest way to evaluate a betting model.

Each month in the test horizon gets its own model:
  - train  = all games with game_date < month_start
  - val    = last 30 days of train (early stopping)
  - test   = the month itself

We refit LightGBM (cls + reg) every month and stitch predictions into a
single `walkforward_preds.parquet`. Then we report aggregate metrics
(log_loss, AUC, Brier, calibration), and run the same betting strategies
as backtest.py — only now over a much longer, unbiased horizon.

Usage:
  python -m src.model.walkforward
  python -m src.model.walkforward --start 2024-05 --end 2025-12
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from tqdm import tqdm

from ..normalize.paths import PROCESSED
from .backtest import _american_to_decimal, _build_market_lines, _implied_prob_american
from .dataset import EXCLUDE_COLS

MODELS = PROCESSED.parent / "models"


def _months(start: date, end: date):
    """Yield (month_start, month_end_inclusive) for each month in [start, end]."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = cur + relativedelta(months=1)
        yield cur, nxt - timedelta(days=1)
        cur = nxt


def _feature_columns(df: pd.DataFrame) -> list[str]:
    feats = [c for c in df.columns if c not in EXCLUDE_COLS]
    # drop features that are entirely NaN — they have no signal
    return [c for c in feats if df[c].notna().any()]


def _train_one_fold(train: pd.DataFrame, feats: list[str]):
    # Use the last 30 days of train as val for early stopping + calibration.
    cutoff = train["game_date"].max() - pd.Timedelta(days=30)
    tr = train[train["game_date"] <= cutoff]
    vl = train[train["game_date"] > cutoff]
    if len(vl) < 50 or len(tr) < 500:
        return None  # not enough data

    Xtr, Xvl = tr[feats], vl[feats]
    cls = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_child_samples=40, reg_lambda=1.0, subsample=0.85,
        colsample_bytree=0.85, random_state=42, verbosity=-1,
    )
    cls.fit(Xtr, tr["home_win"].astype(int),
            eval_set=[(Xvl, vl["home_win"].astype(int))],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    cal = CalibratedClassifierCV(FrozenEstimator(cls), method="isotonic")
    cal.fit(Xvl, vl["home_win"].astype(int))

    reg = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_child_samples=40, reg_lambda=1.0, subsample=0.85,
        colsample_bytree=0.85, random_state=42, verbosity=-1,
        objective="regression",
    )
    reg.fit(Xtr, tr["total_runs"].astype(float),
            eval_set=[(Xvl, vl["total_runs"].astype(float))],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(50, verbose=False)])
    return cal, reg


def run(start: date, end: date) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.dropna(subset=["home_win", "total_runs"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    feats = _feature_columns(df)
    print(f"using {len(feats)} features over {len(df):,} completed games")

    all_preds: list[pd.DataFrame] = []
    months = list(_months(start, end))
    for m_start, m_end in tqdm(months, desc="walkforward"):
        train = df[df["game_date"] < pd.Timestamp(m_start)]
        test = df[(df["game_date"] >= pd.Timestamp(m_start)) &
                  (df["game_date"] <= pd.Timestamp(m_end))]
        if test.empty:
            continue
        models = _train_one_fold(train, feats)
        if models is None:
            continue
        cal, reg = models
        p_home_model = cal.predict_proba(test[feats])[:, 1]
        # Blend with market — tune weight by accuracy on val (matches production).
        cutoff = train["game_date"].max() - pd.Timedelta(days=30)
        vl = train[train["game_date"] > cutoff]
        if "market_p_home" in vl.columns:
            mp_val = vl["market_p_home"].clip(1e-6, 1 - 1e-6).values
            p_val = cal.predict_proba(vl[feats])[:, 1]
            y_val = vl["home_win"].astype(int).values
            best_w, best_acc = 1.0, -1.0
            for w in np.arange(0.0, 1.01, 0.05):
                pv = w * mp_val + (1 - w) * p_val
                acc = ((pv > 0.5) == y_val).mean()
                if acc > best_acc:
                    best_acc, best_w = acc, w
            mp_test = test["market_p_home"].clip(1e-6, 1 - 1e-6).values
            p_home = np.where(np.isnan(mp_test), p_home_model,
                              best_w * mp_test + (1 - best_w) * p_home_model)
        else:
            p_home = p_home_model
        pred_total = reg.predict(test[feats])
        # Stash raw model pred for post-hoc blend experiments (C — disagreement-aware blend, etc.)
        out = test[["game_pk", "game_date", "season", "home_team_abbrev",
                    "away_team_abbrev", "home_score", "away_score",
                    "home_win", "total_runs", "market_p_home",
                    "market_over_under"]].copy()
        out["p_home"] = p_home
        out["p_home_model_raw"] = p_home_model
        out["best_w_fold"] = best_w if "market_p_home" in vl.columns else 1.0
        out["pred_total"] = pred_total
        out["fold_month"] = f"{m_start.year}-{m_start.month:02d}"
        all_preds.append(out)
        continue  # skip the original out append below
        out = test[["game_pk", "game_date", "season", "home_team_abbrev",
                    "away_team_abbrev", "home_score", "away_score",
                    "home_win", "total_runs", "market_p_home",
                    "market_over_under"]].copy()
        out["p_home"] = p_home
        out["pred_total"] = pred_total
        out["fold_month"] = f"{m_start.year}-{m_start.month:02d}"
        all_preds.append(out)

    res = pd.concat(all_preds, ignore_index=True)
    out_path = PROCESSED / "walkforward_preds.parquet"
    res.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path} ({len(res):,} predictions)")
    return res


def _aggregate_metrics(preds: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS  (walk-forward, fully out-of-sample)")
    print("=" * 60)
    y = preds["home_win"].astype(int)
    p = preds["p_home"].clip(1e-6, 1 - 1e-6)
    print(f"n games:        {len(preds):,}")
    print(f"base rate:      {y.mean():.4f}")
    print(f"log_loss:       {log_loss(y, p):.4f}")
    print(f"brier:          {brier_score_loss(y, p):.4f}")
    print(f"AUC:            {roc_auc_score(y, p):.4f}")
    print(f"accuracy@0.5:   {((p > 0.5) == y).mean():.4f}")
    print(f"\ntotal runs:")
    yr = preds["total_runs"].astype(float)
    pr = preds["pred_total"].astype(float)
    print(f"MAE:  {mean_absolute_error(yr, pr):.4f}")
    print(f"RMSE: {mean_squared_error(yr, pr) ** 0.5:.4f}")

    print("\n--- per-fold metrics ---")
    g = preds.groupby("fold_month").apply(
        lambda d: pd.Series({
            "n": len(d),
            "log_loss": log_loss(d["home_win"].astype(int), d["p_home"].clip(1e-6, 1 - 1e-6)),
            "AUC": roc_auc_score(d["home_win"].astype(int), d["p_home"]) if d["home_win"].nunique() > 1 else np.nan,
            "acc": ((d["p_home"] > 0.5) == d["home_win"].astype(int)).mean(),
        }),
        include_groups=False,
    )
    print(g.to_string())


def _backtest(preds: pd.DataFrame, edge_threshold: float = 0.03,
              bankroll0: float = 1000.0, kelly_frac: float = 0.25) -> None:
    lines = _build_market_lines()
    df = preds.merge(lines, on="game_pk", how="inner")
    if df.empty:
        print("\n[backtest] no overlap with market lines")
        return

    df["dec_h"] = df["home_ml"].map(_american_to_decimal)
    df["dec_a"] = df["away_ml"].map(_american_to_decimal)
    p_h = df["home_ml"].map(_implied_prob_american)
    p_a = df["away_ml"].map(_implied_prob_american)
    total = p_h + p_a
    df["p_mkt_h"] = p_h / total
    df["p_mkt_a"] = p_a / total

    df["edge_h"] = df["p_home"] - df["p_mkt_h"]
    df["edge_a"] = (1 - df["p_home"]) - df["p_mkt_a"]

    df["side"] = np.where(
        (df["edge_h"] >= edge_threshold) & (df["edge_h"] >= df["edge_a"]), "HOME",
        np.where((df["edge_a"] >= edge_threshold) & (df["edge_a"] > df["edge_h"]),
                 "AWAY", "NONE"),
    )
    df["bet_p"] = np.where(df["side"] == "HOME", df["p_home"],
                  np.where(df["side"] == "AWAY", 1 - df["p_home"], np.nan))
    df["bet_dec"] = np.where(df["side"] == "HOME", df["dec_h"],
                   np.where(df["side"] == "AWAY", df["dec_a"], np.nan))
    df["bet_won"] = np.where(df["side"] == "HOME", df["home_win"] == 1,
                    np.where(df["side"] == "AWAY", df["home_win"] == 0, np.nan))

    df = df.sort_values("game_date").reset_index(drop=True)
    bets = df[df["side"] != "NONE"].copy()
    if bets.empty:
        print(f"\n[backtest edge>={edge_threshold:.0%}] zero bets")
        return

    # Flat
    pnl_flat = np.where(bets["bet_won"] == 1.0, 100 * (bets["bet_dec"] - 1), -100)
    flat_total = pnl_flat.sum()
    flat_roi = flat_total / (100 * len(bets))

    # Quarter-Kelly with rolling bankroll
    bankroll = bankroll0
    pnl_qk = np.zeros(len(bets))
    stake_qk = np.zeros(len(bets))
    for i, row in bets.reset_index().iterrows():
        b = row["bet_dec"] - 1
        f = max(0.0, (b * row["bet_p"] - (1 - row["bet_p"])) / b) * kelly_frac
        f = min(f, 0.05)
        stake = bankroll * f
        if row["bet_won"] == 1.0:
            pnl_qk[i] = stake * b
        else:
            pnl_qk[i] = -stake
        stake_qk[i] = stake
        bankroll += pnl_qk[i]

    print(f"\n=== Backtest edge>={edge_threshold:.0%} | bets {len(bets):,}/{len(df):,} "
          f"({len(bets)/len(df):.1%}) ===")
    print(f"  win rate:        {(bets['bet_won']==1).mean():.1%}")
    print(f"  ROI flat-$100:   {flat_roi:+.2%}  (P&L ${flat_total:+,.0f} on ${100*len(bets):,} risked)")
    print(f"  Bankroll 1/4K:   ${bankroll:,.0f}  (from ${bankroll0:,.0f})")
    if stake_qk.sum() > 0:
        print(f"  ROI 1/4K:        {pnl_qk.sum() / stake_qk.sum():+.2%}")
    # By edge bucket
    bets["edge_chosen"] = np.where(bets["side"] == "HOME", bets["edge_h"], bets["edge_a"])
    bets["bucket"] = pd.cut(bets["edge_chosen"],
                              [0, 0.03, 0.05, 0.08, 0.15, 1],
                              labels=["0-3%", "3-5%", "5-8%", "8-15%", "15%+"],
                              include_lowest=True)
    print("\n  ROI by edge bucket:")
    for b_, sub in bets.groupby("bucket", observed=True):
        wr = (sub["bet_won"] == 1).mean()
        roi = ((np.where(sub["bet_won"] == 1.0, 100 * (sub["bet_dec"] - 1), -100)).sum() / (100 * len(sub))) if len(sub) else 0
        print(f"    {str(b_):8s}  n={len(sub):4d}  WR={wr:.1%}  ROI={roi:+.2%}")
    # ROI per month
    bets["month"] = bets["game_date"].dt.to_period("M")
    print("\n  ROI per month (flat-$100):")
    for mo, sub in bets.groupby("month", observed=True):
        pnl_mo = np.where(sub["bet_won"] == 1.0, 100 * (sub["bet_dec"] - 1), -100)
        roi_mo = pnl_mo.sum() / (100 * len(sub))
        sign = "+" if roi_mo >= 0 else "-"
        print(f"    {str(mo)}  n={len(sub):3d}  WR={(sub['bet_won']==1).mean():.0%}  ROI={sign}{abs(roi_mo):.1%}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-05",
                   help="YYYY-MM. Default 2024-05 (waiting for L30 features to warm up).")
    p.add_argument("--end", default="2025-12", help="YYYY-MM")
    a = p.parse_args()
    start = date.fromisoformat(a.start + "-01")
    end_m = date.fromisoformat(a.end + "-01")
    end = (end_m + relativedelta(months=1)) - timedelta(days=1)
    preds = run(start, end)
    _aggregate_metrics(preds)
    for th in (0.03, 0.05, 0.08):
        _backtest(preds, edge_threshold=th)


if __name__ == "__main__":
    main()
