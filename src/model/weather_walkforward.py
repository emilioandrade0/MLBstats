"""Walk-forward: does wind direction help WIN and/or TOTALS prediction?

Tests wind_out_mph + wind_speed + is_indoor against baseline on:
  1. Win classifier (acc, AUC)
  2. Totals regressor (MAE) — the primary hypothesis

Run:
  python -m src.model.weather_walkforward
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_absolute_error

PROCESSED = Path("data/processed")
WIND_FEATS = ["wind_out_mph", "wind_speed", "is_indoor"]


def _feature_pool(df):
    drop = {"game_pk","game_date","season","game_type","status",
            "home_team_abbrev","away_team_abbrev","venue_id","venue_name","roof_type",
            "probable_home_pitcher_id","probable_away_pitcher_id",
            "home_score","away_score","home_win","total_runs","market_p_home",
            "f5_home_score","f5_away_score","f5_home_won","f5_tied",
            "f5_total_runs","f5_run_diff"}
    return [c for c in df.columns if c not in drop and df[c].dtype.kind in "biufc"]


def _wf_cls(df, feats, cat):
    df = df.sort_values("game_date").reset_index(drop=True)
    df["ym"] = df["game_date"].dt.to_period("M")
    warm = pd.Period("2024-04", freq="M")
    rows = []
    for m in sorted(df["ym"].unique()):
        if m <= warm: continue
        tr = df["ym"] < m; te = df["ym"] == m
        if tr.sum() < 1000 or te.sum() == 0: continue
        Xtr = df.loc[tr, feats].copy(); Xte = df.loc[te, feats].copy()
        for c in cat:
            if c in Xtr.columns:
                Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").fillna(-1).astype("int32")
                Xte[c] = pd.to_numeric(Xte[c], errors="coerce").fillna(-1).astype("int32")
        ytr = df.loc[tr, "home_win"].astype(int); yte = df.loc[te, "home_win"].astype(int)
        if ytr.nunique() < 2: continue
        m_ = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=10,
            min_child_samples=40, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1)
        m_.fit(Xtr, ytr, categorical_feature=cat if cat else "auto")
        p = m_.predict_proba(Xte)[:, 1]
        for prob, y in zip(p, yte):
            rows.append({"p": float(prob), "y": int(y)})
    o = pd.DataFrame(rows); o["pred"] = (o["p"] >= 0.5).astype(int)
    return (o["pred"]==o["y"]).mean(), roc_auc_score(o["y"], o["p"]), log_loss(o["y"], o["p"].clip(1e-6,1-1e-6))


def _wf_reg(df, feats):
    df = df.sort_values("game_date").reset_index(drop=True)
    df["ym"] = df["game_date"].dt.to_period("M")
    warm = pd.Period("2024-04", freq="M")
    preds, actuals = [], []
    for m in sorted(df["ym"].unique()):
        if m <= warm: continue
        tr = df["ym"] < m; te = df["ym"] == m
        if tr.sum() < 1000 or te.sum() == 0: continue
        Xtr = df.loc[tr, feats]; Xte = df.loc[te, feats]
        ytr = df.loc[tr, "total_runs"]; yte = df.loc[te, "total_runs"]
        m_ = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=10,
            min_child_samples=40, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1)
        m_.fit(Xtr, ytr)
        preds.extend(m_.predict(Xte)); actuals.extend(yte)
    return mean_absolute_error(actuals, preds)


def main():
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "home_win" not in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.dropna(subset=["home_score", "away_score", "total_runs"]).copy()

    pool = _feature_pool(df)
    base = [c for c in pool if c not in WIND_FEATS]
    full = base + [c for c in WIND_FEATS if c in df.columns]
    cat = ["home_team_id", "away_team_id"]
    print(f"baseline: {len(base)}  +wind: {len(full)}")
    print()
    print("=== WIN classifier ===")
    ab, ub, lb = _wf_cls(df, base, [c for c in cat if c in base])
    af, uf, lf = _wf_cls(df, full, [c for c in cat if c in full])
    print(f"  baseline: acc={ab:.4f}  AUC={ub:.4f}  ll={lb:.4f}")
    print(f"  +wind:    acc={af:.4f}  AUC={uf:.4f}  ll={lf:.4f}")
    print(f"  Δ acc={ (af-ab)*100:+.3f}pp  Δ AUC={(uf-ub)*1000:+.2f}")
    print()
    print("=== TOTALS regressor (lower MAE = better) ===")
    mb = _wf_reg(df, base)
    mf = _wf_reg(df, full)
    print(f"  baseline MAE: {mb:.4f}")
    print(f"  +wind MAE:    {mf:.4f}")
    print(f"  Δ MAE: {(mf-mb):+.4f}  ({'better' if mf < mb else 'worse'})")


if __name__ == "__main__":
    main()
