"""Walk-forward: does ANY market blend beat model-only?

The base model already has market_logit_p_home as a feature, so blending its
output with raw market again may be redundant double-counting. This script
tests, across all monthly walk-forward folds:
  - model only (raw)
  - 2-bucket blend (current production: agree=0.50, disagree=0.80)
  - flat blends at w = 0.2, 0.35, 0.5

Run:
  python -m src.model.blend_walkforward
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

PROCESSED = Path("data/processed")
MODELS = Path("data/models")


def main():
    with open(MODELS / "lgb_cls.pkl", "rb") as f:
        base = pickle.load(f)
    feats = base["feature_names"]

    df = pd.read_parquet(PROCESSED / "train.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "home_win" not in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    feats = [f for f in feats if f in df.columns]

    df = df.sort_values("game_date").reset_index(drop=True)
    df["ym"] = df["game_date"].dt.to_period("M")
    months = sorted(df["ym"].unique())
    warm = pd.Period("2024-04", freq="M")

    rows = []
    for m in months:
        if m <= warm:
            continue
        tr = df["ym"] < m
        te = df["ym"] == m
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        Xtr = df.loc[tr, feats].copy()
        Xte = df.loc[te, feats].copy()
        for c in ["home_team_id", "away_team_id"]:
            if c in Xtr.columns:
                Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").fillna(-1).astype("int32")
                Xte[c] = pd.to_numeric(Xte[c], errors="coerce").fillna(-1).astype("int32")
        ytr = df.loc[tr, "home_win"].astype(int)
        yte = df.loc[te, "home_win"].astype(int)
        if ytr.nunique() < 2:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=10,
            min_child_samples=40, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1,
        )
        model.fit(Xtr, ytr, categorical_feature=[c for c in ["home_team_id","away_team_id"] if c in feats])
        p_model = model.predict_proba(Xte)[:, 1]
        mkt = df.loc[te, "market_p_home"].values
        for pm, mp, y in zip(p_model, mkt, yte):
            rows.append({"p_model": pm, "p_market": mp, "y": int(y)})

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["p_market"])
    print(f"Walk-forward games: {len(out):,}")
    print()

    def _ev(name, p):
        p = np.clip(p, 1e-6, 1-1e-6)
        acc = accuracy_score(out["y"], (p >= 0.5).astype(int))
        ll  = log_loss(out["y"], p)
        auc = roc_auc_score(out["y"], p)
        print(f"  {name:32s}  acc={acc:.4f}  AUC={auc:.4f}  ll={ll:.4f}")
        return acc

    pm = out["p_model"].values
    mp = out["p_market"].values
    print("=== Walk-forward comparison (all folds) ===")
    _ev("model only", pm)
    _ev("market only", mp)
    agree = (pm > 0.5) == (mp > 0.5)
    w2 = np.where(agree, 0.50, 0.80)
    _ev("2-bucket blend (current prod)", w2 * mp + (1 - w2) * pm)
    print()
    print("=== Flat blend sweep ===")
    best_acc, best_w = 0, None
    for w in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]:
        a = _ev(f"flat blend w={w:.2f}", w * mp + (1 - w) * pm)
        if a > best_acc:
            best_acc, best_w = a, w
    print()
    print("=== 2-bucket sweep (agree weight, disagree fixed higher) ===")
    for wa in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
        wd = min(wa + 0.15, 0.95)
        w2s = np.where(agree, wa, wd)
        _ev(f"2-bucket agree={wa:.2f} disagree={wd:.2f}", w2s * mp + (1 - w2s) * pm)
    print()
    print(f"Best flat weight: w={best_w:.2f}  acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
