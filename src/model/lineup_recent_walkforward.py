"""Walk-forward: does lineup_recent (L15 top-4 batter form) improve the model?

Compares:
  BASELINE  — current production features (no lineup_recent)
  +lineup_recent — adds the 21 new lineup_recent columns

Uses the production hyperparameters (leaves=10, reg=1.0) and team_id as
categorical. Reports overall + by confidence band, with focus on the
sweet band (5pp+ edge) since that's what produces +EV bets in production.

Run:
  python -m src.model.lineup_recent_walkforward
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score

PROCESSED = Path("data/processed")
MODELS = Path("data/models")


def _load_feats():
    with open(MODELS / "lgb_cls.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved.get("feature_names", [])


def _wf(df, feats, cat_features, label):
    df = df.sort_values("game_date").reset_index(drop=True)
    df["ym"] = df["game_date"].dt.to_period("M")
    months = sorted(df["ym"].unique())
    warm = pd.Period("2024-04", freq="M")
    rows = []
    for m in months:
        if m <= warm:
            continue
        tr_mask = df["ym"] < m
        te_mask = df["ym"] == m
        if tr_mask.sum() < 1000 or te_mask.sum() == 0:
            continue
        Xtr = df.loc[tr_mask, feats].copy()
        Xte = df.loc[te_mask, feats].copy()
        for c in cat_features:
            if c in Xtr.columns:
                Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").fillna(-1).astype("int32")
                Xte[c] = pd.to_numeric(Xte[c], errors="coerce").fillna(-1).astype("int32")
        ytr = df.loc[tr_mask, "home_win"].astype(int)
        yte = df.loc[te_mask, "home_win"].astype(int)
        if ytr.nunique() < 2:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03,
            num_leaves=10, min_child_samples=40,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1,
        )
        kw = {"categorical_feature": cat_features} if cat_features else {}
        model.fit(Xtr, ytr, **kw)
        p = model.predict_proba(Xte)[:, 1]
        for prob, y in zip(p, yte):
            rows.append({"p": float(prob), "y": int(y)})
    out = pd.DataFrame(rows)
    out["pred"] = (out["p"] >= 0.5).astype(int)
    acc = (out["pred"] == out["y"]).mean()
    auc = roc_auc_score(out["y"], out["p"])
    ll = log_loss(out["y"], out["p"].clip(1e-6, 1-1e-6))
    high = out[np.abs(out["p"] - 0.5) >= 0.10]
    sweet = out[(np.abs(out["p"] - 0.5) >= 0.05) & (np.abs(out["p"] - 0.5) < 0.10)]
    h_acc = (high["pred"] == high["y"]).mean() if len(high) else float("nan")
    s_acc = (sweet["pred"] == sweet["y"]).mean() if len(sweet) else float("nan")
    print(f"{label:>30s}: N={len(out):,}  acc={acc:.4f}  AUC={auc:.4f}  ll={ll:.4f}  "
          f"sweet(5-10pp,N={len(sweet)}):{s_acc:.4f}  high(10pp+,N={len(high)}):{h_acc:.4f}")
    return out, acc, auc, ll


def main():
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "home_win" not in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    saved = _load_feats()
    lr_feats = [c for c in df.columns if (
        c.startswith("lineup_top4_recent_") or c.startswith("lineup_hot_bats_count_")
        or c.startswith("lineup_cold_bats_count_") or c.startswith("lineup_recent_n_with_data")
    )]
    print(f"lineup_recent feats in train: {len(lr_feats)}")
    feats_base = [c for c in saved if c in df.columns]
    feats_base_no_lr = [c for c in feats_base if c not in lr_feats]
    feats_with_lr = feats_base_no_lr + lr_feats
    print(f"baseline:        {len(feats_base_no_lr)} feats")
    print(f"+ lineup_recent: {len(feats_with_lr)} feats  (added {len(feats_with_lr)-len(feats_base_no_lr)})")
    print()

    cat = ["home_team_id", "away_team_id"]
    cat_base = [c for c in cat if c in feats_base_no_lr]
    cat_lr   = [c for c in cat if c in feats_with_lr]
    print("=" * 100)
    out_b, acc_b, auc_b, ll_b = _wf(df, feats_base_no_lr, cat_base, "BASELINE")
    out_l, acc_l, auc_l, ll_l = _wf(df, feats_with_lr,    cat_lr,   "+ lineup_recent")

    print()
    print("=" * 100)
    print(f"Delta acc: {(acc_l - acc_b)*100:+.3f}pp")
    print(f"Delta AUC: {(auc_l - auc_b)*1000:+.2f} (x1000)")
    print(f"Delta logloss: {(ll_l - ll_b):+.5f}  (negative = better calibration)")
    print("=" * 100)


if __name__ == "__main__":
    main()
