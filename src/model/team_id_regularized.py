"""Walk-forward test: team_id with stronger regularization to prevent
high-confidence overshooting.

Tries 3 configurations:
  BASELINE          — no team_id
  TEAM_ID + reg=0.1 — default reg (the bad one we saw)
  TEAM_ID + reg=0.5 — stronger L1+L2
  TEAM_ID + reg=1.0 — even stronger

If a stronger-reg config recovers the high-conf band without hurting overall,
we have a usable version.

Run:
  python -m src.model.team_id_regularized
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


def _walkforward(df, feats, cat_features=None, reg=0.1, num_leaves=31, label=""):
    df = df.sort_values("game_date").reset_index(drop=True)
    df["ym"] = df["game_date"].dt.to_period("M")
    months = sorted(df["ym"].unique())
    warm_cutoff = pd.Period("2024-04", freq="M")
    rows = []
    for m in months:
        if m <= warm_cutoff:
            continue
        train_mask = df["ym"] < m
        test_mask  = df["ym"] == m
        if train_mask.sum() < 1000 or test_mask.sum() == 0:
            continue
        Xtr = df.loc[train_mask, feats].copy()
        ytr = df.loc[train_mask, "home_win"].astype(int)
        Xte = df.loc[test_mask,  feats].copy()
        yte = df.loc[test_mask,  "home_win"].astype(int)
        if ytr.nunique() < 2:
            continue
        if cat_features:
            for c in cat_features:
                Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").fillna(-1).astype("int32")
                Xte[c] = pd.to_numeric(Xte[c], errors="coerce").fillna(-1).astype("int32")
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=num_leaves,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=reg, reg_lambda=reg, random_state=42, verbose=-1,
        )
        fit_kwargs = {}
        if cat_features:
            fit_kwargs["categorical_feature"] = cat_features
        model.fit(Xtr, ytr, **fit_kwargs)
        p = model.predict_proba(Xte)[:, 1]
        for prob, y in zip(p, yte):
            rows.append({"p": float(prob), "y": int(y)})
    out = pd.DataFrame(rows)
    out["pred"] = (out["p"] >= 0.5).astype(int)
    acc = (out["pred"] == out["y"]).mean()
    auc = roc_auc_score(out["y"], out["p"])
    ll  = log_loss(out["y"], out["p"].clip(1e-6, 1-1e-6))
    high = out[np.abs(out["p"] - 0.5) >= 0.20]
    acc_high = (high["pred"] == high["y"]).mean() if len(high) else float("nan")
    sweet = out[(np.abs(out["p"] - 0.5) >= 0.10) & (np.abs(out["p"] - 0.5) < 0.20)]
    acc_sweet = (sweet["pred"] == sweet["y"]).mean() if len(sweet) else float("nan")
    print(f"{label:>32s}: acc={acc:.4f}  AUC={auc:.4f}  ll={ll:.4f}  "
          f"sweet(10-20pp,N={len(sweet)}):{acc_sweet:.4f}  high(20pp+,N={len(high)}):{acc_high:.4f}")
    return out, acc, auc


def main():
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "home_win" not in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    feats_base = [c for c in _load_feats() if c in df.columns]
    feats_team = feats_base + ["home_team_id", "away_team_id"]

    print("=" * 100)
    _walkforward(df, feats_base, label="BASELINE (no team_id, reg=0.1)")

    print("=" * 100)
    print("Testing team_id with progressively stronger regularization:")
    print("=" * 100)
    for reg, nl in [(0.1, 31), (0.5, 31), (1.0, 31), (0.5, 16), (1.0, 16), (2.0, 16)]:
        _walkforward(df, feats_team,
                     cat_features=["home_team_id", "away_team_id"],
                     reg=reg, num_leaves=nl,
                     label=f"+team_id reg={reg}  leaves={nl}")


if __name__ == "__main__":
    main()
