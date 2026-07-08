"""Grid search around the current best config to find better hyperparameters.

Current best (already in production):
  num_leaves=16, reg_alpha=1.0, reg_lambda=1.0, min_child_samples=20
  with home_team_id + away_team_id as categorical features.

We sweep:
  - num_leaves: 10, 14, 16, 20, 24
  - reg: 0.5, 1.0, 1.5, 2.0, 3.0
  - min_child_samples: 20, 40, 60
  - + adding venue_id as a 3rd categorical

Output: ranked table of configs by accuracy + AUC + log_loss.

Run:
  python -m src.model.hyperparam_sweep
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


def _wf(df, feats, cat_features, num_leaves, reg, min_child, label):
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
            num_leaves=num_leaves, min_child_samples=min_child,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=reg, reg_lambda=reg,
            random_state=42, verbose=-1,
        )
        kw = {}
        if cat_features:
            kw["categorical_feature"] = cat_features
        model.fit(Xtr, ytr, **kw)
        p = model.predict_proba(Xte)[:, 1]
        for prob, y in zip(p, yte):
            rows.append({"p": float(prob), "y": int(y)})
    out = pd.DataFrame(rows)
    out["pred"] = (out["p"] >= 0.5).astype(int)
    acc = (out["pred"] == out["y"]).mean()
    auc = roc_auc_score(out["y"], out["p"])
    ll = log_loss(out["y"], out["p"].clip(1e-6, 1-1e-6))
    high = out[np.abs(out["p"] - 0.5) >= 0.20]
    acc_high = (high["pred"] == high["y"]).mean() if len(high) else float("nan")
    sweet = out[(np.abs(out["p"] - 0.5) >= 0.10) & (np.abs(out["p"] - 0.5) < 0.20)]
    acc_sweet = (sweet["pred"] == sweet["y"]).mean() if len(sweet) else float("nan")
    return {
        "label": label, "acc": acc, "auc": auc, "log_loss": ll,
        "high_n": len(high), "high_acc": acc_high,
        "sweet_n": len(sweet), "sweet_acc": acc_sweet,
    }


def main():
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "home_win" not in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    feats_base = [c for c in _load_feats() if c in df.columns]
    # The current production model already has team_id baked in. Strip it here
    # so the sweep starts from the SAME baseline as the original team_id test.
    feats_no_team = [c for c in feats_base if c not in ("home_team_id", "away_team_id")]
    feats_with_team = feats_no_team + ["home_team_id", "away_team_id"]
    print(f"baseline feats: {len(feats_no_team)}    +team: {len(feats_with_team)}")
    has_venue = "venue_id" in df.columns
    feats_with_team_venue = feats_with_team + (["venue_id"] if has_venue else [])
    print(f"+venue: {len(feats_with_team_venue)}  (has venue_id: {has_venue})")
    print()

    configs = [
        # (feats, cat, leaves, reg, min_child, label)
        (feats_no_team,         [],                                                  16, 1.0, 20, "NO team_id (reference)"),
        # Vary num_leaves around 16
        (feats_with_team,       ["home_team_id","away_team_id"],                     10, 1.0, 20, "L=10 reg=1.0"),
        (feats_with_team,       ["home_team_id","away_team_id"],                     14, 1.0, 20, "L=14 reg=1.0"),
        (feats_with_team,       ["home_team_id","away_team_id"],                     16, 1.0, 20, "L=16 reg=1.0  ← prod"),
        (feats_with_team,       ["home_team_id","away_team_id"],                     20, 1.0, 20, "L=20 reg=1.0"),
        (feats_with_team,       ["home_team_id","away_team_id"],                     24, 1.0, 20, "L=24 reg=1.0"),
        # Vary regularization at L=16
        (feats_with_team,       ["home_team_id","away_team_id"],                     16, 0.5, 20, "L=16 reg=0.5"),
        (feats_with_team,       ["home_team_id","away_team_id"],                     16, 1.5, 20, "L=16 reg=1.5"),
        (feats_with_team,       ["home_team_id","away_team_id"],                     16, 2.0, 20, "L=16 reg=2.0"),
        (feats_with_team,       ["home_team_id","away_team_id"],                     16, 3.0, 20, "L=16 reg=3.0"),
        # Vary min_child_samples
        (feats_with_team,       ["home_team_id","away_team_id"],                     16, 1.0, 40, "L=16 reg=1.0 minC=40"),
        (feats_with_team,       ["home_team_id","away_team_id"],                     16, 1.0, 60, "L=16 reg=1.0 minC=60"),
        # Best combinations
        (feats_with_team,       ["home_team_id","away_team_id"],                     14, 1.5, 40, "L=14 reg=1.5 minC=40"),
        (feats_with_team,       ["home_team_id","away_team_id"],                     20, 2.0, 40, "L=20 reg=2.0 minC=40"),
    ]
    if has_venue:
        configs += [
            (feats_with_team_venue, ["home_team_id","away_team_id","venue_id"],      16, 1.0, 20, "+venue L=16 reg=1.0"),
            (feats_with_team_venue, ["home_team_id","away_team_id","venue_id"],      20, 1.5, 40, "+venue L=20 reg=1.5 minC=40"),
        ]

    results = []
    for feats, cat, L, reg, mc, label in configs:
        print(f"running: {label} ...")
        r = _wf(df, feats, cat, L, reg, mc, label)
        results.append(r)
        print(f"  acc={r['acc']:.4f}  auc={r['auc']:.4f}  ll={r['log_loss']:.4f}  "
              f"sweet({r['sweet_n']}):{r['sweet_acc']:.4f}  high({r['high_n']}):{r['high_acc']:.4f}")

    print()
    print("=" * 100)
    print("Ranked by overall accuracy:")
    print(f"{'rank':>4s} {'config':<32s} {'acc':>8s} {'AUC':>7s} {'logl':>7s} {'sweet':>8s} {'high':>8s}")
    for i, r in enumerate(sorted(results, key=lambda x: -x["acc"]), 1):
        print(f"{i:>4d} {r['label']:<32s} {r['acc']:>8.4f} {r['auc']:>7.4f} {r['log_loss']:>7.4f} "
              f"{r['sweet_acc']:>8.4f} {r['high_acc']:>8.4f}")

    print()
    print("Ranked by HIGH-CONFIDENCE accuracy (where +EV picks live):")
    print(f"{'rank':>4s} {'config':<32s} {'acc':>8s} {'AUC':>7s} {'logl':>7s} {'sweet':>8s} {'high':>8s}")
    for i, r in enumerate(sorted(results, key=lambda x: -x["high_acc"]), 1):
        print(f"{i:>4d} {r['label']:<32s} {r['acc']:>8.4f} {r['auc']:>7.4f} {r['log_loss']:>7.4f} "
              f"{r['sweet_acc']:>8.4f} {r['high_acc']:>8.4f}")


if __name__ == "__main__":
    main()
