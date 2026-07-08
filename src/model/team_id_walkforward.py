"""Walk-forward test: does adding home_team_id + away_team_id as categorical
features improve model accuracy?

Compares:
  BASELINE: 125 features from current production model
  +TEAM_ID: same 125 + home_team_id + away_team_id (LightGBM categorical)

Both trained monthly walk-forward, full warmup, same hyperparameters.

Run:
  python -m src.model.team_id_walkforward
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


def _walkforward(df, feats, cat_features=None, label=""):
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
        # Cast categorical cols to int (LightGBM needs int or category)
        if cat_features:
            for c in cat_features:
                Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").fillna(-1).astype("int32")
                Xte[c] = pd.to_numeric(Xte[c], errors="coerce").fillna(-1).astype("int32")
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
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
    print(f"{label:>22s}: N={len(out):,}  acc={acc:.4f}  AUC={auc:.4f}  log_loss={ll:.4f}")
    return out, acc, auc


def main():
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "home_win" not in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    feats_baseline = [c for c in _load_feats() if c in df.columns]
    print(f"baseline: {len(feats_baseline)} features")

    if "home_team_id" not in df.columns or "away_team_id" not in df.columns:
        print("ERROR: home_team_id / away_team_id not in train.parquet")
        return
    feats_with_team = feats_baseline + ["home_team_id", "away_team_id"]
    print(f"+ team_id: {len(feats_with_team)} features  (added 2 categorical: home_team_id, away_team_id)")
    print()

    print("=== Walk-forward comparison ===")
    out_base, acc_base, auc_base = _walkforward(df, feats_baseline, label="BASELINE")
    out_team, acc_team, auc_team = _walkforward(
        df, feats_with_team,
        cat_features=["home_team_id", "away_team_id"],
        label="+ team_id (cat)"
    )

    print()
    print("=" * 60)
    print(f"Delta acc: {(acc_team - acc_base)*100:+.3f}pp")
    print(f"Delta AUC: {(auc_team - auc_base)*1000:+.2f} (x1000)")
    print("=" * 60)

    # Per-team accuracy breakdown
    df_test = df[df["ym"] > pd.Period("2024-04", freq="M")].copy().reset_index(drop=True) if False else None
    # Compare by confidence band
    print()
    print("By confidence band (|p - 0.5|):")
    print(f"{'band':>10s} {'N':>7s} {'acc_base':>9s} {'acc_team':>9s} {'delta':>8s}")
    for lo, hi, label in [
        (0.00, 0.03, "0-3"),
        (0.03, 0.06, "3-6"),
        (0.06, 0.10, "6-10"),
        (0.10, 0.20, "10-20"),
        (0.20, 1.00, "20+"),
    ]:
        sb = out_base[(np.abs(out_base["p"] - 0.5) >= lo) & (np.abs(out_base["p"] - 0.5) < hi)]
        st = out_team[(np.abs(out_team["p"] - 0.5) >= lo) & (np.abs(out_team["p"] - 0.5) < hi)]
        a_b = (sb["pred"] == sb["y"]).mean() if len(sb) else float("nan")
        a_t = (st["pred"] == st["y"]).mean() if len(st) else float("nan")
        print(f"{label:>10s} {len(sb):>7,} {a_b:>9.4f} {a_t:>9.4f} {(a_t-a_b)*100:>+7.2f}pp")


if __name__ == "__main__":
    main()
