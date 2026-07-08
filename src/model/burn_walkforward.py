"""Walk-forward test: model accuracy WITH vs WITHOUT burn features.

Loads the current train.parquet (which includes burn features), runs walk-forward
twice — once with burn cols included, once with them dropped — and reports the
delta.

Run:
  python -m src.model.burn_walkforward
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


def _load_feats():
    with open(MODELS / "lgb_cls.pkl", "rb") as f:
        saved = pickle.load(f)
    return saved.get("feature_names", [])


def _walkforward(df: pd.DataFrame, feats: list[str], label: str):
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
        Xtr = df.loc[train_mask, feats]
        ytr = df.loc[train_mask, "home_win"].astype(int)
        Xte = df.loc[test_mask,  feats]
        yte = df.loc[test_mask,  "home_win"].astype(int)
        if ytr.nunique() < 2:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
        )
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        for idx, prob, y in zip(df.loc[test_mask].index, p, yte):
            rows.append({"idx": idx, "p": float(prob), "y": int(y)})
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

    saved_feats = _load_feats()
    # Saved feats are baseline. Burn cols only exist in current train.parquet.
    feats_no_burn = [c for c in saved_feats if c in df.columns]
    burn_feats    = [c for c in df.columns if c.startswith("burn_")]
    feats_all     = feats_no_burn + [c for c in burn_feats if c not in feats_no_burn]
    print(f"total features: {len(feats_all)} (burn: {len(burn_feats)}, baseline: {len(feats_no_burn)})")
    print()

    print("=== Walk-forward comparison ===")
    out_no, acc_no, auc_no = _walkforward(df, feats_no_burn, "BASELINE (no burn)")
    out_yes, acc_yes, auc_yes = _walkforward(df, feats_all, "WITH burn features")

    print()
    print("=" * 60)
    print(f"Delta acc: {(acc_yes - acc_no)*100:+.3f}pp")
    print(f"Delta AUC: {(auc_yes - auc_no)*1000:+.2f} (x1000)")
    print("=" * 60)

    # Edge-band breakdown using market-blended prob (approximation: just model p)
    # Show accuracy by confidence
    print()
    print("By confidence band (|p - 0.5|):")
    print(f"{'band':>10s} {'N':>7s} {'acc_no':>8s} {'acc_yes':>9s} {'delta':>8s}")
    for lo, hi, label in [
        (0.00, 0.03, "0-3"),
        (0.03, 0.06, "3-6"),
        (0.06, 0.10, "6-10"),
        (0.10, 0.20, "10-20"),
        (0.20, 1.00, "20+"),
    ]:
        sub_no  = out_no [(np.abs(out_no ["p"] - 0.5) >= lo) & (np.abs(out_no ["p"] - 0.5) < hi)]
        sub_yes = out_yes[(np.abs(out_yes["p"] - 0.5) >= lo) & (np.abs(out_yes["p"] - 0.5) < hi)]
        a_no  = (sub_no ["pred"] == sub_no ["y"]).mean() if len(sub_no)  else float("nan")
        a_yes = (sub_yes["pred"] == sub_yes["y"]).mean() if len(sub_yes) else float("nan")
        print(f"{label:>10s} {len(sub_no):>7,} {a_no:>8.4f} {a_yes:>9.4f} {(a_yes-a_no)*100:>+7.2f}pp")


if __name__ == "__main__":
    main()
