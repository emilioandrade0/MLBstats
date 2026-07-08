"""NRFI/YRFI walk-forward — can we predict first-inning runs above 50%?

Combines existing model features (team/pitcher/lineup form) with NRFI-specific
first-inning features, trains a classifier on the yrfi target, and reports
accuracy overall AND at confidence-limited coverage levels.

Run:
  python -m src.model.nrfi_walkforward
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

PROCESSED = Path("data/processed")


def main():
    tr = pd.read_parquet(PROCESSED / "train.parquet")
    tr["game_date"] = pd.to_datetime(tr["game_date"])
    tgt = pd.read_parquet(PROCESSED / "nrfi_targets.parquet")
    nf = pd.read_parquet(PROCESSED / "features_nrfi.parquet")

    df = tr.merge(tgt, on="game_pk", how="inner").merge(nf, on="game_pk", how="left")
    df = df.dropna(subset=["yrfi"]).copy()
    df = df.sort_values("game_date").reset_index(drop=True)

    nrfi_feats = [c for c in nf.columns if c != "game_pk"]
    # Use a focused feature set: NRFI-specific + the most relevant existing ones
    base_relevant = [c for c in df.columns if any(s in c for s in (
        "starter_k_pct_l15", "starter_bb_pct_l15", "starter_xwoba_l15",
        "off_xwoba_l30", "off_k_pct_l30", "lineup_top4", "off_bb_pct_l30",
        "home_elo_pre", "away_elo_pre", "park_runs_factor", "weather_temp_f",
    ))]
    feats = list(dict.fromkeys(nrfi_feats + base_relevant))
    feats = [c for c in feats if c in df.columns and df[c].dtype.kind in "biufc"]
    print(f"NRFI features: {len(nrfi_feats)}  total used: {len(feats)}")
    print(f"games: {len(df):,}  YRFI base rate: {df['yrfi'].mean():.4f}")
    print()

    df["ym"] = df["game_date"].dt.to_period("M")
    warm = pd.Period("2024-04", freq="M")
    rows = []
    for m in sorted(df["ym"].unique()):
        if m <= warm: continue
        trm = df["ym"] < m; tem = df["ym"] == m
        if trm.sum() < 800 or tem.sum() == 0: continue
        Xtr = df.loc[trm, feats]; Xte = df.loc[tem, feats]
        ytr = df.loc[trm, "yrfi"].astype(int); yte = df.loc[tem, "yrfi"].astype(int)
        if ytr.nunique() < 2: continue
        mdl = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.03, num_leaves=12,
            min_child_samples=40, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1)
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)[:, 1]
        for prob, y in zip(p, yte):
            rows.append({"p": float(prob), "y": int(y)})

    o = pd.DataFrame(rows)
    o["pred"] = (o["p"] >= 0.5).astype(int)
    o["correct"] = (o["pred"] == o["y"]).astype(int)
    o["conf"] = np.abs(o["p"] - 0.5)
    acc = o["correct"].mean()
    auc = roc_auc_score(o["y"], o["p"])
    print(f"=== NRFI/YRFI walk-forward ({len(o):,} games) ===")
    print(f"  overall: acc={acc:.4f}  AUC={auc:.4f}  ll={log_loss(o['y'], o['p'].clip(1e-6,1-1e-6)):.4f}")
    print(f"  (vs 50% coin flip, base rate {o['y'].mean():.4f})")
    print()
    o = o.sort_values("conf", ascending=False).reset_index(drop=True)
    N = len(o)
    print("Accuracy by coverage (most confident X%):")
    print(f"{'coverage':>10s} {'n':>7s} {'accuracy':>9s} {'min_conf':>9s}")
    for cov in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]:
        k = max(1, int(N * cov))
        sub = o.head(k)
        print(f"{cov*100:>9.1f}% {k:>7d} {sub['correct'].mean():>9.4f} {sub['conf'].min()*100:>8.1f}pp")


if __name__ == "__main__":
    main()
