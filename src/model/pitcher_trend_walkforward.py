"""Walk-forward: do pitcher trend + blowup features lift global accuracy?

Baseline = current production features (no trend/blowup)
+ trend = adds 5 trend/blowup feature groups × 3 (h/a/diff) = 15 cols

Run:
  python -m src.model.pitcher_trend_walkforward
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score

PROCESSED = Path("data/processed")
MODELS = Path("data/models")


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
    high = out[np.abs(out["p"] - 0.5) >= 0.09]
    h_acc = (high["pred"] == high["y"]).mean() if len(high) else float("nan")
    print(f"{label:>22s}: acc={acc:.4f}  AUC={auc:.4f}  ll={ll:.4f}  "
          f"high(9pp+,N={len(high)}):{h_acc:.4f}")
    return acc, auc, ll


def main():
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "home_win" not in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    trend_feats = [c for c in df.columns if any(
        s in c for s in ("trend_l5", "blowup_rate_l10", "disaster_count_l10")
    )]
    print(f"trend/blowup feats in train: {len(trend_feats)}")
    drop = {"game_pk","game_date","season","game_type","status",
            "home_team_abbrev","away_team_abbrev","venue_id","venue_name","roof_type",
            "probable_home_pitcher_id","probable_away_pitcher_id",
            "home_score","away_score","home_win","total_runs","market_p_home",
            "f5_home_score","f5_away_score","f5_home_won","f5_tied",
            "f5_total_runs","f5_run_diff"}
    all_feats = [c for c in df.columns if c not in drop and df[c].dtype.kind in "biufc"]
    base = [c for c in all_feats if c not in trend_feats]
    full = base + trend_feats
    print(f"baseline (no trend): {len(base)}  / +trend: {len(full)}")
    print()
    cat = ["home_team_id", "away_team_id"]
    print("=" * 90)
    a_b, u_b, l_b = _wf(df, base, [c for c in cat if c in base], "BASELINE")
    a_f, u_f, l_f = _wf(df, full, [c for c in cat if c in full], "+ pitcher_trend")
    print()
    print("=" * 90)
    print(f"Δ acc:    {(a_f-a_b)*100:+.3f}pp")
    print(f"Δ AUC:    {(u_f-u_b)*1000:+.2f} (x1000)")
    print(f"Δ logloss:{(l_f-l_b):+.5f}")


if __name__ == "__main__":
    main()
