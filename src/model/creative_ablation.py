"""Ablation walk-forward for creative features.

Tests baseline vs each creative group to find which (if any) adds signal:
  - streak (momentum)
  - fatigue (games_last_7d, day_after_night, rest_advantage)
  - line movement (line_move_home, line_move_total)
  - ALL together

Run:
  python -m src.model.creative_ablation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

PROCESSED = Path("data/processed")

GROUPS = {
    "streak":   ["streak_signed_h", "streak_signed_a", "streak_signed_diff"],
    "fatigue":  ["games_last_7d_h", "games_last_7d_a", "games_last_7d_diff",
                 "day_after_night_h", "day_after_night_a", "day_after_night_diff",
                 "rest_advantage_h", "rest_advantage_a", "rest_advantage_diff"],
    "linemove": ["line_move_home", "line_move_total"],
}
ALL_CREATIVE = [c for g in GROUPS.values() for c in g]


def _wf(df, feats, cat):
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
        m_.fit(Xtr, ytr, categorical_feature=[c for c in cat if c in feats] or "auto")
        p = m_.predict_proba(Xte)[:, 1]
        for prob, y in zip(p, yte):
            rows.append({"p": float(prob), "y": int(y)})
    o = pd.DataFrame(rows); o["pred"] = (o["p"] >= 0.5).astype(int)
    return (o["pred"]==o["y"]).mean(), roc_auc_score(o["y"], o["p"]), log_loss(o["y"], o["p"].clip(1e-6,1-1e-6))


def main():
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "home_win" not in df.columns:
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    drop = {"game_pk","game_date","season","game_type","status",
            "home_team_abbrev","away_team_abbrev","venue_id","venue_name","roof_type",
            "probable_home_pitcher_id","probable_away_pitcher_id",
            "home_score","away_score","home_win","total_runs","market_p_home",
            "f5_home_score","f5_away_score","f5_home_won","f5_tied",
            "f5_total_runs","f5_run_diff"}
    pool = [c for c in df.columns if c not in drop and df[c].dtype.kind in "biufc"]
    base = [c for c in pool if c not in ALL_CREATIVE]
    cat = ["home_team_id", "away_team_id"]

    print(f"baseline features: {len(base)}")
    print("=" * 80)
    ab, ub, lb = _wf(df, base, cat)
    print(f"{'BASELINE':>20s}: acc={ab:.4f}  AUC={ub:.4f}  ll={lb:.4f}")
    print("-" * 80)
    for name, cols in GROUPS.items():
        cols = [c for c in cols if c in df.columns]
        a, u, l = _wf(df, base + cols, cat)
        print(f"{'+ '+name:>20s}: acc={a:.4f}  AUC={u:.4f}  ll={l:.4f}  "
              f"(Δacc {(a-ab)*100:+.2f}pp  ΔAUC {(u-ub)*1000:+.1f})")
    print("-" * 80)
    a, u, l = _wf(df, base + [c for c in ALL_CREATIVE if c in df.columns], cat)
    print(f"{'+ ALL':>20s}: acc={a:.4f}  AUC={u:.4f}  ll={l:.4f}  "
          f"(Δacc {(a-ab)*100:+.2f}pp  ΔAUC {(u-ub)*1000:+.1f})")


if __name__ == "__main__":
    main()
