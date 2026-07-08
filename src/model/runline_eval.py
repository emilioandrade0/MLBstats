"""Run-line evaluation — baseline (lean from ML pick) vs dedicated RL classifier.

Baseline: if model picks home -> assume home covers -1.5; if away -> assume away covers +1.5.
Model:    train binary classifier on `home_won_by_2_plus` with the full feature set.

Both evaluated via walk-forward monthly folds (same scheme as moneyline).

Run:
  python -m src.model.runline_eval
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
TRAIN_PATH = PROCESSED / "train.parquet"


def _load():
    df = pd.read_parquet(TRAIN_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Need final scores
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_margin"] = df["home_score"] - df["away_score"]
    # Standard MLB run line is ±1.5 — exactly one side covers (no push)
    df["home_cover_rl"] = (df["home_margin"] >= 2).astype(int)       # HOME -1.5
    df["away_cover_rl"] = (df["home_margin"] <= 1).astype(int)       # AWAY +1.5 (complement)
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Reuse the same feature set the production ML classifier uses."""
    cls_path = MODELS / "lgb_cls.pkl"
    if cls_path.exists():
        with open(cls_path, "rb") as f:
            saved = pickle.load(f)
        feats = saved.get("feature_names")
        if feats:
            return [c for c in feats if c in df.columns]
    # fallback: numeric non-target columns
    drop = {"game_pk", "game_date", "season", "home_score", "away_score",
            "home_margin", "home_cover_rl", "away_cover_rl",
            "home_win", "home_team_abbrev", "away_team_abbrev"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]


def _walk_forward(df: pd.DataFrame, feats: list[str], target: str) -> pd.DataFrame:
    """Monthly walk-forward: train on everything strictly before month M, predict M."""
    df = df.sort_values("game_date").reset_index(drop=True)
    df["ym"] = df["game_date"].dt.to_period("M")
    months = sorted(df["ym"].unique())
    out_rows = []
    # Need at least one full season as warm-up
    warm_cutoff = pd.Period("2024-04", freq="M")
    for m in months:
        if m <= warm_cutoff:
            continue
        train_mask = df["ym"] < m
        test_mask  = df["ym"] == m
        if train_mask.sum() < 1000 or test_mask.sum() == 0:
            continue
        Xtr = df.loc[train_mask, feats]
        ytr = df.loc[train_mask, target].astype(int)
        Xte = df.loc[test_mask,  feats]
        yte = df.loc[test_mask,  target].astype(int)
        if ytr.nunique() < 2:
            continue
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
        )
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        for idx, prob, pr, y in zip(df.loc[test_mask].index, p, pred, yte):
            out_rows.append({
                "idx": idx, "month": str(m), "y": int(y),
                "p": float(prob), "pred": int(pr),
            })
    return pd.DataFrame(out_rows)


def main():
    df = _load()
    print(f"loaded {len(df):,} games with final scores  ({df['game_date'].min().date()} -> {df['game_date'].max().date()})")
    base_rate_home = df["home_cover_rl"].mean()
    base_rate_away = df["away_cover_rl"].mean()
    print(f"base rates — home covers -1.5: {base_rate_home:.3f}   away covers +1.5: {1 - base_rate_away:.3f}")
    print()

    feats = _feature_cols(df)
    print(f"using {len(feats)} features")
    print()

    # ── Baseline: derive RL pick from ML model walk-forward predictions ──────
    # We need the ML model's p_home for each game. Load saved ensemble walk-forward
    # results if available, else regenerate quickly with a single LightGBM.
    print("=== BASELINE: derive RL pick from ML p_home ===")
    ml_results = _walk_forward(df, feats, target="home_win") if "home_win" not in df.columns or True else None
    # Re-derive home_win in case it's missing
    if "home_win" not in df.columns:
        df["home_win"] = (df["home_margin"] > 0).astype(int)
    ml_wf = _walk_forward(df, feats, target="home_win")
    # Build baseline RL prediction: pick HOME RL if model picks HOME (p_home >= 0.5)
    base = ml_wf.copy()
    base = base.merge(df[["home_cover_rl", "away_cover_rl"]], left_on="idx", right_index=True)
    # Predicted side: HOME if p_home >= 0.5 else AWAY
    base["picked_home"] = base["pred"] == 1
    # Did the lean win? Home pick wins if home covered -1.5; Away pick wins if away covered +1.5
    base["rl_won"] = np.where(base["picked_home"],
                              base["home_cover_rl"] == 1,
                              base["away_cover_rl"] == 1)
    n = len(base)
    acc_base = base["rl_won"].mean()
    print(f"  N = {n:,}   RL accuracy (lean) = {acc_base:.4f}   ({base['rl_won'].sum():,} hits / {n:,})")
    print(f"  picked HOME -1.5: {base['picked_home'].sum():,}   AWAY +1.5: {(~base['picked_home']).sum():,}")

    # By confidence buckets
    base["conf"] = np.abs(base["p"] - 0.5)
    for lo, hi, label in [(0, .03, '0-3pp'), (.03, .06, '3-6pp'),
                          (.06, .10, '6-10pp'), (.10, 1.0, '10pp+')]:
        sub = base[(base["conf"] >= lo) & (base["conf"] < hi)]
        if len(sub):
            print(f"    conf {label:>7s}: N={len(sub):,}  acc={sub['rl_won'].mean():.4f}")
    print()

    # ── Dedicated RL model: predict home_cover_rl directly ────────────────────
    print("=== DEDICATED MODEL: predict home_cover_rl directly ===")
    rl_wf = _walk_forward(df, feats, target="home_cover_rl")
    rl_wf = rl_wf.merge(df[["home_cover_rl", "away_cover_rl"]], left_on="idx", right_index=True)
    # Pick the side with higher predicted probability above 0.5
    # Strategy A: always pick HOME if p >= 0.5, else AWAY
    rl_wf["picked_home"] = rl_wf["pred"] == 1
    rl_wf["rl_won"] = np.where(rl_wf["picked_home"],
                                rl_wf["home_cover_rl"] == 1,
                                rl_wf["away_cover_rl"] == 1)
    n2 = len(rl_wf)
    acc_model = rl_wf["rl_won"].mean()
    print(f"  N = {n2:,}   RL accuracy (model) = {acc_model:.4f}")
    print(f"  picked HOME -1.5: {rl_wf['picked_home'].sum():,}   AWAY +1.5: {(~rl_wf['picked_home']).sum():,}")
    print(f"  log_loss = {log_loss(rl_wf['y'], rl_wf['p'].clip(1e-6, 1-1e-6)):.4f}   AUC = {roc_auc_score(rl_wf['y'], rl_wf['p']):.4f}")

    rl_wf["conf"] = np.abs(rl_wf["p"] - 0.5)
    for lo, hi, label in [(0, .03, '0-3pp'), (.03, .06, '3-6pp'),
                          (.06, .10, '6-10pp'), (.10, 1.0, '10pp+')]:
        sub = rl_wf[(rl_wf["conf"] >= lo) & (rl_wf["conf"] < hi)]
        if len(sub):
            print(f"    conf {label:>7s}: N={len(sub):,}  acc={sub['rl_won'].mean():.4f}")
    print()

    # ── Confidence-thresholded picks (skip uncertain games) ───────────────────
    print("=== CONFIDENCE-THRESHOLDED PICKS ===")
    print("Bet only when model is confident (|p - 0.5| > threshold)")
    print(f"{'thresh':>8s} {'N_base':>8s} {'acc_base':>9s} {'N_model':>8s} {'acc_model':>10s}")
    for thr in [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]:
        sub_base = base[np.abs(base["p"] - 0.5) >= thr]
        sub_model = rl_wf[np.abs(rl_wf["p"] - 0.5) >= thr]
        ab = sub_base["rl_won"].mean() if len(sub_base) else float('nan')
        am = sub_model["rl_won"].mean() if len(sub_model) else float('nan')
        print(f"{thr:>8.2f} {len(sub_base):>8,} {ab:>9.4f} {len(sub_model):>8,} {am:>10.4f}")
    print()

    # Trivial baselines for context
    away_base = (df["away_cover_rl"] == 1).mean()
    home_base = (df["home_cover_rl"] == 1).mean()
    print("=== TRIVIAL BASELINES ===")
    print(f"Always pick AWAY +1.5 (the dog): {away_base:.4f}")
    print(f"Always pick HOME -1.5 (the fav): {home_base:.4f}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"BASELINE (lean from ML):    {acc_base:.4f}")
    print(f"DEDICATED MODEL (cover RL): {acc_model:.4f}")
    delta = acc_model - acc_base
    print(f"Delta vs lean:               {delta:+.4f}  ({delta*100:+.2f}pp)")
    print(f"Vs always-away baseline:     {acc_model - away_base:+.4f}  (model {'beats' if acc_model > away_base else 'loses to'} trivial)")
    print("=" * 60)
    return acc_base, acc_model


if __name__ == "__main__":
    main()
