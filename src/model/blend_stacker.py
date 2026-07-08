"""Per-segment blend stacker — learns contextual market-vs-model weighting.

Replaces the static 2-context blend (agree=0.50, disagree=0.80) with a
learned meta-model that takes [p_model, p_market, context features] and
outputs a calibrated p_home.

Training:
  - Base model trained on train set (2023-2024) — already done by train.py
  - Stacker trained on val set (2025-01..2025-07) — this script
  - Stacker evaluated on test set (2025-08..2025-12)

Inputs to stacker per game:
  - p_model (LightGBM raw output from base classifier)
  - p_market (de-vigged market probability)
  - p_diff = p_model - p_market (signed disagreement)
  - p_disagree = |p_model - p_market| (magnitude)
  - direction_agree (1 if same side picked)
  - context: market_n_providers, market_p_home_std (market confidence),
             starter_xwoba_l15_diff (pitcher edge),
             win_pct_l30_diff (team strength edge),
             home_elo_pre, elo_diff_pre

Output: data/models/blend_stacker.pkl
  {model, feature_names}

Run:
  python -m src.model.blend_stacker
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from .dataset import load_split, CATEGORICAL_FEATURES

PROCESSED = Path("data/processed")
MODELS = Path("data/models")
OUT = MODELS / "blend_stacker.pkl"


CONTEXT_FEATURES = [
    "market_n_providers",
    "market_p_home_std",
    "starter_xwoba_l15_diff",
    "win_pct_l30_diff",
    "home_elo_pre",
    "elo_diff_pre",
    "run_diff_l10_diff",
    "off_xwoba_l30_diff",
    "def_xwoba_l30_diff",
]


def _devig_market(p_home: pd.Series) -> pd.Series:
    """Market p_home in train.parquet is already devigged. Return as-is."""
    return p_home


def main():
    split = load_split()
    print(f"train: {len(split.X_train):,}  val: {len(split.X_val):,}  test: {len(split.X_test):,}")

    # ── Step 1: load base classifier predictions (raw, uncalibrated) ─────────
    with open(MODELS / "lgb_cls.pkl", "rb") as f:
        base = pickle.load(f)
    base_model = base["model"]
    feats = base["feature_names"]
    feats = [f for f in feats if f in split.X_val.columns]

    def _prep_X(X):
        Xc = X[feats].copy()
        for c in CATEGORICAL_FEATURES:
            if c in Xc.columns:
                Xc[c] = pd.to_numeric(Xc[c], errors="coerce").fillna(-1).astype("int32")
        return Xc

    p_model_val  = base_model.predict_proba(_prep_X(split.X_val))[:, 1]
    p_model_test = base_model.predict_proba(_prep_X(split.X_test))[:, 1]
    y_val  = split.y_val_cls.values
    y_test = split.y_test_cls.values
    p_mkt_val  = split.X_val["market_p_home"].values  if "market_p_home" in split.X_val.columns else np.full(len(split.X_val), np.nan)
    p_mkt_test = split.X_test["market_p_home"].values if "market_p_home" in split.X_test.columns else np.full(len(split.X_test), np.nan)

    # Need market_p_home — load from train.parquet (it's in meta cols only)
    if np.all(np.isnan(p_mkt_val)):
        # Re-load full train.parquet to get market_p_home
        full = pd.read_parquet(PROCESSED / "train.parquet")
        m = full[["game_pk", "market_p_home"]]
        p_mkt_val  = split.meta_val.merge(m,  on="game_pk", how="left")["market_p_home"].values
        p_mkt_test = split.meta_test.merge(m, on="game_pk", how="left")["market_p_home"].values
    p_mkt_val  = np.clip(p_mkt_val,  1e-6, 1-1e-6)
    p_mkt_test = np.clip(p_mkt_test, 1e-6, 1-1e-6)

    # ── Step 2: build stacker feature matrix ─────────────────────────────────
    def _make_stacker_X(X, p_model, p_market):
        out = pd.DataFrame({
            "p_model":     p_model,
            "p_market":    p_market,
            "p_diff":      p_model - p_market,
            "p_disagree":  np.abs(p_model - p_market),
            "direction_agree": ((p_model > 0.5) == (p_market > 0.5)).astype(int),
        })
        for c in CONTEXT_FEATURES:
            if c in X.columns:
                out[c] = X[c].values
            else:
                out[c] = np.nan
        return out.fillna(0)  # fillna only after building so we have a clean numeric matrix

    Xs_val  = _make_stacker_X(split.X_val,  p_model_val,  p_mkt_val)
    Xs_test = _make_stacker_X(split.X_test, p_model_test, p_mkt_test)

    # Filter out rows without market info (rare in our window)
    mask_val  = ~pd.isna(p_mkt_val)
    mask_test = ~pd.isna(p_mkt_test)
    Xs_val_f  = Xs_val[mask_val].reset_index(drop=True)
    Xs_test_f = Xs_test[mask_test].reset_index(drop=True)
    y_val_f   = y_val[mask_val]
    y_test_f  = y_test[mask_test]

    print(f"\nStacker training set: {len(Xs_val_f):,} games (val)")
    print(f"Stacker test set:     {len(Xs_test_f):,} games (test)")
    print(f"Features: {list(Xs_val_f.columns)}")

    # ── Step 3: baselines (what we have today) ────────────────────────────────
    print("\n=== Baselines on TEST ===")
    def _ev(name, p):
        p = np.clip(p, 1e-6, 1-1e-6)
        acc = accuracy_score(y_test_f, (p >= 0.5).astype(int))
        ll  = log_loss(y_test_f, p)
        auc = roc_auc_score(y_test_f, p)
        br  = brier_score_loss(y_test_f, p)
        print(f"  {name:30s}  acc={acc:.4f}  AUC={auc:.4f}  ll={ll:.4f}  Brier={br:.4f}")
        return acc

    _ev("model only",   p_model_test[mask_test])
    _ev("market only",  p_mkt_test[mask_test])
    # current 2-bucket blend
    pm = p_model_test[mask_test]; mp = p_mkt_test[mask_test]
    agree = (pm > 0.5) == (mp > 0.5)
    w = np.where(agree, 0.50, 0.80)
    blend_2bucket = w * mp + (1-w) * pm
    _ev("2-bucket blend (current)", blend_2bucket)
    # naive 0.55 blend
    _ev("flat 0.55 blend",  0.55 * mp + 0.45 * pm)

    # ── Step 4: train stacker ─────────────────────────────────────────────────
    print("\n=== Training stackers ===")
    # Logistic regression — interpretable
    sc = StandardScaler()
    Xs_val_n  = sc.fit_transform(Xs_val_f)
    Xs_test_n = sc.transform(Xs_test_f)
    lr = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs", random_state=42)
    lr.fit(Xs_val_n, y_val_f)
    p_lr = lr.predict_proba(Xs_test_n)[:, 1]
    lr_acc = _ev("Logistic stacker", p_lr)

    # Shallow LightGBM — non-linear interactions
    gbm = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.02, num_leaves=8,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    gbm.fit(Xs_val_f, y_val_f)
    p_gbm = gbm.predict_proba(Xs_test_f)[:, 1]
    gbm_acc = _ev("LightGBM stacker", p_gbm)

    # Pick the winner
    use_gbm = gbm_acc >= lr_acc
    chosen = ("gbm", gbm) if use_gbm else ("lr", (sc, lr))
    print(f"\nChosen: {chosen[0]}")

    # ── Step 5: save bundle ───────────────────────────────────────────────────
    with open(OUT, "wb") as f:
        pickle.dump({
            "kind":            chosen[0],
            "model":           chosen[1],
            "feature_names":   list(Xs_val_f.columns),
            "context_features": CONTEXT_FEATURES,
        }, f)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
