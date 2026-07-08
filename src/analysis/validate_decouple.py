"""Validate the accuracy/value decoupling on the holdout window.

Reproduces both production probabilities and confirms:
  1. DISPLAY accuracy (market-heavy blend) >= old blend accuracy
  2. VALUE edges (model-only) are preserved (not shrunk to zero)

Run:
  python -m src.analysis.validate_decouple
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

PROCESSED = Path("data/processed")
MODELS = Path("data/models")

SHRINK_FACTOR = 0.75
SHRINK_ANCHOR = 0.55
W_AGREE_OLD, W_DIS_OLD = 0.50, 0.80
W_AGREE_NEW, W_DIS_NEW = 0.70, 0.85


def _shrink(p):
    out = p.copy()
    high = p > SHRINK_ANCHOR
    out[high] = SHRINK_ANCHOR + (p[high] - SHRINK_ANCHOR) * SHRINK_FACTOR
    low = p < (1 - SHRINK_ANCHOR)
    out[low] = (1 - SHRINK_ANCHOR) - ((1 - SHRINK_ANCHOR) - p[low]) * SHRINK_FACTOR
    return np.clip(out, 1e-6, 1 - 1e-6)


def _blend(p_model, mp, wa, wd):
    agree = (p_model > 0.5) == (mp > 0.5)
    w = np.where(agree, wa, wd)
    return _shrink(w * mp + (1 - w) * p_model)


def main():
    with open(MODELS / "ensemble.pkl", "rb") as f:
        ens = pickle.load(f)
    model = ens["model"]
    feats = ens["feature_names"]

    df = pd.read_parquet(PROCESSED / "train.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["home_score", "away_score", "market_p_home"]).copy()
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    # Holdout: last 120 days
    cutoff = df["game_date"].max() - pd.Timedelta(days=120)
    sub = df[df["game_date"] >= cutoff].copy()
    feats = [f for f in feats if f in sub.columns]

    X = sub[feats].copy()
    for c in ["home_team_id", "away_team_id"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-1).astype("int32")
    p_model = model.predict_proba(X)[:, 1]
    mp = sub["market_p_home"].values
    y = sub["home_win"].values

    print(f"Holdout window: {len(sub):,} games "
          f"({sub['game_date'].min().date()} → {sub['game_date'].max().date()})")
    print()

    def _ev(name, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        acc = accuracy_score(y, (p >= 0.5).astype(int))
        auc = roc_auc_score(y, p)
        ll  = log_loss(y, p)
        print(f"  {name:34s}  acc={acc:.4f}  AUC={auc:.4f}  ll={ll:.4f}")
        return acc

    print("=== DISPLAY probability (who-wins accuracy) ===")
    _ev("model only (raw+shrink)",      _shrink(p_model))
    _ev("OLD blend (agree=0.50/0.80)",  _blend(p_model, mp, W_AGREE_OLD, W_DIS_OLD))
    _ev("NEW blend (agree=0.70/0.85)",  _blend(p_model, mp, W_AGREE_NEW, W_DIS_NEW))
    _ev("market only",                  mp)
    print()

    # ── Value edge preservation ──────────────────────────────────────────────
    print("=== VALUE edges (model-only vs old blended) ===")
    p_value = _shrink(p_model)         # what the new value path uses
    p_old_value = _blend(p_model, mp, W_AGREE_OLD, W_DIS_OLD)  # old value path
    book_imp = mp  # approx de-vigged implied (market_p_home is already devigged)
    edge_new = np.abs(np.maximum(p_value - book_imp, (1-p_value) - (1-book_imp)))
    edge_old = np.abs(np.maximum(p_old_value - book_imp, (1-p_old_value) - (1-book_imp)))
    print(f"  mean |edge| model-only:  {edge_new.mean()*100:.2f}pp")
    print(f"  mean |edge| old blend:   {edge_old.mean()*100:.2f}pp")
    print(f"  games with edge>=5pp model-only: {(edge_new>=0.05).sum()}")
    print(f"  games with edge>=5pp old blend:  {(edge_old>=0.05).sum()}")
    print(f"  games with edge>=9pp model-only: {(edge_new>=0.09).sum()}")
    print(f"  games with edge>=9pp old blend:  {(edge_old>=0.09).sum()}")


if __name__ == "__main__":
    main()
