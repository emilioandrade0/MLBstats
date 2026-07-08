"""Find the optimal edge threshold for the new model (with lineup_recent).

The old threshold (5pp) was calibrated to the old model. The new model is
better-calibrated, so more games naturally cross 5pp — but the marginal ones
are noise. Sweep thresholds from 4pp to 10pp and find the best EV.

Also tests filter combinations:
  - skip favs < 1.85 (already on)
  - skip low-conf < 2pp (already on)
  - try tighter low-conf < 3pp

Run:
  python -m src.analysis.edge_threshold_sweep
"""
from __future__ import annotations

import pickle
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
MODELS = Path("data/models")


def _build_picks(days: int = 90):
    train = pd.read_parquet(PROCESSED / "train.parquet")
    train["game_date"] = pd.to_datetime(train["game_date"])
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    sub = train[(train["game_date"] >= cutoff)
                & train["home_score"].notna()
                & train["away_score"].notna()].copy()
    sub["home_win"] = (sub["home_score"] > sub["away_score"]).astype(int)

    with open(MODELS / "lgb_cls.pkl", "rb") as f:
        clf = pickle.load(f)
    feats = clf["feature_names"]
    X = sub[feats].copy()
    for c in ["home_team_id", "away_team_id"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-1).astype("int32")
    sub["p_home"] = clf["calibrator"].predict_proba(X)[:, 1]

    # Market lines
    oc = pd.read_parquet(PROCESSED / "odds_close.parquet")
    sharp = oc[oc["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    sharp["home_ml"] = sharp["home_ml_close"].fillna(sharp["home_ml_current"]).fillna(sharp["home_ml_top"])
    sharp["away_ml"] = sharp["away_ml_close"].fillna(sharp["away_ml_current"]).fillna(sharp["away_ml_top"])
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id").agg(
        home_ml=("home_ml", "median"), away_ml=("away_ml", "median")
    ).reset_index()
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    matched = xref[xref["_merge"]=="both"][["espn_event_id","game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    ml = agg.merge(matched, on="espn_event_id", how="inner") \
            .drop(columns=["espn_event_id"]).drop_duplicates("game_pk")
    sub = sub.merge(ml, on="game_pk", how="left")

    def _am_to_dec(v):
        if pd.isna(v): return np.nan
        v = float(v)
        return 1 + (v/100 if v > 0 else 100/abs(v))

    sub["dec_home"] = sub["home_ml"].apply(_am_to_dec)
    sub["dec_away"] = sub["away_ml"].apply(_am_to_dec)
    sub["book_imp_home"] = 1/sub["dec_home"]
    sub["book_imp_away"] = 1/sub["dec_away"]
    sub["edge_home"] = sub["p_home"] - sub["book_imp_home"]
    sub["edge_away"] = (1-sub["p_home"]) - sub["book_imp_away"]
    sub["best_edge"] = np.maximum(sub["edge_home"], sub["edge_away"])
    sub["pick_side"] = np.where(sub["edge_home"] >= sub["edge_away"], "HOME", "AWAY")
    sub["pick_dec"] = np.where(sub["pick_side"]=="HOME", sub["dec_home"], sub["dec_away"])
    sub["pick_won"] = np.where(sub["pick_side"]=="HOME", sub["home_win"]==1, sub["home_win"]==0)
    sub["profit_u"] = np.where(sub["pick_won"], sub["pick_dec"]-1, -1.0)
    sub["conf_pp"] = np.abs(sub["p_home"] - 0.5) * 100
    return sub.dropna(subset=["pick_dec"])


def _apply_filters(df, edge_min, fav_min_dec=None, low_conf_min=None):
    edge_pp = df["best_edge"] * 100
    keep = edge_pp >= edge_min
    if fav_min_dec is not None:
        keep &= ~((df["pick_dec"] < fav_min_dec) & (edge_pp < edge_min + 2))
    if low_conf_min is not None:
        keep &= df["conf_pp"] >= low_conf_min
    return df[keep]


def main():
    df = _build_picks(days=90)
    print(f"Audited {len(df):,} games · last 90 days "
          f"({df['game_date'].min().date()} to {df['game_date'].max().date()})")
    print()
    print("=" * 100)
    print(f"{'edge_min':>9s} {'fav_filter':>11s} {'low_conf':>9s} {'N':>5s} {'WR':>7s} "
          f"{'units':>8s} {'ROI':>8s}  comment")
    print("=" * 100)

    configs = [
        (4, None, None,  "edge>=4pp, no filters"),
        (5, None, None,  "edge>=5pp, no filters"),
        (6, None, None,  "edge>=6pp"),
        (7, None, None,  "edge>=7pp"),
        (8, None, None,  "edge>=8pp"),
        (9, None, None,  "edge>=9pp"),
        (10, None, None, "edge>=10pp"),
        # With filters at each threshold
        (5, 1.85, 2.0, "edge>=5 + favs>=1.85 + conf>=2pp (OLD prod)"),
        (6, 1.85, 2.0, "edge>=6 + favs>=1.85 + conf>=2pp"),
        (7, 1.85, 2.0, "edge>=7 + favs>=1.85 + conf>=2pp"),
        (8, 1.85, 2.0, "edge>=8 + favs>=1.85 + conf>=2pp"),
        (7, 1.85, 3.0, "edge>=7 + favs>=1.85 + conf>=3pp (tighter)"),
        (8, 1.85, 3.0, "edge>=8 + favs>=1.85 + conf>=3pp"),
        (6, 1.6,  2.0, "edge>=6 + favs>=1.60 + conf>=2pp (looser fav)"),
        (6, 2.0,  3.0, "edge>=6 + favs>=2.00 + conf>=3pp (only dogs/even)"),
    ]
    for edge_min, fav_min_dec, low_conf, comment in configs:
        f = _apply_filters(df, edge_min, fav_min_dec, low_conf)
        n = len(f)
        if n == 0:
            print(f"{edge_min:>9} {str(fav_min_dec):>11} {str(low_conf):>9} {n:>5}  (no picks)")
            continue
        wr = f["pick_won"].mean() * 100
        units = f["profit_u"].sum()
        roi = units / n * 100
        print(f"{edge_min:>9} {str(fav_min_dec):>11} {str(low_conf):>9} {n:>5} {wr:>6.1f}% "
              f"{units:>+8.2f} {roi:>+7.2f}%  {comment}")


if __name__ == "__main__":
    main()
