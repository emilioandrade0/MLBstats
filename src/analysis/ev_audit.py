"""EV leak audit — where are we actually losing/winning money?

For all completed games in the recent window, replay the production pick logic
and break down realized P&L by:
  - ev_grade bucket (sweet / marginal / overconfident / no_edge)
  - edge band (% diff between model and book)
  - decimal odds bucket (heavy fav, fav, even, dog, big dog)
  - home vs away pick
  - confidence band (|p_model - 0.5|)
  - slot of day
  - day-of-week
  - month (recency trend)

Each segment shows: N, win rate, units P&L, ROI.

Reveals systematic leaks we can patch with new rules / filters.

Run:
  python -m src.analysis.ev_audit [--days 90]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
MODELS = Path("data/models")


def _decimal_to_implied(d):
    return 1.0 / d if d and d > 0 else None


def _build_picks(days: int):
    """Replay production logic and produce one row per pick."""
    train = pd.read_parquet(PROCESSED / "train.parquet")
    train["game_date"] = pd.to_datetime(train["game_date"])
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    sub = train[(train["game_date"] >= cutoff)
                & train["home_score"].notna()
                & train["away_score"].notna()].copy()
    if sub.empty:
        return None
    sub["home_win"] = (sub["home_score"] > sub["away_score"]).astype(int)

    # VALUE probability = standalone lgb_cls calibrator, NO market blend and
    # NO shrink. This is exactly what edge_threshold_sweep.py used to set the
    # 9pp threshold, and what api._raw_model_phome now mirrors for the value path.
    with open(MODELS / "lgb_cls.pkl", "rb") as f:
        clf = pickle.load(f)
    feats = clf["feature_names"]
    X = sub[feats].copy()
    for c in ["home_team_id", "away_team_id"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-1).astype("int32")
    sub["p_home"] = clf["calibrator"].predict_proba(X)[:, 1]

    # Build market lines from odds_close
    oc = pd.read_parquet(PROCESSED / "odds_close.parquet")
    sharp = oc[oc["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    sharp["home_ml"] = (sharp["home_ml_close"].fillna(sharp["home_ml_current"])
                                              .fillna(sharp["home_ml_top"]))
    sharp["away_ml"] = (sharp["away_ml_close"].fillna(sharp["away_ml_current"])
                                              .fillna(sharp["away_ml_top"]))
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id").agg(
        home_ml=("home_ml", "median"),
        away_ml=("away_ml", "median"),
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
    # De-vig
    s = sub["book_imp_home"] + sub["book_imp_away"]
    sub["mkt_p_home"] = sub["book_imp_home"]/s
    sub["mkt_p_away"] = sub["book_imp_away"]/s

    # Edge vs book (with vig — what EV is computed on)
    sub["edge_home"] = sub["p_home"] - sub["book_imp_home"]
    sub["edge_away"] = (1-sub["p_home"]) - sub["book_imp_away"]
    sub["best_edge"] = np.maximum(sub["edge_home"], sub["edge_away"])
    sub["pick_side"] = np.where(sub["edge_home"] >= sub["edge_away"], "HOME", "AWAY")
    # If no side has positive edge, still produce the model pick (for accuracy
    # diagnostics) but mark grade as no_edge.
    sub["pick_dec"] = np.where(sub["pick_side"]=="HOME", sub["dec_home"], sub["dec_away"])
    sub["pick_p_model"] = np.where(sub["pick_side"]=="HOME", sub["p_home"], 1-sub["p_home"])
    sub["pick_won"] = np.where(sub["pick_side"]=="HOME", sub["home_win"]==1, sub["home_win"]==0)

    # ── NEW grading (matches api.py _value_block recalibration) ─────────────
    # Quality over quantity: only edge >= 5pp counts as playable.
    e = sub["best_edge"]
    edge_pp = e * 100
    sub["ev_grade"] = np.where(edge_pp >= 5.0, "sweet",
                       np.where(edge_pp >= 0.0, "overconfident", "no_edge"))
    # Filter 1: skip favs <1.85 with edge <5pp (no-op since sweet requires >=5pp)
    fav_filter = (sub["ev_grade"] == "sweet") & (sub["pick_dec"] < 1.85) & (edge_pp < 5.0)
    sub.loc[fav_filter, "ev_grade"] = "no_edge"
    # Filter 2: skip low-conf picks (|p-0.5| < 2pp) within sweet
    conf_pp = np.abs(sub["p_home"] - 0.5) * 100
    low_conf = (sub["ev_grade"] == "sweet") & (conf_pp < 2.0)
    sub.loc[low_conf, "ev_grade"] = "no_edge"

    # Profit (1u stake): win = dec-1, loss = -1
    sub["profit_u"] = np.where(sub["pick_won"], sub["pick_dec"]-1, -1.0)
    return sub.dropna(subset=["pick_dec", "pick_p_model"])


def _segment(df, key, name):
    g = df.groupby(key).agg(
        n=("profit_u", "size"),
        wr=("pick_won", "mean"),
        units=("profit_u", "sum"),
        avg_dec=("pick_dec", "mean"),
        avg_edge=("best_edge", "mean"),
    ).reset_index()
    g["roi_pct"] = (g["units"]/g["n"]*100).round(2)
    g["wr"] = (g["wr"]*100).round(1)
    g["avg_dec"] = g["avg_dec"].round(2)
    g["avg_edge_pp"] = (g["avg_edge"]*100).round(2)
    g = g.drop(columns=["avg_edge"])
    print(f"\n=== Segment: {name} ===")
    print(g.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    args = ap.parse_args()

    df = _build_picks(args.days)
    if df is None or len(df) == 0:
        print("No data in window.")
        return
    print(f"Audited {len(df):,} games · last {args.days} days "
          f"({df['game_date'].min().date()} to {df['game_date'].max().date()})")
    print(f"Overall: WR={df['pick_won'].mean()*100:.1f}%  units={df['profit_u'].sum():+.2f}  "
          f"ROI={df['profit_u'].sum()/len(df)*100:+.2f}%")

    # EV grade
    _segment(df, "ev_grade", "by ev_grade")

    # Edge band
    df["edge_band"] = pd.cut(df["best_edge"]*100,
        bins=[-100, -2, 0, 1, 2, 3, 5, 100],
        labels=["very_neg", "neg", "0-1pp", "1-2pp(sweet)", "2-3pp", "3-5pp", "5pp+"])
    _segment(df, "edge_band", "by edge band")

    # Decimal odds bucket (favs vs dogs)
    df["dec_bucket"] = pd.cut(df["pick_dec"],
        bins=[0, 1.4, 1.6, 1.8, 2.0, 2.3, 3.0, 99],
        labels=["heavy_fav(<1.4)", "fav(1.4-1.6)", "mid_fav(1.6-1.8)", "even(1.8-2.0)",
                "small_dog(2.0-2.3)", "dog(2.3-3.0)", "big_dog(3.0+)"])
    _segment(df, "dec_bucket", "by decimal odds bucket")

    # Side
    _segment(df, "pick_side", "HOME vs AWAY pick")

    # Confidence band
    df["conf_pp"] = (np.abs(df["p_home"] - 0.5)*100).astype(int)
    df["conf_band"] = pd.cut(df["conf_pp"],
        bins=[-1, 2, 4, 6, 10, 15, 100],
        labels=["0-2pp", "2-4pp", "4-6pp", "6-10pp", "10-15pp", "15pp+"])
    _segment(df, "conf_band", "by model confidence band")

    # Day of week
    df["dow"] = df["game_date"].dt.day_name().str[:3]
    _segment(df, "dow", "by day of week")

    # Month (recency trend)
    df["month"] = df["game_date"].dt.to_period("M").astype(str)
    _segment(df, "month", "by month")

    # ── Only playable picks (sweet/marginal — what we actually bet) ──
    print("\n\n" + "=" * 80)
    print("LIMITED to playable picks (sweet + marginal):")
    print("=" * 80)
    play = df[df["ev_grade"].isin(["sweet", "marginal"])]
    if len(play):
        print(f"N={len(play):,}  WR={play['pick_won'].mean()*100:.1f}%  "
              f"units={play['profit_u'].sum():+.2f}  ROI={play['profit_u'].sum()/len(play)*100:+.2f}%")
        _segment(play, "dec_bucket", "playable · by odds bucket")
        _segment(play, "pick_side", "playable · HOME vs AWAY")
        _segment(play, "conf_band", "playable · by confidence")


if __name__ == "__main__":
    main()
