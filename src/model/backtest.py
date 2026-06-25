"""Backtest the home-win classifier against the actual market.

Walks through the held-out test period game by game, applies our model's
probability, compares with the real (devigged) market probability, and bets
whichever side has positive expected value.

Three strategies:
  - flat_100: $100 on every value bet (edge >= threshold)
  - kelly:    Kelly criterion sizing  (f = (bp - q) / b)
  - qkelly:   1/4 Kelly (safer, standard for non-stationary markets)

We use the median moneyline across providers as the "market" line.

Outputs metrics to stdout and a per-game CSV at data/processed/backtest.parquet.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED

MODELS = PROCESSED.parent / "models"


def _american_to_decimal(ml: float) -> float:
    if not np.isfinite(ml) or ml == 0:
        return np.nan
    return (ml / 100 + 1) if ml > 0 else (100 / (-ml) + 1)


def _implied_prob_american(ml: float) -> float:
    if not np.isfinite(ml) or ml == 0:
        return np.nan
    return (-ml / (-ml + 100)) if ml < 0 else (100 / (ml + 100))


def _build_market_lines() -> pd.DataFrame:
    """Median home + away moneyline per game_pk (real lines you'd actually book)."""
    odds = pd.read_parquet(PROCESSED / "odds.parquet")
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    odds = odds.dropna(subset=["home_moneyline", "away_moneyline"])
    line = odds.groupby("espn_event_id").agg(
        home_ml=("home_moneyline", "median"),
        away_ml=("away_moneyline", "median"),
    ).reset_index()
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    return line.merge(matched, on="espn_event_id", how="inner") \
               .drop(columns=["espn_event_id"]) \
               .drop_duplicates("game_pk")


def _score_test() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    df = df.dropna(subset=["home_win", "total_runs"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    test = df[(df["game_date"] > "2025-07-31") & (df["game_date"] <= "2025-12-31")].copy()

    with open(MODELS / "lgb_cls.pkl", "rb") as f:
        b = pickle.load(f)
    X = test[b["feature_names"]]
    test["p_model_home"] = b["calibrator"].predict_proba(X)[:, 1]
    return test[["game_pk", "game_date", "home_team_abbrev", "away_team_abbrev",
                 "home_score", "away_score", "home_win", "p_model_home"]]


def _kelly(p: float, dec_odds: float) -> float:
    """Kelly fraction. Capped at 0 (no bet when negative EV)."""
    if not np.isfinite(p) or not np.isfinite(dec_odds) or dec_odds <= 1:
        return 0.0
    b = dec_odds - 1
    f = (b * p - (1 - p)) / b
    return max(0.0, f)


def simulate(edge_threshold: float = 0.03, kelly_frac: float = 0.25,
             bankroll0: float = 1_000.0) -> pd.DataFrame:
    pred = _score_test()
    lines = _build_market_lines()
    df = pred.merge(lines, on="game_pk", how="inner")
    if df.empty:
        raise SystemExit("No overlap between predictions and market lines.")

    df["dec_home"] = df["home_ml"].map(_american_to_decimal)
    df["dec_away"] = df["away_ml"].map(_american_to_decimal)
    p_h_raw = df["home_ml"].map(_implied_prob_american)
    p_a_raw = df["away_ml"].map(_implied_prob_american)
    total = p_h_raw + p_a_raw
    df["p_mkt_home"] = p_h_raw / total
    df["p_mkt_away"] = p_a_raw / total

    df["edge_home"] = df["p_model_home"] - df["p_mkt_home"]
    df["edge_away"] = (1 - df["p_model_home"]) - df["p_mkt_away"]

    # Choose the side with bigger positive edge (skip if neither passes threshold).
    df["side"] = np.where(
        (df["edge_home"] >= edge_threshold) & (df["edge_home"] >= df["edge_away"]),
        "HOME",
        np.where(
            (df["edge_away"] >= edge_threshold) & (df["edge_away"] > df["edge_home"]),
            "AWAY", "NONE",
        ),
    )
    df["bet_p"] = np.where(df["side"] == "HOME", df["p_model_home"],
                  np.where(df["side"] == "AWAY", 1 - df["p_model_home"], np.nan))
    df["bet_dec"] = np.where(df["side"] == "HOME", df["dec_home"],
                   np.where(df["side"] == "AWAY", df["dec_away"], np.nan))
    df["bet_won"] = np.where(df["side"] == "HOME", df["home_win"] == 1,
                    np.where(df["side"] == "AWAY", df["home_win"] == 0, np.nan))

    df = df.sort_values("game_date").reset_index(drop=True)

    # Flat $100
    flat = np.where(df["side"] == "NONE", 0.0,
            np.where(df["bet_won"] == 1.0, 100 * (df["bet_dec"] - 1), -100))
    df["pnl_flat100"] = flat
    df["bankroll_flat100"] = bankroll0 + flat.cumsum()

    # Quarter-Kelly with rolling bankroll
    bankroll = bankroll0
    pnl_qk = np.zeros(len(df))
    stake_qk = np.zeros(len(df))
    for i, row in df.iterrows():
        if row["side"] == "NONE":
            continue
        f = _kelly(row["bet_p"], row["bet_dec"]) * kelly_frac
        f = min(f, 0.05)  # cap at 5% of bankroll per bet
        stake = bankroll * f
        if row["bet_won"] == 1.0:
            pnl_qk[i] = stake * (row["bet_dec"] - 1)
        else:
            pnl_qk[i] = -stake
        stake_qk[i] = stake
        bankroll += pnl_qk[i]
    df["stake_qkelly"] = stake_qk
    df["pnl_qkelly"] = pnl_qk
    df["bankroll_qkelly"] = bankroll0 + pnl_qk.cumsum()

    # ROI summary
    bets = df[df["side"] != "NONE"]
    print(f"\n=== Edge threshold: {edge_threshold:.0%}  |  bets: {len(bets):,} / {len(df):,}  "
          f"({len(bets)/len(df):.1%}) ===")
    if len(bets):
        wr = (bets["bet_won"] == 1).mean()
        roi_flat = bets["pnl_flat100"].sum() / (100 * len(bets))
        roi_qk = bets["pnl_qkelly"].sum() / bets["stake_qkelly"].sum() \
            if bets["stake_qkelly"].sum() > 0 else 0
        print(f"  win rate:           {wr:.1%}")
        print(f"  ROI flat-$100:      {roi_flat:+.2%}  (profit ${bets['pnl_flat100'].sum():+.0f} on ${100*len(bets):,} risked)")
        print(f"  ROI 1/4 Kelly:      {roi_qk:+.2%}  (final bankroll ${df['bankroll_qkelly'].iloc[-1]:,.0f} from ${bankroll0:,.0f})")
        print(f"  max drawdown (QK):  ${(df['bankroll_qkelly'] - df['bankroll_qkelly'].cummax()).min():,.0f}")

        # Edge-bucket calibration
        print("\n  ROI by edge bucket:")
        bets = bets.copy()
        bets["edge_chosen"] = np.where(bets["side"] == "HOME", bets["edge_home"], bets["edge_away"])
        bets["bucket"] = pd.cut(bets["edge_chosen"],
                                 [0, 0.05, 0.10, 0.15, 0.25, 1],
                                 labels=["3-5%", "5-10%", "10-15%", "15-25%", "25%+"],
                                 include_lowest=True)
        for b, sub in bets.groupby("bucket", observed=True):
            wr_b = (sub["bet_won"] == 1).mean()
            roi_b = sub["pnl_flat100"].sum() / (100 * len(sub))
            print(f"    {str(b):8s} n={len(sub):4d}  WR={wr_b:.1%}  ROI={roi_b:+.2%}")
    return df


def main() -> None:
    print("\n" + "=" * 60)
    print("BACKTEST — test period 2025-08-01 to 2025-12-31")
    print("=" * 60)
    for th in (0.03, 0.05, 0.08):
        simulate(edge_threshold=th)
    df = simulate(edge_threshold=0.03)
    out = PROCESSED / "backtest.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
