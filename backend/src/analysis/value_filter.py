"""Filter bets to only +EV games and measure profitability.

The honest fix: don't bet every game. Only bet games where our model's
probability of winning EXCEEDS the book's implied probability (after vig).
That's the textbook definition of a +EV bet — and it's the ONLY way to
overcome vig long-term.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


def shrink(p, factor=0.75, anchor=0.55):
    out = p.copy()
    high = p > anchor; low = p < 1 - anchor
    out[high] = anchor + (p[high] - anchor) * factor
    out[low] = (1 - anchor) - ((1 - anchor) - p[low]) * factor
    return out.clip(1e-6, 1 - 1e-6)


def ml_to_dec(ml):
    return np.where(ml > 0, ml / 100 + 1, 100 / (-ml) + 1)


def main() -> None:
    p = pd.read_parquet(PROCESSED / "walkforward_preds.parquet") \
          .dropna(subset=["market_p_home", "p_home_model_raw"]).copy()
    mp = p["market_p_home"].clip(1e-6, 1 - 1e-6)
    pm = p["p_home_model_raw"].clip(1e-6, 1 - 1e-6)
    agree = (pm > 0.5) == (mp > 0.5)
    w = np.where(agree, 0.50, 0.80)
    blended = w * mp + (1 - w) * pm
    p["p_final"] = shrink(blended)
    p["pick_home"] = (p["p_final"] > 0.5).astype(int)
    p["our_prob"] = np.where(p["pick_home"] == 1, p["p_final"], 1 - p["p_final"])
    p["won"] = np.where(p["pick_home"] == 1, p["home_win"] == 1,
                                              p["home_win"] == 0).astype(int)

    # Attach real DK / ESPN BET lines
    odds = pd.read_parquet(PROCESSED / "odds_close.parquet")
    sharp = odds[odds["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    sharp["home_ml"] = sharp["home_ml_close"].fillna(sharp["home_ml_current"]) \
                                              .fillna(sharp["home_ml_top"])
    sharp["away_ml"] = sharp["away_ml_close"].fillna(sharp["away_ml_current"]) \
                                              .fillna(sharp["away_ml_top"])
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id").agg(
        home_ml=("home_ml", "median"),
        away_ml=("away_ml", "median"),
    ).reset_index()
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    agg = agg.merge(matched, on="espn_event_id", how="inner").drop_duplicates("game_pk")
    p = p.merge(agg, on="game_pk", how="inner")

    p["dec_odds"] = np.where(p["pick_home"] == 1, ml_to_dec(p["home_ml"]),
                                                  ml_to_dec(p["away_ml"]))
    p["implied"] = 1.0 / p["dec_odds"]
    p["ev_per_dollar"] = p["our_prob"] * (p["dec_odds"] - 1) - (1 - p["our_prob"])
    p["edge_vs_book"] = p["our_prob"] - p["implied"]

    print(f"Total picks: {len(p):,}")
    print(f"Avg EV/dollar (all picks): {p['ev_per_dollar'].mean():+.4f}\n")

    print("Filtro: solo apostar games donde our_prob > implied (edge >= X)")
    print(f"  {'threshold':>12}  {'n':>5}  {'win_rate':>9}  {'avg_EV/$100':>11}  "
          f"{'flat_$100_ROI':>13}  {'total_$_won':>11}")
    for thresh in [-0.05, 0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        sub = p[p["edge_vs_book"] >= thresh]
        if len(sub) == 0:
            continue
        wr = sub["won"].mean()
        ev = sub["ev_per_dollar"].mean()
        won_payoffs = np.where(sub["won"] == 1, 100 * (sub["dec_odds"] - 1), -100)
        profit = won_payoffs.sum()
        roi = profit / (100 * len(sub))
        print(f"  {thresh:>+12.2%}  {len(sub):>5}  {wr:>9.4f}  "
              f"${ev*100:>+10.2f}  {roi:>+12.2%}  ${profit:>+10,.0f}")

    print("\nLectura:")
    print("  - 'flat_$100_ROI' es ROI apostando $100 en cada game que pasa el filtro.")
    print("  - Si la columna sale POSITIVA, el filtro encuentra edge real explotable.")
    print("  - Si sale NEGATIVA, el filtro no es suficiente para vencer la vig.\n")


if __name__ == "__main__":
    main()
