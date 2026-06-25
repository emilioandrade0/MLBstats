"""Test: in slots where the model historically loses, does FLIPPING the pick
(betting the opposite side) actually win?

Logic:
  - If the model picks HOME but model loses systematically in this slot,
    maybe HOME is overestimated → AWAY at the AWAY price could be +EV.
  - Compare 4 strategies per slot:
      (a) model_pick: bet model's side at its price
      (b) flip_pick:  bet the opposite side at that side's price
      (c) mkt_pick:   bet the market's favored side
      (d) skip:       do nothing (ROI = 0 by definition)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..normalize.paths import PROCESSED


def _american_to_decimal(a):
    if a is None or not np.isfinite(a) or a == 0:
        return None
    return 1 + (a / 100.0 if a > 0 else 100.0 / -a)


def main() -> None:
    wp = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet")[["game_pk", "first_pitch_utc"]]
    odds = pd.read_parquet(PROCESSED / "odds_close.parquet")
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")

    sharp = odds[odds["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    for side in ("home", "away"):
        sharp[f"{side}_ml"] = (
            sharp[f"{side}_ml_close"]
            .fillna(sharp[f"{side}_ml_current"])
            .fillna(sharp[f"{side}_ml_top"])
        )
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id").agg(
        home_ml=("home_ml", "median"), away_ml=("away_ml", "median")
    ).reset_index()
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    ml = agg.merge(matched, on="espn_event_id", how="inner").drop_duplicates("game_pk")
    ml["dec_home"] = ml["home_ml"].apply(_american_to_decimal)
    ml["dec_away"] = ml["away_ml"].apply(_american_to_decimal)

    df = wp.merge(games, on="game_pk", how="left").merge(
        ml[["game_pk", "dec_home", "dec_away"]], on="game_pk", how="left"
    )
    df["first_pitch_utc"] = pd.to_datetime(df["first_pitch_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["first_pitch_utc", "dec_home", "dec_away"]).copy()
    df["et"] = df["first_pitch_utc"].dt.tz_convert("America/New_York")
    df["et_date"] = df["et"].dt.date
    df["slot"] = df.groupby("et_date")["et"].rank(method="first").astype(int)

    # 4 strategies' per-game PnL.
    df["model_home"] = df["p_home"] >= 0.5
    df["mkt_home"]   = df["market_p_home"] >= 0.5
    df["flip_home"]  = ~df["model_home"]

    def pnl(home_pick_mask):
        won = np.where(home_pick_mask, df["home_win"] == 1, df["home_win"] == 0)
        dec = np.where(home_pick_mask, df["dec_home"], df["dec_away"])
        return np.where(won, dec - 1, -1.0)

    df["pnl_model"] = pnl(df["model_home"].values)
    df["pnl_flip"]  = pnl(df["flip_home"].values)
    df["pnl_mkt"]   = pnl(df["mkt_home"].values)
    df["model_correct"] = (df["model_home"].values == (df["home_win"].values == 1))
    df["flip_correct"]  = ~df["model_correct"]

    print(f"global n={len(df):,}")
    print(f"  model: acc={df['model_correct'].mean():.4f}  roi/$={df['pnl_model'].mean():+.4f}")
    print(f"  flip : acc={df['flip_correct'].mean():.4f}  roi/$={df['pnl_flip'].mean():+.4f}")
    print(f"  mkt  : acc={(df['mkt_home']==(df['home_win']==1)).mean():.4f}  roi/$={df['pnl_mkt'].mean():+.4f}")
    print()

    print("=" * 100)
    print("PER-SLOT: ROI for model / flip / market / skip")
    print("=" * 100)
    print(f"{'slot':>4} {'n':>5} | {'mod_acc':>7} {'mod_roi':>8} | "
          f"{'flp_acc':>7} {'flp_roi':>8} | {'mkt_acc':>7} {'mkt_roi':>8} | "
          f"{'best':>7}")

    summary = []
    for slot, g in df.groupby("slot"):
        if len(g) < 50:
            continue
        n = len(g)
        mod_acc = g["model_correct"].mean()
        flp_acc = g["flip_correct"].mean()
        mkt_acc = (g["mkt_home"] == (g["home_win"] == 1)).mean()
        mod_roi = g["pnl_model"].mean()
        flp_roi = g["pnl_flip"].mean()
        mkt_roi = g["pnl_mkt"].mean()
        # Best of the 4 (skip = 0)
        candidates = {"model": mod_roi, "flip": flp_roi, "mkt": mkt_roi, "skip": 0.0}
        best = max(candidates, key=candidates.get)
        print(f"{slot:>4} {n:>5} | {mod_acc:>7.4f} {mod_roi:>+8.4f} | "
              f"{flp_acc:>7.4f} {flp_roi:>+8.4f} | {mkt_acc:>7.4f} {mkt_roi:>+8.4f} | "
              f"{best:>7}")
        summary.append({"slot": slot, "n": n,
                        "mod_roi": mod_roi, "flip_roi": flp_roi,
                        "mkt_roi": mkt_roi, "best": best})

    print()
    print("=" * 100)
    print("FLIP STRATEGY EVALUATION — does flipping beat model on losing slots?")
    print("=" * 100)
    sdf = pd.DataFrame(summary)
    losing_model_slots = sdf[sdf["mod_roi"] < -0.02]["slot"].tolist()
    print(f"Losing-model slots (ROI < -2%): {losing_model_slots}")
    for slot in losing_model_slots:
        g = df[df["slot"] == slot]
        # Paired t-test: pnl_flip vs pnl_model
        t, p = stats.ttest_rel(g["pnl_flip"], g["pnl_model"])
        edge = g["pnl_flip"].mean() - g["pnl_model"].mean()
        print(f"  slot {slot:>2}: flip - model = {edge:+.4f}  t={t:+.2f}  p={p:.3f}  "
              f"{'FLIP BETTER' if edge > 0 else 'FLIP WORSE'}")

    # Aggregate across all losing slots: pretend we use a hybrid strategy.
    print()
    print("=" * 100)
    print("HYBRID STRATEGY — 'use model on good slots, flip on bad slots'")
    print("=" * 100)
    good_slots = {1, 9}
    bad_slots  = {4, 5, 6, 12, 13, 14, 15}
    df["strategy"] = np.where(
        df["slot"].isin(good_slots), "model",
        np.where(df["slot"].isin(bad_slots), "flip", "model")
    )
    df["pnl_hybrid"] = np.where(df["strategy"] == "flip", df["pnl_flip"], df["pnl_model"])
    df["correct_hybrid"] = np.where(df["strategy"] == "flip", df["flip_correct"], df["model_correct"])
    print(f"  pure model : acc={df['model_correct'].mean():.4f}  roi/$={df['pnl_model'].mean():+.4f}")
    print(f"  flip-on-bad: acc={df['correct_hybrid'].mean():.4f}  roi/$={df['pnl_hybrid'].mean():+.4f}")
    print(f"  delta accuracy: {(df['correct_hybrid'].mean() - df['model_correct'].mean())*100:+.2f}pp")
    print(f"  delta ROI/$:    {df['pnl_hybrid'].mean() - df['pnl_model'].mean():+.4f}")

    # Per bad slot, what fraction of model picks would flip?
    print()
    print("Effect on each bad slot:")
    for slot in sorted(bad_slots):
        g = df[df["slot"] == slot]
        if len(g) < 50:
            continue
        flipped = g[g["model_home"] != g["flip_home"]]  # always 100% of rows
        print(f"  slot {slot:>2}: n={len(g):>3}  "
              f"model_acc={g['model_correct'].mean():.3f}  "
              f"flip_acc ={g['flip_correct'].mean():.3f}  "
              f"market_acc={(g['mkt_home']==(g['home_win']==1)).mean():.3f}")


if __name__ == "__main__":
    main()
