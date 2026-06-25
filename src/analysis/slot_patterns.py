"""Test whether the slot of a game within the day (1st, 2nd, … Nth) carries
predictive signal independent of teams/odds.

Hypothesis: early-day matinees, primetime games, and late-West-Coast games
might be priced or played differently — different umpire pools, different
public attention, different lineup completeness when the line is set.

Method:
  1. For each date, rank games by first_pitch_utc → slot 1, 2, …
  2. Group walk-forward predictions by slot and report:
       n, home win rate, mean p_home, model acc, market acc,
       blind-bet ROI on the model pick (at the median DK/EB price), and
       a simple Bonferroni-aware test against the global rate.
  3. Spit out a table the user can eyeball before deciding to add a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..normalize.paths import PROCESSED


def _american_to_decimal(a: float | None) -> float | None:
    if a is None or not np.isfinite(a) or a == 0:
        return None
    return 1 + (a / 100.0 if a > 0 else 100.0 / -a)


def main() -> None:
    wp = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet")[
        ["game_pk", "first_pitch_utc"]
    ].copy()
    odds = pd.read_parquet(PROCESSED / "odds_close.parquet")
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")

    # Median DK/EB closing ML per game_pk, in decimal.
    sharp = odds[odds["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    for side in ("home", "away"):
        sharp[f"{side}_ml"] = (
            sharp[f"{side}_ml_close"]
            .fillna(sharp[f"{side}_ml_current"])
            .fillna(sharp[f"{side}_ml_top"])
        )
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id").agg(
        home_ml=("home_ml", "median"),
        away_ml=("away_ml", "median"),
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

    # Rank by start time within each calendar date (US Eastern — closest to
    # the "feel" of a slate).
    df["et"] = df["first_pitch_utc"].dt.tz_convert("America/New_York")
    df["et_date"] = df["et"].dt.date
    df["slot"] = df.groupby("et_date")["et"].rank(method="first").astype(int)
    df["games_in_day"] = df.groupby("et_date")["et"].transform("size")
    df["hour_et"] = df["et"].dt.hour

    # Model pick + ROI assuming bet $1 at median DK/EB on the pick.
    df["pick_home"] = df["p_home"] >= 0.5
    df["pick_won"] = np.where(df["pick_home"], df["home_win"] == 1, df["home_win"] == 0)
    df["pick_dec"] = np.where(df["pick_home"], df["dec_home"], df["dec_away"])
    df["pnl"] = np.where(df["pick_won"], df["pick_dec"] - 1, -1.0)

    # Market accuracy: pick the side the market favors.
    df["mkt_pick_home"] = df["market_p_home"] >= 0.5
    df["mkt_correct"] = np.where(
        df["mkt_pick_home"], df["home_win"] == 1, df["home_win"] == 0
    )
    df["model_correct"] = df["pick_won"]

    # Edge in pp.
    df["edge_pp"] = (df["p_home"] - df["market_p_home"]) * 100

    global_acc = df["model_correct"].mean()
    global_mkt = df["mkt_correct"].mean()
    global_roi = df["pnl"].mean()

    print(f"global  n={len(df):,}  model_acc={global_acc:.4f}  "
          f"mkt_acc={global_mkt:.4f}  roi/$={global_roi:+.4f}\n")

    # ---------- by slot (1 = first game of the day) ----------
    print("=" * 88)
    print("BY SLOT (first game of the day = 1)")
    print("=" * 88)
    print(f"{'slot':>4} {'n':>6} {'home%':>7} {'mkt_acc':>8} {'mod_acc':>8} "
          f"{'d_mkt(pp)':>9} {'ROI/$':>8} {'p_acc':>7} {'p_roi':>7}")
    rows = []
    for slot, g in df.groupby("slot"):
        if len(g) < 50:
            continue
        n = len(g)
        mkt = g["mkt_correct"].mean()
        mod = g["model_correct"].mean()
        # Two-sided binomial vs global model accuracy
        _, p_acc = stats.binomtest(int(g["model_correct"].sum()), n, global_acc).pvalue, None
        p_acc = stats.binomtest(int(g["model_correct"].sum()), n, global_acc).pvalue
        # One-sample t for ROI vs 0 (and vs global)
        t_roi, p_roi = stats.ttest_1samp(g["pnl"], 0.0)
        rows.append({
            "slot": slot, "n": n, "home_pct": g["home_win"].mean(),
            "mkt_acc": mkt, "mod_acc": mod,
            "delta_mkt": mod - mkt, "roi": g["pnl"].mean(),
            "p_acc": p_acc, "p_roi": p_roi,
        })
        print(f"{slot:>4} {n:>6} {g['home_win'].mean()*100:>6.1f}% "
              f"{mkt:>7.4f} {mod:>7.4f} {(mod-mkt)*100:>+8.2f}pp "
              f"{g['pnl'].mean():>+8.4f} {p_acc:>7.3f} {p_roi:>7.3f}")

    # ---------- by ET hour bucket (more interpretable) ----------
    print()
    print("=" * 88)
    print("BY ET HOUR BUCKET")
    print("=" * 88)
    buckets = [
        ("matinee (≤14)",  df["hour_et"] <= 14),
        ("afternoon (15-17)", df["hour_et"].between(15, 17)),
        ("primetime (18-20)", df["hour_et"].between(18, 20)),
        ("late (21-23)",    df["hour_et"].between(21, 23)),
        ("west-coast (≥21 & home AL/NL west)",  df["hour_et"] >= 21),
    ]
    print(f"{'bucket':>40} {'n':>6} {'home%':>7} {'mkt_acc':>8} {'mod_acc':>8} {'ROI/$':>8} {'p':>6}")
    for name, mask in buckets:
        g = df[mask]
        if len(g) < 50:
            continue
        n = len(g)
        roi = g["pnl"].mean()
        _, p_roi = stats.ttest_1samp(g["pnl"], 0.0)
        print(f"{name:>40} {n:>6} {g['home_win'].mean()*100:>6.1f}% "
              f"{g['mkt_correct'].mean():>7.4f} {g['model_correct'].mean():>7.4f} "
              f"{roi:>+8.4f} {p_roi:>6.3f}")

    # ---------- "first game of the day" vs "rest" ----------
    print()
    print("=" * 88)
    print("FIRST GAME OF DAY vs REST")
    print("=" * 88)
    first = df[df["slot"] == 1]
    rest = df[df["slot"] > 1]
    for label, g in [("first only", first), ("rest", rest)]:
        n = len(g)
        roi = g["pnl"].mean()
        _, p = stats.ttest_1samp(g["pnl"], 0.0)
        print(f"  {label:>12}: n={n:>5}  mod_acc={g['model_correct'].mean():.4f}  "
              f"mkt_acc={g['mkt_correct'].mean():.4f}  roi/$={roi:+.4f}  p={p:.3f}")

    # 2-sample t between first and rest
    t, p = stats.ttest_ind(first["pnl"], rest["pnl"], equal_var=False)
    print(f"  first vs rest ROI:  t={t:+.2f}  p={p:.3f}")

    # ---------- Bonferroni note ----------
    n_slots_tested = sum(1 for r in rows)
    print(f"\nBonferroni threshold across {n_slots_tested} slots: "
          f"p < {0.05 / max(n_slots_tested, 1):.4f}")
    print("(Anything above this is consistent with chance.)")

    # Save full per-slot table for follow-up.
    if rows:
        pd.DataFrame(rows).to_parquet(PROCESSED / "slot_analysis.parquet", index=False)
        print(f"\nWrote {PROCESSED / 'slot_analysis.parquet'}")


if __name__ == "__main__":
    main()
