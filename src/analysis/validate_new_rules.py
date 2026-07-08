"""Validate each proposed rule on walk-forward BEFORE adding to the system.

Rules tested (each replaces the model's natural pick under certain conditions):
  Rule 1 — month-aware  : in July, if model picks favorite, flip to dog
  Rule 2 — fav-band fade: if fav decimal in [1.60, 1.70), flip to dog
  Rule 3a — team boost  : (no-op — trusting model where it already wins)
  Rule 3b — team fade   : fade ATL/MIN/BAL at home (flip model pick)

Decision rule: only adopt if BOTH accuracy AND ROI improve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


def _american_to_decimal(a):
    if a is None or not np.isfinite(a) or a == 0:
        return None
    return 1 + (a / 100.0 if a > 0 else 100.0 / -a)


def load():
    wp = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet")
    odds = pd.read_parquet(PROCESSED / "odds_close.parquet")
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")

    sharp = odds[odds["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    for s in ("home", "away"):
        sharp[f"{s}_ml"] = (sharp[f"{s}_ml_close"]
                            .fillna(sharp[f"{s}_ml_current"])
                            .fillna(sharp[f"{s}_ml_top"]))
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id").agg(
        home_ml=("home_ml", "median"), away_ml=("away_ml", "median")
    ).reset_index()
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    ml = agg.merge(matched, on="espn_event_id", how="inner").drop_duplicates("game_pk")
    ml["dec_home"] = ml["home_ml"].apply(_american_to_decimal)
    ml["dec_away"] = ml["away_ml"].apply(_american_to_decimal)

    wp["game_date"] = pd.to_datetime(wp["game_date"])
    df = wp.merge(games[["game_pk", "first_pitch_utc"]], on="game_pk", how="left")
    df = df.merge(ml[["game_pk", "dec_home", "dec_away"]], on="game_pk", how="left")
    df = df.dropna(subset=["p_home", "home_win", "dec_home", "dec_away"]).copy()

    df["month"] = df["game_date"].dt.month
    df["model_home"] = df["p_home"] >= 0.5
    df["mkt_home"]   = df["market_p_home"] >= 0.5
    df["fav_dec"]    = np.minimum(df["dec_home"], df["dec_away"])
    df["dog_dec"]    = np.maximum(df["dec_home"], df["dec_away"])
    df["model_picks_fav"] = (
        ((df["model_home"]) & (df["dec_home"] < df["dec_away"])) |
        ((~df["model_home"]) & (df["dec_away"] < df["dec_home"]))
    )
    return df


def stats_for(pick_home: pd.Series, df: pd.DataFrame) -> dict:
    won = df["home_win"] == 1
    correct = np.where(pick_home, won, ~won)
    dec = np.where(pick_home, df["dec_home"], df["dec_away"])
    pnl = np.where(correct, dec - 1, -1.0)
    return {"acc": correct.mean(), "roi": pnl.mean(),
            "n": len(df), "n_correct": int(correct.sum())}


def main() -> None:
    df = load()
    baseline = stats_for(df["model_home"], df)
    print(f"BASELINE model picks: n={baseline['n']:,}  "
          f"acc={baseline['acc']:.4f}  roi/$={baseline['roi']:+.4f}\n")

    # ───────────────────────────────────────────────────────────────────
    # Rule 1 — July: if model picks favorite, flip to dog
    # ───────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("RULE 1 — July: flip when model picks favorite")
    print("=" * 80)
    rule1_flip = (df["month"] == 7) & df["model_picks_fav"]
    pick1 = df["model_home"].copy()
    pick1.loc[rule1_flip] = ~pick1.loc[rule1_flip]
    s1 = stats_for(pick1, df)
    n_changed = rule1_flip.sum()
    print(f"  changed picks: {n_changed:,} ({n_changed/len(df)*100:.1f}%)")
    print(f"  baseline   : acc={baseline['acc']:.4f}  roi={baseline['roi']:+.4f}")
    print(f"  with rule  : acc={s1['acc']:.4f}  roi={s1['roi']:+.4f}")
    print(f"  delta      : acc {(s1['acc']-baseline['acc'])*100:+.2f}pp  "
          f"roi {(s1['roi']-baseline['roi'])*100:+.2f}pp")
    rule1_ok = s1["acc"] > baseline["acc"] and s1["roi"] > baseline["roi"]
    print(f"  VERDICT    : {'✅ ADOPT' if rule1_ok else '❌ SKIP'}")

    # Just on July subset
    july = df[df["month"] == 7]
    july_base = stats_for(july["model_home"], july)
    july_pick = july["model_home"].copy()
    july_flip = july["model_picks_fav"]
    july_pick.loc[july_flip] = ~july_pick.loc[july_flip]
    july_with = stats_for(july_pick, july)
    print(f"  [july only]  n={len(july):,}  base acc={july_base['acc']:.4f} "
          f"roi={july_base['roi']:+.4f}  →  with rule acc={july_with['acc']:.4f} "
          f"roi={july_with['roi']:+.4f}")

    # ───────────────────────────────────────────────────────────────────
    # Rule 2 — Fav decimal in [1.60, 1.70): flip to dog
    # ───────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("RULE 2 — Fav @ [1.60, 1.70): flip when model picks favorite")
    print("=" * 80)
    rule2_flip = (df["fav_dec"] >= 1.60) & (df["fav_dec"] < 1.70) & df["model_picks_fav"]
    pick2 = df["model_home"].copy()
    pick2.loc[rule2_flip] = ~pick2.loc[rule2_flip]
    s2 = stats_for(pick2, df)
    n_changed = rule2_flip.sum()
    print(f"  changed picks: {n_changed:,} ({n_changed/len(df)*100:.1f}%)")
    print(f"  baseline   : acc={baseline['acc']:.4f}  roi={baseline['roi']:+.4f}")
    print(f"  with rule  : acc={s2['acc']:.4f}  roi={s2['roi']:+.4f}")
    print(f"  delta      : acc {(s2['acc']-baseline['acc'])*100:+.2f}pp  "
          f"roi {(s2['roi']-baseline['roi'])*100:+.2f}pp")
    rule2_ok = s2["acc"] > baseline["acc"] and s2["roi"] > baseline["roi"]
    print(f"  VERDICT    : {'✅ ADOPT' if rule2_ok else '❌ SKIP'}")

    # Just on subset
    sub = df[(df["fav_dec"] >= 1.60) & (df["fav_dec"] < 1.70)]
    sub_base = stats_for(sub["model_home"], sub)
    sub_pick = sub["model_home"].copy()
    sub_flip = sub["model_picks_fav"]
    sub_pick.loc[sub_flip] = ~sub_pick.loc[sub_flip]
    sub_with = stats_for(sub_pick, sub)
    print(f"  [subset]  n={len(sub):,}  base acc={sub_base['acc']:.4f} "
          f"roi={sub_base['roi']:+.4f}  →  with rule acc={sub_with['acc']:.4f} "
          f"roi={sub_with['roi']:+.4f}")

    # ───────────────────────────────────────────────────────────────────
    # Rule 3 — Team fade (ATL/MIN/BAL @ home → flip pick)
    # ───────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("RULE 3 — Fade ATL/MIN/BAL at home (flip if model picks them)")
    print("=" * 80)
    fade_teams = {"ATL", "MIN", "BAL"}
    rule3_flip = (df["home_team_abbrev"].isin(fade_teams) & df["model_home"])
    pick3 = df["model_home"].copy()
    pick3.loc[rule3_flip] = ~pick3.loc[rule3_flip]
    s3 = stats_for(pick3, df)
    n_changed = rule3_flip.sum()
    print(f"  changed picks: {n_changed:,} ({n_changed/len(df)*100:.1f}%)")
    print(f"  baseline   : acc={baseline['acc']:.4f}  roi={baseline['roi']:+.4f}")
    print(f"  with rule  : acc={s3['acc']:.4f}  roi={s3['roi']:+.4f}")
    print(f"  delta      : acc {(s3['acc']-baseline['acc'])*100:+.2f}pp  "
          f"roi {(s3['roi']-baseline['roi'])*100:+.2f}pp")
    rule3_ok = s3["acc"] > baseline["acc"] and s3["roi"] > baseline["roi"]
    print(f"  VERDICT    : {'✅ ADOPT' if rule3_ok else '❌ SKIP'}")

    # ───────────────────────────────────────────────────────────────────
    # Bonus: Heavy-dog model boost (model picks dog at decimal 3.00-3.50)
    # ───────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("BONUS — Heavy dog confidence: when model picks dog @ [3.00, 3.50)")
    print("        does it actually win more than baseline? (no flip — just check)")
    print("=" * 80)
    model_picks_dog = ~df["model_picks_fav"]
    pick_dec = np.where(df["model_home"], df["dec_home"], df["dec_away"])
    heavy_dog_mask = model_picks_dog & (pick_dec >= 3.00) & (pick_dec < 3.50)
    sub = df[heavy_dog_mask]
    if len(sub) > 50:
        won = sub["home_win"] == 1
        correct = np.where(sub["model_home"], won, ~won)
        dec = np.where(sub["model_home"], sub["dec_home"], sub["dec_away"])
        pnl = np.where(correct, dec - 1, -1.0)
        print(f"  n={len(sub):,}  acc={correct.mean():.4f}  roi={pnl.mean():+.4f}")
        print(f"  vs baseline ROI {baseline['roi']:+.4f} → "
              f"this slice {pnl.mean()-baseline['roi']:+.4f} better")

    # ───────────────────────────────────────────────────────────────────
    # COMBINED: apply all adopted rules together
    # ───────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("COMBINED — Apply ALL rules with positive verdict together")
    print("=" * 80)
    pick_all = df["model_home"].copy()
    flips = pd.Series(False, index=df.index)
    if rule1_ok:
        flips |= rule1_flip
    if rule2_ok:
        flips |= rule2_flip
    if rule3_ok:
        flips |= rule3_flip
    pick_all.loc[flips] = ~pick_all.loc[flips]
    sc = stats_for(pick_all, df)
    print(f"  total games flipped: {flips.sum():,} ({flips.sum()/len(df)*100:.1f}%)")
    print(f"  baseline : acc={baseline['acc']:.4f}  roi={baseline['roi']:+.4f}")
    print(f"  combined : acc={sc['acc']:.4f}  roi={sc['roi']:+.4f}")
    print(f"  delta    : acc {(sc['acc']-baseline['acc'])*100:+.2f}pp  "
          f"roi {(sc['roi']-baseline['roi'])*100:+.2f}pp")

    return {
        "rule1_ok": rule1_ok, "rule2_ok": rule2_ok, "rule3_ok": rule3_ok,
        "rule1_flip_mask": rule1_flip, "rule2_flip_mask": rule2_flip,
        "rule3_flip_mask": rule3_flip,
    }


if __name__ == "__main__":
    main()
