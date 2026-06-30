"""Lineup recent form features — last 15 games per starter, aggregated per lineup.

The existing `features_lineup.parquet` uses CAREER xWOBA vs SP-hand for matchup
analysis. This module fills the recency gap: it computes each starter's L15
batting metrics from prior games (shift(1) — no lookahead) and aggregates
across the top-4 spots in the order.

Hypothesis: top-of-order batters in hot/cold streaks add signal not captured
by team-level offensive metrics (which mix in bench, pitchers, late-game subs).

Features per (game_pk, side):
  lineup_top4_recent_ops_l15     — avg OPS of starters in batting_order 100-400
  lineup_top4_recent_iso_l15     — avg ISO (slg - avg)
  lineup_top4_recent_k_pct_l15   — avg K rate
  lineup_top4_recent_bb_pct_l15  — avg BB rate
  lineup_hot_bats_count_l15      — # starters with recent OPS > 0.800
  lineup_cold_bats_count_l15     — # starters with recent OPS < 0.600
  lineup_recent_n_with_data      — coverage / data quality

Output: data/processed/features_lineup_recent.parquet

Run:
  python -m src.features.lineup_recent
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_lineup_recent.parquet"

WINDOW = 15
MIN_PA = 20   # need at least 20 PA in L15 to count


def _player_recent_form() -> pd.DataFrame:
    """Per (player, game_pk) the L15 batting stats from PRIOR games."""
    pb = pd.read_parquet(
        PROCESSED / "player_box.parquet",
        columns=["game_pk", "side", "player_id", "batting_order",
                 "started_batting",
                 "bat_atBats", "bat_hits", "bat_baseOnBalls",
                 "bat_hitByPitch", "bat_strikeOuts", "bat_plateAppearances",
                 "bat_totalBases", "bat_sacFlies"],
    )
    # Need game dates for chronological ordering
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date"],
    )
    games["game_date"] = pd.to_datetime(games["game_date"])
    pb = pb.merge(games, on="game_pk", how="left")
    pb = pb.dropna(subset=["game_date"])

    # Keep only players who actually batted (had a PA in the game)
    pb = pb[pb["bat_plateAppearances"].fillna(0) > 0].copy()
    pb["player_id"] = pb["player_id"].astype("Int64")
    # batting_order is stored as string ("100", "200", ...). Convert to int.
    pb["batting_order"] = pd.to_numeric(pb["batting_order"], errors="coerce").astype("Int64")

    # Sort chronologically per player
    pb = pb.sort_values(["player_id", "game_date", "game_pk"]).reset_index(drop=True)

    # Fill numeric NaN with 0 so rolling sums work
    num_cols = ["bat_atBats", "bat_hits", "bat_baseOnBalls", "bat_hitByPitch",
                "bat_strikeOuts", "bat_plateAppearances", "bat_totalBases", "bat_sacFlies"]
    for c in num_cols:
        pb[c] = pd.to_numeric(pb[c], errors="coerce").fillna(0)

    # Per-player rolling SUMS over the last WINDOW games, shift(1) to use only PRIOR games
    grp = pb.groupby("player_id", sort=False)
    def _roll_sum(s):
        return s.shift(1).rolling(WINDOW, min_periods=3).sum()

    pb["pa_l15"]   = grp["bat_plateAppearances"].transform(_roll_sum)
    pb["ab_l15"]   = grp["bat_atBats"].transform(_roll_sum)
    pb["h_l15"]    = grp["bat_hits"].transform(_roll_sum)
    pb["bb_l15"]   = grp["bat_baseOnBalls"].transform(_roll_sum)
    pb["hbp_l15"]  = grp["bat_hitByPitch"].transform(_roll_sum)
    pb["k_l15"]    = grp["bat_strikeOuts"].transform(_roll_sum)
    pb["tb_l15"]   = grp["bat_totalBases"].transform(_roll_sum)
    pb["sf_l15"]   = grp["bat_sacFlies"].transform(_roll_sum)

    # Derive rates only when sample is meaningful
    pa = pb["pa_l15"]
    ab = pb["ab_l15"]
    enough = pa >= MIN_PA

    pb["avg_l15"]    = np.where(enough & (ab > 0), pb["h_l15"] / ab.replace(0, np.nan), np.nan)
    obp_denom = ab + pb["bb_l15"] + pb["hbp_l15"] + pb["sf_l15"]
    pb["obp_l15"]    = np.where(enough & (obp_denom > 0),
                                 (pb["h_l15"] + pb["bb_l15"] + pb["hbp_l15"]) / obp_denom.replace(0, np.nan),
                                 np.nan)
    pb["slg_l15"]    = np.where(enough & (ab > 0), pb["tb_l15"] / ab.replace(0, np.nan), np.nan)
    pb["ops_l15"]    = pb["obp_l15"] + pb["slg_l15"]
    pb["iso_l15"]    = pb["slg_l15"] - pb["avg_l15"]
    pb["k_pct_l15"]  = np.where(enough & (pa > 0), pb["k_l15"] / pa.replace(0, np.nan), np.nan)
    pb["bb_pct_l15"] = np.where(enough & (pa > 0), pb["bb_l15"] / pa.replace(0, np.nan), np.nan)

    return pb[["game_pk", "side", "player_id", "batting_order", "started_batting",
                "ops_l15", "iso_l15", "k_pct_l15", "bb_pct_l15", "avg_l15"]]


def build() -> Path:
    print(f"computing player L{WINDOW} recent form...")
    p = _player_recent_form()
    # Top-4 starters: batting_order in {100, 200, 300, 400} AND started_batting
    top = p[(p["started_batting"] == True) & (p["batting_order"].isin([100, 200, 300, 400]))].copy()
    print(f"  {len(top):,} top-4 starter rows across {top['game_pk'].nunique():,} games")

    # Aggregate per (game_pk, side)
    grp = top.groupby(["game_pk", "side"])

    def _agg(g: pd.DataFrame) -> pd.Series:
        ops = g["ops_l15"].dropna()
        return pd.Series({
            "lineup_top4_recent_ops_l15":    float(ops.mean())              if len(ops) else np.nan,
            "lineup_top4_recent_iso_l15":    float(g["iso_l15"].mean())     if g["iso_l15"].notna().any() else np.nan,
            "lineup_top4_recent_k_pct_l15":  float(g["k_pct_l15"].mean())   if g["k_pct_l15"].notna().any() else np.nan,
            "lineup_top4_recent_bb_pct_l15": float(g["bb_pct_l15"].mean())  if g["bb_pct_l15"].notna().any() else np.nan,
            "lineup_hot_bats_count_l15":     int((ops > 0.800).sum())       if len(ops) else 0,
            "lineup_cold_bats_count_l15":    int((ops < 0.600).sum())       if len(ops) else 0,
            "lineup_recent_n_with_data":     int(len(ops)),
        })

    out = grp.apply(_agg, include_groups=False).reset_index()
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} rows)")
    nan_frac = out.drop(columns=["game_pk", "side"]).isna().mean()
    print("NaN fractions:")
    print(nan_frac.round(3).to_string())
    print()
    print("Sample means:")
    print(out.drop(columns=["game_pk", "side"]).mean(numeric_only=True).round(3).to_string())
    return OUT


if __name__ == "__main__":
    build()
