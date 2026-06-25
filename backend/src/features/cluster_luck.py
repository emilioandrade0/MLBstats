"""Cluster luck features — BABIP and LOB% regression signals.

Two well-documented sabermetric signals that capture LUCK vs SKILL:

  BABIP (Batting Average on Balls In Play):
      (H - HR) / (AB - K - HR + SF)
  League BABIP ~ .300. Teams above .310 or below .280 are getting luck/unluck
  on balls dropping for hits, and tend to REGRESS to league mean.

  LOB% (Left On Base %):
      (H + BB + HBP - R) / (H + BB + HBP - 1.4*HR)
  League LOB% ~ .720. Teams above .780 are stranding runners better than skill
  predicts (lucky); below .660 are unlucky in cluster-hitting. Both regress.

We compute both for OFFENSE (their batters) and DEFENSE (their pitchers).
Then we expose:
  - babip_off_l30, babip_def_l30
  - lob_off_l30,   lob_def_l30
  - babip_off_dev_l30 = babip_off_l30 - league_mean  (positive = lucky)
  - lob_off_dev_l30   = lob_off_l30 - league_mean
  ... and same _def variants

Output: data/processed/features_cluster_luck.parquet
Key: (game_pk, team)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED

# League means (approximate, used as anchor for deviation)
LEAGUE_BABIP = 0.295
LEAGUE_LOB = 0.720


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def build() -> Path:
    tb = pd.read_parquet(PROCESSED / "team_box.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet", columns=[
        "game_pk", "game_date", "season", "game_type",
        "home_team_abbrev", "away_team_abbrev",
    ])
    g = games[games["game_type"] == "R"].copy()
    g["game_date"] = pd.to_datetime(g["game_date"])

    # Attach team and game_date to team_box rows
    g_short = g[["game_pk", "game_date", "season", "home_team_abbrev", "away_team_abbrev"]]
    tb = tb.merge(g_short, on="game_pk", how="inner")
    tb["team"] = np.where(tb["side"] == "home",
                          tb["home_team_abbrev"], tb["away_team_abbrev"])
    # Numeric versions of needed columns
    for c in ["bat_hits", "bat_atBats", "bat_homeRuns", "bat_strikeOuts",
              "bat_sacFlies", "bat_baseOnBalls", "bat_hitByPitch", "bat_runs",
              "pit_hits", "pit_atBats", "pit_homeRuns", "pit_strikeOuts",
              "pit_baseOnBalls", "pit_hitByPitch", "pit_runs"]:
        if c in tb.columns:
            tb[c] = _safe_num(tb[c])

    # Per-game numerators/denominators (sum-aggregable for rolling)
    # BABIP offense numerator & denominator
    tb["bab_off_num"] = tb["bat_hits"] - tb["bat_homeRuns"]
    tb["bab_off_den"] = tb["bat_atBats"] - tb["bat_strikeOuts"] - tb["bat_homeRuns"] + tb["bat_sacFlies"]
    # LOB% offense
    on_base = tb["bat_hits"] + tb["bat_baseOnBalls"] + tb["bat_hitByPitch"]
    tb["lob_off_num"] = on_base - tb["bat_runs"]
    tb["lob_off_den"] = on_base - 1.4 * tb["bat_homeRuns"]
    # BABIP defense (against)
    tb["bab_def_num"] = tb["pit_hits"] - tb["pit_homeRuns"]
    tb["bab_def_den"] = tb["pit_atBats"] - tb["pit_strikeOuts"] - tb["pit_homeRuns"]
    # LOB% defense
    on_base_d = tb["pit_hits"] + tb["pit_baseOnBalls"] + tb["pit_hitByPitch"]
    tb["lob_def_num"] = on_base_d - tb["pit_runs"]
    tb["lob_def_den"] = on_base_d - 1.4 * tb["pit_homeRuns"]

    df = tb.sort_values(["team", "season", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)

    # Rolling LAGGED sums per (team, season)
    g_ts = df.groupby(["team", "season"], sort=False)
    def rsum(col, window, min_p=5):
        return g_ts[col].apply(
            lambda s: s.shift(1).rolling(window, min_periods=min_p).sum()
        ).reset_index(level=[0, 1], drop=True)

    for stat in ["bab_off_num", "bab_off_den", "lob_off_num", "lob_off_den",
                  "bab_def_num", "bab_def_den", "lob_def_num", "lob_def_den"]:
        df[f"{stat}_l30"] = rsum(stat, 30)

    # Compute ratios
    df["babip_off_l30"] = df["bab_off_num_l30"] / df["bab_off_den_l30"].replace(0, np.nan)
    df["lob_off_l30"]   = df["lob_off_num_l30"] / df["lob_off_den_l30"].replace(0, np.nan)
    df["babip_def_l30"] = df["bab_def_num_l30"] / df["bab_def_den_l30"].replace(0, np.nan)
    df["lob_def_l30"]   = df["lob_def_num_l30"] / df["lob_def_den_l30"].replace(0, np.nan)

    # Deviation from league mean — positive = lucky
    df["babip_off_dev"] = df["babip_off_l30"] - LEAGUE_BABIP
    df["lob_off_dev"]   = df["lob_off_l30"]   - LEAGUE_LOB
    df["babip_def_dev"] = df["babip_def_l30"] - LEAGUE_BABIP
    df["lob_def_dev"]   = df["lob_def_l30"]   - LEAGUE_LOB

    keep = [
        "game_pk", "team",
        "babip_off_l30", "lob_off_l30", "babip_def_l30", "lob_def_l30",
        "babip_off_dev", "lob_off_dev", "babip_def_dev", "lob_def_dev",
    ]
    out = df[keep].copy()
    p = PROCESSED / "features_cluster_luck.parquet"
    out.to_parquet(p, index=False)
    print(f"wrote {p} ({len(out):,} rows)")
    print(f"  median BABIP offense L30: {out['babip_off_l30'].median():.3f} (league ~{LEAGUE_BABIP})")
    print(f"  median LOB%  offense L30: {out['lob_off_l30'].median():.3f}  (league ~{LEAGUE_LOB})")
    print(f"  median BABIP defense L30: {out['babip_def_l30'].median():.3f}")
    print(f"  median LOB%  defense L30: {out['lob_def_l30'].median():.3f}")
    # Distribution of luck
    lucky_off = (out["babip_off_dev"] > 0.015).sum()
    unlucky_off = (out["babip_off_dev"] < -0.015).sum()
    print(f"  BABIP-off > +.015 (lucky):   {lucky_off:,}")
    print(f"  BABIP-off < -.015 (unlucky): {unlucky_off:,}")
    return p


if __name__ == "__main__":
    build()
