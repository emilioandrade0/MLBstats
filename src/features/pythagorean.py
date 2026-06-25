"""Pythagorean expectation features.

Bill James's Pythagorean expectation predicts a team's "true" win % from
runs scored and allowed:
    pyth_wpct = RS^x / (RS^x + RA^x)
with x = 1.83 (Pythagenpat, the MLB-specific exponent).

Key insight: actual wins are noisier than run differential. A team that wins
10/15 games but has been outscored has been LUCKY and will regress down. A
team with positive run diff but only 7/15 wins has been UNLUCKY.

Features per (game_pk, team):
  - pyth_wpct_l30        expected win % from L30 run diff
  - pyth_minus_actual_l30  positive = lucky / negative = unlucky
  - pyth_run_diff_l30    raw RS - RA over L30 (per game)

Both LAGGED (using games BEFORE current).

Output: data/processed/features_pythagorean.parquet
Key: (game_pk, team)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED

PYTHAG_EXP = 1.83  # Pythagenpat exponent (more accurate than 2.0 for MLB)


def build() -> Path:
    games = pd.read_parquet(PROCESSED / "games.parquet", columns=[
        "game_pk", "game_date", "season", "game_type",
        "home_team_abbrev", "away_team_abbrev",
        "home_score", "away_score",
    ])
    g = games[games["game_type"] == "R"].copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g.dropna(subset=["home_score", "away_score", "home_team_abbrev", "away_team_abbrev"])
    g["home_won"] = (g["home_score"] > g["away_score"]).astype(int)

    # Long format: one row per (team, game) with runs_scored / runs_allowed
    home = g[["game_pk", "game_date", "season", "home_team_abbrev",
              "home_score", "away_score", "home_won"]].copy()
    home.columns = ["game_pk", "game_date", "season", "team",
                    "runs_scored", "runs_allowed", "won"]
    away = g[["game_pk", "game_date", "season", "away_team_abbrev",
              "away_score", "home_score", "home_won"]].copy()
    away["away_won"] = 1 - away["home_won"]
    away = away[["game_pk", "game_date", "season", "away_team_abbrev",
                  "away_score", "home_score", "away_won"]]
    away.columns = ["game_pk", "game_date", "season", "team",
                    "runs_scored", "runs_allowed", "won"]
    df = pd.concat([home, away], ignore_index=True)
    df = df.sort_values(["team", "season", "game_date", "game_pk"], kind="mergesort") \
            .reset_index(drop=True)

    g_team_season = df.groupby(["team", "season"], sort=False)

    # Rolling LAGGED stats — strictly using games BEFORE the current one
    def rolling_shift_sum(s, window, min_periods=5):
        return s.shift(1).rolling(window, min_periods=min_periods).sum()

    df["rs_l30"] = g_team_season["runs_scored"].apply(
        lambda s: rolling_shift_sum(s, 30)
    ).reset_index(level=[0, 1], drop=True)
    df["ra_l30"] = g_team_season["runs_allowed"].apply(
        lambda s: rolling_shift_sum(s, 30)
    ).reset_index(level=[0, 1], drop=True)
    df["wins_l30"] = g_team_season["won"].apply(
        lambda s: rolling_shift_sum(s, 30)
    ).reset_index(level=[0, 1], drop=True)
    df["games_l30"] = g_team_season["won"].apply(
        lambda s: s.shift(1).rolling(30, min_periods=5).count()
    ).reset_index(level=[0, 1], drop=True)

    # Pythagorean expected win %
    rs_safe = df["rs_l30"].clip(lower=0.1)
    ra_safe = df["ra_l30"].clip(lower=0.1)
    df["pyth_wpct_l30"] = (rs_safe ** PYTHAG_EXP) / (
        (rs_safe ** PYTHAG_EXP) + (ra_safe ** PYTHAG_EXP)
    )
    # Actual win pct
    df["actual_wpct_l30"] = df["wins_l30"] / df["games_l30"].replace(0, np.nan)
    # Luck deviation: positive = lucky (won more than runs suggest)
    df["pyth_minus_actual_l30"] = df["actual_wpct_l30"] - df["pyth_wpct_l30"]
    # Run differential per game (less correlated with wpct)
    df["pyth_run_diff_pg_l30"] = (df["rs_l30"] - df["ra_l30"]) / df["games_l30"].replace(0, np.nan)

    # Same for L10 (more recent regression signal)
    df["rs_l10"] = g_team_season["runs_scored"].apply(
        lambda s: rolling_shift_sum(s, 10, 3)
    ).reset_index(level=[0, 1], drop=True)
    df["ra_l10"] = g_team_season["runs_allowed"].apply(
        lambda s: rolling_shift_sum(s, 10, 3)
    ).reset_index(level=[0, 1], drop=True)
    rs10_safe = df["rs_l10"].clip(lower=0.1)
    ra10_safe = df["ra_l10"].clip(lower=0.1)
    df["pyth_wpct_l10"] = (rs10_safe ** PYTHAG_EXP) / (
        (rs10_safe ** PYTHAG_EXP) + (ra10_safe ** PYTHAG_EXP)
    )

    keep = [
        "game_pk", "team",
        "pyth_wpct_l30", "pyth_wpct_l10",
        "pyth_minus_actual_l30", "pyth_run_diff_pg_l30",
    ]
    out = df[keep].copy()
    p = PROCESSED / "features_pythagorean.parquet"
    out.to_parquet(p, index=False)
    print(f"wrote {p} ({len(out):,} rows)")
    print(f"  median pyth_wpct_l30:           {out['pyth_wpct_l30'].median():.3f}")
    print(f"  pyth_minus_actual_l30 stats:    "
          f"mean={out['pyth_minus_actual_l30'].mean():.4f} "
          f"std={out['pyth_minus_actual_l30'].std():.4f}")
    # Distribution of luck
    lucky = (out["pyth_minus_actual_l30"] > 0.07).sum()
    unlucky = (out["pyth_minus_actual_l30"] < -0.07).sum()
    print(f"  extremely lucky team-games (>+.070): {lucky}")
    print(f"  extremely unlucky team-games (<-.070): {unlucky}")
    return p


if __name__ == "__main__":
    build()
