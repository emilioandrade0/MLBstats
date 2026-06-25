"""Tanking / late-season features.

Error analysis found that August/September games and bad teams (CWS 2024, etc.)
account for many of the model's blown picks. The rolling L30 features don't
capture WHY a team is bad — sometimes they're just bad, sometimes they
deliberately tanked, sometimes they got hit by injuries and traded stars away.

Features computed per (game_pk, team):
  - wins_to_date              cumulative wins this season BEFORE this game
  - games_played_to_date      cumulative games played
  - win_pct_to_date           wins / games (with sensible default early in season)
  - is_below_500              1 if win_pct < .500 and >= 30 games played
  - is_below_400              1 if win_pct < .400 (true tanking signal)
  - is_post_trade_deadline    1 if game_date > Jul 31
  - is_september_callups      1 if game_date >= Sep 1
  - days_until_oct_1          calendar countdown
  - is_tanking                1 if below_400 AND post_trade_deadline

Output: data/processed/features_tanking.parquet
Key: (game_pk, team)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


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
    g["away_won"] = 1 - g["home_won"]

    # Long format: one row per (team, game)
    home_rows = g[["game_pk", "game_date", "season", "home_team_abbrev", "home_won"]].copy()
    home_rows.columns = ["game_pk", "game_date", "season", "team", "won"]
    away_rows = g[["game_pk", "game_date", "season", "away_team_abbrev", "away_won"]].copy()
    away_rows.columns = ["game_pk", "game_date", "season", "team", "won"]
    df = pd.concat([home_rows, away_rows], ignore_index=True)
    df = df.sort_values(["team", "season", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)

    # Cumulative season W-L — LAGGED (does not include current game)
    gts = df.groupby(["team", "season"], sort=False)
    df["games_played_to_date"] = gts.cumcount()
    df["wins_to_date"] = gts["won"].cumsum().shift(1).fillna(0)
    # First game of each (team, season) gets 0 wins / 0 games — that shift cascades, fix:
    # We need wins_to_date to reset within the group. shift(1) without group-aware fillna leaks.
    df["wins_to_date"] = gts["won"].apply(
        lambda s: s.shift(1).fillna(0).cumsum()
    ).reset_index(level=[0, 1], drop=True)

    safe_games = df["games_played_to_date"].replace(0, np.nan)
    df["win_pct_to_date"] = (df["wins_to_date"] / safe_games).fillna(0.5)

    # Calendar features
    df["is_post_trade_deadline"] = (df["game_date"] >= pd.Timestamp(
        df["season"].astype(int).astype(str) + "-08-01"
    ) if False else ((df["game_date"].dt.month > 7) |
                       ((df["game_date"].dt.month == 7) & (df["game_date"].dt.day >= 31)))).astype(int)
    df["is_september_callups"] = (df["game_date"].dt.month >= 9).astype(int)
    # Days until end of regular season (rough: Oct 1)
    df["days_until_oct_1"] = (
        pd.to_datetime(df["season"].astype(int).astype(str) + "-10-01") - df["game_date"]
    ).dt.days.clip(lower=0, upper=200)

    # Tanking flags — meaningful only after ~30 games sample
    enough = df["games_played_to_date"] >= 30
    df["is_below_500"] = (enough & (df["win_pct_to_date"] < 0.500)).astype(int)
    df["is_below_400"] = (enough & (df["win_pct_to_date"] < 0.400)).astype(int)
    df["is_tanking"] = (df["is_below_400"] & df["is_post_trade_deadline"]).astype(int)

    keep = [
        "game_pk", "team",
        "games_played_to_date", "wins_to_date", "win_pct_to_date",
        "is_below_500", "is_below_400", "is_tanking",
        "is_post_trade_deadline", "is_september_callups", "days_until_oct_1",
    ]
    out = df[keep].copy()
    p = PROCESSED / "features_tanking.parquet"
    out.to_parquet(p, index=False)
    print(f"wrote {p} ({len(out):,} rows)")
    print(f"  rows tagged is_below_500: {out['is_below_500'].sum():,}")
    print(f"  rows tagged is_below_400: {out['is_below_400'].sum():,}")
    print(f"  rows tagged is_tanking:   {out['is_tanking'].sum():,}")
    return p


if __name__ == "__main__":
    build()
