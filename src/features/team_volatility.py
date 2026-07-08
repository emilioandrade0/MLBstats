"""Team volatility features — std of runs scored/allowed L10.

Existing model has L10 *means* (runs_scored_l10, runs_allowed_l10). Two teams
can have the same mean but very different volatility:
  Team A: 3,3,3,3,3 runs scored (mean=3, std=0)
  Team B: 0,0,8,4,3 runs scored (mean=3, std=3.2)
Team B is much more unpredictable. Volatile offenses tend to over-perform
in close games (variance helps the underdog), under-perform when fav.

Output: data/processed/features_team_volatility.parquet
  game_pk | side | runs_scored_std_l10 | runs_allowed_std_l10 | run_diff_std_l10

Run:
  python -m src.features.team_volatility
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_team_volatility.parquet"

WINDOW = 10
MIN_P = 4


def build() -> Path:
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "game_type",
                 "home_team_abbrev", "away_team_abbrev",
                 "home_score", "away_score"],
    )
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games[games["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    games = games.dropna(subset=["home_score", "away_score"])

    # Build per-team timeline (one row per team per game)
    rows = []
    for _, r in games.iterrows():
        rows.append({"game_pk": r["game_pk"], "game_date": r["game_date"],
                      "team": r["home_team_abbrev"],
                      "runs_scored": r["home_score"],
                      "runs_allowed": r["away_score"], "side": "home"})
        rows.append({"game_pk": r["game_pk"], "game_date": r["game_date"],
                      "team": r["away_team_abbrev"],
                      "runs_scored": r["away_score"],
                      "runs_allowed": r["home_score"], "side": "away"})

    tdf = pd.DataFrame(rows)
    tdf = tdf.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)
    tdf["run_diff"] = tdf["runs_scored"] - tdf["runs_allowed"]

    g = tdf.groupby("team", sort=False)
    def _roll_std(s):
        return s.shift(1).rolling(window=WINDOW, min_periods=MIN_P).std()

    tdf["runs_scored_std_l10"]  = g["runs_scored"].transform(_roll_std)
    tdf["runs_allowed_std_l10"] = g["runs_allowed"].transform(_roll_std)
    tdf["run_diff_std_l10"]     = g["run_diff"].transform(_roll_std)

    out = tdf[["game_pk", "side",
                "runs_scored_std_l10", "runs_allowed_std_l10", "run_diff_std_l10"]].copy()
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} rows)")
    nan_frac = out.drop(columns=["game_pk","side"]).isna().mean()
    print("NaN fractions:")
    print(nan_frac.round(3).to_string())
    print()
    print("Sample stats:")
    print(out.drop(columns=["game_pk","side"]).describe().round(3).to_string())
    return OUT


if __name__ == "__main__":
    build()
