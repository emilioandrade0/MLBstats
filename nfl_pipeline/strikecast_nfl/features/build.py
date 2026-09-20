"""Assemble the final training / prediction table.

Joins games + Elo + EPA + market lines into one row per game with:
  - Target columns (only populated for FINAL games): home_win, cover_home, total_over
  - Features (all pre-kickoff safe): elo_diff, epa_diff, market lines, rest days, etc.

Output: data/processed/train.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import PROCESSED


def _rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game (walk-forward safe)."""
    long = pd.concat([
        games[["game_id", "kickoff_utc", "home_team"]].rename(columns={"home_team": "team"}),
        games[["game_id", "kickoff_utc", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True).sort_values(["team", "kickoff_utc"])
    long["prev_kickoff"] = long.groupby("team")["kickoff_utc"].shift(1)
    long["rest_days"] = (long["kickoff_utc"] - long["prev_kickoff"]).dt.total_seconds() / 86400.0
    home_rest = long.merge(games[["game_id", "home_team"]], left_on=["game_id", "team"],
                           right_on=["game_id", "home_team"])[["game_id", "rest_days"]] \
                    .rename(columns={"rest_days": "home_rest"})
    away_rest = long.merge(games[["game_id", "away_team"]], left_on=["game_id", "team"],
                           right_on=["game_id", "away_team"])[["game_id", "rest_days"]] \
                    .rename(columns={"rest_days": "away_rest"})
    return games.merge(home_rest, on="game_id", how="left") \
                .merge(away_rest, on="game_id", how="left")


def build() -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    elo = pd.read_parquet(PROCESSED / "features_elo.parquet")
    epa = pd.read_parquet(PROCESSED / "features_epa.parquet")

    df = games.merge(elo, on="game_id", how="left").merge(epa, on="game_id", how="left")
    df = _rest_days(df)

    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    for col in ["home_epa_off_per_play_l4", "away_epa_off_per_play_l4",
                "home_epa_def_per_play_l4", "away_epa_def_per_play_l4",
                "home_epa_off_per_play_l8", "away_epa_off_per_play_l8",
                "home_success_rate_off_l4", "away_success_rate_off_l4"]:
        if col not in df.columns:
            df[col] = np.nan
    df["epa_off_diff_l4"] = df["home_epa_off_per_play_l4"] - df["away_epa_off_per_play_l4"]
    df["epa_def_diff_l4"] = df["home_epa_def_per_play_l4"] - df["away_epa_def_per_play_l4"]
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)

    # Targets (only for final games)
    is_final = df["status"] == "final"
    df["home_win"] = np.where(is_final, (df["home_score"] > df["away_score"]).astype(float), np.nan)
    df["margin"] = np.where(is_final, df["home_score"] - df["away_score"], np.nan)
    # nflreadpy convention: spread_line > 0 means home team is favored by that many points.
    # Home covers iff home wins by MORE than the spread → margin - spread_line > 0.
    df["cover_home"] = np.where(is_final & df["spread_line"].notna(),
                                (df["margin"] - df["spread_line"] > 0).astype(float), np.nan)
    df["total_over"] = np.where(is_final & df["total_line"].notna(),
                                (df["home_score"] + df["away_score"] > df["total_line"]).astype(float), np.nan)

    path = PROCESSED / "train.parquet"
    df.to_parquet(path, index=False)
    print(f"train.parquet: {len(df):,} rows, {len(df.columns)} cols  "
          f"(with_target={int(is_final.sum()):,})")
    return df


if __name__ == "__main__":
    build()
