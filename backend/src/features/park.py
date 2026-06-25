"""Park factors — multi-season rolling, lagged.

For each venue, park_runs_factor at game G = (avg runs/game at venue in prior season)
                                            / (avg runs/game league-wide in prior season)

Uses prior-season totals so today's game cannot leak into its own park factor.

Output: data/processed/features_park.parquet
Key: venue_id, season
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..normalize.paths import PROCESSED


def build() -> Path:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    g = games[games["game_type"] == "R"].copy()
    g["total_runs"] = g["home_score"] + g["away_score"]
    g["total_hr_proxy"] = g["total_runs"]  # placeholder — would refine with team_box HRs

    # Per (season, venue) averages
    venue_year = g.groupby(["season", "venue_id"]).agg(
        venue_runs_per_game=("total_runs", "mean"),
        venue_games=("total_runs", "size"),
    ).reset_index()
    league_year = g.groupby("season").agg(
        league_runs_per_game=("total_runs", "mean"),
    ).reset_index()
    yr = venue_year.merge(league_year, on="season")
    yr["park_runs_factor_curr"] = yr["venue_runs_per_game"] / yr["league_runs_per_game"]

    # Use PRIOR season as the factor for this season (no leakage).
    yr["season_for_use"] = yr["season"] + 1
    park = yr[["season_for_use", "venue_id", "park_runs_factor_curr"]].rename(
        columns={"season_for_use": "season", "park_runs_factor_curr": "park_runs_factor"}
    )
    out = PROCESSED / "features_park.parquet"
    park.to_parquet(out, index=False)
    print(f"wrote {out} ({len(park):,} rows)")
    return out


if __name__ == "__main__":
    build()
