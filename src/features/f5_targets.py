"""Compute First-5-Innings (F5) outcomes from play-by-play.

For each game, capture the score at the END of inning 5 (after both top and
bottom halves of the 5th). F5 is its own betting market:
  - F5 moneyline: which team is winning through 5 innings (can be a tie)
  - F5 total: runs scored by both teams in innings 1-5
  - F5 spread: half-run line, e.g. home -0.5

Compared to full-game ML:
  - F5 depends almost entirely on STARTING pitchers (bullpen variance removed)
  - Less inherent variance → easier to model
  - Books often less sharp on F5 → more potential edge

Output: data/processed/f5_targets.parquet
Key: game_pk
Cols: f5_away_score, f5_home_score, f5_home_won, f5_total_runs,
      f5_tied (binary — possible in F5, impossible in full game),
      f5_run_diff (home - away)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..normalize.paths import PROCESSED


def build() -> Path:
    plays = pd.read_parquet(
        PROCESSED / "plays.parquet",
        columns=["game_pk", "inning", "is_top", "away_score", "home_score",
                 "at_bat_index"],
    )
    plays = plays.dropna(subset=["inning", "game_pk"]).copy()
    plays["inning"] = plays["inning"].astype(int)

    # Filter to plays through inning 5 (innings 1-5 inclusive)
    through_5 = plays[plays["inning"] <= 5].copy()

    # Last play of inning 5 (or earlier if game ended) per game has the cumulative score
    through_5 = through_5.sort_values(["game_pk", "at_bat_index"], kind="mergesort")
    f5 = through_5.groupby("game_pk").tail(1)[
        ["game_pk", "away_score", "home_score", "inning"]
    ].reset_index(drop=True)
    f5 = f5.rename(columns={
        "away_score": "f5_away_score",
        "home_score": "f5_home_score",
        "inning": "f5_last_inning_reached",
    })

    # Some games may end before completing 5 innings (rain, mercy rule). Drop those.
    f5 = f5[f5["f5_last_inning_reached"] >= 5].drop(columns=["f5_last_inning_reached"])

    f5["f5_total_runs"] = f5["f5_home_score"] + f5["f5_away_score"]
    f5["f5_run_diff"] = f5["f5_home_score"] - f5["f5_away_score"]
    f5["f5_home_won"] = (f5["f5_home_score"] > f5["f5_away_score"]).astype("Int64")
    f5["f5_tied"]    = (f5["f5_home_score"] == f5["f5_away_score"]).astype("Int64")

    out = PROCESSED / "f5_targets.parquet"
    f5.to_parquet(out, index=False)
    print(f"wrote {out} ({len(f5):,} rows)")
    print(f"  F5 home wins:  {f5['f5_home_won'].mean():.3f}")
    print(f"  F5 tied:       {f5['f5_tied'].mean():.3f}")
    print(f"  F5 avg total:  {f5['f5_total_runs'].mean():.2f}")
    print(f"  F5 avg margin: {f5['f5_run_diff'].abs().mean():.2f}")
    return out


if __name__ == "__main__":
    build()
