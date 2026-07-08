"""Calendar features — slate size and day-of-week per game.

Daily-pattern audit (src/analysis/daily_patterns.py) found:
  - Slate 6-8 games → dogs +4.94pp deviation (strong signal)
  - Wed/Thu → dogs +3pp vs Tuesday (weaker signal)

These are game-level features (both teams share them), so they don't follow
the _h/_a/_diff convention.

Output: data/processed/features_calendar.parquet
  game_pk · slate_size · dow (int 0=Mon..6=Sun) · is_small_slate

Run:
  python -m src.features.calendar
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_calendar.parquet"


def build() -> Path:
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "game_type",
                 "home_team_abbrev", "away_team_abbrev"],
    )
    games["game_date"] = pd.to_datetime(games["game_date"])
    # Regular + playoff games only (exclude exhibitions which inflate slate counts)
    games = games[games["game_type"].isin(["R", "F", "D", "L", "W"])].copy()

    # Slate size = # games on that game_date
    slate = games.groupby(games["game_date"].dt.date).size().rename("slate_size")
    games["_d"] = games["game_date"].dt.date
    games = games.merge(slate, left_on="_d", right_index=True, how="left")
    games["dow"] = games["game_date"].dt.dayofweek.astype("int32")  # 0=Mon, 6=Sun
    games["is_small_slate"] = ((games["slate_size"] >= 6) & (games["slate_size"] <= 8)).astype("int8")
    out = games[["game_pk", "slate_size", "dow", "is_small_slate"]].copy()
    out["slate_size"] = out["slate_size"].astype("int32")
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} rows)")
    print(f"  slate_size:    min={out['slate_size'].min()} median={out['slate_size'].median():.0f} max={out['slate_size'].max()}")
    print(f"  dow distrib:   {out['dow'].value_counts().sort_index().to_dict()}")
    print(f"  small slates:  {out['is_small_slate'].sum():,}  ({out['is_small_slate'].mean()*100:.1f}%)")
    return OUT


if __name__ == "__main__":
    build()
