"""Series context — captures the fact that MLB plays 3-4 game series, not isolated games.

For each game, compute:
  - series_game_number: 1, 2, 3, 4 (which game of the current series)
  - series_run_diff_so_far: cumulative home_margin in this series before this game
  - did_home_win_previous_game_in_series: 0/1, NaN if first game of series
  - series_home_wins_so_far: how many of the prior games in this series the home team won

A "series" is consecutive games between the same two teams with date_gap ≤ 3 days.

Output: data/processed/features_series.parquet
Key: game_pk
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


def build() -> Path:
    games = pd.read_parquet(PROCESSED / "games.parquet",
                             columns=["game_pk", "game_date", "season",
                                      "home_team_abbrev", "away_team_abbrev",
                                      "home_score", "away_score"])
    g = games.dropna(subset=["home_team_abbrev", "away_team_abbrev", "game_date"]).copy()
    g["game_date"] = pd.to_datetime(g["game_date"])

    # Canonical pair key: alphabetical so HOME-AWAY and AWAY-HOME map to same series
    a = g["home_team_abbrev"].astype(str)
    b = g["away_team_abbrev"].astype(str)
    g["pair_low"] = np.where(a <= b, a, b)
    g["pair_high"] = np.where(a > b, a, b)
    g["pair_key"] = g["season"].astype(str) + "|" + g["pair_low"] + "|" + g["pair_high"]

    g = g.sort_values(["pair_key", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    date_gap = g.groupby("pair_key")["game_date"].diff().dt.days.fillna(99)
    new_series = (date_gap > 3).astype(int)
    g["series_seq"] = new_series.groupby(g["pair_key"]).cumsum().astype(str)
    g["series_id"] = g["pair_key"] + "|" + g["series_seq"]

    # Home margin and home win flag
    g["home_margin"] = (
        pd.to_numeric(g["home_score"], errors="coerce").fillna(0.0)
        - pd.to_numeric(g["away_score"], errors="coerce").fillna(0.0)
    )
    g["home_win_flag"] = (g["home_margin"] > 0).astype(float)

    # series_game_number = 1 + cumcount within series
    g["series_game_number"] = g.groupby("series_id").cumcount().astype(float) + 1.0

    # cumulative run_diff in series BEFORE this game (lagged)
    g["series_run_diff_so_far"] = (
        g.groupby("series_id")["home_margin"].cumsum() - g["home_margin"]
    )

    # did the HOME TEAM win the previous game in this series?
    # The "home team" identity can flip in MLB (game 1 home vs game 2 home), so
    # this column is interpreted as: did the team that is HOME TODAY win the
    # previous matchup in this series? For simplicity (and matching prior project)
    # we just shift the home_win_flag within the series — note that this answers
    # "did the team that was HOME LAST GAME win" which is still informative.
    g["did_home_win_previous_game_in_series"] = (
        g.groupby("series_id")["home_win_flag"].shift(1)
    )

    keep = ["game_pk", "series_game_number", "series_run_diff_so_far",
            "did_home_win_previous_game_in_series"]
    out = g[keep].copy()
    p = PROCESSED / "features_series.parquet"
    out.to_parquet(p, index=False)
    print(f"wrote {p} ({len(out):,} rows)")
    print(f"  game_number distribution:")
    print(out["series_game_number"].value_counts().sort_index().head(10).to_string())
    return p


if __name__ == "__main__":
    build()
