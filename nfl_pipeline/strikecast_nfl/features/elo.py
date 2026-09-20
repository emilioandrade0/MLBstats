"""Dynamic Elo ratings — walk-forward safe.

For every game, we emit `home_elo_pre` and `away_elo_pre` — ratings BEFORE
kickoff. After the game we update in-place (Elo K-factor, MOV multiplier a la
FiveThirtyEight). Season resets regress 25% toward the league mean of 1500.

No leakage: the update for game N is applied AFTER we've written the pre-game
features for game N.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..paths import PROCESSED

K = 20.0
HFA = 55.0            # home-field advantage in Elo points (~2.5 pts spread)
SEASON_REGRESS = 0.25
BASE = 1500.0


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _mov_mult(point_diff: int, elo_diff: float) -> float:
    """FiveThirtyEight margin-of-victory multiplier."""
    return float(np.log(max(abs(point_diff), 1) + 1) * (2.2 / ((elo_diff if point_diff > 0 else -elo_diff) * 0.001 + 2.2)))


def build() -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    games = games.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)

    ratings: dict[str, float] = defaultdict(lambda: BASE)
    last_season: dict[str, int] = {}
    rows = []

    for _, g in games.iterrows():
        home, away, season = g["home_team"], g["away_team"], int(g["season"])
        # Season-boundary regression
        for t in (home, away):
            if last_season.get(t) is not None and season != last_season[t]:
                ratings[t] = ratings[t] * (1 - SEASON_REGRESS) + BASE * SEASON_REGRESS
            last_season[t] = season

        home_pre, away_pre = ratings[home], ratings[away]
        rows.append({
            "game_id": g["game_id"],
            "home_elo_pre": home_pre,
            "away_elo_pre": away_pre,
            "elo_diff": (home_pre + HFA) - away_pre,
        })

        # Update only if final
        if pd.notna(g.get("home_score")) and pd.notna(g.get("away_score")):
            hs, as_ = int(g["home_score"]), int(g["away_score"])
            actual = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            exp = _expected(home_pre + HFA, away_pre)
            elo_diff = (home_pre + HFA) - away_pre
            mult = _mov_mult(hs - as_, elo_diff)
            delta = K * mult * (actual - exp)
            ratings[home] = home_pre + delta
            ratings[away] = away_pre - delta

    out = pd.DataFrame(rows)
    path = PROCESSED / "features_elo.parquet"
    out.to_parquet(path, index=False)
    print(f"features_elo.parquet: {len(out):,} rows")
    return out


if __name__ == "__main__":
    build()
