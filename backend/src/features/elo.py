"""ELO ratings — universal team-strength baseline.

For each game in chronological order:
  1. Read each team's pre-game ELO (default 1500).
  2. Expected win prob = 1 / (1 + 10**((opp_elo - my_elo - home_adv) / 400))
  3. After the game, update with K * (actual - expected).

The PRE-game ELO is what we save — no leakage. Default params K=16, home_adv=35
match what worked in the previous MLB project.

Output: data/processed/features_elo.parquet
Key: game_pk
Columns: home_elo_pre, away_elo_pre, elo_diff_pre, expected_home_win_elo
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..normalize.paths import PROCESSED

K = 16.0
HOME_ADV = 35.0
ELO_INIT = 1500.0


def build() -> Path:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    # Use only completed games for ELO updates; we still emit pre-game ELO for
    # scheduled future games (their ELO comes from history up to that date).
    g = games.dropna(subset=["game_date", "home_team_abbrev", "away_team_abbrev"]).copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    # Chronological order with stable tiebreak
    g = g.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)

    elo: dict[str, float] = {}
    home_pre, away_pre, exp_home = [], [], []

    for _, r in g.iterrows():
        h = r["home_team_abbrev"]
        a = r["away_team_abbrev"]
        he = elo.get(h, ELO_INIT)
        ae = elo.get(a, ELO_INIT)
        diff = ae - (he + HOME_ADV)
        eh = 1.0 / (1.0 + 10 ** (diff / 400))

        home_pre.append(he)
        away_pre.append(ae)
        exp_home.append(eh)

        # Update only if the game was played (we have scores).
        hs = r.get("home_score")
        as_ = r.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            actual_home = 1.0 if hs > as_ else 0.0
            elo[h] = he + K * (actual_home - eh)
            elo[a] = ae + K * ((1 - actual_home) - (1 - eh))

    out = pd.DataFrame({
        "game_pk": g["game_pk"].values,
        "home_elo_pre": home_pre,
        "away_elo_pre": away_pre,
        "elo_diff_pre": [h - a for h, a in zip(home_pre, away_pre)],
        "expected_home_win_elo": exp_home,
    })
    p = PROCESSED / "features_elo.parquet"
    out.to_parquet(p, index=False)
    print(f"wrote {p} ({len(out):,} rows)")
    print(f"  final ELOs (top 10):")
    final = pd.Series(elo).sort_values(ascending=False).head(10)
    print(final.to_string())
    return p


if __name__ == "__main__":
    build()
