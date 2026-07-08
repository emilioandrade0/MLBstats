"""Burn features — yesterday's "carne al asador" signal per team.

Reads burn_analysis.parquet (which already shift(1)s prev-game state) and
exposes the burn score and component signals at game_pk × side level for
the modeling pipeline.

Output: data/processed/features_burn.parquet
  game_pk | side | burn_score | bp_ip_yest | extra_inn_yest | comeback_yest

Run standalone:
  python -m src.features.burn
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_burn.parquet"


def build() -> Path:
    burn_path = PROCESSED / "burn_analysis.parquet"
    if not burn_path.exists():
        raise FileNotFoundError(
            f"{burn_path} missing — run src.analysis.burn_analysis first"
        )
    burn = pd.read_parquet(burn_path)
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "home_team_abbrev", "away_team_abbrev"],
    )

    # Burn columns (already shift(1) — represent yesterday's state)
    feat_cols = [
        "burn_score",
        "bullpen_ip_yest",
        "bullpen_apps_yest",
        "extra_innings_yest",
        "long_game_yest",
        "used_closer_close_yest",
        "comeback_won_yest",
        "days_rest",
    ]

    keep = burn[["game_pk", "team"] + feat_cols].copy()
    # Convert booleans to floats so LightGBM treats them as numeric
    for c in ["extra_innings_yest", "long_game_yest",
              "used_closer_close_yest", "comeback_won_yest"]:
        keep[c] = keep[c].astype(float)

    # Map (game_pk, team) → side
    home = games[["game_pk", "home_team_abbrev"]].rename(
        columns={"home_team_abbrev": "team"}
    )
    home["side"] = "home"
    away = games[["game_pk", "away_team_abbrev"]].rename(
        columns={"away_team_abbrev": "team"}
    )
    away["side"] = "away"
    side_map = pd.concat([home, away], ignore_index=True)

    out = keep.merge(side_map, on=["game_pk", "team"], how="inner")
    out = out[["game_pk", "side"] + feat_cols].drop_duplicates(["game_pk", "side"])

    # Rename to model-friendly names with side suffix downstream in build.py
    out = out.rename(columns={
        "burn_score":             "burn_score",
        "bullpen_ip_yest":        "burn_bp_ip",
        "bullpen_apps_yest":      "burn_bp_apps",
        "extra_innings_yest":     "burn_extra_inn",
        "long_game_yest":         "burn_long_game",
        "used_closer_close_yest": "burn_closer",
        "comeback_won_yest":      "burn_comeback",
        "days_rest":              "burn_days_rest",
    })

    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} rows)")
    nan_frac = out.drop(columns=["game_pk", "side"]).isna().mean()
    print("NaN fractions:")
    print(nan_frac.round(3).to_string())
    return OUT


if __name__ == "__main__":
    build()
