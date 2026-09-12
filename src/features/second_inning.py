"""Second-inning scoring profile features.

The league-wide second inning is a low-scoring spot after adjusting for
opportunities. This module measures which teams deviate from that baseline
using only games played before the current game.

Output: data/processed/features_second_inning.parquet
  game_pk | side | si_* rolling feature columns
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..normalize.paths import PROCESSED

OUT = PROCESSED / "features_second_inning.parquet"
WINDOW_SHORT = 30
WINDOW_LONG = 60
MIN_SHORT = 10
MIN_LONG = 20


def _score_after_half(plays: pd.DataFrame, inning: int, is_top: bool, prefix: str) -> pd.DataFrame:
    mask = (plays["inning"] == inning) & (plays["is_top"] == is_top)
    out = (
        plays.loc[mask]
        .sort_values(["game_pk", "at_bat_index"], kind="mergesort")
        .groupby("game_pk", as_index=False)
        .tail(1)[["game_pk", "away_score", "home_score"]]
        .rename(columns={
            "away_score": f"{prefix}_away_score",
            "home_score": f"{prefix}_home_score",
        })
    )
    return out


def _rolling_mean(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def build() -> Path:
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=[
            "game_pk", "game_date", "game_type",
            "home_team_abbrev", "away_team_abbrev",
            "home_score", "away_score",
        ],
    )
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games[games["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    games = games.dropna(subset=[
        "game_date", "home_team_abbrev", "away_team_abbrev",
        "home_score", "away_score",
    ])

    plays = pd.read_parquet(
        PROCESSED / "plays.parquet",
        columns=["game_pk", "at_bat_index", "inning", "is_top", "away_score", "home_score"],
    )
    plays = plays.dropna(subset=["game_pk", "inning", "at_bat_index"]).copy()
    plays["inning"] = plays["inning"].astype(int)

    top1 = _score_after_half(plays, inning=1, is_top=True, prefix="top1")
    bot1 = _score_after_half(plays, inning=1, is_top=False, prefix="bot1")
    top2 = _score_after_half(plays, inning=2, is_top=True, prefix="top2")
    bot2 = _score_after_half(plays, inning=2, is_top=False, prefix="bot2")

    inn = top2.merge(top1, on="game_pk", how="left")
    inn = inn.merge(bot1, on="game_pk", how="left")
    inn = inn.merge(bot2, on="game_pk", how="left")
    inn["away_si_runs"] = (
        pd.to_numeric(inn["top2_away_score"], errors="coerce")
        - pd.to_numeric(inn["top1_away_score"], errors="coerce").fillna(0)
    )
    inn["home_si_runs"] = (
        pd.to_numeric(inn["bot2_home_score"], errors="coerce")
        - pd.to_numeric(inn["bot1_home_score"], errors="coerce").fillna(0)
    )
    inn["away_si_runs"] = inn["away_si_runs"].clip(lower=0)
    inn["home_si_runs"] = inn["home_si_runs"].clip(lower=0)

    games = games.merge(inn[["game_pk", "away_si_runs", "home_si_runs"]], on="game_pk", how="left")
    games = games.dropna(subset=["away_si_runs", "home_si_runs"]).copy()

    rows = []
    for r in games.itertuples(index=False):
        rows.append({
            "game_pk": r.game_pk,
            "game_date": r.game_date,
            "team": r.home_team_abbrev,
            "side": "home",
            "si_runs_scored": float(r.home_si_runs),
            "si_runs_allowed": float(r.away_si_runs),
        })
        rows.append({
            "game_pk": r.game_pk,
            "game_date": r.game_date,
            "team": r.away_team_abbrev,
            "side": "away",
            "si_runs_scored": float(r.away_si_runs),
            "si_runs_allowed": float(r.home_si_runs),
        })

    tdf = pd.DataFrame(rows)
    tdf = tdf.sort_values(["team", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    tdf["si_scored"] = (tdf["si_runs_scored"] > 0).astype(float)
    tdf["si_allowed"] = (tdf["si_runs_allowed"] > 0).astype(float)
    tdf["si_run_diff"] = tdf["si_runs_scored"] - tdf["si_runs_allowed"]

    grp = tdf.groupby("team", sort=False)
    for col in ["si_runs_scored", "si_runs_allowed", "si_scored", "si_allowed", "si_run_diff"]:
        tdf[f"{col}_l30"] = grp[col].transform(
            lambda x: _rolling_mean(x, WINDOW_SHORT, MIN_SHORT)
        )
        tdf[f"{col}_l60"] = grp[col].transform(
            lambda x: _rolling_mean(x, WINDOW_LONG, MIN_LONG)
        )

    feature_cols = [
        "si_runs_scored_l30", "si_runs_allowed_l30",
        "si_scored_l30", "si_allowed_l30", "si_run_diff_l30",
        "si_runs_scored_l60", "si_runs_allowed_l60",
        "si_scored_l60", "si_allowed_l60", "si_run_diff_l60",
    ]
    out = tdf[["game_pk", "side"] + feature_cols].copy()
    out.to_parquet(OUT, index=False)

    print(f"wrote {OUT} ({len(out):,} rows, {len(feature_cols)} features x 2 sides)")
    print("NaN fractions:")
    print(out[feature_cols].isna().mean().round(3).to_string())
    print("\nSample means:")
    print(out[feature_cols].mean().round(3).to_string())
    return OUT


if __name__ == "__main__":
    build()
