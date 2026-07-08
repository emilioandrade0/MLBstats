"""Game-flow features — rolling patterns from how games are actually played.

Three feature groups (all computed with shift(1) — strictly prior games):

  1. fi_score_rate_l30   : frequency team scores in the 1st inning (last 30 games)
     fi_score_first_l30  : frequency team scores FIRST in the game
     Rationale: teams that score in inning 1 win ~60% vs ~45% baseline.
     Measures lineup quality *and* early-game aggression not captured by xWOBA.

  2. lead_hold_6_rate_l30 : % of times team holds a lead after 6 innings
     comeback_6_rate_l30  : % of times team was trailing after 6 and won anyway
     Rationale: pure bullpen hold quality. More precise than bullpen_ip_l5d.

  3. runs_std_l20          : std dev of runs scored last 20 games
     runs_scored_median_l20: median runs scored (robust center)
     Rationale: feast-or-famine teams are mispriced; consistent scorers are safer.

Output: data/processed/features_game_flow.parquet
  game_pk | side (home/away) | all feature columns

Run standalone:
  python -m src.features.game_flow
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_game_flow.parquet"

WINDOW_30 = 30
WINDOW_20 = 20
MIN_P = 10


def _last_play_per_half(plays: pd.DataFrame, inning: int, is_top: bool) -> pd.DataFrame:
    """Return the last at-bat of the given inning half per game."""
    mask = (plays["inning"] == inning) & (plays["is_top"] == is_top)
    return (
        plays[mask]
        .sort_values("at_bat_index")
        .groupby("game_pk")
        .last()[["away_score", "home_score"]]
    )


def _rolling(series: pd.Series, window: int, agg: str = "mean") -> pd.Series:
    s = series.shift(1)
    if agg == "mean":
        return s.rolling(window, min_periods=MIN_P).mean()
    if agg == "std":
        return s.rolling(window, min_periods=MIN_P).std()
    if agg == "median":
        return s.rolling(window, min_periods=MIN_P).median()
    raise ValueError(agg)


def build() -> Path:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    plays = pd.read_parquet(PROCESSED / "plays.parquet")

    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_team_abbrev", "away_team_abbrev",
                                  "home_score", "away_score"])
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. FIRST-INNING SCORING
    # ─────────────────────────────────────────────────────────────────────────
    # Away: last play of top of inning 1 → away_score > 0 means away scored
    top1  = _last_play_per_half(plays, inning=1, is_top=True)[["away_score"]]
    top1.columns  = ["fi_away_score"]
    # Home: last play of bottom of inning 1 → home_score > 0 means home scored
    bot1  = _last_play_per_half(plays, inning=1, is_top=False)[["home_score"]]
    bot1.columns  = ["fi_home_score"]

    fi = top1.join(bot1, how="outer").reset_index()
    fi["away_fi"] = (fi["fi_away_score"] > 0).astype(float)
    fi["home_fi"] = (fi["fi_home_score"] > 0).astype(float)
    # Who scored first?
    fi["away_score_first"] = (
        fi["fi_away_score"].fillna(0) > fi["fi_home_score"].fillna(0)
    ).astype(float)
    fi["home_score_first"] = (
        fi["fi_home_score"].fillna(0) > fi["fi_away_score"].fillna(0)
    ).astype(float)

    games = games.merge(fi[["game_pk", "away_fi", "home_fi",
                             "away_score_first", "home_score_first"]],
                        on="game_pk", how="left")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. LEAD AFTER 6 INNINGS
    # ─────────────────────────────────────────────────────────────────────────
    # Use last play of inning 6 (covers both halves)
    i6 = (
        plays[plays["inning"] == 6]
        .sort_values("at_bat_index")
        .groupby("game_pk")
        .last()[["away_score", "home_score"]]
        .rename(columns={"away_score": "away_after6", "home_score": "home_after6"})
        .reset_index()
    )
    games = games.merge(i6, on="game_pk", how="left")

    games["home_led_after6"]     = (games["home_after6"] > games["away_after6"]).astype(float)
    games["away_led_after6"]     = (games["away_after6"] > games["home_after6"]).astype(float)
    games["home_trailed_after6"] = (games["home_after6"] < games["away_after6"]).astype(float)
    games["away_trailed_after6"] = (games["away_after6"] < games["home_after6"]).astype(float)

    # Hold = was leading after 6 AND won
    games["home_held_lead6"]    = ((games["home_led_after6"] == 1) & (games["home_win"] == 1)).astype(float)
    games["away_held_lead6"]    = ((games["away_led_after6"] == 1) & (games["home_win"] == 0)).astype(float)
    # Comeback after 6 = was trailing after 6 AND won
    games["home_comeback6"]     = ((games["home_trailed_after6"] == 1) & (games["home_win"] == 1)).astype(float)
    games["away_comeback6"]     = ((games["away_trailed_after6"] == 1) & (games["home_win"] == 0)).astype(float)

    # ─────────────────────────────────────────────────────────────────────────
    # Build per-team timeline
    # ─────────────────────────────────────────────────────────────────────────
    rows = []
    for _, r in games.iterrows():
        hw = int(r["home_win"])
        for side, team, won in [("home", r["home_team_abbrev"], hw == 1),
                                 ("away", r["away_team_abbrev"], hw == 0)]:
            runs = float(r["home_score"] if side == "home" else r["away_score"])
            rows.append({
                "team": team,
                "game_pk": r["game_pk"],
                "game_date": r["game_date"],
                "won": float(won),
                "runs_scored": runs,
                # first-inning
                "fi_scored": float(r.get(f"{side}_fi", np.nan) or 0),
                "fi_score_first": float(r.get(f"{side}_score_first", np.nan) or 0),
                # lead after 6
                "led_after6":     float(r.get(f"{side}_led_after6", np.nan) or 0),
                "held_lead6":     float(r.get(f"{side}_held_lead6", np.nan) or 0),
                "trailed_after6": float(r.get(f"{side}_trailed_after6", np.nan) or 0),
                "comeback6":      float(r.get(f"{side}_comeback6", np.nan) or 0),
            })

    tdf = pd.DataFrame(rows)
    tdf = tdf.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Rolling features
    # ─────────────────────────────────────────────────────────────────────────
    grp30 = tdf.groupby("team", group_keys=False)
    grp20 = tdf.groupby("team", group_keys=False)

    tdf["fi_score_rate_l30"]    = grp30["fi_scored"].transform(
        lambda x: _rolling(x, WINDOW_30))
    tdf["fi_first_rate_l30"]    = grp30["fi_score_first"].transform(
        lambda x: _rolling(x, WINDOW_30))
    tdf["lead_hold_6_rate_l30"] = grp30["held_lead6"].transform(
        lambda x: _rolling(x, WINDOW_30))
    tdf["comeback_6_rate_l30"]  = grp30["comeback6"].transform(
        lambda x: _rolling(x, WINDOW_30))
    tdf["runs_std_l20"]         = grp20["runs_scored"].transform(
        lambda x: _rolling(x, WINDOW_20, "std"))
    tdf["runs_median_l20"]      = grp20["runs_scored"].transform(
        lambda x: _rolling(x, WINDOW_20, "median"))

    feat_cols = [
        "fi_score_rate_l30", "fi_first_rate_l30",
        "lead_hold_6_rate_l30", "comeback_6_rate_l30",
        "runs_std_l20", "runs_median_l20",
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # Join back to game_pk × side
    # ─────────────────────────────────────────────────────────────────────────
    gm_teams = games[["game_pk", "home_team_abbrev", "away_team_abbrev"]].copy()

    home_side = (
        tdf.merge(gm_teams, left_on=["game_pk", "team"],
                  right_on=["game_pk", "home_team_abbrev"], how="inner")
        [["game_pk"] + feat_cols]
        .assign(side="home")
    )
    away_side = (
        tdf.merge(gm_teams, left_on=["game_pk", "team"],
                  right_on=["game_pk", "away_team_abbrev"], how="inner")
        [["game_pk"] + feat_cols]
        .assign(side="away")
    )

    out = pd.concat([home_side, away_side], ignore_index=True)
    out.to_parquet(OUT, index=False)

    nan_frac = out[feat_cols].isna().mean()
    print(f"wrote {OUT}  ({len(out):,} rows, {len(feat_cols)} features x 2 sides)")
    print("NaN fractions:\n", nan_frac.round(3).to_string())

    # Quick sanity print
    print("\nSample means (all games):")
    print(out[feat_cols].mean().round(3).to_string())
    return OUT


if __name__ == "__main__":
    build()
