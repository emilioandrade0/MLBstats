"""Situational filter columns for pattern-hunting.

For every game we compute a boolean per filter. Downstream analysis (API +
frontend) picks subsets by ANDing filters and reports accuracy + ROI on that
slice — like sabermetric splits, but for NFL walk-forward.

Filters implemented (initial catalog — easy to extend):

REST / FATIGUE
  short_week_home / away        team plays with ≤5 days rest (Thu-after-Sun)
  off_bye_home / away           team coming off 10+ day rest (bye)
  rest_edge_home / away         team has ≥3 days more rest than opponent
  short_week_either             at least one team on short week

SITUATIONAL
  divisional                    both teams share a division (heuristic by team pair)
  primetime                     kickoff between 20:00 and 04:00 ET (MNF/SNF/TNF)
  playoffs                      game_type != REG
  early_season                  week ≤ 4
  late_season                   week ≥ 15

WEATHER (outdoor only)
  outdoor                       roof in {outdoors, open}
  windy                         wind ≥ 15 mph
  cold                          temp ≤ 32°F
  dome                          roof in {dome, closed}

MARKET / LINE
  big_home_fav                  spread_line ≥ 7 (home favored by 7+)
  big_home_dog                  spread_line ≤ -7 (home dog by 7+)
  home_dog                      spread_line < 0
  low_total                     total_line ≤ 40
  high_total                    total_line ≥ 50
  pickem                        |spread_line| ≤ 2.5

STREAKS (last 3 games ATS, walk-forward safe)
  home_ats_hot_l3               home covered 3/3 last games
  away_ats_hot_l3
  home_ats_cold_l3              home covered 0/3
  away_ats_cold_l3

Output: data/processed/game_filters.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import PROCESSED

# NFL divisions (2024 alignment). Not historically perfect but good enough for filter slicing.
DIVISIONS = {
    "AFC_E": {"BUF", "MIA", "NE", "NYJ"},
    "AFC_N": {"BAL", "CIN", "CLE", "PIT"},
    "AFC_S": {"HOU", "IND", "JAX", "TEN"},
    "AFC_W": {"DEN", "KC", "LV", "LAC"},
    "NFC_E": {"DAL", "NYG", "PHI", "WAS"},
    "NFC_N": {"CHI", "DET", "GB", "MIN"},
    "NFC_S": {"ATL", "CAR", "NO", "TB"},
    "NFC_W": {"ARI", "LA", "SF", "SEA"},
}

TEAM_TO_DIV = {t: d for d, teams in DIVISIONS.items() for t in teams}


def _team_rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game (walk-forward safe)."""
    long = pd.concat([
        games[["game_id", "kickoff_utc", "home_team"]].rename(columns={"home_team": "team"}),
        games[["game_id", "kickoff_utc", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True).sort_values(["team", "kickoff_utc"])
    long["prev_kickoff"] = long.groupby("team")["kickoff_utc"].shift(1)
    long["rest_days"] = (long["kickoff_utc"] - long["prev_kickoff"]).dt.total_seconds() / 86400.0
    return long


def _ats_streak(games: pd.DataFrame) -> pd.DataFrame:
    """Last-3 ATS record per team going INTO the game (excludes current)."""
    finals = games[games["status"] == "final"].dropna(subset=["spread_line", "home_score", "away_score"]).copy()
    finals["margin_home"] = finals["home_score"] - finals["away_score"]
    finals["home_covered"] = (finals["margin_home"] - finals["spread_line"]) > 0

    long_home = finals[["game_id", "kickoff_utc", "home_team", "home_covered"]] \
        .rename(columns={"home_team": "team", "home_covered": "covered"})
    long_away = finals[["game_id", "kickoff_utc", "away_team", "home_covered"]] \
        .rename(columns={"away_team": "team"})
    long_away["covered"] = ~long_away["home_covered"]
    long_away = long_away.drop(columns=["home_covered"])
    long = pd.concat([long_home, long_away], ignore_index=True) \
             .sort_values(["team", "kickoff_utc"])

    # rolling 3-game sum of covers PRIOR to current game
    long["ats_l3_covers"] = long.groupby("team")["covered"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum()
    )
    long["ats_l3_games"] = long.groupby("team")["covered"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).count()
    )
    return long[["game_id", "team", "ats_l3_covers", "ats_l3_games"]]


def build() -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    g = games.copy()
    g["kickoff_utc"] = pd.to_datetime(g["kickoff_utc"], errors="coerce", utc=True)

    # ---- rest ----
    rest = _team_rest_days(g)
    home_rest = rest.merge(g[["game_id", "home_team"]], left_on=["game_id", "team"],
                           right_on=["game_id", "home_team"])[["game_id", "rest_days"]] \
                    .rename(columns={"rest_days": "home_rest"})
    away_rest = rest.merge(g[["game_id", "away_team"]], left_on=["game_id", "team"],
                           right_on=["game_id", "away_team"])[["game_id", "rest_days"]] \
                    .rename(columns={"rest_days": "away_rest"})
    g = g.merge(home_rest, on="game_id", how="left").merge(away_rest, on="game_id", how="left")

    g["short_week_home"] = (g["home_rest"] <= 5).fillna(False)
    g["short_week_away"] = (g["away_rest"] <= 5).fillna(False)
    g["short_week_either"] = g["short_week_home"] | g["short_week_away"]
    g["off_bye_home"] = (g["home_rest"] >= 10).fillna(False)
    g["off_bye_away"] = (g["away_rest"] >= 10).fillna(False)
    g["rest_edge_home"] = ((g["home_rest"] - g["away_rest"]) >= 3).fillna(False)
    g["rest_edge_away"] = ((g["away_rest"] - g["home_rest"]) >= 3).fillna(False)

    # ---- situational ----
    g["divisional"] = g.apply(
        lambda r: TEAM_TO_DIV.get(r["home_team"]) is not None
                  and TEAM_TO_DIV.get(r["home_team"]) == TEAM_TO_DIV.get(r["away_team"]),
        axis=1,
    )
    # primetime: kickoff hour local (approx via ET = UTC-4/5 depending on DST; we use UTC-5 conservative)
    hour_et = ((g["kickoff_utc"].dt.hour - 5) % 24)
    g["primetime"] = ((hour_et >= 20) | (hour_et <= 4)).fillna(False)
    g["playoffs"] = g["game_type"].isin(["WC", "DIV", "CON", "SB"])
    g["early_season"] = (g["week"] <= 4)
    g["late_season"] = (g["week"] >= 15) & (~g["playoffs"])

    # ---- weather ----
    g["outdoor"] = g["roof"].isin(["outdoors", "open"])
    g["dome"] = g["roof"].isin(["dome", "closed"])
    g["windy"] = (g["wind"] >= 15).fillna(False) & g["outdoor"]
    g["cold"] = (g["temp"] <= 32).fillna(False) & g["outdoor"]

    # ---- market ----
    g["big_home_fav"] = (g["spread_line"] >= 7).fillna(False)
    g["big_home_dog"] = (g["spread_line"] <= -7).fillna(False)
    g["home_dog"] = (g["spread_line"] < 0).fillna(False)
    g["low_total"] = (g["total_line"] <= 40).fillna(False)
    g["high_total"] = (g["total_line"] >= 50).fillna(False)
    g["pickem"] = (g["spread_line"].abs() <= 2.5).fillna(False)

    # ---- streaks ----
    streak = _ats_streak(g)
    home_streak = streak.merge(g[["game_id", "home_team"]],
                               left_on=["game_id", "team"], right_on=["game_id", "home_team"])
    home_streak = home_streak[["game_id", "ats_l3_covers", "ats_l3_games"]] \
        .rename(columns={"ats_l3_covers": "home_ats_l3_c", "ats_l3_games": "home_ats_l3_n"})
    away_streak = streak.merge(g[["game_id", "away_team"]],
                               left_on=["game_id", "team"], right_on=["game_id", "away_team"])
    away_streak = away_streak[["game_id", "ats_l3_covers", "ats_l3_games"]] \
        .rename(columns={"ats_l3_covers": "away_ats_l3_c", "ats_l3_games": "away_ats_l3_n"})
    g = g.merge(home_streak, on="game_id", how="left").merge(away_streak, on="game_id", how="left")

    g["home_ats_hot_l3"] = (g["home_ats_l3_c"] == 3) & (g["home_ats_l3_n"] == 3)
    g["away_ats_hot_l3"] = (g["away_ats_l3_c"] == 3) & (g["away_ats_l3_n"] == 3)
    g["home_ats_cold_l3"] = (g["home_ats_l3_c"] == 0) & (g["home_ats_l3_n"] == 3)
    g["away_ats_cold_l3"] = (g["away_ats_l3_c"] == 0) & (g["away_ats_l3_n"] == 3)

    FILTER_COLS = [
        # rest
        "short_week_home", "short_week_away", "short_week_either",
        "off_bye_home", "off_bye_away",
        "rest_edge_home", "rest_edge_away",
        # situational
        "divisional", "primetime", "playoffs", "early_season", "late_season",
        # weather
        "outdoor", "dome", "windy", "cold",
        # market
        "big_home_fav", "big_home_dog", "home_dog", "low_total", "high_total", "pickem",
        # streaks
        "home_ats_hot_l3", "away_ats_hot_l3", "home_ats_cold_l3", "away_ats_cold_l3",
    ]
    for c in FILTER_COLS:
        g[c] = g[c].astype(bool)

    out = g[["game_id"] + FILTER_COLS]
    path = PROCESSED / "game_filters.parquet"
    out.to_parquet(path, index=False)

    # Print a coverage summary
    print(f"game_filters.parquet: {len(out):,} rows, {len(FILTER_COLS)} filters")
    counts = out[FILTER_COLS].sum().sort_values(ascending=False)
    for name, n in counts.items():
        pct = n / len(out) * 100
        print(f"  {name:<20} {int(n):>5}  ({pct:5.1f}%)")
    return out


FILTER_CATALOG = {
    "rest": {
        "label": "Descanso y fatiga",
        "filters": [
            ("short_week_home", "Home short week (≤5 días)"),
            ("short_week_away", "Away short week (≤5 días)"),
            ("short_week_either", "Alguno con short week"),
            ("off_bye_home", "Home post-bye (≥10 días)"),
            ("off_bye_away", "Away post-bye (≥10 días)"),
            ("rest_edge_home", "Home con ≥3 días más descanso"),
            ("rest_edge_away", "Away con ≥3 días más descanso"),
        ],
    },
    "situational": {
        "label": "Situacional",
        "filters": [
            ("divisional", "Juego divisional"),
            ("primetime", "Prime time (MNF/SNF/TNF)"),
            ("playoffs", "Postemporada"),
            ("early_season", "Inicio de temporada (S1-4)"),
            ("late_season", "Cierre de temporada (S15+)"),
        ],
    },
    "weather": {
        "label": "Clima / venue",
        "filters": [
            ("outdoor", "Aire libre"),
            ("dome", "Domo"),
            ("windy", "Viento ≥15 mph (outdoor)"),
            ("cold", "Frío ≤32°F (outdoor)"),
        ],
    },
    "market": {
        "label": "Mercado / línea",
        "filters": [
            ("big_home_fav", "Home fav ≥7"),
            ("big_home_dog", "Home dog ≥7"),
            ("home_dog", "Home dog cualquiera"),
            ("low_total", "Total ≤40"),
            ("high_total", "Total ≥50"),
            ("pickem", "Pick'em (|spread| ≤2.5)"),
        ],
    },
    "streaks": {
        "label": "Rachas ATS",
        "filters": [
            ("home_ats_hot_l3", "Home 3/3 ATS (últimos 3)"),
            ("away_ats_hot_l3", "Away 3/3 ATS (últimos 3)"),
            ("home_ats_cold_l3", "Home 0/3 ATS"),
            ("away_ats_cold_l3", "Away 0/3 ATS"),
        ],
    },
}


if __name__ == "__main__":
    build()
