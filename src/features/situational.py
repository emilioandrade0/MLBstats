"""Situational features — conservative same-day mirror-pair context.

Feature groups:

  MIRROR PAIRS (same-day anti-correlated teams, no lookahead)
    mirror_pair_signal     : signed score from an earlier-finished paired team on
                             the same date. Positive helps this team, negative
                             hurts this team.
    mirror_pair_abs_signal : absolute strength of that earlier paired signal
    mirror_pair_seen       : 1 if any valid earlier paired team signal exists

  POST-UPSET FAVORITE (team-specific, shrunk)
    upset_fav60_ctx              : current game follows a home 3+ run loss as
                                   60%+ favorite, and team is favorite again
    upset_fav60_shrunk_edge      : team-specific next-game edge after that spot,
                                   shrunk toward the global prior mean
    upset_fav60_team_n_prior     : prior team sample count for that context

Output: data/processed/features_situational.parquet
  game_pk | side (home/away) | all feature columns

Run standalone:
  python -m src.features.situational
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_situational.parquet"

MIRROR_MIN_N = 50
MIRROR_MIN_DELTA = 0.14
MIRROR_MAX_P = 0.05
UPSET_FAV_THRESHOLD = 0.60
UPSET_MARGIN_THRESHOLD = 3.0
UPSET_SHRINK_K = 8.0
UPSET_MIN_TEAM_N = 3.0


def _team_game_rows(games: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    played = games.merge(market[["game_pk", "market_p_home"]], on="game_pk", how="left")
    played["game_date_only"] = pd.to_datetime(played["game_date"]).dt.date
    played["first_pitch_utc"] = pd.to_datetime(played["first_pitch_utc"], utc=True, errors="coerce")
    played["duration_minutes"] = pd.to_numeric(played["duration_minutes"], errors="coerce")
    played["end_utc"] = played["first_pitch_utc"] + pd.to_timedelta(
        played["duration_minutes"].fillna(180), unit="m"
    )

    home = played[[
        "season", "game_pk", "game_date", "game_date_only", "home_team_abbrev",
        "home_score", "away_score", "market_p_home", "first_pitch_utc", "end_utc",
    ]].copy()
    home = home.rename(columns={
        "home_team_abbrev": "team",
        "home_score": "runs_for",
        "away_score": "runs_against",
    })
    home["side"] = "home"
    home["won"] = np.where(
        home["runs_for"].notna() & home["runs_against"].notna(),
        (home["runs_for"] > home["runs_against"]).astype(float),
        np.nan,
    )
    home["team_pregame_p"] = home["market_p_home"]

    away = played[[
        "season", "game_pk", "game_date", "game_date_only", "away_team_abbrev",
        "away_score", "home_score", "market_p_home", "first_pitch_utc", "end_utc",
    ]].copy()
    away = away.rename(columns={
        "away_team_abbrev": "team",
        "away_score": "runs_for",
        "home_score": "runs_against",
    })
    away["side"] = "away"
    away["won"] = np.where(
        away["runs_for"].notna() & away["runs_against"].notna(),
        (away["runs_for"] > away["runs_against"]).astype(float),
        np.nan,
    )
    away["team_pregame_p"] = np.where(
        away["market_p_home"].notna(), 1.0 - away["market_p_home"], np.nan
    )

    tdf = pd.concat([home, away], ignore_index=True)
    tdf["fav_flag"] = np.where(
        tdf["team_pregame_p"] > 0.5, 1.0,
        np.where(tdf["team_pregame_p"] < 0.5, 0.0, np.nan),
    )
    return tdf


def _mirror_pair_map(team_games: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Return late_team -> [(early_team, delta), ...] for robust ordered anti-pairs."""
    single = team_games.groupby(["team", "game_date_only"]).filter(lambda g: len(g) == 1).copy()
    single = single.dropna(subset=["won", "fav_flag", "first_pitch_utc", "end_utc"])

    rows: list[dict[str, object]] = []
    for _, grp in single.groupby("game_date_only", sort=False):
        vals = grp.to_dict("records")
        for i, early in enumerate(vals):
            for j, late in enumerate(vals):
                if i == j:
                    continue
                if int(early["fav_flag"]) == int(late["fav_flag"]):
                    continue
                if late["first_pitch_utc"] < early["end_utc"]:
                    continue
                rows.append({
                    "early_team": early["team"],
                    "late_team": late["team"],
                    "early_win": int(early["won"]),
                    "late_win": int(late["won"]),
                })

    ordered = pd.DataFrame(rows)
    if ordered.empty:
        return {}

    pair_map: dict[str, list[tuple[str, float]]] = {}
    for (early_team, late_team), grp in ordered.groupby(["early_team", "late_team"], sort=False):
        if len(grp) < MIRROR_MIN_N:
            continue
        early_wins = grp["early_win"] == 1
        if not early_wins.any():
            continue
        if grp["early_win"].nunique() < 2 or grp["late_win"].nunique() < 2:
            continue
        late_lose_if_early_win = float((1 - grp.loc[early_wins, "late_win"]).mean())
        late_lose_base = float((1 - grp["late_win"]).mean())
        delta = late_lose_if_early_win - late_lose_base
        phi = float(np.corrcoef(grp["early_win"], grp["late_win"])[0, 1])
        both_win = int(((grp["early_win"] == 1) & (grp["late_win"] == 1)).sum())
        early_win_late_lose = int(((grp["early_win"] == 1) & (grp["late_win"] == 0)).sum())
        early_lose_late_win = int(((grp["early_win"] == 0) & (grp["late_win"] == 1)).sum())
        both_lose = int(((grp["early_win"] == 0) & (grp["late_win"] == 0)).sum())
        _, p_value = fisher_exact([
            [both_win, early_win_late_lose],
            [early_lose_late_win, both_lose],
        ])
        if delta < MIRROR_MIN_DELTA or p_value > MIRROR_MAX_P or phi >= 0:
            continue
        pair_map.setdefault(late_team, []).append((early_team, delta))

    for late_team in pair_map:
        pair_map[late_team] = sorted(pair_map[late_team], key=lambda x: x[1], reverse=True)
    return pair_map


def _add_upset_favorite_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add conservative post-upset favorite features with chronological shrinkage."""
    tdf = team_games.sort_values(["team", "game_date", "game_pk"]).copy()
    tdf["won"] = pd.to_numeric(tdf["won"], errors="coerce")
    tdf["margin"] = pd.to_numeric(tdf["runs_for"] - tdf["runs_against"], errors="coerce").abs()
    tdf["was_home"] = (tdf["side"] == "home").astype(float)
    tdf["was_fav60"] = (tdf["team_pregame_p"] >= UPSET_FAV_THRESHOLD).astype(float)
    tdf["lost_home_big_as_fav60"] = (
        (tdf["won"] == 0.0) &
        (tdf["was_home"] == 1.0) &
        (tdf["was_fav60"] == 1.0) &
        (tdf["margin"] >= UPSET_MARGIN_THRESHOLD)
    ).astype(float)
    tdf["prev_lost_home_big_as_fav60"] = tdf.groupby("team")["lost_home_big_as_fav60"].shift(1).fillna(0.0)
    tdf["current_fav60"] = (tdf["team_pregame_p"] >= UPSET_FAV_THRESHOLD).astype(float)
    tdf["upset_fav60_ctx"] = (
        (tdf["prev_lost_home_big_as_fav60"] == 1.0) &
        (tdf["current_fav60"] == 1.0)
    ).astype(float)

    eligible = (tdf["upset_fav60_ctx"] == 1.0) & tdf["won"].notna()
    tdf["ctx_win"] = np.where(eligible, tdf["won"], 0.0)
    tdf["ctx_n"] = eligible.astype(float)

    team_grp = tdf.groupby("team", sort=False)
    tdf["upset_fav60_team_n_prior"] = team_grp["ctx_n"].cumsum() - tdf["ctx_n"]
    tdf["team_ctx_wins_prior"] = team_grp["ctx_win"].cumsum() - tdf["ctx_win"]

    chrono = tdf.sort_values(["game_date", "game_pk", "team"]).copy()
    chrono["global_ctx_n_prior"] = chrono["ctx_n"].cumsum() - chrono["ctx_n"]
    chrono["global_ctx_wins_prior"] = chrono["ctx_win"].cumsum() - chrono["ctx_win"]
    global_mean = chrono["global_ctx_wins_prior"] / chrono["global_ctx_n_prior"].replace(0.0, np.nan)
    chrono["global_ctx_mean_prior"] = global_mean.fillna(0.5)
    tdf = tdf.merge(
        chrono[["game_pk", "team", "global_ctx_mean_prior"]],
        on=["game_pk", "team"],
        how="left",
    )

    denom = tdf["upset_fav60_team_n_prior"] + UPSET_SHRINK_K
    numer = tdf["team_ctx_wins_prior"] + UPSET_SHRINK_K * tdf["global_ctx_mean_prior"]
    shrunk = numer / denom.replace(0.0, np.nan)
    edge = shrunk - tdf["global_ctx_mean_prior"]

    active = (tdf["upset_fav60_ctx"] == 1.0) & (tdf["upset_fav60_team_n_prior"] >= UPSET_MIN_TEAM_N)
    tdf["upset_fav60_shrunk_edge"] = np.where(active, edge.fillna(0.0), 0.0)
    tdf["upset_fav60_team_n_prior"] = np.where(
        tdf["upset_fav60_ctx"] == 1.0,
        tdf["upset_fav60_team_n_prior"],
        0.0,
    )

    return tdf[["game_pk", "team", "upset_fav60_ctx", "upset_fav60_shrunk_edge", "upset_fav60_team_n_prior"]]
def build() -> Path:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    market = pd.read_parquet(PROCESSED / "features_market.parquet")
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.dropna(subset=["home_team_abbrev", "away_team_abbrev"])

    team_games = _team_game_rows(games, market)
    mirror_pair_map = _mirror_pair_map(team_games)
    upset_features = _add_upset_favorite_features(team_games)

    single = team_games.groupby(["team", "game_date_only"]).filter(lambda g: len(g) == 1).copy()
    single = single.dropna(subset=["fav_flag", "first_pitch_utc", "end_utc"])
    played_single = single.dropna(subset=["won"]).copy()
    day_groups = {
        day: grp[["team", "won", "fav_flag", "end_utc"]].to_dict("records")
        for day, grp in played_single.groupby("game_date_only", sort=False)
    }

    signal_rows: list[dict[str, object]] = []
    for _, row in single.iterrows():
        team = row["team"]
        day = row["game_date_only"]
        current_flag = int(row["fav_flag"])
        current_start = row["first_pitch_utc"]
        best_signal = 0.0
        best_abs = 0.0
        seen = 0.0

        for early_team, delta in mirror_pair_map.get(team, []):
            for cand in day_groups.get(day, []):
                if cand["team"] != early_team:
                    continue
                if int(cand["fav_flag"]) == current_flag:
                    continue
                if current_start < cand["end_utc"]:
                    continue
                seen = 1.0
                signal = -delta if int(cand["won"]) == 1 else delta
                if abs(signal) > best_abs:
                    best_signal = float(signal)
                    best_abs = float(abs(signal))

        signal_rows.append({
            "game_pk": row["game_pk"],
            "team": team,
            "mirror_pair_signal": best_signal,
            "mirror_pair_abs_signal": best_abs,
            "mirror_pair_seen": seen,
        })

    sig = pd.DataFrame(signal_rows)
    sig = sig.merge(upset_features, on=["game_pk", "team"], how="left")
    gm_teams = games[["game_pk", "home_team_abbrev", "away_team_abbrev"]].copy()

    home_side = (
        sig.merge(gm_teams, left_on=["game_pk", "team"],
                  right_on=["game_pk", "home_team_abbrev"], how="inner")
        [[
            "game_pk",
            "mirror_pair_signal", "mirror_pair_abs_signal", "mirror_pair_seen",
            "upset_fav60_ctx", "upset_fav60_shrunk_edge", "upset_fav60_team_n_prior",
        ]]
        .assign(side="home")
    )
    away_side = (
        sig.merge(gm_teams, left_on=["game_pk", "team"],
                  right_on=["game_pk", "away_team_abbrev"], how="inner")
        [[
            "game_pk",
            "mirror_pair_signal", "mirror_pair_abs_signal", "mirror_pair_seen",
            "upset_fav60_ctx", "upset_fav60_shrunk_edge", "upset_fav60_team_n_prior",
        ]]
        .assign(side="away")
    )

    out = pd.concat([home_side, away_side], ignore_index=True)
    out.to_parquet(OUT, index=False)

    feat_cols = [
        "mirror_pair_signal", "mirror_pair_abs_signal", "mirror_pair_seen",
        "upset_fav60_ctx", "upset_fav60_shrunk_edge", "upset_fav60_team_n_prior",
    ]
    nan_frac = out[feat_cols].isna().mean()
    print(f"wrote {OUT}  ({len(out):,} rows, {len(feat_cols)} features x 2 sides)")
    print("NaN fractions:\n", nan_frac.round(3).to_string())
    print("\nSample means:")
    print(out[feat_cols].mean().round(3).to_string())
    return OUT


if __name__ == "__main__":
    build()
