from __future__ import annotations

from pathlib import Path

import pandas as pd


SIGNAL_COLUMNS = [
    "home_next_series_after_swept",
    "away_next_series_after_swept",
    "next_series_after_swept_edge",
    "next_series_after_swept_progress_edge",
    "both_next_series_after_swept",
]


def build_team_series(team_games: pd.DataFrame) -> pd.DataFrame:
    games = team_games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["is_home"] = games["is_home"].astype(str).str.lower().eq("true")
    games = games.sort_values(["team", "season", "game_date", "game_pk"], kind="mergesort")
    previous = games.groupby(["team", "season"], sort=False).shift(1)
    boundary = (
        previous["opponent"].isna()
        | games["opponent"].ne(previous["opponent"])
        | games["is_home"].ne(previous["is_home"])
        | games["game_date"].sub(previous["game_date"]).dt.days.gt(3)
    )
    games["series_number"] = boundary.groupby([games["team"], games["season"]]).cumsum().astype(int)
    games["series_game_number"] = games.groupby(
        ["team", "season", "series_number"], sort=False
    ).cumcount() + 1
    return games


def swept_next_series_membership(team_games: pd.DataFrame, exact_series: pd.DataFrame) -> pd.DataFrame:
    games = build_team_series(team_games)
    series_lookup = {
        (str(team), int(season), int(game_pk)): int(series_number)
        for team, season, game_pk, series_number in games[
            ["team", "season", "game_pk", "series_number"]
        ].itertuples(index=False, name=None)
    }
    swept = exact_series[exact_series["swept_against"].eq(1)]
    triggered: set[tuple[str, int, int]] = set()
    trigger_end: dict[tuple[str, int, int], pd.Timestamp] = {}
    for row in swept.itertuples(index=False):
        first_pk = int(str(row.game_pks).split(",")[0])
        current_number = series_lookup.get((str(row.team), int(row.season), first_pk))
        if current_number is None:
            continue
        next_key = (str(row.team), int(row.season), current_number + 1)
        triggered.add(next_key)
        trigger_end[next_key] = pd.Timestamp(row.end_date)

    membership = games[
        games.apply(
            lambda row: (str(row["team"]), int(row["season"]), int(row["series_number"])) in triggered,
            axis=1,
        )
    ].copy()
    membership["swept_series_end"] = membership.apply(
        lambda row: trigger_end[(str(row["team"]), int(row["season"]), int(row["series_number"]))],
        axis=1,
    )
    if not membership.empty and not (membership["game_date"] > membership["swept_series_end"]).all():
        raise RuntimeError("La señal de siguiente serie usa un juego anterior al cierre de la barrida")
    return membership[
        ["game_pk", "team", "series_game_number", "swept_series_end"]
    ].drop_duplicates(["game_pk", "team"])


def attach_swept_next_series_features(root: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = root / "work" / "sweep_report_20260824"
    team_games = pd.read_csv(work / "games_final_regular.csv")
    exact_series = pd.read_csv(work / "team_series.csv")
    membership = swept_next_series_membership(team_games, exact_series)
    lookup = {
        (int(game_pk), str(team)): int(game_number)
        for game_pk, team, game_number in membership[
            ["game_pk", "team", "series_game_number"]
        ].itertuples(index=False, name=None)
    }

    result = frame.copy()
    home_numbers = [lookup.get((int(pk), str(team)), 0) for pk, team in result[["gamePk", "home"]].itertuples(index=False, name=None)]
    away_numbers = [lookup.get((int(pk), str(team)), 0) for pk, team in result[["gamePk", "away"]].itertuples(index=False, name=None)]
    result["home_next_series_after_swept"] = [float(number > 0) for number in home_numbers]
    result["away_next_series_after_swept"] = [float(number > 0) for number in away_numbers]
    result["next_series_after_swept_edge"] = (
        result["away_next_series_after_swept"] - result["home_next_series_after_swept"]
    )
    result["next_series_after_swept_progress_edge"] = [
        float(away - home) for home, away in zip(home_numbers, away_numbers)
    ]
    result["both_next_series_after_swept"] = (
        result["home_next_series_after_swept"] * result["away_next_series_after_swept"]
    )

    flagged_games = result[
        result["home_next_series_after_swept"].eq(1)
        | result["away_next_series_after_swept"].eq(1)
    ]
    audit = {
        "eligibleSweeps": int(exact_series["swept_against"].sum()),
        "teamGamesFlagged": int(len(membership)),
        "gamesFlagged": int(len(flagged_games)),
        "bothTeamsFlagged": int(result["both_next_series_after_swept"].sum()),
        "firstFlaggedDate": str(flagged_games["date"].min()) if len(flagged_games) else None,
        "lastFlaggedDate": str(flagged_games["date"].max()) if len(flagged_games) else None,
        "futureOutcomeColumnsUsed": [],
    }
    return result, audit
