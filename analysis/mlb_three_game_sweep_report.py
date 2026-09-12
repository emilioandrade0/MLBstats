from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GAMES = Path("data/processed/games.parquet")
DEFAULT_SCHEDULE = Path("data/raw/statsapi_schedule")
DEFAULT_OUTPUT = Path("work/sweep_report_20260824")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1 + (z * z / total)
    centre = (p + (z * z / (2 * total))) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total)))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_schedule_metadata(schedule_root: Path) -> tuple[dict[int, dict[str, object]], pd.DataFrame]:
    metadata: dict[int, dict[str, object]] = {}
    final_records: dict[int, dict[str, object]] = {}
    for path in sorted(schedule_root.glob("20??/20??-??.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for date_block in payload.get("dates", []):
            for game in date_block.get("games", []):
                game_pk = game.get("gamePk")
                if game_pk is None:
                    continue
                record = {
                    "games_in_series": game.get("gamesInSeries"),
                    "series_game_number": game.get("seriesGameNumber"),
                    "series_description": game.get("seriesDescription"),
                    "source_file": path.as_posix(),
                }
                previous = metadata.get(int(game_pk))
                if previous is None:
                    metadata[int(game_pk)] = record
                else:
                    # Prefer a record with explicit series fields over a shallow schedule record.
                    old_score = sum(
                        previous.get(key) is not None for key in ("games_in_series", "series_game_number")
                    )
                    new_score = sum(
                        record.get(key) is not None for key in ("games_in_series", "series_game_number")
                    )
                    if new_score >= old_score:
                        metadata[int(game_pk)] = record
                status = game.get("status", {})
                teams = game.get("teams", {})
                away = teams.get("away", {})
                home = teams.get("home", {})
                away_team = away.get("team", {})
                home_team = home.get("team", {})
                if (
                    game.get("gameType") == "R"
                    and status.get("abstractGameState") == "Final"
                    and away.get("score") is not None
                    and home.get("score") is not None
                    and away_team.get("id") is not None
                    and home_team.get("id") is not None
                ):
                    final_records[int(game_pk)] = {
                        "game_pk": int(game_pk),
                        "season": int(game.get("season")),
                        "game_date": game.get("officialDate"),
                        "first_pitch_utc": game.get("gameDate"),
                        "game_type": game.get("gameType"),
                        "status": status.get("abstractGameState"),
                        "status_code": status.get("statusCode"),
                        "double_header": game.get("doubleHeader"),
                        "game_number": game.get("gameNumber"),
                        "away_team_id": int(away_team.get("id")),
                        "away_team_abbrev": away_team.get("abbreviation"),
                        "away_team_name": away_team.get("name"),
                        "home_team_id": int(home_team.get("id")),
                        "home_team_abbrev": home_team.get("abbreviation"),
                        "home_team_name": home_team.get("name"),
                        "away_score": away.get("score"),
                        "home_score": home.get("score"),
                    }
    supplement = pd.DataFrame(final_records.values())
    return metadata, supplement


def load_final_regular_games(path: Path, schedule_supplement: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = [
        "game_pk",
        "season",
        "game_date",
        "first_pitch_utc",
        "game_type",
        "status",
        "status_code",
        "double_header",
        "game_number",
        "away_team_id",
        "away_team_abbrev",
        "away_team_name",
        "home_team_id",
        "home_team_abbrev",
        "home_team_name",
        "away_score",
        "home_score",
    ]
    parquet_raw = pd.read_parquet(path, columns=columns)
    source_rows = len(parquet_raw)
    duplicate_source_ids = int(parquet_raw.duplicated("game_pk").sum())
    parquet_final_mask = (
        parquet_raw["game_type"].eq("R")
        & parquet_raw["status"].eq("Final")
        & parquet_raw["away_score"].notna()
        & parquet_raw["home_score"].notna()
    )
    parquet_final_ids = set(parquet_raw.loc[parquet_final_mask, "game_pk"].dropna().astype(int).tolist())
    supplement_ids = set(schedule_supplement["game_pk"].dropna().astype(int).tolist()) if len(schedule_supplement) else set()
    supplemental_games_added = len(supplement_ids - parquet_final_ids)
    raw = pd.concat([schedule_supplement.reindex(columns=columns), parquet_raw], ignore_index=True)
    games = raw.loc[
        raw["game_type"].eq("R")
        & raw["status"].eq("Final")
        & raw["away_score"].notna()
        & raw["home_score"].notna()
    ].copy()
    games = games.drop_duplicates("game_pk", keep="last")
    games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
    games["first_pitch_dt"] = pd.to_datetime(games["first_pitch_utc"], errors="coerce", utc=True)
    fallback = games["game_date"].dt.tz_localize("UTC") + pd.Timedelta(hours=12)
    games["sort_time"] = games["first_pitch_dt"].fillna(fallback)
    games = games.dropna(
        subset=[
            "season",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_team_abbrev",
            "away_team_abbrev",
        ]
    )
    games["season"] = games["season"].astype(int)
    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce").astype(int)
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce").astype(int)
    games["home_win"] = games["home_score"] > games["away_score"]
    games = games.sort_values(["season", "sort_time", "game_pk"], kind="mergesort").reset_index(drop=True)

    # Franchise IDs are stable even when branding/abbreviations change (OAK -> ATH).
    identity_rows = pd.concat(
        [
            games[["game_date", "sort_time", "home_team_id", "home_team_abbrev", "home_team_name"]].rename(
                columns={
                    "home_team_id": "team_id",
                    "home_team_abbrev": "team_abbrev",
                    "home_team_name": "team_name",
                }
            ),
            games[["game_date", "sort_time", "away_team_id", "away_team_abbrev", "away_team_name"]].rename(
                columns={
                    "away_team_id": "team_id",
                    "away_team_abbrev": "team_abbrev",
                    "away_team_name": "team_name",
                }
            ),
        ],
        ignore_index=True,
    ).sort_values(["game_date", "sort_time"], kind="mergesort")
    canonical = identity_rows.groupby("team_id", sort=False).last()
    abbrev_map = canonical["team_abbrev"].to_dict()
    name_map = canonical["team_name"].to_dict()
    games["home_team_abbrev"] = games["home_team_id"].map(abbrev_map)
    games["away_team_abbrev"] = games["away_team_id"].map(abbrev_map)
    games["home_team_name"] = games["home_team_id"].map(name_map)
    games["away_team_name"] = games["away_team_id"].map(name_map)
    coverage = {
        "source_rows": source_rows,
        "source_duplicate_game_ids": duplicate_source_ids,
        "final_schedule_records": int(len(schedule_supplement)),
        "schedule_final_games_added_to_processed_finals": int(supplemental_games_added),
        "final_regular_games": len(games),
        "first_final_date": games["game_date"].min().date().isoformat(),
        "last_final_date": games["game_date"].max().date().isoformat(),
        "seasons": sorted(games["season"].unique().tolist()),
        "final_games_by_season": {
            str(int(k)): int(v) for k, v in games.groupby("season").size().items()
        },
    }
    return games, coverage


def add_schedule_metadata(games: pd.DataFrame, metadata: dict[int, dict[str, object]]) -> pd.DataFrame:
    meta_rows = []
    for game_pk, record in metadata.items():
        meta_rows.append(
            {
                "game_pk": game_pk,
                "meta_games_in_series": record.get("games_in_series"),
                "meta_series_game_number": record.get("series_game_number"),
                "meta_series_description": record.get("series_description"),
                "meta_source_file": record.get("source_file"),
            }
        )
    if not meta_rows:
        for col in (
            "meta_games_in_series",
            "meta_series_game_number",
            "meta_series_description",
            "meta_source_file",
        ):
            games[col] = np.nan
        return games
    meta = pd.DataFrame(meta_rows)
    return games.merge(meta, on="game_pk", how="left", validate="one_to_one")


def make_team_games(games: pd.DataFrame) -> pd.DataFrame:
    home = games[
        [
            "game_pk",
            "season",
            "game_date",
            "sort_time",
            "home_team_id",
            "home_team_abbrev",
            "home_team_name",
            "away_team_id",
            "away_team_abbrev",
            "away_team_name",
            "home_score",
            "away_score",
        ]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_abbrev": "team",
            "home_team_name": "team_name",
            "away_team_id": "opponent_id",
            "away_team_abbrev": "opponent",
            "away_team_name": "opponent_name",
            "home_score": "runs_for",
            "away_score": "runs_against",
        }
    )
    home["is_home"] = True
    away = games[
        [
            "game_pk",
            "season",
            "game_date",
            "sort_time",
            "away_team_id",
            "away_team_abbrev",
            "away_team_name",
            "home_team_id",
            "home_team_abbrev",
            "home_team_name",
            "away_score",
            "home_score",
        ]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_abbrev": "team",
            "away_team_name": "team_name",
            "home_team_id": "opponent_id",
            "home_team_abbrev": "opponent",
            "home_team_name": "opponent_name",
            "away_score": "runs_for",
            "home_score": "runs_against",
        }
    )
    away["is_home"] = False
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["win"] = (team_games["runs_for"] > team_games["runs_against"]).astype(int)
    team_games["run_diff"] = team_games["runs_for"] - team_games["runs_against"]
    team_games = team_games.sort_values(
        ["team", "season", "sort_time", "game_pk"], kind="mergesort"
    ).reset_index(drop=True)
    team_games["team_game_number"] = team_games.groupby(["team", "season"]).cumcount() + 1
    return team_games


def identify_three_game_series(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = games.copy()
    work["oriented_matchup"] = (
        work["season"].astype(str)
        + "|"
        + work["away_team_id"].astype("Int64").astype(str)
        + "@"
        + work["home_team_id"].astype("Int64").astype(str)
    )
    work = work.sort_values(["oriented_matchup", "game_date", "sort_time", "game_pk"], kind="mergesort")
    gap = work.groupby("oriented_matchup")["game_date"].diff().dt.days.fillna(99)
    work["series_sequence"] = (gap > 3).astype(int).groupby(work["oriented_matchup"]).cumsum()
    work["series_id"] = work["oriented_matchup"] + "|" + work["series_sequence"].astype(str)
    work["inferred_series_game_number"] = work.groupby("series_id").cumcount() + 1
    group_size = work.groupby("series_id")["game_pk"].transform("size")
    candidates = work.loc[group_size.eq(3)].copy()

    eligibility_rows = []
    eligible_ids: list[str] = []
    for series_id, group in candidates.groupby("series_id", sort=False):
        totals = pd.to_numeric(group["meta_games_in_series"], errors="coerce").dropna().astype(int)
        numbers = pd.to_numeric(group["meta_series_game_number"], errors="coerce").dropna().astype(int)
        has_conflict = bool((totals != 3).any())
        if has_conflict:
            quality = "EXCLUIDA: metadata indica otra longitud"
            eligible = False
        elif len(totals) == 3 and set(numbers.tolist()) == {1, 2, 3}:
            quality = "CONFIRMADA"
            eligible = True
        elif len(totals) > 0:
            quality = "RESPALDADA PARCIALMENTE"
            eligible = True
        else:
            quality = "INFERIDA"
            eligible = True
        eligibility_rows.append(
            {
                "series_id": series_id,
                "eligible": int(eligible),
                "series_quality": quality,
                "metadata_games": int(len(totals)),
                "metadata_numbers": ",".join(map(str, sorted(numbers.unique().tolist()))),
                "metadata_lengths": ",".join(map(str, sorted(totals.unique().tolist()))),
            }
        )
        if eligible:
            eligible_ids.append(series_id)

    eligibility = pd.DataFrame(eligibility_rows)
    eligible_games = candidates.loc[candidates["series_id"].isin(eligible_ids)].copy()
    eligible_games = eligible_games.merge(
        eligibility[["series_id", "series_quality"]], on="series_id", how="left", validate="many_to_one"
    )
    return eligible_games, eligibility


def build_series_tables(series_games: pd.DataFrame, team_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_rows: list[dict[str, object]] = []
    team_rows: list[dict[str, object]] = []

    team_games_by_key: dict[tuple[str, int], pd.DataFrame] = {
        key: group.reset_index(drop=True)
        for key, group in team_games.groupby(["team", "season"], sort=False)
    }
    team_index: dict[tuple[str, int, int], int] = {}
    for (team, season), group in team_games_by_key.items():
        for idx, game_pk in enumerate(group["game_pk"].astype(int).tolist()):
            team_index[(team, int(season), game_pk)] = idx

    for series_id, group in series_games.groupby("series_id", sort=False):
        g = group.sort_values(["sort_time", "game_pk"], kind="mergesort").reset_index(drop=True)
        home_team = str(g.iloc[0]["home_team_abbrev"])
        away_team = str(g.iloc[0]["away_team_abbrev"])
        home_name = str(g.iloc[0]["home_team_name"])
        away_name = str(g.iloc[0]["away_team_name"])
        season = int(g.iloc[0]["season"])
        home_wins = int(g["home_win"].sum())
        away_wins = 3 - home_wins
        sweep_team = home_team if home_wins == 3 else away_team if away_wins == 3 else None
        swept_team = away_team if home_wins == 3 else home_team if away_wins == 3 else None
        home_runs = int(g["home_score"].sum())
        away_runs = int(g["away_score"].sum())
        scoreline = " | ".join(
            f"{row.away_team_abbrev} {int(row.away_score)}-{int(row.home_score)} {row.home_team_abbrev}"
            for row in g.itertuples()
        )
        game_pks = ",".join(str(int(x)) for x in g["game_pk"].tolist())
        series_row = {
            "series_id": series_id,
            "season": season,
            "start_date": g["game_date"].min().date().isoformat(),
            "end_date": g["game_date"].max().date().isoformat(),
            "away_team": away_team,
            "away_team_name": away_name,
            "home_team": home_team,
            "home_team_name": home_name,
            "away_wins": away_wins,
            "home_wins": home_wins,
            "away_runs": away_runs,
            "home_runs": home_runs,
            "sweep": int(sweep_team is not None),
            "sweep_team": sweep_team,
            "swept_team": swept_team,
            "series_quality": str(g.iloc[0]["series_quality"]),
            "scoreline": scoreline,
            "game_pks": game_pks,
        }
        series_rows.append(series_row)

        last_game_pk = int(g.iloc[-1]["game_pk"])
        for team, team_name, opponent, opponent_name, wins, runs_for, runs_against, location in (
            (home_team, home_name, away_team, away_name, home_wins, home_runs, away_runs, "Casa"),
            (away_team, away_name, home_team, home_name, away_wins, away_runs, home_runs, "Visitante"),
        ):
            key = (team, season)
            idx = team_index[(team, season, last_game_pk)]
            schedule = team_games_by_key[key]
            future = schedule.iloc[idx + 1 : idx + 4]
            next_game = future.iloc[0] if len(future) else None
            row: dict[str, object] = {
                "series_id": series_id,
                "season": season,
                "start_date": series_row["start_date"],
                "end_date": series_row["end_date"],
                "team": team,
                "team_name": team_name,
                "opponent": opponent,
                "opponent_name": opponent_name,
                "series_location": location,
                "series_wins": wins,
                "series_losses": 3 - wins,
                "sweep_for": int(wins == 3),
                "swept_against": int(wins == 0),
                "series_runs_for": runs_for,
                "series_runs_against": runs_against,
                "series_run_diff": runs_for - runs_against,
                "series_quality": series_row["series_quality"],
                "game_pks": game_pks,
                "scoreline": scoreline,
            }
            if next_game is None:
                row.update(
                    {
                        "next_game_available": 0,
                        "next_game_date": None,
                        "days_to_next_game": None,
                        "next_opponent": None,
                        "next_location": None,
                        "next_win": None,
                        "next_runs_for": None,
                        "next_runs_against": None,
                        "next_run_diff": None,
                    }
                )
            else:
                row.update(
                    {
                        "next_game_available": 1,
                        "next_game_date": next_game["game_date"].date().isoformat(),
                        "days_to_next_game": int((next_game["game_date"] - g["game_date"].max()).days),
                        "next_opponent": str(next_game["opponent"]),
                        "next_location": "Casa" if bool(next_game["is_home"]) else "Visitante",
                        "next_win": int(next_game["win"]),
                        "next_runs_for": int(next_game["runs_for"]),
                        "next_runs_against": int(next_game["runs_against"]),
                        "next_run_diff": int(next_game["run_diff"]),
                    }
                )
            row["next3_games"] = int(len(future))
            row["next3_wins"] = int(future["win"].sum()) if len(future) else 0
            row["next3_losses"] = int(len(future) - future["win"].sum()) if len(future) else 0
            team_rows.append(row)

    series = pd.DataFrame(series_rows).sort_values(["end_date", "series_id"]).reset_index(drop=True)
    team_series = pd.DataFrame(team_rows).sort_values(["team", "end_date", "series_id"]).reset_index(drop=True)
    return series, team_series


def summarize_teams(team_series: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    team_names = (
        team_games.sort_values(["game_date", "sort_time"]).groupby("team")["team_name"].last().to_dict()
    )
    for team in sorted(team_games["team"].unique()):
        s = team_series.loc[team_series["team"].eq(team)]
        g = team_games.loc[team_games["team"].eq(team)]
        opportunities = len(s)
        sweeps = int(s["sweep_for"].sum())
        swept = int(s["swept_against"].sum())
        sweep_low, sweep_high = wilson_interval(sweeps, opportunities)
        swept_low, swept_high = wilson_interval(swept, opportunities)
        after_sweep = s.loc[s["sweep_for"].eq(1) & s["next_game_available"].eq(1)]
        after_swept = s.loc[s["swept_against"].eq(1) & s["next_game_available"].eq(1)]
        home_series = s.loc[s["series_location"].eq("Casa")]
        road_series = s.loc[s["series_location"].eq("Visitante")]
        baseline_win_rate = float(g["win"].mean()) if len(g) else math.nan
        next_win_rate = float(after_sweep["next_win"].mean()) if len(after_sweep) else math.nan
        rebound_win_rate = float(after_swept["next_win"].mean()) if len(after_swept) else math.nan
        next3_games = int(after_sweep["next3_games"].sum())
        next3_wins = int(after_sweep["next3_wins"].sum())
        rows.append(
            {
                "team": team,
                "team_name": team_names.get(team, team),
                "three_game_series": opportunities,
                "sweeps": sweeps,
                "sweep_rate": sweeps / opportunities if opportunities else math.nan,
                "sweep_rate_ci_low": sweep_low,
                "sweep_rate_ci_high": sweep_high,
                "times_swept": swept,
                "swept_rate": swept / opportunities if opportunities else math.nan,
                "swept_rate_ci_low": swept_low,
                "swept_rate_ci_high": swept_high,
                "net_sweeps": sweeps - swept,
                "net_sweep_rate": (sweeps - swept) / opportunities if opportunities else math.nan,
                "game_wins": int(g["win"].sum()),
                "games": len(g),
                "baseline_win_rate": baseline_win_rate,
                "after_sweep_games": len(after_sweep),
                "after_sweep_wins": int(after_sweep["next_win"].sum()),
                "after_sweep_win_rate": next_win_rate,
                "after_sweep_delta_vs_baseline": next_win_rate - baseline_win_rate
                if len(after_sweep)
                else math.nan,
                "after_sweep_next3_games": next3_games,
                "after_sweep_next3_wins": next3_wins,
                "after_sweep_next3_win_rate": next3_wins / next3_games if next3_games else math.nan,
                "after_swept_games": len(after_swept),
                "after_swept_wins": int(after_swept["next_win"].sum()),
                "after_swept_win_rate": rebound_win_rate,
                "after_swept_delta_vs_baseline": rebound_win_rate - baseline_win_rate
                if len(after_swept)
                else math.nan,
                "home_three_game_series": len(home_series),
                "home_sweeps": int(home_series["sweep_for"].sum()),
                "home_sweep_rate": float(home_series["sweep_for"].mean()) if len(home_series) else math.nan,
                "road_three_game_series": len(road_series),
                "road_sweeps": int(road_series["sweep_for"].sum()),
                "road_sweep_rate": float(road_series["sweep_for"].mean()) if len(road_series) else math.nan,
                "avg_run_diff_in_sweeps": float(s.loc[s["sweep_for"].eq(1), "series_run_diff"].mean())
                if sweeps
                else math.nan,
                "unique_opponents_swept": int(s.loc[s["sweep_for"].eq(1), "opponent"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def summarize_team_seasons(team_series: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    team_names = team_games.groupby(["season", "team"])["team_name"].last().to_dict()
    for (season, team), s in team_series.groupby(["season", "team"], sort=True):
        g = team_games.loc[team_games["season"].eq(season) & team_games["team"].eq(team)]
        after = s.loc[s["sweep_for"].eq(1) & s["next_game_available"].eq(1)]
        after_swept = s.loc[s["swept_against"].eq(1) & s["next_game_available"].eq(1)]
        opportunities = len(s)
        sweeps = int(s["sweep_for"].sum())
        swept = int(s["swept_against"].sum())
        rows.append(
            {
                "season": int(season),
                "team": team,
                "team_name": team_names.get((season, team), team),
                "three_game_series": opportunities,
                "sweeps": sweeps,
                "sweep_rate": sweeps / opportunities if opportunities else math.nan,
                "times_swept": swept,
                "swept_rate": swept / opportunities if opportunities else math.nan,
                "net_sweeps": sweeps - swept,
                "games": len(g),
                "game_wins": int(g["win"].sum()),
                "baseline_win_rate": float(g["win"].mean()) if len(g) else math.nan,
                "after_sweep_games": len(after),
                "after_sweep_wins": int(after["next_win"].sum()),
                "after_sweep_win_rate": float(after["next_win"].mean()) if len(after) else math.nan,
                "after_swept_games": len(after_swept),
                "after_swept_wins": int(after_swept["next_win"].sum()),
                "after_swept_win_rate": float(after_swept["next_win"].mean()) if len(after_swept) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_league_seasons(series: pd.DataFrame, team_series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, all_s in series.groupby("season", sort=True):
        ts = team_series.loc[team_series["season"].eq(season)]
        sweepers = ts.loc[ts["sweep_for"].eq(1)]
        swept = ts.loc[ts["swept_against"].eq(1)]
        after = sweepers.loc[sweepers["next_game_available"].eq(1)]
        rebound = swept.loc[swept["next_game_available"].eq(1)]
        next3_games = int(after["next3_games"].sum())
        next3_wins = int(after["next3_wins"].sum())
        rows.append(
            {
                "season": int(season),
                "three_game_series": len(all_s),
                "swept_series": int(all_s["sweep"].sum()),
                "sweep_frequency": float(all_s["sweep"].mean()),
                "after_sweep_games": len(after),
                "after_sweep_wins": int(after["next_win"].sum()),
                "after_sweep_win_rate": float(after["next_win"].mean()) if len(after) else math.nan,
                "after_sweep_next3_games": next3_games,
                "after_sweep_next3_wins": next3_wins,
                "after_sweep_next3_win_rate": next3_wins / next3_games if next3_games else math.nan,
                "after_swept_games": len(rebound),
                "after_swept_wins": int(rebound["next_win"].sum()),
                "after_swept_win_rate": float(rebound["next_win"].mean()) if len(rebound) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def round_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    float_cols = result.select_dtypes(include=["float", "float64", "float32"]).columns
    result[float_cols] = result[float_cols].round(8)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MLB sweeps in inferred three-game regular-season series.")
    parser.add_argument("--games", type=Path, default=DEFAULT_GAMES)
    parser.add_argument("--schedule-root", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata, schedule_supplement = load_schedule_metadata(args.schedule_root)
    games, coverage = load_final_regular_games(args.games, schedule_supplement)
    games = add_schedule_metadata(games, metadata)
    team_games = make_team_games(games)
    series_games, eligibility = identify_three_game_series(games)
    series, team_series = build_series_tables(series_games, team_games)
    team_summary = summarize_teams(team_series, team_games)
    team_season = summarize_team_seasons(team_series, team_games)
    league_season = summarize_league_seasons(series, team_series)

    ranking_good = team_summary.sort_values(
        ["net_sweep_rate", "sweep_rate", "sweeps", "team"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking_good.insert(0, "overall_rank", np.arange(1, len(ranking_good) + 1))
    ranking_sweepers = team_summary.sort_values(
        ["sweep_rate", "sweeps", "net_sweep_rate", "team"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking_sweepers.insert(0, "sweeper_rank", np.arange(1, len(ranking_sweepers) + 1))
    ranking_vulnerable = team_summary.sort_values(
        ["swept_rate", "times_swept", "net_sweep_rate", "team"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking_vulnerable.insert(0, "vulnerability_rank", np.arange(1, len(ranking_vulnerable) + 1))
    ranking_low_sweep = team_summary.sort_values(
        ["sweep_rate", "sweeps", "net_sweep_rate", "team"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking_low_sweep.insert(0, "low_sweep_rank", np.arange(1, len(ranking_low_sweep) + 1))

    sweep_events = team_series.loc[team_series["sweep_for"].eq(1)]
    swept_events = team_series.loc[team_series["swept_against"].eq(1)]
    after_sweep = sweep_events.loc[sweep_events["next_game_available"].eq(1)]
    after_swept = swept_events.loc[swept_events["next_game_available"].eq(1)]
    next3_games = int(after_sweep["next3_games"].sum())
    next3_wins = int(after_sweep["next3_wins"].sum())

    quality_counts = series["series_quality"].value_counts().to_dict()
    checks = {
        "teams_found": int(team_games["team"].nunique()),
        "eligible_series_have_three_games": bool(series_games.groupby("series_id").size().eq(3).all()),
        "team_rows_equal_two_per_series": bool(len(team_series) == 2 * len(series)),
        "sweeps_equal_times_swept": bool(team_summary["sweeps"].sum() == team_summary["times_swept"].sum()),
        "sweep_events_equal_series_sweeps": bool(len(sweep_events) == int(series["sweep"].sum())),
        "unique_final_game_ids": bool(not games["game_pk"].duplicated().any()),
    }
    if not all(checks.values()) or checks["teams_found"] != 30:
        raise RuntimeError(f"Quality checks failed: {checks}")

    summary = {
        "definition": {
            "scope": "MLB regular-season games with final scores",
            "series_rule": "Same oriented matchup and season; consecutive completed games no more than 3 calendar days apart; exactly 3 completed games; exclude when MLB schedule metadata says a different series length.",
            "sweep_rule": "One team wins all 3 games.",
            "next_game_rule": "The team's next completed regular-season game in the same season.",
            "ranking_rule": "Overall good-to-bad ranking uses net sweep rate = (sweeps - times swept) / three-game series. Dedicated rankings also show sweep rate and vulnerability separately.",
        },
        "coverage": coverage,
        "series": {
            "eligible_three_game_series": int(len(series)),
            "swept_three_game_series": int(series["sweep"].sum()),
            "league_sweep_frequency": float(series["sweep"].mean()),
            "series_quality_counts": {str(k): int(v) for k, v in quality_counts.items()},
            "candidate_groups_of_three": int(len(eligibility)),
            "excluded_by_schedule_metadata": int((eligibility["eligible"].eq(0)).sum()),
        },
        "after_sweep": {
            "sweeps_with_next_game": int(len(after_sweep)),
            "next_game_wins": int(after_sweep["next_win"].sum()),
            "next_game_win_rate": float(after_sweep["next_win"].mean()),
            "next_three_games": next3_games,
            "next_three_wins": next3_wins,
            "next_three_win_rate": next3_wins / next3_games,
        },
        "after_being_swept": {
            "events_with_next_game": int(len(after_swept)),
            "next_game_wins": int(after_swept["next_win"].sum()),
            "next_game_win_rate": float(after_swept["next_win"].mean()),
        },
        "top_overall": ranking_good.head(10)[
            ["overall_rank", "team", "team_name", "three_game_series", "sweeps", "sweep_rate", "times_swept", "swept_rate", "net_sweep_rate"]
        ].to_dict(orient="records"),
        "top_sweepers": ranking_sweepers.head(10)[
            ["sweeper_rank", "team", "team_name", "three_game_series", "sweeps", "sweep_rate"]
        ].to_dict(orient="records"),
        "most_vulnerable": ranking_vulnerable.head(10)[
            ["vulnerability_rank", "team", "team_name", "three_game_series", "times_swept", "swept_rate"]
        ].to_dict(orient="records"),
        "lowest_sweep_rates": ranking_low_sweep.head(10)[
            ["low_sweep_rank", "team", "team_name", "three_game_series", "sweeps", "sweep_rate"]
        ].to_dict(orient="records"),
        "checks": checks,
    }

    outputs = {
        "games_final_regular.csv": team_games[
            [
                "game_pk",
                "season",
                "game_date",
                "team",
                "team_name",
                "opponent",
                "opponent_name",
                "is_home",
                "runs_for",
                "runs_against",
                "win",
                "run_diff",
            ]
        ],
        "all_three_game_series.csv": series,
        "team_series.csv": team_series,
        "ranking_good_to_bad.csv": ranking_good,
        "ranking_best_sweepers.csv": ranking_sweepers,
        "ranking_most_vulnerable.csv": ranking_vulnerable,
        "ranking_lowest_sweep_rate.csv": ranking_low_sweep,
        "team_season.csv": team_season,
        "league_season.csv": league_season,
        "series_eligibility_qc.csv": eligibility,
    }
    for filename, frame in outputs.items():
        round_for_csv(frame).to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    workbook_data = {
        "summary": summary,
        "ranking_good": json.loads(ranking_good.to_json(orient="records", date_format="iso")),
        "ranking_sweepers": json.loads(ranking_sweepers.to_json(orient="records", date_format="iso")),
        "ranking_vulnerable": json.loads(ranking_vulnerable.to_json(orient="records", date_format="iso")),
        "ranking_low_sweep": json.loads(ranking_low_sweep.to_json(orient="records", date_format="iso")),
        "team_season": json.loads(team_season.to_json(orient="records", date_format="iso")),
        "league_season": json.loads(league_season.to_json(orient="records", date_format="iso")),
        "all_series": json.loads(series.to_json(orient="records", date_format="iso")),
        "team_series": json.loads(team_series.to_json(orient="records", date_format="iso")),
        "team_games": json.loads(
            team_games[
                [
                    "game_pk",
                    "season",
                    "game_date",
                    "team",
                    "team_name",
                    "opponent",
                    "opponent_name",
                    "is_home",
                    "runs_for",
                    "runs_against",
                    "win",
                    "run_diff",
                ]
            ].to_json(orient="records", date_format="iso")
        ),
    }
    (args.output_dir / "workbook_data.json").write_text(
        json.dumps(workbook_data, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
