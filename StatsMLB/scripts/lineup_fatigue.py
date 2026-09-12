from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    "avgConsecutiveStarts",
    "weightedConsecutiveStarts",
    "longStreakCount",
    "startsLast3Days",
    "paLast3Days",
    "paLast7Days",
    "shortRestCount",
    "starHeavyCount",
    "weightedFatigue",
    "previousExtraInnings",
    "previousDoubleheader",
    "lossWeightedFatigue",
    "winWeightedFatigue",
]

TEST_FEATURES = [
    "edge_startsLast3Days",
    "edge_paLast3Days",
    "edge_paLast7Days",
    "edge_shortRestCount",
]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def attach_lineup_fatigue_features(root: Path, feature_rows: pd.DataFrame, cutoff_date: str):
    """Attach lineup workload known before first pitch and export live player states."""
    game_columns = [
        "game_pk", "first_pitch_utc", "final_innings", "double_header", "game_number",
    ]
    game_times = pd.read_parquet(root / "data" / "processed" / "games.parquet", columns=game_columns)
    game_times = game_times.drop_duplicates("game_pk")
    meta = feature_rows[["gamePk", "date", "season", "home", "away", "homeWin"]].merge(
        game_times, left_on="gamePk", right_on="game_pk", how="left"
    )
    meta["sortTime"] = pd.to_datetime(meta["first_pitch_utc"], utc=True, errors="coerce")
    fallback = pd.to_datetime(meta["date"], utc=True) + pd.to_timedelta(meta.groupby("date").cumcount(), unit="s")
    meta["sortTime"] = meta["sortTime"].fillna(fallback)
    meta["extraInnings"] = pd.to_numeric(meta["final_innings"], errors="coerce").gt(9).astype(float)
    double_header = meta["double_header"].astype(str).str.upper().ne("N") | pd.to_numeric(
        meta["game_number"], errors="coerce"
    ).fillna(1).gt(1)
    meta["doubleheaderGame"] = double_header.astype(float)

    columns = [
        "game_pk", "side", "player_id", "position", "started_batting", "batting_order",
        "appeared_batting", "bat_plateAppearances", "bat_atBats", "bat_hits", "bat_doubles",
        "bat_triples", "bat_homeRuns", "bat_baseOnBalls", "bat_hitByPitch", "bat_sacFlies",
    ]
    players = pd.read_parquet(root / "data" / "processed" / "player_box.parquet", columns=columns)
    players["side"] = players["side"].astype(str).str.lower()
    players = players.merge(
        meta[["gamePk", "sortTime", "home", "away", "homeWin"]],
        left_on="game_pk", right_on="gamePk", how="inner",
    )
    players["team"] = np.where(players["side"].eq("home"), players["home"], players["away"])
    numeric = [
        "bat_plateAppearances", "bat_atBats", "bat_hits", "bat_doubles", "bat_triples",
        "bat_homeRuns", "bat_baseOnBalls", "bat_hitByPitch", "bat_sacFlies",
    ]
    for column in numeric:
        players[column] = pd.to_numeric(players[column], errors="coerce").fillna(0.0)
    players["started"] = players["started_batting"].eq(True).astype(float)
    players["appeared"] = (
        players["appeared_batting"].eq(True) | players["started_batting"].eq(True)
        | players["bat_plateAppearances"].gt(0)
    ).astype(float)
    singles = (players["bat_hits"] - players["bat_doubles"] - players["bat_triples"] - players["bat_homeRuns"]).clip(lower=0)
    players["woba_num"] = (
        .69 * players["bat_baseOnBalls"] + .72 * players["bat_hitByPitch"] + .89 * singles
        + 1.27 * players["bat_doubles"] + 1.62 * players["bat_triples"] + 2.10 * players["bat_homeRuns"]
    )
    players["woba_den"] = (
        players["bat_atBats"] + players["bat_baseOnBalls"]
        + players["bat_hitByPitch"] + players["bat_sacFlies"]
    )
    players = players.sort_values(["team", "player_id", "sortTime", "game_pk"]).reset_index(drop=True)
    grouped = players.groupby(["team", "player_id"], sort=False)
    players["prior_woba_num"] = grouped["woba_num"].cumsum() - players["woba_num"]
    players["prior_woba_den"] = grouped["woba_den"].cumsum() - players["woba_den"]
    players["prior_starts"] = grouped["started"].cumsum() - players["started"]
    players["prior_appearances"] = grouped["appeared"].cumsum() - players["appeared"]
    players["prior_quality"] = (
        (players["prior_woba_num"] + .320 * 50) / (players["prior_woba_den"] + 50)
        + .06 * (players["prior_starts"] + 1.5) / (players["prior_appearances"] + 3)
    )
    players["post_quality"] = (
        (players["prior_woba_num"] + players["woba_num"] + .320 * 50)
        / (players["prior_woba_den"] + players["woba_den"] + 50)
        + .06 * (players["prior_starts"] + players["started"] + 1.5)
        / (players["prior_appearances"] + players["appeared"] + 3)
    )
    players = players.sort_values(["sortTime", "game_pk", "side", "player_id"]).reset_index(drop=True)

    groups = {(int(game_pk), str(side)): group for (game_pk, side), group in players.groupby(["game_pk", "side"], sort=False)}
    history: dict[str, dict[int, dict]] = defaultdict(dict)
    team_index: dict[str, int] = defaultdict(int)
    previous_team_game: dict[str, dict] = {}
    latest_quality: dict[tuple[str, int], float] = {}
    team_rows = []

    for game in meta.sort_values(["sortTime", "gamePk"]).itertuples():
        for side in ("home", "away"):
            group = groups.get((int(game.gamePk), side))
            if group is None or group.empty:
                continue
            team = str(getattr(game, side))
            index = team_index[team]
            starters = group[
                group["started_batting"].eq(True)
                & ~group["position"].astype(str).eq("P")
                & group["player_id"].notna()
            ].copy()
            if len(starters) < 8:
                continue
            starters["player_id"] = starters["player_id"].astype(int)
            now = game.sortTime
            player_metrics = []
            for row in starters.itertuples():
                player_id = int(row.player_id)
                state = history[team].get(player_id, {"events": deque(), "lastStartIndex": -99, "streak": 0})
                events = list(state["events"])
                recent3 = [event for event in events if (now - event[0]).total_seconds() <= 72 * 3600]
                recent7 = [event for event in events if (now - event[0]).total_seconds() <= 168 * 3600]
                last_at = events[-1][0] if events else None
                consecutive = float(state["streak"] if state["lastStartIndex"] == index - 1 else 0)
                starts3 = float(sum(event[1] for event in recent3))
                pa3 = float(sum(event[2] for event in recent3))
                pa7 = float(sum(event[2] for event in recent7))
                short_rest = float(last_at is not None and (now - last_at).total_seconds() <= 30 * 3600)
                quality = float(row.prior_quality)
                fatigue = .32 * consecutive + .28 * starts3 + .20 * (pa3 / 4.3) + .12 * (pa7 / 12.9) + .65 * short_rest
                player_metrics.append({
                    "playerId": player_id, "quality": quality, "consecutive": consecutive,
                    "starts3": starts3, "pa3": pa3, "pa7": pa7, "shortRest": short_rest,
                    "fatigue": fatigue,
                })

            qualities = np.asarray([item["quality"] for item in player_metrics], dtype=float)
            weights = np.maximum(qualities, .01)
            fatigue_values = np.asarray([item["fatigue"] for item in player_metrics], dtype=float)
            star_cut = float(np.quantile(qualities, .67))
            signals = {
                "avgConsecutiveStarts": _mean([item["consecutive"] for item in player_metrics]),
                "weightedConsecutiveStarts": float(np.average([item["consecutive"] for item in player_metrics], weights=weights)),
                "longStreakCount": float(sum(item["consecutive"] >= 5 for item in player_metrics)),
                "startsLast3Days": _mean([item["starts3"] for item in player_metrics]),
                "paLast3Days": _mean([item["pa3"] for item in player_metrics]),
                "paLast7Days": _mean([item["pa7"] for item in player_metrics]),
                "shortRestCount": float(sum(item["shortRest"] for item in player_metrics)),
                "starHeavyCount": float(sum(item["quality"] >= star_cut and item["fatigue"] >= 2.25 for item in player_metrics)),
                "weightedFatigue": float(np.average(fatigue_values, weights=weights)),
                "previousExtraInnings": float(previous_team_game.get(team, {}).get("extraInnings", 0.0)),
                "previousDoubleheader": float(previous_team_game.get(team, {}).get("doubleheaderGame", 0.0)),
            }
            previous_win = previous_team_game.get(team, {}).get("win")
            signals["lossWeightedFatigue"] = float(previous_win == 0) * signals["weightedFatigue"]
            signals["winWeightedFatigue"] = float(previous_win == 1) * signals["weightedFatigue"]
            win = int(game.homeWin if side == "home" else 1 - game.homeWin)
            team_rows.append({
                "gamePk": int(game.gamePk), "date": str(game.date), "season": int(game.season),
                "sortTime": now, "side": side, "team": team, "win": win,
                "previousWin": previous_win, **signals,
            })

            for row in group.itertuples():
                if pd.isna(row.player_id) or str(row.position) == "P":
                    continue
                player_id = int(row.player_id)
                state = history[team].setdefault(player_id, {"events": deque(), "lastStartIndex": -99, "streak": 0})
                if bool(row.appeared):
                    state["events"].append((now, int(bool(row.started)), float(row.bat_plateAppearances)))
                while state["events"] and (now - state["events"][0][0]).total_seconds() > 8 * 24 * 3600:
                    state["events"].popleft()
                if bool(row.started) and str(row.position) != "P":
                    state["streak"] = state["streak"] + 1 if state["lastStartIndex"] == index - 1 else 1
                    state["lastStartIndex"] = index
                latest_quality[(team, player_id)] = float(row.post_quality)
            previous_team_game[team] = {
                "win": win, "extraInnings": float(game.extraInnings),
                "doubleheaderGame": float(game.doubleheaderGame), "playedAt": now,
            }
            team_index[team] += 1

    team_frame = pd.DataFrame(team_rows)
    home = team_frame[team_frame["side"].eq("home")][["gamePk", *SIGNAL_COLUMNS]].rename(
        columns={name: f"home_{name}" for name in SIGNAL_COLUMNS}
    )
    away = team_frame[team_frame["side"].eq("away")][["gamePk", *SIGNAL_COLUMNS]].rename(
        columns={name: f"away_{name}" for name in SIGNAL_COLUMNS}
    )
    joined = home.merge(away, on="gamePk", how="outer")
    for name in SIGNAL_COLUMNS:
        joined[f"edge_{name}"] = joined[f"home_{name}"] - joined[f"away_{name}"]
    result = feature_rows.merge(joined, on="gamePk", how="left")

    team_frame["fatigueBucket"] = pd.cut(
        team_frame["weightedFatigue"], [-np.inf, 1.2, 2.0, np.inf],
        labels=["Carga baja", "Carga media", "Carga alta"],
    )
    descriptive = []
    for bucket in ("Carga baja", "Carga media", "Carga alta"):
        sample = team_frame[team_frame["fatigueBucket"].eq(bucket)]
        latest = sample[sample["season"].eq(2026)]
        descriptive.append({
            "label": bucket, "games": len(sample), "wins": int(sample["win"].sum()),
            "winRate": sample["win"].mean(), "winRate2026": latest["win"].mean(),
        })

    latest_player_workloads: dict[str, dict[str, dict]] = defaultdict(dict)
    for team, team_players in history.items():
        current_index = team_index[team]
        for player_id, state in team_players.items():
            events = list(state["events"])
            latest_player_workloads[team][str(player_id)] = {
                "streak": int(state["streak"] if state["lastStartIndex"] == current_index - 1 else 0),
                "lastStartIndex": int(state["lastStartIndex"]),
                "teamGameIndex": int(current_index),
                "events": [[event[0].isoformat(), int(event[1]), float(event[2])] for event in events],
                "quality": float(latest_quality.get((team, player_id), .35)),
            }
    latest_team_states = {
        team: {
            "win": int(value["win"]), "extraInnings": float(value["extraInnings"]),
            "doubleheaderGame": float(value["doubleheaderGame"]), "playedAt": value["playedAt"].isoformat(),
        }
        for team, value in previous_team_game.items()
    }
    audit = {
        "method": "Carga individual del lineup calculada antes del primer lanzamiento con titularidades, PA y descanso de 3/7 dias; monthly expanding walk-forward.",
        "cutoffDate": cutoff_date,
        "coverage": {
            "teamGames": len(team_frame), "lineupComparisons": int(team_frame["weightedFatigue"].notna().sum()),
            "trackedPlayers": sum(len(value) for value in latest_player_workloads.values()),
        },
        "descriptive": descriptive,
        "latestPlayerWorkloads": dict(latest_player_workloads),
        "latestTeamStates": latest_team_states,
        "candidate": None,
        "combinedWithRotationQuality": None,
        "recommendation": "PRUEBA: solo se aplica cuando MLB confirma ambos lineups y al menos siete bateadores tienen historial de carga.",
    }
    return result, audit
