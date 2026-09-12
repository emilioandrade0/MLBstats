from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    "qualityDelta",
    "replacementDelta",
    "retentionQuality",
    "top4QualityDelta",
    "lossQualityDelta",
    "lossReplacementDelta",
    "lossRetentionQuality",
    "lossTop4QualityDelta",
]

TEST_FEATURES = [f"edge_{column}" for column in SIGNAL_COLUMNS]


def attach_rotation_quality_features(root: Path, feature_rows: pd.DataFrame, cutoff_date: str):
    """Build pregame-only lineup-quality deltas and the live player-rating snapshot."""
    game_times = pd.read_parquet(
        root / "data" / "processed" / "games.parquet",
        columns=["game_pk", "first_pitch_utc"],
    ).drop_duplicates("game_pk")
    meta = feature_rows[["gamePk", "date", "season", "home", "away", "homeWin"]].merge(
        game_times, left_on="gamePk", right_on="game_pk", how="left"
    )
    meta["sortTime"] = pd.to_datetime(meta["first_pitch_utc"], utc=True, errors="coerce")
    fallback = pd.to_datetime(meta["date"], utc=True) + pd.to_timedelta(meta.groupby("date").cumcount(), unit="s")
    meta["sortTime"] = meta["sortTime"].fillna(fallback)

    columns = [
        "game_pk", "side", "player_id", "position", "started_batting", "batting_order",
        "bat_plateAppearances", "bat_atBats", "bat_hits", "bat_doubles", "bat_triples", "bat_homeRuns",
        "bat_baseOnBalls", "bat_hitByPitch", "bat_sacFlies",
    ]
    players = pd.read_parquet(root / "data" / "processed" / "player_box.parquet", columns=columns)
    players["side"] = players["side"].astype(str).str.lower()
    players = players.merge(
        meta[["gamePk", "sortTime", "home", "away", "homeWin"]],
        left_on="game_pk", right_on="gamePk", how="inner",
    )
    players["team"] = np.where(players["side"].eq("home"), players["home"], players["away"])
    players = players.sort_values(["team", "player_id", "sortTime", "game_pk"]).reset_index(drop=True)
    numeric = [
        "bat_plateAppearances", "bat_atBats", "bat_hits", "bat_doubles", "bat_triples",
        "bat_homeRuns", "bat_baseOnBalls", "bat_hitByPitch", "bat_sacFlies",
    ]
    for column in numeric:
        players[column] = pd.to_numeric(players[column], errors="coerce").fillna(0.0)
    singles = (players["bat_hits"] - players["bat_doubles"] - players["bat_triples"] - players["bat_homeRuns"]).clip(lower=0)
    players["woba_num"] = (
        .69 * players["bat_baseOnBalls"] + .72 * players["bat_hitByPitch"] + .89 * singles
        + 1.27 * players["bat_doubles"] + 1.62 * players["bat_triples"] + 2.10 * players["bat_homeRuns"]
    )
    players["woba_den"] = (
        players["bat_atBats"] + players["bat_baseOnBalls"]
        + players["bat_hitByPitch"] + players["bat_sacFlies"]
    )
    players["started"] = players["started_batting"].eq(True).astype(float)
    players["roster_game"] = 1.0
    grouped = players.groupby(["team", "player_id"], sort=False)
    players["prior_woba_num"] = grouped["woba_num"].cumsum() - players["woba_num"]
    players["prior_woba_den"] = grouped["woba_den"].cumsum() - players["woba_den"]
    players["prior_starts"] = grouped["started"].cumsum() - players["started"]
    players["prior_roster_games"] = grouped["roster_game"].cumsum() - 1.0
    players["prior_quality"] = (
        (players["prior_woba_num"] + .320 * 50) / (players["prior_woba_den"] + 50)
        + .06 * (players["prior_starts"] + 1.5) / (players["prior_roster_games"] + 3)
    )
    players["post_quality"] = (
        (players["prior_woba_num"] + players["woba_num"] + .320 * 50)
        / (players["prior_woba_den"] + players["woba_den"] + 50)
        + .06 * (players["prior_starts"] + players["started"] + 1.5)
        / (players["prior_roster_games"] + 4)
    )
    players["slot"] = (pd.to_numeric(players["batting_order"], errors="coerce") // 100).astype("Int64")
    hitters = players[~players["position"].astype(str).eq("P")].sort_values(
        ["sortTime", "game_pk", "side", "player_id"]
    )

    team_rows = []
    latest_post_quality: dict[tuple[str, int], float] = {}
    previous_by_team: dict[str, dict] = {}
    for (game_pk, side), group in hitters.groupby(["game_pk", "side"], sort=False):
        first = group.iloc[0]
        team = str(first["team"])
        starters = group[group["started_batting"].eq(True) & group["player_id"].notna()].copy()
        if len(starters) < 8:
            continue
        starters["player_id"] = starters["player_id"].astype(int)
        current_quality = {int(row.player_id): float(row.prior_quality) for row in starters.itertuples()}
        current_players = set(current_quality)
        current_top4 = set(starters.loc[starters["slot"].le(4), "player_id"].astype(int))
        roster_quality = {
            int(row.player_id): float(row.prior_quality)
            for row in group[group["player_id"].notna()].itertuples()
        }
        previous = previous_by_team.get(team)
        signals = {name: np.nan for name in SIGNAL_COLUMNS}
        if previous:
            previous_players = previous["players"]
            previous_top4 = previous["top4"]
            previous_quality_now = {
                player: roster_quality.get(
                    player,
                    latest_post_quality.get((team, player), previous["quality"].get(player, .35)),
                )
                for player in previous_players
            }
            current_mean = float(np.mean(list(current_quality.values())))
            previous_mean = float(np.mean(list(previous_quality_now.values())))
            incoming = current_players - previous_players
            outgoing = previous_players - current_players
            incoming_mean = float(np.mean([current_quality[player] for player in incoming])) if incoming else current_mean
            outgoing_mean = float(np.mean([previous_quality_now[player] for player in outgoing])) if outgoing else previous_mean
            previous_weight = sum(max(value, .01) for value in previous_quality_now.values())
            retained_weight = sum(max(previous_quality_now[player], .01) for player in current_players & previous_players)
            current_top4_mean = float(np.mean([current_quality[player] for player in current_top4])) if current_top4 else current_mean
            previous_top4_mean = float(np.mean([previous_quality_now[player] for player in previous_top4])) if previous_top4 else previous_mean
            signals.update({
                "qualityDelta": current_mean - previous_mean,
                "replacementDelta": incoming_mean - outgoing_mean if incoming or outgoing else 0.0,
                "retentionQuality": retained_weight / previous_weight if previous_weight else np.nan,
                "top4QualityDelta": current_top4_mean - previous_top4_mean,
            })
            loss = float(previous["win"] == 0)
            signals.update({
                "lossQualityDelta": loss * signals["qualityDelta"],
                "lossReplacementDelta": loss * signals["replacementDelta"],
                "lossRetentionQuality": loss * signals["retentionQuality"],
                "lossTop4QualityDelta": loss * signals["top4QualityDelta"],
            })
        win = int(first["homeWin"] if side == "home" else 1 - first["homeWin"])
        team_rows.append({
            "gamePk": int(game_pk), "date": str(first["sortTime"].date()), "side": side,
            "team": team, "sortTime": first["sortTime"], "win": win,
            "previousWin": np.nan if previous is None else int(previous["win"]), **signals,
        })
        previous_by_team[team] = {
            "players": current_players, "top4": current_top4,
            "quality": current_quality, "win": win,
        }
        for row in group[group["player_id"].notna()].itertuples():
            latest_post_quality[(team, int(row.player_id))] = float(row.post_quality)

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

    after_loss = team_frame[team_frame["previousWin"].eq(0) & team_frame["qualityDelta"].notna()].copy()
    after_loss["bucket"] = pd.cut(
        after_loss["qualityDelta"], [-np.inf, -.003, .003, np.inf],
        labels=["Lineup de menor calidad", "Calidad similar", "Lineup de mayor calidad"],
    )
    descriptive = []
    for bucket in ["Lineup de menor calidad", "Calidad similar", "Lineup de mayor calidad"]:
        sample = after_loss[after_loss["bucket"].eq(bucket)]
        latest = sample[sample["sortTime"].dt.year.eq(2026)]
        descriptive.append({
            "label": bucket, "games": len(sample), "wins": int(sample["win"].sum()),
            "winRate": sample["win"].mean(), "winRate2026": latest["win"].mean(),
        })

    latest_player_ratings: dict[str, dict[str, float]] = defaultdict(dict)
    for (team, player_id), quality in latest_post_quality.items():
        latest_player_ratings[team][str(player_id)] = quality
    audit = {
        "method": "Calidad suavizada de bateadores, titularidad y sustituciones calculadas solo con juegos anteriores; monthly expanding walk-forward.",
        "cutoffDate": cutoff_date,
        "coverage": {
            "teamGames": len(team_frame),
            "qualityComparisons": int(team_frame["qualityDelta"].notna().sum()),
            "ratedPlayers": len(latest_post_quality),
        },
        "descriptive": descriptive,
        "latestPlayerRatings": dict(latest_player_ratings),
        "candidate": None,
        "combinedWithCoach": None,
        "recommendation": "PRUEBA: requiere ambas alineaciones confirmadas; compara la calidad de quienes entran, salen y permanecen.",
    }
    return result, audit
