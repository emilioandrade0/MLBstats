from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    "wobaL5", "wobaL10", "xbhRateL5", "xbhRateL10",
    "disciplineL5", "disciplineL10", "conversionGapL5", "conversionGapL10",
    "lobRateL5", "lobRateL10", "runRateL5", "runRateL10",
    "processTrend", "conversionTrend",
]


def _mean(entries: list[dict], key: str) -> float:
    return float(np.mean([entry[key] for entry in entries])) if entries else float("nan")


def attach_sustainable_offense(root: Path, feature_rows: pd.DataFrame, cutoff_date: str):
    columns = [
        "game_pk", "side", "bat_runs", "bat_hits", "bat_doubles", "bat_triples", "bat_homeRuns",
        "bat_baseOnBalls", "bat_strikeOuts", "bat_hitByPitch", "bat_atBats", "bat_leftOnBase",
        "bat_sacFlies", "bat_totalBases", "bat_plateAppearances",
    ]
    box = pd.read_parquet(root / "data" / "processed" / "team_box.parquet", columns=columns)
    box["side"] = box["side"].astype(str).str.lower()
    numeric = [column for column in columns if column not in {"game_pk", "side"}]
    for column in numeric:
        box[column] = pd.to_numeric(box[column], errors="coerce").fillna(0.0)
    box_map = {(int(game_pk), str(side)): group.iloc[0] for (game_pk, side), group in box.groupby(["game_pk", "side"], sort=False)}

    histories: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    league_runs = 100.0
    league_woba_num = 135.0
    rows = []

    def signals(team: str) -> dict[str, float]:
        entries = list(histories[team])
        windows = {5: entries[-5:], 10: entries[-10:]}
        result = {}
        for window, items in windows.items():
            result[f"wobaL{window}"] = _mean(items, "woba")
            result[f"xbhRateL{window}"] = _mean(items, "xbhRate")
            result[f"disciplineL{window}"] = _mean(items, "discipline")
            result[f"conversionGapL{window}"] = _mean(items, "conversionGap")
            result[f"lobRateL{window}"] = _mean(items, "lobRate")
            result[f"runRateL{window}"] = _mean(items, "runRate")
        result["processTrend"] = result["wobaL5"] - result["wobaL10"]
        result["conversionTrend"] = result["conversionGapL5"] - result["conversionGapL10"]
        return result

    for game in feature_rows.sort_values(["date", "gamePk"]).itertuples():
        home = str(game.home)
        away = str(game.away)
        home_signals = signals(home)
        away_signals = signals(away)
        row = {"gamePk": int(game.gamePk)}
        for name in SIGNAL_COLUMNS:
            row[f"home_{name}"] = home_signals[name]
            row[f"away_{name}"] = away_signals[name]
            row[f"edge_{name}"] = home_signals[name] - away_signals[name]
        rows.append(row)

        conversion_scale = league_runs / league_woba_num if league_woba_num > 0 else .74
        current = []
        for side, team in (("home", home), ("away", away)):
            item = box_map.get((int(game.gamePk), side))
            if item is None:
                continue
            singles = max(0.0, item.bat_hits - item.bat_doubles - item.bat_triples - item.bat_homeRuns)
            woba_num = (
                .69 * item.bat_baseOnBalls + .72 * item.bat_hitByPitch + .89 * singles
                + 1.27 * item.bat_doubles + 1.62 * item.bat_triples + 2.10 * item.bat_homeRuns
            )
            denominator = max(1.0, item.bat_atBats + item.bat_baseOnBalls + item.bat_hitByPitch + item.bat_sacFlies)
            pa = max(1.0, item.bat_plateAppearances)
            opportunity = max(1.0, item.bat_hits + item.bat_baseOnBalls + item.bat_hitByPitch + item.bat_leftOnBase)
            entry = {
                "woba": woba_num / denominator,
                "xbhRate": (item.bat_doubles + item.bat_triples + item.bat_homeRuns) / pa,
                "discipline": (item.bat_baseOnBalls - item.bat_strikeOuts) / pa,
                "conversionGap": item.bat_runs - conversion_scale * woba_num,
                "lobRate": item.bat_leftOnBase / opportunity,
                "runRate": item.bat_runs / pa,
                "date": str(game.date),
            }
            histories[team].append(entry)
            current.append((float(item.bat_runs), float(woba_num)))
        league_runs += sum(value[0] for value in current)
        league_woba_num += sum(value[1] for value in current)

    signal_frame = pd.DataFrame(rows)
    result = feature_rows.merge(signal_frame, on="gamePk", how="left")
    team_rows = []
    for game in result.itertuples():
        for side, won in (("home", int(game.homeWin)), ("away", 1 - int(game.homeWin))):
            team_rows.append({
                "gamePk": int(game.gamePk), "date": str(game.date), "season": int(game.season),
                "team": str(getattr(game, side)), "win": won, "wobaL10": getattr(game, f"{side}_wobaL10"),
            })
    team_frame = pd.DataFrame(team_rows)
    team_frame["bucket"] = pd.cut(
        team_frame["wobaL10"], [-np.inf, .29, .34, np.inf],
        labels=["Proceso ofensivo frío", "Proceso ofensivo medio", "Proceso ofensivo fuerte"],
    )
    descriptive = []
    for bucket in ("Proceso ofensivo frío", "Proceso ofensivo medio", "Proceso ofensivo fuerte"):
        sample = team_frame[team_frame["bucket"].eq(bucket)]
        latest = sample[sample["season"].eq(2026)]
        descriptive.append({
            "label": bucket, "games": len(sample), "wins": int(sample["win"].sum()),
            "winRate": sample["win"].mean(), "winRate2026": latest["win"].mean(),
        })
    latest_team_signals = {team: signals(team) for team in sorted(histories)}
    audit = {
        "method": "Proceso ofensivo agregado en 5/10 juegos con wOBA aproximada, extrabases, disciplina, LOB y brecha carreras-creación; monthly expanding walk-forward.",
        "cutoffDate": cutoff_date,
        "coverage": {"teamGames": len(team_frame), "teams": len(latest_team_signals)},
        "descriptive": descriptive,
        "latestTeamSignals": latest_team_signals,
        "candidate": None,
        "recommendation": "PRUEBA: agrega proceso ofensivo reciente; no utiliza el resultado del partido que intenta predecir.",
    }
    return result, audit
