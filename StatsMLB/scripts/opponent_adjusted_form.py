from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    "adjustedResultL5",
    "adjustedResultL10",
    "adjustedResultL20",
    "adjustedMarginL5",
    "adjustedMarginL10",
    "adjustedMarginL20",
    "opponentStrengthL10",
    "opponentStrengthL20",
    "qualityResultL10",
    "qualityResultL20",
    "formTrend",
    "marginTrend",
]

# Only these two pre-selected features passed the chronological gate when they
# complemented lineup fatigue + rotation quality. Keep the wider signal set for
# the audit, but do not expose the design-period variants to the production model.
TEST_FEATURES = ["edge_opponentStrengthL10", "edge_opponentStrengthL20"]


def _mean(entries: list[dict], key: str) -> float:
    return float(np.mean([entry[key] for entry in entries])) if entries else float("nan")


def attach_opponent_adjusted_form(feature_rows: pd.DataFrame, cutoff_date: str):
    """Build opponent-adjusted recent form using only information known before each game."""
    strength_history = defaultdict(lambda: {"games": 0, "wins": 0})
    performance_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    rows = []

    def strength(team: str) -> float:
        record = strength_history[team]
        return (record["wins"] + 10) / (record["games"] + 20)

    def signals(team: str) -> dict[str, float]:
        entries = list(performance_history[team])
        windows = {5: entries[-5:], 10: entries[-10:], 20: entries[-20:]}
        result = {
            f"adjustedResultL{window}": _mean(items, "adjustedResult")
            for window, items in windows.items()
        }
        result.update({
            f"adjustedMarginL{window}": _mean(items, "adjustedMargin")
            for window, items in windows.items()
        })
        result.update({
            "opponentStrengthL10": _mean(windows[10], "opponentStrength"),
            "opponentStrengthL20": _mean(windows[20], "opponentStrength"),
            "qualityResultL10": _mean(windows[10], "qualityResult"),
            "qualityResultL20": _mean(windows[20], "qualityResult"),
        })
        result["formTrend"] = result["adjustedResultL5"] - result["adjustedResultL20"]
        result["marginTrend"] = result["adjustedMarginL5"] - result["adjustedMarginL20"]
        return result

    for game in feature_rows.sort_values(["date", "gamePk"]).itertuples():
        home = str(game.home)
        away = str(game.away)
        home_strength = strength(home)
        away_strength = strength(away)
        home_signals = signals(home)
        away_signals = signals(away)
        row = {"gamePk": int(game.gamePk)}
        for name in SIGNAL_COLUMNS:
            row[f"home_{name}"] = home_signals[name]
            row[f"away_{name}"] = away_signals[name]
            row[f"edge_{name}"] = home_signals[name] - away_signals[name]
        rows.append(row)

        home_win = int(game.homeWin)
        for team, opponent, won, run_margin, own_strength, opponent_strength, location in (
            (home, away, home_win, float(game.homeScore - game.awayScore), home_strength, away_strength, 1.0),
            (away, home, 1 - home_win, float(game.awayScore - game.homeScore), away_strength, home_strength, -1.0),
        ):
            expected = 1 / (1 + math.exp(-(4.0 * (own_strength - opponent_strength) + .16 * location)))
            margin_score = math.tanh(run_margin / 4.0)
            performance_history[team].append({
                "opponent": opponent,
                "opponentStrength": opponent_strength,
                "expectedWin": expected,
                "adjustedResult": won - expected,
                "adjustedMargin": margin_score - (2 * expected - 1),
                "qualityResult": won * opponent_strength - (1 - won) * (1 - opponent_strength),
                "win": won,
                "runMargin": run_margin,
                "date": str(game.date),
            })
        strength_history[home]["games"] += 1
        strength_history[home]["wins"] += home_win
        strength_history[away]["games"] += 1
        strength_history[away]["wins"] += 1 - home_win

    signal_frame = pd.DataFrame(rows)
    result = feature_rows.merge(signal_frame, on="gamePk", how="left")
    team_rows = []
    for game in result.itertuples():
        for side, won in (("home", int(game.homeWin)), ("away", 1 - int(game.homeWin))):
            team_rows.append({
                "gamePk": int(game.gamePk), "date": str(game.date), "season": int(game.season),
                "team": str(getattr(game, side)), "win": won,
                "adjustedResultL10": getattr(game, f"{side}_adjustedResultL10"),
            })
    team_frame = pd.DataFrame(team_rows)
    team_frame["bucket"] = pd.cut(
        team_frame["adjustedResultL10"], [-np.inf, -.08, .08, np.inf],
        labels=["Rindió peor de lo esperado", "Rindió como se esperaba", "Rindió mejor de lo esperado"],
    )
    descriptive = []
    for bucket in ("Rindió peor de lo esperado", "Rindió como se esperaba", "Rindió mejor de lo esperado"):
        sample = team_frame[team_frame["bucket"].eq(bucket)]
        latest = sample[sample["season"].eq(2026)]
        descriptive.append({
            "label": bucket, "games": len(sample), "wins": int(sample["win"].sum()),
            "winRate": sample["win"].mean(), "winRate2026": latest["win"].mean(),
        })
    latest_team_signals = {team: signals(team) for team in sorted(performance_history)}
    audit = {
        "method": "Resultados y margenes recientes comparados contra una expectativa congelada por fuerza previa del rival y localia; monthly expanding walk-forward.",
        "cutoffDate": cutoff_date,
        "coverage": {"teamGames": len(team_frame), "teams": len(latest_team_signals)},
        "descriptive": descriptive,
        "latestTeamSignals": latest_team_signals,
        "candidate": None,
        "combinedWithFatigue": None,
        "recommendation": "Se activa solo si supera diseño 2024-2025 y validación 2026 H1/H2 sin retrocesos.",
    }
    return result, audit
