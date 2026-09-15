from __future__ import annotations

import json
import math
import os
import re
import urllib.request
import base64
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rotation_quality import TEST_FEATURES as ROTATION_QUALITY_TEST_FEATURES
from rotation_quality import attach_rotation_quality_features
from lineup_fatigue import TEST_FEATURES as LINEUP_FATIGUE_TEST_FEATURES
from lineup_fatigue import attach_lineup_fatigue_features
from opponent_adjusted_form import TEST_FEATURES as OPPONENT_FORM_TEST_FEATURES
from opponent_adjusted_form import attach_opponent_adjusted_form
from season_l10_audit import write_season_l10_audit


ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "sweep_report_20260824"
PUBLIC_DATA = APP / "public" / "data"
TODAY = date.today()

STATUS_LABELS = {
    "sweep": "Barrió 3-0",
    "swept": "Fue barrido 0-3",
    "lost_1_2": "Perdió la serie 1-2",
    "won_2_1": "Ganó la serie 2-1",
    "none": "Sin serie exacta de 3 previa",
}

FEATURES = [
    "strength_diff",
    "recent_diff",
    "run_diff_diff",
    "home_sweep",
    "away_sweep",
    "home_swept",
    "away_swept",
    "home_lost_1_2",
    "away_lost_1_2",
    "both_sweep",
    "both_swept",
    "home_swept_vs_away_lost_1_2",
    "away_swept_vs_home_lost_1_2",
    "market_logit_p_home",
    "market_n_providers",
    "market_p_home_std",
    "days_since_last_game_diff",
]

BEST_PLAYERS_TEST_FEATURES = [
    f"edge_prev_{prefix}_{result}"
    for prefix in ("top5_ge4", "core8_ge7", "star_pa_ge50", "star_weight_ge80")
    for result in ("win", "loss")
]

COACH_ROTATION_SIGNAL_COLUMNS = [
    "prevLoss", "changes", "orderMove", "top4Changed",
    "lossRotation", "lossOrderMove", "lossTop4Changed",
]
COACH_ROTATION_TEST_FEATURES = [
    f"{side}_{column}"
    for column in COACH_ROTATION_SIGNAL_COLUMNS
    for side in ("home", "away")
]

FACTOR_KEYS = ["localia", "strength", "recent", "runDiff", "series", "marketSchedule", "bestPlayersTest", "coachRotationTest", "rotationQualityTest", "lineupFatigueTest", "opponentFormTest"]
FACTOR_GROUPS = {
    "strength": ["strength_diff"],
    "recent": ["recent_diff"],
    "runDiff": ["run_diff_diff"],
    "series": FEATURES[3:13],
    "marketSchedule": ["market_logit_p_home", "market_n_providers", "market_p_home_std", "days_since_last_game_diff"],
    "bestPlayersTest": BEST_PLAYERS_TEST_FEATURES,
    "coachRotationTest": COACH_ROTATION_TEST_FEATURES,
    "rotationQualityTest": ROTATION_QUALITY_TEST_FEATURES,
    "lineupFatigueTest": LINEUP_FATIGUE_TEST_FEATURES,
    "opponentFormTest": OPPONENT_FORM_TEST_FEATURES,
}
DEFAULT_FACTOR_MASK = 1 + 4 + 8 + 32  # Localia + Forma L10 + Carreras + Mercado/descanso.
MODEL_WORKERS = min(4, os.cpu_count() or 1)


def safe_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else round(float(value), 8)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def compact_walkforward_probabilities(walkforward: dict):
    """Pack probabilities as uint8 base64 while preserving every >= .5 decision."""
    for day in walkforward["days"].values():
        for game in day["games"]:
            packed = bytes(
                128 if value is None or not math.isfinite(value)
                else min(255, max(0, int(value * 256)))
                for value in game.pop("maskProbabilities")
            )
            game["maskProbabilitiesEncoded"] = base64.b64encode(packed).decode("ascii")
    walkforward["probabilityEncoding"] = "uint8-base64-midpoint"
    return walkforward


def write_walkforward_artifacts(walkforward: dict):
    """Write a small manifest plus bounded yearly day files for reliable loading and hosting."""
    days = walkforward.pop("days")
    day_files = []
    for year in sorted({str(day)[:4] for day in days}):
        filename = f"walkforward-{year}.json"
        yearly_days = {day: value for day, value in days.items() if str(day).startswith(year)}
        (PUBLIC_DATA / filename).write_text(
            json.dumps(clean({"days": yearly_days}), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        day_files.append(filename)
    walkforward["dayFiles"] = day_files
    walkforward["days"] = {}
    (PUBLIC_DATA / "walkforward.json").write_text(
        json.dumps(clean(walkforward), ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def load_previous_walkforward_predictions():
    manifest_path = PUBLIC_DATA / "walkforward.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        manifest.get("factorKeys") != FACTOR_KEYS
        or manifest.get("probabilityEncoding") != "uint8-base64-midpoint"
    ):
        return {}
    expected_length = 2 ** len(FACTOR_KEYS)
    previous = {}
    for filename in manifest.get("dayFiles", []):
        path = PUBLIC_DATA / str(filename)
        if not path.exists():
            continue
        try:
            days = json.loads(path.read_text(encoding="utf-8")).get("days", {})
        except (OSError, json.JSONDecodeError):
            continue
        for day in days.values():
            for game in day.get("games", []):
                try:
                    packed = base64.b64decode(game["maskProbabilitiesEncoded"], validate=True)
                    game_pk = int(game["gamePk"])
                    home_probability = float(game["homeProbability"])
                except (KeyError, TypeError, ValueError):
                    continue
                if len(packed) != expected_length or not math.isfinite(home_probability):
                    continue
                previous[game_pk] = {
                    "packed": packed,
                    "homeProbability": home_probability,
                }
    return previous


def context_code(row: pd.Series) -> str:
    wins = int(row["series_wins"])
    losses = int(row["series_losses"])
    if wins == 3:
        return "sweep"
    if losses == 3:
        return "swept"
    if wins == 1 and losses == 2:
        return "lost_1_2"
    if wins == 2 and losses == 1:
        return "won_2_1"
    return "none"


def build_context_map(series: pd.DataFrame):
    context = {}
    eligible = series[
        series["next_game_available"].eq(1)
        & series["next_game_date"].notna()
        & series["next_opponent"].notna()
    ].copy()
    for _, row in eligible.iterrows():
        key = (str(row["next_game_date"])[:10], str(row["team"]), str(row["next_opponent"]))
        context[key] = {
            "code": context_code(row),
            "label": STATUS_LABELS[context_code(row)],
            "seriesStart": str(row["start_date"])[:10],
            "seriesEnd": str(row["end_date"])[:10],
            "opponent": row["opponent"],
            "wins": int(row["series_wins"]),
            "losses": int(row["series_losses"]),
            "runDiff": int(row["series_run_diff"]),
            "nextLocation": row["next_location"],
        }
    return context, eligible


def pregame_features(games: pd.DataFrame, context_map):
    history = defaultdict(lambda: {"games": 0, "wins": 0, "runsFor": 0, "runsAgainst": 0, "recent": deque(maxlen=10)})
    rows = []

    def snapshot(team):
        item = history[team]
        games_played = item["games"]
        recent = item["recent"]
        return {
            "games": games_played,
            "wins": item["wins"],
            "strength": (item["wins"] + 10) / (games_played + 20),
            "recent": (sum(recent) + 2.5) / (len(recent) + 5),
            "runDiffPerGame": (item["runsFor"] - item["runsAgainst"]) / (games_played + 10),
            "runsFor": item["runsFor"],
            "runsAgainst": item["runsAgainst"],
            "recentGames": len(recent),
        }

    for _, game in games.sort_values(["season", "sort_time", "game_pk"]).iterrows():
        home = str(game["home_team_abbrev"])
        away = str(game["away_team_abbrev"])
        game_date = str(game["game_date"])[:10]
        hs, aw = snapshot(home), snapshot(away)
        home_ctx = context_map.get((game_date, home, away), {"code": "none", "label": STATUS_LABELS["none"]})
        away_ctx = context_map.get((game_date, away, home), {"code": "none", "label": STATUS_LABELS["none"]})
        hc, ac = home_ctx["code"], away_ctx["code"]
        features = {
            "strength_diff": hs["strength"] - aw["strength"],
            "recent_diff": hs["recent"] - aw["recent"],
            "run_diff_diff": float(np.clip(hs["runDiffPerGame"] - aw["runDiffPerGame"], -3, 3)),
            "home_sweep": float(hc == "sweep"),
            "away_sweep": float(ac == "sweep"),
            "home_swept": float(hc == "swept"),
            "away_swept": float(ac == "swept"),
            "home_lost_1_2": float(hc == "lost_1_2"),
            "away_lost_1_2": float(ac == "lost_1_2"),
            "both_sweep": float(hc == "sweep" and ac == "sweep"),
            "both_swept": float(hc == "swept" and ac == "swept"),
            "home_swept_vs_away_lost_1_2": float(hc == "swept" and ac == "lost_1_2"),
            "away_swept_vs_home_lost_1_2": float(ac == "swept" and hc == "lost_1_2"),
        }
        rows.append({
            "gamePk": int(game["game_pk"]),
            "date": game_date,
            "season": int(game["season"]),
            "home": home,
            "away": away,
            "homeWin": int(bool(game["home_win"])),
            "homeScore": int(game["home_score"]),
            "awayScore": int(game["away_score"]),
            "homeContext": home_ctx,
            "awayContext": away_ctx,
            "homeSnapshot": hs,
            "awaySnapshot": aw,
            **features,
        })
        home_win = int(bool(game["home_win"]))
        home_score, away_score = int(game["home_score"]), int(game["away_score"])
        for team, won, runs_for, runs_against in (
            (home, home_win, home_score, away_score),
            (away, 1 - home_win, away_score, home_score),
        ):
            item = history[team]
            item["games"] += 1
            item["wins"] += won
            item["runsFor"] += runs_for
            item["runsAgainst"] += runs_against
            item["recent"].append(won)
    return pd.DataFrame(rows), history


def features_for_mask(mask: int):
    selected = []
    for index, key in enumerate(FACTOR_KEYS):
        if key == "localia" or not mask & (1 << index):
            continue
        selected.extend(FACTOR_GROUPS[key])
    return selected


def fit_mask_model(train: pd.DataFrame, mask: int):
    features = features_for_mask(mask)
    use_localia = bool(mask & 1)
    if not features:
        probability = float(train["homeWin"].mean()) if use_localia else 0.5
        return {"kind": "constant", "probability": probability, "inputFeatures": []}
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.55, max_iter=2500, random_state=42, fit_intercept=use_localia),
    )
    pipeline.fit(train[features], train["homeWin"])
    return pipeline


def predict_mask(model, frame: pd.DataFrame, mask: int):
    if isinstance(model, dict):
        return np.full(len(frame), model["probability"], dtype=float)
    return model.predict_proba(frame[features_for_mask(mask)])[:, 1]


def serialize_mask_model(model, mask: int):
    if isinstance(model, dict):
        return model
    features = features_for_mask(mask)
    imputer = model.named_steps["simpleimputer"]
    scaler = model.named_steps["standardscaler"]
    logistic_model = model.named_steps["logisticregression"]
    return {
        "kind": "logistic",
        "inputFeatures": features,
        "imputeValues": list(imputer.statistics_),
        "indicatorFeatures": list(imputer.indicator_.features_),
        "means": list(scaler.mean_),
        "scales": list(scaler.scale_),
        "coefficients": list(logistic_model.coef_[0]),
        "intercept": float(logistic_model.intercept_[0]) if len(logistic_model.intercept_) else 0.0,
    }


def attach_market_schedule_features(feature_rows: pd.DataFrame):
    source = ROOT / "data" / "processed" / "train.parquet"
    columns = ["game_pk", *FACTOR_GROUPS["marketSchedule"]]
    market = pd.read_parquet(source, columns=columns).drop_duplicates("game_pk")
    result = feature_rows.merge(market, left_on="gamePk", right_on="game_pk", how="left")
    return result.drop(columns=["game_pk"])


def attach_coach_rotation_features(feature_rows: pd.DataFrame, cutoff_date: str):
    game_times = pd.read_parquet(
        ROOT / "data" / "processed" / "games.parquet",
        columns=["game_pk", "first_pitch_utc"],
    ).drop_duplicates("game_pk")
    meta = feature_rows[["gamePk", "date", "season", "home", "away", "homeWin"]].merge(
        game_times, left_on="gamePk", right_on="game_pk", how="left"
    )
    meta["sortTime"] = pd.to_datetime(meta["first_pitch_utc"], utc=True, errors="coerce")
    fallback = pd.to_datetime(meta["date"], utc=True) + pd.to_timedelta(meta.groupby("date").cumcount(), unit="s")
    meta["sortTime"] = meta["sortTime"].fillna(fallback)

    players = pd.read_parquet(
        ROOT / "data" / "processed" / "player_box.parquet",
        columns=["game_pk", "side", "player_id", "position", "started_batting", "batting_order"],
    )
    players["side"] = players["side"].astype(str).str.lower()
    players = players[players["started_batting"].eq(True) & ~players["position"].astype(str).eq("P")].copy()
    players["slot"] = (pd.to_numeric(players["batting_order"], errors="coerce") // 100).astype("Int64")
    lineups = {}
    for key, group in players.groupby(["game_pk", "side"], sort=False):
        lineups[key] = {
            "players": set(group["player_id"].dropna().astype(int)),
            "slots": {
                int(row["player_id"]): int(row["slot"])
                for _, row in group.dropna(subset=["player_id", "slot"]).iterrows()
            },
            "top4": set(group.loc[group["slot"].le(4), "player_id"].dropna().astype(int)),
        }

    team_rows = []
    for _, game in meta.sort_values(["sortTime", "gamePk"]).iterrows():
        for side, won in (("home", int(game["homeWin"])), ("away", 1 - int(game["homeWin"]))):
            lineup = lineups.get((int(game["gamePk"]), side))
            if not lineup:
                continue
            team_rows.append({
                "gamePk": int(game["gamePk"]),
                "date": str(game["date"]),
                "season": int(game["season"]),
                "sortTime": game["sortTime"],
                "team": str(game[side]),
                "side": side,
                "win": won,
                **lineup,
            })
    team_frame = pd.DataFrame(team_rows).sort_values(["team", "sortTime", "gamePk"]).reset_index(drop=True)

    metrics = []
    for _, group in team_frame.groupby("team", sort=False):
        previous = None
        for index, row in group.iterrows():
            item = {
                "index": index, "prevLoss": np.nan, "changes": np.nan, "orderMove": np.nan,
                "top4Changed": np.nan, "gapHours": np.nan,
            }
            if previous is not None:
                overlap = row["players"] & previous["players"]
                order_moves = [
                    abs(row["slots"][player] - previous["slots"][player])
                    for player in overlap if player in row["slots"] and player in previous["slots"]
                ]
                item.update({
                    "prevLoss": float(previous["win"] == 0),
                    "changes": float(len(row["players"] - previous["players"])),
                    "orderMove": float(np.mean(order_moves)) if order_moves else np.nan,
                    "top4Changed": float(len(row["top4"] - previous["top4"])),
                    "gapHours": (row["sortTime"] - previous["sortTime"]).total_seconds() / 3600,
                })
            metrics.append(item)
            previous = row
    metric_frame = pd.DataFrame(metrics).set_index("index")
    team_frame = team_frame.join(metric_frame)
    team_frame["lossRotation"] = team_frame["prevLoss"] * team_frame["changes"]
    team_frame["lossOrderMove"] = team_frame["prevLoss"] * team_frame["orderMove"]
    team_frame["lossTop4Changed"] = team_frame["prevLoss"] * team_frame["top4Changed"]

    home_signals = team_frame[team_frame["side"].eq("home")][["gamePk", *COACH_ROTATION_SIGNAL_COLUMNS]].rename(
        columns={column: f"home_{column}" for column in COACH_ROTATION_SIGNAL_COLUMNS}
    )
    away_signals = team_frame[team_frame["side"].eq("away")][["gamePk", *COACH_ROTATION_SIGNAL_COLUMNS]].rename(
        columns={column: f"away_{column}" for column in COACH_ROTATION_SIGNAL_COLUMNS}
    )
    result = feature_rows.merge(home_signals.merge(away_signals, on="gamePk", how="outer"), on="gamePk", how="left")

    team_frame["bucket"] = pd.cut(
        team_frame["changes"], [-.1, 1, 2, 99], labels=["0–1 cambios", "2 cambios", "3+ cambios"]
    )
    descriptive = []
    for previous_loss, result_label in ((1, "Después de perder"), (0, "Después de ganar")):
        for bucket in ("0–1 cambios", "2 cambios", "3+ cambios"):
            sample = team_frame[team_frame["prevLoss"].eq(previous_loss) & team_frame["bucket"].eq(bucket)]
            latest = sample[sample["season"].eq(2026)]
            descriptive.append({
                "label": f"{result_label}: {bucket}", "previousResult": "loss" if previous_loss else "win",
                "rotation": bucket, "games": len(sample), "wins": int(sample["win"].sum()),
                "winRate": sample["win"].mean(), "winRate2026": latest["win"].mean(),
            })

    latest_team_states = {}
    for _, row in team_frame.groupby("team", sort=False).tail(1).iterrows():
        latest_team_states[str(row["team"])] = {
            "gamePk": int(row["gamePk"]),
            "playedAt": row["sortTime"].isoformat(),
            "win": int(row["win"]),
            "players": sorted(int(player) for player in row["players"]),
            "slots": {str(player): int(slot) for player, slot in row["slots"].items()},
            "top4": sorted(int(player) for player in row["top4"]),
        }
    audit = {
        "method": "Alineaciones titulares de posición comparadas antes de cada juego; monthly expanding walk-forward.",
        "cutoffDate": cutoff_date,
        "coverage": {
            "teamGames": len(team_frame),
            "rotationComparisons": int(team_frame["changes"].notna().sum()),
            "averageLineupSize": np.mean([len(value["players"]) for value in lineups.values()]),
        },
        "descriptive": descriptive,
        "latestTeamStates": latest_team_states,
        "candidate": None,
        "recommendation": "PRUEBA: requiere ambas alineaciones confirmadas; si faltan, el juego conserva el modelo base.",
    }
    return result, audit


PAIR_MIN_PRIOR_GAMES = 10


def build_indication_pair_signal(
    feature_rows: pd.DataFrame,
    away_column: str,
    home_column: str,
    labeler,
    title: str,
):
    columns = ["gamePk", "date", "season", "homeWin", away_column, home_column]
    eligible = feature_rows[columns].dropna(subset=[away_column, home_column, "homeWin"]).copy()
    eligible["dateKey"] = eligible["date"].astype(str).str[:10]
    eligible["awayLabel"] = eligible[away_column].map(labeler)
    eligible["homeLabel"] = eligible[home_column].map(labeler)
    eligible["pairKey"] = eligible["awayLabel"] + "|" + eligible["homeLabel"]
    eligible = eligible.sort_values(["dateKey", "gamePk"]).reset_index(drop=True)

    history = defaultdict(lambda: {"games": 0, "homeWins": 0})
    evaluated = defaultdict(lambda: {"games": 0, "correct": 0, "homeCorrect": 0})
    for _, same_day in eligible.groupby("dateKey", sort=True):
        # Every game on the same date uses the snapshot available before that date.
        for row in same_day.itertuples():
            prior = history[row.pairKey]
            if prior["games"] < PAIR_MIN_PRIOR_GAMES:
                continue
            predicted_home = prior["homeWins"] * 2 >= prior["games"]
            actual_home = bool(row.homeWin)
            score = evaluated[row.pairKey]
            score["games"] += 1
            score["correct"] += int(predicted_home == actual_home)
            score["homeCorrect"] += int(actual_home)
        for row in same_day.itertuples():
            prior = history[row.pairKey]
            prior["games"] += 1
            prior["homeWins"] += int(bool(row.homeWin))

    rows = []
    for pair_key, sample in eligible.groupby("pairKey", sort=False):
        away_label = str(sample.iloc[0]["awayLabel"])
        home_label = str(sample.iloc[0]["homeLabel"])
        games = len(sample)
        home_wins = int(sample["homeWin"].sum())
        away_wins = games - home_wins
        latest = sample[sample["season"].eq(2026)]
        latest_games = len(latest)
        latest_home_wins = int(latest["homeWin"].sum())
        wf = evaluated[pair_key]
        wf_accuracy = wf["correct"] / wf["games"] if wf["games"] else None
        wf_home_accuracy = wf["homeCorrect"] / wf["games"] if wf["games"] else None
        leader = "home" if home_wins > away_wins else "away" if away_wins > home_wins else "even"
        rows.append({
            "key": pair_key,
            "awayLabel": away_label,
            "homeLabel": home_label,
            "games": games,
            "awayWins": away_wins,
            "homeWins": home_wins,
            "leader": leader,
            "leaderWinRate": max(home_wins, away_wins) / games if games else None,
            "games2026": latest_games,
            "awayWins2026": latest_games - latest_home_wins,
            "homeWins2026": latest_home_wins,
            "walkforwardGames": wf["games"],
            "walkforwardCorrect": wf["correct"],
            "walkforwardAccuracy": wf_accuracy,
            "walkforwardHomeAccuracy": wf_home_accuracy,
            "deltaVsHomePoints": (wf_accuracy - wf_home_accuracy) * 100 if wf_accuracy is not None else None,
        })
    rows.sort(key=lambda item: (-item["games"], item["awayLabel"], item["homeLabel"]))
    total_evaluated = sum(item["walkforwardGames"] for item in rows)
    total_correct = sum(item["walkforwardCorrect"] for item in rows)
    total_home_correct = sum(item["homeCorrect"] for item in evaluated.values())
    accuracy = total_correct / total_evaluated if total_evaluated else None
    home_accuracy = total_home_correct / total_evaluated if total_evaluated else None
    return {
        "title": title,
        "eligibleGames": len(eligible),
        "pairCount": len(rows),
        "summary": {
            "games": total_evaluated,
            "correct": total_correct,
            "accuracy": accuracy,
            "homeBaselineAccuracy": home_accuracy,
            "deltaVsHomePoints": (accuracy - home_accuracy) * 100 if accuracy is not None else None,
        },
        "rows": rows,
    }


def build_indication_pair_audit(feature_rows: pd.DataFrame, cutoff_date: str):
    def change_label(value):
        changes = int(round(float(value)))
        return f"{changes} cambio" if changes == 1 else f"{changes} cambios" if changes < 4 else "4+ cambios"

    def quality_label(value):
        value = float(value)
        return "Mejoró" if value > .003 else "Bajó" if value < -.003 else "Similar"

    def fatigue_label(value):
        value = float(value)
        return "Carga baja" if value <= 1.2 else "Carga media" if value <= 2.0 else "Carga alta"

    return {
        "method": (
            "Cruces visitante-local calculados con información previa al juego. La accuracy es expanding walk-forward por fecha: "
            f"cada cruce empieza a pronosticar después de {PAIR_MIN_PRIOR_GAMES} antecedentes y nunca aprende de juegos del mismo día."
        ),
        "cutoffDate": cutoff_date,
        "minimumPriorGames": PAIR_MIN_PRIOR_GAMES,
        "signals": {
            "coachRotation": build_indication_pair_signal(
                feature_rows, "away_changes", "home_changes", change_label, "Cambios de titulares",
            ),
            "rotationQuality": build_indication_pair_signal(
                feature_rows, "away_qualityDelta", "home_qualityDelta", quality_label, "Calidad de la rotación",
            ),
            "lineupFatigue": build_indication_pair_signal(
                feature_rows, "away_weightedFatigue", "home_weightedFatigue", fatigue_label, "Fatiga del lineup",
            ),
        },
    }


def finalize_coach_rotation_audit(audit: dict, walkforward: dict):
    coach_mask = DEFAULT_FACTOR_MASK | (1 << FACTOR_KEYS.index("coachRotationTest"))
    combinations = {item["mask"]: item for item in walkforward["combinations"]}
    baseline = combinations[DEFAULT_FACTOR_MASK]
    candidate = combinations[coach_mask]
    paired = []
    candidate_brier = []
    fold_deltas = defaultdict(lambda: {"base": 0, "coach": 0, "games": 0})
    for day in walkforward["days"].values():
        for game in day["games"]:
            actual_home = int(game["homeScore"] > game["awayScore"])
            base_probability = game["maskProbabilities"][DEFAULT_FACTOR_MASK]
            coach_probability = game["maskProbabilities"][coach_mask]
            base_correct = int((base_probability >= .5) == actual_home)
            coach_correct = int((coach_probability >= .5) == actual_home)
            paired.append(coach_correct - base_correct)
            candidate_brier.append((coach_probability - actual_home) ** 2)
            fold = fold_deltas[game["foldMonth"]]
            fold["base"] += base_correct
            fold["coach"] += coach_correct
            fold["games"] += 1
    delta = np.asarray(paired, dtype=float)
    rng = np.random.default_rng(20260824)
    bootstrap = np.array([
        delta[rng.integers(0, len(delta), len(delta))].mean() * 100
        for _ in range(10000)
    ])
    folds = [
        {"month": month, **values, "deltaCorrect": values["coach"] - values["base"]}
        for month, values in sorted(fold_deltas.items())
    ]
    audit["candidate"] = {
        "mask": coach_mask,
        "games": candidate["games"],
        "correct": candidate["correct"],
        "accuracy": candidate["accuracy"],
        "deltaPoints": (candidate["accuracy"] - baseline["accuracy"]) * 100,
        "brier": float(np.mean(candidate_brier)),
        "baseline": {
            "mask": DEFAULT_FACTOR_MASK, "correct": baseline["correct"], "accuracy": baseline["accuracy"],
            "brier": walkforward["summary"]["brier"],
        },
        "seasons": candidate["seasons"],
        "delta2026Points": (
            candidate["seasons"]["2026"]["accuracy"] - baseline["seasons"]["2026"]["accuracy"]
        ) * 100,
        "ci95Points": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
        "foldsBetter": sum(item["deltaCorrect"] > 0 for item in folds),
        "foldsEqual": sum(item["deltaCorrect"] == 0 for item in folds),
        "foldsWorse": sum(item["deltaCorrect"] < 0 for item in folds),
        "status": "PRUEBA",
    }
    return audit


def rotation_quality_candidate_metrics(walkforward: dict, mask: int, baseline_mask: int = DEFAULT_FACTOR_MASK):
    combinations = {item["mask"]: item for item in walkforward["combinations"]}
    baseline = combinations[baseline_mask]
    candidate = combinations[mask]
    paired = []
    candidate_brier = []
    baseline_brier = []
    fold_deltas = defaultdict(lambda: {"base": 0, "candidate": 0, "games": 0})
    for day in walkforward["days"].values():
        for game in day["games"]:
            actual_home = int(game["homeScore"] > game["awayScore"])
            base_probability = game["maskProbabilities"][baseline_mask]
            candidate_probability = game["maskProbabilities"][mask]
            base_correct = int((base_probability >= .5) == actual_home)
            candidate_correct = int((candidate_probability >= .5) == actual_home)
            paired.append(candidate_correct - base_correct)
            candidate_brier.append((candidate_probability - actual_home) ** 2)
            baseline_brier.append((base_probability - actual_home) ** 2)
            fold = fold_deltas[game["foldMonth"]]
            fold["base"] += base_correct
            fold["candidate"] += candidate_correct
            fold["games"] += 1
    delta = np.asarray(paired, dtype=float)
    rng = np.random.default_rng(20260824)
    bootstrap = np.array([
        delta[rng.integers(0, len(delta), len(delta))].mean() * 100
        for _ in range(10000)
    ])
    folds = [
        {"month": month, **values, "deltaCorrect": values["candidate"] - values["base"]}
        for month, values in sorted(fold_deltas.items())
    ]
    return {
        "mask": mask,
        "games": candidate["games"],
        "correct": candidate["correct"],
        "accuracy": candidate["accuracy"],
        "deltaPoints": (candidate["accuracy"] - baseline["accuracy"]) * 100,
        "brier": float(np.mean(candidate_brier)),
        "baseline": {
            "mask": baseline_mask, "correct": baseline["correct"],
            "accuracy": baseline["accuracy"], "brier": float(np.mean(baseline_brier)),
        },
        "seasons": candidate["seasons"],
        "delta2026Points": (
            candidate["seasons"]["2026"]["accuracy"] - baseline["seasons"]["2026"]["accuracy"]
        ) * 100,
        "ci95Points": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
        "foldsBetter": sum(item["deltaCorrect"] > 0 for item in folds),
        "foldsEqual": sum(item["deltaCorrect"] == 0 for item in folds),
        "foldsWorse": sum(item["deltaCorrect"] < 0 for item in folds),
        "status": "PRUEBA",
    }


def finalize_rotation_quality_audit(audit: dict, walkforward: dict):
    quality_bit = 1 << FACTOR_KEYS.index("rotationQualityTest")
    coach_bit = 1 << FACTOR_KEYS.index("coachRotationTest")
    audit["candidate"] = rotation_quality_candidate_metrics(
        walkforward, DEFAULT_FACTOR_MASK | quality_bit
    )
    audit["combinedWithCoach"] = rotation_quality_candidate_metrics(
        walkforward, DEFAULT_FACTOR_MASK | coach_bit | quality_bit
    )
    return audit


def finalize_lineup_fatigue_audit(audit: dict, walkforward: dict):
    fatigue_bit = 1 << FACTOR_KEYS.index("lineupFatigueTest")
    quality_bit = 1 << FACTOR_KEYS.index("rotationQualityTest")
    candidate = rotation_quality_candidate_metrics(walkforward, DEFAULT_FACTOR_MASK | fatigue_bit)
    combined = rotation_quality_candidate_metrics(
        walkforward, DEFAULT_FACTOR_MASK | quality_bit | fatigue_bit
    )
    paired_by_period = defaultdict(lambda: {"base": [], "candidate": []})
    for day in walkforward["days"].values():
        for game in day["games"]:
            actual_home = int(game["homeScore"] > game["awayScore"])
            date_value = str(game["date"])
            if date_value < "2026-01-01":
                period = "design_2024_2025"
            elif date_value <= "2026-06-30":
                period = "h1_2026"
            else:
                period = "h2_2026"
            base_probability = game["maskProbabilities"][DEFAULT_FACTOR_MASK]
            candidate_probability = game["maskProbabilities"][DEFAULT_FACTOR_MASK | fatigue_bit]
            paired_by_period[period]["base"].append(int((base_probability >= .5) == actual_home))
            paired_by_period[period]["candidate"].append(int((candidate_probability >= .5) == actual_home))
    for period, values in paired_by_period.items():
        base = np.asarray(values["base"], dtype=float)
        result = np.asarray(values["candidate"], dtype=float)
        candidate[period] = {
            "games": len(result), "accuracy": result.mean(),
            "deltaPoints": (result.mean() - base.mean()) * 100,
        }
    candidate["selectionRule"] = (
        "Seleccionado con 2024-2025; despues exigio mejora en 2026 y no retroceder en H1/H2."
    )
    candidate["passesGate"] = bool(
        candidate["design_2024_2025"]["deltaPoints"] > 0
        and candidate["delta2026Points"] > 0
        and candidate["h1_2026"]["deltaPoints"] >= 0
        and candidate["h2_2026"]["deltaPoints"] >= 0
    )
    audit["candidate"] = candidate
    audit["combinedWithRotationQuality"] = combined
    return audit


def finalize_opponent_form_audit(audit: dict, walkforward: dict):
    fatigue_bit = 1 << FACTOR_KEYS.index("lineupFatigueTest")
    quality_bit = 1 << FACTOR_KEYS.index("rotationQualityTest")
    opponent_bit = 1 << FACTOR_KEYS.index("opponentFormTest")
    reference_mask = DEFAULT_FACTOR_MASK | fatigue_bit | quality_bit
    candidate_mask = reference_mask | opponent_bit
    candidate = rotation_quality_candidate_metrics(
        walkforward, candidate_mask, baseline_mask=reference_mask
    )
    paired_by_period = defaultdict(lambda: {"base": [], "candidate": []})
    for day in walkforward["days"].values():
        for game in day["games"]:
            actual_home = int(game["homeScore"] > game["awayScore"])
            date_value = str(game["date"])
            if date_value < "2026-01-01":
                period = "design_2024_2025"
            elif date_value <= "2026-06-30":
                period = "h1_2026"
            else:
                period = "h2_2026"
            base_probability = game["maskProbabilities"][reference_mask]
            candidate_probability = game["maskProbabilities"][candidate_mask]
            paired_by_period[period]["base"].append(int((base_probability >= .5) == actual_home))
            paired_by_period[period]["candidate"].append(int((candidate_probability >= .5) == actual_home))
    for period, values in paired_by_period.items():
        base = np.asarray(values["base"], dtype=float)
        result = np.asarray(values["candidate"], dtype=float)
        candidate[period] = {
            "games": len(result), "accuracy": result.mean(),
            "deltaPoints": (result.mean() - base.mean()) * 100,
        }
    candidate["selectionRule"] = (
        "Variables elegidas solo con 2024-2025; despues exigieron mejora sobre Fatiga + Calidad "
        "en 2026 y no retroceder en H1/H2."
    )
    candidate["passesGate"] = bool(
        candidate["design_2024_2025"]["deltaPoints"] > 0
        and candidate["delta2026Points"] > 0
        and candidate["h1_2026"]["deltaPoints"] >= 0
        and candidate["h2_2026"]["deltaPoints"] >= 0
    )
    audit["candidate"] = candidate
    audit["recommendation"] = (
        "PRUEBA: aportó valor como complemento de Fatiga lineup + Calidad rotación. "
        "Permanece apagado por defecto y no se recomienda usarlo aislado."
    )
    return audit


def fit_model(feature_rows: pd.DataFrame):
    train = feature_rows[feature_rows["season"].le(2025)]
    test = feature_rows[feature_rows["season"].eq(2026)]
    total_masks = 2 ** len(FACTOR_KEYS) - 1
    def fit_one(mask: int):
        return mask, fit_mask_model(train, mask)

    models = {}
    with ThreadPoolExecutor(max_workers=MODEL_WORKERS) as executor:
        for completed, (mask, model) in enumerate(
            executor.map(fit_one, range(1, total_masks + 1)), start=1
        ):
            models[mask] = model
            if completed == 1 or completed % 256 == 0 or completed == total_masks:
                print(
                    f"  Modelo actual: {completed}/{total_masks} combinaciones "
                    f"({MODEL_WORKERS} núcleos)",
                    flush=True,
                )
    probabilities = predict_mask(models[DEFAULT_FACTOR_MASK], test, DEFAULT_FACTOR_MASK)
    predictions = probabilities >= 0.5
    baseline_probability = np.full(len(test), train["homeWin"].mean())
    metrics = {
        "trainRange": "2023-2025",
        "holdoutRange": f"2026-01-01 a {test['date'].max()}",
        "trainGames": len(train),
        "holdoutGames": len(test),
        "accuracy": accuracy_score(test["homeWin"], predictions),
        "brier": brier_score_loss(test["homeWin"], probabilities),
        "logLoss": log_loss(test["homeWin"], probabilities),
        "homeBaselineAccuracy": max(test["homeWin"].mean(), 1 - test["homeWin"].mean()),
        "homeWinRateTrain": train["homeWin"].mean(),
        "baselineBrier": brier_score_loss(test["homeWin"], baseline_probability),
    }
    return {
        "defaultMask": DEFAULT_FACTOR_MASK,
        "factorKeys": FACTOR_KEYS,
        "factorGroups": FACTOR_GROUPS,
        "modelsByMask": {str(mask): serialize_mask_model(model, mask) for mask, model in models.items()},
        "metrics": metrics,
    }


def build_walkforward(feature_rows: pd.DataFrame, coverage_audit: dict, full_rebuild: bool = False):
    frame = feature_rows.copy()
    frame["date_dt"] = pd.to_datetime(frame["date"])
    first_test_month = pd.Timestamp("2024-03-01")
    test_months = sorted(frame.loc[frame["date_dt"].ge(first_test_month), "date_dt"].dt.to_period("M").unique())
    predictions = []
    folds = []
    temporal_violations = 0
    previous_predictions = {} if full_rebuild else load_previous_walkforward_predictions()
    preserved_historical_predictions = 0
    cached_folds_skipped = 0
    print(
        f"  Walk-forward cache: {len(previous_predictions):,} juegos con predicciones previas "
        f"(full_rebuild={full_rebuild})",
        flush=True,
    )
    combination_totals = {
        mask: {
            "games": 0,
            "correct": 0,
            "seasons": defaultdict(lambda: {"games": 0, "correct": 0}),
            "days": defaultdict(lambda: {"games": 0, "correct": 0}),
        }
        for mask in range(1, 2 ** len(FACTOR_KEYS))
    }
    total_periods = len(test_months)
    today_ts = pd.Timestamp.today().normalize()
    for period_index, period in enumerate(test_months, start=1):
        month_start = period.start_time
        month_end = period.end_time.normalize()
        train = frame[frame["date_dt"].lt(month_start)]
        test = frame[frame["date_dt"].between(month_start, month_end)]
        if len(train) < 1000 or test.empty:
            continue
        train_through = train["date_dt"].max()
        if train_through >= test["date_dt"].min():
            temporal_violations += 1

        # Cache fast-path: si el mes ya cerró y TODOS sus juegos tienen mask
        # probabilities cacheadas del run anterior, saltar el fit+predict de las
        # 2047 combinaciones. Los resultados walk-forward de folds cerradas son
        # inmutables por definición (mismo train set = mismos resultados), así
        # que reusar es 100% seguro. Solo recomputamos la fold del mes actual.
        is_closed_month = month_end < today_ts
        test_game_pks = [int(row["gamePk"]) for _, row in test.iterrows()]
        all_cached = (
            not full_rebuild
            and is_closed_month
            and previous_predictions
            and all(gpk in previous_predictions for gpk in test_game_pks)
        )

        if all_cached:
            cached_folds_skipped += 1
            print(
                f"  Walk-forward {period_index}/{total_periods}: {period} — cache ({len(test)} juegos, sin reentrenar)",
                flush=True,
            )
            mask_probabilities = np.full((len(test), 2 ** len(FACTOR_KEYS)), np.nan, dtype=float)
            preserved_defaults = {}
            for test_index, gpk in enumerate(test_game_pks):
                previous = previous_predictions[gpk]
                mask_probabilities[test_index, :] = (
                    np.frombuffer(previous["packed"], dtype=np.uint8).astype(float) / 256.0
                )
                preserved_defaults[test_index] = previous["homeProbability"]
                preserved_historical_predictions += 1
        else:
            print(
                f"  Walk-forward {period_index}/{total_periods}: {period} "
                f"({len(train):,} entrenamiento, {len(test):,} prueba)",
                flush=True,
            )
            mask_probabilities = np.full((len(test), 2 ** len(FACTOR_KEYS)), np.nan, dtype=float)
            def predict_one(mask: int):
                mask_model = fit_mask_model(train, mask)
                return mask, predict_mask(mask_model, test, mask)

            with ThreadPoolExecutor(max_workers=MODEL_WORKERS) as executor:
                predicted_masks = executor.map(predict_one, range(1, 2 ** len(FACTOR_KEYS)))
                for mask, mask_values in predicted_masks:
                    mask_probabilities[:, mask] = mask_values

            preserved_defaults = {}
            for test_index, (_, row) in enumerate(test.iterrows()):
                previous = previous_predictions.get(int(row["gamePk"]))
                if previous is None:
                    continue
                mask_probabilities[test_index, :] = (
                    np.frombuffer(previous["packed"], dtype=np.uint8).astype(float) / 256.0
                )
                preserved_defaults[test_index] = previous["homeProbability"]
                preserved_historical_predictions += 1

        test_dates = test["date"].astype(str).to_numpy()
        day_indices = {day: np.flatnonzero(test_dates == day) for day in np.unique(test_dates)}
        for mask in range(1, 2 ** len(FACTOR_KEYS)):
            mask_values = mask_probabilities[:, mask]
            hits = (mask_values >= 0.5).astype(int) == test["homeWin"].to_numpy()
            total = combination_totals[mask]
            total["games"] += len(test)
            total["correct"] += int(hits.sum())
            season = str(period.year)
            total["seasons"][season]["games"] += len(test)
            total["seasons"][season]["correct"] += int(hits.sum())
            for day, indices in day_indices.items():
                total["days"][day]["games"] += int(len(indices))
                total["days"][day]["correct"] += int(hits[indices].sum())
        probabilities = mask_probabilities[:, DEFAULT_FACTOR_MASK]
        for test_index, previous_probability in preserved_defaults.items():
            probabilities[test_index] = previous_probability
        fold_correct = 0
        for test_index, ((_, row), home_probability) in enumerate(zip(test.iterrows(), probabilities)):
            predicted_home = bool(home_probability >= 0.5)
            actual_home = bool(row["homeWin"])
            correct = int(predicted_home == actual_home)
            fold_correct += correct
            predictions.append({
                "gamePk": int(row["gamePk"]),
                "date": row["date"],
                "foldMonth": str(period),
                "trainedThrough": train_through.date().isoformat(),
                "away": row["away"],
                "home": row["home"],
                "awayScore": int(row["awayScore"]),
                "homeScore": int(row["homeScore"]),
                "homeProbability": float(home_probability),
                "predictedWinner": row["home"] if predicted_home else row["away"],
                "predictedProbability": float(home_probability if predicted_home else 1 - home_probability),
                "actualWinner": row["home"] if actual_home else row["away"],
                "correct": correct,
                "maskProbabilities": [None if not math.isfinite(value) else float(value) for value in mask_probabilities[test_index]],
                "homeContext": row["homeContext"],
                "awayContext": row["awayContext"],
            })
        folds.append({
            "month": str(period),
            "trainedThrough": train_through.date().isoformat(),
            "trainGames": int(len(train)),
            "games": int(len(test)),
            "correct": int(fold_correct),
            "accuracy": fold_correct / len(test),
        })

    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame.empty:
        raise RuntimeError("Walk-forward produced no predictions")
    prediction_frame["actualHomeWin"] = (prediction_frame["actualWinner"] == prediction_frame["home"]).astype(int)
    prediction_frame["brierError"] = (prediction_frame["homeProbability"] - prediction_frame["actualHomeWin"]) ** 2
    days = {}
    for day, group in prediction_frame.groupby("date", sort=True):
        records = [
            {key: value for key, value in row.items() if key not in {"actualHomeWin", "brierError"}}
            for row in group.to_dict("records")
        ]
        days[str(day)] = {
            "date": str(day),
            "games": records,
            "total": int(len(group)),
            "correct": int(group["correct"].sum()),
            "accuracy": group["correct"].mean(),
            "brier": group["brierError"].mean(),
        }

    seasons = []
    prediction_frame["season"] = prediction_frame["date"].str[:4].astype(int)
    for season, group in prediction_frame.groupby("season"):
        seasons.append({
            "season": int(season),
            "games": int(len(group)),
            "correct": int(group["correct"].sum()),
            "accuracy": group["correct"].mean(),
            "brier": group["brierError"].mean(),
            "homeBaselineAccuracy": group["actualHomeWin"].mean(),
        })

    confidence_bands = []
    band_specs = [(0.50, 0.55, "50–54.9%"), (0.55, 0.60, "55–59.9%"), (0.60, 1.01, "60%+")]
    for low, high, label in band_specs:
        group = prediction_frame[prediction_frame["predictedProbability"].ge(low) & prediction_frame["predictedProbability"].lt(high)]
        confidence_bands.append({
            "label": label,
            "games": int(len(group)),
            "correct": int(group["correct"].sum()),
            "accuracy": group["correct"].mean() if len(group) else None,
            "averagePrediction": group["predictedProbability"].mean() if len(group) else None,
        })

    daily = pd.DataFrame([{"date": key, **{k: v for k, v in value.items() if k != "games"}} for key, value in days.items()])
    combinations = []
    for mask, values in combination_totals.items():
        seasons_for_mask = {}
        for season, season_values in values["seasons"].items():
            seasons_for_mask[season] = {
                **season_values,
                "accuracy": season_values["correct"] / season_values["games"],
            }
        day_values = list(values["days"].values())
        fifteen_game_days = [day for day in day_values if day["games"] == 15]
        fifteen_game_correct = sum(day["correct"] for day in fifteen_game_days)
        fifteen_game_games = len(fifteen_game_days) * 15
        winning_days = sum(day["correct"] > day["games"] / 2 for day in day_values)
        even_days = sum(day["correct"] == day["games"] / 2 for day in day_values)
        losing_days = sum(day["correct"] < day["games"] / 2 for day in day_values)
        combinations.append({
            "mask": mask,
            "games": values["games"],
            "correct": values["correct"],
            "accuracy": values["correct"] / values["games"],
            "seasons": seasons_for_mask,
            "days": len(day_values),
            "winningDays": winning_days,
            "evenDays": even_days,
            "losingDays": losing_days,
            "perfectDays": sum(day["correct"] == day["games"] for day in day_values),
            "fifteenGameDays": len(fifteen_game_days),
            "fifteenGameWinningDays": sum(day["correct"] >= 8 for day in fifteen_game_days),
            "fifteenGameGames": fifteen_game_games,
            "fifteenGameCorrect": fifteen_game_correct,
            "fifteenGameAccuracy": fifteen_game_correct / fifteen_game_games if fifteen_game_games else 0,
            "fifteenGamePerfectDays": sum(day["correct"] == 15 for day in fifteen_game_days),
        })
    combinations.sort(key=lambda item: (-item["accuracy"], bin(item["mask"]).count("1"), item["mask"]))
    best_winning_days = min(
        combinations,
        key=lambda item: (-item["winningDays"], -item["accuracy"], bin(item["mask"]).count("1"), item["mask"]),
    )
    best_fifteen_game_days = min(
        combinations,
        key=lambda item: (
            -item["fifteenGameAccuracy"],
            -item["fifteenGameWinningDays"],
            -item["accuracy"],
            bin(item["mask"]).count("1"),
            item["mask"],
        ),
    )
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "mode": "Monthly expanding walk-forward",
            "rule": "Cada mes se entrena solo con juegos terminados antes del primer día de ese mes y se congela el modelo para todos los juegos del mes.",
            "firstPredictionDate": prediction_frame["date"].min(),
            "lastPredictionDate": prediction_frame["date"].max(),
            "temporalLeakageViolations": temporal_violations,
            "preservedHistoricalPredictions": preserved_historical_predictions,
            "predictionHistoryRule": "Los gamePk ya publicados conservan sus probabilidades; sólo se calculan juegos nuevos.",
            "coverageAudit": coverage_audit,
        },
        "summary": {
            "games": int(len(prediction_frame)),
            "correct": int(prediction_frame["correct"].sum()),
            "accuracy": prediction_frame["correct"].mean(),
            "brier": prediction_frame["brierError"].mean(),
            "homeBaselineAccuracy": prediction_frame["actualHomeWin"].mean(),
            "folds": len(folds),
            "days": len(days),
            "winningDays": int((daily["accuracy"] > 0.5).sum()),
            "evenDays": int((daily["accuracy"] == 0.5).sum()),
            "losingDays": int((daily["accuracy"] < 0.5).sum()),
            "perfectDays": int((daily["accuracy"] == 1).sum()),
        },
        "defaultMask": DEFAULT_FACTOR_MASK,
        "bestWinningDaysMask": best_winning_days["mask"],
        "bestFifteenGameAccuracyMask": best_fifteen_game_days["mask"],
        "factorKeys": FACTOR_KEYS,
        "combinations": combinations,
        "seasons": seasons,
        "confidenceBands": confidence_bands,
        "folds": folds,
        "days": days,
    }


def _series_game_wins(row: pd.Series):
    wins = []
    for score in str(row.get("scoreline", "")).split(" | "):
        match = re.match(r"^([A-Z]+) (\d+)-(\d+) ([A-Z]+)$", score.strip())
        if not match:
            return []
        away, away_score, home_score, home = match.groups()
        winner = away if int(away_score) > int(home_score) else home
        wins.append(winner == str(row["team"]))
    return wins


def _series_rate_record(frame: pd.DataFrame):
    games = int(len(frame))
    wins = int(frame["sweep_for"].sum()) if games else 0
    return {"games": games, "sweeps": wins, "rate": wins / games if games else None}


def _series_rate_index(frame: pd.DataFrame, key: str):
    result = {}
    for value, group in frame.groupby(key, dropna=False):
        result[str(value)] = _series_rate_record(group)
    return result


def _series_probability_metrics(records: list[dict]):
    if not records:
        return {
            "events": 0, "sweeps": 0, "observedRate": None, "averageProbability": None,
            "brier": None, "accuracyAt50": None, "correctAt50": 0,
        }
    probabilities = np.asarray([row["probability"] for row in records], dtype=float)
    outcomes = np.asarray([row["outcome"] for row in records], dtype=int)
    correct = int(((probabilities >= .5).astype(int) == outcomes).sum())
    return {
        "events": int(len(records)),
        "sweeps": int(outcomes.sum()),
        "observedRate": float(outcomes.mean()),
        "averageProbability": float(probabilities.mean()),
        "brier": float(np.mean((probabilities - outcomes) ** 2)),
        "accuracyAt50": float(correct / len(records)),
        "correctAt50": correct,
    }


def build_series_sweep_audit(series: pd.DataFrame, walkforward: dict, cutoff_date: str):
    """Evaluate exact three-game sweeps using only information available before each stage."""
    frame = series.copy()
    frame["start_dt"] = pd.to_datetime(frame["start_date"])
    frame["end_dt"] = pd.to_datetime(frame["end_date"])
    frame["sweep_for"] = frame["sweep_for"].astype(int)
    sequences = frame.apply(_series_game_wins, axis=1)
    frame["won_game_1"] = [len(values) == 3 and bool(values[0]) for values in sequences]
    frame["won_first_2"] = [len(values) == 3 and bool(values[0]) and bool(values[1]) for values in sequences]
    frame["previous_sweep"] = (
        frame.sort_values(["team", "start_dt", "end_dt"])
        .groupby("team")["sweep_for"].shift(1).fillna(0).astype(int)
    )

    prediction_by_pk = {}
    for day in walkforward.get("days", {}).values():
        for game in day.get("games", []):
            prediction_by_pk[int(game["gamePk"])] = game

    def team_game_probability(game, team):
        home_probability = float(game["homeProbability"])
        return home_probability if str(game["home"]) == str(team) else 1 - home_probability

    def conditional_probability(history, current, stage):
        eligible = history[history["won_game_1"]] if stage == 1 else history[history["won_first_2"]]
        location = eligible[eligible["series_location"].eq(current["series_location"])]
        base_frame = location if len(location) >= 40 else eligible
        base = float(base_frame["sweep_for"].mean()) if len(base_frame) else .5

        def smoothed(sample, strength=18):
            return (float(sample["sweep_for"].sum()) + strength * base) / (len(sample) + strength)

        team_sample = eligible[eligible["team"].eq(current["team"])]
        opponent_sample = eligible[eligible["opponent"].eq(current["opponent"])]
        team_rate = smoothed(team_sample)
        opponent_rate = smoothed(opponent_sample)
        probability = .55 * team_rate + .45 * opponent_rate
        if int(current.get("previous_sweep", 0)):
            repeat_sample = eligible[eligible["previous_sweep"].eq(1)]
            repeat_rate = smoothed(repeat_sample, 28)
            probability = .85 * probability + .15 * repeat_rate
        return float(np.clip(probability, .01, .99))

    stage_records = {"beforeSeries": [], "afterGame1": [], "afterGame2": []}
    temporal_violations = 0
    for _, row in frame.sort_values(["start_dt", "team"]).iterrows():
        try:
            game_pks = [int(value) for value in str(row["game_pks"]).split(",")]
        except ValueError:
            continue
        if len(game_pks) != 3 or any(game_pk not in prediction_by_pk for game_pk in game_pks):
            continue
        games = [prediction_by_pk[game_pk] for game_pk in game_pks]
        fold_start = pd.Timestamp(str(games[0]["foldMonth"]) + "-01")
        history = frame[frame["end_dt"].lt(fold_start)]
        if history.empty:
            continue
        trained_through = pd.Timestamp(games[0]["trainedThrough"])
        if trained_through >= pd.Timestamp(games[0]["date"]):
            temporal_violations += 1
        p1 = team_game_probability(games[0], row["team"])
        p2 = team_game_probability(games[1], row["team"])
        p3 = team_game_probability(games[2], row["team"])
        common = {
            "seriesId": str(row["series_id"]), "season": int(row["season"]),
            "team": str(row["team"]), "opponent": str(row["opponent"]),
            "location": str(row["series_location"]), "outcome": int(row["sweep_for"]),
            "previousSweep": int(row["previous_sweep"]),
        }
        stage_records["beforeSeries"].append({
            **common, "gamePk": game_pks[0],
            "probability": p1 * conditional_probability(history, row, 1),
        })
        if bool(row["won_game_1"]):
            stage_records["afterGame1"].append({
                **common, "gamePk": game_pks[1],
                "probability": p2 * conditional_probability(history, row, 2),
            })
        if bool(row["won_first_2"]):
            stage_records["afterGame2"].append({**common, "gamePk": game_pks[2], "probability": p3})

    stages = {}
    for stage, records in stage_records.items():
        stages[stage] = {
            **_series_probability_metrics(records),
            "seasons": {
                str(season): _series_probability_metrics([row for row in records if row["season"] == season])
                for season in sorted({row["season"] for row in records})
            },
        }

    after_game_1 = frame[frame["won_game_1"]].copy()
    after_game_2 = frame[frame["won_first_2"]].copy()
    repeat = frame[frame["previous_sweep"].eq(1)].copy()
    double_walkforward = [row for row in stage_records["beforeSeries"] if row["previousSweep"] == 1]

    estimator = {}
    for key, eligible in (("afterGame1", after_game_1), ("afterGame2", after_game_2)):
        estimator[key] = {
            "overall": _series_rate_record(eligible),
            "byLocation": _series_rate_index(eligible, "series_location"),
            "byTeam": _series_rate_index(eligible, "team"),
            "byOpponent": _series_rate_index(eligible, "opponent"),
        }

    latest_completed = {}
    for team, group in frame.sort_values(["end_dt", "start_dt"]).groupby("team"):
        row = group.iloc[-1]
        latest_completed[str(team)] = {
            "endDate": str(row["end_date"])[:10], "opponent": str(row["opponent"]),
            "swept": bool(row["sweep_for"]), "wins": int(row["series_wins"]),
            "losses": int(row["series_losses"]),
        }

    team_rows = []
    for team, group in frame.groupby("team"):
        conversions = after_game_2[after_game_2["team"].eq(team)]
        repeats = repeat[repeat["team"].eq(team)]
        team_rows.append({
            "team": str(team), "name": str(group.iloc[-1]["team_name"]),
            "series": int(len(group)), "sweeps": int(group["sweep_for"].sum()),
            "sweepRate": float(group["sweep_for"].mean()),
            "stage2Games": int(len(conversions)), "stage2Sweeps": int(conversions["sweep_for"].sum()),
            "stage2Rate": float(conversions["sweep_for"].mean()) if len(conversions) else None,
            "repeatAttempts": int(len(repeats)), "repeatSweeps": int(repeats["sweep_for"].sum()),
            "repeatRate": float(repeats["sweep_for"].mean()) if len(repeats) else None,
        })
    team_rows.sort(key=lambda row: (-(row["repeatRate"] or -1), -row["repeatAttempts"], row["team"]))

    return {
        "status": "PRUEBA",
        "cutoffDate": cutoff_date,
        "method": (
            "Walk-forward expansivo mensual. En cada etapa se usa la probabilidad prepartido ya congelada del juego actual "
            "y tasas condicionales calculadas únicamente con series terminadas antes del mes evaluado."
        ),
        "temporalLeakageViolations": temporal_violations,
        "historical": {
            "beforeSeries": _series_rate_record(frame),
            "afterGame1": _series_rate_record(after_game_1),
            "afterGame2": _series_rate_record(after_game_2),
            "afterPreviousSweep": _series_rate_record(repeat),
            "doubleSweepUnconditional": {
                "pairs": int(len(frame) - frame["team"].nunique()),
                "doubleSweeps": int(repeat["sweep_for"].sum()),
                "rate": float(repeat["sweep_for"].sum() / (len(frame) - frame["team"].nunique())),
            },
        },
        "walkforward": {"stages": stages, "doubleSweep": _series_probability_metrics(double_walkforward)},
        "estimator": estimator,
        "latestCompleted": latest_completed,
        "teams": team_rows,
        "recommendation": (
            "Se publica como contexto de serie. No modifica el ganador del juego: el equipo que llega 2-0 convirtió cerca de la mitad "
            "de sus oportunidades y la señal de serie previa no mejoró el accuracy general."
        ),
    }


def audit_source_coverage(team_games: pd.DataFrame):
    canonical = pd.read_parquet(
        ROOT / "data" / "processed" / "games.parquet",
        columns=["game_pk", "game_date", "game_type", "status", "away_score", "home_score"],
    )
    canonical = canonical[
        canonical["game_type"].eq("R")
        & canonical["status"].eq("Final")
        & canonical["away_score"].notna()
        & canonical["home_score"].notna()
    ].drop_duplicates("game_pk")
    canonical["game_date"] = pd.to_datetime(canonical["game_date"]).dt.strftime("%Y-%m-%d")
    derived_ids = set(team_games["game_pk"].astype(int).unique())
    canonical_ids = set(canonical["game_pk"].astype(int))
    missing_ids = canonical_ids - derived_ids
    missing = canonical[canonical["game_pk"].isin(missing_ids)]
    missing_by_date = {str(key): int(value) for key, value in missing.groupby("game_date").size().items()}
    row_counts = team_games.groupby("game_pk").size()
    invalid_pairs = row_counts[row_counts.ne(2)]
    if missing_ids:
        detail = ", ".join(f"{day}: {count}" for day, count in missing_by_date.items())
        raise RuntimeError(
            "El derivado de series está desactualizado y omite juegos finales presentes en games.parquet "
            f"({len(missing_ids)} juegos; {detail}). Ejecuta primero mlb_three_game_sweep_report.py."
        )
    if len(invalid_pairs):
        raise RuntimeError(f"El derivado tiene {len(invalid_pairs)} game_pk sin exactamente dos filas de equipo.")
    return {
        "status": "COMPLETE",
        "derivedFinalGames": int(team_games["game_pk"].nunique()),
        "canonicalProcessedFinals": int(len(canonical_ids)),
        "coveredCanonicalFinals": int(len(canonical_ids & derived_ids)),
        "supplementalScheduleFinals": int(len(derived_ids - canonical_ids)),
        "missingCanonicalFinals": 0,
        "datesWithMissingGames": 0,
    }


def team_metrics(games: pd.DataFrame, series: pd.DataFrame):
    current = games[games["season"].eq(int(games["season"].max()))].copy()
    long_rows = []
    for _, game in current.sort_values(["sort_time", "game_pk"]).iterrows():
        for side in ("home", "away"):
            other = "away" if side == "home" else "home"
            runs_for = int(game[f"{side}_score"])
            runs_against = int(game[f"{other}_score"])
            long_rows.append({
                "team": game[f"{side}_team_abbrev"],
                "teamName": game[f"{side}_team_name"],
                "date": str(game["game_date"])[:10],
                "isHome": side == "home",
                "win": int(runs_for > runs_against),
                "runsFor": runs_for,
                "runsAgainst": runs_against,
            })
    long = pd.DataFrame(long_rows)
    ranking = pd.read_csv(WORK / "ranking_good_to_bad.csv")
    ranking_index = ranking.set_index("team")
    latest_series = series[series["season"].eq(int(games["season"].max()))].sort_values(["end_date", "start_date"]).groupby("team").tail(1).set_index("team")
    result = {}
    for team, group in long.groupby("team"):
        group = group.sort_values("date")
        home = group[group["isHome"]]
        away = group[~group["isHome"]]
        last10 = group.tail(10)
        row = ranking_index.loc[team] if team in ranking_index.index else None
        last = latest_series.loc[team] if team in latest_series.index else None
        result[team] = {
            "team": team,
            "name": group.iloc[-1]["teamName"],
            "games": len(group),
            "wins": int(group["win"].sum()),
            "losses": int(len(group) - group["win"].sum()),
            "winPct": group["win"].mean(),
            "last10Wins": int(last10["win"].sum()),
            "last10Losses": int(len(last10) - last10["win"].sum()),
            "last10Pct": last10["win"].mean(),
            "runsFor": int(group["runsFor"].sum()),
            "runsAgainst": int(group["runsAgainst"].sum()),
            "runDiff": int(group["runsFor"].sum() - group["runsAgainst"].sum()),
            "runDiffPerGame": (group["runsFor"].sum() - group["runsAgainst"].sum()) / len(group),
            "lastGameDate": str(group.iloc[-1]["date"]),
            "homeWinPct": home["win"].mean() if len(home) else None,
            "awayWinPct": away["win"].mean() if len(away) else None,
            "afterSweep": {"wins": int(row["after_sweep_wins"]), "games": int(row["after_sweep_games"]), "rate": row["after_sweep_win_rate"]} if row is not None else None,
            "afterSwept": {"wins": int(row["after_swept_wins"]), "games": int(row["after_swept_games"]), "rate": row["after_swept_win_rate"]} if row is not None else None,
            "sweeps": int(row["sweeps"]) if row is not None else None,
            "timesSwept": int(row["times_swept"]) if row is not None else None,
            "sweepRate": row["sweep_rate"] if row is not None else None,
            "sweptRate": row["swept_rate"] if row is not None else None,
            "latestExact3Series": {
                "start": str(last["start_date"])[:10],
                "end": str(last["end_date"])[:10],
                "opponent": last["opponent"],
                "wins": int(last["series_wins"]),
                "losses": int(last["series_losses"]),
                "runDiff": int(last["series_run_diff"]),
            } if last is not None else None,
        }
    return result


def previous_game_audit(feature_rows: pd.DataFrame, cutoff_date: str):
    canonical = pd.read_parquet(
        ROOT / "data" / "processed" / "games.parquet",
        columns=["game_pk", "game_date", "home_team_abbrev", "away_team_abbrev", "home_score", "away_score", "home_hits", "away_hits", "home_lob", "away_lob"],
    )
    box = pd.read_parquet(
        ROOT / "data" / "processed" / "team_box.parquet",
        columns=["game_pk", "side", "bat_homeRuns"],
    )
    box["side"] = box["side"].astype(str).str.lower()
    home_runs = box.pivot_table(index="game_pk", columns="side", values="bat_homeRuns", aggfunc="first").rename(columns={"home": "home_hr", "away": "away_hr"})
    canonical = canonical.merge(home_runs.reset_index(), on="game_pk", how="left")
    home_rows = pd.DataFrame({
        "gamePk": canonical["game_pk"], "date": pd.to_datetime(canonical["game_date"]), "team": canonical["home_team_abbrev"], "side": "home",
        "hits": canonical["home_hits"], "lob": canonical["home_lob"], "hr": canonical["home_hr"],
        "win": (canonical["home_score"] > canonical["away_score"]).astype(float),
    })
    away_rows = pd.DataFrame({
        "gamePk": canonical["game_pk"], "date": pd.to_datetime(canonical["game_date"]), "team": canonical["away_team_abbrev"], "side": "away",
        "hits": canonical["away_hits"], "lob": canonical["away_lob"], "hr": canonical["away_hr"],
        "win": (canonical["away_score"] > canonical["home_score"]).astype(float),
    })
    team_rows = pd.concat([home_rows, away_rows], ignore_index=True).sort_values(["team", "date", "gamePk"])
    for column in ["hits", "lob", "hr", "win"]:
        team_rows[f"prev_{column}"] = team_rows.groupby("team")[column].shift(1)
    team_rows["prev_loss"] = 1 - team_rows["prev_win"]
    for metric, thresholds in (("lob", [8, 10, 12]), ("hits", [10, 12, 14]), ("hr", [1, 2, 3])):
        for threshold in thresholds:
            for result, result_column in (("loss", "prev_loss"), ("win", "prev_win")):
                team_rows[f"prev_{metric}_ge{threshold}_{result}"] = (
                    team_rows[f"prev_{metric}"].ge(threshold) & team_rows[result_column].eq(1)
                ).astype(float)
    lag_columns = [column for column in team_rows.columns if column.startswith("prev_")]
    home_lag = team_rows[team_rows["side"].eq("home")][["gamePk", *lag_columns]].rename(columns={column: f"home_{column}" for column in lag_columns})
    away_lag = team_rows[team_rows["side"].eq("away")][["gamePk", *lag_columns]].rename(columns={column: f"away_{column}" for column in lag_columns})
    frame = feature_rows.merge(home_lag.merge(away_lag, on="gamePk", how="outer"), on="gamePk", how="left")
    for column in lag_columns:
        frame[f"edge_{column}"] = frame[f"home_{column}"] - frame[f"away_{column}"]
    frame["date_dt"] = pd.to_datetime(frame["date"])
    periods = sorted(frame.loc[frame["date_dt"].ge(pd.Timestamp("2024-03-01")), "date_dt"].dt.to_period("M").unique())
    base_features = features_for_mask(DEFAULT_FACTOR_MASK)
    groups = {
        "Diferencia de hits del juego anterior": ["edge_prev_hits"],
        "LOB alto separado por victoria/derrota": [f"edge_prev_lob_ge{threshold}_{result}" for threshold in (8, 10, 12) for result in ("loss", "win")],
        "Hits altos separados por victoria/derrota": [f"edge_prev_hits_ge{threshold}_{result}" for threshold in (10, 12, 14) for result in ("loss", "win")],
        "HR separados por victoria/derrota": [f"edge_prev_hr_ge{threshold}_{result}" for threshold in (1, 2, 3) for result in ("loss", "win")],
    }

    def evaluate(extra_features):
        records = []
        features = base_features + extra_features
        for period in periods:
            history = frame[frame["date_dt"].lt(period.start_time)]
            test = frame[frame["date_dt"].between(period.start_time, period.end_time.normalize())]
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.55, max_iter=2500, random_state=42))
            model.fit(history[features], history["homeWin"])
            probability = model.predict_proba(test[features])[:, 1]
            records.extend(zip(test["date"], test["homeWin"].astype(int), probability))
        out = pd.DataFrame(records, columns=["date", "actual", "probability"])
        out["correct"] = (out["probability"].ge(0.5).astype(int) == out["actual"]).astype(int)
        out["season"] = out["date"].str[:4]
        return out

    baseline_frame = evaluate([])
    baseline_accuracy = baseline_frame["correct"].mean()
    baseline_2026 = baseline_frame.loc[baseline_frame["season"].eq("2026"), "correct"].mean()
    reasons = {
        "Diferencia de hits del juego anterior": "Mejoró el total, pero no añadió aciertos en 2026 y el intervalo todavía incluye cero.",
        "LOB alto separado por victoria/derrota": "La ganancia reciente no compensó la pérdida global y el comportamiento cambió entre temporadas.",
        "Hits altos separados por victoria/derrota": "Empeoró tanto el total como 2026.",
        "HR separados por victoria/derrota": "No superó la base y también retrocedió en 2026.",
    }
    candidates = []
    rng = np.random.default_rng(20260824)
    for name, extra_features in groups.items():
        candidate = evaluate(extra_features)
        candidate_2026 = candidate.loc[candidate["season"].eq("2026"), "correct"].mean()
        delta = candidate["correct"].to_numpy() - baseline_frame["correct"].to_numpy()
        item = {
            "name": name,
            "accuracy": candidate["correct"].mean(),
            "correct": int(candidate["correct"].sum()),
            "deltaPoints": delta.mean() * 100,
            "accuracy2026": candidate_2026,
            "delta2026Points": (candidate_2026 - baseline_2026) * 100,
            "status": "NO APLICADO",
            "reason": reasons[name],
        }
        if name == "Diferencia de hits del juego anterior":
            boot = [delta[rng.integers(0, len(delta), len(delta))].mean() * 100 for _ in range(5000)]
            item["ci95Points"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        candidates.append(item)

    descriptive = []
    for label, metric, threshold, previous_win in (
        ("Dejó 10+ en base y perdió", "prev_lob", 10, 0),
        ("Conectó 10+ hits y perdió", "prev_hits", 10, 0),
        ("Conectó 2+ HR y perdió", "prev_hr", 2, 0),
        ("Dejó 10+ en base y ganó", "prev_lob", 10, 1),
    ):
        sample = team_rows[team_rows[metric].ge(threshold) & team_rows["prev_win"].eq(previous_win)]
        latest = sample[sample["date"].dt.year.eq(2026)]
        descriptive.append({"label": label, "games": len(sample), "nextWinRate": sample["win"].mean(), "nextWinRate2026": latest["win"].mean()})
    return {
        "method": "Monthly expanding walk-forward; every fold uses only games before that month.",
        "cutoffDate": cutoff_date,
        "games": len(baseline_frame),
        "baseline": {"accuracy": baseline_accuracy, "correct": int(baseline_frame["correct"].sum()), "accuracy2026": baseline_2026},
        "candidates": candidates,
        "descriptive": descriptive,
    }


def best_players_audit(feature_rows: pd.DataFrame, cutoff_date: str):
    game_times = pd.read_parquet(
        ROOT / "data" / "processed" / "games.parquet",
        columns=["game_pk", "first_pitch_utc"],
    ).drop_duplicates("game_pk")
    meta = feature_rows[["gamePk", "date", "season", "home", "away", "homeWin"]].merge(
        game_times, left_on="gamePk", right_on="game_pk", how="left"
    )
    meta["sortTime"] = pd.to_datetime(meta["first_pitch_utc"], utc=True, errors="coerce")
    fallback = pd.to_datetime(meta["date"], utc=True) + pd.to_timedelta(meta.groupby("date").cumcount(), unit="s")
    meta["sortTime"] = meta["sortTime"].fillna(fallback)

    players = pd.read_parquet(
        ROOT / "data" / "processed" / "player_box.parquet",
        columns=[
            "game_pk", "side", "team_id", "player_id", "position", "started_batting",
            "bat_plateAppearances", "bat_atBats", "bat_hits", "bat_doubles", "bat_triples",
            "bat_homeRuns", "bat_baseOnBalls", "bat_hitByPitch", "bat_sacFlies",
        ],
    )
    players["side"] = players["side"].astype(str).str.lower()
    players = players.merge(meta[["gamePk", "season", "sortTime"]], left_on="game_pk", right_on="gamePk", how="inner")
    players = players.sort_values(["team_id", "player_id", "sortTime", "game_pk"]).reset_index(drop=True)
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
    players["woba_den"] = players["bat_atBats"] + players["bat_baseOnBalls"] + players["bat_hitByPitch"] + players["bat_sacFlies"]
    players["started"] = players["started_batting"].eq(True).astype(float)
    players["roster_game"] = 1.0
    player_group = players.groupby(["team_id", "player_id"], sort=False)
    players["prior_woba_num"] = player_group["woba_num"].cumsum() - players["woba_num"]
    players["prior_woba_den"] = player_group["woba_den"].cumsum() - players["woba_den"]
    players["prior_starts"] = player_group["started"].cumsum() - players["started"]
    players["prior_roster_games"] = player_group["roster_game"].cumsum() - 1.0
    players["prior_woba"] = (players["prior_woba_num"] + .320 * 50) / (players["prior_woba_den"] + 50)
    players["prior_start_rate"] = (players["prior_starts"] + 1.5) / (players["prior_roster_games"] + 3)
    players["star_score"] = players["prior_woba"] + .06 * players["prior_start_rate"]

    hitters = players[~players["position"].astype(str).eq("P")].copy()
    hitters["star_rank"] = hitters.groupby(["game_pk", "side"])["star_score"].rank(method="first", ascending=False)
    hitters["core_rank"] = hitters.groupby(["game_pk", "side"])["prior_start_rate"].rank(method="first", ascending=False)
    hitters["is_top5"] = hitters["star_rank"].le(5)
    hitters["is_core8"] = hitters["core_rank"].le(8)
    hitters["top5_started_value"] = hitters["is_top5"].astype(float) * hitters["started"]
    hitters["core8_started_value"] = hitters["is_core8"].astype(float) * hitters["started"]
    hitters["star_pa_value"] = hitters["bat_plateAppearances"] * hitters["is_top5"].astype(float)
    hitters["rank_weight_started"] = np.where(hitters["is_top5"], 6 - hitters["star_rank"], 0) * hitters["started"]
    usage = hitters.groupby(["game_pk", "side"], as_index=False).agg(
        top5_started=("top5_started_value", "sum"),
        core8_started=("core8_started_value", "sum"),
        star_pa=("star_pa_value", "sum"),
        total_pa=("bat_plateAppearances", "sum"),
        star_weight_started=("rank_weight_started", "sum"),
    )
    usage["star_pa_share"] = usage["star_pa"] / usage["total_pa"].replace(0, np.nan)
    usage["star_weight_share"] = usage["star_weight_started"] / 15.0
    lineup = pd.read_parquet(
        ROOT / "data" / "processed" / "features_lineup.parquet",
        columns=["game_pk", "side", "lineup_xwoba_vs_sp_hand", "lineup_top4_xwoba_vs_sp_hand"],
    )
    lineup["side"] = lineup["side"].astype(str).str.lower()
    usage = usage.merge(lineup, on=["game_pk", "side"], how="left").merge(
        meta[["gamePk", "sortTime", "home", "away", "homeWin"]], left_on="game_pk", right_on="gamePk", how="left"
    )
    usage["team"] = np.where(usage["side"].eq("home"), usage["home"], usage["away"])
    usage["current_win"] = np.where(usage["side"].eq("home"), usage["homeWin"], 1 - usage["homeWin"])
    usage = usage.sort_values(["team", "sortTime", "game_pk"])
    for column in ["lineup_xwoba_vs_sp_hand", "lineup_top4_xwoba_vs_sp_hand"]:
        grouped = usage.groupby("team", sort=False)[column]
        prior_mean = grouped.transform(lambda series: series.shift(1).rolling(10, min_periods=5).mean())
        prior_std = grouped.transform(lambda series: series.shift(1).rolling(30, min_periods=10).std()).clip(lower=.01)
        usage[f"{column}_relative"] = usage[column] - prior_mean
        usage[f"{column}_z"] = (usage[column] - prior_mean) / prior_std

    previous_columns = [
        "top5_started", "core8_started", "star_pa_share", "star_weight_share",
        "lineup_xwoba_vs_sp_hand_relative", "lineup_top4_xwoba_vs_sp_hand_relative",
    ]
    for column in previous_columns:
        usage[f"prev_{column}"] = usage.groupby("team")[column].shift(1)
    usage["prev_win"] = usage.groupby("team")["current_win"].shift(1)
    usage["prev_loss"] = 1 - usage["prev_win"]
    threshold_conditions = {
        "top5_ge4": usage["prev_top5_started"].ge(4),
        "core8_ge7": usage["prev_core8_started"].ge(7),
        "star_pa_ge50": usage["prev_star_pa_share"].ge(.50),
        "star_weight_ge80": usage["prev_star_weight_share"].ge(.80),
    }
    for prefix, condition in threshold_conditions.items():
        for result, result_column in (("win", "prev_win"), ("loss", "prev_loss")):
            usage[f"prev_{prefix}_{result}"] = (condition & usage[result_column].eq(1)).astype(float)

    lag_columns = [column for column in usage.columns if column.startswith("prev_")]
    home_lag = usage[usage["side"].eq("home")][["game_pk", *lag_columns]].rename(columns={column: f"home_{column}" for column in lag_columns})
    away_lag = usage[usage["side"].eq("away")][["game_pk", *lag_columns]].rename(columns={column: f"away_{column}" for column in lag_columns})
    audit_frame = feature_rows.merge(home_lag.merge(away_lag, on="game_pk", how="outer"), left_on="gamePk", right_on="game_pk", how="left")
    for column in lag_columns:
        audit_frame[f"edge_{column}"] = audit_frame[f"home_{column}"] - audit_frame[f"away_{column}"]
    audit_frame["date_dt"] = pd.to_datetime(audit_frame["date"])
    periods = sorted(audit_frame.loc[audit_frame["date_dt"].ge(pd.Timestamp("2024-03-01")), "date_dt"].dt.to_period("M").unique())
    base_features = features_for_mask(DEFAULT_FACTOR_MASK)
    groups = {
        "Top 5 por resultado": [f"edge_prev_top5_ge4_{result}" for result in ("win", "loss")],
        "Peso de estrellas por resultado": [f"edge_prev_star_weight_ge80_{result}" for result in ("win", "loss")],
        "Todas las medidas por resultado": BEST_PLAYERS_TEST_FEATURES,
        "Calidad del lineup anterior": ["edge_prev_lineup_xwoba_vs_sp_hand_relative", "edge_prev_lineup_top4_xwoba_vs_sp_hand_relative"],
        "Uso continuo de estrellas": ["edge_prev_top5_started", "edge_prev_star_pa_share", "edge_prev_star_weight_share"],
    }

    def evaluate(extra_features):
        records = []
        features = base_features + extra_features
        for period in periods:
            history = audit_frame[audit_frame["date_dt"].lt(period.start_time)]
            test = audit_frame[audit_frame["date_dt"].between(period.start_time, period.end_time.normalize())]
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=.55, max_iter=2500, random_state=42))
            model.fit(history[features], history["homeWin"])
            probability = model.predict_proba(test[features])[:, 1]
            records.extend(zip(test["date"], test["homeWin"].astype(int), probability))
        out = pd.DataFrame(records, columns=["date", "actual", "probability"])
        out["correct"] = (out["probability"].ge(.5).astype(int) == out["actual"]).astype(int)
        out["season"] = out["date"].str[:4]
        return out

    baseline_frame = evaluate([])
    baseline_accuracy = baseline_frame["correct"].mean()
    baseline_2026 = baseline_frame.loc[baseline_frame["season"].eq("2026"), "correct"].mean()
    candidates = []
    rng = np.random.default_rng(20260824)
    for name, features in groups.items():
        candidate = evaluate(features)
        candidate_2026 = candidate.loc[candidate["season"].eq("2026"), "correct"].mean()
        delta = candidate["correct"].to_numpy() - baseline_frame["correct"].to_numpy()
        boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() * 100 for _ in range(5000)])
        ci = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
        delta_points = delta.mean() * 100
        delta_2026 = (candidate_2026 - baseline_2026) * 100
        if delta_points > 0 and delta_2026 > 0 and ci[0] > 0:
            status = "IMPLEMENTAR"
            reason = "Mejoró total y 2026 con un intervalo positivo."
        elif delta_points > 0 and delta_2026 > 0:
            status = "OBSERVACIÓN"
            reason = "Mejoró total y 2026, pero el intervalo todavía incluye cero; requiere más juegos antes de activarse."
        else:
            status = "NO APLICADO"
            reason = "La mejora histórica no se sostuvo en 2026." if delta_points > 0 else "No superó la base walk-forward."
        candidates.append({
            "name": name, "accuracy": candidate["correct"].mean(), "correct": int(candidate["correct"].sum()),
            "deltaPoints": delta_points, "accuracy2026": candidate_2026, "delta2026Points": delta_2026,
            "ci95Points": ci, "status": status, "reason": reason,
            "seasons": {str(season): part["correct"].mean() for season, part in candidate.groupby("season")},
        })
    candidates.sort(key=lambda item: -item["accuracy"])

    usage["next_win"] = usage.groupby("team")["current_win"].shift(-1)
    descriptive = []
    scenarios = {
        "4+ de las 5 estrellas iniciaron": usage["top5_started"].ge(4),
        "Estrellas tomaron 50%+ de PA": usage["star_pa_share"].ge(.50),
        "Lineup 1.0σ+ sobre su nivel": usage["lineup_xwoba_vs_sp_hand_z"].ge(1.0),
    }
    for label, condition in scenarios.items():
        for result_label, previous_win in (("Ganó", 1), ("Perdió", 0)):
            sample = usage[condition & usage["current_win"].eq(previous_win) & usage["next_win"].notna()]
            latest = sample[sample["sortTime"].dt.year.eq(2026)]
            descriptive.append({
                "label": f"{label} y {result_label.lower()}", "games": len(sample),
                "nextWinRate": sample["next_win"].mean(), "nextWinRate2026": latest["next_win"].mean(),
            })
    latest_team_signals = {}
    for _, row in usage.groupby("team", sort=False).tail(1).iterrows():
        signals = {}
        current_conditions = {
            "top5_ge4": row["top5_started"] >= 4,
            "core8_ge7": row["core8_started"] >= 7,
            "star_pa_ge50": row["star_pa_share"] >= .50,
            "star_weight_ge80": row["star_weight_share"] >= .80,
        }
        for prefix, condition in current_conditions.items():
            signals[f"prev_{prefix}_win"] = float(condition and row["current_win"] == 1)
            signals[f"prev_{prefix}_loss"] = float(condition and row["current_win"] == 0)
        latest_team_signals[str(row["team"])] = {
            "gamePk": int(row["game_pk"]),
            "playedAt": row["sortTime"].isoformat(),
            "signals": signals,
        }

    audit = {
        "method": "Estrellas definidas con producción y titularidad acumuladas antes de cada juego; evaluación monthly expanding walk-forward.",
        "cutoffDate": cutoff_date,
        "coverage": {"teamGames": len(usage), "lineupQualityGames": int(usage["lineup_xwoba_vs_sp_hand"].notna().sum())},
        "games": len(baseline_frame),
        "baseline": {"accuracy": baseline_accuracy, "correct": int(baseline_frame["correct"].sum()), "accuracy2026": baseline_2026},
        "candidates": candidates,
        "descriptive": descriptive,
        "latestTeamSignals": latest_team_signals,
        "recommendation": "PRUEBA opcional: la variante 'Todas las medidas por resultado' queda disponible, apagada por defecto. Mejoró total y 2026, pero su intervalo todavía incluye cero.",
    }
    return audit_frame, audit


def scenario_stats(eligible: pd.DataFrame):
    paired = eligible.copy()
    paired["pair_key"] = paired.apply(
        lambda r: f"{str(r['next_game_date'])[:10]}|{'-'.join(sorted([str(r['team']), str(r['next_opponent'])]))}", axis=1
    )
    matchups = []
    for _, group in paired.groupby("pair_key"):
        if len(group) != 2:
            continue
        home_rows = group[group["next_location"].eq("Casa")]
        if len(home_rows) != 1:
            continue
        home = home_rows.iloc[0]
        away = group[group.index != home.name].iloc[0]
        matchups.append({
            "homeCode": context_code(home),
            "awayCode": context_code(away),
            "homeWin": int(home["next_win"]),
            "homeRunDiff": int(home["series_run_diff"]),
            "awayRunDiff": int(away["series_run_diff"]),
        })
    both_sweep = [m for m in matchups if m["homeCode"] == "sweep" and m["awayCode"] == "sweep"]
    both_swept = [m for m in matchups if m["homeCode"] == "swept" and m["awayCode"] == "swept"]
    swept_vs_12 = [m for m in matchups if {m["homeCode"], m["awayCode"]} == {"swept", "lost_1_2"}]

    def summary(rows):
        wins = sum(m["homeWin"] for m in rows)
        return {"games": len(rows), "homeWins": wins, "awayWins": len(rows) - wins, "homeWinRate": wins / len(rows) if rows else None}

    swept_wins = sum(m["homeWin"] if m["homeCode"] == "swept" else 1 - m["homeWin"] for m in swept_vs_12)
    swept_home = [m for m in swept_vs_12 if m["homeCode"] == "swept"]
    swept_away = [m for m in swept_vs_12 if m["awayCode"] == "swept"]
    return {
        "bothSweep": summary(both_sweep),
        "bothSwept": summary(both_swept),
        "sweptVsLost12": {
            "games": len(swept_vs_12),
            "sweptWins": swept_wins,
            "lost12Wins": len(swept_vs_12) - swept_wins,
            "sweptWinRate": swept_wins / len(swept_vs_12) if swept_vs_12 else None,
            "sweptAtHome": {"games": len(swept_home), "wins": sum(m["homeWin"] for m in swept_home)},
            "sweptAway": {"games": len(swept_away), "wins": sum(1 - m["homeWin"] for m in swept_away)},
        },
    }


def rankings():
    good = pd.read_csv(WORK / "ranking_good_to_bad.csv")
    vulnerable = pd.read_csv(WORK / "ranking_most_vulnerable.csv")
    columns = [
        "team", "team_name", "three_game_series", "sweeps", "sweep_rate", "times_swept", "swept_rate",
        "net_sweeps", "net_sweep_rate", "baseline_win_rate", "after_sweep_games", "after_sweep_wins",
        "after_sweep_win_rate", "after_swept_games", "after_swept_wins", "after_swept_win_rate",
    ]
    return {
        "goodToBad": good[columns].to_dict("records"),
        "mostVulnerable": vulnerable[columns].to_dict("records"),
        "afterSweep": good.sort_values(["after_sweep_win_rate", "after_sweep_games"], ascending=False)[columns].to_dict("records"),
        "afterSwept": good.sort_values(["after_swept_win_rate", "after_swept_games"], ascending=False)[columns].to_dict("records"),
    }


def fetch_today():
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={TODAY.isoformat()}&hydrate=probablePitcher,team,linescore"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
        games = []
        for block in payload.get("dates", []):
            for game in block.get("games", []):
                teams = game.get("teams", {})
                games.append({
                    "gamePk": game.get("gamePk"),
                    "gameDate": game.get("gameDate"),
                    "officialDate": game.get("officialDate"),
                    "status": game.get("status", {}).get("detailedState"),
                    "abstractState": game.get("status", {}).get("abstractGameState"),
                    "venue": game.get("venue", {}).get("name"),
                    "away": {
                        "team": teams.get("away", {}).get("team", {}).get("abbreviation"),
                        "name": teams.get("away", {}).get("team", {}).get("name"),
                        "score": teams.get("away", {}).get("score"),
                        "pitcher": teams.get("away", {}).get("probablePitcher", {}).get("fullName"),
                    },
                    "home": {
                        "team": teams.get("home", {}).get("team", {}).get("abbreviation"),
                        "name": teams.get("home", {}).get("team", {}).get("name"),
                        "score": teams.get("home", {}).get("score"),
                        "pitcher": teams.get("home", {}).get("probablePitcher", {}).get("fullName"),
                    },
                })
        return {"sourceState": "LIVE SCHEDULE", "source": "MLB StatsAPI", "retrievedAt": datetime.now(timezone.utc).isoformat(), "games": games}
    except Exception as exc:
        return {"sourceState": "LOCAL FALLBACK", "source": "No fue posible consultar MLB StatsAPI", "retrievedAt": datetime.now(timezone.utc).isoformat(), "error": str(exc), "games": []}


def american_to_decimal(value):
    odds = safe_float(value, default=float("nan"))
    if not math.isfinite(odds) or odds == 0:
        return None
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def build_current_odds():
    source = ROOT / "data" / "processed" / "odds_pinnacle.parquet"
    if not source.exists():
        return {
            "sourceState": "UNAVAILABLE",
            "source": "odds_pinnacle.parquet no encontrado",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "games": [],
        }
    frame = pd.read_parquet(source)
    required = {"event_id", "commence_time", "away_team", "home_team", "bookmaker_name", "last_update", "home_ml", "away_ml"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"odds_pinnacle.parquet no contiene las columnas requeridas: {sorted(missing)}")
    frame = frame.sort_values("last_update").drop_duplicates(["event_id", "bookmaker_name"], keep="last")
    games = []
    for event_id, group in frame.groupby("event_id", sort=False):
        row = group.iloc[-1]
        books = []
        for _, book in group.sort_values("bookmaker_name").iterrows():
            away_decimal = american_to_decimal(book["away_ml"])
            home_decimal = american_to_decimal(book["home_ml"])
            if away_decimal is None and home_decimal is None:
                continue
            books.append({
                "name": str(book["bookmaker_name"]),
                "awayAmerican": safe_float(book["away_ml"], None),
                "homeAmerican": safe_float(book["home_ml"], None),
                "awayDecimal": away_decimal,
                "homeDecimal": home_decimal,
                "lastUpdated": str(book["last_update"]),
            })
        best_away = max((book for book in books if book["awayDecimal"] is not None), key=lambda book: book["awayDecimal"], default=None)
        best_home = max((book for book in books if book["homeDecimal"] is not None), key=lambda book: book["homeDecimal"], default=None)
        games.append({
            "eventId": str(event_id),
            "commenceTime": str(row["commence_time"]),
            "awayTeam": str(row["away_team"]),
            "homeTeam": str(row["home_team"]),
            "lastUpdated": max((book["lastUpdated"] for book in books), default=None),
            "books": books,
            "bestAway": {"book": best_away["name"], "american": best_away["awayAmerican"], "decimal": best_away["awayDecimal"]} if best_away else None,
            "bestHome": {"book": best_home["name"], "american": best_home["homeAmerican"], "decimal": best_home["homeDecimal"]} if best_home else None,
        })
    return {
        "sourceState": "REAL DATA",
        "source": "STRIKECAST data/processed/odds_pinnacle.parquet",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceUpdatedAt": max((game["lastUpdated"] for game in games if game["lastUpdated"]), default=None),
        "oddsFormat": "american_and_decimal",
        "selectionRule": "Mejor momio decimal disponible entre las casas presentes para cada lado.",
        "games": sorted(games, key=lambda game: (game["commenceTime"], game["awayTeam"])),
    }


def build_today_context(series: pd.DataFrame, games: pd.DataFrame, today_schedule: dict):
    current_season = int(games["season"].max())
    latest_final = pd.concat(
        [
            games[["game_date", "home_team_abbrev"]].rename(columns={"home_team_abbrev": "team"}),
            games[["game_date", "away_team_abbrev"]].rename(columns={"away_team_abbrev": "team"}),
        ],
        ignore_index=True,
    ).groupby("team")["game_date"].max()
    current_series = series[series["season"].eq(current_season)].copy()
    current_series["end_date_dt"] = pd.to_datetime(current_series["end_date"])
    latest_series = current_series.sort_values(["end_date_dt", "start_date"]).groupby("team").tail(1).set_index("team")
    result = {}
    for matchup in today_schedule.get("games", []):
        away = "ATH" if matchup["away"]["team"] == "OAK" else matchup["away"]["team"]
        home = "ATH" if matchup["home"]["team"] == "OAK" else matchup["home"]["team"]
        for team, opponent in ((away, home), (home, away)):
            if team not in latest_series.index or team not in latest_final.index:
                continue
            row = latest_series.loc[team]
            if pd.Timestamp(row["end_date_dt"]).normalize() != pd.Timestamp(latest_final.loc[team]).normalize():
                continue
            gap = (pd.Timestamp(TODAY) - pd.Timestamp(row["end_date_dt"]).normalize()).days
            if gap < 1 or gap > 3:
                continue
            code = context_code(row)
            result[f"{team}|{opponent}"] = {
                "code": code,
                "label": STATUS_LABELS[code],
                "seriesStart": str(row["start_date"])[:10],
                "seriesEnd": str(row["end_date"])[:10],
                "opponent": row["opponent"],
                "wins": int(row["series_wins"]),
                "losses": int(row["series_losses"]),
                "runDiff": int(row["series_run_diff"]),
                "nextLocation": "Casa" if team == home else "Visitante",
            }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Genera walkforward + JSONs para StatsMLB.")
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Ignora el cache de walkforward.json anterior y reentrena TODAS las folds "
             "históricas. Usar cuando cambia el feature set, hyperparams o hay dudas de "
             "integridad. Toma ~35 min en vez de ~5 min.",
    )
    args = parser.parse_args()
    _full_rebuild = args.full_rebuild

    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    print("  Cargando histórico final y contexto de series...", flush=True)
    team_games = pd.read_csv(WORK / "games_final_regular.csv")
    coverage_audit = audit_source_coverage(team_games)
    team_games["is_home"] = team_games["is_home"].astype(str).str.lower().eq("true")
    home = team_games[team_games["is_home"]].set_index("game_pk")
    away = team_games[~team_games["is_home"]].set_index("game_pk")
    common = home.index.intersection(away.index)
    games = pd.DataFrame({
        "game_pk": common,
        "season": home.loc[common, "season"].to_numpy(),
        "game_date": home.loc[common, "game_date"].to_numpy(),
        "home_team_abbrev": home.loc[common, "team"].to_numpy(),
        "home_team_name": home.loc[common, "team_name"].to_numpy(),
        "away_team_abbrev": away.loc[common, "team"].to_numpy(),
        "away_team_name": away.loc[common, "team_name"].to_numpy(),
        "home_score": home.loc[common, "runs_for"].to_numpy(),
        "away_score": away.loc[common, "runs_for"].to_numpy(),
    })
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["sort_time"] = games["game_date"].dt.tz_localize("UTC") + pd.to_timedelta(games.groupby("game_date").cumcount(), unit="s")
    games["home_win"] = games["home_score"] > games["away_score"]
    series = pd.read_csv(WORK / "team_series.csv")
    context_map, eligible = build_context_map(series)
    print("  Construyendo variables previas a cada juego...", flush=True)
    feature_rows, _ = pregame_features(games, context_map)
    feature_rows = attach_market_schedule_features(feature_rows)
    print("  Variables base y mercado listas.", flush=True)
    summary = json.loads((WORK / "summary.json").read_text(encoding="utf-8"))
    previous_game = previous_game_audit(feature_rows, summary["coverage"]["last_final_date"])
    print("  Auditoría del juego anterior lista.", flush=True)
    feature_rows, coach_rotation = attach_coach_rotation_features(feature_rows, summary["coverage"]["last_final_date"])
    print("  Rotación del coach lista.", flush=True)
    feature_rows, best_players = best_players_audit(feature_rows, summary["coverage"]["last_final_date"])
    print("  Uso de mejores jugadores listo.", flush=True)
    feature_rows, rotation_quality = attach_rotation_quality_features(
        ROOT, feature_rows, summary["coverage"]["last_final_date"]
    )
    print("  Calidad de rotación lista.", flush=True)
    feature_rows, lineup_fatigue = attach_lineup_fatigue_features(
        ROOT, feature_rows, summary["coverage"]["last_final_date"]
    )
    print("  Fatiga del lineup lista.", flush=True)
    feature_rows, opponent_form = attach_opponent_adjusted_form(
        feature_rows, summary["coverage"]["last_final_date"]
    )
    print("  Forma ajustada por rival lista.", flush=True)
    indication_pairs = build_indication_pair_audit(feature_rows, summary["coverage"]["last_final_date"])
    print("  Cruces de indicaciones listos.", flush=True)
    print("  Entrenando combinaciones para los juegos actuales...", flush=True)
    model_config = fit_model(feature_rows)
    today_schedule = fetch_today()
    today_context = {
        f"{team}|{opponent}": value
        for (day, team, opponent), value in context_map.items()
        if day == TODAY.isoformat()
    }
    today_context.update(build_today_context(series, games, today_schedule))
    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "cutoffDate": summary["coverage"]["last_final_date"],
        "definition": summary["definition"],
        "coverage": summary["coverage"],
        "seriesSummary": summary["series"],
        "afterSweepSummary": summary["after_sweep"],
        "afterSweptSummary": summary["after_being_swept"],
        "rankings": rankings(),
        "scenarios": scenario_stats(eligible),
        "teams": team_metrics(games, series),
        "todayContext": today_context,
        "model": model_config,
        "previousGameAudit": previous_game,
        "bestPlayersAudit": best_players,
        "coachRotationAudit": coach_rotation,
        "rotationQualityAudit": rotation_quality,
        "lineupFatigueAudit": lineup_fatigue,
        "opponentFormAudit": opponent_form,
        "indicationPairAudit": indication_pairs,
        "statusLabels": STATUS_LABELS,
        "additionalAnalyses": [
            {"name": "Abridor probable", "status": "Disponible parcialmente", "detail": "Comparar calidad reciente, descanso y mano del abridor; se activa cuando existan estadísticas verificadas del pitcher."},
            {"name": "Bullpen y fatiga", "status": "Siguiente integración", "detail": "Entradas lanzadas en 1, 3 y 5 días, relevistas no disponibles y carreras recientes."},
            {"name": "Lineup, rotación y fatiga", "status": "Activo · pruebas", "detail": "Cantidad, calidad y carga reciente del lineup están disponibles como pruebas cuando MLB confirma ambas alineaciones; faltan lesiones verificadas."},
            {"name": "Viaje", "status": "Siguiente integración", "detail": "Cambio de ciudad, distancia recorrida y día después de juego nocturno; el diferencial de descanso ya está activo junto con mercado."},
            {"name": "Parque y clima", "status": "Siguiente integración", "detail": "Factor del estadio, temperatura, viento, techo y condiciones que cambian el entorno de carreras."},
            {"name": "Cara a cara", "status": "Exploratorio", "detail": "Resultados recientes entre ambos equipos, siempre separados de la fuerza general para evitar muestras engañosas."},
        ],
    }
    print("  Ejecutando validación walk-forward mensual...", flush=True)
    walkforward = build_walkforward(feature_rows, coverage_audit, full_rebuild=_full_rebuild)
    output["seriesSweepAudit"] = build_series_sweep_audit(
        series, walkforward, summary["coverage"]["last_final_date"]
    )
    print("  Walk-forward experimental de barridas listo.", flush=True)
    output["coachRotationAudit"] = finalize_coach_rotation_audit(output["coachRotationAudit"], walkforward)
    output["rotationQualityAudit"] = finalize_rotation_quality_audit(output["rotationQualityAudit"], walkforward)
    output["lineupFatigueAudit"] = finalize_lineup_fatigue_audit(output["lineupFatigueAudit"], walkforward)
    output["opponentFormAudit"] = finalize_opponent_form_audit(output["opponentFormAudit"], walkforward)
    try:
        pb = pd.read_parquet(
            ROOT / "data" / "processed" / "player_box.parquet",
            columns=["game_pk", "player_id", "player_name"],
        ).dropna(subset=["player_id", "player_name"])
        pb = pb.sort_values("game_pk")
        pb = pb.drop_duplicates(subset=["player_id"], keep="last")
        output["playerNames"] = {
            str(int(row["player_id"])): str(row["player_name"])
            for _, row in pb.iterrows()
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  playerNames lookup skipped: {exc}", flush=True)
        output["playerNames"] = {}
    (PUBLIC_DATA / "stats.json").write_text(
        json.dumps(clean(output), ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    walkforward = compact_walkforward_probabilities(walkforward)
    write_walkforward_artifacts(walkforward)
    print("  Generando auditoría estricta Temporada + Forma L10...", flush=True)
    write_season_l10_audit(WORK / "games_final_regular.csv", PUBLIC_DATA)
    (PUBLIC_DATA / "today-snapshot.json").write_text(json.dumps(clean(today_schedule), ensure_ascii=False, indent=2), encoding="utf-8")
    current_odds = build_current_odds()
    (PUBLIC_DATA / "current-odds.json").write_text(json.dumps(clean(current_odds), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(clean({"stats": str(PUBLIC_DATA / "stats.json"), "model": model_config["metrics"], "walkforward": walkforward["summary"], "today": today_schedule["sourceState"], "todayContextSignals": len(today_context), "currentOdds": {"state": current_odds["sourceState"], "games": len(current_odds["games"]), "updatedAt": current_odds.get("sourceUpdatedAt")}}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
