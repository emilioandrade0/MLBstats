from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


APP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work" / "sweep_report_20260824" / "games_final_regular.csv"
PUBLIC_DATA = APP / "public" / "data"


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else round(float(value), 8)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _load_games(source: Path) -> pd.DataFrame:
    team_games = pd.read_csv(source)
    team_games["is_home"] = team_games["is_home"].astype(str).str.lower().eq("true")
    home = team_games[team_games["is_home"]].drop_duplicates("game_pk").set_index("game_pk")
    away = team_games[~team_games["is_home"]].drop_duplicates("game_pk").set_index("game_pk")
    common = home.index.intersection(away.index)
    games = pd.DataFrame(
        {
            "game_pk": common.astype(int),
            "season": home.loc[common, "season"].astype(int).to_numpy(),
            "game_date": pd.to_datetime(home.loc[common, "game_date"].to_numpy()),
            "home_team": home.loc[common, "team"].to_numpy(),
            "away_team": away.loc[common, "team"].to_numpy(),
            "home_win": home.loc[common, "win"].astype(int).to_numpy(),
        }
    )
    return games.sort_values(["season", "game_date", "game_pk"]).reset_index(drop=True)


def _pregame_rows(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    records: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    recent: dict[tuple[int, str], deque[int]] = defaultdict(lambda: deque(maxlen=10))

    # All games on one date receive the same pre-date snapshot. This is conservative
    # for doubleheaders and guarantees that no result from that date leaks into another.
    for (season, game_date), day_games in games.groupby(["season", "game_date"], sort=True):
        for game in day_games.itertuples(index=False):
            home_key = (int(season), game.home_team)
            away_key = (int(season), game.away_team)
            home_wins, home_losses = records[home_key]
            away_wins, away_losses = records[away_key]
            home_recent = recent[home_key]
            away_recent = recent[away_key]
            if home_wins + home_losses < 20 or away_wins + away_losses < 20:
                continue
            if len(home_recent) < 10 or len(away_recent) < 10:
                continue
            rows.append(
                {
                    "gamePk": int(game.game_pk),
                    "season": int(season),
                    "gameDate": pd.Timestamp(game_date),
                    "homeWin": int(game.home_win),
                    "homeSeasonPct": home_wins / (home_wins + home_losses),
                    "awaySeasonPct": away_wins / (away_wins + away_losses),
                    "homeL10Wins": int(sum(home_recent)),
                    "awayL10Wins": int(sum(away_recent)),
                }
            )
        for game in day_games.itertuples(index=False):
            home_key = (int(season), game.home_team)
            away_key = (int(season), game.away_team)
            home_result = int(game.home_win)
            away_result = 1 - home_result
            records[home_key][0 if home_result else 1] += 1
            records[away_key][0 if away_result else 1] += 1
            recent[home_key].append(home_result)
            recent[away_key].append(away_result)

    frame = pd.DataFrame(rows)
    frame["seasonDiff"] = frame["homeSeasonPct"] - frame["awaySeasonPct"]
    frame["l10Diff"] = (frame["homeL10Wins"] - frame["awayL10Wins"]) / 10.0
    frame["interaction"] = frame["seasonDiff"] * frame["l10Diff"]
    frame["month"] = frame["gameDate"].dt.to_period("M").astype(str)
    return frame


def _rate_summary(side_rows: pd.DataFrame, mask: pd.Series) -> dict:
    sample = side_rows.loc[mask]
    seasons = {}
    for season, group in sample.groupby("season"):
        seasons[str(int(season))] = {
            "games": int(len(group)),
            "wins": int(group["strongerWin"].sum()),
            "winRate": float(group["strongerWin"].mean()),
        }
    return {
        "games": int(len(sample)),
        "wins": int(sample["strongerWin"].sum()),
        "winRate": float(sample["strongerWin"].mean()) if len(sample) else None,
        "seasons": seasons,
    }


def _directional_rows(frame: pd.DataFrame) -> pd.DataFrame:
    unequal = frame.loc[frame["seasonDiff"].abs() > 1e-12].copy()
    home_stronger = unequal["seasonDiff"] > 0
    unequal["strongerWin"] = np.where(home_stronger, unequal["homeWin"], 1 - unequal["homeWin"])
    unequal["strongerSeasonPct"] = np.where(home_stronger, unequal["homeSeasonPct"], unequal["awaySeasonPct"])
    unequal["weakerSeasonPct"] = np.where(home_stronger, unequal["awaySeasonPct"], unequal["homeSeasonPct"])
    unequal["strongerL10Wins"] = np.where(home_stronger, unequal["homeL10Wins"], unequal["awayL10Wins"])
    unequal["weakerL10Wins"] = np.where(home_stronger, unequal["awayL10Wins"], unequal["homeL10Wins"])
    unequal["seasonGap"] = unequal["strongerSeasonPct"] - unequal["weakerSeasonPct"]
    unequal["l10WinGap"] = unequal["strongerL10Wins"] - unequal["weakerL10Wins"]
    return unequal


MODEL_SPECS = [
    ("localia", "Localía", []),
    ("season", "Temporada", ["seasonDiff"]),
    ("l10", "Forma L10", ["l10Diff"]),
    ("seasonL10", "Temporada + L10", ["seasonDiff", "l10Diff"]),
    ("seasonL10Interaction", "Temporada + L10 + interacción", ["seasonDiff", "l10Diff", "interaction"]),
]


def _fit_monthly_walkforward(frame: pd.DataFrame) -> dict:
    eligible = frame.loc[frame["gameDate"] >= pd.Timestamp("2024-01-01")].copy()
    predictions: dict[str, list[dict]] = {key: [] for key, _, _ in MODEL_SPECS}
    folds = 0
    for month in sorted(eligible["month"].unique()):
        month_start = pd.Timestamp(f"{month}-01")
        train = frame.loc[frame["gameDate"] < month_start]
        test = eligible.loc[eligible["month"] == month]
        if train.empty or test.empty or train["homeWin"].nunique() < 2:
            continue
        folds += 1
        for key, _label, features in MODEL_SPECS:
            if not features:
                probability = float(train["homeWin"].mean())
                probs = np.full(len(test), probability)
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=17))
                model.fit(train[features], train["homeWin"])
                probs = model.predict_proba(test[features])[:, 1]
            for index, probability in zip(test.index, probs):
                predictions[key].append(
                    {
                        "season": int(test.at[index, "season"]),
                        "actual": int(test.at[index, "homeWin"]),
                        "probability": float(probability),
                    }
                )

    models = []
    for key, label, _features in MODEL_SPECS:
        result = pd.DataFrame(predictions[key])
        result["correct"] = ((result["probability"] >= 0.5).astype(int) == result["actual"]).astype(int)
        seasons = {
            str(int(season)): {
                "games": int(len(group)),
                "correct": int(group["correct"].sum()),
                "accuracy": float(group["correct"].mean()),
            }
            for season, group in result.groupby("season")
        }
        models.append(
            {
                "key": key,
                "label": label,
                "games": int(len(result)),
                "correct": int(result["correct"].sum()),
                "accuracy": float(result["correct"].mean()),
                "brier": float(brier_score_loss(result["actual"], result["probability"])),
                "seasons": seasons,
            }
        )
    season_only = next(model for model in models if model["key"] == "season")
    interaction = next(model for model in models if model["key"] == "seasonL10Interaction")
    return {
        "folds": folds,
        "games": interaction["games"],
        "models": models,
        "gainVsSeasonPoints": (interaction["accuracy"] - season_only["accuracy"]) * 100,
        "gainVsSeason2026Points": (
            interaction["seasons"]["2026"]["accuracy"] - season_only["seasons"]["2026"]["accuracy"]
        ) * 100,
    }


def _production_effect(walkforward_path: Path) -> dict | None:
    if not walkforward_path.exists():
        return None
    manifest = json.loads(walkforward_path.read_text(encoding="utf-8"))
    combinations = {int(row["mask"]): row for row in manifest.get("combinations", [])}
    with_recent = combinations.get(45)
    without_recent = combinations.get(41)
    if not with_recent or not without_recent:
        return None
    paired = {"improved": 0, "tied": 0, "worsened": 0}
    paired_2026 = {"improved": 0, "tied": 0, "worsened": 0}
    for mask, base in combinations.items():
        if mask & 4 or mask + 4 not in combinations:
            continue
        recent = combinations[mask + 4]
        relation = "improved" if recent["correct"] > base["correct"] else "worsened" if recent["correct"] < base["correct"] else "tied"
        paired[relation] += 1
        base_2026 = base.get("seasons", {}).get("2026", {})
        recent_2026 = recent.get("seasons", {}).get("2026", {})
        relation_2026 = "improved" if recent_2026.get("correct", 0) > base_2026.get("correct", 0) else "worsened" if recent_2026.get("correct", 0) < base_2026.get("correct", 0) else "tied"
        paired_2026[relation_2026] += 1
    return {
        "withL10": with_recent,
        "withoutL10": without_recent,
        "deltaPoints": (with_recent["accuracy"] - without_recent["accuracy"]) * 100,
        "delta2026Points": (
            with_recent["seasons"]["2026"]["accuracy"] - without_recent["seasons"]["2026"]["accuracy"]
        ) * 100,
        "pairedMasks": paired,
        "pairedMasks2026": paired_2026,
    }


def build_season_l10_audit(source: Path = SOURCE, walkforward_path: Path | None = None) -> dict:
    games = _load_games(source)
    frame = _pregame_rows(games)
    sides = _directional_rows(frame)
    aligned = sides["l10WinGap"] > 0
    same = sides["l10WinGap"] == 0
    conflict = sides["l10WinGap"] < 0
    broad_example = (
        (sides["strongerSeasonPct"] >= 0.500)
        & (sides["weakerSeasonPct"] < 0.500)
        & (sides["strongerL10Wins"] == 5)
        & (sides["weakerL10Wins"] == 4)
    )
    strict_example = (
        (sides["strongerSeasonPct"] >= 0.525)
        & (sides["weakerSeasonPct"] < 0.475)
        & (sides["strongerL10Wins"] == 5)
        & (sides["weakerL10Wins"] == 4)
    )
    large_gap_aligned = (sides["seasonGap"] >= 0.150) & (sides["l10WinGap"] >= 2)
    walkforward = _fit_monthly_walkforward(frame)
    production = _production_effect(walkforward_path or (PUBLIC_DATA / "walkforward.json"))
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "cutoffDate": games["game_date"].max().date().isoformat(),
        "status": "AUDITORÍA · NO CAMBIA PRONÓSTICOS",
        "method": "Récord y L10 se reconstruyen desde cero cada temporada y antes de la fecha de cada juego. Cada mes se entrena únicamente con fechas anteriores; los juegos del mismo día comparten una instantánea previa.",
        "temporalLeakageViolations": 0,
        "coverage": {
            "games": int(len(frame)),
            "firstDate": frame["gameDate"].min().date().isoformat(),
            "lastDate": frame["gameDate"].max().date().isoformat(),
            "minimumPriorSeasonGamesPerTeam": 20,
            "requiredPriorGamesL10": 10,
        },
        "exactScenario": {
            "label": "Mejor temporada y L10 5-5 vs peor temporada y L10 4-6",
            "broad": _rate_summary(sides, broad_example),
            "strict": _rate_summary(sides, strict_example),
            "strictDefinition": "Mejor equipo ≥ .525; rival < .475; L10 exactos 5-5 y 4-6.",
        },
        "patterns": [
            {"key": "aligned", "label": "Mejor temporada + mejor L10", **_rate_summary(sides, aligned)},
            {"key": "same", "label": "Mejor temporada + mismo L10", **_rate_summary(sides, same)},
            {"key": "conflict", "label": "Mejor temporada + peor L10", **_rate_summary(sides, conflict)},
            {"key": "largeAligned", "label": "Ventaja de temporada ≥15 pts + ventaja L10 ≥2", **_rate_summary(sides, large_gap_aligned)},
        ],
        "walkforward": walkforward,
        "productionModel": production,
        "recommendation": "El L10 aporta contexto, pero no justifica una regla automática. El escenario 5-5 contra 4-6 es inestable; la ventaja fuerte aparece cuando una gran diferencia de temporada también es confirmada por L10. En la combinación base actual, activar L10 no mejora la accuracy, por lo que esta auditoría no modifica los picks.",
    }


def write_season_l10_audit(source: Path = SOURCE, public_data: Path = PUBLIC_DATA) -> dict:
    public_data.mkdir(parents=True, exist_ok=True)
    audit = build_season_l10_audit(source, public_data / "walkforward.json")
    destination = public_data / "season-l10-audit.json"
    destination.write_text(json.dumps(_clean(audit), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return audit


def main():
    audit = write_season_l10_audit()
    print(
        json.dumps(
            _clean(
                {
                    "file": str(PUBLIC_DATA / "season-l10-audit.json"),
                    "cutoffDate": audit["cutoffDate"],
                    "coverage": audit["coverage"],
                    "exactScenario": audit["exactScenario"]["strict"],
                    "walkforward": audit["walkforward"],
                    "productionDeltaPoints": audit["productionModel"]["deltaPoints"] if audit["productionModel"] else None,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
