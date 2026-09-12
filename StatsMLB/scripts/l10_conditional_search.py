from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


APP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "work" / "sweep_report_20260824" / "games_final_regular.csv"
OUTPUT = APP / "outputs" / "l10-conditional-search.json"
BASE_FEATURES = [
    "run_diff_diff",
    "market_logit_p_home",
    "market_n_providers",
    "market_p_home_std",
    "days_since_last_game_diff",
]


def load_frame() -> pd.DataFrame:
    team_games = pd.read_csv(SOURCE)
    team_games["is_home"] = team_games["is_home"].astype(str).str.lower().eq("true")
    home = team_games[team_games["is_home"]].drop_duplicates("game_pk").set_index("game_pk")
    away = team_games[~team_games["is_home"]].drop_duplicates("game_pk").set_index("game_pk")
    common = home.index.intersection(away.index)
    games = pd.DataFrame({
        "gamePk": common.astype(int),
        "season": home.loc[common, "season"].astype(int).to_numpy(),
        "date": home.loc[common, "game_date"].astype(str).str[:10].to_numpy(),
        "home": home.loc[common, "team"].to_numpy(),
        "away": away.loc[common, "team"].to_numpy(),
        "homeWin": home.loc[common, "win"].astype(int).to_numpy(),
        "homeScore": home.loc[common, "runs_for"].astype(int).to_numpy(),
        "awayScore": away.loc[common, "runs_for"].astype(int).to_numpy(),
    }).sort_values(["season", "date", "gamePk"]).reset_index(drop=True)

    overall = defaultdict(lambda: {"games": 0, "runsFor": 0, "runsAgainst": 0})
    seasonal = defaultdict(lambda: {"games": 0, "wins": 0, "recent": deque(maxlen=10)})
    rows = []
    for (season, day), group in games.groupby(["season", "date"], sort=True):
        for game in group.itertuples(index=False):
            home_overall, away_overall = overall[game.home], overall[game.away]
            home_season, away_season = seasonal[(season, game.home)], seasonal[(season, game.away)]
            home_games, away_games = home_season["games"], away_season["games"]
            season_diff = (
                home_season["wins"] / home_games - away_season["wins"] / away_games
                if home_games and away_games else np.nan
            )
            l10_diff = (
                (sum(home_season["recent"]) - sum(away_season["recent"])) / 10.0
                if len(home_season["recent"]) == 10 and len(away_season["recent"]) == 10 else np.nan
            )
            home_run_rate = (home_overall["runsFor"] - home_overall["runsAgainst"]) / (home_overall["games"] + 10)
            away_run_rate = (away_overall["runsFor"] - away_overall["runsAgainst"]) / (away_overall["games"] + 10)
            rows.append({
                "gamePk": game.gamePk,
                "date": day,
                "season": int(season),
                "homeWin": game.homeWin,
                "season_diff": season_diff,
                "l10_diff": l10_diff,
                "season_games_min": min(home_games, away_games),
                "run_diff_diff": float(np.clip(home_run_rate - away_run_rate, -3, 3)),
            })
        for game in group.itertuples(index=False):
            for team, won, runs_for, runs_against in (
                (game.home, game.homeWin, game.homeScore, game.awayScore),
                (game.away, 1 - game.homeWin, game.awayScore, game.homeScore),
            ):
                overall[team]["games"] += 1
                overall[team]["runsFor"] += runs_for
                overall[team]["runsAgainst"] += runs_against
                item = seasonal[(season, team)]
                item["games"] += 1
                item["wins"] += won
                item["recent"].append(won)

    frame = pd.DataFrame(rows)
    market = pd.read_parquet(
        ROOT / "data" / "processed" / "train.parquet",
        columns=["game_pk", *BASE_FEATURES[1:]],
    ).drop_duplicates("game_pk")
    frame = frame.merge(market, left_on="gamePk", right_on="game_pk", how="left").drop(columns="game_pk")
    frame["date_dt"] = pd.to_datetime(frame["date"])
    return frame


def add_candidate(frame: pd.DataFrame, season_threshold: float, l10_threshold: int, kind: str) -> pd.DataFrame:
    result = frame.copy()
    eligible = (
        result["season_games_min"].ge(20)
        & result["season_diff"].abs().ge(season_threshold)
        & result["l10_diff"].abs().ge(l10_threshold / 10)
        & result["season_diff"].mul(result["l10_diff"]).gt(0)
    )
    if kind == "binary":
        values = np.sign(result["season_diff"])
    elif kind == "l10":
        values = result["l10_diff"]
    else:
        values = result["season_diff"] * result["l10_diff"]
    result["l10_confirmed"] = np.where(eligible, values, 0.0)
    return result


def evaluate(frame: pd.DataFrame, extra_feature: bool) -> pd.DataFrame:
    features = [*BASE_FEATURES, *( ["l10_confirmed"] if extra_feature else [])]
    records = []
    months = sorted(frame.loc[frame["date_dt"].ge(pd.Timestamp("2024-03-01")), "date_dt"].dt.to_period("M").unique())
    for period in months:
        train = frame.loc[frame["date_dt"].lt(period.start_time)]
        test = frame.loc[frame["date_dt"].between(period.start_time, period.end_time.normalize())]
        if len(train) < 1000 or test.empty:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.55, max_iter=2500, random_state=42),
        )
        model.fit(train[features], train["homeWin"])
        probability = model.predict_proba(test[features])[:, 1]
        records.extend({
            "date": date,
            "actual": int(actual),
            "probability": float(prob),
        } for date, actual, prob in zip(test["date"], test["homeWin"], probability))
    result = pd.DataFrame(records)
    result["correct"] = (result["probability"].ge(0.5).astype(int) == result["actual"]).astype(int)
    result["season"] = result["date"].str[:4].astype(int)
    return result


def summarize(result: pd.DataFrame) -> dict:
    daily = result.groupby("date")["correct"].agg(["sum", "count"])
    def slice_metrics(group: pd.DataFrame) -> dict:
        if group.empty:
            return {"games": 0, "correct": 0, "accuracy": None}
        return {"games": len(group), "correct": int(group["correct"].sum()), "accuracy": float(group["correct"].mean())}
    return {
        **slice_metrics(result),
        "brier": float(brier_score_loss(result["actual"], result["probability"])),
        "winningDays": int((daily["sum"] > daily["count"] / 2).sum()),
        "days": int(len(daily)),
        "design2024_2025": slice_metrics(result[result["season"].isin([2024, 2025])]),
        "year2026": slice_metrics(result[result["season"].eq(2026)]),
        "h1_2026": slice_metrics(result[(result["season"].eq(2026)) & (result["date"].str[5:7].astype(int).le(6))]),
        "h2_2026": slice_metrics(result[(result["season"].eq(2026)) & (result["date"].str[5:7].astype(int).ge(7))]),
    }


def main():
    frame = load_frame()
    baseline_predictions = evaluate(frame.assign(l10_confirmed=0.0), False)
    baseline = summarize(baseline_predictions)
    candidates = []
    for season_threshold in (0.05, 0.075, 0.10, 0.125, 0.15):
        for l10_threshold in (1, 2, 3, 4):
            for kind in ("binary", "l10", "interaction"):
                candidate_frame = add_candidate(frame, season_threshold, l10_threshold, kind)
                predictions = evaluate(candidate_frame, True)
                summary = summarize(predictions)
                summary.update({
                    "seasonThreshold": season_threshold,
                    "l10Threshold": l10_threshold,
                    "kind": kind,
                    "deltaPoints": (summary["accuracy"] - baseline["accuracy"]) * 100,
                    "deltaDesignPoints": (summary["design2024_2025"]["accuracy"] - baseline["design2024_2025"]["accuracy"]) * 100,
                    "delta2026Points": (summary["year2026"]["accuracy"] - baseline["year2026"]["accuracy"]) * 100,
                    "deltaH1Points": (summary["h1_2026"]["accuracy"] - baseline["h1_2026"]["accuracy"]) * 100,
                    "deltaH2Points": (summary["h2_2026"]["accuracy"] - baseline["h2_2026"]["accuracy"]) * 100,
                    "deltaWinningDays": summary["winningDays"] - baseline["winningDays"],
                })
                candidates.append(summary)
    candidates.sort(key=lambda item: (-item["deltaDesignPoints"], -item["delta2026Points"], -item["deltaPoints"]))
    selected = candidates[0]
    selected["passesGate"] = all(selected[key] > 0 for key in ("deltaPoints", "delta2026Points", "deltaH1Points", "deltaH2Points"))
    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "method": "Umbrales elegidos únicamente por 2024-2025; 2026 y sus dos mitades se conservan como verificación.",
        "baseline": baseline,
        "selected": selected,
        "candidates": candidates,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "selected": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
