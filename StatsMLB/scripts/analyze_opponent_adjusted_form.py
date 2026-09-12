from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_data as bd
from opponent_adjusted_form import SIGNAL_COLUMNS, attach_opponent_adjusted_form


def base_frame():
    team_games = pd.read_csv(bd.WORK / "games_final_regular.csv")
    team_games["is_home"] = team_games["is_home"].astype(str).str.lower().eq("true")
    home = team_games[team_games["is_home"]].set_index("game_pk")
    away = team_games[~team_games["is_home"]].set_index("game_pk")
    common = home.index.intersection(away.index)
    games = pd.DataFrame({
        "game_pk": common, "season": home.loc[common, "season"].to_numpy(),
        "game_date": home.loc[common, "game_date"].to_numpy(),
        "home_team_abbrev": home.loc[common, "team"].to_numpy(), "home_team_name": home.loc[common, "team_name"].to_numpy(),
        "away_team_abbrev": away.loc[common, "team"].to_numpy(), "away_team_name": away.loc[common, "team_name"].to_numpy(),
        "home_score": home.loc[common, "runs_for"].to_numpy(), "away_score": away.loc[common, "runs_for"].to_numpy(),
    })
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["sort_time"] = games["game_date"].dt.tz_localize("UTC") + pd.to_timedelta(games.groupby("game_date").cumcount(), unit="s")
    games["home_win"] = games["home_score"] > games["away_score"]
    series = pd.read_csv(bd.WORK / "team_series.csv")
    context_map, _ = bd.build_context_map(series)
    frame, _ = bd.pregame_features(games, context_map)
    return bd.attach_market_schedule_features(frame)


def evaluate(frame: pd.DataFrame, features: list[str]):
    work = frame.copy()
    work["date_dt"] = pd.to_datetime(work["date"])
    periods = sorted(work.loc[work["date_dt"].ge("2024-03-01"), "date_dt"].dt.to_period("M").unique())
    rows = []
    for period in periods:
        train = work[work["date_dt"].lt(period.start_time)]
        test = work[work["date_dt"].between(period.start_time, period.end_time.normalize())]
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(C=.55, max_iter=2500, random_state=42),
        )
        model.fit(train[features], train["homeWin"])
        probabilities = model.predict_proba(test[features])[:, 1]
        rows.extend((date, int(actual), float(probability), str(period)) for date, actual, probability in zip(test["date"], test["homeWin"], probabilities))
    result = pd.DataFrame(rows, columns=["date", "actual", "probability", "fold"])
    result["correct"] = result["probability"].ge(.5).astype(int).eq(result["actual"]).astype(int)
    result["season"] = result["date"].str[:4]
    return result


def summarize(candidate: pd.DataFrame, baseline: pd.DataFrame, name: str, features: list[str]):
    delta = candidate["correct"].to_numpy() - baseline["correct"].to_numpy()
    folds = candidate.groupby("fold")["correct"].sum() - baseline.groupby("fold")["correct"].sum()
    rng = np.random.default_rng(20260824)
    boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() * 100 for _ in range(10000)])
    seasons = {}
    for season in sorted(candidate["season"].unique()):
        sample = candidate[candidate["season"].eq(season)]
        base = baseline[baseline["season"].eq(season)]
        seasons[season] = {
            "accuracy": sample["correct"].mean(), "base": base["correct"].mean(),
            "deltaPoints": (sample["correct"].mean() - base["correct"].mean()) * 100,
        }
    design = candidate[candidate["season"].isin(["2024", "2025"])]
    design_base = baseline[baseline["season"].isin(["2024", "2025"])]
    periods = {}
    for label, low, high in (("h1_2026", "2026-01-01", "2026-06-30"), ("h2_2026", "2026-07-01", "2026-12-31")):
        sample = candidate[candidate["date"].between(low, high)]
        base = baseline[baseline["date"].between(low, high)]
        periods[label] = {"games": len(sample), "accuracy": sample["correct"].mean(), "deltaPoints": (sample["correct"].mean() - base["correct"].mean()) * 100}
    return {
        "name": name, "features": features, "games": len(candidate), "correct": int(candidate["correct"].sum()),
        "accuracy": candidate["correct"].mean(), "deltaPoints": delta.mean() * 100,
        "designAccuracy": design["correct"].mean(), "designDeltaPoints": (design["correct"].mean() - design_base["correct"].mean()) * 100,
        "holdoutAccuracy": seasons["2026"]["accuracy"], "holdoutDeltaPoints": seasons["2026"]["deltaPoints"],
        "ci95Points": [np.quantile(boot, .025), np.quantile(boot, .975)],
        "foldsBetter": int((folds > 0).sum()), "foldsEqual": int((folds == 0).sum()), "foldsWorse": int((folds < 0).sum()),
        "seasons": seasons, **periods,
    }


def main():
    frame, audit = attach_opponent_adjusted_form(base_frame(), "2026-08-23")
    base_features = bd.features_for_mask(bd.DEFAULT_FACTOR_MASK)
    groups = {
        "adjusted_l10": ["edge_adjustedResultL10", "edge_adjustedMarginL10"],
        "opponent_context": ["edge_opponentStrengthL10", "edge_opponentStrengthL20"],
        "quality_outcomes": ["edge_qualityResultL10", "edge_qualityResultL20"],
        "multiwindow_results": ["edge_adjustedResultL5", "edge_adjustedResultL10", "edge_adjustedResultL20", "edge_formTrend"],
        "multiwindow_margin": ["edge_adjustedMarginL5", "edge_adjustedMarginL10", "edge_adjustedMarginL20", "edge_marginTrend"],
        "compact": ["edge_adjustedResultL10", "edge_adjustedMarginL10", "edge_opponentStrengthL10", "edge_formTrend"],
        "all_adjusted_form": [f"edge_{name}" for name in SIGNAL_COLUMNS],
    }
    baseline = evaluate(frame, base_features)
    results = [summarize(evaluate(frame, base_features + features), baseline, name, features) for name, features in groups.items()]
    eligible_design = [item for item in results if item["seasons"]["2024"]["deltaPoints"] >= 0 and item["seasons"]["2025"]["deltaPoints"] >= 0]
    selected = sorted(eligible_design or results, key=lambda item: (-item["designDeltaPoints"], -item["accuracy"]))[0]
    selected["passesGate"] = bool(
        selected["designDeltaPoints"] > 0 and selected["holdoutDeltaPoints"] > 0
        and selected["h1_2026"]["deltaPoints"] >= 0 and selected["h2_2026"]["deltaPoints"] >= 0
    )
    results.sort(key=lambda item: (-item["designDeltaPoints"], -item["accuracy"]))
    print(json.dumps(bd.clean({
        "selectionRule": "Elegir con 2024-2025 entre variantes no negativas en ambos años; exigir mejora 2026 y no retroceder en H1/H2.",
        "coverage": audit["coverage"], "baseline": summarize(baseline, baseline, "baseline", []),
        "selected": selected, "results": results,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
