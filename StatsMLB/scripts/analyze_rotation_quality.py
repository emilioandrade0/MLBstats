from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_data as bd


SIGNALS = [
    "qualityDelta",
    "replacementDelta",
    "retentionQuality",
    "top4QualityDelta",
    "lossQualityDelta",
    "lossReplacementDelta",
    "lossRetentionQuality",
    "lossTop4QualityDelta",
]


def build_quality_features(feature_rows: pd.DataFrame):
    game_times = pd.read_parquet(
        bd.ROOT / "data" / "processed" / "games.parquet",
        columns=["game_pk", "first_pitch_utc"],
    ).drop_duplicates("game_pk")
    meta = feature_rows[["gamePk", "date", "season", "home", "away", "homeWin"]].merge(
        game_times, left_on="gamePk", right_on="game_pk", how="left"
    )
    meta["sortTime"] = pd.to_datetime(meta["first_pitch_utc"], utc=True, errors="coerce")
    fallback = pd.to_datetime(meta["date"], utc=True) + pd.to_timedelta(meta.groupby("date").cumcount(), unit="s")
    meta["sortTime"] = meta["sortTime"].fillna(fallback)

    columns = [
        "game_pk", "side", "team_id", "player_id", "position", "started_batting", "batting_order",
        "bat_plateAppearances", "bat_atBats", "bat_hits", "bat_doubles", "bat_triples", "bat_homeRuns",
        "bat_baseOnBalls", "bat_hitByPitch", "bat_sacFlies",
    ]
    players = pd.read_parquet(bd.ROOT / "data" / "processed" / "player_box.parquet", columns=columns)
    players["side"] = players["side"].astype(str).str.lower()
    players = players.merge(meta[["gamePk", "sortTime", "home", "away", "homeWin"]], left_on="game_pk", right_on="gamePk", how="inner")
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
    players["woba_den"] = players["bat_atBats"] + players["bat_baseOnBalls"] + players["bat_hitByPitch"] + players["bat_sacFlies"]
    players["started"] = players["started_batting"].eq(True).astype(float)
    players["roster_game"] = 1.0
    grouped = players.groupby(["team", "player_id"], sort=False)
    players["prior_woba_num"] = grouped["woba_num"].cumsum() - players["woba_num"]
    players["prior_woba_den"] = grouped["woba_den"].cumsum() - players["woba_den"]
    players["prior_starts"] = grouped["started"].cumsum() - players["started"]
    players["prior_roster_games"] = grouped["roster_game"].cumsum() - 1.0
    players["post_woba_num"] = players["prior_woba_num"] + players["woba_num"]
    players["post_woba_den"] = players["prior_woba_den"] + players["woba_den"]
    players["post_starts"] = players["prior_starts"] + players["started"]
    players["post_roster_games"] = players["prior_roster_games"] + 1.0
    players["prior_quality"] = (
        (players["prior_woba_num"] + .320 * 50) / (players["prior_woba_den"] + 50)
        + .06 * (players["prior_starts"] + 1.5) / (players["prior_roster_games"] + 3)
    )
    players["post_quality"] = (
        (players["post_woba_num"] + .320 * 50) / (players["post_woba_den"] + 50)
        + .06 * (players["post_starts"] + 1.5) / (players["post_roster_games"] + 3)
    )
    players["slot"] = (pd.to_numeric(players["batting_order"], errors="coerce") // 100).astype("Int64")
    hitters = players[~players["position"].astype(str).eq("P")].copy()
    hitters = hitters.sort_values(["sortTime", "game_pk", "side", "player_id"])

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
        signals = {name: np.nan for name in SIGNALS}
        previous_win = np.nan
        if previous:
            previous_win = float(previous["win"])
            previous_players = previous["players"]
            previous_top4 = previous["top4"]
            previous_quality_now = {
                player: roster_quality.get(player, latest_post_quality.get((team, player), previous["quality"].get(player, .35)))
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
        team_rows.append({
            "gamePk": int(game_pk), "side": side, "team": team,
            "sortTime": first["sortTime"], "win": int(first["homeWin"] if side == "home" else 1 - first["homeWin"]),
            "previousWin": previous_win, **signals,
        })
        previous_by_team[team] = {
            "players": current_players, "top4": current_top4, "quality": current_quality,
            "win": int(first["homeWin"] if side == "home" else 1 - first["homeWin"]),
        }
        for row in group[group["player_id"].notna()].itertuples():
            latest_post_quality[(team, int(row.player_id))] = float(row.post_quality)

    team_frame = pd.DataFrame(team_rows)
    home = team_frame[team_frame["side"].eq("home")][["gamePk", *SIGNALS]].rename(columns={name: f"home_{name}" for name in SIGNALS})
    away = team_frame[team_frame["side"].eq("away")][["gamePk", *SIGNALS]].rename(columns={name: f"away_{name}" for name in SIGNALS})
    joined = home.merge(away, on="gamePk", how="outer")
    for name in SIGNALS:
        joined[f"edge_{name}"] = joined[f"home_{name}"] - joined[f"away_{name}"]
    return feature_rows.merge(joined, on="gamePk", how="left"), team_frame


def evaluate(frame: pd.DataFrame, features: list[str]):
    frame = frame.copy()
    frame["date_dt"] = pd.to_datetime(frame["date"])
    periods = sorted(frame.loc[frame["date_dt"].ge(pd.Timestamp("2024-03-01")), "date_dt"].dt.to_period("M").unique())
    records = []
    for period in periods:
        train = frame[frame["date_dt"].lt(period.start_time)]
        test = frame[frame["date_dt"].between(period.start_time, period.end_time.normalize())]
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=.55, max_iter=2500, random_state=42))
        model.fit(train[features], train["homeWin"])
        probability = model.predict_proba(test[features])[:, 1]
        records.extend((date, int(actual), float(prob), str(period)) for date, actual, prob in zip(test["date"], test["homeWin"], probability))
    out = pd.DataFrame(records, columns=["date", "actual", "probability", "fold"])
    out["correct"] = (out["probability"].ge(.5).astype(int) == out["actual"]).astype(int)
    out["season"] = out["date"].str[:4]
    return out


def main():
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
    frame = bd.attach_market_schedule_features(frame)
    frame, _ = bd.attach_coach_rotation_features(frame, str(frame["date"].max()))
    frame, team_frame = build_quality_features(frame)
    base_features = bd.features_for_mask(bd.DEFAULT_FACTOR_MASK)
    groups = {
        "quality_delta": ["edge_qualityDelta"],
        "replacement_delta": ["edge_replacementDelta"],
        "retention_top4": ["edge_retentionQuality", "edge_top4QualityDelta"],
        "after_loss_quality": ["edge_lossQualityDelta", "edge_lossReplacementDelta", "edge_lossRetentionQuality", "edge_lossTop4QualityDelta"],
        "all_rotation_quality": [f"edge_{name}" for name in SIGNALS],
        "coach_existing": bd.COACH_ROTATION_TEST_FEATURES,
        "coach_plus_all_rotation_quality": bd.COACH_ROTATION_TEST_FEATURES + [f"edge_{name}" for name in SIGNALS],
    }
    baseline = evaluate(frame, base_features)
    rng = np.random.default_rng(20260824)
    results = []
    for name, extra in groups.items():
        candidate = evaluate(frame, base_features + extra)
        delta = candidate["correct"].to_numpy() - baseline["correct"].to_numpy()
        boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() * 100 for _ in range(10000)])
        folds = candidate.groupby("fold")["correct"].sum() - baseline.groupby("fold")["correct"].sum()
        seasons = {}
        for season in sorted(candidate["season"].unique()):
            c = candidate[candidate["season"].eq(season)]
            b = baseline[baseline["season"].eq(season)]
            seasons[season] = {"accuracy": c["correct"].mean(), "base": b["correct"].mean(), "deltaPoints": (c["correct"].mean() - b["correct"].mean()) * 100}
        h1 = candidate[candidate["date"].between("2026-01-01", "2026-06-30")]
        h1_base = baseline[baseline["date"].between("2026-01-01", "2026-06-30")]
        h2 = candidate[candidate["date"].ge("2026-07-01")]
        h2_base = baseline[baseline["date"].ge("2026-07-01")]
        results.append({
            "name": name, "features": extra, "games": len(candidate), "correct": int(candidate["correct"].sum()),
            "accuracy": candidate["correct"].mean(), "baseCorrect": int(baseline["correct"].sum()), "baseAccuracy": baseline["correct"].mean(),
            "deltaPoints": delta.mean() * 100, "ci95Points": [np.quantile(boot, .025), np.quantile(boot, .975)],
            "foldsBetter": int((folds > 0).sum()), "foldsEqual": int((folds == 0).sum()), "foldsWorse": int((folds < 0).sum()),
            "foldDeltas": {str(key): int(value) for key, value in folds.items()},
            "h1_2026": {"accuracy": h1["correct"].mean(), "deltaPoints": (h1["correct"].mean() - h1_base["correct"].mean()) * 100},
            "h2_2026": {"accuracy": h2["correct"].mean(), "deltaPoints": (h2["correct"].mean() - h2_base["correct"].mean()) * 100},
            "seasons": seasons,
        })
    results.sort(key=lambda item: -item["accuracy"])
    print(json.dumps(bd.clean({"coverage": {"teamGames": len(team_frame), "comparisons": int(team_frame["qualityDelta"].notna().sum())}, "results": results}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
