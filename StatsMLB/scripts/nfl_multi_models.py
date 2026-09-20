"""Train multiple NFL winner models via strict walk-forward and evaluate
whether a naive-average consensus beats every individual model on OOS.

Motors:
  M_full   - all features (matches the original walkforward baseline)
  M_market - spread_line, total_line, home_moneyline, away_moneyline
  M_elo    - elo_diff, home_elo_pre, away_elo_pre
  M_epa    - EPA rolling (offense/defense per play, 4/8 game windows)

Each fitted via LightGBM + isotonic calibration exactly like walkforward.py,
with season-level chronological splits (no leakage). The consensus prediction
is the simple average of the four probabilities. It's only persisted if its
OOS accuracy (2024-2025) beats the best individual.

Outputs:
  data/processed/nfl_multi_preds.parquet - per-game predictions per motor
  data/processed/nfl_multi_meta.json     - accuracy report + consensus decision
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV


ROOT = Path(__file__).resolve().parents[2]
NFL_APP = ROOT / "StatsMLB" / "StrikeCast NFL"
PROCESSED = NFL_APP / "data" / "processed"
OUT_PARQUET = PROCESSED / "nfl_multi_preds.parquet"
OUT_META = PROCESSED / "nfl_multi_meta.json"

OOS_SEASONS = [2024, 2025]

COACH_FEATURES = [
    # Diferenciales (home minus away) del estilo del equipo la temporada previa
    "coach_prev_ppg_scored_diff", "coach_prev_ppg_allowed_diff",
    "coach_prev_win_pct_diff", "coach_prev_ats_diff",
    "coach_prev_over_diff", "coach_prev_close_diff", "coach_prev_blowout_diff",
    "coach_prev_home_perf_diff",
    # Season-to-date (misma temporada, hasta el juego antes)
    "coach_std_ppg_scored_diff", "coach_std_ppg_allowed_diff",
    "coach_std_win_pct_diff", "coach_std_games_played_diff",
]

MOTORS: dict[str, list[str]] = {
    "m_full": [
        "elo_diff", "home_elo_pre", "away_elo_pre",
        "epa_off_diff_l4", "epa_def_diff_l4",
        "home_epa_off_per_play_l4", "away_epa_off_per_play_l4",
        "home_epa_def_per_play_l4", "away_epa_def_per_play_l4",
        "home_epa_off_per_play_l8", "away_epa_off_per_play_l8",
        "spread_line", "total_line",
        "home_rest", "away_rest", "rest_diff",
        "is_dome", "temp", "wind",
    ],
    "m_market": ["spread_line", "total_line", "home_moneyline", "away_moneyline"],
    "m_elo": ["elo_diff", "home_elo_pre", "away_elo_pre"],
    "m_epa": [
        "epa_off_diff_l4", "epa_def_diff_l4",
        "home_epa_off_per_play_l4", "away_epa_off_per_play_l4",
        "home_epa_def_per_play_l4", "away_epa_def_per_play_l4",
        "home_epa_off_per_play_l8", "away_epa_off_per_play_l8",
        "home_epa_def_per_play_l8", "away_epa_def_per_play_l8",
    ],
    "m_coach": COACH_FEATURES,
}


def compute_coach_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera features de estilo por equipo/temporada (no leakage: solo usa juegos previos)."""
    from collections import defaultdict
    df = df.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)
    prev_season_stats: dict[str, dict] = defaultdict(dict)
    curr_season_games: dict[str, list] = defaultdict(list)
    curr_season_by_team: dict[str, int] = defaultdict(lambda: -1)  # season tracker

    rows = []
    for _, g in df.iterrows():
        home, away = str(g["home_team"]), str(g["away_team"])
        season = int(g["season"])

        # Roll over: si nueva temporada, congelamos current->previous
        for team in (home, away):
            if curr_season_by_team[team] != season:
                games = curr_season_games[team]
                if games:
                    scored = np.array([x["scored"] for x in games])
                    allowed = np.array([x["allowed"] for x in games])
                    won = np.array([x["won"] for x in games if x["won"] is not None])
                    ats = [x["ats"] for x in games if x["ats"] is not None]
                    over = [x["over"] for x in games if x["over"] is not None]
                    close = np.array([1 if abs(x["scored"] - x["allowed"]) <= 3 else 0 for x in games])
                    blow = np.array([1 if x["won"] and (x["scored"] - x["allowed"]) >= 14 else 0 for x in games])
                    home_games = [x for x in games if x["side"] == "home"]
                    away_games = [x for x in games if x["side"] == "away"]
                    home_won = [x["won"] for x in home_games if x["won"] is not None]
                    away_won = [x["won"] for x in away_games if x["won"] is not None]
                    home_wr = float(np.mean(home_won)) if home_won else 0.5
                    away_wr = float(np.mean(away_won)) if away_won else 0.5
                    prev_season_stats[team] = {
                        "ppg_scored": float(scored.mean()),
                        "ppg_allowed": float(allowed.mean()),
                        "win_pct": float(won.mean()) if len(won) else 0.5,
                        "ats_pct": float(np.mean(ats)) if ats else 0.5,
                        "over_pct": float(np.mean(over)) if over else 0.5,
                        "close_pct": float(close.mean()) if len(close) else 0.0,
                        "blowout_pct": float(blow.mean()) if len(blow) else 0.0,
                        "home_perf_gap": home_wr - away_wr,
                    }
                curr_season_games[team] = []
                curr_season_by_team[team] = season

        def snapshot(team):
            prev = prev_season_stats.get(team, {})
            curr = curr_season_games[team]
            if curr:
                s = np.array([x["scored"] for x in curr])
                a = np.array([x["allowed"] for x in curr])
                w = [x["won"] for x in curr if x["won"] is not None]
                return prev, {
                    "ppg_scored": float(s.mean()), "ppg_allowed": float(a.mean()),
                    "win_pct": float(np.mean(w)) if w else 0.5, "played": float(len(curr)),
                }
            return prev, {"ppg_scored": 22.0, "ppg_allowed": 22.0, "win_pct": 0.5, "played": 0.0}

        hp, hs = snapshot(home)
        ap, as_ = snapshot(away)

        def d(k, hp, ap, default=0.0):
            return (hp.get(k, default) or default) - (ap.get(k, default) or default)

        row = {"game_id": g["game_id"]}
        row["coach_prev_ppg_scored_diff"] = d("ppg_scored", hp, ap, 22.0)
        row["coach_prev_ppg_allowed_diff"] = d("ppg_allowed", hp, ap, 22.0)
        row["coach_prev_win_pct_diff"] = d("win_pct", hp, ap, 0.5)
        row["coach_prev_ats_diff"] = d("ats_pct", hp, ap, 0.5)
        row["coach_prev_over_diff"] = d("over_pct", hp, ap, 0.5)
        row["coach_prev_close_diff"] = d("close_pct", hp, ap, 0.0)
        row["coach_prev_blowout_diff"] = d("blowout_pct", hp, ap, 0.0)
        row["coach_prev_home_perf_diff"] = d("home_perf_gap", hp, ap, 0.0)
        row["coach_std_ppg_scored_diff"] = d("ppg_scored", hs, as_, 22.0)
        row["coach_std_ppg_allowed_diff"] = d("ppg_allowed", hs, as_, 22.0)
        row["coach_std_win_pct_diff"] = d("win_pct", hs, as_, 0.5)
        row["coach_std_games_played_diff"] = hs["played"] - as_["played"]
        rows.append(row)

        # Post: actualizar current season con este juego si es final
        if str(g.get("status")) == "final" and pd.notna(g.get("home_score")) and pd.notna(g.get("away_score")):
            hsc = float(g["home_score"]); asc = float(g["away_score"])
            spread = float(g["spread_line"]) if pd.notna(g.get("spread_line")) else 0.0
            total_line = float(g["total_line"]) if pd.notna(g.get("total_line")) else None
            total_actual = hsc + asc
            over = None if total_line is None else 1 if total_actual > total_line else 0 if total_actual < total_line else None
            home_covered = None if spread == 0 else 1 if (hsc + spread) > asc else 0 if (hsc + spread) < asc else None
            for side, scored, allowed, ats_val in (("home", hsc, asc, home_covered),
                                                   ("away", asc, hsc, None if home_covered is None else 1 - home_covered)):
                team = str(g[f"{side}_team"])
                won = 1 if scored > allowed else 0 if scored < allowed else None
                curr_season_games[team].append({
                    "side": side, "scored": scored, "allowed": allowed,
                    "won": won, "ats": ats_val, "over": over,
                })

    feats = pd.DataFrame(rows)
    return df.merge(feats, on="game_id", how="left")


@dataclass
class Cfg:
    start_season: int = 2015
    end_season: int = 2026
    min_train: int = 500
    n_estimators: int = 400
    lr: float = 0.05
    max_depth: int = 5
    seed: int = 42


def fit_model(X, y, cfg: Cfg):
    base = LGBMClassifier(
        n_estimators=cfg.n_estimators, learning_rate=cfg.lr, max_depth=cfg.max_depth,
        num_leaves=31, min_child_samples=20, random_state=cfg.seed, verbose=-1,
    )
    n_splits = min(5, max(2, len(y) // 200))
    return CalibratedClassifierCV(base, method="isotonic", cv=n_splits).fit(X, y)


def walkforward_all(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    df = df.sort_values("kickoff_utc").reset_index(drop=True)
    all_features = sorted({c for feats in MOTORS.values() for c in feats})
    df[all_features] = df[all_features].apply(pd.to_numeric, errors="coerce")

    predictions: list[pd.DataFrame] = []
    seasons = sorted(df["season"].unique())
    seasons = [s for s in seasons if cfg.start_season <= s <= cfg.end_season]

    for season in seasons:
        past = df[df["season"] < season]
        current = df[df["season"] == season].sort_values(["week", "kickoff_utc"])
        weeks = sorted(current["week"].dropna().unique().tolist())
        train_pool = past.copy()
        print(f"\n-- Season {season}  (train pool: {len(train_pool):,} games)")

        for wk in weeks:
            wk_games = current[current["week"] == wk]
            row = wk_games[["game_id", "season", "week", "status"]].copy()

            y = train_pool["home_win"].dropna()
            if len(y) < cfg.min_train:
                for motor in MOTORS:
                    row[f"{motor}_home_win"] = np.nan
            else:
                for motor, feats in MOTORS.items():
                    X_train = train_pool.loc[y.index, feats].fillna(-999)
                    model = fit_model(X_train, y.astype(int), cfg)
                    X_pred = wk_games[feats].fillna(-999)
                    row[f"{motor}_home_win"] = model.predict_proba(X_pred)[:, 1]

            predictions.append(row)
            finals = wk_games[wk_games["status"] == "final"]
            if not finals.empty:
                train_pool = pd.concat([train_pool, finals], ignore_index=True)
            print(f"  wk {int(wk):>2}: {len(wk_games):>2} games  (pool now {len(train_pool):,})")

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def evaluate(preds: pd.DataFrame, df: pd.DataFrame) -> dict:
    truth = df[["game_id", "season", "home_win"]]
    m = preds.merge(truth, on=["game_id", "season"], how="left", suffixes=("", "_truth"))
    oos = m[(m["season"].isin(OOS_SEASONS)) & m["home_win"].notna()].copy()
    if oos.empty:
        return {"note": "No OOS games available", "individuals": {}, "consensus": None, "consensusUsed": False}

    def brier(p, y): return float(((p - y) ** 2).mean())
    def acc(p, y): return float(((p >= .5).astype(int) == y).mean())

    y_true = oos["home_win"].astype(int).values
    ind_metrics = {}
    for motor in MOTORS:
        col = f"{motor}_home_win"
        p = oos[col].astype(float).values
        ind_metrics[motor] = {
            "n": int(len(p)),
            "accuracy": acc(p, y_true),
            "brier": brier(p, y_true),
        }

    consensus_p = oos[[f"{motor}_home_win" for motor in MOTORS]].mean(axis=1).values
    consensus_acc = acc(consensus_p, y_true)
    consensus_brier = brier(consensus_p, y_true)
    best_individual_acc = max(v["accuracy"] for v in ind_metrics.values())
    consensus_used = consensus_acc > best_individual_acc

    return {
        "oosSeasons": OOS_SEASONS,
        "individuals": ind_metrics,
        "consensus": {
            "accuracy": consensus_acc,
            "brier": consensus_brier,
            "bestIndividual": best_individual_acc,
            "improvement": consensus_acc - best_individual_acc,
        },
        "consensusUsed": bool(consensus_used),
    }


def main() -> None:
    df = pd.read_parquet(PROCESSED / "train.parquet")
    print(f"[nfl_multi_models] computando features de coach/estilo por equipo…")
    df = compute_coach_features(df)
    cfg = Cfg()
    print(f"[nfl_multi_models] {len(df):,} juegos totales, entrenando {len(MOTORS)} motores...")
    preds = walkforward_all(df, cfg)
    print(f"\n[nfl_multi_models] {len(preds):,} predicciones generadas")

    meta = evaluate(preds, df)
    print("\n=== EVALUACION OOS ({} + {}) ===".format(*OOS_SEASONS))
    for motor, m in meta["individuals"].items():
        print(f"  {motor:10s}  n={m['n']:4d}  acc={m['accuracy']:.4f}  brier={m['brier']:.4f}")
    c = meta["consensus"]
    print(f"  {'CONSENSO':10s}         acc={c['accuracy']:.4f}  brier={c['brier']:.4f}  "
          f"delta_vs_best={c['improvement']:+.4f}")
    print(f"  -> {'USADO' if meta['consensusUsed'] else 'DESCARTADO'} "
          f"(regla: consensus > best_individual)")

    preds["consensus_home_win"] = preds[[f"{motor}_home_win" for motor in MOTORS]].mean(axis=1)
    preds["consensus_used"] = meta["consensusUsed"]

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(OUT_PARQUET, index=False)
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n-> {OUT_PARQUET}")
    print(f"-> {OUT_META}")


if __name__ == "__main__":
    main()
