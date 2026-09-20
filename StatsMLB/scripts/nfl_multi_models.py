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
}


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
