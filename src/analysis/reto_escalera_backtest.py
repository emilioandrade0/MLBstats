"""Leakage-safe selection sweep for Reto Escalera.

Consumes monthly walk-forward predictions, chooses at most one real moneyline
pick per day, tunes selection rules on completed games through 2025, and
reports 2026 as an untouched temporal holdout.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..model.backtest import (
    _american_to_decimal,
    _implied_prob_american,
)
from ..normalize.paths import PROCESSED


OUT = PROCESSED / "reto_escalera_sweep"
DEV_END = pd.Timestamp("2025-12-31")
HOLDOUT_START = pd.Timestamp("2026-01-01")
PRED_COL = "p_home_model_raw"  # excludes globally researched post-process rules


@dataclass(frozen=True)
class Config:
    name: str
    odds_min: float
    odds_max: float
    side: str
    rank: str
    min_prob: float = 0.0
    require_agree: bool = False
    soft_quality: bool = False


def _configs() -> list[Config]:
    out = [
        Config("current_like", 1.80, 2.20, "model", "confidence", .55, False, True),
        Config("current_like_agree", 1.80, 2.20, "model", "confidence", .55, True, True),
    ]
    for lo, hi in [
        (1.60, 2.40), (1.70, 2.30), (1.70, 2.10),
        (1.75, 2.15), (1.80, 2.20), (1.80, 2.10), (1.85, 2.15),
    ]:
        for side in ("model", "edge"):
            for rank in ("confidence", "edge", "ev"):
                for min_prob in (0.0, .52, .55, .58, .60, .62, .65):
                    for agree in (False, True):
                        name = f"o{lo:.2f}-{hi:.2f}_{side}_{rank}_p{min_prob:.2f}_a{int(agree)}"
                        out.append(Config(name, lo, hi, side, rank, min_prob, agree))
    return out


def _universe() -> pd.DataFrame:
    pred = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    pred["game_date"] = pd.to_datetime(pred["game_date"])
    pred = pred.dropna(subset=["home_win", PRED_COL, "market_p_home"]).copy()
    # Mirror the endpoint's actual line source: median closing/current/top ML
    # from its two designated sharp books. The older generic backtest helper
    # has sparse 2026 coverage and is not the production Reto universe.
    odds = pd.read_parquet(PROCESSED / "odds_close.parquet")
    odds = odds[odds["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    odds["home_ml"] = odds["home_ml_close"].fillna(odds["home_ml_current"]).fillna(odds["home_ml_top"])
    odds["away_ml"] = odds["away_ml_close"].fillna(odds["away_ml_current"]).fillna(odds["away_ml_top"])
    odds = odds.dropna(subset=["home_ml", "away_ml"])
    lines = odds.groupby("espn_event_id").agg(
        home_ml=("home_ml", "median"), away_ml=("away_ml", "median")
    ).reset_index()
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    lines = lines.merge(matched, on="espn_event_id", how="inner").drop_duplicates("game_pk")
    d = pred.merge(lines, on="game_pk", how="inner", validate="one_to_one")
    d["dec_h"] = d["home_ml"].map(_american_to_decimal)
    d["dec_a"] = d["away_ml"].map(_american_to_decimal)
    raw_h = d["home_ml"].map(_implied_prob_american)
    raw_a = d["away_ml"].map(_implied_prob_american)
    vig = raw_h + raw_a
    d["mkt_h"] = raw_h / vig
    d["mkt_a"] = raw_a / vig
    return d.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def _daily_picks(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = d.copy()
    if cfg.side == "edge":
        choose_h = (x[PRED_COL] - x["mkt_h"]) >= ((1 - x[PRED_COL]) - x["mkt_a"])
    else:
        choose_h = x[PRED_COL] >= .5
    x["pick_prob"] = np.where(choose_h, x[PRED_COL], 1 - x[PRED_COL])
    x["pick_home"] = choose_h
    x["decimal"] = np.where(choose_h, x["dec_h"], x["dec_a"])
    x["mkt_prob"] = np.where(choose_h, x["mkt_h"], x["mkt_a"])
    x["won"] = np.where(choose_h, x["home_win"] == 1, x["home_win"] == 0)
    x["agree"] = choose_h == (x["market_p_home"] >= .5)
    x["edge"] = x["pick_prob"] - x["mkt_prob"]
    x["ev"] = x["pick_prob"] * x["decimal"] - 1
    eligible = x["decimal"].between(cfg.odds_min, cfg.odds_max)
    if not cfg.soft_quality:
        eligible &= x["pick_prob"] >= cfg.min_prob
        if cfg.require_agree:
            eligible &= x["agree"]
    x = x[eligible].copy()
    if x.empty:
        return x
    score_col = {"confidence": "pick_prob", "edge": "edge", "ev": "ev"}[cfg.rank]
    x["rank_score"] = x[score_col]
    if cfg.soft_quality:
        x.loc[x["pick_prob"] < cfg.min_prob, "rank_score"] *= .5
        if cfg.require_agree:
            x.loc[~x["agree"], "rank_score"] *= .5
    # Stable, pre-declared tiebreak: score, probability, then game id.
    x = x.sort_values(["game_date", "rank_score", "pick_prob", "game_pk"],
                      ascending=[True, False, False, True])
    return x.drop_duplicates("game_date", keep="first")


def _ladder_counts(wins: np.ndarray, steps: int) -> tuple[int, int]:
    started = completed = run = 0
    for won in wins.astype(bool):
        if run == 0:
            started += 1
        if won:
            run += 1
            if run == steps:
                completed += 1
                run = 0
        else:
            run = 0
    return started, completed


def _metrics(picks: pd.DataFrame, period: str, cfg: Config) -> dict:
    if picks.empty:
        return {"config": cfg.name, "period": period, "n_days": 0}
    won = picks["won"].astype(bool).to_numpy()
    profit = np.where(won, picks["decimal"].to_numpy() - 1, -1.0)
    wr = float(won.mean())
    # Conservative tuning target: Wilson lower bound rewards repeatability,
    # while a small ROI term breaks ties without chasing one lucky ladder.
    n = len(won)
    z = 1.96
    center = (wr + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(wr*(1-wr)/n + z*z/(4*n*n)) / (1 + z*z/n)
    row = {
        "config": cfg.name, "period": period, "n_days": n,
        "win_rate": wr, "flat_roi": float(profit.mean()),
        "avg_odds": float(picks["decimal"].mean()),
        "avg_edge": float(picks["edge"].mean()),
        "wilson_low": float(center-half),
        "selection_score": float(center-half + .10*profit.mean()),
    }
    for steps in (3, 5, 10):
        starts, wins_ = _ladder_counts(won, steps)
        row[f"l{steps}_started"] = starts
        row[f"l{steps}_won"] = wins_
        row[f"l{steps}_rate"] = wins_ / starts if starts else 0.0
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    universe = _universe()
    dev = universe[universe["game_date"] <= DEV_END]
    hold = universe[universe["game_date"] >= HOLDOUT_START]
    rows: list[dict] = []
    picks_by_name: dict[str, pd.DataFrame] = {}
    configs = _configs()
    for cfg in configs:
        dp = _daily_picks(dev, cfg)
        hp = _daily_picks(hold, cfg)
        rows.append(_metrics(dp, "dev_through_2025", cfg))
        rows.append(_metrics(dp[dp["game_date"].dt.year == 2024], "dev_2024", cfg))
        rows.append(_metrics(dp[dp["game_date"].dt.year == 2025], "dev_2025", cfg))
        rows.append(_metrics(hp, "holdout_2026", cfg))
        picks_by_name[cfg.name] = hp
    result = pd.DataFrame(rows)
    dev_scores = result[(result.period == "dev_through_2025") & (result.n_days >= 100)]
    winner = dev_scores.sort_values(["selection_score", "n_days"], ascending=False).iloc[0]
    best_name = str(winner.config)
    yearly = result[result.period.isin(["dev_2024", "dev_2025"]) & (result.n_days >= 50)]
    robust = yearly.groupby("config").agg(
        worst_year_score=("selection_score", "min"),
        worst_year_wr=("win_rate", "min"),
        total_days=("n_days", "sum"),
    ).reset_index().sort_values(["worst_year_score", "total_days"], ascending=False)
    robust_name = str(robust.iloc[0].config)
    baseline_names = ["current_like", "current_like_agree", best_name, robust_name]
    comparison = result[result.config.isin(baseline_names)].copy()
    result.sort_values(["period", "selection_score"], ascending=[True, False]).to_csv(
        OUT / "all_configs.csv", index=False)
    picks_by_name[best_name].to_csv(OUT / "best_holdout_picks.csv", index=False)
    summary = {
        "protocol": {
            "prediction_source": "monthly expanding walk-forward raw model predictions",
            "post_process_rules": "excluded to avoid research leakage from globally tuned adjustments",
            "development_end": DEV_END.date().isoformat(),
            "holdout_start": HOLDOUT_START.date().isoformat(),
            "winner_chosen_only_on": "dev_through_2025",
            "minimum_development_pick_days": 100,
        },
        "best_config": asdict(next(c for c in configs if c.name == best_name)),
        "robust_config": asdict(next(c for c in configs if c.name == robust_name)),
        "comparison": comparison.replace({np.nan: None}).to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
