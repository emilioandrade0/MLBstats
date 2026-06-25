"""Error analysis on the walk-forward predictions.

Instead of guessing new features, find the games the model gets WRONG
(especially confidently wrong) and look for systematic patterns.

The output ranks failure modes by:
  - Slice with worst log_loss / accuracy
  - Slice where model is over/under-confident
  - Largest absolute miscalibrations
  - Examples of biggest blown picks

From this we know which features to ADD or which existing features to fix.

Usage:
    python -m src.analysis.errors
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..normalize.paths import PROCESSED


def _load() -> pd.DataFrame:
    preds = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet", columns=[
        "game_pk", "venue_id", "venue_name", "day_night", "weather_temp_f",
        "weather_condition", "weather_wind", "roof_type", "scheduled_innings",
        "season", "game_type", "double_header", "game_number",
        "probable_home_pitcher_id", "probable_away_pitcher_id",
    ])
    df = preds.merge(games, on="game_pk", how="left")
    df["game_date"] = pd.to_datetime(df["game_date_x"] if "game_date_x" in df.columns else df["game_date"])
    df["p_home_c"] = df["p_home"].clip(1e-6, 1 - 1e-6)
    df["y"] = df["home_win"].astype(int)
    df["error"] = (df["y"] - df["p_home_c"]).abs()
    df["log_loss"] = -(df["y"] * np.log(df["p_home_c"]) + (1 - df["y"]) * np.log(1 - df["p_home_c"]))
    df["confidence"] = np.maximum(df["p_home_c"], 1 - df["p_home_c"])
    df["pick_home"] = (df["p_home_c"] >= 0.5).astype(int)
    df["correct"] = (df["pick_home"] == df["y"]).astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    """Where is the model over/under-confident? Bucket by predicted prob."""
    df = df.copy()
    df["bucket"] = pd.cut(df["p_home_c"], np.arange(0, 1.05, 0.05))
    cal = df.groupby("bucket", observed=True).agg(
        n=("y", "size"),
        mean_pred=("p_home_c", "mean"),
        actual_rate=("y", "mean"),
    )
    cal["calibration_gap"] = cal["actual_rate"] - cal["mean_pred"]
    cal["sign"] = np.where(cal["calibration_gap"] > 0.02, "model UNDERconfident",
                  np.where(cal["calibration_gap"] < -0.02, "model OVERconfident", "calibrated"))
    return cal.round(3)


# ──────────────────────────────────────────────────────────────────────────────
def _slice_table(df: pd.DataFrame, by: str, min_n: int = 50) -> pd.DataFrame:
    g = df.groupby(by).agg(
        n=("y", "size"),
        log_loss=("log_loss", "mean"),
        accuracy=("correct", "mean"),
        mean_pred=("p_home_c", "mean"),
        actual_rate=("y", "mean"),
    )
    g = g[g["n"] >= min_n].copy()
    g["calibration_gap"] = g["actual_rate"] - g["mean_pred"]
    return g.sort_values("log_loss", ascending=False).round(3)


def worst_venues(df: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    return _slice_table(df, "venue_name", min_n=60).head(top)


def best_venues(df: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    g = _slice_table(df, "venue_name", min_n=60)
    return g.sort_values("log_loss").head(top)


def by_day_night(df: pd.DataFrame) -> pd.DataFrame:
    return _slice_table(df, "day_night", min_n=100)


def by_dow(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["game_date"].dt.day_name()
    return _slice_table(df, "dow", min_n=100)


def by_temp_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_bkt"] = pd.cut(df["weather_temp_f"], [0, 50, 65, 80, 95, 130],
                              labels=["<50°", "50-65°", "65-80°", "80-95°", "95°+"])
    return _slice_table(df, "temp_bkt", min_n=80)


def by_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["game_date"].dt.month
    return _slice_table(df, "month", min_n=100)


def by_doubleheader(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dh_flag"] = (df["double_header"].fillna("N").astype(str) != "N").astype(int)
    df["dh_label"] = df["dh_flag"].map({0: "single game", 1: "double-header"})
    return _slice_table(df, "dh_label", min_n=20)


def by_pick_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["conf_bkt"] = pd.cut(df["confidence"],
                              [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01],
                              labels=["50-55%", "55-60%", "60-65%", "65-70%", "70-80%", "80%+"])
    return _slice_table(df, "conf_bkt", min_n=50)


def biggest_blown_picks(df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """Games where the model was confidently wrong."""
    g = df[df["correct"] == 0].sort_values("confidence", ascending=False).head(top)
    return g[["game_date", "away_team_abbrev", "home_team_abbrev",
              "away_score", "home_score", "p_home", "market_p_home",
              "venue_name", "day_night", "weather_temp_f"]].copy().round(3)


def by_market_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    """When model and market DISAGREE, who is right more often?"""
    df = df.copy()
    df["model_pick"] = (df["p_home_c"] > 0.5).astype(int)
    df["market_pick"] = (df["market_p_home"] > 0.5).astype(int)
    df = df.dropna(subset=["market_p_home"])
    df["disagreement"] = (df["model_pick"] != df["market_pick"]).astype(int)
    agree = df[df["disagreement"] == 0]
    disagree = df[df["disagreement"] == 1]
    out = pd.DataFrame({
        "n": [len(agree), len(disagree)],
        "model_acc": [agree["correct"].mean(), disagree["correct"].mean()],
        "market_acc": [
            (agree["market_pick"] == agree["y"]).mean(),
            (disagree["market_pick"] == disagree["y"]).mean(),
        ],
    }, index=["agree", "disagree"]).round(3)
    out["model_minus_market"] = (out["model_acc"] - out["market_acc"]).round(3)
    return out


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    df = _load()
    print(f"loaded {len(df):,} walk-forward predictions, base rate = {df['y'].mean():.3f}")
    print(f"overall log_loss = {df['log_loss'].mean():.4f}, accuracy = {df['correct'].mean():.4f}")
    print()

    print("=" * 78)
    print("1. CALIBRATION — donde se equivoca por sobre/sub-confianza")
    print("=" * 78)
    print(calibration_table(df).to_string())

    print("\n" + "=" * 78)
    print("2. PEORES VENUES — donde mas pifia el modelo")
    print("=" * 78)
    print(worst_venues(df).to_string())

    print("\n" + "=" * 78)
    print("3. MEJORES VENUES — donde mejor predice")
    print("=" * 78)
    print(best_venues(df).to_string())

    print("\n" + "=" * 78)
    print("4. DIA vs NOCHE")
    print("=" * 78)
    print(by_day_night(df).to_string())

    print("\n" + "=" * 78)
    print("5. POR DIA DE LA SEMANA")
    print("=" * 78)
    print(by_dow(df).to_string())

    print("\n" + "=" * 78)
    print("6. POR TEMPERATURA")
    print("=" * 78)
    print(by_temp_bucket(df).to_string())

    print("\n" + "=" * 78)
    print("7. POR MES DE LA TEMPORADA")
    print("=" * 78)
    print(by_month(df).to_string())

    print("\n" + "=" * 78)
    print("8. DOBLE JORNADA vs SOLO")
    print("=" * 78)
    print(by_doubleheader(df).to_string())

    print("\n" + "=" * 78)
    print("9. CONFIANZA DEL PICK — calibracion por bucket")
    print("=" * 78)
    print(by_pick_confidence(df).to_string())

    print("\n" + "=" * 78)
    print("10. CUANDO MODELO Y MERCADO ESTAN EN DESACUERDO, QUIEN GANA?")
    print("=" * 78)
    print(by_market_disagreement(df).to_string())

    print("\n" + "=" * 78)
    print("11. PEORES BLOWN PICKS — modelo confiadamente equivocado")
    print("=" * 78)
    print(biggest_blown_picks(df).to_string())


if __name__ == "__main__":
    main()
