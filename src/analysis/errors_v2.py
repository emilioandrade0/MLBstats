"""Re-run error analysis with the NEW production logic.

After C (disagree-aware blend) + D (park interactions) + shrink, the errors
have a different shape. We need to find what's still systematically wrong
so we can target the NEXT layer of improvements.

Compares OLD blend (per-fold tuned, no shrink) vs NEW blend (disagree-aware,
shrink, park interactions) on the same walk-forward set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


W_AGREE = 0.50
W_DISAGREE = 0.80
SHRINK = 0.5
ANCHOR = 0.55


def shrink_arr(p: np.ndarray) -> np.ndarray:
    out = p.copy()
    high = p > ANCHOR; low = p < (1 - ANCHOR)
    out[high] = ANCHOR + (p[high] - ANCHOR) * SHRINK
    out[low] = (1 - ANCHOR) - ((1 - ANCHOR) - p[low]) * SHRINK
    return np.clip(out, 1e-6, 1 - 1e-6)


def production_blend(model_raw: np.ndarray, market: np.ndarray) -> np.ndarray:
    agree = (model_raw > 0.5) == (market > 0.5)
    w = np.where(agree, W_AGREE, W_DISAGREE)
    blended = w * market + (1 - w) * model_raw
    return shrink_arr(blended)


def load_with_new_preds() -> pd.DataFrame:
    preds = pd.read_parquet(PROCESSED / "walkforward_preds.parquet")
    preds = preds.dropna(subset=["market_p_home", "p_home_model_raw"]).copy()
    games = pd.read_parquet(PROCESSED / "games.parquet", columns=[
        "game_pk", "venue_name", "day_night", "weather_temp_f",
        "weather_condition", "scheduled_innings", "double_header",
    ])
    df = preds.merge(games, on="game_pk", how="left")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["y"] = df["home_win"].astype(int)
    # NEW production prediction
    df["p_new"] = production_blend(
        df["p_home_model_raw"].clip(1e-6, 1 - 1e-6).values,
        df["market_p_home"].clip(1e-6, 1 - 1e-6).values,
    )
    # Old prediction (from saved walkforward run)
    df["p_old"] = df["p_home"].clip(1e-6, 1 - 1e-6)
    df["pick_new"] = (df["p_new"] >= 0.5).astype(int)
    df["pick_old"] = (df["p_old"] >= 0.5).astype(int)
    df["correct_new"] = (df["pick_new"] == df["y"]).astype(int)
    df["correct_old"] = (df["pick_old"] == df["y"]).astype(int)
    df["confidence_new"] = np.maximum(df["p_new"], 1 - df["p_new"])
    df["log_loss_new"] = -(df["y"] * np.log(df["p_new"]) +
                            (1 - df["y"]) * np.log(1 - df["p_new"]))
    return df


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = pd.cut(df["p_new"], np.arange(0, 1.05, 0.05))
    cal = df.groupby("bucket", observed=True).agg(
        n=("y", "size"),
        mean_pred=("p_new", "mean"),
        actual_rate=("y", "mean"),
    )
    cal["gap"] = cal["actual_rate"] - cal["mean_pred"]
    return cal[cal["n"] >= 15].round(3)


def _slice(df: pd.DataFrame, by: str, min_n: int = 50) -> pd.DataFrame:
    g = df.groupby(by).agg(
        n=("y", "size"),
        ll=("log_loss_new", "mean"),
        acc_new=("correct_new", "mean"),
        acc_old=("correct_old", "mean"),
        mean_pred=("p_new", "mean"),
        actual=("y", "mean"),
    )
    g = g[g["n"] >= min_n].copy()
    g["delta_acc"] = g["acc_new"] - g["acc_old"]
    g["gap"] = g["actual"] - g["mean_pred"]
    return g.sort_values("ll", ascending=False).round(3)


def venues_now(df, top=8):
    return _slice(df, "venue_name", 60).head(top)


def venues_best(df, top=6):
    return _slice(df, "venue_name", 60).sort_values("ll").head(top)


def by_month_now(df):
    df = df.copy()
    df["month"] = df["game_date"].dt.month
    return _slice(df, "month", 100)


def by_dow_now(df):
    df = df.copy()
    df["dow"] = df["game_date"].dt.day_name()
    return _slice(df, "dow", 100)


def by_confidence_now(df):
    df = df.copy()
    df["conf_bkt"] = pd.cut(df["confidence_new"],
                              [0.5, 0.52, 0.55, 0.58, 0.62, 0.70, 1.01],
                              labels=["50-52%", "52-55%", "55-58%", "58-62%", "62-70%", "70%+"])
    return _slice(df, "conf_bkt", 30)


def disagree_subset(df: pd.DataFrame) -> pd.DataFrame:
    """When model and market disagree, who's right now (after C blend)?"""
    df = df.copy()
    df["disagree"] = ((df["p_home_model_raw"] > 0.5) != (df["market_p_home"] > 0.5)).astype(int)
    market_pick = (df["market_p_home"] > 0.5).astype(int)
    df["market_correct"] = (market_pick == df["y"]).astype(int)
    g = df.groupby("disagree").agg(
        n=("y", "size"),
        model_new_acc=("correct_new", "mean"),
        model_old_acc=("correct_old", "mean"),
        market_acc=("market_correct", "mean"),
    ).round(3)
    g["new_minus_market"] = g["model_new_acc"] - g["market_acc"]
    g["new_minus_old"] = g["model_new_acc"] - g["model_old_acc"]
    return g


def biggest_blown_picks_now(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    g = df[df["correct_new"] == 0].sort_values("confidence_new", ascending=False).head(top)
    return g[["game_date", "away_team_abbrev", "home_team_abbrev",
              "away_score", "home_score", "p_new", "market_p_home",
              "venue_name"]].round(3)


def picks_only_new_got_right(df: pd.DataFrame) -> pd.DataFrame:
    """Games where the NEW model was correct but the OLD model was wrong."""
    g = df[(df["correct_new"] == 1) & (df["correct_old"] == 0)]
    print(f"  NEW correct but OLD wrong: {len(g):,}")
    return g[["game_date", "away_team_abbrev", "home_team_abbrev",
              "away_score", "home_score", "p_new", "p_old", "market_p_home"]].round(3).head(15)


def picks_only_old_got_right(df: pd.DataFrame) -> pd.DataFrame:
    """Games where the OLD model was correct but the NEW model is now wrong."""
    g = df[(df["correct_old"] == 1) & (df["correct_new"] == 0)]
    print(f"  NEW wrong but OLD right: {len(g):,}")
    return g[["game_date", "away_team_abbrev", "home_team_abbrev",
              "away_score", "home_score", "p_new", "p_old", "market_p_home"]].round(3).head(15)


def main() -> None:
    df = load_with_new_preds()
    print(f"loaded {len(df):,} predictions")
    print(f"OLD accuracy:  {df['correct_old'].mean():.4f}")
    print(f"NEW accuracy:  {df['correct_new'].mean():.4f}")
    print(f"NEW log_loss:  {df['log_loss_new'].mean():.4f}")
    print()

    print("=" * 78)
    print("1. CALIBRACION AHORA — donde aun hay over/under-confianza")
    print("=" * 78)
    print(calibration(df).to_string())

    print("\n" + "=" * 78)
    print("2. PEORES VENUES (con NEW model)")
    print("=" * 78)
    print(venues_now(df).to_string())

    print("\n" + "=" * 78)
    print("3. MEJORES VENUES (con NEW model)")
    print("=" * 78)
    print(venues_best(df).to_string())

    print("\n" + "=" * 78)
    print("4. POR MES (sigue agosto/septiembre malo?)")
    print("=" * 78)
    print(by_month_now(df).to_string())

    print("\n" + "=" * 78)
    print("5. POR DIA DE LA SEMANA")
    print("=" * 78)
    print(by_dow_now(df).to_string())

    print("\n" + "=" * 78)
    print("6. POR NIVEL DE CONFIANZA")
    print("=" * 78)
    print(by_confidence_now(df).to_string())

    print("\n" + "=" * 78)
    print("7. CUANDO MODELO Y MERCADO ESTAN EN DESACUERDO (DESPUES DE C)")
    print("=" * 78)
    print(disagree_subset(df).to_string())

    print("\n" + "=" * 78)
    print("8. BIGGEST BLOWN PICKS — donde NEW model erro confiadamente")
    print("=" * 78)
    print(biggest_blown_picks_now(df).to_string())

    print("\n" + "=" * 78)
    print("9. PICKS QUE EL NEW SALVO vs PICKS QUE EL NEW PERDIO")
    print("=" * 78)
    n_only_new = ((df["correct_new"] == 1) & (df["correct_old"] == 0)).sum()
    n_only_old = ((df["correct_old"] == 1) & (df["correct_new"] == 0)).sum()
    print(f"  NEW arregla:  {n_only_new}")
    print(f"  NEW rompe:    {n_only_old}")
    print(f"  Neto:         {n_only_new - n_only_old} games more correct")


if __name__ == "__main__":
    main()
