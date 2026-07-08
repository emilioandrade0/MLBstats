"""Market bias tile feature — comprime patrones contextuales en un solo numero.

Idea: el signal check reveló que el mercado tiene sesgos sistematicos en
combinaciones especificas de (mes, dow, hora, slate size). Esos sesgos varian
de -15pp a +15pp de residuo (home_won - market_p_home). Meter month/dow/hour
como features crudas fallo antes porque LightGBM con 10 leaves no combina
5 features debiles en algo util.

Este modulo pre-agrega el sesgo por casilla contextual y lo entrega como UNA
feature. LightGBM solo tiene que aprender: "cuando este numero es +2pp, subir
p_home; cuando es -2pp, bajarlo". Trivial de aprender.

Feature emitida (por game_pk):
  market_bias_tile_pp  — residuo esperado (home_won - market_p_home) * 100 en la
                          casilla de este juego, calculado con juegos ESTRICTAMENTE
                          anteriores al game_date del target, con shrinkage
                          bayesiano hacia 0.

Casilla = (month, dow, hour_bin, slate_bin) donde:
  hour_bin  = 'tarde' (<=15h ET), 'anochecer' (16-19), 'noche' (20+)
  slate_bin = 'peq' (<=7), 'med' (8-12), 'gnd' (>12)

Shrinkage: smoothed = (n * tile_mean) / (n + k),  k=100
  → tile con n=100 juegos se cree la mitad; tile con n=500 se cree ~85%

Output: data/processed/features_market_tile.parquet  (game_pk, market_bias_tile_pp)

Run standalone:
  python -m src.features.market_tile
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED

OUT = PROCESSED / "features_market_tile.parquet"
SHRINK_K = 30.0  # menos shrinkage porque tile grueso (month, dow) tiene mas N por casilla


def _bin_hour(h):
    if pd.isna(h):
        return "unk"
    if h <= 15:
        return "tarde"
    if h <= 19:
        return "anochecer"
    return "noche"


def _bin_slate(s):
    if pd.isna(s):
        return "unk"
    if s <= 7:
        return "peq"
    if s <= 12:
        return "med"
    return "gnd"


def build() -> Path:
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "first_pitch_utc",
                 "game_type", "home_score", "away_score"],
    )
    games = games[games["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["first_pitch_utc"] = pd.to_datetime(games["first_pitch_utc"], utc=True, errors="coerce")

    # Necesitamos el market_p_home para calcular residual — viene de features_market
    market = pd.read_parquet(PROCESSED / "features_market.parquet",
                              columns=["game_pk", "market_p_home"])
    df = games.merge(market, on="game_pk", how="left")
    df = df.dropna(subset=["market_p_home", "home_score", "away_score"])
    df["home_won"] = (df["home_score"] > df["away_score"]).astype(int)
    df["residual"] = df["home_won"] - df["market_p_home"]

    # Dimensiones contextuales
    et = df["first_pitch_utc"].dt.tz_convert("America/New_York")
    df["month"] = et.dt.month
    df["dow"] = et.dt.dayofweek
    df["hour"] = et.dt.hour
    df["et_date"] = et.dt.date

    slate = df.groupby("et_date").size().rename("slate_size").reset_index()
    df = df.merge(slate, on="et_date", how="left")

    # Tile grueso: solo (month, dow). 84 casillas, ~65-90 juegos c/u.
    # Version anterior (mes+dow+hour+slate) daba tiles con n=30-50 y r=0.0016.
    df["tile"] = df["month"].astype(str) + "|" + df["dow"].astype(str)

    # ── Lagged tile stats: para cada juego, computar tile_mean usando solo
    # juegos anteriores en su misma casilla. Sort por fecha, expanding mean.
    df = df.sort_values(["tile", "game_date", "game_pk"]).reset_index(drop=True)
    grp = df.groupby("tile", sort=False)
    df["tile_lagged_sum"] = grp["residual"].transform(lambda s: s.shift(1).expanding().sum())
    df["tile_lagged_n"]   = grp["residual"].transform(lambda s: s.shift(1).expanding().count())
    # Shrinkage: pesar hacia 0 (residuo esperado = 0 si no hay sesgo)
    n = df["tile_lagged_n"].fillna(0.0)
    mean = df["tile_lagged_sum"] / n.replace(0, np.nan)
    df["market_bias_tile_pp"] = (n / (n + SHRINK_K) * mean.fillna(0.0)) * 100.0

    out = df[["game_pk", "market_bias_tile_pp"]].copy()
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out):,} rows)")
    print(f"  distribucion: mean={out['market_bias_tile_pp'].mean():+.3f}  "
          f"std={out['market_bias_tile_pp'].std():.3f}  "
          f"p10/50/90={out['market_bias_tile_pp'].quantile(0.1):+.2f}/"
          f"{out['market_bias_tile_pp'].quantile(0.5):+.2f}/"
          f"{out['market_bias_tile_pp'].quantile(0.9):+.2f}")

    # Signal check: correlacion con home_won
    chk = df.dropna(subset=["market_bias_tile_pp"])
    r = chk["market_bias_tile_pp"].corr(chk["home_won"])
    print(f"\nSignal check: corr(market_bias_tile_pp, home_won) = {r:.4f}")
    print(f"  (r>0.05 = señal explotable, r<0.02 = ruido)")

    # Bucketed home win rate
    chk["bucket"] = pd.cut(chk["market_bias_tile_pp"], [-99, -3, -1, 1, 3, 99],
                            labels=["-3+ pp (fade home)", "-1 a -3", "flat", "+1 a +3", "+3+ (fade away)"])
    print()
    print("Home win rate por cubeta de market_bias_tile_pp (lagged):")
    print(chk.groupby("bucket", observed=False)["home_won"].agg(["mean", "count"]).round(4).to_string())

    return OUT


if __name__ == "__main__":
    build()
