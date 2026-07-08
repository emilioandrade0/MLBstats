"""Cleaner market features — use ONLY DraftKings + ESPN BET close.moneyLine.

Why: our previous features_market took the median across 20 ESPN "providers".
Most of them are regional licenses of the same book and have only the top-level
moneyLine (which may be opening or stale). DraftKings and ESPN BET have well-
populated close.moneyLine (~95% coverage) so they give the cleanest closing-line
signal we can get from ESPN.

Output: data/processed/features_market.parquet  (overwrites previous version)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


SHARP_BOOKS = {"DraftKings", "ESPN BET"}


def _implied(ml):
    if pd.isna(ml) or ml == 0:
        return np.nan
    return -ml / (-ml + 100) if ml < 0 else 100.0 / (ml + 100)


def build() -> Path:
    df = pd.read_parquet(PROCESSED / "odds_close.parquet")
    df = df[df["provider_name"].isin(SHARP_BOOKS)].copy()

    # Prefer close.moneyLine; fall back to current; then top-level.
    df["home_ml"] = df["home_ml_close"].fillna(df["home_ml_current"]).fillna(df["home_ml_top"])
    df["away_ml"] = df["away_ml_close"].fillna(df["away_ml_current"]).fillna(df["away_ml_top"])
    df = df.dropna(subset=["home_ml", "away_ml"])

    df["p_h"] = df["home_ml"].map(_implied)
    df["p_a"] = df["away_ml"].map(_implied)
    total = df["p_h"] + df["p_a"]
    valid = total.between(1.00, 1.20)  # legitimate overround
    df = df.loc[valid].copy()
    df["p_h_devig"] = df["p_h"] / total[valid]

    # Opening implied prob (undevigged — we care about DIRECTION of movement)
    df["p_h_open"] = df["home_ml_open"].map(_implied)
    df["p_a_open"] = df["away_ml_open"].map(_implied)
    total_open = df["p_h_open"] + df["p_a_open"]
    valid_open = total_open.between(1.00, 1.20)
    df["p_h_open_devig"] = np.where(valid_open, df["p_h_open"] / total_open, np.nan)

    # totals + spreads
    df["total_close_num"] = pd.to_numeric(df["total_close"], errors="coerce")
    df["total_open_num"]  = pd.to_numeric(df["total_open"],  errors="coerce")
    df["spread_close_num"] = pd.to_numeric(df["spread_close"], errors="coerce")
    df["spread_open_num"]  = pd.to_numeric(df["spread_open"],  errors="coerce")

    market = df.groupby("espn_event_id").agg(
        market_p_home=("p_h_devig", "mean"),  # mean across DK + ESPN BET (n=1 or 2)
        market_p_home_open=("p_h_open_devig", "mean"),  # promedio opening
        market_over_under=("total_close_num", "mean"),
        market_over_under_open=("total_open_num", "mean"),
        market_spread=("spread_close_num", "mean"),
        market_spread_open=("spread_open_num", "mean"),
        market_n_providers=("p_h_devig", "count"),
        market_p_home_std=("p_h_devig", "std"),
    ).reset_index()

    # Signal: dinero sharp cuando la linea se mueve entre open y close.
    # market_line_shift_home_pp > 0 = dinero entro al HOME, book subio p_home
    # market_line_shift_home_pp < 0 = dinero al AWAY (sharps corrigen home overvalued)
    market["market_line_shift_home_pp"] = (
        (market["market_p_home"] - market["market_p_home_open"]) * 100
    )
    market["market_line_shift_abs_pp"] = market["market_line_shift_home_pp"].abs()
    market["market_total_shift"]  = market["market_over_under"] - market["market_over_under_open"]
    # NOTA: market_spread_shift removido — spread_close en el dataset es la ODDS
    # de la run line (americanos), no el spread en runs; shift = ruido. Solo
    # dejamos ML shift y total shift que si son limpios.
    # Tampoco emitimos los opens al modelo (redundante con close + shift)
    market = market.drop(columns=[
        "market_p_home_open", "market_over_under_open", "market_spread_open"
    ])
    # std with n=1 is NaN — replace with 0 so the model doesn't treat
    # single-book games as a special cohort.
    market["market_p_home_std"] = market["market_p_home_std"].fillna(0.0)

    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    market = market.merge(matched, on="espn_event_id", how="inner") \
                   .drop_duplicates("game_pk", keep="first")

    out = PROCESSED / "features_market.parquet"
    market.to_parquet(out, index=False)
    print(f"wrote {out} ({len(market):,} games)")
    print(f"  median market_p_home: {market['market_p_home'].median():.3f}")
    print(f"  games with both books: {(market['market_n_providers']==2).sum():,}")
    print(f"  games with only one book: {(market['market_n_providers']==1).sum():,}")
    return out


if __name__ == "__main__":
    build()
