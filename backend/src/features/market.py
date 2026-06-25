"""Convert ESPN odds → devigged market probabilities per game.

For each game we have multiple providers (DraftKings, ESPN BET, Caesars, etc.).
Each provider's moneyline implies a probability with a vig (overround). We:

  1. Convert american moneyline → raw implied probability for each side.
  2. De-vig by proportional method: p_devigged = p_raw / (p_home_raw + p_away_raw).
     (Simple but well-behaved; equivalent to Shin's method when vig is small.)
  3. Take the median across providers (robust to a single book mispricing).

Output: data/processed/features_market.parquet
Key: game_pk
Columns: market_p_home, market_over_under, market_spread, market_n_providers
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED


def _implied_prob_from_american(ml: float) -> float:
    if ml is None or not np.isfinite(ml) or ml == 0:
        return np.nan
    if ml < 0:
        return -ml / (-ml + 100)
    return 100.0 / (ml + 100)


def build() -> Path:
    odds = pd.read_parquet(PROCESSED / "odds.parquet")
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")

    # 1. raw implied per side
    odds["p_home_raw"] = odds["home_moneyline"].astype(float).map(_implied_prob_from_american)
    odds["p_away_raw"] = odds["away_moneyline"].astype(float).map(_implied_prob_from_american)
    odds = odds.dropna(subset=["p_home_raw", "p_away_raw"])
    # 2. de-vig (proportional)
    total = odds["p_home_raw"] + odds["p_away_raw"]
    # Sanity filter: legitimate two-way books overround ~1.02–1.10.
    valid = total.between(1.00, 1.20)
    odds = odds.loc[valid].copy()
    odds["p_home_devig"] = odds["p_home_raw"] / total[valid]

    # Numeric over/under and spread — both come in as strings sometimes.
    odds["over_under_num"] = pd.to_numeric(odds["over_under"], errors="coerce")
    odds["spread_num"] = pd.to_numeric(odds["spread"], errors="coerce")

    # 3. median across providers per ESPN event
    market = odds.groupby("espn_event_id").agg(
        market_p_home=("p_home_devig", "median"),
        market_over_under=("over_under_num", "median"),
        market_spread=("spread_num", "median"),
        market_n_providers=("p_home_devig", "count"),
        market_p_home_std=("p_home_devig", "std"),
    ).reset_index()

    # Map ESPN event_id → MLB game_pk via xref (matched only)
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    market = market.merge(matched, on="espn_event_id", how="inner")

    # Drop duplicate game_pk (doubleheader collisions on same key) — keep first.
    market = market.drop_duplicates("game_pk", keep="first")

    out = PROCESSED / "features_market.parquet"
    market.to_parquet(out, index=False)
    print(f"wrote {out} ({len(market):,} games with market data)")
    print(f"  median market_p_home: {market['market_p_home'].median():.3f}")
    print(f"  median n_providers:   {int(market['market_n_providers'].median())}")
    return out


if __name__ == "__main__":
    build()
