"""Lineup features — what the market doesn't have at opening line.

For each game's actual starting lineup, look up each batter's rolling Statcast
splits vs the opposing starter's hand (L or R), then aggregate.

Pipeline:
  1. From pitches.parquet → per-(batter, p_throws, game) PA stats
  2. Per (batter, hand) rolling 30-game stats, lagged
  3. Identify each game's home/away lineup from player_box.parquet
     (starters = batting_order set, appeared_batting=True)
  4. Identify each game's home/away starting pitcher's hand
  5. For each lineup batter, merge in their rolling-vs-opposing-hand stats
  6. Aggregate to per-(game_pk, side) lineup features

Output: data/processed/features_lineup.parquet
Key: game_pk, side
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..normalize.paths import PROCESSED


def _per_game_batter_hand(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, batter, p_throws) — PA-end aggregates."""
    p = pitches[pitches["woba_denom"].fillna(0) > 0].copy()
    p["is_k"] = p["events"].fillna("").str.contains("strikeout", case=False).astype(int)
    p["is_bb"] = p["events"].fillna("").isin(["walk", "intent_walk"]).astype(int)
    p["is_barrel"] = (p["launch_speed_angle"].fillna(0) == 6).astype(int)
    p["is_batted"] = p["launch_speed"].notna()
    out = p.groupby(["batter", "p_throws", "game_pk", "game_date"], dropna=False).agg(
        pa=("woba_denom", "sum"),
        xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        xwoba_n=("estimated_woba_using_speedangle", "count"),
        k=("is_k", "sum"),
        bb=("is_bb", "sum"),
        barrels=("is_barrel", "sum"),
        batted=("is_batted", "sum"),
    ).reset_index()
    return out


def _add_rolling(df: pd.DataFrame, window: int = 30, min_periods: int = 5) -> pd.DataFrame:
    """Per (batter, p_throws), rolling sums of last `window` PA-games, lagged by 1."""
    df = df.sort_values(["batter", "p_throws", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    g = df.groupby(["batter", "p_throws"], sort=False)
    cols = ["pa", "xwoba_sum", "xwoba_n", "k", "bb", "barrels", "batted"]
    out = {}
    for c in cols:
        shifted = g[c].shift(1)
        out[f"{c}_rs"] = shifted.groupby([df["batter"], df["p_throws"]]).transform(
            lambda s: s.rolling(window, min_periods=min_periods).sum()
        )
    rolled = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
    # ratios
    rolled["bvh_xwoba"] = rolled["xwoba_sum_rs"] / rolled["xwoba_n_rs"].replace(0, np.nan)
    rolled["bvh_k_pct"] = rolled["k_rs"] / rolled["pa_rs"].replace(0, np.nan)
    rolled["bvh_bb_pct"] = rolled["bb_rs"] / rolled["pa_rs"].replace(0, np.nan)
    rolled["bvh_barrel_rate"] = rolled["barrels_rs"] / rolled["batted_rs"].replace(0, np.nan)
    return rolled[["game_pk", "game_date", "batter", "p_throws", "pa_rs",
                    "bvh_xwoba", "bvh_k_pct", "bvh_bb_pct", "bvh_barrel_rate"]]


def _starter_hand_per_game(pitches: pd.DataFrame) -> pd.DataFrame:
    """The first pitcher to throw for each side per game IS the starter.

    Top of 1st → home team is pitching → home starter
    Bot of 1st → away team is pitching → away starter

    For each (game_pk, inning_topbot), pick the pitcher whose AB has the minimum
    at_bat_number on that side. That's the starter.
    """
    s = pitches.copy()
    s["side_starter"] = np.where(
        s["inning_topbot"].astype(str).str.startswith("T"), "home", "away"
    )
    # First AB per (game_pk, side) is the starter's debut
    idx = s.groupby(["game_pk", "side_starter"])["at_bat_number"].idxmin()
    out = s.loc[idx, ["game_pk", "side_starter", "pitcher", "p_throws"]]
    return out.rename(columns={"pitcher": "starter_id", "p_throws": "starter_hand"})


def _lineups(player_box: pd.DataFrame) -> pd.DataFrame:
    """Starting lineup per (game_pk, side). Returns one row per batter in the lineup.

    MLB Stats API encodes batting_order as 3-digit strings '100', '200', ... '900'
    for the starting 9 (multiples of 100). Pinch hitters get '101', '202', etc.
    We filter to starters only (ends in '00').
    """
    pb = player_box.copy()
    pb = pb.dropna(subset=["batting_order", "player_id"])
    pb["order_str"] = pb["batting_order"].astype(str)
    is_starter = pb["order_str"].str.endswith("00")
    starters = pb[is_starter].copy()
    starters["order_spot"] = pd.to_numeric(starters["order_str"].str[:1], errors="coerce")
    starters = starters.dropna(subset=["order_spot"])
    starters["order_spot"] = starters["order_spot"].astype(int)
    return starters[["game_pk", "side", "player_id", "order_spot"]].rename(
        columns={"player_id": "batter"}
    )


def build() -> Path:
    print("[1/5] reading pitches…")
    pitches = pd.read_parquet(
        PROCESSED / "pitches.parquet",
        columns=["game_pk", "game_date", "batter", "pitcher", "p_throws",
                 "inning_topbot", "at_bat_number", "pitch_number",
                 "woba_denom", "estimated_woba_using_speedangle", "events",
                 "launch_speed", "launch_speed_angle"],
    )
    pitches["batter"] = pitches["batter"].astype("Int64")
    pitches["pitcher"] = pitches["pitcher"].astype("Int64")

    print("[2/5] computing per-game batter×hand stats + rolling…")
    pgb = _per_game_batter_hand(pitches)
    bvh = _add_rolling(pgb, window=30, min_periods=5)
    bvh["batter"] = bvh["batter"].astype("Int64")

    print("[3/5] identifying starter hand per game/side…")
    starter = _starter_hand_per_game(pitches)
    starter["starter_id"] = starter["starter_id"].astype("Int64")
    # Pivot: per game, get home_starter_hand and away_starter_hand
    sw = starter.pivot_table(index="game_pk", columns="side_starter",
                              values="starter_hand", aggfunc="first").reset_index()
    sw.columns.name = None
    sw = sw.rename(columns={"home": "home_starter_hand", "away": "away_starter_hand"})

    print("[4/5] reading lineups from player_box…")
    pb = pd.read_parquet(PROCESSED / "player_box.parquet",
                          columns=["game_pk", "side", "player_id", "batting_order"])
    pb["player_id"] = pb["player_id"].astype("Int64")
    lineups = _lineups(pb)

    # Each lineup batter's opposing starter hand
    # home lineup faces away starter; away lineup faces home starter
    lineups = lineups.merge(sw, on="game_pk", how="left")
    lineups["opp_starter_hand"] = np.where(lineups["side"] == "home",
                                            lineups["away_starter_hand"],
                                            lineups["home_starter_hand"])

    # Merge in the batter's rolling stats vs the opposing starter hand
    bvh_lookup = bvh.rename(columns={"p_throws": "opp_starter_hand"})
    merged = lineups.merge(
        bvh_lookup, on=["game_pk", "batter", "opp_starter_hand"], how="left",
    )

    print("[5/5] aggregating to lineup features…")
    # Weight: top-of-order batters get more PAs vs starter (~4 PAs for #1, ~2 for #9).
    # Approximate weights: linear decay from 1.0 → 0.5 across spots 1..9.
    merged["w"] = 1.0 - (merged["order_spot"] - 1) * (0.5 / 8)
    # Weighted means
    def _wavg(g: pd.DataFrame, col: str) -> float:
        x = g[col]
        w = g["w"]
        mask = x.notna() & w.notna()
        if mask.sum() == 0:
            return np.nan
        return float((x[mask] * w[mask]).sum() / w[mask].sum())

    agg = merged.groupby(["game_pk", "side"]).apply(
        lambda g: pd.Series({
            "lineup_xwoba_vs_sp_hand": _wavg(g, "bvh_xwoba"),
            "lineup_k_pct_vs_sp_hand": _wavg(g, "bvh_k_pct"),
            "lineup_bb_pct_vs_sp_hand": _wavg(g, "bvh_bb_pct"),
            "lineup_barrel_vs_sp_hand": _wavg(g, "bvh_barrel_rate"),
            "lineup_top4_xwoba_vs_sp_hand": _wavg(g[g["order_spot"] <= 4], "bvh_xwoba"),
            "lineup_top4_k_pct_vs_sp_hand": _wavg(g[g["order_spot"] <= 4], "bvh_k_pct"),
            "lineup_n_with_data": int(g["bvh_xwoba"].notna().sum()),
        }),
        include_groups=False,
    ).reset_index()

    out = PROCESSED / "features_lineup.parquet"
    agg.to_parquet(out, index=False)
    print(f"\nwrote {out} ({len(agg):,} rows)")
    print("coverage:")
    print(f"  games with both sides: {agg.groupby('game_pk').size().eq(2).sum():,}")
    print(f"  mean batters w/ data:  {agg['lineup_n_with_data'].mean():.1f}/9")
    return out


if __name__ == "__main__":
    build()
