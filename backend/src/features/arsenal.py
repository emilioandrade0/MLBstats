"""Lineup × pitcher arsenal matchup features.

Concept: each batter has a profile of how they hit different pitch types
(FF/SI/SL/CH/CU/FC). Each starter throws their pitches in specific proportions.
A lineup's true matchup score for today is the weighted xwOBA of each batter
against the SPECIFIC arsenal mix the opposing starter will throw.

Pipeline:
  1. Per (game_pk, batter, pitch_type) PA-end xwOBA aggregates
  2. Per (batter, pitch_type) rolling 30-game lagged xwOBA
  3. Per (game_pk, pitcher) pitch-type usage % (arsenal of THIS appearance)
  4. Per pitcher, rolling lagged arsenal (last 5 appearances)
  5. Match each game's lineup vs opposing starter's rolling arsenal
  6. Weighted score: sum(arsenal_pct[pt] * batter_xwoba_vs_pt[pt])
  7. Aggregate to (game_pk, side) — full lineup + top-4 hitters

Output: data/processed/features_arsenal.parquet
Key: (game_pk, side)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED

PITCH_TYPES = ["FF", "SI", "SL", "CH", "CU", "FC"]   # 95% of all MLB pitches
ALL_PT = PITCH_TYPES + ["OTHER"]
BVPT_WINDOW = 30   # rolling games per (batter, pitch_type)
ARSENAL_WINDOW = 5 # rolling appearances per pitcher


def _label_pt(pitches: pd.DataFrame) -> pd.DataFrame:
    pitches = pitches.copy()
    pitches["pt"] = pitches["pitch_type"].where(
        pitches["pitch_type"].isin(PITCH_TYPES), "OTHER"
    )
    return pitches


def _per_game_batter_vs_pt(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, batter, pt) at PA-end: PA, xwOBA sums."""
    pa = pitches[pitches["woba_denom"].fillna(0) > 0].copy()
    out = pa.groupby(["batter", "pt", "game_pk", "game_date"], dropna=False).agg(
        pa=("woba_denom", "sum"),
        xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        xwoba_n=("estimated_woba_using_speedangle", "count"),
    ).reset_index()
    return out


def _rolling_lagged_bvpt(df: pd.DataFrame, window: int = BVPT_WINDOW) -> pd.DataFrame:
    """Per (batter, pt), rolling lagged sum of last `window` games, then ratio."""
    df = df.sort_values(["batter", "pt", "game_date", "game_pk"], kind="mergesort") \
            .reset_index(drop=True)
    g = df.groupby(["batter", "pt"], sort=False)
    out = {}
    for c in ["pa", "xwoba_sum", "xwoba_n"]:
        out[c + "_rs"] = g[c].transform(
            lambda s: s.shift(1).rolling(window, min_periods=3).sum()
        )
    rolled = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
    rolled["bvpt_xwoba"] = rolled["xwoba_sum_rs"] / rolled["xwoba_n_rs"].replace(0, np.nan)
    return rolled[["game_pk", "batter", "pt", "bvpt_xwoba"]]


def _per_game_arsenal(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per (game_pk, pitcher) pitch-type usage % (wide format, cols = PT)."""
    g = pitches.groupby(["pitcher", "game_pk", "game_date", "pt"]).size().reset_index(name="n")
    tot = g.groupby(["pitcher", "game_pk", "game_date"])["n"].sum().rename("total")
    g = g.merge(tot, on=["pitcher", "game_pk", "game_date"])
    g["pct"] = g["n"] / g["total"]
    wide = g.pivot_table(
        index=["pitcher", "game_pk", "game_date"],
        columns="pt", values="pct", fill_value=0,
    ).reset_index()
    wide.columns.name = None
    for pt in ALL_PT:
        if pt not in wide.columns:
            wide[pt] = 0.0
    return wide[["pitcher", "game_pk", "game_date"] + ALL_PT]


def _rolling_lagged_arsenal(df: pd.DataFrame, window: int = ARSENAL_WINDOW) -> pd.DataFrame:
    """Per pitcher, rolling lagged mean of arsenal % over last `window` appearances."""
    df = df.sort_values(["pitcher", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("pitcher", sort=False)
    for pt in ALL_PT:
        df[f"{pt}_l{window}"] = g[pt].transform(
            lambda s: s.shift(1).rolling(window, min_periods=2).mean()
        )
    return df[["game_pk", "pitcher"] + [f"{pt}_l{window}" for pt in ALL_PT]]


def _starters_per_game_side(pitches: pd.DataFrame) -> pd.DataFrame:
    """First pitcher per side = the starter. inning_topbot=Top → home pitcher."""
    s = pitches.copy()
    s["side"] = np.where(
        s["inning_topbot"].astype(str).str.startswith("T"), "home", "away"
    )
    idx = s.groupby(["game_pk", "side"])["at_bat_number"].idxmin()
    return s.loc[idx, ["game_pk", "side", "pitcher"]].rename(columns={"pitcher": "starter_id"})


def _lineups_starters(player_box_path: Path, starters: pd.DataFrame) -> pd.DataFrame:
    """Per (game_pk, side, batter, order_spot) with opposing starter id attached."""
    pb = pd.read_parquet(player_box_path,
                          columns=["game_pk", "side", "player_id", "batting_order"])
    pb = pb.dropna(subset=["batting_order", "player_id"]).copy()
    pb["order_str"] = pb["batting_order"].astype(str)
    pb = pb[pb["order_str"].str.endswith("00")]
    pb["order_spot"] = pd.to_numeric(pb["order_str"].str[:1], errors="coerce")
    pb = pb.dropna(subset=["order_spot"])
    pb["order_spot"] = pb["order_spot"].astype(int)
    pb = pb.rename(columns={"player_id": "batter"})
    pb["batter"] = pb["batter"].astype("Int64")
    # Pivot starter id per side
    starters["starter_id"] = starters["starter_id"].astype("Int64")
    sw = starters.pivot_table(index="game_pk", columns="side", values="starter_id",
                                aggfunc="first").reset_index()
    sw.columns.name = None
    sw = sw.rename(columns={"home": "home_starter", "away": "away_starter"})
    pb = pb.merge(sw, on="game_pk", how="left")
    pb["opp_starter"] = np.where(pb["side"] == "home",
                                   pb["away_starter"], pb["home_starter"])
    return pb[["game_pk", "side", "batter", "order_spot", "opp_starter"]]


def build() -> Path:
    print("[1/8] reading pitches…")
    pitches = pd.read_parquet(
        PROCESSED / "pitches.parquet",
        columns=["game_pk", "game_date", "batter", "pitcher", "pitch_type",
                 "inning_topbot", "at_bat_number",
                 "woba_denom", "estimated_woba_using_speedangle"],
    )
    pitches["batter"] = pitches["batter"].astype("Int64")
    pitches["pitcher"] = pitches["pitcher"].astype("Int64")
    pitches = _label_pt(pitches)

    print("[2/8] per-game batter×pt aggregates…")
    bvpt = _per_game_batter_vs_pt(pitches)

    print("[3/8] rolling lagged batter xwOBA per pitch type…")
    bvpt_rolled = _rolling_lagged_bvpt(bvpt)

    print("[4/8] pivoting batter splits to wide…")
    bvpt_wide = bvpt_rolled.pivot_table(
        index=["game_pk", "batter"], columns="pt", values="bvpt_xwoba",
    ).reset_index()
    bvpt_wide.columns.name = None
    for pt in ALL_PT:
        col = pt
        if col not in bvpt_wide.columns:
            bvpt_wide[col] = np.nan
    bvpt_wide = bvpt_wide.rename(columns={pt: f"xwoba_vs_{pt}" for pt in ALL_PT})

    print("[5/8] per-game pitcher arsenal + rolling…")
    arsenal = _per_game_arsenal(pitches)
    arsenal_rolled = _rolling_lagged_arsenal(arsenal)

    print("[6/8] identifying starters + lineups…")
    starters = _starters_per_game_side(pitches)
    pb_lineups = _lineups_starters(PROCESSED / "player_box.parquet", starters)

    print("[7/8] joining batter splits + opp starter arsenal…")
    # Merge in batter xwOBA per pt
    pb_lineups = pb_lineups.merge(bvpt_wide, on=["game_pk", "batter"], how="left")
    # Merge in opp starter's rolling arsenal
    arsenal_lookup = arsenal_rolled.rename(columns={"pitcher": "opp_starter"})
    pb_lineups = pb_lineups.merge(arsenal_lookup, on=["game_pk", "opp_starter"], how="left")

    print("[8/8] computing weighted matchup + aggregating…")
    # Per batter: weighted xwOBA against the specific arsenal mix
    weighted = pd.Series(0.0, index=pb_lineups.index)
    weight_sum = pd.Series(0.0, index=pb_lineups.index)
    for pt in ALL_PT:
        pct = pb_lineups.get(f"{pt}_l{ARSENAL_WINDOW}")
        xw = pb_lineups.get(f"xwoba_vs_{pt}")
        if pct is None or xw is None:
            continue
        mask = xw.notna() & pct.notna()
        contrib_pct = pct.where(mask, 0).fillna(0)
        contrib_xw = xw.where(mask, 0).fillna(0)
        weighted = weighted + contrib_pct * contrib_xw
        weight_sum = weight_sum + contrib_pct
    pb_lineups["batter_xwoba_vs_arsenal"] = (
        weighted / weight_sum.where(weight_sum > 0, np.nan)
    )

    def _wmean(g, col):
        # Linear weights — top of order bats more times against starter
        w = 1.0 - (g["order_spot"] - 1) * (0.5 / 8)
        x = g[col]; m = x.notna() & w.notna()
        if m.sum() == 0:
            return np.nan
        return float((x[m] * w[m]).sum() / w[m].sum())

    agg = pb_lineups.groupby(["game_pk", "side"]).apply(
        lambda g: pd.Series({
            "lineup_xwoba_vs_arsenal":      _wmean(g, "batter_xwoba_vs_arsenal"),
            "lineup_top4_xwoba_vs_arsenal": _wmean(g[g["order_spot"] <= 4], "batter_xwoba_vs_arsenal"),
            "lineup_arsenal_n_with_data":   int(g["batter_xwoba_vs_arsenal"].notna().sum()),
        }),
        include_groups=False,
    ).reset_index()

    out = PROCESSED / "features_arsenal.parquet"
    agg.to_parquet(out, index=False)
    print(f"\nwrote {out} ({len(agg):,} rows)")
    print(f"  mean batters w/ arsenal data: {agg['lineup_arsenal_n_with_data'].mean():.1f}/9")
    print(f"  median xwOBA vs arsenal: {agg['lineup_xwoba_vs_arsenal'].median():.3f}")
    return out


if __name__ == "__main__":
    build()
