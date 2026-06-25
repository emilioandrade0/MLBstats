"""Per-(pitcher, season) aggregated stats: W-L, ERA, WHIP, K/9, BB/9.

Aggregates `player_box.parquet` across all appearances per pitcher per season.
Decisions (W/L/SV) come from `games.parquet` (winning_pitcher_id etc).

Output: data/processed/features_pitcher_season.parquet
Key: (player_id, season)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED
from .bullpen_load import _parse_ip


def build() -> Path:
    pb = pd.read_parquet(PROCESSED / "player_box.parquet", columns=[
        "game_pk", "player_id", "pit_inningsPitched", "pit_runs", "pit_earnedRuns",
        "pit_hits", "pit_baseOnBalls", "pit_strikeOuts", "pit_homeRuns",
        "pit_battersFaced", "appeared_pitching",
    ])
    pb = pb[pb["appeared_pitching"] == True].copy()
    pb["player_id"] = pb["player_id"].astype("Int64")

    games = pd.read_parquet(PROCESSED / "games.parquet", columns=[
        "game_pk", "season", "winning_pitcher_id", "losing_pitcher_id",
        "save_pitcher_id",
    ])
    games["season"] = games["season"].astype("Int64")

    pb = pb.merge(games[["game_pk", "season"]], on="game_pk", how="left")

    pb["ip_dec"] = pb["pit_inningsPitched"].map(_parse_ip)
    for c in ["pit_runs", "pit_earnedRuns", "pit_hits", "pit_baseOnBalls",
              "pit_strikeOuts", "pit_homeRuns", "pit_battersFaced"]:
        pb[c] = pd.to_numeric(pb[c], errors="coerce").fillna(0.0)

    season = pb.groupby(["player_id", "season"], dropna=False).agg(
        ip=("ip_dec", "sum"),
        runs=("pit_runs", "sum"),
        er=("pit_earnedRuns", "sum"),
        hits=("pit_hits", "sum"),
        bb=("pit_baseOnBalls", "sum"),
        k=("pit_strikeOuts", "sum"),
        hr=("pit_homeRuns", "sum"),
        bf=("pit_battersFaced", "sum"),
        appearances=("game_pk", "count"),
    ).reset_index()

    # Decisions per (pitcher, season) from games.parquet
    def _dec(col: str, name: str) -> pd.DataFrame:
        sub = games.dropna(subset=[col]).copy()
        sub[col] = sub[col].astype("Int64")
        out = sub.groupby([col, "season"]).size().reset_index(name=name)
        return out.rename(columns={col: "player_id"})

    wins = _dec("winning_pitcher_id", "wins")
    losses = _dec("losing_pitcher_id", "losses")
    saves = _dec("save_pitcher_id", "saves")

    season = season.merge(wins, on=["player_id", "season"], how="left")
    season = season.merge(losses, on=["player_id", "season"], how="left")
    season = season.merge(saves, on=["player_id", "season"], how="left")
    for c in ("wins", "losses", "saves"):
        season[c] = season[c].fillna(0).astype(int)

    ip_safe = season["ip"].where(season["ip"] > 0, np.nan)
    bf_safe = season["bf"].where(season["bf"] > 0, np.nan)
    season["era"] = (season["er"] * 9 / ip_safe).round(2)
    season["whip"] = ((season["bb"] + season["hits"]) / ip_safe).round(2)
    season["k9"] = (season["k"] * 9 / ip_safe).round(1)
    season["bb9"] = (season["bb"] * 9 / ip_safe).round(2)
    season["k_pct"] = (season["k"] / bf_safe).round(3)
    season["bb_pct"] = (season["bb"] / bf_safe).round(3)
    season["hr9"] = (season["hr"] * 9 / ip_safe).round(2)

    # ----- xwOBA vs LHB and vs RHB from pitches.parquet -----
    print("computing xwOBA vs LHB / RHB from pitches…")
    pitches = pd.read_parquet(PROCESSED / "pitches.parquet", columns=[
        "pitcher", "game_year", "stand", "woba_denom",
        "estimated_woba_using_speedangle",
    ])
    pa = pitches[pitches["woba_denom"].fillna(0) > 0].copy()
    pa["pitcher"] = pa["pitcher"].astype("Int64")
    pa["game_year"] = pa["game_year"].astype("Int64")
    wxb = pa.groupby(["pitcher", "game_year", "stand"], dropna=False).agg(
        xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        xwoba_n=("estimated_woba_using_speedangle", "count"),
        pa=("woba_denom", "sum"),
    ).reset_index()
    wxb["xwoba"] = wxb["xwoba_sum"] / wxb["xwoba_n"].replace(0, np.nan)
    wxb = wxb.rename(columns={"pitcher": "player_id", "game_year": "season"})
    pivot = wxb.pivot_table(
        index=["player_id", "season"], columns="stand",
        values="xwoba", aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"L": "xwoba_vs_lhb", "R": "xwoba_vs_rhb"})
    if "xwoba_vs_lhb" in pivot.columns:
        pivot["xwoba_vs_lhb"] = pivot["xwoba_vs_lhb"].round(3)
    if "xwoba_vs_rhb" in pivot.columns:
        pivot["xwoba_vs_rhb"] = pivot["xwoba_vs_rhb"].round(3)
    season = season.merge(pivot, on=["player_id", "season"], how="left")

    out = PROCESSED / "features_pitcher_season.parquet"
    season.to_parquet(out, index=False)
    print(f"wrote {out} ({len(season):,} rows)")
    print(f"  pitchers w/ >=20 IP in 2025: {((season['season']==2025) & (season['ip']>=20)).sum()}")
    print(f"  pitchers w/ >=10 IP in 2026: {((season['season']==2026) & (season['ip']>=10)).sum()}")
    return out


if __name__ == "__main__":
    build()
