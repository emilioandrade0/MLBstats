"""Bullpen quality features — que tan BUENOS son los relievers del team.

El modelo YA sabe cuanto tira el bullpen (IP acumulado); NO sabe si son buenos.
Este modulo agrega rate stats season-to-date del bullpen entero, estrictamente
LAGGED (solo apariciones antes de la fecha del target — cero data leakage).

Features per (game_pk, side):
  bullpen_k9_season             — strikeouts por 9 IP season-to-date
  bullpen_bb9_season            — walks por 9 IP
  bullpen_k_minus_bb_pct        — (K-BB)/PA, well-correlated con ERA verdadera
  bullpen_hr9_season            — HR por 9 IP
  bullpen_fip_season            — Fielding-Independent Pitching = (13*HR + 3*BB - 2*K)/IP + 3.10
  bullpen_ip_season             — IP acumulado (proxy de trust del manager)
  bullpen_whip_season           — walks + hits per IP

Data source: player_box.parquet (basicos por juego + pitcher).
Starter detection via pitches.parquet (primer pitcher del team en el juego).

Output: data/processed/features_bullpen_quality.parquet  (game_pk, side)

Run standalone:
  python -m src.features.bullpen_quality
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED

OUT = PROCESSED / "features_bullpen_quality.parquet"
FIP_CONSTANT = 3.10  # ajuste tipico para MLB moderna


def _parse_ip(v) -> float:
    """MLB Stats API stores IP como '5.2' = 5 innings + 2 outs. Convert a decimal."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    s = str(v)
    if "." in s:
        try:
            whole, frac = s.split(".", 1)
            return float(whole) + int(frac[0]) / 3.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _starters_per_game_side(pitches_path: Path) -> pd.DataFrame:
    """Para cada (game_pk, pitching side), el pitcher que abrio (minor at_bat_number)."""
    p = pd.read_parquet(pitches_path, columns=[
        "game_pk", "inning_topbot", "pitcher", "at_bat_number",
    ])
    p["side"] = np.where(p["inning_topbot"].astype(str).str.startswith("T"),
                          "home", "away")
    idx = p.groupby(["game_pk", "side"])["at_bat_number"].idxmin()
    return p.loc[idx, ["game_pk", "side", "pitcher"]].rename(columns={"pitcher": "starter_id"})


def build() -> Path:
    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "season", "game_type",
                 "home_team_abbrev", "away_team_abbrev"],
    )
    games = games[games["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    print("loading starters and player box...")
    starters = _starters_per_game_side(PROCESSED / "pitches.parquet")

    pb = pd.read_parquet(
        PROCESSED / "player_box.parquet",
        columns=["game_pk", "side", "player_id", "appeared_pitching",
                 "pit_inningsPitched", "pit_battersFaced",
                 "pit_strikeOuts", "pit_baseOnBalls", "pit_homeRuns",
                 "pit_hits", "pit_earnedRuns"],
    )
    pb = pb[pb["appeared_pitching"] == True].copy()
    pb["player_id"] = pb["player_id"].astype("Int64")
    starters["starter_id"] = starters["starter_id"].astype("Int64")

    # Filtrar a solo relievers (no starters)
    pb = pb.merge(starters, on=["game_pk", "side"], how="left")
    pb["is_starter"] = (pb["player_id"] == pb["starter_id"])
    rel = pb[~pb["is_starter"]].copy()

    # Convert a numericos y parse IP
    rel["ip"]      = rel["pit_inningsPitched"].map(_parse_ip)
    rel["k"]       = pd.to_numeric(rel["pit_strikeOuts"], errors="coerce").fillna(0)
    rel["bb"]      = pd.to_numeric(rel["pit_baseOnBalls"], errors="coerce").fillna(0)
    rel["hr"]      = pd.to_numeric(rel["pit_homeRuns"], errors="coerce").fillna(0)
    rel["pa"]      = pd.to_numeric(rel["pit_battersFaced"], errors="coerce").fillna(0)
    rel["hits"]    = pd.to_numeric(rel["pit_hits"], errors="coerce").fillna(0)

    # Attach game_date + team a cada aparicion
    home_pk = games[["game_pk", "game_date", "season", "home_team_abbrev"]].rename(
        columns={"home_team_abbrev": "team"})
    home_pk["side"] = "home"
    away_pk = games[["game_pk", "game_date", "season", "away_team_abbrev"]].rename(
        columns={"away_team_abbrev": "team"})
    away_pk["side"] = "away"
    lookup = pd.concat([home_pk, away_pk], ignore_index=True)
    rel = rel.merge(lookup, on=["game_pk", "side"], how="left")
    rel = rel.dropna(subset=["team", "game_date", "season"])

    # ── Agregar por (team, season) via expanding SHIFTED (estrictamente lagged) ──
    # Primero suma todas las contribuciones del team en cada juego
    team_game = rel.groupby(["team", "season", "game_date", "game_pk"]).agg(
        ip=("ip","sum"), k=("k","sum"), bb=("bb","sum"),
        hr=("hr","sum"), pa=("pa","sum"), hits=("hits","sum"),
    ).reset_index()

    # Ordenar cronologicamente por team-season, y expanding sum shifted por 1
    team_game = team_game.sort_values(["team", "season", "game_date", "game_pk"]).reset_index(drop=True)
    grp = team_game.groupby(["team", "season"], sort=False)
    # SEASON expanding (lifetime)
    for col in ["ip", "k", "bb", "hr", "pa", "hits"]:
        team_game[f"{col}_cum"] = grp[col].transform(lambda s: s.shift(1).expanding().sum())
    # RECENT rolling — ultimos 10 team games (bullpen form)
    WINDOW = 10
    for col in ["ip", "k", "bb", "hr", "pa", "hits"]:
        team_game[f"{col}_l{WINDOW}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=3).sum())

    # ── Rate stats SEASON ──
    ip = team_game["ip_cum"];  pa = team_game["pa_cum"]
    ip_safe = ip.where(ip > 0); pa_safe = pa.where(pa > 0)
    team_game["bullpen_ip_season"]           = ip.fillna(0.0)
    team_game["bullpen_k9_season"]           = 9 * team_game["k_cum"] / ip_safe
    team_game["bullpen_bb9_season"]          = 9 * team_game["bb_cum"] / ip_safe
    team_game["bullpen_hr9_season"]          = 9 * team_game["hr_cum"] / ip_safe
    team_game["bullpen_k_minus_bb_pct"]      = (team_game["k_cum"] - team_game["bb_cum"]) / pa_safe
    team_game["bullpen_whip_season"]         = (team_game["bb_cum"] + team_game["hits_cum"]) / ip_safe
    team_game["bullpen_fip_season"] = (
        (13 * team_game["hr_cum"] + 3 * team_game["bb_cum"] - 2 * team_game["k_cum"]) / ip_safe
        + FIP_CONSTANT
    )

    # ── Rate stats RECENT (last 10 team games) ──
    ip_r = team_game[f"ip_l{WINDOW}"];   pa_r = team_game[f"pa_l{WINDOW}"]
    ip_rs = ip_r.where(ip_r > 0);         pa_rs = pa_r.where(pa_r > 0)
    team_game["bullpen_k9_recent"]           = 9 * team_game[f"k_l{WINDOW}"] / ip_rs
    team_game["bullpen_bb9_recent"]          = 9 * team_game[f"bb_l{WINDOW}"] / ip_rs
    team_game["bullpen_hr9_recent"]          = 9 * team_game[f"hr_l{WINDOW}"] / ip_rs
    team_game["bullpen_fip_recent"] = (
        (13 * team_game[f"hr_l{WINDOW}"] + 3 * team_game[f"bb_l{WINDOW}"] - 2 * team_game[f"k_l{WINDOW}"]) / ip_rs
        + FIP_CONSTANT
    )

    # ── DELTA: recent form vs season baseline (donde vive el edge potencial)
    # positivo en FIP = bullpen EMPEORANDO (deberia bajar p_home si es home team)
    # positivo en K9  = bullpen MEJORANDO
    team_game["bullpen_fip_form_delta"] = team_game["bullpen_fip_recent"] - team_game["bullpen_fip_season"]
    team_game["bullpen_k9_form_delta"]  = team_game["bullpen_k9_recent"]  - team_game["bullpen_k9_season"]

    # Emit por (game_pk, side)  — join back con lookup
    out_cols = [c for c in team_game.columns if c.startswith("bullpen_")]
    team_game = team_game.merge(lookup, on=["team", "season", "game_date", "game_pk"], how="inner")
    out = team_game[["game_pk", "side"] + out_cols].copy()

    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out):,} rows)")
    print()
    print("Distribuciones de features nuevas:")
    for c in out_cols:
        s = out[c].dropna()
        if len(s) == 0: continue
        print(f"  {c:32s}  n={len(s):5d}  mean={s.mean():+.3f}  std={s.std():.3f}  "
              f"p10/50/90={s.quantile(0.1):+.2f}/{s.quantile(0.5):+.2f}/{s.quantile(0.9):+.2f}")

    return OUT


if __name__ == "__main__":
    build()
