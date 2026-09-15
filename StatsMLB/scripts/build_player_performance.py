"""Build per-player recent performance snapshot for card modals.

Aggregates the last 30 days of games from ``player_box.parquet`` into per-player
batting and pitching stats plus a form label (hot / normal / cold) so the UI can
show a clear read-out of who is coming into the game well or in a slump.

Only players who actually appeared in the last 30 days are published; anyone
without recent action is omitted so we never invent numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "StatsMLB" / "public" / "data" / "player-performance.json"
WINDOW_DAYS = 30
MIN_BAT_PA = 15
MIN_PIT_BF = 20


def label_batter(ops: float, k_pct: float, avg: float) -> str:
    if np.isnan(ops):
        return "sin datos"
    if ops >= .820 and k_pct <= .28:
        return "racha caliente"
    if ops >= .720:
        return "sólido"
    if ops <= .580 or (k_pct >= .32 and avg <= .200):
        return "racha fría"
    return "normal"


def label_pitcher(era: float, whip: float, k9: float) -> str:
    if np.isnan(era):
        return "sin datos"
    if era <= 3.20 and whip <= 1.15:
        return "dominante"
    if era <= 4.20:
        return "sólido"
    if era >= 5.50 or whip >= 1.50:
        return "en problemas"
    return "normal"


def main() -> None:
    pb = pd.read_parquet(ROOT / "data" / "processed" / "player_box.parquet")
    games = pd.read_parquet(
        ROOT / "data" / "processed" / "games.parquet",
        columns=["game_pk", "game_date"],
    )
    games["game_date"] = pd.to_datetime(games["game_date"])
    latest = games["game_date"].max()
    cutoff = latest - pd.Timedelta(days=WINDOW_DAYS)
    active_games = games[games["game_date"] >= cutoff]["game_pk"].astype(int).tolist()
    frame = pb[pb["game_pk"].isin(active_games)].copy()

    batters = frame[frame["appeared_batting"].fillna(False).astype(bool)].copy()
    bat_agg = batters.groupby(["player_id", "player_name"], dropna=True).agg(
        pa=("bat_plateAppearances", "sum"),
        ab=("bat_atBats", "sum"),
        h=("bat_hits", "sum"),
        d=("bat_doubles", "sum"),
        t=("bat_triples", "sum"),
        hr=("bat_homeRuns", "sum"),
        bb=("bat_baseOnBalls", "sum"),
        so=("bat_strikeOuts", "sum"),
        rbi=("bat_rbi", "sum"),
        runs=("bat_runs", "sum"),
        sb=("bat_stolenBases", "sum"),
        games=("game_pk", "nunique"),
    ).reset_index()
    bat_agg["avg"] = np.where(bat_agg["ab"] > 0, bat_agg["h"] / bat_agg["ab"], np.nan)
    tb = bat_agg["h"] + bat_agg["d"] + 2 * bat_agg["t"] + 3 * bat_agg["hr"]
    bat_agg["slg"] = np.where(bat_agg["ab"] > 0, tb / bat_agg["ab"], np.nan)
    obp_num = bat_agg["h"] + bat_agg["bb"]
    obp_den = bat_agg["ab"] + bat_agg["bb"]
    bat_agg["obp"] = np.where(obp_den > 0, obp_num / obp_den, np.nan)
    bat_agg["ops"] = bat_agg["obp"] + bat_agg["slg"]
    bat_agg["kPct"] = np.where(bat_agg["pa"] > 0, bat_agg["so"] / bat_agg["pa"], np.nan)
    bat_agg["bbPct"] = np.where(bat_agg["pa"] > 0, bat_agg["bb"] / bat_agg["pa"], np.nan)

    batters_out = {}
    for _, row in bat_agg[bat_agg["pa"] >= MIN_BAT_PA].iterrows():
        pid = str(int(row["player_id"]))
        batters_out[pid] = {
            "kind": "batter",
            "name": str(row["player_name"]),
            "games": int(row["games"]),
            "pa": int(row["pa"]),
            "avg": None if np.isnan(row["avg"]) else round(float(row["avg"]), 3),
            "ops": None if np.isnan(row["ops"]) else round(float(row["ops"]), 3),
            "hr": int(row["hr"]),
            "rbi": int(row["rbi"]),
            "runs": int(row["runs"]),
            "sb": int(row["sb"]),
            "kPct": None if np.isnan(row["kPct"]) else round(float(row["kPct"]), 3),
            "bbPct": None if np.isnan(row["bbPct"]) else round(float(row["bbPct"]), 3),
            "label": label_batter(float(row["ops"]) if not np.isnan(row["ops"]) else float("nan"),
                                  float(row["kPct"]) if not np.isnan(row["kPct"]) else float("nan"),
                                  float(row["avg"]) if not np.isnan(row["avg"]) else float("nan")),
        }

    pitchers = frame[frame["appeared_pitching"].fillna(False).astype(bool)].copy()
    def ip_to_outs(value):
        if pd.isna(value):
            return 0
        s = str(value)
        try:
            whole = int(s.split(".")[0])
            frac = int(s.split(".")[1]) if "." in s else 0
        except ValueError:
            return 0
        return whole * 3 + min(frac, 2)
    pitchers["outs"] = pitchers["pit_inningsPitched"].map(ip_to_outs)
    pit_agg = pitchers.groupby(["player_id", "player_name"], dropna=True).agg(
        bf=("pit_battersFaced", "sum"),
        outs=("outs", "sum"),
        h=("pit_hits", "sum"),
        er=("pit_earnedRuns", "sum"),
        bb=("pit_baseOnBalls", "sum"),
        so=("pit_strikeOuts", "sum"),
        hr=("pit_homeRuns", "sum"),
        games=("game_pk", "nunique"),
    ).reset_index()
    pit_agg["ip"] = pit_agg["outs"] / 3.0
    pit_agg["era"] = np.where(pit_agg["outs"] > 0, (pit_agg["er"] * 27) / pit_agg["outs"], np.nan)
    pit_agg["whip"] = np.where(pit_agg["outs"] > 0, ((pit_agg["bb"] + pit_agg["h"]) * 3) / pit_agg["outs"], np.nan)
    pit_agg["k9"] = np.where(pit_agg["outs"] > 0, (pit_agg["so"] * 27) / pit_agg["outs"], np.nan)
    pit_agg["bb9"] = np.where(pit_agg["outs"] > 0, (pit_agg["bb"] * 27) / pit_agg["outs"], np.nan)

    pitchers_out = {}
    for _, row in pit_agg[pit_agg["bf"] >= MIN_PIT_BF].iterrows():
        pid = str(int(row["player_id"]))
        pitchers_out[pid] = {
            "kind": "pitcher",
            "name": str(row["player_name"]),
            "games": int(row["games"]),
            "ip": None if np.isnan(row["ip"]) else round(float(row["ip"]), 1),
            "era": None if np.isnan(row["era"]) else round(float(row["era"]), 2),
            "whip": None if np.isnan(row["whip"]) else round(float(row["whip"]), 2),
            "k9": None if np.isnan(row["k9"]) else round(float(row["k9"]), 1),
            "bb9": None if np.isnan(row["bb9"]) else round(float(row["bb9"]), 1),
            "so": int(row["so"]),
            "hr": int(row["hr"]),
            "label": label_pitcher(float(row["era"]) if not np.isnan(row["era"]) else float("nan"),
                                   float(row["whip"]) if not np.isnan(row["whip"]) else float("nan"),
                                   float(row["k9"]) if not np.isnan(row["k9"]) else float("nan")),
        }

    merged = {**batters_out, **pitchers_out}
    payload = {
        "generatedAt": pd.Timestamp.now("UTC").isoformat(),
        "windowDays": WINDOW_DAYS,
        "cutoffDate": str(latest.date()),
        "players": merged,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"player performance: {len(batters_out)} bateadores + {len(pitchers_out)} pitchers -> {OUT}")


if __name__ == "__main__":
    main()
