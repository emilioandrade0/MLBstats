"""Build player_box.parquet — one row per (game_pk, player_id) with both batting + pitching stats.

A player can appear once per game per role. We emit one row per player and roll
batting + pitching stats into the same row (both can be non-null for two-way
players or position-player-pitching cameos).
"""
from __future__ import annotations

from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from .paths import RAW, PROCESSED, SEASONS

BAT = [
    "atBats", "runs", "hits", "doubles", "triples", "homeRuns", "rbi", "baseOnBalls",
    "intentionalWalks", "strikeOuts", "hitByPitch", "stolenBases", "caughtStealing",
    "groundOuts", "airOuts", "flyOuts", "leftOnBase", "totalBases", "plateAppearances",
    "groundIntoDoublePlay", "sacBunts", "sacFlies",
]
PIT = [
    "inningsPitched", "battersFaced", "atBats", "hits", "runs", "earnedRuns",
    "homeRuns", "baseOnBalls", "strikeOuts", "hitByPitch", "intentionalWalks",
    "pitchesThrown", "strikes", "balls", "wildPitches", "balks", "groundOuts",
    "airOuts", "flyOuts",
]


def _player_rows(feed: dict) -> list[dict]:
    out: list[dict] = []
    box = feed.get("liveData", {}).get("boxscore", {})
    teams = (box or {}).get("teams", {}) or {}
    game_pk = feed.get("gamePk")
    for side in ("away", "home"):
        side_box = teams.get(side, {}) or {}
        team = side_box.get("team", {}) or {}
        team_id = team.get("id")
        team_abbrev = team.get("abbreviation")
        batting_order_set = set((side_box.get("battingOrder") or []))
        batters = set(side_box.get("batters") or [])
        pitchers = set(side_box.get("pitchers") or [])
        for pid_key, p in (side_box.get("players") or {}).items():
            person = p.get("person", {}) or {}
            pid = person.get("id")
            stats = p.get("stats", {}) or {}
            bat = stats.get("batting", {}) or {}
            pit = stats.get("pitching", {}) or {}
            position = (p.get("position", {}) or {}).get("abbreviation")
            row = {
                "game_pk": game_pk,
                "side": side,
                "team_id": team_id,
                "team_abbrev": team_abbrev,
                "player_id": pid,
                "player_name": person.get("fullName"),
                "position": position,
                "jersey": p.get("jerseyNumber"),
                "batting_order": (p.get("battingOrder") or None),
                "started_batting": pid in batting_order_set if batting_order_set else None,
                "appeared_batting": pid in batters,
                "appeared_pitching": pid in pitchers,
            }
            for f in BAT:
                row[f"bat_{f}"] = bat.get(f)
            for f in PIT:
                row[f"pit_{f}"] = pit.get(f)
            out.append(row)
    return out


def build() -> Path:
    rows: list[dict] = []
    for season in SEASONS:
        d = RAW / "statsapi_feed" / str(season)
        files = sorted(d.glob("*.json"))
        for p in tqdm(files, desc=f"player_box {season}"):
            try:
                feed = orjson.loads(p.read_bytes())
            except Exception:
                continue
            rows.extend(_player_rows(feed))
    df = pd.DataFrame(rows)
    out = PROCESSED / "player_box.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)")
    return out


if __name__ == "__main__":
    build()
