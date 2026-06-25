"""Build team_box.parquet — one row per (game_pk, side) with full team batting + pitching."""
from __future__ import annotations

from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from .paths import RAW, PROCESSED, SEASONS

BATTING_FIELDS = [
    "runs", "hits", "doubles", "triples", "homeRuns", "rbi", "baseOnBalls",
    "intentionalWalks", "strikeOuts", "hitByPitch", "atBats", "avg", "obp",
    "slg", "ops", "stolenBases", "caughtStealing", "groundOuts", "airOuts",
    "flyOuts", "leftOnBase", "sacBunts", "sacFlies", "groundIntoDoublePlay",
    "totalBases", "plateAppearances", "pickoffs", "catchersInterference",
]
PITCHING_FIELDS = [
    "runs", "earnedRuns", "hits", "homeRuns", "strikeOuts", "baseOnBalls",
    "intentionalWalks", "hitByPitch", "atBats", "battersFaced", "inningsPitched",
    "era", "whip", "pitchesThrown", "strikes", "balls", "wildPitches", "balks",
    "groundOuts", "airOuts", "flyOuts", "doubles", "triples",
    "stolenBases", "caughtStealing", "pickoffs",
]


def _side_row(feed: dict, side: str) -> dict | None:
    box = (feed.get("liveData", {}).get("boxscore", {}) or {})
    teams = box.get("teams", {}) or {}
    side_box = teams.get(side, {}) or {}
    if not side_box:
        return None
    team = side_box.get("team", {}) or {}
    ts = side_box.get("teamStats", {}) or {}
    bat = ts.get("batting", {}) or {}
    pit = ts.get("pitching", {}) or {}
    fld = ts.get("fielding", {}) or {}
    info = side_box.get("info", []) or []
    row = {
        "game_pk": feed.get("gamePk"),
        "side": side,
        "team_id": team.get("id"),
        "team_abbrev": team.get("abbreviation"),
        "team_name": team.get("name"),
        "batters_used": len(side_box.get("batters") or []),
        "pitchers_used": len(side_box.get("pitchers") or []),
        "fielding_errors": fld.get("errors"),
        "fielding_putouts": fld.get("putOuts"),
        "fielding_assists": fld.get("assists"),
    }
    for f in BATTING_FIELDS:
        row[f"bat_{f}"] = bat.get(f)
    for f in PITCHING_FIELDS:
        row[f"pit_{f}"] = pit.get(f)
    return row


def build() -> Path:
    rows: list[dict] = []
    for season in SEASONS:
        d = RAW / "statsapi_feed" / str(season)
        files = sorted(d.glob("*.json"))
        for p in tqdm(files, desc=f"team_box {season}"):
            try:
                feed = orjson.loads(p.read_bytes())
            except Exception:
                continue
            for side in ("away", "home"):
                row = _side_row(feed, side)
                if row is not None:
                    rows.append(row)
    df = pd.DataFrame(rows)
    out = PROCESSED / "team_box.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)")
    return out


if __name__ == "__main__":
    build()
