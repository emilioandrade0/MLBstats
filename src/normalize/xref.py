"""Build games_xref.parquet — mapping ESPN event_id ↔ StatsAPI game_pk.

Two gotchas:
  1. ESPN exposes the start as UTC. A 7pm PT night game shows up in UTC as the
     NEXT calendar day. MLB Stats API exposes `officialDate` in venue-local time.
     We shift the ESPN UTC datetime back 12 hours before slicing the date — that
     covers every US time zone without flipping early-afternoon games.
  2. Team abbreviations differ:
       MLB   -> ESPN
       AZ    -> ARI
       CWS   -> CHW
       SFG/TBR/etc — handled here.
     We canonicalize both sides to ESPN form.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from .paths import RAW, PROCESSED, SEASONS

# Map MLB-style codes -> ESPN-style canonical form.
MLB_TO_CANONICAL = {
    "AZ": "ARI",
    "CWS": "CHW",
    "WAS": "WSH",
    "SFG": "SF",
    "TBR": "TB",
    "KCR": "KC",
    "SDP": "SD",
}
# Both sources occasionally use these — make idempotent.
def _canon(abbr: str | None) -> str | None:
    if abbr is None:
        return None
    return MLB_TO_CANONICAL.get(abbr, abbr)


def _local_date_from_utc(iso_utc: str | None) -> str | None:
    if not iso_utc:
        return None
    s = iso_utc.replace("Z", "+00:00")
    try:
        # ESPN sometimes returns YYYY-MM-DDTHH:MMZ (no seconds). Pad if needed.
        if len(s) < 25 and "+" in s and s.count(":") == 2:
            pass
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Last resort: parse YYYY-MM-DDTHH:MMZ
        try:
            dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%MZ")
        except ValueError:
            return iso_utc[:10]
    return (dt - timedelta(hours=12)).date().isoformat()


def _espn_rows(season: int) -> list[dict]:
    d = RAW / "summary" / str(season)
    out: list[dict] = []
    for p in tqdm(sorted(d.glob("*.json")), desc=f"espn {season}"):
        try:
            summ = orjson.loads(p.read_bytes())
        except Exception:
            continue
        header = summ.get("header") or {}
        comps = header.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        teams = comp.get("competitors") or []
        away = next((t for t in teams if t.get("homeAway") == "away"), None)
        home = next((t for t in teams if t.get("homeAway") == "home"), None)
        if not (away and home):
            continue
        out.append({
            "espn_event_id": p.stem,
            "season": season,
            "date_local": _local_date_from_utc(comp.get("date")),
            "away_abbrev": _canon((away.get("team") or {}).get("abbreviation")),
            "home_abbrev": _canon((home.get("team") or {}).get("abbreviation")),
        })
    return out


def build() -> Path:
    espn_rows: list[dict] = []
    for s in SEASONS:
        espn_rows.extend(_espn_rows(s))
    espn = pd.DataFrame(espn_rows)

    games = pd.read_parquet(PROCESSED / "games.parquet")
    mlb = games[["game_pk", "season", "game_date", "away_team_abbrev",
                 "home_team_abbrev", "game_number"]].copy()
    mlb["away_team_abbrev"] = mlb["away_team_abbrev"].map(_canon)
    mlb["home_team_abbrev"] = mlb["home_team_abbrev"].map(_canon)
    mlb = mlb.rename(columns={
        "game_date": "date_local",
        "away_team_abbrev": "away_abbrev",
        "home_team_abbrev": "home_abbrev",
    })

    merged = espn.merge(
        mlb, on=["date_local", "away_abbrev", "home_abbrev"],
        how="outer", indicator=True,
    )
    out = PROCESSED / "games_xref.parquet"
    merged.to_parquet(out, index=False)
    matched = (merged["_merge"] == "both").sum()
    only_espn = (merged["_merge"] == "left_only").sum()
    only_mlb = (merged["_merge"] == "right_only").sum()
    print(f"wrote {out}: matched={matched:,} | espn-only={only_espn:,} | mlb-only={only_mlb:,}")
    return out


if __name__ == "__main__":
    build()
