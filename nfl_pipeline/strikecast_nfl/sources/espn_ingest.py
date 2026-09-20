"""ESPN NFL odds + scoreboard ingest — free, multi-book, no API key.

Recycled from STRIKECAST's espn_client.py with sport switched to NFL.

Endpoints used:
  - site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=YYYYMMDD
        Daily scoreboard with event ids, teams, kickoff, current odds preview.
  - sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{id}/competitions/{id}/odds
        Multi-provider odds (ESPN BET, Caesars, DK, FanDuel, etc.) per event.

Outputs:
  data/raw/espn/scoreboard/{YYYY-MM-DD}.json
  data/raw/espn/odds/{season}/{event_id}.json

Usage:
    python -m strikecast_nfl.sources.espn_ingest --date 2025-09-07
    python -m strikecast_nfl.sources.espn_ingest --season 2024   # every game of a past season
    python -m strikecast_nfl.sources.espn_ingest --week current  # scoreboard for the next 8 days
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import orjson
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..paths import RAW

SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

OUT_SCOREBOARD = RAW / "espn" / "scoreboard"
OUT_ODDS = RAW / "espn" / "odds"


class ESPNClient:
    def __init__(self, timeout: float = 30.0, concurrency: int = 8):
        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        self._client = httpx.AsyncClient(timeout=timeout, http2=True, limits=limits)
        self._sem = asyncio.Semaphore(concurrency)

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    async def _get(self, url: str, params: dict | None = None) -> dict[str, Any]:
        async with self._sem:
            r = await self._client.get(url, params=params or {})
            r.raise_for_status()
            return r.json()

    async def scoreboard(self, day: date) -> dict[str, Any]:
        return await self._get(f"{SITE_BASE}/scoreboard",
                               params={"dates": day.strftime("%Y%m%d")})

    async def event_odds(self, event_id: str) -> dict[str, Any]:
        return await self._get(
            f"{CORE_BASE}/events/{event_id}/competitions/{event_id}/odds"
        )


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


async def ingest_day(client: ESPNClient, day: date, fetch_odds: bool = True) -> list[str]:
    sb = await client.scoreboard(day)
    _save_json(OUT_SCOREBOARD / f"{day.isoformat()}.json", sb)
    events = sb.get("events") or []
    event_ids = [str(e.get("id")) for e in events if e.get("id")]
    season = sb.get("season", {}).get("year") or day.year

    if fetch_odds and event_ids:
        async def _one(eid: str) -> None:
            try:
                payload = await client.event_odds(eid)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return  # no odds for that event yet
                raise
            _save_json(OUT_ODDS / str(season) / f"{eid}.json", payload)

        await asyncio.gather(*(_one(eid) for eid in event_ids))

    return event_ids


async def ingest_range(start: date, end: date, fetch_odds: bool = True) -> None:
    client = ESPNClient()
    try:
        day = start
        while day <= end:
            ids = await ingest_day(client, day, fetch_odds=fetch_odds)
            print(f"  {day}: {len(ids):>2} events")
            day += timedelta(days=1)
    finally:
        await client.close()


def _season_range(season: int) -> tuple[date, date]:
    """NFL season roughly runs Sep of `season` -> Feb of `season+1` (Super Bowl)."""
    return date(season, 8, 1), date(season + 1, 2, 28)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="Single day YYYY-MM-DD")
    g.add_argument("--range", nargs=2, metavar=("START", "END"),
                   help="Date range YYYY-MM-DD YYYY-MM-DD")
    g.add_argument("--season", type=int, help="Full NFL season (Aug..Feb)")
    g.add_argument("--week", choices=["current"], help="'current' = today .. +8 days")
    ap.add_argument("--no-odds", action="store_true", help="Skip per-event odds (scoreboard only)")
    args = ap.parse_args()

    if args.date:
        d = date.fromisoformat(args.date)
        start, end = d, d
    elif args.range:
        start, end = date.fromisoformat(args.range[0]), date.fromisoformat(args.range[1])
    elif args.season:
        start, end = _season_range(args.season)
    else:  # --week current
        today = datetime.now(timezone.utc).date()
        start, end = today, today + timedelta(days=8)

    print(f"ESPN ingest: {start} -> {end}  (odds={not args.no_odds})")
    asyncio.run(ingest_range(start, end, fetch_odds=not args.no_odds))
    print("Done.")


if __name__ == "__main__":
    main()
