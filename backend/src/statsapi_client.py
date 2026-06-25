"""MLB Stats API client — the official (free, undocumented but stable) source.

Two key endpoints:
  * /api/v1/schedule    — enumerate games (gamePk) over a date range; rich hydrates
                          give us probable pitchers, weather, venue, decisions...
  * /api/v1.1/game/{gamePk}/feed/live — the comprehensive bundle:
        - gameData: teams, players, venue, weather, probablePitchers, officials
        - liveData.boxscore: full box for both teams + per-player stats
        - liveData.linescore: inning-by-inning + current state
        - liveData.plays.allPlays: every plate appearance with playEvents,
          each pitch has pitchData (start/end speed, spin rate, break, location)
          and hitData (launch speed/angle, total distance, trajectory).
          When the game has Statcast tracking, the pitch-level data is there.

We keep one /feed/live per game on disk — that's the single source of truth.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

BASE = "https://statsapi.mlb.com/api"

# Everything useful you can pull in a single schedule call.
SCHEDULE_HYDRATE = ",".join([
    "probablePitcher",
    "decisions",
    "linescore(matchup,runners)",
    "team",
    "venue(timezone,location)",
    "weather",
    "gameInfo",
    "seriesStatus",
    "broadcasts(all)",
    "flags",
    "review",
    "officials",
    "homeRuns",
    "scoringplays",
    "stats",
])

HEADERS = {
    "User-Agent": "mlb-slec-ingest/0.1 (research)",
    "Accept": "application/json",
}


class StatsAPIClient:
    def __init__(self, concurrency: int = 8, timeout: float = 60.0):
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=HEADERS,
            http2=True,
            limits=httpx.Limits(max_connections=concurrency * 2),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "StatsAPIClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(
            (httpx.TransportError, httpx.HTTPStatusError, httpx.ReadTimeout)
        ),
    )
    async def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        async with self._sem:
            r = await self._client.get(url, params=params)
            r.raise_for_status()
            return r.json()

    async def schedule(self, start: str, end: str, sport_id: int = 1) -> dict[str, Any]:
        """Schedule for [start, end] inclusive. sport_id=1 is MLB."""
        return await self._get(
            f"{BASE}/v1/schedule",
            params={
                "sportId": sport_id,
                "startDate": start,
                "endDate": end,
                "hydrate": SCHEDULE_HYDRATE,
            },
        )

    async def feed_live(self, game_pk: int) -> dict[str, Any]:
        """Comprehensive game bundle. v1.1 returns nested liveData with all plays + pitches."""
        return await self._get(f"{BASE}/v1.1/game/{game_pk}/feed/live")

    async def boxscore(self, game_pk: int) -> dict[str, Any]:
        return await self._get(f"{BASE}/v1/game/{game_pk}/boxscore")

    async def content(self, game_pk: int) -> dict[str, Any]:
        """Editorial content: recaps, highlights, media."""
        return await self._get(f"{BASE}/v1/game/{game_pk}/content")
