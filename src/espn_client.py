"""Thin async client over ESPN's public MLB endpoints.

ESPN exposes two useful hosts:
  - site.api.espn.com : scoreboard + summary (game detail bundle)
  - sports.core.api.espn.com : granular resources (athletes, teams, odds, plays...)

These endpoints are undocumented; field shape can change. We persist the raw
JSON so the parsing layer can be re-run without re-downloading.
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

SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
CORE_BASE = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"

DEFAULT_HEADERS = {
    "User-Agent": "mlb-slec-ingest/0.1 (research)",
    "Accept": "application/json",
}


class ESPNClient:
    def __init__(
        self,
        concurrency: int = 8,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._sem = asyncio.Semaphore(concurrency)
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            headers=DEFAULT_HEADERS,
            http2=True,
            limits=httpx.Limits(max_connections=concurrency * 2),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ESPNClient":
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
            if r.status_code in (429, 500, 502, 503, 504):
                r.raise_for_status()
            r.raise_for_status()
            return r.json()

    async def scoreboard(self, yyyymmdd: str) -> dict[str, Any]:
        """Scoreboard for a single calendar date (UTC). Includes event ids + odds preview."""
        return await self._get(
            f"{SITE_BASE}/scoreboard",
            params={"dates": yyyymmdd, "limit": 200},
        )

    async def summary(self, event_id: str) -> dict[str, Any]:
        """Full game bundle: boxscore, plays, lineups, odds, win-probability, leaders, injuries, weather."""
        return await self._get(f"{SITE_BASE}/summary", params={"event": event_id})

    async def event_odds(self, event_id: str) -> dict[str, Any]:
        """Per-provider odds history. Lives on the core host."""
        return await self._get(f"{CORE_BASE}/events/{event_id}/competitions/{event_id}/odds")

    async def event_probabilities(self, event_id: str) -> dict[str, Any]:
        return await self._get(
            f"{CORE_BASE}/events/{event_id}/competitions/{event_id}/probabilities",
            params={"limit": 1000},
        )

    async def teams(self) -> dict[str, Any]:
        return await self._get(f"{SITE_BASE}/teams", params={"limit": 100})
