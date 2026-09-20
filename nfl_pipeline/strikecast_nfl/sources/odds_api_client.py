"""the-odds-api.com client — sharp reference (Pinnacle) for NFL.

Recycled from STRIKECAST with SPORT switched to NFL. Free tier: 500 req/month.
Each /odds call is 1 credit per region listed.

    $env:ODDS_API_KEY = "your-key"
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"


def _key() -> str:
    k = os.environ.get("ODDS_API_KEY")
    if not k:
        raise SystemExit("ODDS_API_KEY not set. Run:  $env:ODDS_API_KEY = 'your-key'")
    return k


class OddsAPIClient:
    def __init__(self, timeout: float = 30.0):
        self._client = httpx.Client(timeout=timeout, http2=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _get(self, path: str, params: dict[str, Any]) -> tuple[Any, dict]:
        params = {**params, "apiKey": _key()}
        r = self._client.get(f"{BASE}{path}", params=params)
        r.raise_for_status()
        headers = {
            "remaining": r.headers.get("x-requests-remaining"),
            "used": r.headers.get("x-requests-used"),
            "last_cost": r.headers.get("x-requests-last"),
        }
        return r.json(), headers

    def events(self) -> tuple[list, dict]:
        return self._get(f"/sports/{SPORT}/events", params={})

    def odds(
        self,
        regions: str = "eu,us",
        markets: str = "h2h,totals,spreads",
        bookmakers: str | None = "pinnacle,draftkings,fanduel,betmgm,caesars,williamhill_us",
        date_format: str = "iso",
    ) -> tuple[list, dict]:
        """Current odds for all upcoming NFL events."""
        params: dict[str, Any] = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
            "dateFormat": date_format,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return self._get(f"/sports/{SPORT}/odds", params=params)
