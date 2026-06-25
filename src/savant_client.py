"""Baseball Savant (Statcast) — pitch-level CSV bulk export.

The /statcast_search/csv endpoint returns one row per pitch with the full
Statcast suite: release_speed, release_spin_rate, release_extension,
plate_x/z, pfx_x/z, sz_top/sz_bot, launch_speed, launch_angle,
hit_distance_sc, hc_x/y, estimated_woba_using_speedangle,
estimated_ba_using_speedangle, woba_value, delta_run_exp, bat_speed,
swing_length, and ~90 more columns.

The web UI caps results so we chunk by week. Per-game queries are also
possible via `game_pk=...` but bulk by date range is faster for backfill.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from io import StringIO

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

BASE = "https://baseballsavant.mlb.com/statcast_search/csv"

# Static params reverse-engineered from the search UI. Empty strings reproduce
# "any" filters; `all=true` removes the 25k row server cap (still risky on long
# ranges — we chunk by week anyway).
COMMON_PARAMS: dict[str, str] = {
    "all": "true",
    "hfPT": "",  # pitch type
    "hfAB": "",
    "hfGT": "R|PO|S|",  # game type: Regular | Postseason | Spring — drop suffix to taste
    "hfPR": "",  # play result
    "hfZ": "",
    "hfStadium": "",
    "hfBBL": "",
    "hfNewZones": "",
    "hfPull": "",
    "hfC": "",
    "hfSea": "",
    "hfSit": "",
    "player_type": "pitcher",
    "hfOuts": "",
    "hfOpponent": "",
    "pitcher_throws": "",
    "batter_stands": "",
    "hfSA": "",
    "game_date_gt": "",
    "game_date_lt": "",
    "hfMo": "",
    "hfTeam": "",
    "home_road": "",
    "hfRO": "",
    "position": "",
    "hfInfield": "",
    "hfOutfield": "",
    "hfInn": "",
    "hfBBT": "",
    "hfFlag": "",
    "metric_1": "",
    "group_by": "name",
    "min_pitches": "0",
    "min_results": "0",
    "min_pas": "0",
    "sort_col": "pitches",
    "player_event_sort": "api_p_release_speed",
    "sort_order": "desc",
    "type": "details",
}

HEADERS = {
    "User-Agent": "mlb-slec-ingest/0.1 (research)",
    "Accept": "text/csv,*/*",
}


class SavantClient:
    def __init__(self, concurrency: int = 4, timeout: float = 120.0):
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=HEADERS,
            http2=True,
            limits=httpx.Limits(max_connections=concurrency * 2),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "SavantClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(
            (httpx.TransportError, httpx.HTTPStatusError, httpx.ReadTimeout)
        ),
    )
    async def fetch_range_csv(self, start: date, end: date, season: int) -> str:
        """Returns raw CSV text for pitches in [start, end] (inclusive)."""
        params = dict(COMMON_PARAMS)
        params["game_date_gt"] = start.isoformat()
        params["game_date_lt"] = end.isoformat()
        params["hfSea"] = f"{season}|"
        async with self._sem:
            r = await self._client.get(BASE, params=params)
            r.raise_for_status()
            return r.text

    async def fetch_range_df(self, start: date, end: date, season: int) -> pd.DataFrame:
        csv = await self.fetch_range_csv(start, end, season)
        if not csv.strip():
            return pd.DataFrame()
        return pd.read_csv(StringIO(csv), low_memory=False)


def week_chunks(start: date, end: date):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=6), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)
