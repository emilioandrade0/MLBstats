"""Ingest from MLB Stats API.

Phase 1: one schedule call per month → collect every gamePk.
         Saved at data/raw/statsapi_schedule/YYYY/YYYY-MM.json.
Phase 2: feed/live per gamePk (the big one).
         Saved at data/raw/statsapi_feed/YYYY/{gamePk}.json.

Re-runs skip files already on disk.

Usage:
    python -m src.ingest_statsapi --start 2023-01-01 --end 2026-12-31
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date
from dateutil.relativedelta import relativedelta

import httpx
from tqdm.asyncio import tqdm

from . import storage
from .statsapi_client import StatsAPIClient


def month_chunks(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = cur + relativedelta(months=1)
        chunk_start = max(cur, start)
        chunk_end = min(nxt - relativedelta(days=1), end)
        yield chunk_start, chunk_end
        cur = nxt


async def fetch_schedule(c: StatsAPIClient, start: date, end: date, force: bool) -> list[tuple[int, int]]:
    key = f"{start.year}-{start.month:02d}"
    if not force and storage.exists("statsapi_schedule", start.year, key):
        sched = storage.load("statsapi_schedule", start.year, key)
    else:
        sched = await c.schedule(start.isoformat(), end.isoformat())
        storage.save("statsapi_schedule", start.year, key, sched)

    out: list[tuple[int, int]] = []
    for day in sched.get("dates", []):
        season_year = int(day["date"][:4])
        for g in day.get("games", []):
            out.append((g["gamePk"], season_year))
    return out


async def fetch_feed(c: StatsAPIClient, game_pk: int, year: int, force: bool) -> None:
    if not force and storage.exists("statsapi_feed", year, str(game_pk)):
        return
    try:
        data = await c.feed_live(game_pk)
    except httpx.HTTPStatusError as e:
        # Some scheduled-but-not-yet-played, cancelled, or non-MLB exhibition games 404.
        if e.response.status_code in (400, 404):
            return
        raise
    storage.save("statsapi_feed", year, str(game_pk), data)


async def run(start: date, end: date, concurrency: int, force: bool) -> None:
    async with StatsAPIClient(concurrency=concurrency) as c:
        # Phase 1: schedules.
        chunks = list(month_chunks(start, end))
        all_pks: list[tuple[int, int]] = []
        for cs, ce in tqdm(chunks, desc="schedule"):
            try:
                pks = await fetch_schedule(c, cs, ce, force)
            except Exception as e:
                print(f"[schedule {cs}..{ce}] {e}")
                continue
            all_pks.extend(pks)

        # Dedup gamePks (a doubleheader can show up across chunks if dates overlap).
        seen: set[int] = set()
        unique: list[tuple[int, int]] = []
        for pk, y in all_pks:
            if pk in seen:
                continue
            seen.add(pk)
            unique.append((pk, y))
        print(f"Found {len(unique)} unique gamePks")

        # Phase 2: feeds.
        async def _do(pk: int, y: int):
            try:
                await fetch_feed(c, pk, y, force)
            except Exception as e:
                print(f"[feed {pk}] {e}")

        await tqdm.gather(*[_do(pk, y) for pk, y in unique], desc="feed/live")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    asyncio.run(run(a.start, a.end, a.concurrency, a.force))


if __name__ == "__main__":
    main()
