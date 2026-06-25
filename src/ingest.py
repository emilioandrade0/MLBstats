"""End-to-end ingest:
  1. For every date in [start, end], pull the scoreboard.
  2. Collect event ids from each scoreboard.
  3. For every event id, pull summary + odds + probabilities concurrently.

Re-runs are safe: files already on disk are skipped unless --force is passed.

Usage:
    python -m src.ingest --start 2002-01-01 --end 2026-06-22
    python -m src.ingest --season 2024
    python -m src.ingest --start 2024-03-20 --end 2024-11-01 --concurrency 12
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date, timedelta

import httpx
from tqdm.asyncio import tqdm

from .espn_client import ESPNClient
from . import storage


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


async def fetch_scoreboard(client: ESPNClient, d: date, force: bool) -> list[str]:
    yyyymmdd = d.strftime("%Y%m%d")
    if not force and storage.exists("scoreboard", d.year, yyyymmdd):
        try:
            sb = storage.load("scoreboard", d.year, yyyymmdd)
        except Exception:
            sb = await client.scoreboard(yyyymmdd)
            storage.save("scoreboard", d.year, yyyymmdd, sb)
    else:
        sb = await client.scoreboard(yyyymmdd)
        storage.save("scoreboard", d.year, yyyymmdd, sb)
    return [ev["id"] for ev in sb.get("events", [])]


async def fetch_event(client: ESPNClient, event_id: str, year: int, force: bool) -> None:
    async def _one(kind: str, coro):
        if not force and storage.exists(kind, year, event_id):
            return
        try:
            data = await coro
        except httpx.HTTPStatusError as e:
            # Secondary endpoints (odds, probabilities) return 400/404 for events
            # that simply don't have that data (spring training, postponed, pre-2010, etc).
            if e.response.status_code in (400, 404) and kind in ("odds", "probabilities"):
                return
            raise
        storage.save(kind, year, event_id, data)

    await asyncio.gather(
        _one("summary", client.summary(event_id)),
        _one("odds", client.event_odds(event_id)),
        _one("probabilities", client.event_probabilities(event_id)),
    )


async def run(start: date, end: date, concurrency: int, force: bool) -> None:
    async with ESPNClient(concurrency=concurrency) as client:
        dates = list(daterange(start, end))
        # Phase 1: scoreboards (gives us the event id universe).
        event_ids: list[tuple[str, int]] = []
        for d in tqdm(dates, desc="scoreboards"):
            try:
                ids = await fetch_scoreboard(client, d, force)
            except Exception as e:
                print(f"[scoreboard {d}] {e}")
                continue
            event_ids.extend((eid, d.year) for eid in ids)

        # De-dup (scoreboard rollovers around midnight UTC can repeat an id on adjacent days).
        seen: set[str] = set()
        unique: list[tuple[str, int]] = []
        for eid, y in event_ids:
            if eid in seen:
                continue
            seen.add(eid)
            unique.append((eid, y))

        # Phase 2: per-event bundle.
        async def _do(eid: str, y: int):
            try:
                await fetch_event(client, eid, y, force)
            except Exception as e:
                print(f"[event {eid}] {e}")

        await tqdm.gather(*[_do(eid, y) for eid, y in unique], desc="events")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", type=date.fromisoformat, help="YYYY-MM-DD inclusive")
    p.add_argument("--season", type=int, help="Shortcut: ingest a full MLB season window")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--force", action="store_true", help="Re-download files already on disk")
    args = p.parse_args()

    if args.season:
        # Spring training through World Series — generous window.
        args.start = date(args.season, 2, 15)
        args.end = date(args.season, 11, 15)
    if not (args.start and args.end):
        p.error("provide --season or both --start and --end")
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run(args.start, args.end, args.concurrency, args.force))


if __name__ == "__main__":
    main()
