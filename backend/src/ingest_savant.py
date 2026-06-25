"""Ingest Statcast pitch-level data from Baseball Savant.

Strategy: walk [start, end] in 7-day windows, save one parquet per chunk.
Parquet (not raw CSV) because the schema is stable and parquet is ~10x smaller.

Output:
    data/raw/savant/YYYY/YYYY-MM-DD__YYYY-MM-DD.parquet

Re-runs skip existing files.

Usage:
    python -m src.ingest_savant --start 2023-01-01 --end 2026-12-31
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

from .savant_client import SavantClient, week_chunks

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "savant"


def _path(season: int, start: date, end: date) -> Path:
    p = OUT / str(season) / f"{start.isoformat()}__{end.isoformat()}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


async def fetch_chunk(c: SavantClient, start: date, end: date, force: bool) -> None:
    # A chunk's season is determined by the start date; spring training in Feb/Mar
    # belongs to that calendar year's season in Savant's `hfSea` filter.
    season = start.year
    out = _path(season, start, end)
    if not force and out.exists():
        return
    df = await c.fetch_range_df(start, end, season)
    if df.empty:
        # Write an empty marker so we don't refetch nothing repeatedly.
        out.write_bytes(b"")
        return
    df.to_parquet(out, index=False)


async def run(start: date, end: date, concurrency: int, force: bool) -> None:
    async with SavantClient(concurrency=concurrency) as c:
        chunks = list(week_chunks(start, end))

        async def _do(cs: date, ce: date):
            try:
                await fetch_chunk(c, cs, ce, force)
            except Exception as e:
                print(f"[savant {cs}..{ce}] {e}")

        await tqdm.gather(*[_do(cs, ce) for cs, ce in chunks], desc="savant")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    asyncio.run(run(a.start, a.end, a.concurrency, a.force))


if __name__ == "__main__":
    main()
