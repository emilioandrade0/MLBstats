"""Probe one date / one event to inspect the shape of every payload.

Run this BEFORE writing the normalized parser so we know what fields exist:
    python -m src.probe 2024-07-15
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date

from .espn_client import ESPNClient


def _keys(obj, prefix="", depth=0, max_depth=3):
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{'  ' * depth}{prefix}{k}: {type(v).__name__}")
            if isinstance(v, (dict, list)) and depth < max_depth:
                _keys(v, prefix="", depth=depth + 1, max_depth=max_depth)
    elif isinstance(obj, list) and obj:
        print(f"{'  ' * depth}[list len={len(obj)}]")
        _keys(obj[0], depth=depth + 1, max_depth=max_depth)


async def main(d: date) -> None:
    async with ESPNClient() as c:
        sb = await c.scoreboard(d.strftime("%Y%m%d"))
        events = sb.get("events", [])
        print(f"\n=== scoreboard {d}: {len(events)} events ===")
        if not events:
            return
        eid = events[0]["id"]
        print(f"\n--- scoreboard event[0] (id={eid}) keys ---")
        _keys(events[0])

        summary = await c.summary(eid)
        print(f"\n--- summary top-level keys ---")
        for k in summary:
            v = summary[k]
            note = f"len={len(v)}" if isinstance(v, (list, dict)) else ""
            print(f"  {k} ({type(v).__name__}) {note}")

        try:
            odds = await c.event_odds(eid)
            print(f"\n--- odds: {len(odds.get('items', []))} providers ---")
        except Exception as e:
            print(f"odds: {e}")

        try:
            prob = await c.event_probabilities(eid)
            print(f"\n--- probabilities: {prob.get('count', 0)} samples ---")
        except Exception as e:
            print(f"probabilities: {e}")


if __name__ == "__main__":
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2024, 7, 15)
    asyncio.run(main(d))
