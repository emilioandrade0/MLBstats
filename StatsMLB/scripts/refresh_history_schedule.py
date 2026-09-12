from __future__ import annotations

import argparse
import json
import tempfile
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_ROOT = ROOT / "data" / "raw" / "statsapi_schedule"
def current_mexico_date() -> date:
    # The local launcher runs on the user's Mexico City workstation.
    return date.today()


def month_bounds(day: date) -> tuple[date, date]:
    start = day.replace(day=1)
    if day.month == 12:
        next_month = date(day.year + 1, 1, 1)
    else:
        next_month = date(day.year, day.month + 1, 1)
    return start, date.fromordinal(next_month.toordinal() - 1)


def fetch_month(day: date) -> tuple[Path, int, int]:
    start, end = month_bounds(day)
    query = urllib.parse.urlencode({
        "sportId": 1,
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "hydrate": "team,linescore",
    })
    request = urllib.request.Request(
        f"https://statsapi.mlb.com/api/v1/schedule?{query}",
        headers={"Accept": "application/json", "User-Agent": "StatsMLB-history-refresh/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)

    dates = payload.get("dates")
    if not isinstance(dates, list):
        raise RuntimeError("MLB StatsAPI no devolvió una lista de fechas válida")
    games = sum(len(block.get("games", [])) for block in dates if isinstance(block, dict))
    finals = sum(
        1
        for block in dates if isinstance(block, dict)
        for game in block.get("games", [])
        if game.get("status", {}).get("abstractGameState") == "Final"
    )
    if games == 0:
        raise RuntimeError("MLB StatsAPI devolvió el mes sin juegos; se conservó el archivo anterior")

    target_dir = SCHEDULE_ROOT / str(day.year)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{day.year}-{day.month:02d}.json"
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=target_dir, suffix=".tmp", delete=False) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    temporary.replace(target)
    return target, games, finals


def main() -> None:
    parser = argparse.ArgumentParser(description="Actualiza el mes completo de MLB sin truncar su calendario local.")
    parser.add_argument("--date", type=date.fromisoformat, default=current_mexico_date())
    args = parser.parse_args()
    target, games, finals = fetch_month(args.date)
    print(json.dumps({"file": str(target), "games": games, "finals": finals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
