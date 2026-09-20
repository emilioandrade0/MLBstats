"""Capture per-book NFL moneylines/spreads/totals from Action Network.

Runs hourly in CI. Determines the active NFL week from the local schedule
and queries `api.actionnetwork.com/web/v1/scoreboard/nfl` for both this week
and next week (games span Thu-Mon). Every book's line is stored in
``data/processed/nfl_odds_timeseries.parquet`` with the current UTC timestamp
so the panel Movimiento de línea can plot how the market moved.

Same endpoint the /api/nfl/live route uses; zero API cost (public JSON).
"""
from __future__ import annotations

import json
import ssl
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "nfl_odds_timeseries.parquet"
SCHEDULE = ROOT / "StatsMLB" / "public" / "data" / "nfl" / "games.json"

UA = "StatsMLB/1.0 (+snapshot-nfl-odds)"


def http_get_json(url: str, timeout: int = 20) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        print(f"  [warn] GET {url[:110]}...: {exc}")
        return None


def active_weeks(today: datetime) -> list[tuple[int, int, str]]:
    """Return (season, week, phase) for the current + next NFL week."""
    if not SCHEDULE.exists():
        return []
    try:
        schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    except Exception:
        return []
    games = schedule if isinstance(schedule, list) else schedule.get("games") or []
    today_ts = today.timestamp()
    upcoming = []
    for g in games:
        kickoff = g.get("kickoff_utc")
        if not kickoff:
            continue
        try:
            k = datetime.fromisoformat(kickoff.replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        season = g.get("season")
        week = g.get("week")
        game_type = (g.get("game_type") or "REG").upper()
        if not season or not week:
            continue
        # window: 3 days before to 7 days after → cubre current + next week
        if today_ts - 3 * 86400 <= k <= today_ts + 7 * 86400:
            regular_weeks = 18 if season >= 2021 else 17
            if game_type == "REG":
                phase, wk = "reg", week
            elif game_type in ("WC", "DIV", "CON", "SB", "POST"):
                phase, wk = "post", week - regular_weeks
            else:
                continue
            if 1 <= wk <= (4 if phase == "post" else 18):
                upcoming.append((season, wk, phase))
    return sorted(set(upcoming))


def scoreboard(season: int, week: int, phase: str) -> list[dict]:
    url = (
        f"https://api.actionnetwork.com/web/v1/scoreboard/nfl"
        f"?period=game&season={season}&week={week}&seasonType={phase}"
    )
    data = http_get_json(url)
    if not data or not isinstance(data, dict):
        return []
    return data.get("games") or []


def main() -> None:
    snapshot_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    today = datetime.now(timezone.utc)
    weeks = active_weeks(today)
    if not weeks:
        # fallback: try just this week reg season
        weeks = [(today.year, max(1, min(18, (today.timetuple().tm_yday - 244) // 7 + 1)), "reg")]
    print(f"[snapshot_nfl_odds] active windows: {weeks}")

    rows: list[dict] = []
    for season, week, phase in weeks:
        games = scoreboard(season, week, phase)
        print(f"  {season} wk{week} {phase}: {len(games)} juegos")
        for g in games:
            if g.get("league_name") != "nfl":
                continue
            source_id = g.get("id")
            teams = g.get("teams") or []
            home_id = g.get("home_team_id")
            away_id = g.get("away_team_id")
            home = next((t.get("abbr") for t in teams if t.get("id") == home_id), None)
            away = next((t.get("abbr") for t in teams if t.get("id") == away_id), None)
            if not source_id or not home or not away:
                continue
            for odd in g.get("odds") or []:
                if odd.get("type") != "game":
                    continue
                book_id = odd.get("book_id")
                if not isinstance(book_id, int):
                    continue
                rows.append({
                    "snapshot_ts": snapshot_ts,
                    "action_game_id": int(source_id),
                    "season": season,
                    "week": week,
                    "phase": phase,
                    "home_team": home,
                    "away_team": away,
                    "kickoff_utc": g.get("start_time"),
                    "book_id": book_id,
                    "home_ml": odd.get("ml_home"),
                    "away_ml": odd.get("ml_away"),
                    "home_spread": odd.get("spread_home"),
                    "total": odd.get("total"),
                    "book_updated_at": odd.get("inserted"),
                })

    if not rows:
        print(f"[snapshot_nfl_odds] 0 filas capturadas")
        return

    new_df = pd.DataFrame(rows)
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["snapshot_ts", "action_game_id", "book_id"], keep="last"
        )
    else:
        combined = new_df
    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT, index=False)
    print(
        f"[snapshot_nfl_odds] +{len(new_df):,} filas ({snapshot_ts}); "
        f"total {len(combined):,} filas en {combined['snapshot_ts'].nunique()} snapshots"
    )


if __name__ == "__main__":
    main()
