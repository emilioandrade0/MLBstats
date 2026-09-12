"""Build games.parquet — one row per game, MLB Stats API as canonical source.

Output columns (selection):
  game_pk, season, game_date, game_type, status, double_header, game_number,
  venue_id, venue_name, venue_city, venue_state, roof_type, turf_type,
  away_team_id, away_team_abbrev, home_team_id, home_team_abbrev,
  away_score, home_score, away_hits, home_hits, away_errors, home_errors,
  scheduled_innings, final_innings, attendance, day_night, first_pitch_utc,
  duration_minutes, weather_condition, weather_temp_f, weather_wind,
  probable_away_pitcher_id, probable_home_pitcher_id,
  winning_pitcher_id, losing_pitcher_id, save_pitcher_id,
  away_record_wins, away_record_losses, home_record_wins, home_record_losses,
  espn_event_id  (joined later via games_xref)
"""
from __future__ import annotations

from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from .paths import RAW, PROCESSED, SEASONS


def _to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def parse_feed(path: Path) -> dict | None:
    try:
        feed = orjson.loads(path.read_bytes())
    except Exception:
        return None

    gd = feed.get("gameData", {})
    ld = feed.get("liveData", {})
    game = gd.get("game", {})
    dt = gd.get("datetime", {})
    status = gd.get("status", {})
    venue = gd.get("venue", {})
    loc = venue.get("location", {}) or {}
    field = venue.get("fieldInfo", {}) or {}
    weather = gd.get("weather", {}) or {}
    ginfo = gd.get("gameInfo", {}) or {}
    teams = gd.get("teams", {}) or {}
    away_t = teams.get("away", {}) or {}
    home_t = teams.get("home", {}) or {}
    away_rec = away_t.get("record", {}) or {}
    home_rec = home_t.get("record", {}) or {}
    probable = gd.get("probablePitchers", {}) or {}

    linescore = ld.get("linescore", {}) or {}
    teams_ls = linescore.get("teams", {}) or {}
    away_ls = teams_ls.get("away", {}) or {}
    home_ls = teams_ls.get("home", {}) or {}

    box = ld.get("boxscore", {}) or {}
    box_teams = box.get("teams", {}) or {}
    away_box = box_teams.get("away", {}) or {}
    home_box = box_teams.get("home", {}) or {}
    away_ts = (away_box.get("teamStats", {}) or {}).get("batting", {}) or {}
    home_ts = (home_box.get("teamStats", {}) or {}).get("batting", {}) or {}

    decisions = ld.get("decisions", {}) or {}
    w = decisions.get("winner", {}) or {}
    l = decisions.get("loser", {}) or {}
    s = decisions.get("save", {}) or {}

    return {
        "game_pk": feed.get("gamePk"),
        "season": _to_int(game.get("season")),
        "game_date": dt.get("officialDate") or dt.get("originalDate"),
        "first_pitch_utc": dt.get("dateTime"),
        "day_night": dt.get("dayNight"),
        "game_type": game.get("type"),
        "double_header": game.get("doubleHeader"),
        "game_number": game.get("gameNumber"),
        "status": status.get("detailedState"),
        "status_code": status.get("statusCode"),
        # venue
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "venue_city": loc.get("city"),
        "venue_state": loc.get("stateAbbrev"),
        "roof_type": field.get("roofType"),
        "turf_type": field.get("turfType"),
        # teams
        "away_team_id": away_t.get("id"),
        "away_team_abbrev": away_t.get("abbreviation"),
        "away_team_name": away_t.get("name"),
        "home_team_id": home_t.get("id"),
        "home_team_abbrev": home_t.get("abbreviation"),
        "home_team_name": home_t.get("name"),
        # records (going into the game; MLB rolls these forward)
        "away_record_wins": away_rec.get("wins"),
        "away_record_losses": away_rec.get("losses"),
        "home_record_wins": home_rec.get("wins"),
        "home_record_losses": home_rec.get("losses"),
        # scoring
        "away_score": away_ls.get("runs"),
        "home_score": home_ls.get("runs"),
        "away_hits": away_ls.get("hits"),
        "home_hits": home_ls.get("hits"),
        "away_errors": away_ls.get("errors"),
        "home_errors": home_ls.get("errors"),
        "away_lob": away_ls.get("leftOnBase"),
        "home_lob": home_ls.get("leftOnBase"),
        "scheduled_innings": linescore.get("scheduledInnings"),
        "final_innings": linescore.get("currentInning"),
        # attendance / duration
        "attendance": ginfo.get("attendance"),
        "duration_minutes": ginfo.get("gameDurationMinutes"),
        # weather
        "weather_condition": weather.get("condition"),
        "weather_temp_f": _to_int(weather.get("temp")),
        "weather_wind": weather.get("wind"),
        # probables
        "probable_away_pitcher_id": (probable.get("away") or {}).get("id"),
        "probable_home_pitcher_id": (probable.get("home") or {}).get("id"),
        # decisions
        "winning_pitcher_id": w.get("id"),
        "losing_pitcher_id": l.get("id"),
        "save_pitcher_id": s.get("id"),
        # totals from boxscore (sanity)
        "away_team_avg": away_ts.get("avg"),
        "home_team_avg": home_ts.get("avg"),
    }


def build(merge: bool = True) -> Path:
    """Regenera games.parquet desde los raw JSONs de statsapi_feed.

    merge=True (default): mezcla los nuevos rows con el games.parquet existente.
    Necesario en CI donde solo se ingieren los feeds de los ultimos N dias — sin
    merge, la historia entera desaparece porque los feeds antiguos no viven en
    el repo (data/raw/ esta gitignored).

    merge=False: rebuild completo desde cero (uso local con archivo raw completo).
    """
    rows: list[dict] = []
    for season in SEASONS:
        d = RAW / "statsapi_feed" / str(season)
        files = sorted(d.glob("*.json"))
        for p in tqdm(files, desc=f"games {season}"):
            row = parse_feed(p)
            if row is not None:
                rows.append(row)
    new_df = pd.DataFrame(rows)
    if "game_pk" in new_df.columns:
        new_df = new_df.drop_duplicates("game_pk", keep="last")

    out = PROCESSED / "games.parquet"
    if merge and out.exists() and not new_df.empty:
        existing = pd.read_parquet(out)
        # Update-in-place: new rows win, existing rows kept for game_pks not
        # present in the fresh feeds (histórico se conserva).
        refreshed_pks = set(new_df["game_pk"].astype("int64").tolist())
        keep_mask = ~existing["game_pk"].astype("int64").isin(refreshed_pks)
        combined = pd.concat([existing[keep_mask], new_df], ignore_index=True)
        combined = combined.drop_duplicates("game_pk", keep="last")
        combined.to_parquet(out, index=False)
        print(f"wrote {out} ({len(combined):,} rows total; {len(new_df):,} refreshed, {int(keep_mask.sum()):,} preserved)")
    else:
        new_df.to_parquet(out, index=False)
        print(f"wrote {out} ({len(new_df):,} rows; full rebuild)")
    return out


if __name__ == "__main__":
    build()
