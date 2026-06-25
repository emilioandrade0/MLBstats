"""Build plays.parquet — one row per plate appearance (PA) from StatsAPI."""
from __future__ import annotations

from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from .paths import RAW, PROCESSED, SEASONS


def _play_rows(feed: dict) -> list[dict]:
    game_pk = feed.get("gamePk")
    plays = (feed.get("liveData", {}).get("plays", {}) or {}).get("allPlays", []) or []
    out: list[dict] = []
    for play in plays:
        res = play.get("result") or {}
        ab = play.get("about") or {}
        cnt = play.get("count") or {}
        mu = play.get("matchup") or {}
        out.append({
            "game_pk": game_pk,
            "at_bat_index": ab.get("atBatIndex"),
            "inning": ab.get("inning"),
            "half_inning": ab.get("halfInning"),
            "is_top": ab.get("isTopInning"),
            "is_complete": ab.get("isComplete"),
            "is_scoring_play": ab.get("isScoringPlay"),
            "captivating_index": ab.get("captivatingIndex"),
            "start_time": ab.get("startTime"),
            "end_time": ab.get("endTime"),
            "result_type": res.get("type"),
            "event": res.get("event"),
            "event_type": res.get("eventType"),
            "description": res.get("description"),
            "rbi": res.get("rbi"),
            "away_score": res.get("awayScore"),
            "home_score": res.get("homeScore"),
            "is_out": res.get("isOut"),
            "balls": cnt.get("balls"),
            "strikes": cnt.get("strikes"),
            "outs": cnt.get("outs"),
            "batter_id": (mu.get("batter") or {}).get("id"),
            "bat_side": (mu.get("batSide") or {}).get("code"),
            "pitcher_id": (mu.get("pitcher") or {}).get("id"),
            "pitch_hand": (mu.get("pitchHand") or {}).get("code"),
            "menOnBase": (mu.get("splits") or {}).get("menOnBase"),
            "n_play_events": len(play.get("playEvents") or []),
        })
    return out


def build() -> Path:
    rows: list[dict] = []
    for season in SEASONS:
        d = RAW / "statsapi_feed" / str(season)
        files = sorted(d.glob("*.json"))
        for p in tqdm(files, desc=f"plays {season}"):
            try:
                feed = orjson.loads(p.read_bytes())
            except Exception:
                continue
            rows.extend(_play_rows(feed))
    df = pd.DataFrame(rows)
    out = PROCESSED / "plays.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)")
    return out


if __name__ == "__main__":
    build()
