"""Inspect one real game from each source to confirm the field shape.

Picks a recent regular-season game (2024-07-20 had a normal slate) and prints:
  - ESPN summary top-level + boxscore structure + odds + plays shape
  - StatsAPI feed/live top-level + allPlays[0] shape + playEvents pitchData
  - Savant CSV columns

Run:
    python -m src.inspect_shapes
"""
from __future__ import annotations

from pathlib import Path

import orjson
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _peek(obj, label: str, depth: int = 0, max_depth: int = 2, max_keys: int = 20):
    pad = "  " * depth
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{pad}{label} dict[{len(keys)} keys]: {keys[:max_keys]}")
        if depth < max_depth:
            for k in keys[:max_keys]:
                _peek(obj[k], k, depth + 1, max_depth, max_keys)
    elif isinstance(obj, list):
        print(f"{pad}{label} list[{len(obj)}]")
        if obj and depth < max_depth:
            _peek(obj[0], "[0]", depth + 1, max_depth, max_keys)
    else:
        v = repr(obj)
        if len(v) > 60:
            v = v[:60] + "..."
        print(f"{pad}{label}: {type(obj).__name__} = {v}")


def pick_game() -> tuple[str, str]:
    """Pick a non-empty file from each source independently. IDs differ across sources."""
    espn = sorted((RAW / "summary" / "2024").glob("*.json"))
    sapi = sorted((RAW / "statsapi_feed" / "2024").glob("*.json"))
    # Pick mid-season to dodge spring training.
    return espn[len(espn) // 2].stem, sapi[len(sapi) // 2].stem


def main() -> None:
    eid, pk = pick_game()
    print(f"Inspecting event_id={eid} (== statsapi gamePk)\n")

    # ---- ESPN summary ----
    print("=" * 60)
    print("ESPN summary")
    print("=" * 60)
    espn = orjson.loads((RAW / "summary" / "2024" / f"{eid}.json").read_bytes())
    print(f"top-level keys: {list(espn.keys())}\n")
    for k in ("header", "boxscore", "winprobability", "predictor", "odds", "pickcenter", "againstTheSpread", "leaders", "injuries", "weather", "venue", "gameInfo", "plays", "rosters", "broadcasts", "format", "seasonseries"):
        if k in espn:
            v = espn[k]
            if isinstance(v, list):
                print(f"  {k}: list[{len(v)}]")
                if v:
                    _peek(v[0], "  [0]", depth=1, max_depth=1)
            elif isinstance(v, dict):
                print(f"  {k}: dict[{len(v)}] keys={list(v.keys())[:10]}")
            else:
                print(f"  {k}: {type(v).__name__}")

    print("\n-- ESPN plays[0] --")
    if espn.get("plays"):
        _peek(espn["plays"][0], "plays[0]", max_depth=2)

    print("\n-- ESPN boxscore.players[0].statistics[0] --")
    try:
        _peek(espn["boxscore"]["players"][0], "boxscore.players[0]", max_depth=3, max_keys=8)
    except Exception as e:
        print(f"  n/a: {e}")

    print("\n-- ESPN odds[0] --")
    if espn.get("odds"):
        _peek(espn["odds"][0], "odds[0]", max_depth=2)

    # ---- StatsAPI feed/live ----
    print("\n" + "=" * 60)
    print("StatsAPI feed/live")
    print("=" * 60)
    feed = orjson.loads((RAW / "statsapi_feed" / "2024" / f"{pk}.json").read_bytes())
    print(f"top-level keys: {list(feed.keys())}")
    print(f"gameData keys: {list(feed.get('gameData', {}).keys())}")
    print(f"liveData keys: {list(feed.get('liveData', {}).keys())}")

    print("\n-- gameData.weather / venue / probablePitchers --")
    gd = feed.get("gameData", {})
    for k in ("weather", "gameInfo", "probablePitchers", "venue", "datetime", "status", "teams"):
        if k in gd:
            _peek(gd[k], f"gameData.{k}", max_depth=1)

    print("\n-- liveData.linescore --")
    _peek(feed["liveData"].get("linescore", {}), "linescore", max_depth=1)

    print("\n-- liveData.boxscore.teams.home top keys --")
    try:
        home = feed["liveData"]["boxscore"]["teams"]["home"]
        print(f"  keys: {list(home.keys())}")
        print(f"  teamStats.batting keys: {list(home['teamStats']['batting'].keys())[:15]}")
        print(f"  teamStats.pitching keys: {list(home['teamStats']['pitching'].keys())[:15]}")
        # one player
        first_pid = next(iter(home["players"]))
        print(f"  players.{first_pid} keys: {list(home['players'][first_pid].keys())}")
        ps = home["players"][first_pid].get("stats", {})
        print(f"    stats keys: {list(ps.keys())}")
    except Exception as e:
        print(f"  n/a: {e}")

    print("\n-- liveData.plays.allPlays[0] --")
    try:
        ap = feed["liveData"]["plays"]["allPlays"][0]
        print(f"  keys: {list(ap.keys())}")
        print(f"  result: {ap.get('result')}")
        print(f"  about: {ap.get('about')}")
        print(f"  matchup keys: {list(ap.get('matchup', {}).keys())}")
        print(f"  count: {ap.get('count')}")
        pe = ap.get("playEvents", [])
        print(f"  playEvents: list[{len(pe)}]")
        if pe:
            # find a pitch event (has pitchData)
            pitch_ev = next((e for e in pe if "pitchData" in e), pe[0])
            print(f"\n  -- pitch event keys --")
            print(f"  {list(pitch_ev.keys())}")
            if "pitchData" in pitch_ev:
                print(f"  pitchData keys: {list(pitch_ev['pitchData'].keys())}")
                if "coordinates" in pitch_ev["pitchData"]:
                    print(f"    coordinates keys: {list(pitch_ev['pitchData']['coordinates'].keys())}")
                if "breaks" in pitch_ev["pitchData"]:
                    print(f"    breaks keys: {list(pitch_ev['pitchData']['breaks'].keys())}")
            if "hitData" in pitch_ev:
                print(f"  hitData keys: {list(pitch_ev['hitData'].keys())}")
            if "details" in pitch_ev:
                print(f"  details keys: {list(pitch_ev['details'].keys())}")
    except Exception as e:
        print(f"  n/a: {e}")

    # ---- Savant ----
    print("\n" + "=" * 60)
    print("Savant Statcast")
    print("=" * 60)
    savant_files = sorted((RAW / "savant" / "2024").glob("*.parquet"))
    savant_files = [f for f in savant_files if f.stat().st_size > 0]
    if savant_files:
        df = pd.read_parquet(savant_files[0])
        print(f"sample file: {savant_files[0].name}, rows={len(df)}, cols={len(df.columns)}")
        print(f"\ncolumns:")
        for col in df.columns:
            print(f"  {col} ({df[col].dtype})")


if __name__ == "__main__":
    main()
