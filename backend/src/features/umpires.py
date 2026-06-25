"""Umpire features — home-plate umpire's prior-season accuracy + run-impact.

Pipeline:
  1. Walk all cached statsapi_feed/*.json → extract HP umpire per game
  2. Fetch per-season umpire aggregates from umpscorecards.com
  3. Per game: HP umpire's PRIOR season stats (no leakage)

Output:
  data/processed/features_umpire_assignment.parquet   game_pk → hp_ump_name
  data/processed/features_umpire.parquet              game_pk → ump_*_priorseason features
"""
from __future__ import annotations

from pathlib import Path

import httpx
import orjson
import pandas as pd
from tqdm import tqdm

from ..normalize.paths import PROCESSED, RAW

UMP_API = "https://umpscorecards.com/api/umpires"


def _hp_umpire(feed: dict) -> str | None:
    """HP umpire lives at liveData.boxscore.officials (not gameData.officials)."""
    box = (feed.get("liveData") or {}).get("boxscore") or {}
    for o in (box.get("officials") or []):
        if (o.get("officialType") or "").lower() == "home plate":
            ump = o.get("official") or {}
            return ump.get("fullName")
    return None


def build_assignment() -> Path:
    rows: list[dict] = []
    for season in (2023, 2024, 2025, 2026):
        d = RAW / "statsapi_feed" / str(season)
        if not d.exists():
            continue
        for p in tqdm(sorted(d.glob("*.json")), desc=f"ump assign {season}"):
            try:
                feed = orjson.loads(p.read_bytes())
            except Exception:
                continue
            rows.append({
                "game_pk": int(p.stem),
                "season": season,
                "hp_umpire": _hp_umpire(feed),
            })
    df = pd.DataFrame(rows)
    out = PROCESSED / "features_umpire_assignment.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows, {df['hp_umpire'].notna().sum():,} with HP ump)")
    return out


def fetch_season(season: int) -> pd.DataFrame:
    """Per-umpire season aggregates (regular season)."""
    r = httpx.get(
        UMP_API,
        params={"startDate": f"{season}-01-01",
                "endDate":   f"{season}-12-31",
                "seasonType": "R"},
        timeout=60.0,
    )
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("rows") or []
    df = pd.DataFrame(rows)
    keep = [
        "umpire", "n",
        "overall_accuracy_wmean", "accuracy_above_x_wmean",
        "consistency_wmean", "total_run_impact_mean",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(columns={
        "umpire": "hp_umpire",
        "n": "ump_games",
        "overall_accuracy_wmean": "ump_acc",
        "accuracy_above_x_wmean": "ump_acc_above_x",
        "consistency_wmean": "ump_consistency",
        "total_run_impact_mean": "ump_run_impact",
    })
    df["season"] = season
    # Normalize accuracy to 0-1 if in 0-100 range
    for c in ("ump_acc", "ump_consistency"):
        if c in df.columns and df[c].notna().any() and df[c].max() > 1.5:
            df[c] = df[c] / 100.0
    return df


def build_features() -> Path:
    seasons = sorted(pd.read_parquet(PROCESSED / "features_umpire_assignment.parquet")["season"].unique())
    print(f"fetching umpscorecards for seasons {seasons}…")
    season_frames: list[pd.DataFrame] = []
    for s in seasons:
        try:
            season_frames.append(fetch_season(int(s)))
        except Exception as e:
            print(f"  [{s}] skipped: {e}")
    ump = pd.concat(season_frames, ignore_index=True)

    # Use PRIOR season as the feature for this game (no leakage).
    ump_prior = ump.copy()
    ump_prior["season"] = ump_prior["season"] + 1

    assign = pd.read_parquet(PROCESSED / "features_umpire_assignment.parquet")
    df = assign.merge(ump_prior, on=["season", "hp_umpire"], how="left")
    keep = ["game_pk", "ump_acc", "ump_acc_above_x", "ump_consistency",
            "ump_run_impact", "ump_games"]
    out = PROCESSED / "features_umpire.parquet"
    df[keep].to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows, "
          f"{df['ump_acc'].notna().sum():,} with prior-season ump stats)")
    return out


if __name__ == "__main__":
    import sys
    if "--assign" in sys.argv or "--all" in sys.argv:
        build_assignment()
    if "--features" in sys.argv or "--all" in sys.argv:
        build_features()
    if not any(a.startswith("--") for a in sys.argv[1:]):
        build_assignment()
        build_features()
