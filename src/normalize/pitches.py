"""Build pitches.parquet — pitch-level Statcast from Savant.

Savant is the source of truth at pitch level (full 119-col Statcast suite).
We concat all per-week parquets, dedupe defensively, sort by (game_pk, at_bat_number, pitch_number).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .paths import RAW, PROCESSED, SEASONS


def build() -> Path:
    parts: list[pd.DataFrame] = []
    for season in SEASONS:
        d = RAW / "savant" / str(season)
        files = sorted(d.glob("*.parquet"))
        files = [f for f in files if f.stat().st_size > 0]
        for p in tqdm(files, desc=f"pitches {season}"):
            try:
                parts.append(pd.read_parquet(p))
            except Exception as e:
                print(f"[{p.name}] {e}")
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not df.empty:
        df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"], keep="last")
        df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="mergesort").reset_index(drop=True)
    out = PROCESSED / "pitches.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)")
    return out


if __name__ == "__main__":
    build()
