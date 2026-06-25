"""Build win_probability.parquet — ESPN win-prob frames per play."""
from __future__ import annotations

from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from .paths import RAW, PROCESSED, SEASONS


def build() -> Path:
    rows: list[dict] = []
    for season in SEASONS:
        d = RAW / "summary" / str(season)
        files = sorted(d.glob("*.json"))
        for p in tqdm(files, desc=f"win_prob {season}"):
            try:
                summ = orjson.loads(p.read_bytes())
            except Exception:
                continue
            for w in summ.get("winprobability") or []:
                rows.append({
                    "espn_event_id": p.stem,
                    "season": season,
                    "play_id": w.get("playId"),
                    "home_win_pct": w.get("homeWinPercentage"),
                    "tie_pct": w.get("tiePercentage"),
                })
    df = pd.DataFrame(rows)
    out = PROCESSED / "win_probability.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)")
    return out


if __name__ == "__main__":
    build()
