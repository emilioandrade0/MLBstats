"""Export StrikeCast walkforward_preds.parquet to public/data/strikecast-history.json
so the StatsMLB calibrator can use it as a bucket dimension.

Run once, then re-run whenever the StrikeCast parquet updates:
  py scripts/export_strikecast_history.py

Assumes the StrikeCast repo lives one level up (../data/processed/walkforward_preds.parquet).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
STRIKECAST_PARQUET = REPO_ROOT.parent / "data" / "processed" / "walkforward_preds.parquet"
OUT_PATH = REPO_ROOT / "public" / "data" / "strikecast-history.json"

if not STRIKECAST_PARQUET.exists():
    raise SystemExit(f"parquet not found: {STRIKECAST_PARQUET}")

df = pd.read_parquet(STRIKECAST_PARQUET)
keep_cols = [c for c in ("game_pk", "game_date", "p_home_model_raw", "p_home", "market_p_home", "home_win") if c in df.columns]
df = df.loc[:, keep_cols].copy()

# Coerce types
df["game_pk"] = df["game_pk"].astype(int)
if "game_date" in df.columns:
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

# Drop rows with no raw model prob (can't use them)
if "p_home_model_raw" in df.columns:
    df = df.dropna(subset=["p_home_model_raw"])

# Round floats for smaller payload
for col in ("p_home_model_raw", "p_home", "market_p_home"):
    if col in df.columns:
        df[col] = df[col].astype(float).round(6)

# Emit as {gamePk: {p_raw, p_blend, market, home_win, date}}
games = {}
for row in df.itertuples(index=False):
    d = row._asdict()
    pk = str(d.pop("game_pk"))
    games[pk] = {
        "date": d.get("game_date"),
        "pRaw": d.get("p_home_model_raw"),
        "pBlend": d.get("p_home"),
        "market": d.get("market_p_home"),
        "homeWin": int(d["home_win"]) if pd.notna(d.get("home_win")) else None,
    }

payload = {
    "generatedAt": pd.Timestamp.utcnow().isoformat(),
    "source": str(STRIKECAST_PARQUET),
    "count": len(games),
    "games": games,
}
OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
print(f"Wrote {len(games)} entries to {OUT_PATH}")
