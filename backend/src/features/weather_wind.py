"""Wind/weather features — scoring-environment signal.

Parses the StatsAPI wind string ("8 mph, Out To CF", "7 mph, In From LF",
"6 mph, L To R") into a directional scoring effect:
  wind_out_mph  — signed: +mph when blowing OUT (boosts offense/HR),
                          -mph when blowing IN (suppresses), 0 for crosswind
  wind_speed    — absolute wind speed (mph)
  is_indoor     — 1 if dome or roof closed (wind irrelevant)
  temp_f        — temperature (warm air carries the ball)

Game-level (both teams share the same conditions), so no _h/_a split.
Primary hypothesis: helps the TOTALS regressor (O/U), maybe win indirectly.

Output: data/processed/features_weather.parquet
  game_pk | wind_out_mph | wind_speed | is_indoor | temp_f

Run:
  python -m src.features.weather_wind
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = PROCESSED / "features_weather.parquet"


def _parse_wind(wind: str | None, roof: str | None) -> tuple[float, float]:
    """Return (wind_out_mph signed, wind_speed abs). Crosswind → out=0."""
    if not wind or pd.isna(wind):
        return 0.0, 0.0
    s = str(wind).strip()
    m = re.match(r"(\d+)\s*mph", s, re.IGNORECASE)
    speed = float(m.group(1)) if m else 0.0
    low = s.lower()
    if "out to" in low:
        return speed, speed          # blowing out → positive
    if "in from" in low:
        return -speed, speed         # blowing in → negative
    # "L To R", "R To L", or "None" → crosswind / neutral
    return 0.0, speed


def build() -> Path:
    g = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_type", "weather_wind",
                 "weather_temp_f", "roof_type"],
    )
    g = g[g["game_type"].isin(["R", "F", "D", "L", "W"])].copy()

    parsed = g.apply(
        lambda r: _parse_wind(r["weather_wind"], r["roof_type"]), axis=1
    )
    g["wind_out_mph"] = [p[0] for p in parsed]
    g["wind_speed"]   = [p[1] for p in parsed]
    # Indoor: dome always closed; retractable we can't be sure, treat as open.
    g["is_indoor"] = (g["roof_type"].astype(str).str.lower() == "dome").astype("int8")
    # Zero out wind effect indoors
    g.loc[g["is_indoor"] == 1, ["wind_out_mph", "wind_speed"]] = 0.0
    g["temp_f"] = pd.to_numeric(g["weather_temp_f"], errors="coerce")

    out = g[["game_pk", "wind_out_mph", "wind_speed", "is_indoor", "temp_f"]].copy()
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} rows)")
    print(f"  wind_out_mph: mean={out['wind_out_mph'].mean():.2f}  "
          f"out>0: {(out['wind_out_mph']>0).sum():,}  in<0: {(out['wind_out_mph']<0).sum():,}  "
          f"neutral: {(out['wind_out_mph']==0).sum():,}")
    print(f"  indoor games: {out['is_indoor'].sum():,}")
    print(f"  temp_f: mean={out['temp_f'].mean():.1f}  NaN={out['temp_f'].isna().sum():,}")
    return OUT


if __name__ == "__main__":
    build()
