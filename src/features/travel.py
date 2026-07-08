"""Travel & jetlag features — physical fatigue the market often underprices.

For each (game_pk, side) computes:
  travel_dist_miles       — great-circle miles from previous game's park to today's
  travel_tz_shift_east    — hours traveled east (body clock harder to advance)
  travel_tz_shift_west    — hours traveled west (easier to delay body clock)
  hours_since_last_game   — first-pitch UTC diff to previous game
  day_after_night_flag    — 1 if previous game was night AND today is day
  consecutive_road_games  — count of consecutive road games including today (0 if home)

Coordinates + timezone zone (0=Eastern, 1=Central, 2=Mountain, 3=Pacific) hardcoded
for the 30 MLB parks. Special venues (Field of Dreams, London Stadium etc.) fall
through to NaN so the model just ignores them.

Output: data/processed/features_travel.parquet
Key: (game_pk, side)

Run standalone:
  python -m src.features.travel
"""
from __future__ import annotations

from math import radians, sin, cos, asin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from ..normalize.paths import PROCESSED

OUT = PROCESSED / "features_travel.parquet"

# (city, state) → (lat, lon, tz_zone: 0=Eastern, 1=Central, 2=Mountain, 3=Pacific).
# tz_zone uses the DAYLIGHT-time offset — during MLB season (Mar–Oct) all US MLB
# cities observe DST except Phoenix (which sits at Pacific-equivalent offset in
# summer, so we mark it 3 to reflect the effective body-clock difference).
PARK_COORDS = {
    ("Bronx", "NY"):           (40.8296, -73.9262, 0),  # Yankee Stadium
    ("Flushing", "NY"):        (40.7571, -73.8458, 0),  # Citi Field
    ("Boston", "MA"):          (42.3467, -71.0972, 0),  # Fenway
    ("Baltimore", "MD"):       (39.2839, -76.6217, 0),  # Camden
    ("Philadelphia", "PA"):    (39.9061, -75.1665, 0),  # Citizens Bank
    ("Pittsburgh", "PA"):      (40.4469, -80.0057, 0),  # PNC
    ("Washington", "DC"):      (38.8730, -77.0074, 0),  # Nationals
    ("Toronto", "ON"):         (43.6414, -79.3894, 0),  # Rogers Centre
    ("Atlanta", "GA"):         (33.8908, -84.4678, 0),  # Truist
    ("Miami", "FL"):           (25.7781, -80.2197, 0),  # loanDepot
    ("St. Petersburg", "FL"):  (27.7683, -82.6534, 0),  # Tropicana
    ("Tampa", "FL"):           (27.9797, -82.5065, 0),  # Steinbrenner (spring training)
    ("Cleveland", "OH"):       (41.4962, -81.6852, 0),  # Progressive
    ("Cincinnati", "OH"):      (39.0975, -84.5080, 0),  # Great American
    ("Detroit", "MI"):         (42.3390, -83.0485, 0),  # Comerica

    ("Chicago", "IL"):         (41.8300, -87.6338, 1),  # Wrigley/Rate/Guaranteed (both Chicago)
    ("Milwaukee", "WI"):       (43.0280, -87.9712, 1),  # American Family
    ("Minneapolis", "MN"):     (44.9817, -93.2775, 1),  # Target
    ("Kansas City", "MO"):     (39.0517, -94.4803, 1),  # Kauffman
    ("St. Louis", "MO"):       (38.6226, -90.1928, 1),  # Busch
    ("Houston", "TX"):         (29.7573, -95.3555, 1),  # Minute Maid / Daikin
    ("Arlington", "TX"):       (32.7473, -97.0817, 1),  # Globe Life

    ("Denver", "CO"):          (39.7559, -104.9942, 2), # Coors
    ("Phoenix", "AZ"):         (33.4453, -112.0667, 3), # Chase — AZ stays Mountain year-round
                                                        # = effective Pacific in summer.

    ("Los Angeles", "CA"):     (34.0739, -118.2400, 3), # Dodger
    ("Anaheim", "CA"):         (33.8003, -117.8827, 3), # Angel
    ("San Diego", "CA"):       (32.7073, -117.1566, 3), # Petco
    ("San Francisco", "CA"):   (37.7786, -122.3893, 3), # Oracle
    ("Oakland", "CA"):         (37.7516, -122.2005, 3), # Coliseum
    ("Sacramento", "CA"):      (38.5807, -121.5133, 3), # Sutter Health
    ("Seattle", "WA"):         (47.5914, -122.3325, 3), # T-Mobile
}


def _haversine_miles(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(x) for x in (lat1, lon1, lat2, lon2)):
        return np.nan
    R = 3958.8  # earth radius in miles
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * R * asin(sqrt(a))


def build() -> Path:
    g = pd.read_parquet(PROCESSED / "games.parquet", columns=[
        "game_pk", "game_date", "first_pitch_utc", "game_type", "day_night",
        "home_team_abbrev", "away_team_abbrev",
        "venue_city", "venue_state",
    ])
    g = g[g["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    g["first_pitch_utc"] = pd.to_datetime(g["first_pitch_utc"], utc=True, errors="coerce")

    # Attach venue coordinates + timezone
    coords = g[["venue_city", "venue_state"]].apply(
        lambda r: PARK_COORDS.get((r["venue_city"], r["venue_state"]), (np.nan, np.nan, np.nan)),
        axis=1, result_type="expand",
    )
    coords.columns = ["lat", "lon", "tz"]
    g = pd.concat([g.reset_index(drop=True), coords.reset_index(drop=True)], axis=1)

    # Long format: one row per (team, game) so we can shift(1) per team
    home = g[["game_pk", "game_date", "first_pitch_utc", "day_night",
              "home_team_abbrev", "lat", "lon", "tz"]].copy()
    home.columns = ["game_pk", "game_date", "first_pitch_utc", "day_night",
                    "team", "lat", "lon", "tz"]
    home["side"] = "home"
    home["is_home"] = 1
    away = g[["game_pk", "game_date", "first_pitch_utc", "day_night",
              "away_team_abbrev", "lat", "lon", "tz"]].copy()
    away.columns = ["game_pk", "game_date", "first_pitch_utc", "day_night",
                    "team", "lat", "lon", "tz"]
    away["side"] = "away"
    away["is_home"] = 0
    tg = pd.concat([home, away], ignore_index=True)
    tg = tg.sort_values(["team", "first_pitch_utc", "game_pk"]).reset_index(drop=True)

    grp = tg.groupby("team", sort=False)
    tg["prev_lat"] = grp["lat"].shift(1)
    tg["prev_lon"] = grp["lon"].shift(1)
    tg["prev_tz"]  = grp["tz"].shift(1)
    tg["prev_fp"]  = grp["first_pitch_utc"].shift(1)
    tg["prev_daynight"] = grp["day_night"].shift(1)
    tg["prev_is_home"]  = grp["is_home"].shift(1)

    # Great-circle miles from prev park
    tg["travel_dist_miles"] = tg.apply(
        lambda r: _haversine_miles(r["prev_lat"], r["prev_lon"], r["lat"], r["lon"]),
        axis=1,
    )
    # Time zone shift: positive = went east, negative = went west
    tz_shift = tg["tz"] - tg["prev_tz"]  # e.g. Pacific(3)→Eastern(0) = -3, i.e. traveled EAST by 3
    tg["travel_tz_shift_east"] = (-tz_shift).clip(lower=0)  # 0 if not east
    tg["travel_tz_shift_west"] = tz_shift.clip(lower=0)     # 0 if not west
    # Hours since last game (via first_pitch_utc)
    tg["hours_since_last_game"] = (
        (tg["first_pitch_utc"] - tg["prev_fp"]).dt.total_seconds() / 3600.0
    )
    # Day-after-night: prev was night, today is day
    tg["day_after_night_flag"] = (
        (tg["prev_daynight"].astype(str).str.lower() == "night") &
        (tg["day_night"].astype(str).str.lower() == "day")
    ).astype(int)

    # Consecutive road games including today (0 if at home)
    def _road_streak(is_home_series: pd.Series) -> pd.Series:
        # 1 - is_home to get road flags; cumulative streak counter
        r = (1 - is_home_series.fillna(1)).astype(int)
        streak = np.zeros(len(r), dtype=int)
        count = 0
        for i, v in enumerate(r.values):
            count = count + 1 if v == 1 else 0
            streak[i] = count
        return pd.Series(streak, index=r.index)

    tg["consecutive_road_games"] = grp["is_home"].transform(_road_streak)

    # Reduced set — round-1 test with all 6 showed +0.07pp acc, -0.22pp AUC,
    # and none of 18 columns cracked top-30 importance. Keeping only the 3 with
    # the clearest physical motive (fatigue: distance + consecutive road + short
    # turnaround). tz_shift and day_after_night dropped as sparse/redundant.
    out = tg[["game_pk", "side",
              "travel_dist_miles",
              "hours_since_last_game",
              "consecutive_road_games"]].copy()

    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(out):,} rows)")
    print()
    print("Distribuciones:")
    for c in ["travel_dist_miles", "hours_since_last_game", "consecutive_road_games"]:
        s = out[c].dropna()
        print(f"  {c:28s}  mean={s.mean():.2f}  p50={s.median():.2f}  p90={s.quantile(0.9):.2f}  max={s.max():.2f}")

    # Signal check: home_win vs travel_dist_miles for AWAY team
    games = pd.read_parquet(PROCESSED / "games.parquet",
                             columns=["game_pk", "home_score", "away_score"]).dropna()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    away_travel = out[out["side"] == "away"][["game_pk", "travel_dist_miles"]]
    chk = games.merge(away_travel, on="game_pk")
    chk = chk.dropna(subset=["travel_dist_miles"])
    print(f"\nSignal check (away team travel vs home_win, n={len(chk):,}):")
    chk["dist_bucket"] = pd.cut(chk["travel_dist_miles"],
                                 [-1, 100, 500, 1000, 1500, 5000],
                                 labels=["<100mi (in-city/rest)", "100-500", "500-1000",
                                         "1000-1500", "1500+"])
    print(chk.groupby("dist_bucket", observed=False)["home_win"].agg(["mean", "count"]).round(4).to_string())
    print("\n(higher home_win = away team likely tired)")
    return OUT


if __name__ == "__main__":
    build()
