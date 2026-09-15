"""Per-date historical snapshots for the game cards.

For every date in the season (2026 by default) we freeze what was known BEFORE
that date:
  * Player performance L30: aggregated from completed games in
    ``[date - windowDays, date - 1]``.
  * Team lineup state: the most recent game each team played strictly before
    the target date, with its player IDs, batting slots and top 4.

Each date lands in ``StatsMLB/public/data/history/YYYY-MM-DD.json`` so the UI
can fetch on demand when the user browses back through the slate tabs.  The
current-day snapshot is also written alongside so it stays consistent with the
existing ``player-performance.json`` / lineup state consumers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "StatsMLB" / "public" / "data" / "history"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WINDOW_DAYS = 30
SEASON = 2026
MIN_BAT_PA = 15
MIN_PIT_BF = 20


def label_batter(ops, k_pct, avg):
    if ops is None or (isinstance(ops, float) and np.isnan(ops)):
        return "sin datos"
    if ops >= .820 and (k_pct is not None and not np.isnan(k_pct) and k_pct <= .28):
        return "racha caliente"
    if ops >= .720:
        return "sólido"
    if ops <= .580 or (k_pct is not None and not np.isnan(k_pct) and k_pct >= .32 and avg is not None and not np.isnan(avg) and avg <= .200):
        return "racha fría"
    return "normal"


def label_pitcher(era, whip):
    if era is None or (isinstance(era, float) and np.isnan(era)):
        return "sin datos"
    if era <= 3.20 and whip is not None and not np.isnan(whip) and whip <= 1.15:
        return "dominante"
    if era <= 4.20:
        return "sólido"
    if era >= 5.50 or (whip is not None and not np.isnan(whip) and whip >= 1.50):
        return "en problemas"
    return "normal"


def ip_to_outs(value):
    if pd.isna(value):
        return 0
    s = str(value)
    try:
        whole = int(s.split(".")[0])
        frac = int(s.split(".")[1]) if "." in s else 0
    except ValueError:
        return 0
    return whole * 3 + min(frac, 2)


def aggregate_batters(frame):
    if frame.empty:
        return {}
    agg = frame.groupby(["player_id", "player_name"], dropna=True).agg(
        pa=("bat_plateAppearances", "sum"),
        ab=("bat_atBats", "sum"),
        h=("bat_hits", "sum"),
        d=("bat_doubles", "sum"),
        t=("bat_triples", "sum"),
        hr=("bat_homeRuns", "sum"),
        bb=("bat_baseOnBalls", "sum"),
        so=("bat_strikeOuts", "sum"),
        rbi=("bat_rbi", "sum"),
        runs=("bat_runs", "sum"),
        sb=("bat_stolenBases", "sum"),
        games=("game_pk", "nunique"),
    ).reset_index()
    agg = agg[agg["pa"] >= MIN_BAT_PA].copy()
    agg["avg"] = np.where(agg["ab"] > 0, agg["h"] / agg["ab"], np.nan)
    tb = agg["h"] + agg["d"] + 2 * agg["t"] + 3 * agg["hr"]
    agg["slg"] = np.where(agg["ab"] > 0, tb / agg["ab"], np.nan)
    obp_num = agg["h"] + agg["bb"]
    obp_den = agg["ab"] + agg["bb"]
    agg["obp"] = np.where(obp_den > 0, obp_num / obp_den, np.nan)
    agg["ops"] = agg["obp"] + agg["slg"]
    agg["kPct"] = np.where(agg["pa"] > 0, agg["so"] / agg["pa"], np.nan)
    agg["bbPct"] = np.where(agg["pa"] > 0, agg["bb"] / agg["pa"], np.nan)
    out = {}
    for _, row in agg.iterrows():
        pid = str(int(row["player_id"]))
        ops = float(row["ops"]) if not np.isnan(row["ops"]) else float("nan")
        k = float(row["kPct"]) if not np.isnan(row["kPct"]) else float("nan")
        avg = float(row["avg"]) if not np.isnan(row["avg"]) else float("nan")
        out[pid] = {
            "kind": "batter",
            "name": str(row["player_name"]),
            "games": int(row["games"]),
            "pa": int(row["pa"]),
            "avg": None if np.isnan(row["avg"]) else round(float(row["avg"]), 3),
            "ops": None if np.isnan(row["ops"]) else round(float(row["ops"]), 3),
            "hr": int(row["hr"]),
            "rbi": int(row["rbi"]),
            "runs": int(row["runs"]),
            "sb": int(row["sb"]),
            "kPct": None if np.isnan(row["kPct"]) else round(float(row["kPct"]), 3),
            "bbPct": None if np.isnan(row["bbPct"]) else round(float(row["bbPct"]), 3),
            "label": label_batter(ops, k, avg),
        }
    return out


def aggregate_pitchers(frame):
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["outs"] = frame["pit_inningsPitched"].map(ip_to_outs)
    agg = frame.groupby(["player_id", "player_name"], dropna=True).agg(
        bf=("pit_battersFaced", "sum"),
        outs=("outs", "sum"),
        h=("pit_hits", "sum"),
        er=("pit_earnedRuns", "sum"),
        bb=("pit_baseOnBalls", "sum"),
        so=("pit_strikeOuts", "sum"),
        hr=("pit_homeRuns", "sum"),
        games=("game_pk", "nunique"),
    ).reset_index()
    agg = agg[agg["bf"] >= MIN_PIT_BF].copy()
    agg["ip"] = agg["outs"] / 3.0
    agg["era"] = np.where(agg["outs"] > 0, (agg["er"] * 27) / agg["outs"], np.nan)
    agg["whip"] = np.where(agg["outs"] > 0, ((agg["bb"] + agg["h"]) * 3) / agg["outs"], np.nan)
    agg["k9"] = np.where(agg["outs"] > 0, (agg["so"] * 27) / agg["outs"], np.nan)
    agg["bb9"] = np.where(agg["outs"] > 0, (agg["bb"] * 27) / agg["outs"], np.nan)
    out = {}
    for _, row in agg.iterrows():
        pid = str(int(row["player_id"]))
        era = float(row["era"]) if not np.isnan(row["era"]) else float("nan")
        whip = float(row["whip"]) if not np.isnan(row["whip"]) else float("nan")
        out[pid] = {
            "kind": "pitcher",
            "name": str(row["player_name"]),
            "games": int(row["games"]),
            "ip": None if np.isnan(row["ip"]) else round(float(row["ip"]), 1),
            "era": None if np.isnan(era) else round(float(era), 2),
            "whip": None if np.isnan(whip) else round(float(whip), 2),
            "k9": None if np.isnan(row["k9"]) else round(float(row["k9"]), 1),
            "bb9": None if np.isnan(row["bb9"]) else round(float(row["bb9"]), 1),
            "so": int(row["so"]),
            "hr": int(row["hr"]),
            "label": label_pitcher(era, whip),
        }
    return out


def main():
    pb = pd.read_parquet(ROOT / "data" / "processed" / "player_box.parquet")
    games = pd.read_parquet(
        ROOT / "data" / "processed" / "games.parquet",
        columns=["game_pk", "game_date", "season", "away_team_abbrev", "home_team_abbrev", "away_score", "home_score", "status"],
    )
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.tz_localize(None)
    games["date"] = games["game_date"].dt.normalize()

    pb = pb.merge(games[["game_pk", "date", "away_team_abbrev", "home_team_abbrev"]], on="game_pk", how="inner")
    pb["team"] = np.where(pb["side"].astype(str).str.lower() == "home", pb["home_team_abbrev"], pb["away_team_abbrev"])

    # Team lineup states per game: for each (game_pk, team) get player IDs of starters.
    starters = pb[pb["started_batting"].fillna(False).astype(bool) & ~pb["position"].astype(str).eq("P")].copy()
    starters["slot"] = (pd.to_numeric(starters["batting_order"], errors="coerce") // 100).astype("Int64")
    team_games = []
    games_indexed = games.set_index("game_pk")
    for (game_pk, team), group in starters.groupby(["game_pk", "team"], sort=False):
        if pd.isna(team) or group["slot"].dropna().empty:
            continue
        gd = group["date"].iloc[0]
        players = sorted(int(x) for x in group["player_id"].dropna().astype(int))
        slots = {str(int(row["player_id"])): int(row["slot"]) for _, row in group.dropna(subset=["player_id", "slot"]).iterrows()}
        top4 = sorted(int(x) for x in group.loc[group["slot"].le(4), "player_id"].dropna().astype(int))
        try:
            g_row = games_indexed.loc[game_pk]
        except KeyError:
            continue
        away_score, home_score = g_row["away_score"], g_row["home_score"]
        if pd.isna(away_score) or pd.isna(home_score):
            won = 0
        elif team == g_row["away_team_abbrev"]:
            won = int(away_score > home_score)
        else:
            won = int(home_score > away_score)
        team_games.append({
            "team": str(team),
            "date": gd,
            "game_pk": int(game_pk),
            "win": won,
            "players": players,
            "slots": slots,
            "top4": top4,
        })
    team_frame = pd.DataFrame(team_games).sort_values(["team", "date", "game_pk"]).reset_index(drop=True)

    # Dates to snapshot: every 2026 date with at least one game
    season_games = games[games["season"] == SEASON].copy()
    snapshot_dates = sorted(season_games["date"].dropna().unique())

    all_dates_pb = pb.copy()
    all_dates_pb["date"] = pd.to_datetime(all_dates_pb["date"]).dt.normalize()

    written = 0
    for target_date in snapshot_dates:
        cutoff_end = target_date - pd.Timedelta(days=1)
        cutoff_start = target_date - pd.Timedelta(days=WINDOW_DAYS)
        window = all_dates_pb[(all_dates_pb["date"] >= cutoff_start) & (all_dates_pb["date"] <= cutoff_end)]
        batters = window[window["appeared_batting"].fillna(False).astype(bool)]
        pitchers = window[window["appeared_pitching"].fillna(False).astype(bool)]
        players_out = {**aggregate_batters(batters), **aggregate_pitchers(pitchers)}

        # Team states: last game per team strictly before target_date
        team_states = {}
        past = team_frame[team_frame["date"] < target_date]
        for team, group in past.groupby("team", sort=False):
            last = group.iloc[-1]
            team_states[str(team)] = {
                "gamePk": int(last["game_pk"]),
                "playedAt": last["date"].isoformat(),
                "win": int(last["win"]),
                "players": last["players"],
                "slots": last["slots"],
                "top4": last["top4"],
            }

        payload = {
            "date": target_date.strftime("%Y-%m-%d"),
            "windowDays": WINDOW_DAYS,
            "cutoffDate": cutoff_end.strftime("%Y-%m-%d"),
            "generatedAt": pd.Timestamp.now("UTC").isoformat(),
            "players": players_out,
            "teamStates": team_states,
        }
        path = OUT_DIR / f"{target_date.strftime('%Y-%m-%d')}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        written += 1

    # Manifest for the frontend (list of available dates)
    manifest = {
        "generatedAt": pd.Timestamp.now("UTC").isoformat(),
        "season": SEASON,
        "windowDays": WINDOW_DAYS,
        "dates": [d.strftime("%Y-%m-%d") for d in snapshot_dates],
    }
    (OUT_DIR.parent / "history-index.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"history: {written} snapshots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
