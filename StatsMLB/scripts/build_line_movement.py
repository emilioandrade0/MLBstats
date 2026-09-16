"""Build per-day line-movement JSONs from raw The Odds API snapshots.

For every raw payload in ``data/raw/odds_api/*.json`` we extract each
bookmaker's h2h prices at that timestamp, then aggregate per game a
compact time series so the UI can render sparkline movement charts.

Match strategy: The Odds API returns team names ("Detroit Tigers"),
which we map to MLB abbreviations + game_pk via ``games.parquet``
(same date + home/away team names).

Output: ``StatsMLB/public/data/line-movement/YYYY-MM-DD.json`` per game
date, plus an index ``line-movement-index.json`` listing available dates.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "odds_api"
ESPN_TS_PARQUET = ROOT / "data" / "processed" / "espn_odds_timeseries.parquet"
OUT_DIR = ROOT / "StatsMLB" / "public" / "data" / "line-movement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRACKED_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "caesars", "williamhill_us",
                 "espn_bet", "bet365", "caesars_sportsbook"]

# ESPN provider name → normalized book key (matches TRACKED_BOOKS keys where possible)
ESPN_BOOK_MAP = {
    "ESPN BET": "espn_bet",
    "DraftKings": "draftkings",
    "FanDuel": "fanduel",
    "BetMGM": "betmgm",
    "Caesars": "caesars",
    "Caesars Sportsbook": "caesars",
    "William Hill": "williamhill_us",
    "William Hill (New Jersey)": "williamhill_us",
    "Bet365": "bet365",
    "Pinnacle": "pinnacle",
}


def american_to_implied(price):
    if price is None:
        return None
    p = float(price)
    if p >= 100:
        return 100.0 / (p + 100.0)
    if p <= -100:
        return -p / (-p + 100.0)
    return None


def devig_pair(home_impl, away_impl):
    if home_impl is None or away_impl is None:
        return None
    total = home_impl + away_impl
    if total <= 0:
        return None
    return home_impl / total


def build_team_lookup(games: pd.DataFrame) -> dict:
    """(date_str, home_full_name, away_full_name) -> game_pk"""
    lookup = {}
    for _, g in games.iterrows():
        date_str = pd.to_datetime(g["game_date"]).strftime("%Y-%m-%d")
        key = (date_str, str(g["home_team_name"]), str(g["away_team_name"]))
        lookup[key] = {
            "game_pk": int(g["game_pk"]),
            "home_abbrev": str(g["home_team_abbrev"]),
            "away_abbrev": str(g["away_team_abbrev"]),
        }
    return lookup


def _ingest_espn_timeseries(aggregate, meta, games: pd.DataFrame) -> int:
    if not ESPN_TS_PARQUET.exists():
        return 0
    ts_df = pd.read_parquet(ESPN_TS_PARQUET)
    try:
        xref = pd.read_parquet(
            ROOT / "data" / "processed" / "games_xref.parquet",
            columns=["espn_event_id", "game_pk", "_merge"],
        )
        xref = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
        xref["game_pk"] = xref["game_pk"].astype(int)
        xref["espn_event_id"] = xref["espn_event_id"].astype(str)
    except Exception:
        return 0
    merged = ts_df.merge(xref, on="espn_event_id", how="inner")
    game_meta = games.set_index("game_pk")[["game_date", "home_team_abbrev", "away_team_abbrev"]].to_dict("index")
    added = 0
    for _, r in merged.iterrows():
        gpk = int(r["game_pk"])
        info = game_meta.get(gpk)
        if not info:
            continue
        game_date = info["game_date"].strftime("%Y-%m-%d")
        book = ESPN_BOOK_MAP.get(str(r["provider_name"]))
        if not book:
            continue
        meta[game_date][gpk] = {
            "commence": info["game_date"].isoformat() + "T00:00:00Z" if hasattr(info["game_date"], "isoformat") else "",
            "home": info["home_team_abbrev"],
            "away": info["away_team_abbrev"],
        }
        aggregate[game_date][gpk][book].append(
            (str(r["snapshot_ts"]), int(r["home_ml"]), int(r["away_ml"]))
        )
        added += 1
    return added


def main() -> None:
    games = pd.read_parquet(
        ROOT / "data" / "processed" / "games.parquet",
        columns=["game_pk", "game_date", "home_team_name", "away_team_name",
                 "home_team_abbrev", "away_team_abbrev"],
    )
    games["game_date"] = pd.to_datetime(games["game_date"])
    lookup = build_team_lookup(games)

    # Aggregate: date -> game_pk -> book -> list of (timestamp, home_ml, away_ml)
    aggregate = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta = defaultdict(dict)  # date -> game_pk -> {commence, home, away}

    espn_added = _ingest_espn_timeseries(aggregate, meta, games)
    if espn_added:
        print(f"[build_line_movement] +{espn_added:,} snapshots ESPN (gratis)")

    files = sorted(RAW_DIR.glob("*.json"))
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for event in payload:
            commence = event.get("commence_time")
            if not commence:
                continue
            game_date = commence[:10]
            home_name = event.get("home_team")
            away_name = event.get("away_team")
            hit = lookup.get((game_date, home_name, away_name))
            if not hit:
                continue
            game_pk = hit["game_pk"]
            meta[game_date][game_pk] = {
                "commence": commence,
                "home": hit["home_abbrev"],
                "away": hit["away_abbrev"],
            }
            for bk in event.get("bookmakers", []):
                key = bk.get("key")
                if key not in TRACKED_BOOKS:
                    continue
                last_update = bk.get("last_update")
                if not last_update:
                    continue
                for market in bk.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outs = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
                    home_ml = outs.get(home_name)
                    away_ml = outs.get(away_name)
                    if home_ml is None or away_ml is None:
                        continue
                    aggregate[game_date][game_pk][key].append(
                        (last_update, int(home_ml), int(away_ml))
                    )

    written = 0
    for date, games_dict in aggregate.items():
        out_games = {}
        for game_pk, books_dict in games_dict.items():
            game_books = {}
            all_home_impls = []
            for book, series in books_dict.items():
                # dedupe by timestamp (kepp last), sort ascending
                series_by_ts = {}
                for ts, hml, aml in series:
                    series_by_ts[ts] = (hml, aml)
                sorted_series = sorted(series_by_ts.items())
                series_out = []
                for ts, (hml, aml) in sorted_series:
                    devigged = devig_pair(american_to_implied(hml), american_to_implied(aml))
                    series_out.append([ts, hml, aml, round(devigged, 4) if devigged else None])
                    if devigged is not None:
                        all_home_impls.append((ts, devigged))
                game_books[book] = series_out

            # Compute open + latest home implied (median across books at each timestamp)
            if not all_home_impls:
                continue
            all_home_impls.sort(key=lambda x: x[0])
            opened_ts, opened_impl = all_home_impls[0]
            latest_ts, latest_impl = all_home_impls[-1]
            out_games[str(game_pk)] = {
                "away": meta[date][game_pk]["away"],
                "home": meta[date][game_pk]["home"],
                "commence": meta[date][game_pk]["commence"],
                "opened": opened_ts,
                "latest": latest_ts,
                "homeImpliedOpen": round(opened_impl, 4),
                "homeImpliedLatest": round(latest_impl, 4),
                "shiftPp": round((latest_impl - opened_impl) * 100, 2),
                "snapshots": len(all_home_impls),
                "books": game_books,
            }
        payload = {
            "date": date,
            "generatedAt": pd.Timestamp.now("UTC").isoformat(),
            "games": out_games,
        }
        (OUT_DIR / f"{date}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        written += 1

    # Index of available dates for the frontend
    dates = sorted(aggregate.keys())
    (OUT_DIR.parent / "line-movement-index.json").write_text(
        json.dumps({
            "generatedAt": pd.Timestamp.now("UTC").isoformat(),
            "dates": dates,
            "trackedBooks": TRACKED_BOOKS,
        }, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"line movement: {written} days -> {OUT_DIR}")


if __name__ == "__main__":
    main()
