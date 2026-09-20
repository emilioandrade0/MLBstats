"""Emit per-date NFL line-movement JSONs from nfl_odds_timeseries.parquet.

Each file at ``StatsMLB/public/data/nfl/line-movement/YYYY-MM-DD.json`` contains
the games kicking off that date with each tracked book's snapshot series
(moneyline, spread, total). The frontend reads them to draw the movement
panel similar to MLB.

Book IDs are Action Network's integer identifiers; the file also carries a
labels map so the UI shows human-readable names.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "processed" / "nfl_odds_timeseries.parquet"
OUT_DIR = ROOT / "StatsMLB" / "public" / "data" / "nfl" / "line-movement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOOK_LABELS = {
    15: "DraftKings", 30: "FanDuel", 68: "BetMGM", 69: "Caesars",
    75: "Pinnacle", 71: "William Hill", 236: "ESPN BET", 76: "PointsBet",
    272: "Fanatics", 274: "bet365", 79: "BetRivers",
}


def american_to_implied(price) -> float | None:
    if price is None or pd.isna(price):
        return None
    p = float(price)
    if p == 0:
        return None
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


def main() -> None:
    if not SRC.exists():
        print(f"[build_nfl_line_movement] source parquet missing: {SRC}")
        return
    df = pd.read_parquet(SRC)
    if df.empty:
        print("[build_nfl_line_movement] parquet vacio")
        return

    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["kickoff_utc"])
    df["game_date"] = df["kickoff_utc"].dt.strftime("%Y-%m-%d")

    written = 0
    for date, group in df.groupby("game_date"):
        games_out: dict[str, dict] = {}
        for gid, gg in group.groupby("action_game_id"):
            first_row = gg.iloc[0]
            home = str(first_row["home_team"])
            away = str(first_row["away_team"])
            commence = first_row["kickoff_utc"].isoformat()
            books_out: dict[str, list] = {}
            all_home_impls: list[tuple[str, float]] = []
            all_spreads: list[tuple[str, float]] = []
            all_totals: list[tuple[str, float]] = []
            for bk, bg in gg.groupby("book_id"):
                by_ts: dict[str, tuple] = {}
                for _, r in bg.iterrows():
                    ts = str(r["snapshot_ts"])
                    hml = None if pd.isna(r["home_ml"]) else int(r["home_ml"])
                    aml = None if pd.isna(r["away_ml"]) else int(r["away_ml"])
                    hs = None if pd.isna(r["home_spread"]) else float(r["home_spread"])
                    tt = None if pd.isna(r["total"]) else float(r["total"])
                    by_ts[ts] = (hml, aml, hs, tt)
                series = []
                for ts, (hml, aml, hs, tt) in sorted(by_ts.items()):
                    devig = devig_pair(american_to_implied(hml), american_to_implied(aml))
                    series.append([ts, hml, aml, round(devig, 4) if devig else None, hs, tt])
                    if devig is not None:
                        all_home_impls.append((ts, devig))
                    if hs is not None:
                        all_spreads.append((ts, hs))
                    if tt is not None:
                        all_totals.append((ts, tt))
                books_out[str(int(bk))] = series
            if not all_home_impls:
                continue
            all_home_impls.sort(key=lambda x: x[0])
            all_spreads.sort(key=lambda x: x[0])
            all_totals.sort(key=lambda x: x[0])
            opened_ts, opened_impl = all_home_impls[0]
            latest_ts, latest_impl = all_home_impls[-1]
            games_out[str(int(gid))] = {
                "away": away,
                "home": home,
                "commence": commence,
                "opened": opened_ts,
                "latest": latest_ts,
                "homeImpliedOpen": round(opened_impl, 4),
                "homeImpliedLatest": round(latest_impl, 4),
                "shiftPp": round((latest_impl - opened_impl) * 100, 2),
                "spreadOpen": all_spreads[0][1] if all_spreads else None,
                "spreadLatest": all_spreads[-1][1] if all_spreads else None,
                "totalOpen": all_totals[0][1] if all_totals else None,
                "totalLatest": all_totals[-1][1] if all_totals else None,
                "snapshots": len(all_home_impls),
                "books": books_out,
            }
        if not games_out:
            continue
        payload = {
            "date": date,
            "generatedAt": pd.Timestamp.now("UTC").isoformat(),
            "bookLabels": BOOK_LABELS,
            "games": games_out,
        }
        (OUT_DIR / f"{date}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        written += 1

    (OUT_DIR.parent / "line-movement-index.json").write_text(
        json.dumps({
            "generatedAt": pd.Timestamp.now("UTC").isoformat(),
            "dates": sorted(df["game_date"].unique().tolist()),
        }, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"[build_nfl_line_movement] {written} fechas -> {OUT_DIR}")


if __name__ == "__main__":
    main()
