"""Pull current MLB odds from the-odds-api.com (includes Pinnacle).

Single call: fetches all upcoming MLB events with Pinnacle + a few US books.
Saves:
  - data/raw/odds_api/{YYYY-MM-DD_HHMM}.json   raw payload (timestamped)
  - data/processed/odds_pinnacle.parquet       latest snapshot, normalized

Cost: ~1 credit (1 region with bookmakers filter). Check remaining via stdout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import orjson
import pandas as pd

from .normalize.paths import RAW, PROCESSED
from .odds_api_client import OddsAPIClient


def _ml_h2h(market: dict, side_name: str) -> float | None:
    for o in market.get("outcomes") or []:
        if o.get("name") == side_name:
            v = o.get("price")
            return float(v) if v is not None else None
    return None


def _total(market: dict) -> tuple[float | None, float | None, float | None]:
    """Return (point, over_price, under_price) from a totals market."""
    pt = None
    over_p, under_p = None, None
    for o in market.get("outcomes") or []:
        nm = (o.get("name") or "").lower()
        if "over" in nm:
            over_p = o.get("price")
            pt = o.get("point")
        elif "under" in nm:
            under_p = o.get("price")
            pt = pt or o.get("point")
    return pt, (float(over_p) if over_p is not None else None), (float(under_p) if under_p is not None else None)


def _spread(market: dict, side_name: str) -> tuple[float | None, float | None]:
    """Return (handicap, price) for the given team's spread outcome."""
    for o in market.get("outcomes") or []:
        if o.get("name") == side_name:
            return (
                float(o["point"]) if o.get("point") is not None else None,
                float(o["price"]) if o.get("price") is not None else None,
            )
    return None, None


def normalize(payload: list) -> pd.DataFrame:
    """Explode events × bookmakers into long-form rows."""
    rows: list[dict] = []
    for ev in payload:
        ev_id = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        for bk in ev.get("bookmakers") or []:
            bk_key = bk.get("key")
            bk_title = bk.get("title")
            last_update = bk.get("last_update")
            row = {
                "event_id": ev_id,
                "commence_time": commence,
                "away_team": away,
                "home_team": home,
                "bookmaker": bk_key,
                "bookmaker_name": bk_title,
                "last_update": last_update,
                "home_ml": None, "away_ml": None,
                "total_point": None, "over_price": None, "under_price": None,
                "home_spread": None, "home_spread_price": None,
                "away_spread": None, "away_spread_price": None,
            }
            for mkt in bk.get("markets") or []:
                k = mkt.get("key")
                if k == "h2h":
                    row["home_ml"] = _ml_h2h(mkt, home)
                    row["away_ml"] = _ml_h2h(mkt, away)
                elif k == "totals":
                    pt, op, up = _total(mkt)
                    row["total_point"] = pt
                    row["over_price"] = op
                    row["under_price"] = up
                elif k == "spreads":
                    h_pt, h_pr = _spread(mkt, home)
                    a_pt, a_pr = _spread(mkt, away)
                    row["home_spread"] = h_pt
                    row["home_spread_price"] = h_pr
                    row["away_spread"] = a_pt
                    row["away_spread_price"] = a_pr
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", default="eu,us",
                        help="Comma-separated regions. 'eu' is needed for Pinnacle.")
    parser.add_argument("--bookmakers", default="pinnacle,draftkings,fanduel,betmgm,caesars,williamhill_us",
                        help="Set to '' to skip filter (costs more credits).")
    args = parser.parse_args()

    out_dir = RAW / "odds_api"
    out_dir.mkdir(parents=True, exist_ok=True)

    with OddsAPIClient() as c:
        payload, hdr = c.odds(
            regions=args.regions,
            bookmakers=args.bookmakers or None,
        )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    raw_path = out_dir / f"{ts}.json"
    raw_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    print(f"saved raw payload: {raw_path}")
    print(f"events returned:   {len(payload)}")
    print(f"credits — last call: {hdr.get('last_cost')}  used so far: {hdr.get('used')}  "
          f"remaining: {hdr.get('remaining')}")

    df = normalize(payload)
    pin = df[df["bookmaker"] == "pinnacle"]
    print(f"normalized rows:   {len(df):,}  (pinnacle rows: {len(pin):,})")

    out = PROCESSED / "odds_pinnacle.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}")

    # Quick visual: show first 5 events with Pinnacle moneyline
    if not pin.empty:
        print("\nSample Pinnacle moneylines:")
        print(pin[["commence_time", "away_team", "home_team", "away_ml", "home_ml",
                   "total_point", "home_spread"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
