"""Pull current NFL odds from the-odds-api.com (Pinnacle + US books).

Outputs:
  data/raw/odds_api/{YYYY-MM-DDTHHMMSSZ}.json   raw timestamped snapshot
  data/processed/odds_api_latest.parquet         normalized long-form
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import orjson
import pandas as pd

from ..paths import PROCESSED, RAW
from .odds_api_client import OddsAPIClient


def _h2h(mkt: dict, side: str) -> float | None:
    for o in mkt.get("outcomes") or []:
        if o.get("name") == side:
            v = o.get("price")
            return float(v) if v is not None else None
    return None


def _totals(mkt: dict) -> tuple[float | None, float | None, float | None]:
    pt = op = up = None
    for o in mkt.get("outcomes") or []:
        nm = (o.get("name") or "").lower()
        if "over" in nm:
            op = o.get("price"); pt = o.get("point")
        elif "under" in nm:
            up = o.get("price"); pt = pt or o.get("point")
    return pt, (float(op) if op is not None else None), (float(up) if up is not None else None)


def _spread(mkt: dict, side: str) -> tuple[float | None, float | None]:
    for o in mkt.get("outcomes") or []:
        if o.get("name") == side:
            return (
                float(o["point"]) if o.get("point") is not None else None,
                float(o["price"]) if o.get("price") is not None else None,
            )
    return None, None


def normalize(payload: list) -> pd.DataFrame:
    rows: list[dict] = []
    for ev in payload:
        home, away = ev.get("home_team"), ev.get("away_team")
        for bk in ev.get("bookmakers") or []:
            row = {
                "event_id": ev.get("id"),
                "commence_time": ev.get("commence_time"),
                "away_team": away, "home_team": home,
                "bookmaker": bk.get("key"), "bookmaker_name": bk.get("title"),
                "last_update": bk.get("last_update"),
                "home_ml": None, "away_ml": None,
                "total_point": None, "over_price": None, "under_price": None,
                "home_spread": None, "home_spread_price": None,
                "away_spread": None, "away_spread_price": None,
            }
            for m in bk.get("markets") or []:
                k = m.get("key")
                if k == "h2h":
                    row["home_ml"] = _h2h(m, home); row["away_ml"] = _h2h(m, away)
                elif k == "totals":
                    row["total_point"], row["over_price"], row["under_price"] = _totals(m)
                elif k == "spreads":
                    row["home_spread"], row["home_spread_price"] = _spread(m, home)
                    row["away_spread"], row["away_spread_price"] = _spread(m, away)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="eu,us")
    ap.add_argument("--bookmakers",
                    default="pinnacle,draftkings,fanduel,betmgm,caesars,williamhill_us")
    args = ap.parse_args()

    out_dir = RAW / "odds_api"
    out_dir.mkdir(parents=True, exist_ok=True)

    with OddsAPIClient() as c:
        payload, hdr = c.odds(regions=args.regions, bookmakers=args.bookmakers or None)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    raw_path = out_dir / f"{ts}.json"
    raw_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    print(f"raw:      {raw_path.name}")
    print(f"events:   {len(payload)}")
    print(f"credits — last: {hdr.get('last_cost')}  used: {hdr.get('used')}  "
          f"remaining: {hdr.get('remaining')}")

    df = normalize(payload)
    out = PROCESSED / "odds_api_latest.parquet"
    df.to_parquet(out, index=False)
    pin = df[df["bookmaker"] == "pinnacle"]
    print(f"rows:     {len(df):,}  (pinnacle: {len(pin):,})")
    print(f"wrote:    {out}")


if __name__ == "__main__":
    main()
