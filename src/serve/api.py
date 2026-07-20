"""FastAPI backend.

Endpoints:
  GET /api/games?start=YYYY-MM-DD&end=YYYY-MM-DD   list of games with prediction
  GET /api/games/{game_pk}                          full matchup detail

Run:
  python -m uvicorn src.serve.api:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import orjson
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles

from ..normalize.paths import PROCESSED, RAW

MODELS = PROCESSED.parent / "models"
STATIC_DIR = Path(__file__).parent / "static"
STATSAPI_LIVE_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

# Background refresh interval. Override with env var if needed.
REFRESH_INTERVAL_SEC = int(os.environ.get("STRIKECAST_REFRESH_SEC", "3600"))  # 1 hour
# Minimum gap between on-request refreshes (lower than background interval —
# we want pageloads to feel snappy without hammering ESPN).
REFRESH_ON_REQUEST_TTL_SEC = int(os.environ.get("STRIKECAST_ONREQ_TTL_SEC", "120"))  # 2 min
_last_refresh = {"ok": 0.0, "started": 0.0, "running": False, "err": None}
_refresh_lock = asyncio.Lock()
_freshness_cache = {"ts": 0.0, "data": None}
_FRESHNESS_TTL_SEC = 60


async def _maybe_refresh_now(blocking: bool = False) -> None:
    """Trigger a data refresh if stale.

    blocking=False (default): fire-and-forget. Response uses whatever's already
    cached; the next request after the refresh completes will see fresh data.
    blocking=True: legacy behavior — wait for refresh to finish before
    returning.

    Throttled to one fetch every REFRESH_ON_REQUEST_TTL_SEC. Never queues a
    second refresh while one is running.
    """
    now = time.time()
    if _last_refresh["ok"] and (now - _last_refresh["ok"]) < REFRESH_ON_REQUEST_TTL_SEC:
        return
    if _refresh_lock.locked():
        if blocking:
            try:
                await asyncio.wait_for(_refresh_lock.acquire(), timeout=4.0)
                _refresh_lock.release()
            except asyncio.TimeoutError:
                pass
        return
    if not blocking:
        # Fire-and-forget — schedule on the event loop, return immediately.
        asyncio.create_task(_run_refresh())
        return
    await _run_refresh()


async def _run_refresh() -> None:
    """The actual refresh work, guarded by _refresh_lock."""
    if _refresh_lock.locked():
        return
    async with _refresh_lock:
        _last_refresh["started"] = time.time()
        _last_refresh["running"] = True
        try:
            from ..refresh import refresh_async
            await refresh_async(days_back=1, days_forward=0,
                                statsapi=True, include_savant=False,
                                verbose=False)
            _cache["market_lines"] = _build_market_lines()
            # Invalidate the /api/games response cache so the next call rebuilds.
            _games_response_cache.clear()
            _last_refresh["ok"] = time.time()
            _last_refresh["err"] = None
        except Exception as ex:
            _last_refresh["err"] = repr(ex)
        finally:
            _last_refresh["running"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Spawn the background data refresher while the server runs.

    Set STRIKECAST_DISABLE_REFRESH=1 in production to skip the loop
    (the slim deploy image doesn't ship raw ESPN JSONs, so rebuilding
    odds_close.parquet on the server would wipe historical rows).
    """
    if os.environ.get("STRIKECAST_DISABLE_REFRESH", "0") == "1":
        _last_refresh["err"] = "refresh disabled by env"
        yield
        return
    task = asyncio.create_task(_refresh_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _refresh_loop():
    """Periodically refresh today's odds + rebuild light parquets.

    Skips past/future days entirely. Only touches today + N days back. Doesn't
    re-download files that already exist unless `force=True`.
    """
    from ..refresh import refresh_async
    # Initial delay so the server can start serving requests first.
    await asyncio.sleep(45)
    while True:
        _last_refresh["started"] = time.time()
        _last_refresh["running"] = True
        try:
            await refresh_async(days_back=1, days_forward=0,
                                statsapi=True, include_savant=False,
                                verbose=False)
            # Invalidate just the market_lines cache so the next request
            # rebuilds it from the freshly-written odds_close.parquet.
            _cache["market_lines"] = _build_market_lines()
            _last_refresh["ok"] = time.time()
            _last_refresh["err"] = None
        except Exception as e:
            _last_refresh["err"] = repr(e)
        finally:
            _last_refresh["running"] = False
        await asyncio.sleep(REFRESH_INTERVAL_SEC)


app = FastAPI(default_response_class=ORJSONResponse, title="STRIKECAST", lifespan=lifespan)

# CORS: comma-separated origins from env (e.g. "https://strikecast.vercel.app,https://strikecast.com").
# Defaults to "*" so localhost dev still works out of the box.
_cors_origins = [o.strip() for o in os.environ.get("STRIKECAST_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# Gzip everything ≥1KB — JSON responses (predictions, WP curves) compress ~80%,
# and the static strike.html (~100KB) drops to ~22KB on the wire.
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/api/health")
def health():
    return {"ok": True, "service": "strikecast"}


@app.get("/api/refresh-status")
def refresh_status():
    """Lightweight status endpoint for the UI footer."""
    now = time.time()
    freshness = _feed_freshness_summary()
    return {
        "interval_sec": REFRESH_INTERVAL_SEC,
        "seconds_since_last_ok": int(now - _last_refresh["ok"]) if _last_refresh["ok"] else None,
        "running_now": _last_refresh["running"],
        "last_error": _last_refresh["err"],
        **freshness,
    }


@app.post("/api/refresh-now")
async def refresh_now():
    """Trigger an immediate refresh in the background (non-blocking)."""
    if _last_refresh["running"]:
        return {"queued": False, "reason": "already running"}
    from ..refresh import refresh_async

    async def _run():
        _last_refresh["running"] = True
        _last_refresh["started"] = time.time()
        try:
            await refresh_async(days_back=1, statsapi=True, verbose=False)
            _cache["market_lines"] = _build_market_lines()
            _last_refresh["ok"] = time.time()
            _last_refresh["err"] = None
        except Exception as e:
            _last_refresh["err"] = repr(e)
        finally:
            _last_refresh["running"] = False

    asyncio.create_task(_run())
    return {"queued": True}


# ---------------- caches ----------------
_cache: dict = {}

# Response cache for /api/games keyed by (start, end). 15s TTL — fast page reloads
# avoid recomputing predictions and re-fetching live feeds, while in-game state
# stays current enough for a sport that scores once every few minutes.
_GAMES_CACHE_TTL_SEC = 15
_games_response_cache: dict[tuple[str, str], tuple[float, list]] = {}


def _merge_runtime_game_context(train: pd.DataFrame) -> pd.DataFrame:
    """Attach game metadata needed only by runtime winner overrides."""
    path = PROCESSED / "games.parquet"
    if not path.exists():
        return train
    wanted = [
        "game_pk", "weather_temp_f",
    ]
    try:
        games = pd.read_parquet(path, columns=wanted)
        missing = [col for col in wanted[1:] if col not in train.columns]
        train = train.merge(games[["game_pk", *missing]], on="game_pk", how="left") if missing else train
        circ_path = PROCESSED / "features_circadian.parquet"
        if circ_path.exists() and "circ_x_day_home" not in train.columns:
            circ = pd.read_parquet(circ_path, columns=["game_pk", "circ_x_day_home"])
            train = train.merge(circ, on="game_pk", how="left")
        travel_path = PROCESSED / "features_travel.parquet"
        if travel_path.exists() and "travel_dist_miles_a" not in train.columns:
            travel = pd.read_parquet(travel_path, columns=["game_pk", "side", "travel_dist_miles"])
            wide = travel.pivot(index="game_pk", columns="side")
            wide.columns = [f"{metric}_{str(side)[0]}" for metric, side in wide.columns]
            train = train.merge(wide.reset_index(), on="game_pk", how="left")
        luck_path = PROCESSED / "features_luck.parquet"
        if luck_path.exists() and "babip_luck_net_a" not in train.columns:
            luck = pd.read_parquet(luck_path, columns=["game_pk", "side", "babip_luck_net"])
            wide = luck.pivot(index="game_pk", columns="side")
            wide.columns = [f"{metric}_{str(side)[0]}" for metric, side in wide.columns]
            train = train.merge(wide.reset_index(), on="game_pk", how="left")
        cluster_path = PROCESSED / "features_cluster_luck.parquet"
        if cluster_path.exists() and "lob_off_dev_h" not in train.columns:
            cluster = pd.read_parquet(cluster_path, columns=["game_pk", "team", "lob_off_dev", "lob_def_dev"])
            home = cluster.rename(columns={"team": "home_team_abbrev", "lob_off_dev": "lob_off_dev_h", "lob_def_dev": "lob_def_dev_h"})
            away = cluster.rename(columns={"team": "away_team_abbrev", "lob_off_dev": "lob_off_dev_a", "lob_def_dev": "lob_def_dev_a"})
            train = train.merge(home, on=["game_pk", "home_team_abbrev"], how="left")
            train = train.merge(away, on=["game_pk", "away_team_abbrev"], how="left")
        comeback_path = PROCESSED / "features_comeback.parquet"
        if comeback_path.exists() and "cb_avg_def_l30_h" not in train.columns:
            comeback = pd.read_parquet(comeback_path, columns=["game_pk", "side", "cb_avg_def_l30"])
            wide = comeback.pivot(index="game_pk", columns="side")
            wide.columns = [f"{metric}_{str(side)[0]}" for metric, side in wide.columns]
            train = train.merge(wide.reset_index(), on="game_pk", how="left")
        flow_path = PROCESSED / "features_game_flow.parquet"
        if flow_path.exists() and "runs_median_l20_h" not in train.columns:
            flow = pd.read_parquet(flow_path, columns=["game_pk", "side", "runs_median_l20"])
            wide = flow.pivot(index="game_pk", columns="side")
            wide.columns = [f"{metric}_{str(side)[0]}" for metric, side in wide.columns]
            train = train.merge(wide.reset_index(), on="game_pk", how="left")
        burn_path = PROCESSED / "features_burn.parquet"
        if burn_path.exists() and "burn_score_h" not in train.columns:
            burn = pd.read_parquet(burn_path, columns=["game_pk", "side", "burn_score"])
            wide = burn.pivot(index="game_pk", columns="side")
            wide.columns = [f"{metric}_{str(side)[0]}" for metric, side in wide.columns]
            train = train.merge(wide.reset_index(), on="game_pk", how="left")
        workload_path = PROCESSED / "features_starter_workload.parquet"
        if workload_path.exists() and "starter_prev_pitches_h" not in train.columns:
            workload = pd.read_parquet(workload_path)
            train = train.merge(workload, on="game_pk", how="left")
        highlev_path = PROCESSED / "features_high_leverage.parquet"
        if highlev_path.exists() and "highlev_pitches_1d_h" not in train.columns:
            highlev = pd.read_parquet(highlev_path)
            train = train.merge(highlev, on="game_pk", how="left")
        catcher_path = PROCESSED / "features_catcher_control.parquet"
        if catcher_path.exists() and "battery_kbb_l8_h" not in train.columns:
            catcher = pd.read_parquet(catcher_path, columns=["game_pk", "side", "battery_kbb_l8"])
            wide = catcher.pivot(index="game_pk", columns="side")
            wide.columns = [f"{metric}_{str(side)[0]}" for metric, side in wide.columns]
            train = train.merge(wide.reset_index(), on="game_pk", how="left")
        return train
    except Exception:
        return train


def _load():
    if "train" in _cache:
        return _cache
    train = pd.read_parquet(PROCESSED / "train.parquet")
    train = _merge_runtime_game_context(train)
    train["game_date"] = pd.to_datetime(train["game_date"]).dt.date
    games = pd.read_parquet(PROCESSED / "games.parquet")
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    ens_path = MODELS / "ensemble.pkl"
    if ens_path.exists():
        with open(ens_path, "rb") as f:
            cls_b = pickle.load(f)
        cls_b["__ensemble__"] = True
    else:
        with open(MODELS / "lgb_cls.pkl", "rb") as f:
            cls_b = pickle.load(f)
        cls_b["__ensemble__"] = False
    # Standalone calibrated classifier — the VALUE/edge probability source.
    # Decoupled from the display blend so edges aren't shrunk toward market.
    # Matches edge_threshold_sweep.py, which calibrated the 9pp threshold here.
    cls_value_b = None
    lgb_cls_path = MODELS / "lgb_cls.pkl"
    if lgb_cls_path.exists():
        with open(lgb_cls_path, "rb") as f:
            cls_value_b = pickle.load(f)
    with open(MODELS / "lgb_reg.pkl", "rb") as f:
        reg_b = pickle.load(f)
    # F5 models — separate ensemble for First-5-Innings market
    f5_cls_b = None
    f5_reg_b = None
    f5_cls_path = MODELS / "f5_cls.pkl"
    f5_reg_path = MODELS / "f5_reg.pkl"
    if f5_cls_path.exists():
        with open(f5_cls_path, "rb") as f:
            f5_cls_b = pickle.load(f)
    if f5_reg_path.exists():
        with open(f5_reg_path, "rb") as f:
            f5_reg_b = pickle.load(f)
    ps_path = PROCESSED / "features_pitcher_season.parquet"
    if ps_path.exists():
        pitcher_season = pd.read_parquet(ps_path)
        pitcher_season["player_id"] = pitcher_season["player_id"].astype("Int64")
        pitcher_season["season"] = pitcher_season["season"].astype("Int64")
        pitcher_season = pitcher_season.set_index(["player_id", "season"])
    else:
        pitcher_season = None

    market_lines = _build_market_lines()
    # Pinnacle sharp-value table (soft book price vs Pinnacle de-vig fair).
    pin_path = PROCESSED / "pinnacle_value.parquet"
    pinnacle_value = None
    if pin_path.exists():
        try:
            pinnacle_value = pd.read_parquet(pin_path).set_index("game_pk")
        except Exception:
            pinnacle_value = None

    _cache.update({"train": train, "games": games, "cls": cls_b, "reg": reg_b,
                   "cls_value": cls_value_b,
                   "pitcher_season": pitcher_season, "market_lines": market_lines,
                   "pinnacle_value": pinnacle_value,
                   "f5_cls": f5_cls_b, "f5_reg": f5_reg_b})
    return _cache


def _build_market_lines():
    """Build the game_pk → moneyline, total, run-line table.

    Merges odds_close (ML + total) with odds.parquet (O/U odds, spread, RL odds).
    Returns None if the underlying parquets are missing.
    """
    odds_close_path = PROCESSED / "odds_close.parquet"
    xref_path = PROCESSED / "games_xref.parquet"
    if not (odds_close_path.exists() and xref_path.exists()):
        return None
    # ── Moneyline + total from odds_close (sharp books) ───────────────────────
    odds = pd.read_parquet(odds_close_path)
    sharp = odds[odds["provider_name"].isin({"DraftKings", "ESPN BET"})].copy()
    sharp["home_ml"] = (sharp["home_ml_close"]
                         .fillna(sharp["home_ml_current"])
                         .fillna(sharp["home_ml_top"]))
    sharp["away_ml"] = (sharp["away_ml_close"]
                         .fillna(sharp["away_ml_current"])
                         .fillna(sharp["away_ml_top"]))
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id").agg(
        home_ml=("home_ml", "median"),
        away_ml=("away_ml", "median"),
        total_close=("total_close", "median"),
        home_rl_odds=("spread_close", "median"),  # home team RL odds (American)
    ).reset_index()
    # ── Over/Under odds + Run Line (spread) from odds.parquet (all providers) ─
    odds_raw_path = PROCESSED / "odds.parquet"
    if odds_raw_path.exists():
        odds_raw = pd.read_parquet(odds_raw_path)
        agg_raw = odds_raw.groupby("espn_event_id").agg(
            over_odds=("over_odds", "median"),
            under_odds=("under_odds", "median"),
            run_line=("spread", "median"),
            away_rl_odds=("away_spread_odds", "median"),
            home_rl_odds=("home_spread_odds", "median"),
        ).reset_index()
        agg = agg.merge(agg_raw, on="espn_event_id", how="left")
    # ── Best available price (line shopping) across all pre-game books ────────
    # The MEDIAN above stays the de-vig "fair" reference; the BEST price is what
    # you'd actually bet at. Taking the best of ~5-13 books is worth ~3-5pp of
    # implied probability (validated on odds.parquet history).
    best = _best_lines_from(odds)
    if best is not None:
        agg = agg.merge(best, on="espn_event_id", how="left")

    xref = pd.read_parquet(xref_path)
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    return agg.merge(matched, on="espn_event_id", how="inner") \
              .drop(columns=["espn_event_id"]).drop_duplicates("game_pk") \
              .set_index("game_pk")


_BOOK_DISPLAY = {
    "ESPN BET": "ESPN BET", "DraftKings": "DraftKings", "DraftKings (old)": "DraftKings",
    "MGM": "BetMGM", "Bet365": "Bet365", "Unibet": "Unibet", "SugarHouse": "SugarHouse",
    "PointsBet": "PointsBet", "BetfairSportsbook": "Betfair",
}


def _clean_book(name: str) -> str:
    if not isinstance(name, str):
        return "?"
    if name.startswith("Caesars"):
        return "Caesars"
    return _BOOK_DISPLAY.get(name, name)


def _best_lines_from(odds: pd.DataFrame) -> pd.DataFrame | None:
    """Per event: best (highest = most favorable) home/away ML + the book name.

    Higher American odds are always better for the bettor, so max() = best price.
    Excludes live-odds feeds and non-book aggregators.
    """
    o = odds[~odds["provider_name"].str.contains("Live", case=False, na=False)].copy()
    o = o[~o["provider_name"].isin(["consensus", "teamrankings"])]
    o["home_ml"] = o["home_ml_close"].fillna(o["home_ml_current"]).fillna(o["home_ml_top"])
    o["away_ml"] = o["away_ml_close"].fillna(o["away_ml_current"]).fillna(o["away_ml_top"])
    o = o.dropna(subset=["home_ml", "away_ml"])
    # Valid American odds are never 0 and never strictly between -100 and +100.
    valid = lambda s: (s.abs() >= 100)
    o = o[valid(o["home_ml"]) & valid(o["away_ml"])]
    if o.empty:
        return None
    rows = []
    for eid, grp in o.groupby("espn_event_id"):
        hi = grp.loc[grp["home_ml"].idxmax()]
        ai = grp.loc[grp["away_ml"].idxmax()]
        rows.append({
            "espn_event_id": eid,
            "best_home_ml": float(hi["home_ml"]),
            "best_home_book": _clean_book(hi["provider_name"]),
            "best_away_ml": float(ai["away_ml"]),
            "best_away_book": _clean_book(ai["provider_name"]),
            "n_books": int(grp["provider_name"].nunique()),
        })
    return pd.DataFrame(rows)


def _insights(r: pd.Series) -> dict:
    """Comparative team stats for the hover-expand panel on Hoy page."""
    def _g(col):
        v = r.get(col)
        return float(v) if pd.notna(v) else None
    return {
        "ins_off_xwoba_h":   _g("off_xwoba_l30_h"),
        "ins_off_xwoba_a":   _g("off_xwoba_l30_a"),
        "ins_def_xwoba_h":   _g("def_xwoba_l30_h"),
        "ins_def_xwoba_a":   _g("def_xwoba_l30_a"),
        "ins_off_barrel_h":  _g("off_barrel_rate_l30_h"),
        "ins_off_barrel_a":  _g("off_barrel_rate_l30_a"),
        "ins_off_k_pct_h":   _g("off_k_pct_l30_h"),
        "ins_off_k_pct_a":   _g("off_k_pct_l30_a"),
        "ins_def_k_pct_h":   _g("def_k_pct_l30_h"),
        "ins_def_k_pct_a":   _g("def_k_pct_l30_a"),
        "ins_win_pct_h":     _g("win_pct_l30_h"),
        "ins_win_pct_a":     _g("win_pct_l30_a"),
        "ins_runs_scored_h": _g("runs_scored_l10_h"),
        "ins_runs_scored_a": _g("runs_scored_l10_a"),
        "ins_runs_allowed_h":_g("runs_allowed_l10_h"),
        "ins_runs_allowed_a":_g("runs_allowed_l10_a"),
        "ins_run_diff_h":    _g("run_diff_l10_h"),
        "ins_run_diff_a":    _g("run_diff_l10_a"),
        "ins_elo_h":         _g("home_elo_pre"),
        "ins_elo_a":         _g("away_elo_pre"),
        "ins_cb_rate_h":     _g("cb_rate_l30_h"),
        "ins_cb_rate_a":     _g("cb_rate_l30_a"),
        "ins_days_rest_h":   _g("days_since_last_game_h"),
        "ins_days_rest_a":   _g("days_since_last_game_a"),
        "ins_starter_era_h": _g("starter_xwoba_l15_h"),
        "ins_starter_era_a": _g("starter_xwoba_l15_a"),
    }


def _game_context(c, r: pd.Series, games_df: pd.DataFrame | None = None) -> dict:
    """Per-game narrative context: weather, venue, burn yesterday per team."""
    out = {
        "ctx_weather_temp": None,
        "ctx_weather_cond": None,
        "ctx_weather_wind": None,
        "ctx_roof_type": None,
        "ctx_burn_away": None,
        "ctx_burn_home": None,
        "ctx_burn_away_tags": [],
        "ctx_burn_home_tags": [],
    }
    # Weather + roof from games table
    if games_df is not None:
        g_row = games_df[games_df["game_pk"] == r["game_pk"]]
        if not g_row.empty:
            g_row = g_row.iloc[0]
            for src, dst in [
                ("weather_temp_f", "ctx_weather_temp"),
                ("weather_condition", "ctx_weather_cond"),
                ("weather_wind", "ctx_weather_wind"),
                ("roof_type", "ctx_roof_type"),
            ]:
                v = g_row.get(src)
                if pd.notna(v):
                    out[dst] = float(v) if dst == "ctx_weather_temp" else str(v)
    # Burn from yesterday — load on demand and cache
    burn = c.get("_burn_today")
    if burn is None:
        burn_path = PROCESSED / "burn_analysis.parquet"
        if burn_path.exists():
            burn = pd.read_parquet(burn_path)
            burn["game_date"] = pd.to_datetime(burn["game_date"])
            c["_burn_today"] = burn
        else:
            burn = pd.DataFrame()
            c["_burn_today"] = burn
    if len(burn):
        gd = pd.to_datetime(r["game_date"])
        for team, key in [(r["away_team_abbrev"], "ctx_burn_away"),
                           (r["home_team_abbrev"], "ctx_burn_home")]:
            sub = burn[(burn["team"] == team) & (burn["game_date"].dt.date == gd.date())]
            if not sub.empty:
                b = sub.iloc[0]
                out[key] = round(float(b["burn_score"]), 1)
                tags = []
                if b.get("extra_innings_yest"):     tags.append("Extras")
                if b.get("long_game_yest"):         tags.append("Largo")
                if b.get("used_closer_close_yest"): tags.append("Cerrador")
                if b.get("comeback_won_yest"):      tags.append("Comeback")
                bp_ip = b.get("bullpen_ip_yest")
                if pd.notna(bp_ip) and float(bp_ip) >= 4.5: tags.append(f"BP {float(bp_ip):.1f}ip")
                out[key + "_tags"] = tags
    return out


_pin_cache: dict = {"ts": 0.0, "data": None}
_PIN_TTL_SEC = 60


def _get_pinnacle_value():
    """Load pinnacle_value.parquet fresh (60s TTL) so intraday pulls appear
    without a server restart — the sharp odds change through the day."""
    now = time.time()
    if _pin_cache["data"] is not None and (now - _pin_cache["ts"]) < _PIN_TTL_SEC:
        return _pin_cache["data"]
    pv = None
    p = PROCESSED / "pinnacle_value.parquet"
    if p.exists():
        try:
            pv = pd.read_parquet(p).set_index("game_pk")
        except Exception:
            pv = None
    _pin_cache.update({"ts": now, "data": pv})
    return pv


def _pinnacle_block(c, game_pk: int) -> dict:
    """Sharp-value fields from Pinnacle: where a soft book pays over fair.

    Pinnacle de-vig ≈ true probability. When a soft book's price implies a
    lower probability than Pinnacle's fair line, that book is paying above fair
    → mechanical +EV independent of our model.
    """
    out = {
        "pin_has": False,
        "pin_fair_home": None, "pin_fair_away": None,
        "pin_value_side": None, "pin_value_book": None,
        "pin_value_edge_pp": None, "pin_value_ml": None,
    }
    pv = _get_pinnacle_value()
    if pv is None or game_pk not in pv.index:
        return out
    row = pv.loc[game_pk]
    out["pin_has"] = True
    out["pin_fair_home"] = _safe(row.get("pin_fair_home"))
    out["pin_fair_away"] = _safe(row.get("pin_fair_away"))
    eh = _safe(row.get("sharp_edge_home_pp"))
    ea = _safe(row.get("sharp_edge_away_pp"))
    # Pick the side with the larger positive sharp edge (>= 1pp to matter)
    best_side, best_edge, best_book, best_ml = None, None, None, None
    if eh is not None and (ea is None or eh >= ea) and eh >= 1.0:
        best_side, best_edge = "HOME", eh
        best_book, best_ml = row.get("best_home_book"), _safe(row.get("best_home_ml"))
    elif ea is not None and ea >= 1.0:
        best_side, best_edge = "AWAY", ea
        best_book, best_ml = row.get("best_away_book"), _safe(row.get("best_away_ml"))
    if best_side:
        _bkdisp = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
                   "caesars": "Caesars", "williamhill_us": "Caesars"}
        out["pin_value_side"]    = best_side
        out["pin_value_book"]    = _bkdisp.get(best_book, best_book) if isinstance(best_book, str) else None
        out["pin_value_edge_pp"] = round(best_edge, 2)
        out["pin_value_ml"]      = best_ml
    return out


def _market_extras(c, game_pk: int) -> dict:
    """Return total line, O/U odds, and run-line odds from market_lines."""
    ml = c.get("market_lines")
    empty = {
        "total_close": None,
        "over_odds": None, "under_odds": None,
        "run_line": None, "home_rl_odds": None,
    }
    if ml is None or game_pk not in ml.index:
        return empty
    row = ml.loc[game_pk]
    return {
        "total_close":  _safe(row.get("total_close")),
        "over_odds":    _safe(row.get("over_odds")),
        "under_odds":   _safe(row.get("under_odds")),
        "run_line":     _safe(row.get("run_line")),
        # spread_close = home team RL odds (American) from odds_close
        "home_rl_odds": _safe(row.get("home_rl_odds")),
    }


def _proj_ks(c, r: pd.Series, feed=None) -> dict:
    """Projected strikeouts per starter.

    Primary: rolling k_pct × pa from train features.
    Fallback: pitcher_season k9 × avg innings/start, using probable pitcher
    IDs from the train row or the live StatsAPI feed.
    """
    def _from_roll(k_pct, pa):
        if k_pct is None or pa is None:
            return None
        try:
            v = float(k_pct) * float(pa)
            return round(v, 1) if np.isfinite(v) else None
        except Exception:
            return None

    away_k = _from_roll(r.get("starter_k_pct_l15_a"), r.get("starter_pa_l15_a"))
    home_k = _from_roll(r.get("starter_k_pct_l15_h"), r.get("starter_pa_l15_h"))

    if away_k is None or home_k is None:
        ps = c.get("pitcher_season")
        season = int(r.get("season", 2026)) if pd.notna(r.get("season", None)) else 2026
        if ps is not None:
            away_pid = _safe(r.get("probable_away_pitcher_id"))
            home_pid = _safe(r.get("probable_home_pitcher_id"))
            # Try probable pitchers from live feed when train row has no IDs
            if feed is not None and (not away_pid or not home_pid):
                gd = feed.get("gameData") or {}
                prob = gd.get("probablePitchers") or {}
                if not away_pid:
                    away_pid = (prob.get("away") or {}).get("id")
                if not home_pid:
                    home_pid = (prob.get("home") or {}).get("id")

            def _from_season(pid):
                if not pid:
                    return None
                try:
                    pid = int(pid)
                    # pitcher_season is indexed by (player_id, season)
                    try:
                        row = ps.loc[(pid, season)]
                    except KeyError:
                        # fallback: most recent season for this pitcher
                        try:
                            sub = ps.loc[pid]
                        except KeyError:
                            return None
                        # sub may be a Series (one season) or DataFrame (multi)
                        row = sub.iloc[-1] if isinstance(sub, pd.DataFrame) else sub
                    # loc returns Series when one row matches, DataFrame when many
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[-1]
                    k9 = float(row["k9"]) if pd.notna(row.get("k9")) else None
                    if k9 is None:
                        return None
                    ip   = float(row["ip"])          if pd.notna(row.get("ip"))          else 0
                    apps = int(row["appearances"])   if pd.notna(row.get("appearances")) else 0
                    avg_inn = (ip / apps) if apps > 0 else 5.5
                    return round(k9 * avg_inn / 9, 1)
                except Exception:
                    return None

            if away_k is None:
                away_k = _from_season(away_pid)
            if home_k is None:
                home_k = _from_season(home_pid)

    return {"proj_k_away": away_k, "proj_k_home": home_k}


def _f5_predict(c, r: pd.Series) -> dict:
    """Return F5 predictions (P(home_won_F5), pred_total_F5)."""
    out = {"f5_p_home": None, "f5_p_away": None, "f5_pred_total": None,
            "f5_fair_home_ml": None, "f5_fair_away_ml": None,
            "f5_pick_abbrev": None, "f5_played": False,
            "f5_home_score": _safe(r.get("f5_home_score")),
            "f5_away_score": _safe(r.get("f5_away_score")),
            "f5_actual_total": _safe(r.get("f5_total_runs")),
            "f5_actual_tied": bool(r.get("f5_tied") == 1) if pd.notna(r.get("f5_tied")) else None,
            "f5_actual_home_won": bool(r.get("f5_home_won") == 1) if pd.notna(r.get("f5_home_won")) else None}
    if c.get("f5_cls") is None or c.get("f5_reg") is None:
        return out
    f5c = c["f5_cls"]; f5r = c["f5_reg"]
    try:
        Xc = pd.DataFrame([r[f5c["feature_names"]].values], columns=f5c["feature_names"])
        Xr = pd.DataFrame([r[f5r["feature_names"]].values], columns=f5r["feature_names"])
        p_home_f5 = float(f5c["model"].predict_proba(Xc)[0, 1])
        total_f5 = float(f5r["model"].predict(Xr)[0])
        out["f5_p_home"] = round(p_home_f5, 4)
        out["f5_p_away"] = round(1 - p_home_f5, 4)
        out["f5_pred_total"] = round(total_f5, 2)
        out["f5_fair_home_ml"] = _to_american(p_home_f5)
        out["f5_fair_away_ml"] = _to_american(1 - p_home_f5)
        out["f5_pick_abbrev"] = r["home_team_abbrev"] if p_home_f5 >= 0.5 else r["away_team_abbrev"]
    except Exception:
        pass
    out["f5_played"] = pd.notna(r.get("f5_total_runs"))
    return out


# Empirical low-confidence zone for the model (see src/analysis/calibration_bands.py).
# Walkforward (n=4,411) shows the market beats the model by 1-5pp accuracy when
# the model's top-side probability is in [50%, 58%]. Specifically the [52, 55] band
# is the worst (n=1131, market 54.47% vs model 49.78%, +4.69pp).
# Rule: in this band, if market disagrees with the model's pick, defer to market.
LOW_CONF_LO = 0.50
LOW_CONF_HI = 0.58

# User-observed dog sweet spot: when the model's underdog has 46-48% probability,
# it wins 50.6% historically (n=828, calibration_bands.py Section C). Used to
# override slot_skip and surface the dog as a marginal pick. Gain is small
# (~+1pp ROI vs not betting), but matches the user's empirical observation.
DOG_LO = 0.46
DOG_HI = 0.48

# Pareto-optimal favorite-band rule (validate_v2.py).
# When the favorite's decimal odds land in [1.50, 1.80) — moderate favorites
# at -200 to -125 — and the model picks the OPPOSITE side from the market,
# the market wins more often. n=433 walkforward games, acc +1.07pp, roi -0.64pp.
# We accept the small ROI hit because the accuracy gain makes UI picks more
# trustworthy.
FAV_BAND_LO = 1.50
FAV_BAND_HI = 1.80
DOG_HIST_WIN_PROB = 0.506

# Rule Lab holdout-positive overrides (src.analysis.rule_lab).
# Chosen on validation, then checked on 2025-08+ holdout:
# model baseline 53.7% -> conservative rule set 56.4% (+2.62pp).
RULE_MODEL_CONF_LO = 0.52
RULE_MODEL_CONF_HI = 0.55
STARTER_XWOBA_EDGE_Q25 = -0.03235585885990984
STARTER_XWOBA_EDGE_Q45 = -0.005290730480286905
STARTER_XWOBA_FOR_MODEL_Q25 = -0.019166523033624386
TEAM_WIN_FOR_MODEL_Q25 = -0.06666666666666665
TEAM_WIN_FOR_MODEL_Q45 = 0.033333333333333326
DISABLED_ACCURACY_REASONS = {
    "slot_flip",
    "slot_sharp",
    "low_conf_market",
    "dog_46_48",
}


# Empirical slot profile from walkforward (n=4,411 — see src/analysis/slot_fade.py).
# Used by _value_block to downgrade/upgrade ev_grade based on time-of-day signal.
#   model_strong: slots where the model genuinely beats market (+ROI in walk-forward)
#   market_sharp: slot where the market is so accurate (~62%) we should defer to it
#   flip_zone   : model loses but flipping the pick was +ROI (mostly via longer dog prices)
#   skip        : model loses, flip didn't beat skip — best action is no bet
SLOT_PROFILES = {
    1: "model_strong",
    9: "model_strong",
    5: "market_sharp",
    2: "flip_zone", 4: "flip_zone", 12: "flip_zone", 14: "flip_zone",
    3: "skip", 6: "skip", 7: "skip", 8: "skip",
    10: "skip", 11: "skip", 13: "skip", 15: "skip",
}


def _value_block(c, game_pk: int, p_home_model: float,
                  away_abbrev: str, home_abbrev: str,
                  slot: int | None = None,
                  context=None) -> dict:
    """Produce fair odds + value pick + EV grade based on REAL book lines (with vig).

    Empirical sweet spot from walk-forward analysis (4,411 games):
      edge vs book ∈ [1%, 2%]  → +11.2% ROI flat-$100 (the sweet spot)
      edge vs book ∈ [0%, 1%]  → +0.8% ROI (marginal but profitable)
      edge vs book ∈ [2%, 3%]  → -6% ROI (model overconfident → SKIP)
      edge vs book ≥ +3%       → progressively WORSE (model delirando)

    ev_grade encodes this: 'sweet', 'marginal', 'overconfident', 'no_edge'.
    """
    out: dict = {
        "fair_home_decimal": _to_decimal(p_home_model),
        "fair_away_decimal": _to_decimal(1 - p_home_model) if np.isfinite(p_home_model) else None,
        "market_home_ml": None,
        "market_away_ml": None,
        "market_home_decimal": None,
        "market_away_decimal": None,
        "market_home_implied": None,
        "market_away_implied": None,
        # Edge vs devigged consensus (kept for UI)
        "home_edge_pp": None,
        "away_edge_pp": None,
        # Edge vs ACTUAL book line (includes vig) — used for EV decisions
        "home_edge_vs_book_pp": None,
        "away_edge_vs_book_pp": None,
        # Recommended bet (NEW logic — uses empirical sweet spot)
        "value_side": None,
        "value_decimal": None,
        "value_edge_pp": None,        # edge vs book (with vig)
        "value_team_abbrev": None,
        "ev_grade": "no_data",        # sweet | marginal | overconfident | no_edge | no_data
        "ev_per_dollar": None,        # raw EV per $1 wagered
        # ── Slot-aware fields (empirical patterns from walkforward) ──
        "slot_profile": SLOT_PROFILES.get(slot, "neutral") if slot else "neutral",
        # Why ev_grade ended up where it did: base | slot_skip | slot_boost |
        # slot_sharp | slot_flip | low_conf_market | dog_46_48 | fav_band_market
        "ev_grade_reason": "base",
        # Alternative pick suggested by the slot profile (for UI to surface)
        "alt_pick": None,            # dict { side, team, decimal, kind }
    }

    def apply_dog_46_48_override() -> bool:
        """Surface the model's 46-48% dog when the market price is playable."""
        dog_side = None
        dog_prob = None
        dog_dec = None
        dog_team = None
        if DOG_LO <= p_home_model < DOG_HI:
            dog_side, dog_prob, dog_dec, dog_team = "HOME", p_home_model, dec_h, home_abbrev
        elif DOG_LO <= (1 - p_home_model) < DOG_HI:
            dog_side, dog_prob, dog_dec, dog_team = "AWAY", 1 - p_home_model, dec_a, away_abbrev
        if dog_side is None or dog_dec is None:
            return False

        book_implied = 1.0 / dog_dec
        out["value_side"]        = dog_side
        out["value_decimal"]     = round(dog_dec, 2)
        out["value_edge_pp"]     = round((DOG_HIST_WIN_PROB - book_implied) * 100, 2)
        out["value_team_abbrev"] = dog_team
        out["ev_per_dollar"]     = round(DOG_HIST_WIN_PROB * (dog_dec - 1) - (1 - DOG_HIST_WIN_PROB), 4)
        out["ev_grade"]          = "no_edge"
        out["ev_grade_reason"]   = "dog_46_48"
        out["alt_pick"]          = None
        return True

    def set_rule_pick(rule_side: str, reason: str) -> None:
        rule_dec = dec_h if rule_side == "HOME" else dec_a
        rule_team = home_abbrev if rule_side == "HOME" else away_abbrev
        out["value_side"]        = rule_side
        out["value_decimal"]     = round(rule_dec, 2)
        out["value_edge_pp"]     = None
        out["value_team_abbrev"] = rule_team
        out["ev_per_dollar"]     = None
        out["ev_grade"]          = "no_edge"
        out["ev_grade_reason"]   = reason
        out["alt_pick"]          = None

    ml = c.get("market_lines")
    if ml is None or game_pk not in ml.index:
        return out
    row = ml.loc[game_pk]
    h_ml = _safe(row["home_ml"])
    a_ml = _safe(row["away_ml"])
    out["market_home_ml"] = h_ml
    out["market_away_ml"] = a_ml
    dec_h = _american_to_decimal(h_ml)
    dec_a = _american_to_decimal(a_ml)
    out["market_home_decimal"] = dec_h
    out["market_away_decimal"] = dec_a
    # De-vigged implied probabilities (consensus estimate)
    p_h_raw = _implied_from_american(h_ml)
    p_a_raw = _implied_from_american(a_ml)
    if (p_h_raw is None or p_a_raw is None or not np.isfinite(p_home_model)
        or dec_h is None or dec_a is None):
        return out
    total = p_h_raw + p_a_raw
    p_h_devig = p_h_raw / total
    p_a_devig = p_a_raw / total
    out["market_home_implied"] = round(p_h_devig, 4)
    out["market_away_implied"] = round(p_a_devig, 4)
    out["home_edge_pp"] = round((p_home_model - p_h_devig) * 100, 2)
    out["away_edge_pp"] = round(((1 - p_home_model) - p_a_devig) * 100, 2)

    # ── Edge against the BOOK LINE (with vig) — what actually matters for EV ──
    book_implied_h = 1.0 / dec_h  # includes vig
    book_implied_a = 1.0 / dec_a
    edge_h_book = p_home_model - book_implied_h
    edge_a_book = (1 - p_home_model) - book_implied_a
    out["home_edge_vs_book_pp"] = round(edge_h_book * 100, 2)
    out["away_edge_vs_book_pp"] = round(edge_a_book * 100, 2)

    # Pick the side with the larger book-edge (or skip if neither is positive)
    if edge_h_book >= edge_a_book and edge_h_book >= 0:
        side, edge_book, dec, team = "HOME", edge_h_book, dec_h, home_abbrev
        prob = p_home_model
    elif edge_a_book >= 0:
        side, edge_book, dec, team = "AWAY", edge_a_book, dec_a, away_abbrev
        prob = 1 - p_home_model
    else:
        # No side has positive edge. Keep the model side as the neutral
        # placeholder so downstream accuracy rules can still override it.
        side = "HOME" if p_home_model >= 0.5 else "AWAY"
        edge_book = edge_h_book if side == "HOME" else edge_a_book
        dec = dec_h if side == "HOME" else dec_a
        team = home_abbrev if side == "HOME" else away_abbrev
        prob = p_home_model if side == "HOME" else 1 - p_home_model

    out["value_side"] = side
    out["value_decimal"] = round(dec, 2)
    out["value_edge_pp"] = round(edge_book * 100, 2)
    out["value_team_abbrev"] = team
    out["ev_per_dollar"] = round(prob * (dec - 1) - (1 - prob), 4)

    # ── Line shopping: execute the pick at the BEST available book price ──────
    # Side selection + edge stay based on the consensus (median) above so the
    # threshold logic is unchanged; we only upgrade the EXECUTION price.
    best_ml = _safe(row.get("best_home_ml")) if side == "HOME" else _safe(row.get("best_away_ml"))
    best_book = row.get("best_home_book") if side == "HOME" else row.get("best_away_book")
    best_dec = _american_to_decimal(best_ml) if best_ml is not None else None
    if best_dec is not None and best_dec >= dec:  # only if it's actually better
        out["value_best_decimal"] = round(best_dec, 2)
        out["value_best_book"]    = best_book if isinstance(best_book, str) else None
        out["value_best_ml"]      = best_ml
        # EV at the best price (what you'd actually realize) + the consensus EV
        out["ev_per_dollar_best"] = round(prob * (best_dec - 1) - (1 - prob), 4)
        out["value_shop_gain_pp"] = round((1.0 / dec - 1.0 / best_dec) * 100, 2)
    else:
        out["value_best_decimal"] = round(dec, 2)
        out["value_best_book"]    = None
        out["value_best_ml"]      = None
        out["ev_per_dollar_best"] = out["ev_per_dollar"]
        out["value_shop_gain_pp"] = 0.0

    # ── Empirical grading — RECALIBRATED for the lineup_recent model ────────
    # New model (139 feats, w/ lineup_recent + team_id) shifts the edge curve.
    # Sweep results on the 90-day audit (src/analysis/edge_threshold_sweep.py):
    #   edge>=5pp  → +0.45% ROI on 276 picks (was OK on old model, no longer)
    #   edge>=7pp  → -5.40% ROI on 168 picks  (noise band)
    #   edge>=8pp  → +5.15% ROI on 129 picks
    #   edge>=9pp  → +13.39% ROI on 103 picks  ← BEST raw threshold
    #   edge>=10pp → +10.02% ROI on  72 picks
    # The favorite filter and low-conf filter from the previous tuning give
    # nothing extra at 9pp+ (the higher edge bar already excludes the noise),
    # so they're disabled.
    edge_pp = edge_book * 100
    if edge_pp >= 9.0:
        out["ev_grade"] = "sweet"          # +13.39% ROI band → BET
    elif edge_pp >= 0.0:
        out["ev_grade"] = "overconfident"  # 0-9pp = noise/loser band → SKIP
    else:
        out["ev_grade"] = "no_edge"        # negative EV → SKIP

    if out["ev_grade"] == "no_edge":
        apply_dog_46_48_override()

    # ── Slot-aware adjustment (post-hoc; see src/analysis/slot_fade.py) ──
    profile = out["slot_profile"]
    if profile == "skip":
        # Walk-forward says: model + flip + market all lose money in these slots.
        # Best action is usually no bet, except for the user-observed 46-48% dog band.
        if apply_dog_46_48_override():
            pass
        elif out["ev_grade"] in ("sweet", "marginal"):
            out["ev_grade_reason"] = "slot_skip"
            out["ev_grade"] = "no_edge"
    elif profile == "model_strong" and out["ev_grade"] == "marginal":
        # Slots 1 and 9 had +3-5% ROI for the model historically — bump marginal
        # picks to sweet to size them up.
        out["ev_grade"] = "sweet"
        out["ev_grade_reason"] = "slot_boost"
    elif profile == "market_sharp":
        # Slot 5: market hit 62.2% acc; model picks ate vig. Make the market
        # side the actual recommendation.
        mkt_home = p_h_devig >= 0.5
        mkt_dec  = dec_h if mkt_home else dec_a
        mkt_team = home_abbrev if mkt_home else away_abbrev
        out["value_side"]        = "HOME" if mkt_home else "AWAY"
        out["value_decimal"]     = round(mkt_dec, 2)
        out["value_edge_pp"]     = None
        out["value_team_abbrev"] = mkt_team
        out["ev_per_dollar"]     = None
        out["ev_grade"]          = "no_edge"
        out["ev_grade_reason"]   = "slot_sharp"
        out["alt_pick"]          = None
    elif profile == "flip_zone":
        # Slots 2/4/12/14: flipping was +ROI. Make the opposite side the
        # actual recommendation instead of a secondary badge.
        flip_home = (side == "AWAY")
        flip_dec  = dec_h if flip_home else dec_a
        flip_team = home_abbrev if flip_home else away_abbrev
        out["value_side"]        = "HOME" if flip_home else "AWAY"
        out["value_decimal"]     = round(flip_dec, 2)
        out["value_edge_pp"]     = None
        out["value_team_abbrev"] = flip_team
        out["ev_per_dollar"]     = None
        out["ev_grade"]          = "no_edge"
        out["ev_grade_reason"]   = "slot_flip"
        out["alt_pick"]          = None

    # ── Low-confidence override (calibration_bands.py) ──
    # Model is systematically beaten by market when its top-side prob is in
    # [50%, 58%]. Rule (validated on walkforward, +1.29pp acc): override only
    # when MODEL and MARKET disagree on direction. If they agree, the existing
    # value pick stays (book-edge picker may have legitimately landed on the
    # dog in the 45-48% band — that's the calibration miss the user noticed).
    p_pick_model = max(p_home_model, 1 - p_home_model)
    if (LOW_CONF_LO <= p_pick_model < LOW_CONF_HI
        and out["ev_grade"] in ("sweet", "marginal", "overconfident")
        and out["ev_grade_reason"] not in ("slot_sharp", "slot_flip", "dog_46_48")):
        model_side = "HOME" if p_home_model >= 0.5 else "AWAY"
        mkt_side   = "HOME" if p_h_devig    >= 0.5 else "AWAY"
        if mkt_side != model_side and mkt_side != out["value_side"]:
            mkt_dec  = dec_h if mkt_side == "HOME" else dec_a
            mkt_team = home_abbrev if mkt_side == "HOME" else away_abbrev
            out["value_side"]        = mkt_side
            out["value_decimal"]     = round(mkt_dec, 2)
            out["value_team_abbrev"] = mkt_team
            # Edge vs book is now irrelevant (we're using market's confidence,
            # not the model's). Keep edge field but flag the reason.
            out["value_edge_pp"]     = None
            out["ev_per_dollar"]     = None
            out["ev_grade"]          = "no_edge"
            out["ev_grade_reason"]   = "low_conf_market"
            # Suppress flip alt — it conflicts with the market override.
            out["alt_pick"]          = None

    # ── Favorite-band market deferral (validate_v2.py) ──
    # When the favorite is in [1.50, 1.80) decimal and the model picks the
    # opposite side from the market, walkforward shows the market is more
    # accurate. Trade-off: +1.07pp acc, -0.64pp ROI. Skip if low_conf already
    # overrode or slot already neutralized.
    fav_dec_actual = min(dec_h, dec_a) if dec_h and dec_a else None
    if (fav_dec_actual is not None
        and FAV_BAND_LO <= fav_dec_actual < FAV_BAND_HI
        and out["ev_grade"] in ("sweet", "marginal", "overconfident")
        and out["ev_grade_reason"] not in ("low_conf_market", "slot_sharp",
                                           "slot_flip", "dog_46_48")):
        model_side = "HOME" if p_home_model >= 0.5 else "AWAY"
        mkt_side   = "HOME" if p_h_devig    >= 0.5 else "AWAY"
        if mkt_side != model_side and mkt_side != out["value_side"]:
            mkt_dec  = dec_h if mkt_side == "HOME" else dec_a
            mkt_team = home_abbrev if mkt_side == "HOME" else away_abbrev
            out["value_side"]        = mkt_side
            out["value_decimal"]     = round(mkt_dec, 2)
            out["value_team_abbrev"] = mkt_team
            out["value_edge_pp"]     = None
            out["ev_per_dollar"]     = None
            out["ev_grade"]          = "no_edge"
            out["ev_grade_reason"]   = "fav_band_market"
            out["alt_pick"]          = None

    # ── Rule Lab accuracy overrides (rule_lab.py) ──
    # These rules target pick accuracy rather than raw EV. They were selected
    # without using the 2025-08+ holdout, then checked there before deployment.
    ctx = context if context is not None else {}
    model_side = "HOME" if p_home_model >= 0.5 else "AWAY"
    mkt_side = "HOME" if p_h_devig >= 0.5 else "AWAY"
    favorite_side = "HOME" if dec_h < dec_a else "AWAY"
    p_pick_model = max(p_home_model, 1 - p_home_model)
    starter_xwoba_edge = None
    starter_xwoba_for_model = None
    starter_rest_edge = None
    team_win_for_model = None
    if ctx is not None:
        sx_a = _safe(ctx.get("starter_xwoba_l15_a"))
        sx_h = _safe(ctx.get("starter_xwoba_l15_h"))
        if sx_a is not None and sx_h is not None:
            starter_xwoba_edge = sx_a - sx_h
            starter_xwoba_for_model = starter_xwoba_edge if model_side == "HOME" else -starter_xwoba_edge
        sr_h = _safe(ctx.get("starter_days_rest_h"))
        sr_a = _safe(ctx.get("starter_days_rest_a"))
        if sr_h is not None and sr_a is not None:
            starter_rest_edge = sr_h - sr_a
        tw_h = _safe(ctx.get("win_pct_l30_h"))
        tw_a = _safe(ctx.get("win_pct_l30_a"))
        if tw_h is not None and tw_a is not None:
            team_win_edge = tw_h - tw_a
            team_win_for_model = team_win_edge if model_side == "HOME" else -team_win_edge

    if ctx is not None and ctx.get("day_night") == "day" and model_side != mkt_side:
        set_rule_pick(favorite_side, "rule_lab_day_disagree_fav")
    if (RULE_MODEL_CONF_LO <= p_pick_model < RULE_MODEL_CONF_HI
        and starter_xwoba_edge is not None
        and STARTER_XWOBA_EDGE_Q25 < starter_xwoba_edge <= STARTER_XWOBA_EDGE_Q45):
        set_rule_pick("AWAY", "rule_lab_lowconf_away_sp")
    if (fav_dec_actual is not None
        and 1.75 <= fav_dec_actual < 1.90
        and _safe(ctx.get("market_over_under")) is not None
        and 7.0 < _safe(ctx.get("market_over_under")) <= 8.0):
        set_rule_pick("AWAY" if model_side == "HOME" else "HOME", "rule_lab_fav_total_flip")
    if (starter_xwoba_for_model is not None
        and starter_xwoba_for_model <= STARTER_XWOBA_FOR_MODEL_Q25
        and starter_rest_edge is not None
        and starter_rest_edge <= -1.0):
        set_rule_pick("HOME", "rule_lab_sp_rest_home")
    if (ctx.get("series_game_number") == 2
        and starter_xwoba_edge is not None
        and starter_xwoba_edge <= STARTER_XWOBA_EDGE_Q25):
        set_rule_pick(favorite_side, "rule_lab_series2_fav")
    if (RULE_MODEL_CONF_LO <= p_pick_model < RULE_MODEL_CONF_HI
        and team_win_for_model is not None
        and TEAM_WIN_FOR_MODEL_Q25 < team_win_for_model <= TEAM_WIN_FOR_MODEL_Q45):
        set_rule_pick("AWAY" if model_side == "HOME" else "HOME", "rule_lab_lowconf_form_flip")

    if out["ev_grade_reason"] in DISABLED_ACCURACY_REASONS:
        out["value_side"] = None
        out["value_decimal"] = None
        out["value_edge_pp"] = None
        out["value_team_abbrev"] = None
        out["ev_per_dollar"] = None
        out["ev_grade"] = "no_edge"
        out["ev_grade_reason"] = "accuracy_holdout_skip"
        out["alt_pick"] = None

    return out


SHRINK_FACTOR = 0.75  # after C+D, the previous 0.5 was over-correcting; 0.75 minimizes ll
SHRINK_ANCHOR = 0.55
W_AGREE    = 0.50  # market weight when model & market agree on direction
W_DISAGREE = 0.80  # market weight when they disagree (trust market more)
# Display-accuracy-optimal weights on the latest walk-forward:
# agree=0.70 keeps some model signal where both sides agree, but when they
# disagree the cleanest winner rule was effectively "trust market" and use a
# slightly sub-0.50 decision boundary.
# NOT applied to the shared blend
# because that path also feeds the value-edge calc, where a market-heavy
# blend collapses the edges. See _accuracy_phome below for the decoupled
# display probability.
W_AGREE_DISPLAY    = 0.70
W_DISAGREE_DISPLAY = 1.00
WINNER_THRESHOLD = 0.475


def _shrink_overconfidence(p: np.ndarray, factor: float = SHRINK_FACTOR,
                            anchor: float = SHRINK_ANCHOR) -> np.ndarray:
    """Pull predictions outside [anchor_low, anchor_high] band toward those
    anchors. Encodes the systematic miscalibration found in error analysis."""
    out = p.copy()
    high = p > anchor
    out[high] = anchor + (p[high] - anchor) * factor
    low = p < (1 - anchor)
    out[low] = (1 - anchor) - ((1 - anchor) - p[low]) * factor
    return np.clip(out, 1e-6, 1 - 1e-6)


def _raw_model_phome(c, X: pd.DataFrame) -> np.ndarray:
    """Model's INDEPENDENT probability for VALUE/edge detection.

    Uses the standalone calibrated lgb_cls — NO market blend, NO shrink. A real
    edge exists only where the model disagrees with the book; blending toward
    market or shrinking toward 0.55 would collapse those edges. This matches
    edge_threshold_sweep.py, which set the 9pp threshold on this exact model.
    """
    cv = c.get("cls_value")
    if cv is not None and "calibrator" in cv:
        return cv["calibrator"].predict_proba(X[cv["feature_names"]])[:, 1]
    # Fallback: ensemble raw model with shrink (less ideal but safe)
    if c["cls"].get("__ensemble__"):
        return _shrink_overconfidence(c["cls"]["model"].predict_proba(X)[:, 1])
    return c["cls"]["calibrator"].predict_proba(X)[:, 1]


def _display_model_phome(c, X: pd.DataFrame) -> np.ndarray:
    """Base winner-model probability used by the display accuracy path."""
    if c["cls"].get("__ensemble__"):
        return c["cls"]["model"].predict_proba(X)[:, 1]
    return c["cls"]["calibrator"].predict_proba(X)[:, 1]


def _blend_phome(c, X: pd.DataFrame, market_p: pd.Series,
                 w_agree: float, w_disagree: float) -> np.ndarray:
    """Disagree-aware market blend + overconfidence shrink.

    The base model already has market_logit_p_home as a feature, but a second
    explicit blend of the OUTPUT toward the de-vigged line still helps display
    accuracy because the market is near the accuracy ceiling (56.7%).
    """
    if c["cls"].get("__ensemble__"):
        p_model = _display_model_phome(c, X)
        mp = market_p.values
        if not np.isfinite(mp[0]):
            return _shrink_overconfidence(p_model)
        agree = (p_model > 0.5) == (mp > 0.5)
        w = np.where(agree, w_agree, w_disagree)
        blended = w * mp + (1 - w) * p_model
        return _shrink_overconfidence(blended)
    return c["cls"]["calibrator"].predict_proba(X)[:, 1]


def _market_band_recalibrate(p_home: np.ndarray, market_p: pd.Series) -> np.ndarray:
    """Small post-blend calibration learned from walk-forward residuals."""
    out = p_home.copy()
    mp = market_p.values
    finite = np.isfinite(mp)
    if not finite.any():
        return out
    band_low_mid = finite & (mp > 0.42) & (mp <= 0.50)
    band_high_mid = finite & (mp > 0.50) & (mp <= 0.60)
    out[band_low_mid] += 0.005
    out[band_high_mid] -= 0.020
    return np.clip(out, 1e-6, 1 - 1e-6)


def _uncertainty_shrink(p_home: np.ndarray, market_p: pd.Series) -> np.ndarray:
    """Shrink medium-market games slightly toward 50/50."""
    out = p_home.copy()
    mp = market_p.values
    mid = np.isfinite(mp) & (mp >= 0.42) & (mp <= 0.60)
    out[mid] = 0.5 + (out[mid] - 0.5) * 0.90
    return np.clip(out, 1e-6, 1 - 1e-6)


def _weak_disagreement_guard(
    p_home: np.ndarray,
    model_raw: np.ndarray,
    market_p: pd.Series,
) -> np.ndarray:
    """Do not fight the market on very weak contra signals."""
    out = p_home.copy()
    mp = market_p.values
    weak = (
        np.isfinite(model_raw)
        & np.isfinite(mp)
        & ((model_raw > 0.5) != (mp > 0.5))
        & (np.abs(model_raw - mp) < 0.08)
    )
    out[weak] = 0.5 + (mp[weak] - 0.5) * 0.98
    return np.clip(out, 1e-6, 1 - 1e-6)


def _lineup_recent_edge_adjust(
    p_home: np.ndarray,
    market_p: pd.Series,
    X: pd.DataFrame,
) -> np.ndarray:
    """Small boost for short home favorites with a clear recent top-4 edge."""
    need = {
        "lineup_top4_recent_ops_l15_h",
        "lineup_top4_recent_ops_l15_a",
        "lineup_recent_n_with_data_h",
        "lineup_recent_n_with_data_a",
    }
    if not need.issubset(X.columns):
        return p_home
    ops_diff = (
        pd.to_numeric(X["lineup_top4_recent_ops_l15_h"], errors="coerce")
        - pd.to_numeric(X["lineup_top4_recent_ops_l15_a"], errors="coerce")
    ).to_numpy()
    n_home = pd.to_numeric(X["lineup_recent_n_with_data_h"], errors="coerce").to_numpy()
    n_away = pd.to_numeric(X["lineup_recent_n_with_data_a"], errors="coerce").to_numpy()
    out = p_home.copy()
    mp = market_p.values
    good = (
        np.isfinite(mp)
        & (mp > 0.50)
        & (mp <= 0.60)
        & np.isfinite(ops_diff)
        & (ops_diff >= 0.03)
        & np.isfinite(n_home)
        & np.isfinite(n_away)
        & (n_home >= 4)
        & (n_away >= 4)
    )
    out[good] += 0.0075
    return np.clip(out, 1e-6, 1 - 1e-6)


def _bullpen_quality_adjust(
    p_home: np.ndarray,
    market_p: pd.Series,
    X: pd.DataFrame,
) -> np.ndarray:
    """Cool short home favorites whose bullpen quality has been worse recently."""
    need = {"bullpen_runs_l5_h", "bullpen_runs_l5_a"}
    if not need.issubset(X.columns):
        return p_home
    runs_diff = (
        pd.to_numeric(X["bullpen_runs_l5_h"], errors="coerce")
        - pd.to_numeric(X["bullpen_runs_l5_a"], errors="coerce")
    ).to_numpy()
    out = p_home.copy()
    mp = market_p.values
    bad = (
        np.isfinite(mp)
        & (mp > 0.50)
        & (mp <= 0.60)
        & np.isfinite(runs_diff)
        & (runs_diff >= 3.0)
    )
    out[bad] -= 0.010
    return np.clip(out, 1e-6, 1 - 1e-6)


def _starter_contact_adjust(
    p_home: np.ndarray,
    market_p: pd.Series,
    X: pd.DataFrame,
) -> np.ndarray:
    """Slightly cool short home favorites with the worse starter contact profile."""
    if "starter_barrel_against_l15_diff" not in X.columns:
        return p_home
    barrel_diff = pd.to_numeric(X["starter_barrel_against_l15_diff"], errors="coerce").to_numpy()
    out = p_home.copy()
    mp = market_p.values
    bad = (
        np.isfinite(mp)
        & (mp > 0.50)
        & (mp <= 0.60)
        & np.isfinite(barrel_diff)
        & (barrel_diff >= 0.020)
    )
    out[bad] -= 0.0025
    return np.clip(out, 1e-6, 1 - 1e-6)


def _starter_market_override(
    p_home: np.ndarray,
    market_p: pd.Series,
    X: pd.DataFrame,
) -> np.ndarray:
    """Avoid over-cooling short home favorites in some ugly-but-still-favored SP spots."""
    need = {"starter_k_pct_l15_diff", "starter_bb_pct_l15_diff", "starter_xwoba_l15_diff"}
    if not need.issubset(X.columns):
        return p_home
    k_diff = pd.to_numeric(X["starter_k_pct_l15_diff"], errors="coerce").to_numpy()
    bb_diff = pd.to_numeric(X["starter_bb_pct_l15_diff"], errors="coerce").to_numpy()
    xwoba_diff = pd.to_numeric(X["starter_xwoba_l15_diff"], errors="coerce").to_numpy()
    out = p_home.copy()
    mp = market_p.values
    keep_home = (
        np.isfinite(mp)
        & (mp > 0.50)
        & (mp <= 0.60)
        & np.isfinite(k_diff)
        & (k_diff <= -0.020)
        & (
            (np.isfinite(bb_diff) & (bb_diff >= 0.010))
            | (np.isfinite(xwoba_diff) & (xwoba_diff >= 0.020))
        )
    )
    out[keep_home] += 0.010
    return np.clip(out, 1e-6, 1 - 1e-6)


def _context_regime_adjust(
    p_home: np.ndarray,
    model_raw: np.ndarray,
    market_p: pd.Series,
    X: pd.DataFrame,
) -> np.ndarray:
    """Regime-level overrides for the worst short-favorite disagreement pockets."""
    need = {
        "starter_barrel_against_l15_diff",
        "starter_k_pct_l15_diff",
        "starter_bb_pct_l15_diff",
        "starter_xwoba_l15_diff",
        "mirror_pair_seen_h",
        "mirror_pair_seen_a",
        "awayonly_win_pct_l20_a",
        "bullpen_apps_l3d_diff",
    }
    if not need.issubset(X.columns):
        return p_home
    barrel_diff = pd.to_numeric(X["starter_barrel_against_l15_diff"], errors="coerce").to_numpy()
    k_diff = pd.to_numeric(X["starter_k_pct_l15_diff"], errors="coerce").to_numpy()
    bb_diff = pd.to_numeric(X["starter_bb_pct_l15_diff"], errors="coerce").to_numpy()
    xwoba_diff = pd.to_numeric(X["starter_xwoba_l15_diff"], errors="coerce").to_numpy()
    mp = market_p.values
    disagree = np.isfinite(model_raw) & np.isfinite(mp) & ((model_raw > 0.5) != (mp > 0.5))
    short = np.isfinite(mp) & (mp > 0.50) & (mp <= 0.60)
    ugly = np.isfinite(k_diff) & (k_diff <= -0.020) & (
        (np.isfinite(bb_diff) & (bb_diff >= 0.010))
        | (np.isfinite(xwoba_diff) & (xwoba_diff >= 0.020))
    )
    barrel_bad = np.isfinite(barrel_diff) & (barrel_diff >= 0.020)
    mirror_any = (
        pd.to_numeric(X["mirror_pair_seen_h"], errors="coerce").fillna(0).to_numpy()
        + pd.to_numeric(X["mirror_pair_seen_a"], errors="coerce").fillna(0).to_numpy()
    ) > 0
    away_road_bad = pd.to_numeric(X["awayonly_win_pct_l20_a"], errors="coerce").fillna(1.0).to_numpy() <= 0.40
    away_pen_taxed = pd.to_numeric(X["bullpen_apps_l3d_diff"], errors="coerce").to_numpy() <= -2.0
    out = p_home.copy()
    cool = short & disagree & (barrel_bad | ugly)
    out[cool] -= 0.015
    out[short & disagree & barrel_bad] -= 0.0075
    out[short & disagree & ugly] -= 0.010
    out[short & disagree & away_road_bad] -= 0.005
    out[(mp > 0.42) & (mp <= 0.50) & away_road_bad] -= 0.010
    out[(mp >= 0.40) & (mp < 0.45) & away_pen_taxed] += 0.015
    out[mirror_any] += 0.0125
    return np.clip(out, 1e-6, 1 - 1e-6)


def _pythag_regression_adjust(
    p_home: np.ndarray,
    market_p: pd.Series,
    X: pd.DataFrame,
) -> np.ndarray:
    """Fade teams that recently overperformed their run profile in ambiguous bands."""
    if "pyth_minus_actual_l30_diff" not in X.columns:
        return p_home
    pyth_luck = pd.to_numeric(X["pyth_minus_actual_l30_diff"], errors="coerce").to_numpy()
    out = p_home.copy()
    mp = market_p.values
    active = (
        np.isfinite(mp)
        & (mp >= 0.42)
        & (mp <= 0.60)
        & np.isfinite(pyth_luck)
        & (np.abs(pyth_luck) >= 0.09)
    )
    out[active] -= np.sign(pyth_luck[active]) * 0.0075
    return np.clip(out, 1e-6, 1 - 1e-6)


def _accuracy_phome(
    c,
    X: pd.DataFrame,
    market_p: pd.Series,
    context_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """DISPLAY / 'who wins' probability — market-heavy blend for max accuracy.

    Latest walk-forward: agree=0.70, disagree=1.00. The model still adds
    ranking signal overall (AUC), but on binary winner picks it loses too much
    when it fights the market, so the disagreement path defers fully to market.
    """
    model_raw = _display_model_phome(c, X)
    blended = _blend_phome(c, X, market_p, W_AGREE_DISPLAY, W_DISAGREE_DISPLAY)
    calibrated = _market_band_recalibrate(blended, market_p)
    shrunk = _uncertainty_shrink(calibrated, market_p)
    guarded = _weak_disagreement_guard(shrunk, model_raw, market_p)
    lineup_adj = _lineup_recent_edge_adjust(guarded, market_p, X)
    bullpen_adj = _bullpen_quality_adjust(lineup_adj, market_p, X)
    starter_adj = _starter_contact_adjust(bullpen_adj, market_p, X)
    override_adj = _starter_market_override(starter_adj, market_p, X)
    regime_adj = _context_regime_adjust(override_adj, model_raw, market_p, X)
    ctx = context_df if context_df is not None else X
    return _pythag_regression_adjust(regime_adj, market_p, ctx)


def _predict_phome(
    c,
    X: pd.DataFrame,
    market_p: pd.Series,
    context_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Back-compat alias → the display/accuracy probability.

    Historical call sites (team-ranking, performance, detail) all want the
    best 'who wins' estimate, so they get the accuracy probability. The VALUE
    path is the only one that must use _raw_model_phome for edge detection.
    """
    return _accuracy_phome(c, X, market_p, context_df=context_df)


def _winner_threshold(
    market_p_home: float | None,
    p_home: float | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    season: int | None = None,
) -> float:
    """Market-context winner boundary tuned for binary accuracy.

    For 2026+, ALL mid bands (0.42-0.60) plus the heavy-dog band (<=0.42)
    collapse to a symmetric 0.500. Rationale: walk-forward audit revealed the
    asymmetric bands were picking HOME 77% of the time in 2026 vs actual home
    win rate 52%. Fixes:
      - mid 0.42-0.56 → 0.500 (was 0.485/0.4975/0.485): +1.4pp
      - heavy dog <=0.42 → 0.500 (was 0.440): +0.67pp (picking HOME 50%→9% on
        heavy home dogs, acc 0.536→0.619 on n=97 2026 games)
      - short home fav 0.56-0.60 → 0.500 (was 0.515): +0.17pp (n=217 2026
        games, acc 0.548→0.558)
    Pre-2026 bands stay untouched — they were already optimized for those seasons.

    Regime gate: para seasons < 2025 el modelo naive con umbral 0.5 supera al
    umbral con bandas (test 2026-07-09: 2023 +2.4pp, 2024 +0.9pp). Los overrides
    solo aplican al regimen 2025+ donde el modelo drifta y necesita ayuda.
    """
    if season is not None and int(season) < 2025:
        return 0.5
    if market_p_home is None or not np.isfinite(market_p_home):
        return WINNER_THRESHOLD
    modern = season is not None and int(season) >= 2026
    if market_p_home <= 0.42:
        thr = 0.500 if modern else 0.440
    elif market_p_home <= 0.50:
        thr = 0.500 if modern else 0.485
    elif market_p_home <= 0.53:
        thr = 0.500 if modern else 0.4975
    elif market_p_home <= 0.56:
        thr = 0.500 if modern else 0.485
    elif market_p_home <= 0.60:
        thr = 0.500 if modern else 0.515
    else:
        thr = 0.440
    if p_home is not None and np.isfinite(p_home) and 0.42 <= market_p_home <= 0.60:
        if 0.46 <= p_home < 0.47:
            thr -= 0.025
        elif 0.46 <= (1 - p_home) < 0.47:
            thr += 0.025
        elif 0.53 <= p_home < 0.54:
            thr += 0.025
        if home_team == "AZ" and 0.55 <= p_home < 0.60:
            thr += 0.015
        elif away_team == "AZ" and 0.55 <= (1 - p_home) < 0.60:
            thr -= 0.015
        if home_team == "NYY" and 0.53 <= p_home < 0.55:
            thr += 0.015
        elif away_team == "NYY" and 0.53 <= (1 - p_home) < 0.55:
            thr -= 0.015
        if home_team == "HOU" and 0.55 <= p_home < 0.60:
            thr += 0.015
        elif away_team == "HOU" and 0.55 <= (1 - p_home) < 0.60:
            thr -= 0.015
        if home_team == "MIN" and 0.53 <= p_home < 0.55:
            thr += 0.010
        elif away_team == "MIN" and 0.53 <= (1 - p_home) < 0.55:
            thr -= 0.010
    return thr


# Trap teams — walk-forward 2023-2026 (n=6,723): the model's pick fails
# below 50% in ALL four seasons for these cases, so we invert it.
#   pick ON LAA          → 0.44/0.32/0.47/0.48 by season (model overrates them)
#   pick AGAINST MIL     → 0.27/0.48/0.46/0.45 (model underrates MIL)
# MIL is normalized with a model-market agreement gate: the flip only fires
# when both probabilities differ by at most 7pp. Exact production backtest:
# pick rate 85.4% -> 81.3%, 2025 +0, 2026 +2 (H1 +1, H2 +1).
# HOU fade tested and rejected (2/4 seasons, failed 2026).
# Re-validate each season — team identity rules decay when rosters change.
TRAP_PICK_TEAMS = {"LAA"}
TRAP_FADE_TEAMS = {"MIL"}
MIL_TRAP_MODEL_MARKET_MAX_GAP = 0.07

# Day-specific fades: (opponent, dayofweek Mon=0..Sun=6) where the pick fails
# below 50% in 3+ seasons of walk-forward, small-n but consistent enough to
# survive protocol.
#   fade WSH on Saturday → 2024 n=19 acc=0.37, 2025 n=21 acc=0.43, 2026 n=10 acc=0.30
# Impact vs threshold+trap baseline: +0.18pp global, +0.50pp in 2026.
# Watched closely; small sample size means high sensitivity to multiple-testing.
DAY_TRAP_FADE = {("WSH", 5)}  # (opponent_abbrev, dayofweek)

# Night-trap fade — equipos donde el modelo picka mal en juegos nocturnos
# en 2026 especifico (regime rule, gated season >= 2026). El pick sobre
# esos equipos en noche se FLIPPEA.
#   fade DET at night → 2026 n=32, acc 0.344 (flip da 0.656)
#   2023-2025 el modelo pegaba bien con DET noche (0.556/0.692/0.586),
#   asi que gate a 2026+. Detectado 2026-07-10. Impacto medido: +0.715pp
#   acc 2026 en n=32, +0.113pp global.
# ⚠ N pequeño (32). Revalidar mensual. Si acc_night_2026 >= 0.45 en
# proximo backtest, quitar la regla.
NIGHT_TRAP_FADE = {"DET"}  # pick sobre este equipo en night game (2026+) → flip

# Pick x day-of-week fade — combos donde el modelo picka MAL sistematicamente
# en 2025-2026. Detectados via sweep (team_picked, dow) con acc<=0.42 en 2026
# Y acc<=0.48 en 2025. Todos fallan holdout 23-24 (regime specific) → gate 2026+.
# Los 6:
#   BOS Sun: 2026 25% (n=8), 2025 40% (n=10)   |   BAL Sat: 2026 38%, 2025 48%
#   HOU Mon: 2026 29% (n=7), 2025 42% (n=12)   |   CLE Mon: 2026 33%, 2025 44%
#   BAL Tue: 2026 36% (n=11), 2025 38% (n=16)  |   CLE Sun: 2026 38%, 2025 44%
# Backtest: +1.13pp acc 2026 (n=48 afectados), gate 2026+ → 0 regresion 23-25.
# ⚠ 6 combos, N chico por combo → high multi-test risk. Revalidar mensual;
# quitar cualquier combo cuyo acc_2026 suba de 0.48 en el proximo check.
# dow: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
PICK_DAY_FADE = {
    ("BOS", 6),  # BOS Sunday
    ("HOU", 0),  # HOU Monday
    ("CLE", 0),  # CLE Monday
    ("BAL", 1),  # BAL Tuesday
    ("BAL", 5),  # BAL Saturday
    ("CLE", 6),  # CLE Sunday
}

# Pick x month fade — combos donde el pick sobre este equipo en este mes
# falla sistematicamente en 2025-2026. Detectados via sweep. Gate 2026+.
#   KC  Jun: 2026 33% (n=6),  2025 20% (n=10)
#   ATL Jun: 2026 38% (n=16), 2025 40% (n=20)
# Backtest: +0.425pp acc 2026 (n=22 afectados). ⚠ N muy chico KC → high
# noise risk. Revalidar mensual; quitar cualquier combo cuyo acc_2026
# suba de 0.48.
PICK_MONTH_FADE = {
    ("KC",  6),  # KC en Junio
    ("ATL", 6),  # ATL en Junio
}

# Night-elite teams — where the model's pick historically hits >0.62 in ALL
# four seasons of walk-forward when the game is played at night. NOT a pick
# override (the model already picks these correctly at high rate); it's a
# confidence badge shown to the user so they know it aligns with a stable
# high-hit-rate pattern.
#   PHI at night → 0.65 / 0.66 / 0.66 / 0.73 (n=26/79/73/33); total 0.668,
#                  +8.3pp vs baseline night acc 0.585.
# Re-validate each season — like the trap flips, team identity rules decay.
NIGHT_ELITE_TEAMS = {"PHI"}


# Head-to-head pair biases — matchups donde el modelo miscalibra sistematicamente
# vs realidad historica (2023-2026). Se aplica como sesgo a p_home ANTES del
# umbral, asi que puede flippear el pick cuando la miscal es lo bastante grande.
# Diccionario: (home_abbrev, away_abbrev) -> bias (positivo favorece home).
#
# Backtest 2026-07-09 sobre 8,819 juegos: agregar estas 8 reglas sube acc
# global +0.22pp y 2026 +0.64pp. En los 193 juegos que cubren, sube
# acc +9.8pp historico y +32.1pp en 2026 (n=28).
# Impacto individual medido:
#   BAL vs BOS -0.09  → +16.7pp acc en el par (18 flippeos)
#   DET vs CLE -0.09  → +16.0pp acc (16 flippeos)
#   DET vs KC  +0.10  → +13.0pp acc (3 flippeos)
#   WSH vs MIA -0.09  → +13.0pp acc (9 flippeos)
#   PIT vs CIN +0.10  → +7.4pp acc (10 flippeos)
#   KC vs MIN  +0.10  → +4.5pp acc (15 flippeos)
#   SEA vs TEX +0.10  → +4.2pp acc (1 flippeo)
#   PHI vs NYM +0.08  → +4.0pp acc (1 flippeo)
#
# Reglas RETIRADAS (probadas y descartadas, ver docs/rules_backlog.md):
#   - Inertes (0 flippeos): LAD/COL, CIN/MIL, AZ/COL
#   - Duelen global: TEX/HOU, SEA/HOU, NYY/BAL, AZ/SF
#   - Neutrales (flippean pero no netean): HOU/SEA, BOS/NYY, TB/TOR
#
# Sesgo aplicado es ~55-65% del gap detectado (conservador vs overfit).
# Re-validar mensualmente tras retrain. Si un par cae por debajo de +/-8pp de
# gap en 2026 o deja de flippear picks, quitarlo y bajarlo al backlog.
H2H_PICK_BIAS: dict[tuple[str, str], float] = {
    ("PIT", "CIN"): +0.10,
    ("BAL", "BOS"): -0.09,
    ("DET", "CLE"): -0.09,
    ("DET", "KC"):  +0.10,
    ("WSH", "MIA"): -0.09,
    ("KC",  "MIN"): +0.10,
    ("SEA", "TEX"): +0.10,
    ("PHI", "NYM"): +0.08,
}

# Venue-specific H2H regression across every ordered matchup. When the same
# home team has dominated this visitor historically, a modest 50-60% model
# favorite is overconfident often enough to justify a conservative away lean.
# The win rate is Beta(2.5, 2.5)-shrunk and uses only games from earlier dates.
H2H_VENUE_MIN_PRIOR = 3
H2H_VENUE_SHRINK_GAMES = 5.0
H2H_VENUE_HOME_WPCT_THRESHOLD = 0.70
H2H_VENUE_REGRESSION_BIAS = -0.07
_h2h_venue_cache: dict[tuple[str, str, str, int, int], float] = {}


# Pitcher-specific pick biases — abridores donde el modelo miscalibra
# sistematicamente vs realidad 2023-2026. Se detectan por diferencia entre
# p_home promedio del modelo y home_won real cuando ese pitcher inicia.
# Diccionario: (player_id, side) -> delta_p_home
#   Para side='H' (pitchea local): delta positivo = home gana mas que predicho.
#   Para side='A' (pitchea visita): delta ya expresado como delta_p_home
#     (positivo = home gana mas cuando este pitcher es visita, i.e. es un
#     pitcher malo; negativo = home gana menos, i.e. el pitcher visitante es
#     buenisimo).
# Sesgo aplicado es 55% del gap detectado, cap +/- 0.12.
# Selection: solo pitchers con concord (mismo signo en 23-25 y en 2026),
# |gap|>=8pp en ambos periodos, y n_2026>=6.
# Backtest 2026-07-09: +0.87pp acc en los 115 juegos afectados en 2026.
# Regla de mantenimiento: revalidar mensual — pitchers rotan de equipo,
# eventos de lesion cambian su rol, la señal decae.
PITCHER_PICK_BIAS: dict[tuple[int, str], float] = {
    # side H — pitcher local (delta p_home directo)
    (554430, 'H'): +0.089,   # Zack Wheeler       n=52, gap_hist +16.2pp, gap_26 +40.5pp
    (650911, 'H'): +0.068,   # Cristopher Sánchez n=58, gap_hist +12.3pp, gap_26 +22.1pp
    (594798, 'H'): +0.087,   # Jacob deGrom       n=29, gap_hist +15.8pp, gap_26 +19.7pp
    (693433, 'H'): +0.109,   # Bryan Woo          n=39, gap_hist +19.9pp, gap_26 +19.5pp
    (519242, 'H'): +0.099,   # Chris Sale         n=40, gap_hist +18.1pp, gap_26 +16.1pp
    (650644, 'H'): -0.084,   # Aaron Civale       n=44, gap_hist -15.2pp, gap_26 -19.7pp
    (694738, 'H'): -0.063,   # Landen Roupp       n=20, gap_hist -11.5pp, gap_26 -15.8pp
    # side A — pitcher visitante (SIGNO INVERTIDO: gap positivo del reporte =
    # away gana mas = p_home baja → delta_p_home negativo)
    (808967, 'A'): -0.085,   # Yoshinobu Yamamoto n=37, gap_hist +15.5pp → delta -0.085
    (686799, 'A'): +0.067,   # Jack Kochanowicz   n=27, gap_hist -12.1pp → delta +0.067
    (607536, 'A'): +0.054,   # Kyle Freeland      n=49, gap_hist  -9.8pp → delta +0.054
}


def _pitcher_bias(home_pid: int | float | None,
                  away_pid: int | float | None,
                  season: int | None = None) -> float:
    """Sesgo agregado a p_home segun los pitchers probables.

    Gate: solo aplica para seasons >= 2025 (regimen actual). Pre-2025 el
    modelo integra bien la calidad del pitcher via features.
    """
    if season is not None and int(season) < 2025:
        return 0.0
    delta = 0.0
    try:
        if home_pid is not None and pd.notna(home_pid):
            delta += PITCHER_PICK_BIAS.get((int(home_pid), 'H'), 0.0)
        if away_pid is not None and pd.notna(away_pid):
            delta += PITCHER_PICK_BIAS.get((int(away_pid), 'A'), 0.0)
    except (ValueError, TypeError):
        pass
    return delta


def _h2h_bias(home_team: str | None, away_team: str | None,
              season: int | None = None) -> float:
    """Sesgo historico a aplicar a p_home para este matchup (0.0 si no aplica).

    Gate: solo aplica para seasons >= 2025 (regimen actual). Pre-2025 el modelo
    predice bien sin ayuda.
    """
    if season is not None and int(season) < 2025:
        return 0.0
    if not home_team or not away_team:
        return 0.0
    return H2H_PICK_BIAS.get((home_team, away_team), 0.0)


def _h2h_venue_regression_bias(c, r, p_home: float) -> float:
    """Fade strong same-venue H2H dominance for modest home favorites."""
    try:
        if not 0.50 <= float(p_home) < 0.60:
            return 0.0
        home = r.get("home_team_abbrev")
        away = r.get("away_team_abbrev")
        game_date = pd.Timestamp(r.get("game_date"))
        if not home or not away or pd.isna(game_date):
            return 0.0
        season = int(r.get("season")) if pd.notna(r.get("season")) else None
        if _h2h_bias(home, away, season):
            return 0.0
        train = c["train"]
        dates = pd.to_datetime(train["game_date"], errors="coerce")
        prior = train[
            (train["home_team_abbrev"] == home)
            & (train["away_team_abbrev"] == away)
            & dates.lt(game_date)
            & train["home_win"].notna()
        ]
        n_prior = len(prior)
        if n_prior < H2H_VENUE_MIN_PRIOR:
            return 0.0
        last_pk = int(pd.to_numeric(prior["game_pk"], errors="coerce").max())
        key = (home, away, game_date.date().isoformat(), n_prior, last_pk)
        cached = _h2h_venue_cache.get(key)
        if cached is not None:
            return cached
        home_wins = float(pd.to_numeric(prior["home_win"], errors="coerce").sum())
        alpha = H2H_VENUE_SHRINK_GAMES / 2.0
        shrunk_wpct = (home_wins + alpha) / (n_prior + H2H_VENUE_SHRINK_GAMES)
        bias = (
            H2H_VENUE_REGRESSION_BIAS
            if shrunk_wpct >= H2H_VENUE_HOME_WPCT_THRESHOLD else 0.0
        )
        _h2h_venue_cache[key] = bias
        return bias
    except Exception:
        return 0.0


HOT_WEATHER_AWAY_BIAS = -0.03
STL_HOME_CALIBRATION_BIAS = 0.07
INTERLEAGUE_NL_BIAS = 0.05
UMPIRE_ACCX_HIGH_THRESHOLD = 1.0360444022338606
UMPIRE_MARKET_AWAY_BIAS = -0.05
CIRCADIAN_HOME_BIAS = 0.03
TRAVEL_RESILIENCE_THRESHOLD_MILES = 1200.0
TRAVEL_RESILIENCE_AWAY_BIAS = -0.07
PYTHAG_LUCK_THRESHOLD = 0.12
PYTHAG_LUCK_REGRESSION_BIAS = 0.03
BABIP_PERSISTENCE_THRESHOLD = 0.07
BABIP_PERSISTENCE_BIAS = 0.02
LOB_PERSISTENCE_THRESHOLD = 0.12
LOB_PERSISTENCE_BIAS = 0.07
CWS_NIGHT_BIAS = 0.07
COMEBACK_DEFICIT_REGRESSION_THRESHOLD = 0.30
COMEBACK_DEFICIT_REGRESSION_BIAS = 0.05
RUNS_MEDIAN_PERSISTENCE_THRESHOLD = 2.0
RUNS_MEDIAN_PERSISTENCE_BIAS = 0.04
BURN_RESILIENCE_THRESHOLD = 14.0
BURN_RESILIENCE_BIAS = 0.06
STARTER_WORKLOAD_REGRESSION_THRESHOLD = 34.0
STARTER_WORKLOAD_REGRESSION_BIAS = 0.04
HIGHLEV_FATIGUE_THRESHOLD = 27.0
HIGHLEV_FATIGUE_BIAS = 0.03
CATCHER_BATTERY_REGIME_THRESHOLD = 2.60
CATCHER_BATTERY_REGIME_BIAS = 0.05
AL_TEAMS = {"BAL", "BOS", "CWS", "CLE", "DET", "HOU", "KC", "LAA", "MIN", "NYY", "ATH", "SEA", "TB", "TEX", "TOR"}
NL_TEAMS = {"AZ", "ATL", "CHC", "CIN", "COL", "LAD", "MIA", "MIL", "NYM", "PHI", "PIT", "SD", "SF", "STL", "WSH"}


def _weather_extreme_bias(r) -> float:
    """Fade the home side at 90+ F, gate 2026+."""
    try:
        if int(r.get("season")) >= 2026 and float(r.get("weather_temp_f")) >= 90:
            return HOT_WEATHER_AWAY_BIAS
    except Exception:
        pass
    return 0.0


def _home_team_calibration_bias(r, p_home: float) -> float:
    """Correct St. Louis home underconfidence in the raw 0.40-0.45 band."""
    try:
        if int(r.get("season")) >= 2025 and r.get("home_team_abbrev") == "STL" and 0.40 <= float(p_home) < 0.45:
            return STL_HOME_CALIBRATION_BIAS
    except Exception:
        pass
    return 0.0


# Home team × p_home band calibration — extensión del enfoque STL de Codex.
# Sweep exhaustivo (30 equipos × 8 bandas) revela 26 combos con gap concordante
# en 2025 y 2026 y magnitud >=5pp. Se seleccionan 15 con evidencia más fuerte,
# excluyendo STL [0.40,0.45) que ya cubre Codex vía _home_team_calibration_bias.
# Backtest top 15: +0.68pp acc 2026, +0.60pp 2025. Gate 2025+.
# Bias: signo del gap 2026 × 0.07.
# Revalidar mensual — si un combo cae bajo |4pp| gap en 2026, retirar.
HOME_BAND_CALIBRATION: list[tuple[str, float, float, float]] = [
    ("CWS", 0.35, 0.40, +0.07),  # gap25 +9.8, gap26 +43.5
    ("WSH", 0.55, 0.60, -0.07),  # gap25 -29.1, gap26 -39.6
    ("PHI", 0.60, 0.65, +0.07),  # gap25 +6.3, gap26 +29.2 (concord 4 años)
    ("CWS", 0.45, 0.50, +0.07),  # gap25 +6.3, gap26 +26.3
    ("MIA", 0.50, 0.55, +0.07),  # gap25 +14.4, gap26 +23.5
    ("NYM", 0.60, 0.65, -0.07),  # gap25 -14.2, gap26 -22.0
    ("CLE", 0.55, 0.60, -0.07),  # gap25 -9.7, gap26 -20.8
    ("PIT", 0.55, 0.60, -0.07),  # gap25 -19.9, gap26 -19.1
    ("STL", 0.55, 0.60, -0.07),  # gap25 -7.8, gap26 -16.7 (banda distinta a Codex)
    ("HOU", 0.40, 0.45, +0.07),  # gap25 +31.8, gap26 +14.1
    ("CLE", 0.50, 0.55, +0.07),  # gap25 +10.0, gap26 +13.9 (concord 4 años)
    ("MIL", 0.50, 0.55, +0.07),  # gap25 +22.1, gap26 +13.9
    ("AZ",  0.40, 0.45, +0.07),  # gap25 +13.8, gap26 +13.5
    ("NYY", 0.60, 0.65, -0.07),  # gap25 -12.4, gap26 -12.7
    ("MIA", 0.40, 0.45, +0.07),  # gap25 +7.6, gap26 +11.9
]
HOME_BAND_BIAS_SCALE = 1.50


def _home_band_calibration_bias(r, p_home: float) -> float:
    """Sesgo por (home_team, banda p_home) — extension del calibration bias
    de Codex a los 15 combos con mayor gap concordante en 25-26. Gate 2025+."""
    try:
        season = int(r.get("season"))
        if season < 2025:
            return 0.0
        home = r.get("home_team_abbrev")
        if not home:
            return 0.0
        p = float(p_home)
        for team, lo, hi, bias in HOME_BAND_CALIBRATION:
            if home == team and lo <= p < hi:
                return bias * HOME_BAND_BIAS_SCALE
    except Exception:
        pass
    return 0.0


def _interleague_nl_bias(r, p_home: float) -> float:
    """Favor the NL side in low-confidence interleague games, gate 2025+."""
    try:
        if int(r.get("season")) < 2025 or not 0.45 <= float(p_home) < 0.50:
            return 0.0
        home = r.get("home_team_abbrev")
        away = r.get("away_team_abbrev")
        if home in NL_TEAMS and away in AL_TEAMS:
            return INTERLEAGUE_NL_BIAS
        if home in AL_TEAMS and away in NL_TEAMS:
            return -INTERLEAGUE_NL_BIAS
    except Exception:
        pass
    return 0.0


def _umpire_market_bias(r) -> float:
    """Lean away with an accurate umpire and a strong away market favorite."""
    try:
        if int(r.get("season")) < 2025:
            return 0.0
        market_home = float(r.get("market_p_home"))
        acc_above_x = float(r.get("ump_acc_above_x"))
        if market_home < 0.45 and acc_above_x >= UMPIRE_ACCX_HIGH_THRESHOLD:
            return UMPIRE_MARKET_AWAY_BIAS
    except Exception:
        pass
    return 0.0


def _circadian_extreme_bias(r, p_home: float) -> float:
    """Favor home when the away body clock is 3+ hours west in a day game.

    Re-tuned 2026-07-17: umbral circ>=3 (era >=2). Sweep mostró que umbral más
    estricto convierte el -1 net baseline en +1 net (4/7). Trigger más raro
    pero direccionalmente correcto.
    """
    try:
        if (int(r.get("season")) >= 2025 and 0.40 <= float(p_home) < 0.50
                and float(r.get("circ_x_day_home")) >= 3):
            return CIRCADIAN_HOME_BIAS
    except Exception:
        pass
    return 0.0


def _travel_resilience_bias(r) -> float:
    """Correct excessive away penalties on extreme travel asymmetry."""
    try:
        if int(r.get("season")) < 2025:
            return 0.0
        away = float(r.get("travel_dist_miles_a"))
        home = float(r.get("travel_dist_miles_h"))
        if away - home >= TRAVEL_RESILIENCE_THRESHOLD_MILES:
            return TRAVEL_RESILIENCE_AWAY_BIAS
    except Exception:
        pass
    return 0.0


def _pythag_luck_bias(r, p_home: float) -> float:
    """Retired: the L30 luck regression hurt true walk-forward accuracy."""
    return 0.0


def _babip_persistence_bias(r, p_home: float) -> float:
    """Follow extreme net BABIP differential near the away-side boundary."""
    try:
        if int(r.get("season")) < 2025 or not 0.40 <= float(p_home) < 0.50:
            return 0.0
        diff = float(r.get("babip_luck_net_h")) - float(r.get("babip_luck_net_a"))
        if abs(diff) >= BABIP_PERSISTENCE_THRESHOLD:
            return float(np.sign(diff)) * BABIP_PERSISTENCE_BIAS
    except Exception:
        pass
    return 0.0


def _lob_persistence_bias(r, p_home: float) -> float:
    """Follow extreme net LOB sequencing differential, gate 2025+."""
    try:
        if int(r.get("season")) < 2025 or not 0.50 <= float(p_home) < 0.60:
            return 0.0
        net_h = float(r.get("lob_def_dev_h")) - float(r.get("lob_off_dev_h"))
        net_a = float(r.get("lob_def_dev_a")) - float(r.get("lob_off_dev_a"))
        diff = net_h - net_a
        if abs(diff) >= LOB_PERSISTENCE_THRESHOLD:
            return float(np.sign(diff)) * LOB_PERSISTENCE_BIAS
    except Exception:
        pass
    return 0.0


def _cws_night_bias(r, p_home: float) -> float:
    """Follow CWS in close night-game bands, gate 2025+."""
    try:
        if (int(r.get("season")) < 2025 or not 0.40 <= float(p_home) < 0.60
                or str(r.get("day_night") or "").lower() != "night"):
            return 0.0
        if r.get("home_team_abbrev") == "CWS":
            return CWS_NIGHT_BIAS
        if r.get("away_team_abbrev") == "CWS":
            return -CWS_NIGHT_BIAS
    except Exception:
        pass
    return 0.0


def _comeback_deficit_regression_bias(r, p_home: float) -> float:
    """Regress extreme average comeback-deficit differential, gate 2025+."""
    try:
        if int(r.get("season")) < 2025 or not 0.50 <= float(p_home) < 0.60:
            return 0.0
        diff = float(r.get("cb_avg_def_l30_h")) - float(r.get("cb_avg_def_l30_a"))
        if abs(diff) >= COMEBACK_DEFICIT_REGRESSION_THRESHOLD:
            return -float(np.sign(diff)) * COMEBACK_DEFICIT_REGRESSION_BIAS
    except Exception:
        pass
    return 0.0


def _runs_median_persistence_bias(r, p_home: float) -> float:
    """Follow an extreme L20 scoring-median differential, gate 2025+."""
    try:
        if int(r.get("season")) < 2025 or not 0.50 <= float(p_home) < 0.60:
            return 0.0
        diff = float(r.get("runs_median_l20_h")) - float(r.get("runs_median_l20_a"))
        if abs(diff) >= RUNS_MEDIAN_PERSISTENCE_THRESHOLD:
            return float(np.sign(diff)) * RUNS_MEDIAN_PERSISTENCE_BIAS
    except Exception:
        pass
    return 0.0


def _burn_resilience_bias(r, p_home: float) -> float:
    """Counter over-penalization of the more-burned bullpen, gate 2025+."""
    try:
        if int(r.get("season")) < 2025 or not 0.50 <= float(p_home) < 0.60:
            return 0.0
        advantage = float(r.get("burn_score_a")) - float(r.get("burn_score_h"))
        if abs(advantage) >= BURN_RESILIENCE_THRESHOLD:
            return -float(np.sign(advantage)) * BURN_RESILIENCE_BIAS
    except Exception:
        pass
    return 0.0


def _starter_workload_regression_bias(r, p_home: float) -> float:
    """Regress an extreme prior-start pitch-count differential, gate 2025+."""
    try:
        if int(r.get("season")) < 2025 or not 0.40 <= float(p_home) < 0.50:
            return 0.0
        advantage = float(r.get("starter_prev_pitches_a")) - float(r.get("starter_prev_pitches_h"))
        if abs(advantage) >= STARTER_WORKLOAD_REGRESSION_THRESHOLD:
            return -float(np.sign(advantage)) * STARTER_WORKLOAD_REGRESSION_BIAS
    except Exception:
        pass
    return 0.0


def _highlev_fatigue_bias(r, p_home: float) -> float:
    """Penalize the side whose top relievers threw much more yesterday."""
    try:
        if int(r.get("season")) < 2025 or not 0.45 <= float(p_home) < 0.55:
            return 0.0
        advantage = float(r.get("highlev_pitches_1d_a")) - float(r.get("highlev_pitches_1d_h"))
        if abs(advantage) >= HIGHLEV_FATIGUE_THRESHOLD:
            return float(np.sign(advantage)) * HIGHLEV_FATIGUE_BIAS
    except Exception:
        pass
    return 0.0


def _catcher_battery_regime_bias(r, p_home: float) -> float:
    """Regress an extreme starter-catcher K/BB edge in the 2026 regime.

    Re-tuned 2026-07-17: banda 0.52-0.58 (era 0.50-0.60). Sweep mostró que la
    banda estrecha convierte el -1 net del baseline en +4 net (6/8, 75% acc).
    """
    try:
        if int(r.get("season")) != 2026 or not 0.52 <= float(p_home) < 0.58:
            return 0.0
        advantage = float(r.get("battery_kbb_l8_h")) - float(r.get("battery_kbb_l8_a"))
        if abs(advantage) >= CATCHER_BATTERY_REGIME_THRESHOLD:
            return -float(np.sign(advantage)) * CATCHER_BATTERY_REGIME_BIAS
    except Exception:
        pass
    return 0.0


def _is_night_elite(pick_home: bool, home_team: str | None,
                    away_team: str | None, day_night: str | None) -> bool:
    """True iff the pick lands on a night-elite team AND game is a night game."""
    if not home_team or not away_team:
        return False
    if str(day_night or "").lower() != "night":
        return False
    picked = home_team if pick_home else away_team
    return picked in NIGHT_ELITE_TEAMS


def _final_pick_home(c, r, p_home: float) -> bool:
    """Pick FINAL replicando exactamente la lógica del card builder.

    Aplica en orden:
      1. Suma de todos los biases a p_home (h2h + pitcher + streaks + momentum
         + Codex biases + band_calibration)
      2. Threshold season-aware sobre p_home ajustado
      3. Trap flips (team/day/night/pickday/pickmonth)
      4. Series double-down trap (2026+, si contexto lo permite)

    Se usa desde /api/performance para dar el mismo numero que Resultados.
    """
    if not np.isfinite(p_home):
        return False
    ht = r.get("home_team_abbrev")
    at = r.get("away_team_abbrev")
    mp = r.get("market_p_home", float("nan"))
    season = int(r["season"]) if pd.notna(r.get("season")) else None
    gd = r.get("game_date")
    hpid = r.get("probable_home_pitcher_id")
    apid = r.get("probable_away_pitcher_id")
    dn = r.get("day_night")
    try:
        bias = (
            _h2h_bias(ht, at, season)
            + _h2h_venue_regression_bias(c, r, p_home)
            + _pitcher_bias(hpid, apid, season)
            + _home_cold_streak_bias(c, r)
            + _away_hot_streak_bias(c, r)
            + _home_blowout_momentum_bias(c, r)
            + _away_cold_streak_bias(c, r)
            # + _weather_extreme_bias(r)  # RETIRED 2026-07-12 (audit)
            + _home_team_calibration_bias(r, p_home)
            + _home_band_calibration_bias(r, p_home)
            + _interleague_nl_bias(r, p_home)
            # + _umpire_market_bias(r)  # RETIRED 2026-07-17 (audit + sweep: -1 net 2026, ninguna variante rescata)
            + _circadian_extreme_bias(r, p_home)
            + _travel_resilience_bias(r)
            + _pythag_luck_bias(r, p_home)
            + _babip_persistence_bias(r, p_home)
            + _lob_persistence_bias(r, p_home)
            + _cws_night_bias(r, p_home)
            + _comeback_deficit_regression_bias(r, p_home)
            + _runs_median_persistence_bias(r, p_home)
            # + _burn_resilience_bias(r, p_home)  # RETIRED 2026-07-17 (audit + A/B pipeline confirman -3 hits, mantener retirado)
            + _starter_workload_regression_bias(r, p_home)
            + _highlev_fatigue_bias(r, p_home)
            + _catcher_battery_regime_bias(r, p_home)
        )
    except Exception:
        bias = 0.0
    p_adj = float(np.clip(p_home + bias, 0.001, 0.999)) if bias else p_home
    raw = bool(p_adj >= _winner_threshold(mp, p_adj, ht, at, season))
    reason = _flip_reason(raw, ht, at, gd, dn, p_home, mp)
    pick_is_home = (not raw) if reason else raw
    # Series double-down trap (2026+): defer to market cuando el modelo
    # repite un pick previamente fallido en la misma serie.
    try:
        if _series_double_down_market(c, r, pick_is_home):
            if pd.notna(mp):
                pick_is_home = bool(float(mp) > 0.5)
    except Exception:
        pass
    return pick_is_home


def _mil_trap_market_agrees(
    home_team: str | None,
    away_team: str | None,
    model_p_home: float | None,
    market_p_home: float | None,
) -> bool:
    """Allow the MIL correction only when model and market differ by <=7pp."""
    if "MIL" not in {home_team, away_team}:
        return False
    try:
        model_home = float(model_p_home)
        market_home = float(market_p_home)
        if not np.isfinite(model_home) or not np.isfinite(market_home):
            return False
        model_mil = model_home if home_team == "MIL" else 1.0 - model_home
        market_mil = market_home if home_team == "MIL" else 1.0 - market_home
        return abs(model_mil - market_mil) <= MIL_TRAP_MODEL_MARKET_MAX_GAP
    except (TypeError, ValueError):
        return False


def _flip_reason(pick_home: bool, home_team: str | None,
                 away_team: str | None,
                 game_date=None,
                 day_night: str | None = None,
                 model_p_home: float | None = None,
                 market_p_home: float | None = None) -> str | None:
    """Return 'team' if a team trap fires, 'day' if a day-of-week trap fires,
    'night' if a night trap fires (2026+), else None.

    Gate: team/day traps solo aplican season >= 2025. Night trap solo 2026+.
    """
    if not home_team or not away_team:
        return None
    # Season gate — extraer año de game_date si esta disponible.
    season_year = None
    if game_date is not None:
        try:
            season_year = pd.Timestamp(game_date).year
            if season_year < 2025:
                return None
        except Exception:
            pass
    picked = home_team if pick_home else away_team
    opp = away_team if pick_home else home_team
    if picked in TRAP_PICK_TEAMS:
        return "team"
    if opp in TRAP_FADE_TEAMS and (
        opp != "MIL"
        or _mil_trap_market_agrees(
            home_team, away_team, model_p_home, market_p_home
        )
    ):
        return "team"
    # Night trap (2026+ regime rule): pick sobre equipo listado en night game
    if (season_year is None or season_year >= 2026) and \
       str(day_night or "").lower() == "night" and picked in NIGHT_TRAP_FADE:
        return "night"
    if game_date is not None:
        try:
            dow = pd.Timestamp(game_date).dayofweek
            if (opp, dow) in DAY_TRAP_FADE:
                return "day"
            # Pick-day fade (2026+ regime): pick sobre equipo en dia especifico
            if (season_year is None or season_year >= 2026) and \
               (picked, dow) in PICK_DAY_FADE:
                return "pickday"
            # Pick-month fade (2026+ regime): pick sobre equipo en mes especifico
            month = pd.Timestamp(game_date).month
            if (season_year is None or season_year >= 2026) and \
               (picked, month) in PICK_MONTH_FADE:
                return "pickmonth"
        except Exception:
            pass
    return None


def _should_flip_pick(pick_home: bool, home_team: str | None,
                      away_team: str | None,
                      game_date=None,
                      day_night: str | None = None,
                      model_p_home: float | None = None,
                      market_p_home: float | None = None) -> bool:
    """True iff any trap rule fires and the pick should be inverted."""
    return _flip_reason(
        pick_home, home_team, away_team, game_date, day_night,
        model_p_home, market_p_home,
    ) is not None


def _pick_home_from_phome(
    p_home: float,
    market_p_home: float | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    season: int | None = None,
    game_date=None,
    home_pid: int | float | None = None,
    away_pid: int | float | None = None,
) -> bool:
    """Winner decision boundary tuned for binary accuracy, not calibration.

    Applies H2H pair bias + pitcher bias BEFORE the threshold, then trap-team
    + day-of-week flips AFTER. Ver H2H_PICK_BIAS / PITCHER_PICK_BIAS /
    TRAP_*_TEAMS / DAY_TRAP_FADE arriba.
    """
    model_p_home = p_home
    if np.isfinite(p_home):
        bias = _h2h_bias(home_team, away_team, season) + _pitcher_bias(home_pid, away_pid, season)
        if bias:
            p_home = float(np.clip(p_home + bias, 0.001, 0.999))
    raw = bool(
        np.isfinite(p_home)
        and p_home >= _winner_threshold(market_p_home, p_home, home_team, away_team, season)
    )
    if _should_flip_pick(
        raw, home_team, away_team, game_date,
        model_p_home=model_p_home, market_p_home=market_p_home,
    ):
        return not raw
    return raw


# Series double-down trap (2026+ regime rule). When the model repeats the pick
# it just MISSED in the previous game of the same series AND disagrees with the
# market, follow the market instead. Walk-forward 2026 (n=36 disagreement
# spots): model hit 36.1%, market 63.9% — consistent in both halves of the
# season (H1 0.389 vs 0.611, H2 0.333 vs 0.667). Season impact: +0.83pp.
# In 2023-2025 the model was GOOD at doubling down (0.58-0.63), so this is a
# regime rule gated to season >= 2026. Re-validate monthly with the retrain.
# Cache: a played previous game's pick/correctness never changes.
_series_dd_cache: dict[int, tuple[bool, bool]] = {}


def _series_double_down_market(c, r, pick_is_home: bool) -> bool:
    """True iff the series double-down trap fires → pick should follow market."""
    try:
        season = int(r["season"]) if pd.notna(r.get("season")) else None
        if season is None or season < 2026:
            return False
        mp = r.get("market_p_home", float("nan"))
        if mp is None or not np.isfinite(mp):
            return False
        mkt_home = bool(mp > 0.5)
        if pick_is_home == mkt_home:
            return False  # only fires when the model fights the market
        train = c["train"]
        gd = r["game_date"]
        prev = train[
            (train["home_team_abbrev"] == r["home_team_abbrev"])
            & (train["away_team_abbrev"] == r["away_team_abbrev"])
            & (train["game_date"] < gd)
            & (train["game_date"] >= gd - timedelta(days=1))
        ]
        prev = prev[prev["home_win"].notna()]
        if prev.empty:
            return False
        pr = prev.sort_values("game_date").iloc[-1]
        pk = int(pr["game_pk"])
        cached = _series_dd_cache.get(pk)
        if cached is None:
            feats = c["cls"]["feature_names"]
            X = pd.DataFrame([pr[feats].values], columns=feats)
            ph = float(_predict_phome(
                c, X, pd.Series([pr.get("market_p_home", float("nan"))]),
                context_df=pd.DataFrame([pr]))[0])
            prev_pick_home = _pick_home_from_phome(
                ph, pr.get("market_p_home", float("nan")),
                pr.get("home_team_abbrev"), pr.get("away_team_abbrev"),
                int(pr["season"]) if pd.notna(pr.get("season")) else None,
                pr.get("game_date"))
            prev_correct = bool(prev_pick_home) == bool(pr["home_win"])
            cached = (bool(prev_pick_home), prev_correct)
            _series_dd_cache[pk] = cached
        prev_pick_home, prev_correct = cached
        if prev_correct:
            return False
        # Same series → same pair, so "repeats the same team" ⟺ same side.
        return prev_pick_home == pick_is_home
    except Exception:
        return False


# Streak-based biases — 2026 regime rules basadas en mean reversion.
# HOME cold streak: equipo local llega con 3+ derrotas → HOME gana +11pp mas
#   de lo predicho en 2026 (n=89). Sesgo +0.05 a p_home.
# AWAY hot streak: equipo visitante llega con 3+ victorias → HOME gana +8.4pp
#   mas de lo predicho en 2026 (n=91). Sesgo +0.07 a p_home (fade away hot).
# Ambos: en 2023-2024 gaps chicos (mean reversion mercado ya price), 2025
#   INVERTIDOS — regime specific 2026. Impacto medido:
#     HOME cold  → +0.22pp acc 2026 real (interaccion con otros overrides)
#     AWAY hot   → +0.50pp acc 2026 backtest
# Gate 2026+. Revalidar mensual — si signo cambia en próximo backtest, quitar.
HOME_COLD_STREAK_BIAS = 0.05
HOME_COLD_STREAK_THRESHOLD = 3   # 3+ consecutive losses entering the game
AWAY_HOT_STREAK_BIAS = 0.07
AWAY_HOT_STREAK_THRESHOLD = 3    # 3+ consecutive wins entering the game
_home_streak_cache: dict[tuple[str, str, int], int] = {}
_away_streak_cache: dict[tuple[str, str, int], int] = {}


def _team_streak_at_date(c, team: str, season: int, game_date, kind: str, threshold: int,
                         cache: dict) -> bool:
    """True si el equipo llega con >= threshold juegos consecutivos del kind
    ('win' o 'loss') entrando a game_date en la season.
    """
    if not team:
        return False
    train = c["train"]
    prior = train[
        (train["season"] == season)
        & (train["game_date"] < game_date)
        & ((train["home_team_abbrev"] == team) | (train["away_team_abbrev"] == team))
        & train["home_win"].notna()
    ].sort_values("game_date")
    if len(prior) < threshold:
        return False
    last_pk = int(prior.iloc[-1]["game_pk"])
    cache_key = (team, str(season), last_pk)
    hit = cache.get(cache_key)
    if hit is None:
        recent = prior.tail(threshold)
        target = 1 if kind == "win" else 0
        all_match = True
        for _, g in recent.iterrows():
            team_won = (g["home_team_abbrev"] == team and g["home_win"] == 1) or \
                       (g["away_team_abbrev"] == team and g["home_win"] == 0)
            outcome = 1 if team_won else 0
            if outcome != target:
                all_match = False
                break
        hit = all_match
        cache[cache_key] = hit
    return hit


def _home_cold_streak_bias(c, r) -> float:
    """Sesgo +HOME_COLD_STREAK_BIAS si local llega con 3+ derrotas, en 2026+."""
    try:
        season = int(r["season"]) if pd.notna(r.get("season")) else None
        if season is None or season < 2026:
            return 0.0
        cold = _team_streak_at_date(c, r.get("home_team_abbrev"), season,
                                    r["game_date"], "loss",
                                    HOME_COLD_STREAK_THRESHOLD, _home_streak_cache)
        return HOME_COLD_STREAK_BIAS if cold else 0.0
    except Exception:
        return 0.0


def _away_hot_streak_bias(c, r) -> float:
    """Sesgo +AWAY_HOT_STREAK_BIAS a p_home si visitante llega con 3+ victorias,
    en 2026+ (fade away hot → mean reversion favorece HOME)."""
    try:
        season = int(r["season"]) if pd.notna(r.get("season")) else None
        if season is None or season < 2026:
            return 0.0
        hot = _team_streak_at_date(c, r.get("away_team_abbrev"), season,
                                   r["game_date"], "win",
                                   AWAY_HOT_STREAK_THRESHOLD, _away_streak_cache)
        return AWAY_HOT_STREAK_BIAS if hot else 0.0
    except Exception:
        return 0.0


# Away cold streak (5+) — visitante viene tan quebrado que sigue perdiendo
# de visita. 2026: gap +7.4pp (n=90 con thr=3, n=28 con thr=5). Concord 4 años.
#   Impacto: +0.212pp acc 2026 con thr>=5 bias +0.05 (gate 2026+).
AWAY_COLD_STREAK_BIAS = 0.05
AWAY_COLD_STREAK_THRESHOLD = 5  # 5+ derrotas consecutivas del away
_away_cold_streak_cache: dict[tuple[str, str, int], bool] = {}


def _away_cold_streak_bias(c, r) -> float:
    """Sesgo +AWAY_COLD_STREAK_BIAS a p_home si visitante llega con 5+ derrotas
    consecutivas, en 2026+ (equipo en caida libre sigue perdiendo)."""
    try:
        season = int(r["season"]) if pd.notna(r.get("season")) else None
        if season is None or season < 2026:
            return 0.0
        cold = _team_streak_at_date(c, r.get("away_team_abbrev"), season,
                                    r["game_date"], "loss",
                                    AWAY_COLD_STREAK_THRESHOLD, _away_cold_streak_cache)
        return AWAY_COLD_STREAK_BIAS if cold else 0.0
    except Exception:
        return 0.0


# Home blowout W momentum — cuando el local viene de ganar por 8+ carreras,
# HOME sigue ganando mas de lo predicho (momentum ganador se sostiene, mercado
# no ajusta lo suficiente). Concordante en las 4 seasons (gaps +1.9/+2.0/+7.1/+8.5pp).
#   Gate 2025+ (en 2023-2024 aplicar globalmente restaba pese al signal —
#   probablemente colisiona con la calidad del modelo naive en ese regime).
#   Impacto medido: +0.113pp global, +0.202pp 2025, +0.354pp 2026 (n=63).
# Revalidar mensual. Si el gap cae bajo +4pp en 2026, quitar.
HOME_BLOWOUT_MOMENTUM_BIAS = 0.07
HOME_BLOWOUT_MARGIN_THRESHOLD = 8  # ganado por 8+ carreras el juego previo
_home_prev_margin_cache: dict[tuple[str, str, int], float] = {}


def _home_blowout_momentum_bias(c, r) -> float:
    """Sesgo +HOME_BLOWOUT_MOMENTUM_BIAS si el local viene de blowout win
    (8+ carreras) el juego previo. Gate season >= 2025."""
    try:
        season = int(r["season"]) if pd.notna(r.get("season")) else None
        if season is None or season < 2025:
            return 0.0
        home_team = r.get("home_team_abbrev")
        if not home_team:
            return 0.0
        gd = r["game_date"]
        train = c["train"]
        prior = train[
            (train["season"] == season)
            & (train["game_date"] < gd)
            & ((train["home_team_abbrev"] == home_team) | (train["away_team_abbrev"] == home_team))
            & train["home_score"].notna()
        ].sort_values("game_date")
        if prior.empty:
            return 0.0
        last_pk = int(prior.iloc[-1]["game_pk"])
        cache_key = (home_team, str(season), last_pk)
        margin = _home_prev_margin_cache.get(cache_key)
        if margin is None:
            g = prior.iloc[-1]
            hs = float(g["home_score"])
            as_ = float(g["away_score"])
            if g["home_team_abbrev"] == home_team:
                margin = hs - as_
            else:
                margin = as_ - hs
            _home_prev_margin_cache[cache_key] = margin
        return HOME_BLOWOUT_MOMENTUM_BIAS if margin >= HOME_BLOWOUT_MARGIN_THRESHOLD else 0.0
    except Exception:
        return 0.0


# Confidence tiers — RECALIBRADO 2026-07-09 sobre 2025-2026 (n=3,876).
# El modelo drifta: la banda "moderado" del regimen viejo pegaba 60% pero
# en 25-26 solo pega 55%. Rehacemos con umbrales mas altos para que cada
# tier refleje su hit rate REAL en el regimen actual.
#   |p-0.5| >= 0.145 → 72.4% hit rate  → LOCK      (4.4% coverage)
#   |p-0.5| >= 0.115 → 66.1%           → FUERTE    (12.8% coverage)
#   |p-0.5| >= 0.045 → 58.0%           → MODERADO  (57.4% coverage)
#   below            → 52.6%           → PAREJO    (42.6% coverage — coin flip)
# Re-calibrar cada 2 meses o tras cambio de modelo.
_TIER_BANDS = [
    (0.145, "lock",     72.4),
    (0.115, "fuerte",   66.1),
    (0.045, "moderado", 58.0),
    (0.0,   "parejo",   52.6),
]


def _confidence_tier(p_home: float) -> dict:
    """Map a display probability to a confidence tier + historical hit rate."""
    if not np.isfinite(p_home):
        return {"confidence_pp": None, "confidence_tier": None, "tier_hit_rate": None}
    conf = abs(p_home - 0.5)
    for thr, name, hit in _TIER_BANDS:
        if conf >= thr:
            return {"confidence_pp": round(conf * 100, 1),
                    "confidence_tier": name,
                    "tier_hit_rate": hit}
    return {"confidence_pp": round(conf * 100, 1),
            "confidence_tier": "parejo", "tier_hit_rate": 52.6}


def _to_american(p: float) -> int:
    if not np.isfinite(p) or p <= 0 or p >= 1:
        return 0
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def _to_decimal(p: float) -> float | None:
    """Probability → decimal odds (no vig). 0.50 → 2.00, 0.40 → 2.50, 0.60 → 1.67."""
    if not np.isfinite(p) or p <= 0 or p >= 1:
        return None
    return round(1.0 / p, 2)


def _american_to_decimal(ml: float | None) -> float | None:
    if ml is None or not np.isfinite(ml) or ml == 0:
        return None
    return round((ml / 100 + 1) if ml > 0 else (100 / (-ml) + 1), 2)


def _implied_from_american(ml: float | None) -> float | None:
    if ml is None or not np.isfinite(ml) or ml == 0:
        return None
    return -ml / (-ml + 100) if ml < 0 else 100.0 / (ml + 100)


def _safe(v):
    """JSON-safe: NaN/inf -> None, numpy scalars -> native."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _read_cached_feed(game_pk: int) -> dict | None:
    for y in (2026, 2025, 2024, 2023):
        p = RAW / "statsapi_feed" / str(y) / f"{game_pk}.json"
        if p.exists():
            return orjson.loads(p.read_bytes())
    return None


def _write_cached_feed(game_pk: int, feed: dict) -> None:
    official = (((feed.get("gameData") or {}).get("datetime") or {}).get("officialDate"))
    year = int((official or date.today().isoformat())[:4])
    out = RAW / "statsapi_feed" / str(year) / f"{game_pk}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_bytes(orjson.dumps(feed))
    tmp.replace(out)
    _freshness_cache.update({"ts": 0.0, "data": None})


def _feed_freshness_summary() -> dict:
    try:
        now = time.time()
        if _freshness_cache["data"] is not None and now - _freshness_cache["ts"] < _FRESHNESS_TTL_SEC:
            return dict(_freshness_cache["data"])
        c = _load()
        train = c["train"]
        today = date.today()
        window_start = today - timedelta(days=7)
        past = train[(train["game_date"] < today) & (train["game_date"] >= window_start)]
        stale = 0
        missing = 0
        checked = 0
        latest = None
        for _, row in past.iterrows():
            checked += 1
            game_date = row["game_date"]
            latest = game_date if latest is None or game_date > latest else latest
            feed = _read_cached_feed(int(row["game_pk"]))
            if feed is None:
                missing += 1
            elif not _game_state(feed).get("is_final"):
                stale += 1
        data = {
            "feed_checked_past": checked,
            "feed_stale_past": stale,
            "feed_missing_past": missing,
            "feed_latest_past_date": latest.isoformat() if latest else None,
            "feed_fresh": stale == 0 and missing == 0,
        }
        _freshness_cache.update({"ts": now, "data": data})
        return dict(data)
    except Exception as ex:
        return {"feed_fresh": None, "feed_freshness_error": repr(ex)}


_live_cache: dict = {}  # game_pk → (timestamp, feed_json)
_LIVE_TTL_SEC = 25


def _fetch_live_feed(game_pk: int) -> dict | None:
    """Fetch fresh feed/live from StatsAPI with a short TTL cache.

    25-second TTL is short enough to keep scores current but long enough that
    13 concurrent slate requests cost only 13 HTTP calls per ~30s.
    """
    import time
    now = time.time()
    cached = _live_cache.get(game_pk)
    if cached and now - cached[0] < _LIVE_TTL_SEC:
        return cached[1]
    req = Request(
        STATSAPI_LIVE_URL.format(game_pk=game_pk),
        headers={"User-Agent": "mlb-slec-live/0.1", "Accept": "application/json"},
    )
    try:
        with urlopen(req, timeout=6) as response:
            data = orjson.loads(response.read())
            _live_cache[game_pk] = (now, data)
            state = _game_state(data)
            if state.get("is_final"):
                _write_cached_feed(game_pk, data)
            return data
    except Exception:
        return None


# Memoized per-process xref for fast game_pk → espn_event_id lookups.
_xref_cache: dict[int, str] = {}


def _espn_event_id_for(game_pk: int) -> str | None:
    if not _xref_cache:
        try:
            xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
            xref = xref[xref["_merge"] == "both"][["game_pk", "espn_event_id"]].dropna()
            for r in xref.itertuples(index=False):
                _xref_cache[int(r.game_pk)] = str(r.espn_event_id)
        except Exception:
            return None
    return _xref_cache.get(int(game_pk))


def _live_p_home(game_pk: int, season: int | None) -> float | None:
    """Latest homeWinPercentage from the cached ESPN probabilities JSON.

    None if no ESPN data yet (game in pregame or never ingested). The on-request
    refresh re-downloads this file for live games so it's reasonably fresh.
    """
    if season is None:
        return None
    eid = _espn_event_id_for(game_pk)
    if not eid:
        return None
    prob_file = RAW / "probabilities" / str(season) / f"{eid}.json"
    if not prob_file.exists():
        return None
    try:
        payload = orjson.loads(prob_file.read_bytes())
    except Exception:
        return None
    items = payload.get("items") or []
    if not items:
        return None
    # Last item by sequenceNumber wins.
    latest = max(items, key=lambda it: int(it.get("sequenceNumber") or 0))
    wp = latest.get("homeWinPercentage")
    return round(float(wp), 4) if wp is not None else None


def _game_state(feed: dict | None) -> dict:
    status = ((feed or {}).get("gameData") or {}).get("status") or {}
    abstract = status.get("abstractGameState")
    coded = status.get("codedGameState")
    return {
        "abstract": abstract,
        "coded": coded,
        "detailed": status.get("detailedState"),
        "is_live": abstract == "Live" or coded in {"I", "M", "N"},
        "is_final": abstract == "Final" or coded == "F",
    }


def _live_linescore(feed: dict | None) -> dict:
    if not feed:
        return {
            "state": _game_state(feed),
            "inning": None,
            "inning_state": None,
            "balls": None,
            "strikes": None,
            "outs": None,
            "bases": {"first": None, "second": None, "third": None},
            "away": {"runs": None, "hits": None, "errors": None, "left_on_base": None},
            "home": {"runs": None, "hits": None, "errors": None, "left_on_base": None},
            "innings": [],
        }
    linescore = (feed.get("liveData") or {}).get("linescore") or {}
    teams = linescore.get("teams") or {}
    offense = linescore.get("offense") or {}

    def _base_runner(base: str) -> dict | None:
        runner = offense.get(base)
        if not runner:
            return None
        return {
            "id": _safe(runner.get("id")),
            "name": runner.get("fullName") or runner.get("name"),
        }

    def _team_totals(side: str) -> dict:
        t = teams.get(side) or {}
        return {
            "runs": _safe(t.get("runs")),
            "hits": _safe(t.get("hits")),
            "errors": _safe(t.get("errors")),
            "left_on_base": _safe(t.get("leftOnBase")),
        }

    innings = []
    for inn in linescore.get("innings") or []:
        away = inn.get("away") or {}
        home = inn.get("home") or {}
        innings.append({
            "num": _safe(inn.get("num")),
            "ordinal": inn.get("ordinalNum"),
            "away": {
                "runs": _safe(away.get("runs")),
                "hits": _safe(away.get("hits")),
                "errors": _safe(away.get("errors")),
                "left_on_base": _safe(away.get("leftOnBase")),
            },
            "home": {
                "runs": _safe(home.get("runs")),
                "hits": _safe(home.get("hits")),
                "errors": _safe(home.get("errors")),
                "left_on_base": _safe(home.get("leftOnBase")),
            },
        })

    return {
        "state": _game_state(feed),
        "inning": _safe(linescore.get("currentInning")),
        "inning_ordinal": linescore.get("currentInningOrdinal"),
        "inning_state": linescore.get("inningState"),
        "inning_half": linescore.get("inningHalf"),
        "balls": _safe(linescore.get("balls")),
        "strikes": _safe(linescore.get("strikes")),
        "outs": _safe(linescore.get("outs")),
        "bases": {
            "first": _base_runner("first"),
            "second": _base_runner("second"),
            "third": _base_runner("third"),
        },
        "away": _team_totals("away"),
        "home": _team_totals("home"),
        "innings": innings,
    }


def _starter_pitching_line(feed: dict | None, side: str, probable_id: int | None = None) -> dict | None:
    if not feed:
        return None
    team_box = (((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}).get(side) or {}
    players = team_box.get("players") or {}
    pitcher_ids = team_box.get("pitchers") or []
    ordered_ids = []
    if probable_id:
        ordered_ids.append(int(probable_id))
    ordered_ids.extend(int(pid) for pid in pitcher_ids if pid and int(pid) not in ordered_ids)

    for pid in ordered_ids:
        player = players.get(f"ID{pid}") or {}
        pitching = ((player.get("stats") or {}).get("pitching") or {})
        if not pitching:
            continue
        if _safe(pitching.get("outs")) in (None, 0) and str(pitching.get("inningsPitched") or "0.0") == "0.0":
            continue
        return {
            "id": pid,
            "name": (player.get("person") or {}).get("fullName"),
            "summary": pitching.get("summary"),
            "innings_pitched": pitching.get("inningsPitched"),
            "hits": _safe(pitching.get("hits")),
            "runs": _safe(pitching.get("runs")),
            "earned_runs": _safe(pitching.get("earnedRuns")),
            "base_on_balls": _safe(pitching.get("baseOnBalls")),
            "strike_outs": _safe(pitching.get("strikeOuts")),
            "home_runs": _safe(pitching.get("homeRuns")),
            "pitches": _safe(pitching.get("numberOfPitches")),
            "strikes": _safe(pitching.get("strikes")),
            "balls": _safe(pitching.get("balls")),
            "batters_faced": _safe(pitching.get("battersFaced")),
            "decision": (
                "W" if _safe(pitching.get("wins")) else
                "L" if _safe(pitching.get("losses")) else
                None
            ),
        }
    return None


def _predict_row(row: pd.Series, c) -> tuple[float, float]:
    X = pd.DataFrame([row[c["cls"]["feature_names"]].values],
                     columns=c["cls"]["feature_names"])
    market_p = pd.Series([row.get("market_p_home", np.nan)])
    p_home = float(_predict_phome(c, X, market_p, context_df=pd.DataFrame([r]))[0])
    total = float(c["reg"]["model"].predict(X)[0])
    return p_home, total


# ---------------- endpoints ----------------
@app.get("/api/team-recent-form")
def team_recent_form(team: str, n: int = 5):
    """Last n *played* games for a team, as W/L results (chronological order,
    oldest → newest). Used by the predictions dashboard's RECENT FORM column."""
    c = _load()
    team_u = (team or "").strip().upper()
    if not team_u:
        raise HTTPException(400, "team is required")
    n = max(1, min(int(n), 20))
    train = c["train"]
    sub = train[
        ((train["away_team_abbrev"].astype(str).str.upper() == team_u)
         | (train["home_team_abbrev"].astype(str).str.upper() == team_u))
        & train["home_score"].notna()
        & train["away_score"].notna()
    ].sort_values("game_date").tail(n)
    out = []
    for _, r in sub.iterrows():
        is_home = str(r["home_team_abbrev"]).upper() == team_u
        hs, as_ = float(r["home_score"]), float(r["away_score"])
        won = (is_home and hs > as_) or ((not is_home) and as_ > hs)
        out.append({
            "date": r["game_date"].isoformat(),
            "opp": r["away_team_abbrev"] if is_home else r["home_team_abbrev"],
            "home": bool(is_home),
            "score_for": int(hs if is_home else as_),
            "score_against": int(as_ if is_home else hs),
            "result": "W" if won else "L",
        })
    return {"team": team_u, "n": len(out), "games": out}


@app.get("/api/team-ranking")
def team_ranking(days: int = 60):
    c = _load()
    today = date.today()
    safe_days = max(1, min(days, 180))
    start = today - timedelta(days=safe_days)
    train = c["train"]
    sub = train[
        (train["game_date"] >= start)
        & (train["game_date"] <= today)
        & train["home_score"].notna()
        & train["away_score"].notna()
    ].copy()
    if sub.empty:
        return {"days": safe_days, "start": start.isoformat(), "end": today.isoformat(), "games": 0, "teams": []}

    feats = c["cls"]["feature_names"]
    probabilities = _predict_phome(c, sub[feats], sub["market_p_home"], context_df=sub)
    seasons_arr = (
        pd.to_numeric(sub.get("season"), errors="coerce")
        if "season" in sub.columns
        else pd.to_datetime(sub["game_date"]).dt.year
    ).to_numpy()
    thresholds = np.array([
        _winner_threshold(mp, ph, ht, at, int(s) if pd.notna(s) else None)
        for mp, ph, ht, at, s in zip(
            sub["market_p_home"].to_numpy(dtype=float),
            probabilities,
            sub["home_team_abbrev"].astype(str),
            sub["away_team_abbrev"].astype(str),
            seasons_arr,
        )
    ], dtype=float)
    sub["model_won"] = (probabilities >= thresholds) == (sub["home_score"] > sub["away_score"])
    sub["model_team"] = np.where(probabilities >= thresholds, sub["home_team_abbrev"], sub["away_team_abbrev"])

    def _ranking_rate(series: pd.Series) -> float | None:
        return round(float(series.mean() * 100), 1) if len(series) else None

    rows = []
    teams = sorted(set(sub["away_team_abbrev"].dropna()) | set(sub["home_team_abbrev"].dropna()))
    for team in teams:
        games_for_team = sub[(sub["away_team_abbrev"] == team) | (sub["home_team_abbrev"] == team)]
        picked = games_for_team[games_for_team["model_team"] == team]
        correct = int(games_for_team["model_won"].sum())
        rows.append({
            "team": team,
            "games": int(len(games_for_team)),
            "correct": correct,
            "incorrect": int(len(games_for_team) - correct),
            "accuracy": _ranking_rate(games_for_team["model_won"]),
            "times_picked": int(len(picked)),
            "picked_accuracy": _ranking_rate(picked["model_won"]),
        })
    rows.sort(key=lambda row: (-(row["accuracy"] or -1), -row["games"], row["team"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {"days": safe_days, "start": start.isoformat(), "end": today.isoformat(), "games": int(len(sub)), "teams": rows}


@app.get("/api/performance")
def performance(days: int = 60):
    c = _load()
    train = c["train"].copy()
    games_df = c["games"][["game_pk", "first_pitch_utc", "day_night"]].copy()
    today = date.today()
    start = today - timedelta(days=max(1, min(days, 180)))
    sub = train[
        (train["game_date"] >= start)
        & (train["game_date"] <= today)
        & train["home_score"].notna()
        & train["away_score"].notna()
    ].copy()
    if sub.empty:
        return {
            "days": days,
            "start": start.isoformat(),
            "end": today.isoformat(),
            "summary": {},
            "teams": [],
            "weekly": [],
            "grades": [],
            "recent": [],
        }

    sub = sub.merge(games_df, on="game_pk", how="left")
    sub["first_pitch_utc"] = pd.to_datetime(sub["first_pitch_utc"], utc=True, errors="coerce")
    sub["_sort_time"] = sub["first_pitch_utc"].fillna(pd.Timestamp("2100-01-01", tz="UTC"))
    sub = sub.sort_values(["game_date", "_sort_time", "game_pk"]).drop(columns=["_sort_time"])
    sub["et_date"] = sub["first_pitch_utc"].dt.tz_convert("America/New_York").dt.date
    sub["slot"] = sub.groupby("et_date").cumcount() + 1

    feats = c["cls"]["feature_names"]
    # ─── Batch predict once (SeedEnsemble is Nx slower per call → batch is critical) ───
    sub = sub.reset_index(drop=True)
    X_all = sub[feats]
    try:
        p_home_all = _predict_phome(c, X_all, sub["market_p_home"], context_df=sub)
    except Exception:
        p_home_all = np.full(len(sub), float("nan"))

    rows = []
    for idx in range(len(sub)):
        r = sub.iloc[idx]
        p_home = float(p_home_all[idx])
        if not np.isfinite(p_home):
            continue
        home_won = bool(r["home_score"] > r["away_score"])
        # Usa _final_pick_home para replicar EXACTAMENTE la logica del card
        # builder (todos los biases + traps + series DD). Antes usaba
        # _pick_home_from_phome que solo tenia threshold+h2h+traps, dando
        # numeros inconsistentes con Resultados. Fix 2026-07-12.
        model_side = "HOME" if _final_pick_home(c, r, p_home) else "AWAY"
        model_won = (model_side == "HOME") == home_won
        market_side = "HOME" if pd.notna(r.get("market_p_home")) and float(r.get("market_p_home")) >= 0.5 else "AWAY"
        market_won = (market_side == "HOME") == home_won if pd.notna(r.get("market_p_home")) else None

        v = _value_block(
            c,
            int(r["game_pk"]),
            p_home,
            r["away_team_abbrev"],
            r["home_team_abbrev"],
            slot=int(r["slot"]) if pd.notna(r.get("slot")) else None,
            context=r,
        )
        model_dec = v.get("market_home_decimal") if model_side == "HOME" else v.get("market_away_decimal")
        value_side = v.get("value_side")
        value_dec = v.get("value_decimal")
        is_value = value_side in ("HOME", "AWAY") and v.get("ev_grade") in ("sweet", "marginal")
        value_won = ((value_side == "HOME") == home_won) if is_value else None
        # display_won matchea lo que la card muestra al usuario en Resultados:
        # cuando hay pick de valor sweet/marginal, usa ese; si no, usa el pick del modelo.
        # Asi el % en Resumen concuerda con el ACCURACY DEL MODELO de Resultados.
        display_side = value_side if is_value else model_side
        display_won = (display_side == "HOME") == home_won
        rows.append({
            "game_pk": int(r["game_pk"]),
            "date": r["game_date"],
            "away": r["away_team_abbrev"],
            "home": r["home_team_abbrev"],
            "score": f"{r['away_team_abbrev']} {int(r['away_score'])} · {int(r['home_score'])} {r['home_team_abbrev']}",
            "p_home": p_home,
            "model_side": model_side,
            "model_team": r["home_team_abbrev"] if model_side == "HOME" else r["away_team_abbrev"],
            "model_won": model_won,
            "display_won": display_won,
            "model_profit": (float(model_dec) - 1.0 if model_won else -1.0) if model_dec else None,
            "market_won": market_won,
            "value_side": value_side,
            "value_team": v.get("value_team_abbrev"),
            "value_won": value_won,
            "value_profit": (float(value_dec) - 1.0 if value_won else -1.0) if is_value and value_dec else None,
            "ev_grade": v.get("ev_grade") or "no_data",
            "reason": v.get("ev_grade_reason") or "base",
            "edge": v.get("value_edge_pp"),
            "decimal": value_dec,
        })

    if not rows:
        return {"days": days, "start": start.isoformat(), "end": today.isoformat(), "summary": {}, "teams": [], "weekly": [], "grades": [], "recent": []}

    df = pd.DataFrame(rows)

    def _rate(series) -> float | None:
        s = series.dropna()
        return round(float(s.mean() * 100), 1) if len(s) else None

    def _roi(series) -> float | None:
        s = series.dropna()
        return round(float(s.sum() / len(s) * 100), 1) if len(s) else None

    value_df = df[df["value_profit"].notna()].copy()
    model_df = df[df["model_profit"].notna()].copy()
    market_df = df[df["market_won"].notna()].copy()

    team_rows = []
    team_abbrevs = sorted(set(df["away"].dropna()) | set(df["home"].dropna()))
    for team in team_abbrevs:
        games_for_team = df[(df["away"] == team) | (df["home"] == team)].copy()
        settled = games_for_team[games_for_team["model_won"].notna()]
        picked = games_for_team[games_for_team["model_team"] == team]
        correct = int(settled["model_won"].sum())
        team_rows.append({
            "team": team,
            "games": int(len(settled)),
            "correct": correct,
            "incorrect": int(len(settled) - correct),
            "accuracy": _rate(settled["model_won"]),
            "times_picked": int(len(picked)),
            "picked_accuracy": _rate(picked["model_won"]),
        })
    team_rows.sort(key=lambda row: (-(row["accuracy"] or -1), -row["games"], row["team"]))
    for rank, row in enumerate(team_rows, start=1):
        row["rank"] = rank

    weekly = []
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W-SUN").astype(str)
    for week, g in df.groupby("week", sort=True):
        vg = g[g["value_profit"].notna()]
        mg = g[g["model_profit"].notna()]
        weekly.append({
            "week": week,
            "games": int(len(g)),
            "model_accuracy": _rate(g["display_won"]),
            "market_accuracy": _rate(g["market_won"]),
            "value_picks": int(len(vg)),
            "value_accuracy": _rate(vg["value_won"]) if len(vg) else None,
            "value_roi": _roi(vg["value_profit"]) if len(vg) else None,
            "model_roi": _roi(mg["model_profit"]) if len(mg) else None,
        })

    daily = []
    for game_day, g in df.groupby("date", sort=False):
        vg = g[g["value_profit"].notna()]
        mg = g[g["model_profit"].notna()]
        daily.append({
            "date": game_day.isoformat(),
            "games": int(len(g)),
            "model_picks": int(len(mg)),
            "model_accuracy": _rate(g["display_won"]),
            "model_roi": _roi(mg["model_profit"]) if len(mg) else None,
            "market_accuracy": _rate(g["market_won"]),
            "value_picks": int(len(vg)),
            "value_accuracy": _rate(vg["value_won"]) if len(vg) else None,
            "value_roi": _roi(vg["value_profit"]) if len(vg) else None,
            "value_profit_units": round(float(vg["value_profit"].sum()), 2) if len(vg) else 0.0,
        })

    grades = []
    for grade, g in df.groupby("ev_grade", sort=True):
        vg = g[g["value_profit"].notna()]
        grades.append({
            "grade": grade,
            "games": int(len(g)),
            "picks": int(len(vg)),
            "accuracy": _rate(vg["value_won"]) if len(vg) else None,
            "roi": _roi(vg["value_profit"]) if len(vg) else None,
        })

    recent = df.sort_values("date", ascending=False).head(12)
    return {
        "days": days,
        "start": start.isoformat(),
        "end": today.isoformat(),
        "summary": {
            "games": int(len(df)),
            "model_picks": int(len(model_df)),
            "model_accuracy": _rate(df["display_won"]),
            "model_roi": _roi(model_df["model_profit"]),
            "market_accuracy": _rate(market_df["market_won"]),
            "value_picks": int(len(value_df)),
            "value_accuracy": _rate(value_df["value_won"]) if len(value_df) else None,
            "value_roi": _roi(value_df["value_profit"]) if len(value_df) else None,
            "value_profit_units": round(float(value_df["value_profit"].sum()), 2) if len(value_df) else 0.0,
        },
        "teams": team_rows,
        "weekly": weekly,
        "daily": daily,
        "grades": grades,
        "recent": [
            {
                "game_pk": int(r.game_pk),
                "date": r.date.isoformat(),
                "matchup": f"{r.away}@{r.home}",
                "score": r.score,
                "pick": r.value_team or r.model_team,
                "grade": r.ev_grade,
                "reason": r.reason,
                "won": bool(r.value_won) if pd.notna(r.value_won) else bool(r.model_won),
                "profit": _safe(r.value_profit if pd.notna(r.value_profit) else r.model_profit),
            }
            for r in recent.itertuples(index=False)
        ],
    }


@app.get("/api/games")
async def games(start: str | None = None, end: str | None = None, days: int = 7, team: str | None = None):
    c = _load()
    train = c["train"]
    today = date.today()
    s = date.fromisoformat(start) if start else today
    e = date.fromisoformat(end) if end else (s + timedelta(days=days))
    team_filter = team.strip().upper() if isinstance(team, str) and team.strip() else None
    # If the window touches today/future, kick off a background refresh so the
    # NEXT request sees fresh odds. We never block the current response on it.
    if e >= today:
        await _maybe_refresh_now(blocking=False)
    # Response cache — fast page reloads avoid the predict + live-feed loop.
    cache_key = (s.isoformat(), e.isoformat(), team_filter)
    cached = _games_response_cache.get(cache_key)
    if cached is not None and (time.time() - cached[0]) < _GAMES_CACHE_TTL_SEC:
        return cached[1]
    sub = train[(train["game_date"] >= s) & (train["game_date"] <= e)].copy()
    # Keep the complete slate while predicting. Slate-level accuracy rules must
    # produce the same pick when this endpoint is filtered to a single team.
    # Merge first_pitch_utc from games.parquet so the response can carry the
    # scheduled start time and the slot-within-day.
    games_df = c["games"][["game_pk", "first_pitch_utc", "day_night"]]
    sub = sub.merge(games_df, on="game_pk", how="left")
    sub["first_pitch_utc"] = pd.to_datetime(sub["first_pitch_utc"], utc=True, errors="coerce")
    # Sort by start time within each date; games without a known start time
    # sink to the bottom (rare — usually only TBD playoff stuff).
    sub["_sort_time"] = sub["first_pitch_utc"].fillna(pd.Timestamp("2100-01-01", tz="UTC"))
    sub = sub.sort_values(["game_date", "_sort_time", "game_pk"]).drop(columns=["_sort_time"])
    # Per-day slot index (1 = first game of the day in ET).
    sub["et_date"] = sub["first_pitch_utc"].dt.tz_convert("America/New_York").dt.date
    sub["slot"] = sub.groupby("et_date").cumcount() + 1
    sub["games_in_day"] = sub.groupby("et_date")["game_pk"].transform("size")
    out = []
    feats = c["cls"]["feature_names"]

    # Parallel-prefetch fresh live state, but ONLY for games that are still
    # in progress or upcoming. FINAL games never change again — fetching them
    # was the biggest source of dead latency on past-date views.
    from concurrent.futures import ThreadPoolExecutor
    refresh_pks: list[int] = []
    for _, r in sub.iterrows():
        if r["game_date"] < today - timedelta(days=1):
            continue
        cached_feed = _read_cached_feed(int(r["game_pk"]))
        if cached_feed is not None and _game_state(cached_feed).get("is_final"):
            continue
        refresh_pks.append(int(r["game_pk"]))
    if refresh_pks:
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_fetch_live_feed, refresh_pks))
    for _, r in sub.iterrows():
        X = pd.DataFrame([r[feats].values], columns=feats)
        try:
            # Display/accuracy probability (market-heavy blend)
            p_home = float(_predict_phome(c, X, pd.Series([r.get("market_p_home", float("nan"))]), context_df=pd.DataFrame([r]))[0])
            # Value/edge probability (model-only, market-free) — for _value_block
            p_home_value = float(_raw_model_phome(c, X)[0])
            total = float(c["reg"]["model"].predict(X)[0])
        except Exception as e:
            import traceback
            print(f"[predict ERROR for pk={r['game_pk']}]: {type(e).__name__}: {e}")
            traceback.print_exc()
            p_home, p_home_value, total = float("nan"), float("nan"), float("nan")
        # For today's slate AND the day before (in case a late game spilled over),
        # always try to fetch fresh state from StatsAPI; cached file is stale by
        # the time the game starts. The TTL cache inside _fetch_live_feed keeps
        # this from being expensive.
        cached_feed = _read_cached_feed(int(r["game_pk"]))
        cached_state = _game_state(cached_feed)
        stale_past_feed = r["game_date"] < today and not cached_state.get("is_final")
        if r["game_date"] >= today - timedelta(days=1) or stale_past_feed:
            fresh = _fetch_live_feed(int(r["game_pk"]))
            if fresh is not None:
                cached_feed = fresh
        live = _live_linescore(cached_feed)
        # Pitcher names from already-loaded cached_feed — zero extra I/O
        _away_pid = _safe(r.get("probable_away_pitcher_id"))
        _home_pid = _safe(r.get("probable_home_pitcher_id"))
        def _pnq(pid, feed=cached_feed):
            if not pid or feed is None:
                return {"name": None, "throws": None}
            try:
                key = f"ID{int(pid)}"
                ps = (feed.get("gameData") or {}).get("players") or {}
                p = ps.get(key) or {}
                return {"name": p.get("fullName"), "throws": (p.get("pitchHand") or {}).get("code")}
            except Exception:
                return {"name": None, "throws": None}
        pitcher_away = _pnq(_away_pid)
        pitcher_home = _pnq(_home_pid)
        gs = live["state"]
        is_final = bool(gs.get("is_final"))
        is_live = bool(gs.get("is_live"))
        # Authoritative scores: from the feed if live or final; else suppress.
        if is_final or is_live:
            away_score = live["away"]["runs"]
            home_score = live["home"]["runs"]
        else:
            away_score = None
            home_score = None
        # Pick label = team abbreviation, not HOME/AWAY.
        pick_abbrev = None
        flip_reason = None
        h2h_applied = False
        h2h_venue_regression_applied = False
        pitcher_applied = False
        cold_streak_applied = False
        hot_streak_applied = False
        blowout_applied = False
        weather_extreme_applied = False
        home_calibration_applied = False
        home_band_calib_applied = False
        interleague_applied = False
        umpire_market_applied = False
        circadian_applied = False
        travel_resilience_applied = False
        pythag_luck_applied = False
        babip_persistence_applied = False
        lob_persistence_applied = False
        cws_night_applied = False
        comeback_deficit_applied = False
        runs_median_applied = False
        burn_resilience_applied = False
        starter_workload_applied = False
        highlev_fatigue_applied = False
        catcher_battery_applied = False
        if np.isfinite(p_home):
            season_int = int(r["season"]) if pd.notna(r.get("season")) else None
            mp = r.get("market_p_home", float("nan"))
            ht = r.get("home_team_abbrev")
            at = r.get("away_team_abbrev")
            gd = r.get("game_date")
            # Streak/momentum biases + H2H pair bias + pitcher bias — sesga p_home antes del umbral
            h2h = _h2h_bias(ht, at, season_int)
            h2h_venue_regression = _h2h_venue_regression_bias(c, r, p_home)
            pbias = _pitcher_bias(_home_pid, _away_pid, season_int)
            cold_bias = _home_cold_streak_bias(c, r)
            hot_bias = _away_hot_streak_bias(c, r)
            blowout_bias = _home_blowout_momentum_bias(c, r)
            away_cold_bias = _away_cold_streak_bias(c, r)
            # Codex overrides — validacion completa 2026-07-10:
            #  ✓ stl_calibration +45.5pp acc en el bucket (n=11)
            #  ✓ interleague     +12.9pp acc (n=116, 31 flippeos)
            #  ✓ umpire_market   +9.3pp acc (n=86, 10 flippeos)
            #  ✓ travel_resil    +10.0pp acc (n=60, 14 flippeos)
            # Audit override 2026-07-12: weather_extreme RETIRADO — perdia
            # en las 3 ventanas (30d 1/3 -1net, 60d 1/4 -2net, 2026 1/4 -2net).
            # Codigo/funcion se mantiene para revalidar en el futuro.
            # circadian y otros mantienen contribucion marginal por interaccion
            # con downstream rules.
            weather_extreme_bias = 0.0  # RETIRED 2026-07-12 (audit)
            home_calibration_bias = _home_team_calibration_bias(r, p_home)
            home_band_calib_bias = _home_band_calibration_bias(r, p_home)
            interleague_bias = _interleague_nl_bias(r, p_home)
            umpire_market_bias = 0.0  # RETIRED 2026-07-17 (audit + sweep: -1 net 2026, ninguna variante rescata)
            circadian_bias = _circadian_extreme_bias(r, p_home)
            travel_resilience_bias = _travel_resilience_bias(r)
            pythag_luck_bias = _pythag_luck_bias(r, p_home)
            babip_persistence_bias = _babip_persistence_bias(r, p_home)
            lob_persistence_bias = _lob_persistence_bias(r, p_home)
            cws_night_bias = _cws_night_bias(r, p_home)
            comeback_deficit_bias = _comeback_deficit_regression_bias(r, p_home)
            runs_median_bias = _runs_median_persistence_bias(r, p_home)
            burn_resilience_bias = 0.0  # RETIRED 2026-07-17 (A/B pipeline confirmó -3 hits al restaurar)
            starter_workload_bias = _starter_workload_regression_bias(r, p_home)
            highlev_fatigue_bias = _highlev_fatigue_bias(r, p_home)
            catcher_battery_bias = _catcher_battery_regime_bias(r, p_home)
            bias = (h2h + h2h_venue_regression + pbias + cold_bias + hot_bias + blowout_bias + away_cold_bias
                    + weather_extreme_bias + home_calibration_bias + home_band_calib_bias
                    + interleague_bias + umpire_market_bias + circadian_bias
                    + travel_resilience_bias + pythag_luck_bias + babip_persistence_bias
                    + lob_persistence_bias + cws_night_bias)
            bias += comeback_deficit_bias
            bias += runs_median_bias
            bias += burn_resilience_bias
            bias += starter_workload_bias
            bias += highlev_fatigue_bias
            bias += catcher_battery_bias
            p_home_biased = float(np.clip(p_home + bias, 0.001, 0.999)) if bias else p_home
            raw_pick_home = bool(p_home_biased >= _winner_threshold(mp, p_home_biased, ht, at, season_int))
            # flags: se prenden si el sesgo cambio el pick raw
            raw_no_bias = bool(p_home >= _winner_threshold(mp, p_home, ht, at, season_int))
            if h2h:
                h2h_applied = raw_pick_home != raw_no_bias
            if h2h_venue_regression:
                p_without_h2h_venue = float(np.clip(
                    p_home + bias - h2h_venue_regression, 0.001, 0.999
                ))
                raw_without_h2h_venue = bool(
                    p_without_h2h_venue >= _winner_threshold(
                        mp, p_without_h2h_venue, ht, at, season_int
                    )
                )
                h2h_venue_regression_applied = raw_pick_home != raw_without_h2h_venue
            pitcher_applied = bool(pbias) and (raw_pick_home != raw_no_bias)
            cold_streak_applied = bool(cold_bias) and (raw_pick_home != raw_no_bias)
            hot_streak_applied = bool(hot_bias) and (raw_pick_home != raw_no_bias)
            blowout_applied = bool(blowout_bias) and (raw_pick_home != raw_no_bias)
            weather_extreme_applied = bool(weather_extreme_bias) and (raw_pick_home != raw_no_bias)
            home_calibration_applied = bool(home_calibration_bias) and (raw_pick_home != raw_no_bias)
            home_band_calib_applied = bool(home_band_calib_bias) and (raw_pick_home != raw_no_bias)
            interleague_applied = bool(interleague_bias) and (raw_pick_home != raw_no_bias)
            umpire_market_applied = bool(umpire_market_bias) and (raw_pick_home != raw_no_bias)
            circadian_applied = bool(circadian_bias) and (raw_pick_home != raw_no_bias)
            travel_resilience_applied = bool(travel_resilience_bias) and (raw_pick_home != raw_no_bias)
            pythag_luck_applied = bool(pythag_luck_bias) and (raw_pick_home != raw_no_bias)
            babip_persistence_applied = bool(babip_persistence_bias) and (raw_pick_home != raw_no_bias)
            lob_persistence_applied = bool(lob_persistence_bias) and (raw_pick_home != raw_no_bias)
            cws_night_applied = bool(cws_night_bias) and (raw_pick_home != raw_no_bias)
            comeback_deficit_applied = bool(comeback_deficit_bias) and (raw_pick_home != raw_no_bias)
            runs_median_applied = bool(runs_median_bias) and (raw_pick_home != raw_no_bias)
            burn_resilience_applied = bool(burn_resilience_bias) and (raw_pick_home != raw_no_bias)
            starter_workload_applied = bool(starter_workload_bias) and (raw_pick_home != raw_no_bias)
            highlev_fatigue_applied = bool(highlev_fatigue_bias) and (raw_pick_home != raw_no_bias)
            catcher_battery_applied = bool(catcher_battery_bias) and (raw_pick_home != raw_no_bias)
            flip_reason = _flip_reason(
                raw_pick_home, ht, at, gd, r.get("day_night"), p_home, mp
            )
            pick_is_home = (not raw_pick_home) if flip_reason else raw_pick_home
            # Series double-down trap: if the model repeats the exact pick it
            # just missed in this series AND fights the market, defer to market.
            series_flip = _series_double_down_market(c, r, pick_is_home)
            if series_flip:
                pick_is_home = bool(float(mp) > 0.5) if pd.notna(mp) else pick_is_home
            pick_abbrev = ht if pick_is_home else at
            night_elite = _is_night_elite(pick_is_home, ht, at, r.get("day_night"))
        else:
            night_elite = False
            series_flip = False
        # Value pick uses the MODEL-ONLY probability so edges aren't shrunk by
        # the display blend. Falls back to display prob if value calc failed.
        p_for_value = p_home_value if np.isfinite(p_home_value) else p_home
        v = _value_block(c, int(r["game_pk"]), p_for_value,
                          r["away_team_abbrev"], r["home_team_abbrev"],
                          slot=int(r["slot"]) if pd.notna(r.get("slot")) else None,
                          context=r)
        f5 = _f5_predict(c, r)
        fpu = r.get("first_pitch_utc")
        first_pitch_iso = (
            pd.Timestamp(fpu).isoformat() if pd.notna(fpu) else None
        )
        out.append({
            "game_pk": int(r["game_pk"]),
            "date": r["game_date"].isoformat(),
            "first_pitch_utc": first_pitch_iso,
            "slot": int(r["slot"]),
            "games_in_day": int(r["games_in_day"]),
            "away_abbrev": r["away_team_abbrev"],
            "home_abbrev": r["home_team_abbrev"],
            "venue": _safe(r.get("venue_name")),
            "p_home": _safe(p_home),
            "p_away": _safe(1 - p_home) if np.isfinite(p_home) else None,
            # Model-only probability (market-free) — drives the value/edge calc
            # and the "Modelo X%" comparison shown under the value pick.
            "p_home_value": _safe(p_home_value),
            "p_away_value": _safe(1 - p_home_value) if np.isfinite(p_home_value) else None,
            # Confidence tier — how much to trust the who-wins pick (coverage vs
            # accuracy). LOCK ~71% hit rate, FUERTE ~68%, MODERADO ~60%.
            **_confidence_tier(p_home),
            # Live ESPN win probability (overrides the pregame model number in
            # the UI for games that are in progress or final).
            "live_p_home": (_live_p_home(int(r["game_pk"]), int(r.get("season")) if pd.notna(r.get("season")) else None)
                            if (is_live or is_final) else None),
            "pred_total": _safe(total),
            "market_p_home": _safe(r.get("market_p_home")),
            "market_total": _safe(r.get("market_over_under")),
            "fair_home_ml": _to_american(p_home) if np.isfinite(p_home) else None,
            "fair_away_ml": _to_american(1 - p_home) if np.isfinite(p_home) else None,
            "pick_abbrev": pick_abbrev,
            # Trap inversion: raw pick was flipped because it landed on a trap
            # team ('team'), or on a day-of-week trap like fade WSH-on-Saturday
            # ('day'). Both flips validated to hit <50% in 3+ seasons.
            "team_flip":      flip_reason == "team",
            "day_flip":       flip_reason == "day",
            "night_flip":     flip_reason == "night",
            "pickday_flip":   flip_reason == "pickday",
            "pickmonth_flip": flip_reason == "pickmonth",
            "flip_reason": flip_reason,
            # Series double-down trap: model repeated the pick it just missed
            # in this series against the market → pick deferred to market side.
            "series_flip": bool(series_flip),
            # Night-elite badge: pick lands on a team with >0.62 acc in ALL 4
            # seasons of night games (currently PHI). Informational — the pick
            # itself is unchanged, only the confidence signal to the user.
            "night_elite": bool(night_elite),
            # H2H pair bias: matchup con miscalibracion historica del modelo
            # (LAD/COL, PIT/CIN, CIN/MIL). Se aplico un sesgo al p_home ANTES
            # del umbral; el badge se prende si el sesgo cambio el pick raw.
            "h2h_bias": bool(h2h_applied),
            # H2H venue regression: the local dominated this exact visitor in
            # prior meetings, but the 50-60% model band historically regresses.
            "h2h_venue_regression_bias": bool(h2h_venue_regression_applied),
            # Pitcher bias: abridor con miscalibracion historica (ej. Wheeler,
            # Sale, deGrom, Sánchez, Yamamoto). Badge se prende si el sesgo
            # cambio el pick raw.
            "pitcher_bias": bool(pitcher_applied),
            # Home cold streak bias: equipo local con 3+ derrotas consecutivas
            # entrando al juego. En 2026 gana +11pp mas de lo predicho (mean
            # reversion). Badge se prende si el sesgo cambio el pick raw.
            "cold_streak_bias": bool(cold_streak_applied),
            # Away hot streak bias: visitante con 3+ victorias consecutivas.
            # En 2026, home gana +8pp mas de lo predicho (mean reversion).
            # Badge se prende si el sesgo cambio el pick raw.
            "hot_streak_bias": bool(hot_streak_applied),
            # Home blowout momentum: local viene de ganar por 8+ carreras el
            # juego previo. Historicamente sigue ganando mas de lo predicho.
            # Badge se prende si el sesgo cambio el pick raw.
            "blowout_bias": bool(blowout_applied),
            # Codex overrides validados 2026-07-10: 4 winners + 2 marginal-
            # positive (weather_extreme/circadian aportan +0.1pp global en
            # pipeline completo aunque son 0 en aislamiento).
            "weather_extreme_bias": bool(weather_extreme_applied),
            "home_calibration_bias": bool(home_calibration_applied),
            "home_band_calib_bias": bool(home_band_calib_applied),
            "interleague_bias": bool(interleague_applied),
            "umpire_market_bias": bool(umpire_market_applied),
            "circadian_bias": bool(circadian_applied),
            "travel_resilience_bias": bool(travel_resilience_applied),
            "pythag_luck_bias": bool(pythag_luck_applied),
            "babip_persistence_bias": bool(babip_persistence_applied),
            "lob_persistence_bias": bool(lob_persistence_applied),
            "cws_night_bias": bool(cws_night_applied),
            "comeback_deficit_bias": bool(comeback_deficit_applied),
            "runs_median_bias": bool(runs_median_applied),
            "burn_resilience_bias": bool(burn_resilience_applied),
            "starter_workload_bias": bool(starter_workload_applied),
            "highlev_fatigue_bias": bool(highlev_fatigue_applied),
            "catcher_battery_bias": bool(catcher_battery_applied),
            "phi_road_slate_flip": False,
            "wsh_strong_slate_flip": False,
            "pitcher_away": pitcher_away,
            "pitcher_home": pitcher_home,
            **v,
            **f5,
            **_market_extras(c, int(r["game_pk"])),
            **_pinnacle_block(c, int(r["game_pk"])),
            **_proj_ks(c, r, feed=cached_feed),
            **_insights(r),
            **_game_context(c, r, games_df=c["games"]),
            "is_played": is_final,
            "is_live": is_live,
            "game_state": gs,
            "live_score": live,
            "home_score": home_score,
            "away_score": away_score,
        })
    # PHI road correction on uncertain slates. The threshold was learned on
    # 2025 and validated unchanged on both halves of 2026. This is a final-pick
    # correction because it must run after traps and series double-down.
    by_date: dict[str, list[dict]] = {}
    for game in out:
        by_date.setdefault(str(game["date"])[:10], []).append(game)
    for game_date, slate in by_date.items():
        confidences = [abs(float(g["p_home"]) - 0.5) for g in slate if g.get("p_home") is not None]
        day_share = float(np.mean([
            str(sub.loc[sub["game_pk"].eq(g["game_pk"]), "day_night"].iloc[0]).lower() == "day"
            for g in slate
        ])) if slate else 0.0
        if int(game_date[:4]) < 2025 or not confidences or float(np.mean(confidences)) > 0.055:
            phi_slate = False
        else:
            phi_slate = True
        for game in slate:
            if phi_slate and game.get("away_abbrev") == "PHI" and game.get("pick_abbrev") == game.get("home_abbrev"):
                game["pick_abbrev"] = "PHI"
                game["phi_road_slate_flip"] = True
            teams = {game.get("away_abbrev"), game.get("home_abbrev")}
            if (
                int(game_date[:4]) >= 2025 and "WSH" in teams
                and game.get("pick_abbrev") != "WSH" and game.get("p_home") is not None
                and abs(float(game["p_home"]) - 0.5) > 0.08 and day_share >= 0.25
            ):
                game["pick_abbrev"] = "WSH"
                game["wsh_strong_slate_flip"] = True
    if team_filter:
        out = [g for g in out if team_filter in {g.get("away_abbrev"), g.get("home_abbrev")}]
    _games_response_cache[cache_key] = (time.time(), out)
    return out


@app.get("/api/games/{game_pk}")
def game_detail(game_pk: int):
    c = _load()
    train = c["train"]
    games = c["games"]

    row = train[train["game_pk"] == game_pk]
    if row.empty:
        raise HTTPException(404, "game not found")
    r = row.iloc[0]
    g = games[games["game_pk"] == game_pk].iloc[0] if not games[games["game_pk"] == game_pk].empty else None
    detail_context = r.copy()
    if g is not None:
        detail_context["day_night"] = g.get("day_night")

    # Compute slot-within-day for this game so the value block can apply the
    # slot-aware EV adjustment.
    detail_slot = None
    if g is not None and pd.notna(g.get("first_pitch_utc")):
        day = pd.to_datetime(g["first_pitch_utc"], utc=True).tz_convert("America/New_York").date()
        day_games = games.copy()
        day_games["first_pitch_utc"] = pd.to_datetime(day_games["first_pitch_utc"], utc=True, errors="coerce")
        day_games = day_games.dropna(subset=["first_pitch_utc"])
        day_games["et_date"] = day_games["first_pitch_utc"].dt.tz_convert("America/New_York").dt.date
        day_games = day_games[day_games["et_date"] == day].sort_values("first_pitch_utc")
        order = day_games["game_pk"].tolist()
        if game_pk in order:
            detail_slot = order.index(game_pk) + 1

    # Read feed/live for player names, weather, and live linescore.
    feed = _read_cached_feed(game_pk)
    state = _game_state(feed)
    if state["is_live"] or not state["is_final"]:
        fresh_feed = _fetch_live_feed(game_pk)
        if fresh_feed:
            feed = fresh_feed

    away_pid = _safe(r.get("probable_away_pitcher_id"))
    home_pid = _safe(r.get("probable_home_pitcher_id"))

    def _player(pid):
        if not pid or feed is None:
            return {"id": pid, "name": None, "throws": None}
        key = f"ID{int(pid)}"
        players = (feed.get("gameData") or {}).get("players") or {}
        p = players.get(key) or {}
        return {
            "id": int(pid),
            "name": p.get("fullName"),
            "throws": (p.get("pitchHand") or {}).get("code"),
            "primary_number": p.get("primaryNumber"),
            "headshot": f"https://midfield.mlbstatic.com/v1/people/{int(pid)}/spots/120" if pid else None,
        }

    away_p = _player(away_pid)
    home_p = _player(home_pid)

    # predictions
    feats = c["cls"]["feature_names"]
    try:
        X = pd.DataFrame([r[feats].values], columns=feats)
        p_home = float(_predict_phome(c, X, pd.Series([r.get("market_p_home", float("nan"))]), context_df=pd.DataFrame([r]))[0])
        total = float(c["reg"]["model"].predict(X)[0])
    except Exception:
        p_home, total = float("nan"), float("nan")

    def _team(side: str):
        abbr = r[f"{side}_team_abbrev"]
        tid = _safe(r.get(f"{side}_team_id"))
        # records from games.parquet
        wins = _safe(g[f"{side}_record_wins"]) if g is not None else None
        losses = _safe(g[f"{side}_record_losses"]) if g is not None else None
        suffix = "_h" if side == "home" else "_a"
        return {
            "abbrev": abbr,
            "id": tid,
            "name": _safe(g[f"{side}_team_name"]) if g is not None else None,
            "wins": wins,
            "losses": losses,
            "logo": f"https://www.mlbstatic.com/team-logos/{tid}.svg" if tid else None,
            "runs_scored_l10": _safe(r.get(f"runs_scored_l10{suffix}")),
            "runs_allowed_l10": _safe(r.get(f"runs_allowed_l10{suffix}")),
            "win_pct_l30": _safe(r.get(f"win_pct_l30{suffix}")),
            "off_xwoba_l30": _safe(r.get(f"off_xwoba_l30{suffix}")),
            "def_xwoba_l30": _safe(r.get(f"def_xwoba_l30{suffix}")),
            "off_barrel_rate_l30": _safe(r.get(f"off_barrel_rate_l30{suffix}")),
            "off_k_pct_l30": _safe(r.get(f"off_k_pct_l30{suffix}")),
            "off_bb_pct_l30": _safe(r.get(f"off_bb_pct_l30{suffix}")),
        }

    def _pitcher_features(side: str, base: dict) -> dict:
        suffix = "_h" if side == "home" else "_a"
        base.update({
            "starter_xwoba_l15": _safe(r.get(f"starter_xwoba_l15{suffix}")),
            "starter_k_pct_l15": _safe(r.get(f"starter_k_pct_l15{suffix}")),
            "starter_bb_pct_l15": _safe(r.get(f"starter_bb_pct_l15{suffix}")),
            "starter_velo_l15": _safe(r.get(f"starter_velo_mean_l15{suffix}")),
            "starter_spin_l15": _safe(r.get(f"starter_spin_mean_l15{suffix}")),
            "starter_barrel_against_l15": _safe(r.get(f"starter_barrel_against_l15{suffix}")),
            "days_rest": _safe(r.get(f"starter_days_rest{suffix}")),
        })
        # Real season aggregates from features_pitcher_season.parquet
        ps = c.get("pitcher_season")
        pid = base.get("id")
        season = _safe(r.get("season"))
        if ps is not None and pid is not None and season is not None:
            try:
                row = ps.loc[(int(pid), int(season))]
                base["season"] = {
                    "wins": int(row["wins"]),
                    "losses": int(row["losses"]),
                    "saves": int(row["saves"]),
                    "ip": _safe(row["ip"]),
                    "era": _safe(row["era"]),
                    "whip": _safe(row["whip"]),
                    "k9": _safe(row["k9"]),
                    "bb9": _safe(row["bb9"]),
                    "hr9": _safe(row["hr9"]),
                    "k_pct": _safe(row["k_pct"]),
                    "bb_pct": _safe(row["bb_pct"]),
                    "xwoba_vs_lhb": _safe(row.get("xwoba_vs_lhb")),
                    "xwoba_vs_rhb": _safe(row.get("xwoba_vs_rhb")),
                }
            except KeyError:
                base["season"] = None
        else:
            base["season"] = None
        return base

    away_p = _pitcher_features("away", away_p)
    home_p = _pitcher_features("home", home_p)

    weather = (feed.get("gameData", {}).get("weather") or {}) if feed else {}
    live_score = _live_linescore(feed)
    feed_final = bool((live_score.get("state") or {}).get("is_final"))

    return {
        "game_pk": int(r["game_pk"]),
        "date": r["game_date"].isoformat(),
        "venue": _safe(r.get("venue_name")),
        "weather": {
            "temp_f": _safe(r.get("weather_temp_f")),
            "condition": weather.get("condition"),
            "wind": weather.get("wind"),
        },
        "away_team": _team("away"),
        "home_team": _team("home"),
        "away_pitcher": away_p,
        "home_pitcher": home_p,
        "market": {
            "p_home": _safe(r.get("market_p_home")),
            "total": _safe(r.get("market_over_under")),
            "spread": _safe(r.get("market_spread")),
            "n_providers": _safe(r.get("market_n_providers")),
        },
        "model": {
            "p_home": _safe(p_home),
            "p_away": _safe(1 - p_home) if np.isfinite(p_home) else None,
            # Live ESPN win probability (None for upcoming games).
            "live_p_home": _live_p_home(game_pk,
                                         int(r.get("season")) if pd.notna(r.get("season")) else None),
            "pred_total_runs": _safe(total),
            "fair_home_ml": _to_american(p_home) if np.isfinite(p_home) else None,
            "fair_away_ml": _to_american(1 - p_home) if np.isfinite(p_home) else None,
        },
        "value": _value_block(c, game_pk, p_home,
                                r["away_team_abbrev"], r["home_team_abbrev"],
                                slot=detail_slot,
                                context=detail_context),
        "slot": detail_slot,
        "f5": _f5_predict(c, r),
        "result": {
            "played": feed_final or pd.notna(r.get("total_runs")),
            "home_score": live_score["home"]["runs"] if feed_final else _safe(r.get("home_score")),
            "away_score": live_score["away"]["runs"] if feed_final else _safe(r.get("away_score")),
        },
        "live_score": live_score,
        "starter_actual": {
            "away": _starter_pitching_line(feed, "away", away_pid),
            "home": _starter_pitching_line(feed, "home", home_pid),
        },
        "park_runs_factor": _safe(r.get("park_runs_factor")),
        "context": _game_context(c, r, games_df=c["games"]),
    }


# ---------- static frontend ----------
@app.get("/api/games/{game_pk}/win_probability")
def game_win_probability(game_pk: int):
    """Return the per-play home-win-probability time series for one game.

    Reads the raw ESPN probabilities JSON (~74 items per game) and enriches
    each play with inning + score from StatsAPI plays.parquet (joined by
    order, since ESPN and StatsAPI sequence chronologically the same way).
    """
    c = _load()
    games = c["games"]
    xref_path = PROCESSED / "games_xref.parquet"
    if not xref_path.exists():
        return {"plays": [], "current_p_home": None}
    xref = pd.read_parquet(xref_path)
    match = xref[(xref["_merge"] == "both") & (xref["game_pk"] == game_pk)]
    if match.empty:
        return {"plays": [], "current_p_home": None}
    espn_event_id = str(match.iloc[0]["espn_event_id"])
    season = None
    g_row = games[games["game_pk"] == game_pk]
    if not g_row.empty:
        season = int(g_row.iloc[0]["season"])
    if season is None:
        return {"plays": [], "current_p_home": None}

    prob_file = RAW / "probabilities" / str(season) / f"{espn_event_id}.json"
    if not prob_file.exists():
        return {"plays": [], "current_p_home": None}
    try:
        payload = orjson.loads(prob_file.read_bytes())
    except Exception:
        return {"plays": [], "current_p_home": None}
    items = payload.get("items") or []
    # Sort by sequenceNumber ASC; items can come out of order in the raw dump.
    items = sorted(items, key=lambda it: int(it.get("sequenceNumber") or 0))

    # Optional enrichment: align with StatsAPI plays.parquet by order so we can
    # tag inning/score on each ESPN play (descriptions can come later).
    plays_path = PROCESSED / "plays.parquet"
    inning_seq: list[dict] = []
    if plays_path.exists():
        try:
            pl = pd.read_parquet(plays_path)
            pl = pl[pl["game_pk"] == game_pk].sort_values("at_bat_index")
            for _, p in pl.iterrows():
                inning_seq.append({
                    "inning": int(p["inning"]) if pd.notna(p["inning"]) else None,
                    "half": p["half_inning"],
                    "away_score": int(p["away_score"]) if pd.notna(p["away_score"]) else None,
                    "home_score": int(p["home_score"]) if pd.notna(p["home_score"]) else None,
                    "event": p.get("event"),
                    "is_scoring": bool(p.get("is_scoring_play")),
                })
        except Exception:
            pass

    plays_out: list[dict] = []
    for i, it in enumerate(items):
        wp = it.get("homeWinPercentage")
        if wp is None:
            continue
        enrich = inning_seq[i] if i < len(inning_seq) else {}
        plays_out.append({
            "idx": i,
            "p_home": round(float(wp), 4),
            "p_away": round(1 - float(wp) - float(it.get("tiePercentage") or 0), 4),
            "inning": enrich.get("inning"),
            "half": enrich.get("half"),
            "away_score": enrich.get("away_score"),
            "home_score": enrich.get("home_score"),
            "event": enrich.get("event"),
            "is_scoring": enrich.get("is_scoring", False),
        })

    current = plays_out[-1]["p_home"] if plays_out else None
    return {
        "plays": plays_out,
        "n_plays": len(plays_out),
        "current_p_home": current,
    }


@app.get("/api/comebacks")
def comebacks(days: int = 90, season: int = 0):
    """Comeback analysis: team comeback rates, daily records, weekly patterns.

    Query params:
      days=90    last N days (ignored if season is set)
      season=0   filter to a specific year (0 = use days)
    """
    cb_path = PROCESSED / "comeback_analysis.parquet"
    if not cb_path.exists():
        raise HTTPException(status_code=503, detail="Run: python -m src.analysis.comeback_analysis first")

    cb = pd.read_parquet(cb_path)
    cb["game_date"] = pd.to_datetime(cb["game_date"])

    if season:
        cb = cb[cb["game_date"].dt.year == int(season)].copy()
    else:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(days))
        cb = cb[cb["game_date"] >= cutoff].copy()

    cb = cb[cb["was_comeback"].notna() & cb["away_score"].notna() & cb["home_score"].notna()]

    # ── Per-team stats ────────────────────────────────────────────────────────
    team_rows = []
    all_abbrevs = (
        set(cb["home_team_abbrev"].dropna().unique()) |
        set(cb["away_team_abbrev"].dropna().unique())
    )
    for abbrev in all_abbrevs:
        home = cb[cb["home_team_abbrev"] == abbrev]
        away = cb[cb["away_team_abbrev"] == abbrev]
        total_games = len(home) + len(away)
        if total_games < 5:
            continue
        h_wins = home[home["home_win"] == 1]
        a_wins = away[away["home_win"] == 0]
        h_cb = h_wins[h_wins["was_comeback"] == True]
        a_cb = a_wins[a_wins["was_comeback"] == True]
        all_cb = pd.concat([h_cb, a_cb])
        total_wins = len(h_wins) + len(a_wins)
        total_cb = len(all_cb)

        wp_vals = all_cb["min_wp_winner"].dropna()
        wp_vals = wp_vals[wp_vals > 0.02]
        avg_min_wp = float(wp_vals.mean()) if len(wp_vals) > 0 else None

        best_momio = None
        if len(all_cb) > 0:
            best_wp_row = all_cb.loc[all_cb["min_wp_winner"].idxmin()]
            bwp = float(best_wp_row["min_wp_winner"]) if pd.notna(best_wp_row["min_wp_winner"]) else None
            if bwp and bwp > 0.02:
                best_momio = round(1.0 / bwp, 2)

        team_rows.append({
            "abbrev": abbrev,
            "games": total_games,
            "wins": total_wins,
            "comebacks": total_cb,
            "cb_rate": round(total_cb / max(total_wins, 1), 4),
            "avg_deficit": round(float(all_cb["max_deficit_winner"].mean()), 2) if total_cb > 0 else 0.0,
            "max_deficit": float(all_cb["max_deficit_winner"].max()) if total_cb > 0 else 0.0,
            "avg_min_wp": round(avg_min_wp, 4) if avg_min_wp is not None else None,
            "best_momio": best_momio,
        })
    team_rows.sort(key=lambda r: r["comebacks"], reverse=True)

    # ── Daily records ─────────────────────────────────────────────────────────
    cb["date_str"] = cb["game_date"].dt.strftime("%Y-%m-%d")
    daily_rows = []
    for date_str, day_df in cb.groupby("date_str"):
        cbs_d = int(day_df["was_comeback"].sum())
        games_d = len(day_df)
        highlights = []
        for _, r in day_df[day_df["was_comeback"] == True].nlargest(5, "max_deficit_winner").iterrows():
            hw = int(r["home_win"])
            winner = r["home_team_abbrev"] if hw else r["away_team_abbrev"]
            loser  = r["away_team_abbrev"] if hw else r["home_team_abbrev"]
            wp = float(r["min_wp_winner"]) if pd.notna(r.get("min_wp_winner")) else None
            highlights.append({
                "winner": winner, "loser": loser,
                "deficit": float(r["max_deficit_winner"]),
                "score": f"{int(r['away_score'])}-{int(r['home_score'])}",
                "min_wp": round(wp, 3) if wp and wp > 0.02 else None,
                "decimal_at_worst": round(1.0 / wp, 2) if wp and wp > 0.02 else None,
            })
        daily_rows.append({
            "date": date_str,
            "games": games_d,
            "comebacks": cbs_d,
            "cb_pct": round(cbs_d / games_d * 100, 1) if games_d else 0.0,
            "highlights": highlights,
        })
    daily_rows.sort(key=lambda r: r["date"], reverse=True)

    # ── Weekly patterns ───────────────────────────────────────────────────────
    cb["week"] = cb["game_date"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    weekly_rows = []
    for week_str, wk in cb.groupby("week"):
        games_w = len(wk)
        cbs_w = int(wk["was_comeback"].sum())
        cb_wk = wk[wk["was_comeback"] == True]
        weekly_rows.append({
            "week_start": week_str,
            "games": games_w,
            "comebacks": cbs_w,
            "cb_pct": round(cbs_w / games_w * 100, 1) if games_w else 0.0,
            "avg_deficit": round(float(cb_wk["max_deficit_winner"].mean()), 2) if cbs_w > 0 else 0.0,
            "zero_comeback_day": cbs_w == 0,
        })
    weekly_rows.sort(key=lambda r: r["week_start"], reverse=True)

    return {
        "days": days if not season else None,
        "season": season if season else None,
        "total_games": len(cb),
        "total_comebacks": int(cb["was_comeback"].sum()),
        "overall_cb_pct": round(float(cb["was_comeback"].mean()) * 100, 1),
        "team_stats": team_rows,
        "daily": daily_rows,
        "weekly": weekly_rows,
    }


@app.get("/api/errors")
def error_analysis(season: int = 0, days: int = 0):
    """Per-team error vulnerability: rate, WR with/without errors, resilience."""
    games = _load()["games"]
    games = games.dropna(subset=["home_errors", "away_errors",
                                  "home_score", "away_score",
                                  "home_team_abbrev", "away_team_abbrev"])
    games = games[games["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    if season:
        games = games[games["game_date"].dt.year == int(season)]
    elif days:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(days))
        games = games[games["game_date"] >= cutoff]
    else:
        games = games[games["game_date"].dt.year == pd.Timestamp.now().year]

    rows = []
    for _, r in games.iterrows():
        hw = int(r["home_win"])
        for side, team in [("home", r["home_team_abbrev"]),
                            ("away", r["away_team_abbrev"])]:
            errs = float(r["home_errors"] if side == "home" else r["away_errors"])
            won  = int(hw == 1 if side == "home" else hw == 0)
            rows.append({"team": team, "errors": errs, "won": won})

    tdf = pd.DataFrame(rows)

    team_rows = []
    for team, grp in tdf.groupby("team"):
        if len(grp) < 20:
            continue
        err_g  = grp[grp["errors"] >= 1]
        no_err = grp[grp["errors"] == 0]
        multi  = grp[grp["errors"] >= 2]

        wr_err    = float(err_g["won"].mean())   if len(err_g)  >= 5 else None
        wr_no_err = float(no_err["won"].mean())  if len(no_err) >= 5 else None
        wr_multi  = float(multi["won"].mean())   if len(multi)  >= 5 else None
        resilience = round(wr_err - wr_no_err, 3) if (wr_err is not None and wr_no_err is not None) else None

        team_rows.append({
            "team": team,
            "games": len(grp),
            "avg_errors": round(float(grp["errors"].mean()), 3),
            "error_game_pct": round(float((grp["errors"] >= 1).mean()), 3),
            "wr_overall": round(float(grp["won"].mean()), 3),
            "wr_no_error": round(wr_no_err, 3) if wr_no_err is not None else None,
            "wr_with_error": round(wr_err, 3) if wr_err is not None else None,
            "wr_multi_error": round(wr_multi, 3) if wr_multi is not None else None,
            "resilience": resilience,
        })

    team_rows.sort(key=lambda r: r["resilience"] if r["resilience"] is not None else -99, reverse=True)
    return {"season": season or pd.Timestamp.now().year, "teams": team_rows}


# ─── Apuestas del día — selector calibrado walk-forward ──────────────────────
# Estrategia S2 (validada 2026-07-18 sobre walkforward_preds.parquet):
# rankear los picks del día por la PROBABILIDAD DE MERCADO DEL LADO ELEGIDO
# (que tan favorito de mercado es nuestro pick), con desempate por confianza
# del pipeline. NO por probabilidad cruda del modelo.
#   Simulación top-N diario (N=5/3/2 según slate), picks threshold-only:
#     - Selección S2: 62.2% (2025) / 57.0% (2026 holdout), positiva en ambas
#       mitades (H1 54.8%, H2 60.3%) vs baseline todos-los-juegos 58.1%/54.9%.
#     - La tabla empírica aprendida en 2025 (S3) fue mejor en desarrollo
#       (63.1%) pero PEOR en holdout (55.2%) — descartada por overfit.
#   El pipeline completo de producción rinde ~5pp más que threshold-only, así
#   que el hit esperado de la selección diaria ronda 62-65%.
# Regla de tamaño: 5 picks si el slate tiene >=10 juegos; 3 si 6-9; 2 si <=5.
# Hit rates históricos por banda de mercado del lado elegido (dev 2025, para
# mostrar como referencia en la UI — no para seleccionar):
_DAILY_PICK_BANDS = [
    (0.70, 1.01, 77.9),
    (0.62, 0.70, 65.2),
    (0.55, 0.62, 56.7),
    (0.45, 0.55, 54.5),
    (0.00, 0.45, 53.9),
]


def _daily_picks_n(slate_size: int) -> int:
    if slate_size >= 10:
        return 5
    if slate_size >= 6:
        return 3
    return 2


@app.get("/api/daily-picks")
async def daily_picks(date: str | None = None):
    """Picks recomendados del día — selección especial optimizada para acertar.

    No elige por probabilidad cruda del modelo: rankea por el perfil donde el
    pipeline históricamente acierta (fuerza de mercado del lado elegido, con
    la confianza del pipeline como desempate). Ver comentario del selector.
    """
    from datetime import date as _date
    target = date or _date.today().isoformat()
    cards = await games(start=target, end=target)
    slate = [g for g in cards if g.get("pick_abbrev") and g.get("p_home") is not None]
    n_slate = len(slate)
    if n_slate == 0:
        return {"date": target, "slate_size": 0, "n_picks": 0, "picks": [],
                "selector": "market-strength", "note": "Sin juegos con pick para esta fecha."}

    scored = []
    for g in slate:
        p_home = float(g["p_home"])
        pick_home = g["pick_abbrev"] == g["home_abbrev"]
        mp = g.get("market_p_home")
        if mp is not None and np.isfinite(mp):
            mkt_pick_p = float(mp) if pick_home else 1.0 - float(mp)
            agree = (float(mp) >= 0.5) == pick_home
        else:
            mkt_pick_p = 0.5
            agree = True
        conf = abs(p_home - 0.5)
        hist_hit = next((h for lo, hi, h in _DAILY_PICK_BANDS if lo <= mkt_pick_p < hi), 55.0)
        scored.append({
            "game_pk": g["game_pk"],
            "first_pitch_utc": g.get("first_pitch_utc"),
            "venue": g.get("venue"),
            "away_abbrev": g["away_abbrev"],
            "home_abbrev": g["home_abbrev"],
            "pick_abbrev": g["pick_abbrev"],
            "pick_is_home": pick_home,
            "p_home": g.get("p_home"),
            "p_away": g.get("p_away"),
            "market_p_home": g.get("market_p_home"),
            "mkt_pick_p": round(mkt_pick_p, 4),
            "agree_market": bool(agree),
            "confidence_tier": g.get("confidence_tier"),
            "tier_hit_rate": g.get("tier_hit_rate"),
            "hist_hit_band": hist_hit,
            "pitcher_home": g.get("pitcher_home"),
            "pitcher_away": g.get("pitcher_away"),
            "is_played": g.get("is_played"),
            "is_live": g.get("is_live"),
            "home_score": g.get("home_score"),
            "away_score": g.get("away_score"),
            "_score": (mkt_pick_p, conf),
        })
    scored.sort(key=lambda x: x["_score"], reverse=True)
    n_picks = _daily_picks_n(n_slate)
    picks = scored[:n_picks]
    for i, p in enumerate(picks, 1):
        p["rank"] = i
        del p["_score"]
        # resultado si ya terminó
        if p.get("is_played") and p.get("home_score") is not None and p.get("away_score") is not None:
            home_won = float(p["home_score"]) > float(p["away_score"])
            p["pick_won"] = bool(p["pick_is_home"] == home_won)
        else:
            p["pick_won"] = None
    hits = sum(1 for p in picks if p.get("pick_won") is True)
    settled = sum(1 for p in picks if p.get("pick_won") is not None)
    return {
        "date": target,
        "slate_size": n_slate,
        "n_picks": n_picks,
        "selector": "market-strength",
        "selection_hist": {"acc_2025": 62.2, "acc_2026": 57.0,
                            "note": "walk-forward threshold-only; el pipeline completo rinde ~5pp más"},
        "settled": settled,
        "hits": hits,
        "picks": picks,
    }


@app.get("/api/burn")
def burn_report(date: str | None = None):
    """Per-team "carne al asador" report for a given date.

    Returns teams that had a heavy burn YESTERDAY (the day before the given
    date), so the user can see who comes in tired today. Also includes the
    cumulative season average for context.

    Params:
      date — ISO date (defaults to today)
    """
    from datetime import date as _date, timedelta
    burn_path = PROCESSED / "burn_analysis.parquet"
    if not burn_path.exists():
        return {"date": None, "burns": [], "season_top": []}
    bn = pd.read_parquet(burn_path)
    bn["game_date"] = pd.to_datetime(bn["game_date"])

    target = _date.fromisoformat(date) if date else _date.today()
    target_ts = pd.Timestamp(target)
    today_slice = bn[bn["game_date"].dt.date == target]
    # Sort by burn_score descending — heaviest burn first
    today_slice = today_slice.sort_values("burn_score", ascending=False)

    def _row(r):
        prev_date = r["prev_game_date"]
        return {
            "team": r["team"],
            "burn_score": float(r["burn_score"]),
            "prev_game_date": prev_date.date().isoformat() if pd.notna(prev_date) else None,
            "days_rest": int(r["days_rest"]) if pd.notna(r["days_rest"]) else None,
            "bullpen_ip_yest": round(float(r["bullpen_ip_yest"]), 1) if pd.notna(r["bullpen_ip_yest"]) else None,
            "bullpen_apps_yest": int(r["bullpen_apps_yest"]) if pd.notna(r["bullpen_apps_yest"]) else None,
            "extra_innings": bool(r["extra_innings_yest"]),
            "long_game": bool(r["long_game_yest"]),
            "used_closer": bool(r["used_closer_close_yest"]),
            "comeback": bool(r["comeback_won_yest"]),
            "won_today": int(r["won_today"]) if pd.notna(r["won_today"]) else None,
        }

    burns = [_row(r) for _, r in today_slice.iterrows()]

    # Season cumulative leaderboard (top 10 most-burned teams)
    season_year = target.year
    season = bn[bn["game_date"].dt.year == season_year]
    season_top = (
        season.groupby("team")
        .agg(
            avg_burn=("burn_score", "mean"),
            n_games=("burn_score", "size"),
            n_high_burn=("burn_score", lambda s: int((s >= 50).sum())),
        )
        .reset_index()
        .sort_values("avg_burn", ascending=False)
        .head(10)
    )
    season_top_list = [
        {"team": r["team"],
         "avg_burn": round(float(r["avg_burn"]), 2),
         "n_games": int(r["n_games"]),
         "n_high_burn": int(r["n_high_burn"])}
        for _, r in season_top.iterrows()
    ]

    return {
        "date": target.isoformat(),
        "burns": burns,
        "season_top": season_top_list,
    }


@app.post("/api/publish")
async def publish_snapshot():
    """Generate today's full predictions snapshot → vercel/snapshot.json.

    Run this locally (while the server is up) to publish today's predictions
    to the Vercel static web app.  After calling this, commit and push
    vercel/snapshot.json so Vercel redeploys with fresh data.
    """
    import datetime as _dt
    today = date.today()
    # Ventana de +/- 2 semanas para que Vercel pueda navegar entre fechas
    # sin backend. ~15 juegos/dia * 29 dias = ~435 juegos = ~1-2MB snapshot.
    win_start = today - timedelta(days=14)
    win_end   = today + timedelta(days=14)

    games_data  = await games(start=win_start.isoformat(), end=win_end.isoformat())
    perf_90     = performance(days=90)  # 3 meses para DailySummaryPage
    perf_2      = performance(days=2)
    ranking_60  = team_ranking(days=60)
    burn_today  = burn_report(date=today.isoformat())

    snapshot = {
        "published_at": _dt.datetime.utcnow().isoformat() + "Z",
        "date": today.isoformat(),
        "games": games_data,
        "performance": perf_90,
        "performance_2d": perf_2,
        "team_ranking": ranking_60,
        "burn": burn_today,
    }

    vercel_dir = Path(__file__).parent.parent.parent / "vercel"
    out = vercel_dir / "snapshot.json"
    out.write_bytes(orjson.dumps(snapshot, option=orjson.OPT_NON_STR_KEYS))

    return {
        "ok": True,
        "published_at": snapshot["published_at"],
        "date": today.isoformat(),
        "games": len(games_data),
        "path": str(out),
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/strike")
def strike():
    return FileResponse(STATIC_DIR / "strike.html")


@app.get("/predict")
def predict_page():
    return FileResponse(STATIC_DIR / "predict.html")
