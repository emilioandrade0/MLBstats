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
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles

from ..normalize.paths import PROCESSED, RAW

MODELS = PROCESSED.parent / "models"
STATIC_DIR = Path(__file__).parent / "static"
STATSAPI_LIVE_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

# Background refresh interval. Override with env var if needed.
REFRESH_INTERVAL_SEC = int(os.environ.get("STRIKECAST_REFRESH_SEC", "900"))  # 15 min
# Minimum gap between on-request refreshes (lower than background interval —
# we want pageloads to feel snappy without hammering ESPN).
REFRESH_ON_REQUEST_TTL_SEC = int(os.environ.get("STRIKECAST_ONREQ_TTL_SEC", "120"))  # 2 min
_last_refresh = {"ok": 0.0, "started": 0.0, "running": False, "err": None}
_refresh_lock = asyncio.Lock()
_freshness_cache = {"ts": 0.0, "data": None}
_FRESHNESS_TTL_SEC = 60


async def _maybe_refresh_now() -> None:
    """Fast on-request refresh: only re-fetches if odds are stale.

    Throttled to one fetch every REFRESH_ON_REQUEST_TTL_SEC. If a refresh is
    already running, the request waits up to 4s for it to finish so the
    response reflects fresh data; after that it returns the (still stale)
    cached values rather than blocking the user.
    """
    now = time.time()
    if _last_refresh["ok"] and (now - _last_refresh["ok"]) < REFRESH_ON_REQUEST_TTL_SEC:
        return
    if _refresh_lock.locked():
        try:
            await asyncio.wait_for(_refresh_lock.acquire(), timeout=4.0)
            _refresh_lock.release()
        except asyncio.TimeoutError:
            pass
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


def _load():
    if _cache:
        return _cache
    train = pd.read_parquet(PROCESSED / "train.parquet")
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

    _cache.update({"train": train, "games": games, "cls": cls_b, "reg": reg_b,
                   "pitcher_season": pitcher_season, "market_lines": market_lines,
                   "f5_cls": f5_cls_b, "f5_reg": f5_reg_b})
    return _cache


def _build_market_lines():
    """Build the game_pk → (home_ml, away_ml, total_close) median table.

    Extracted so refresh-on-request can rebuild without restarting the server.
    Returns None if the underlying parquets are missing.
    """
    odds_close_path = PROCESSED / "odds_close.parquet"
    xref_path = PROCESSED / "games_xref.parquet"
    if not (odds_close_path.exists() and xref_path.exists()):
        return None
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
    ).reset_index()
    xref = pd.read_parquet(xref_path)
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    return agg.merge(matched, on="espn_event_id", how="inner") \
              .drop(columns=["espn_event_id"]).drop_duplicates("game_pk") \
              .set_index("game_pk")


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
DOG_HIST_WIN_PROB = 0.506


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
                  slot: int | None = None) -> dict:
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
        # slot_sharp | slot_flip | low_conf_market | dog_46_48
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
        out["ev_grade"]          = "marginal"
        out["ev_grade_reason"]   = "dog_46_48"
        out["alt_pick"]          = None
        return True

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
        # No side has positive edge — explicitly NO bet
        out["ev_grade"] = "no_edge"
        apply_dog_46_48_override()
        return out

    out["value_side"] = side
    out["value_decimal"] = round(dec, 2)
    out["value_edge_pp"] = round(edge_book * 100, 2)
    out["value_team_abbrev"] = team
    out["ev_per_dollar"] = round(prob * (dec - 1) - (1 - prob), 4)

    # Empirical grading (from value_filter.py walk-forward analysis)
    edge_pp = edge_book * 100
    if edge_pp >= 2.0:
        out["ev_grade"] = "overconfident"  # model unreliable → SKIP
    elif edge_pp >= 1.0:
        out["ev_grade"] = "sweet"          # +11% ROI band → BET
    elif edge_pp >= 0.0:
        out["ev_grade"] = "marginal"       # +1% ROI band → bet smaller
    else:
        out["ev_grade"] = "no_edge"        # negative EV

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
        out["ev_grade"]          = "marginal"
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
        out["ev_grade"]          = "marginal"
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
            out["ev_grade"]          = "marginal"
            out["ev_grade_reason"]   = "low_conf_market"
            # Suppress flip alt — it conflicts with the market override.
            out["alt_pick"]          = None
    return out


SHRINK_FACTOR = 0.75  # after C+D, the previous 0.5 was over-correcting; 0.75 minimizes ll
SHRINK_ANCHOR = 0.55
W_AGREE    = 0.50  # market weight when model & market agree on direction
W_DISAGREE = 0.80  # market weight when they disagree (trust market more)


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


def _predict_phome(c, X: pd.DataFrame, market_p: pd.Series) -> np.ndarray:
    """Disagree-aware blend + overconfidence shrink.

    Error analysis #2 showed that when our model picks a different side than
    the market, the market is right +2.1pp more often. So when there's
    DIRECTIONAL disagreement, we lean harder on market.

    Walk-forward (4,411 games): w_agree=0.50, w_disagree=0.80 yields:
      ll=0.6778  acc=0.5695  Brier=0.2425
    vs market only:
      ll=0.6760  acc=0.5622  Brier=0.2417
    → +0.73pp accuracy ABOVE market while remaining well-calibrated.
    """
    if c["cls"].get("__ensemble__"):
        p_model = c["cls"]["model"].predict_proba(X)[:, 1]
        mp = market_p.values
        if not np.isfinite(mp[0]):
            # No market line — fall back to pure model
            return _shrink_overconfidence(p_model)
        # Directional agreement check
        agree = (p_model > 0.5) == (mp > 0.5)
        w = np.where(agree, W_AGREE, W_DISAGREE)
        blended = w * mp + (1 - w) * p_model
        return _shrink_overconfidence(blended)
    return c["cls"]["calibrator"].predict_proba(X)[:, 1]


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
    p_home = float(c["cls"]["calibrator"].predict_proba(X)[0, 1])
    total = float(c["reg"]["model"].predict(X)[0])
    return p_home, total


# ---------------- endpoints ----------------
@app.get("/api/games")
async def games(start: str | None = None, end: str | None = None, days: int = 7):
    c = _load()
    train = c["train"]
    today = date.today()
    s = date.fromisoformat(start) if start else today
    e = date.fromisoformat(end) if end else (s + timedelta(days=days))
    # If the window touches today/future, opportunistically refresh ESPN odds
    # so the user always sees fresh market_lines. Throttled via _last_refresh.
    if e >= today:
        await _maybe_refresh_now()
    sub = train[(train["game_date"] >= s) & (train["game_date"] <= e)].copy()
    # Merge first_pitch_utc from games.parquet so the response can carry the
    # scheduled start time and the slot-within-day.
    games_df = c["games"][["game_pk", "first_pitch_utc"]]
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

    # Parallel-prefetch fresh live state for today/yesterday so the per-game
    # loop below uses the cache instead of blocking on sequential HTTP.
    from concurrent.futures import ThreadPoolExecutor
    refresh_pks = [int(r["game_pk"]) for _, r in sub.iterrows()
                   if r["game_date"] >= today - timedelta(days=1)]
    if refresh_pks:
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_fetch_live_feed, refresh_pks))
    for _, r in sub.iterrows():
        X = pd.DataFrame([r[feats].values], columns=feats)
        try:
            p_home = float(_predict_phome(c, X, pd.Series([r.get("market_p_home", float("nan"))]))[0])
            total = float(c["reg"]["model"].predict(X)[0])
        except Exception as e:
            import traceback
            print(f"[predict ERROR for pk={r['game_pk']}]: {type(e).__name__}: {e}")
            traceback.print_exc()
            p_home, total = float("nan"), float("nan")
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
        if np.isfinite(p_home):
            pick_abbrev = r["home_team_abbrev"] if p_home >= 0.5 else r["away_team_abbrev"]
        # Value pick (positive-edge side at real market price) + decimal odds
        v = _value_block(c, int(r["game_pk"]), p_home,
                          r["away_team_abbrev"], r["home_team_abbrev"],
                          slot=int(r["slot"]) if pd.notna(r.get("slot")) else None)
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
            "pred_total": _safe(total),
            "market_p_home": _safe(r.get("market_p_home")),
            "market_total": _safe(r.get("market_over_under")),
            "fair_home_ml": _to_american(p_home) if np.isfinite(p_home) else None,
            "fair_away_ml": _to_american(1 - p_home) if np.isfinite(p_home) else None,
            "pick_abbrev": pick_abbrev,
            **v,
            **f5,
            "is_played": is_final,
            "is_live": is_live,
            "game_state": gs,
            "live_score": live,
            "home_score": home_score,
            "away_score": away_score,
        })
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
        p_home = float(_predict_phome(c, X, pd.Series([r.get("market_p_home", float("nan"))]))[0])
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
            "pred_total_runs": _safe(total),
            "fair_home_ml": _to_american(p_home) if np.isfinite(p_home) else None,
            "fair_away_ml": _to_american(1 - p_home) if np.isfinite(p_home) else None,
        },
        "value": _value_block(c, game_pk, p_home,
                                r["away_team_abbrev"], r["home_team_abbrev"],
                                slot=detail_slot),
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
    }


# ---------- static frontend ----------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/strike")
def strike():
    return FileResponse(STATIC_DIR / "strike.html")
