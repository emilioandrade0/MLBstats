"""Motor de momios — walk-forward simple sobre closing lines por equipo.

Lee `odds_close.parquet` + `games.parquet` + `games_xref.parquet` desde
STRIKECAST y produce dos JSON en StatsMLB/public/data/:

  - odds-walkforward.json     Métricas globales del walk-forward por temporada.
  - odds-predictions-today.json  Predicciones para el slate de hoy.

Modelo (por diseño simple):
  * naive:        momio predicho = último momio del equipo
  * roll5/roll10: media móvil de los últimos 5 / 10 momios del equipo
  * blend:        promedio(naive, roll5)  → suele batir a los tres por sí solo

Métricas:
  * MAE en centavos americanos
  * MAE en decimal
  * Acierto de dirección (¿subió o bajó respecto al juego previo?)
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PUBLIC_DATA = APP / "public" / "data"

SHARP_BOOKS = {"DraftKings", "ESPN BET"}
SHARP_BOOK_KEYS_ODDSAPI = {"draftkings", "fanduel", "betmgm", "pinnacle"}
MIN_HISTORY = 6          # juegos previos requeridos para predecir
WALK_START_SEASON = 2026 # ventana de test (2023-2025 = train implícito por historia)
RAW_ODDS_API_DIR = ROOT / "data" / "raw" / "odds_api"


def american_to_decimal(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v == 0:
        return None
    return 1 + (v / 100 if v > 0 else 100 / abs(v))


def load_moneylines() -> pd.DataFrame:
    """Devuelve un frame juego-nivel: game_pk, game_date, season, home/away teams, home_ml, away_ml."""
    odds = pd.read_parquet(PROCESSED / "odds_close.parquet")
    sharp = odds[odds["provider_name"].isin(SHARP_BOOKS)].copy()
    for side in ("home", "away"):
        sharp[f"{side}_ml"] = (
            sharp[f"{side}_ml_close"]
            .fillna(sharp[f"{side}_ml_current"])
            .fillna(sharp[f"{side}_ml_top"])
        )
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id", as_index=False).agg(
        home_ml=("home_ml", "median"),
        away_ml=("away_ml", "median"),
    )

    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    agg = agg.merge(matched, on="espn_event_id", how="inner").drop_duplicates("game_pk")

    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "season", "home_team_abbrev", "away_team_abbrev"],
    ).drop_duplicates("game_pk")
    games["game_date"] = pd.to_datetime(games["game_date"])

    df = games.merge(agg[["game_pk", "home_ml", "away_ml"]], on="game_pk", how="inner")
    return df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """Un renglón por (game, equipo) con el momio de SU lado en ese juego."""
    home_side = df.rename(columns={
        "home_team_abbrev": "team",
        "away_team_abbrev": "opponent",
        "home_ml": "ml",
        "away_ml": "opp_ml",
    }).assign(is_home=1)
    away_side = df.rename(columns={
        "away_team_abbrev": "team",
        "home_team_abbrev": "opponent",
        "away_ml": "ml",
        "home_ml": "opp_ml",
    }).assign(is_home=0)
    cols = ["game_pk", "game_date", "season", "team", "opponent", "is_home", "ml", "opp_ml"]
    stacked = pd.concat([home_side[cols], away_side[cols]], ignore_index=True)
    return stacked.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)


def add_walkforward_features(team_hist: pd.DataFrame) -> pd.DataFrame:
    """Para cada renglón: features usando SOLO juegos previos del mismo equipo."""
    out = team_hist.copy()
    grouped = out.groupby("team", sort=False)["ml"]

    # shift(1) para excluir el juego actual.
    prev = grouped.shift(1)
    out["prev_ml"] = prev
    out["roll5"] = grouped.shift(1).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    out["roll10"] = grouped.shift(1).rolling(10, min_periods=5).mean().reset_index(level=0, drop=True)
    out["roll10_std"] = grouped.shift(1).rolling(10, min_periods=5).std().reset_index(level=0, drop=True)
    out["games_seen"] = grouped.cumcount()  # cantidad de juegos PREVIOS
    return out


def american_direction(a: float, b: float) -> int:
    """Signo de a-b tratando el momio americano de forma monotónica sobre prob implícita.

    Convertimos a probabilidad implícita para comparar magnitudes reales.
    """
    def imp(x):
        if x is None or not np.isfinite(x):
            return np.nan
        return abs(x) / (abs(x) + 100) if x < 0 else 100 / (x + 100)
    ia, ib = imp(a), imp(b)
    if np.isnan(ia) or np.isnan(ib):
        return 0
    return int(np.sign(ia - ib))


def eval_predictions(actual: np.ndarray, predicted: np.ndarray, prev: np.ndarray) -> dict:
    """MAE americana, MAE decimal, y % acierto de dirección vs momio previo."""
    mask = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(prev)
    a, p, pv = actual[mask], predicted[mask], prev[mask]
    if len(a) == 0:
        return {"n": 0, "mae_american": None, "mae_decimal": None, "direction_accuracy": None}

    mae_am = float(np.mean(np.abs(a - p)))
    dec_a = np.array([american_to_decimal(x) for x in a], dtype=float)
    dec_p = np.array([american_to_decimal(x) for x in p], dtype=float)
    mae_dec = float(np.nanmean(np.abs(dec_a - dec_p)))

    actual_dir = np.array([american_direction(x, y) for x, y in zip(a, pv)])
    pred_dir = np.array([american_direction(x, y) for x, y in zip(p, pv)])
    considered = actual_dir != 0
    if considered.sum() == 0:
        dir_acc = None
    else:
        dir_acc = float(np.mean(actual_dir[considered] == pred_dir[considered]))

    return {
        "n": int(len(a)),
        "mae_american": round(mae_am, 2),
        "mae_decimal": round(mae_dec, 4),
        "direction_accuracy": round(dir_acc, 4) if dir_acc is not None else None,
    }


def run_walkforward(team_hist: pd.DataFrame) -> dict:
    df = add_walkforward_features(team_hist)
    df = df[df["games_seen"] >= MIN_HISTORY].copy()

    # Predicciones por modelo.
    df["pred_naive"] = df["prev_ml"]
    df["pred_roll5"] = df["roll5"]
    df["pred_roll10"] = df["roll10"]
    df["pred_blend"] = (df["prev_ml"] + df["roll5"]) / 2.0

    seasons = sorted(df["season"].dropna().unique().tolist())
    by_season = {}
    for season in seasons:
        sub = df[df["season"].eq(season)]
        by_season[int(season)] = {
            "games": int(len(sub)),
            "models": {
                model: eval_predictions(
                    sub["ml"].to_numpy(dtype=float),
                    sub[f"pred_{model}"].to_numpy(dtype=float),
                    sub["prev_ml"].to_numpy(dtype=float),
                )
                for model in ("naive", "roll5", "roll10", "blend")
            },
        }

    overall = df
    summary = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "STRIKECAST data/processed/odds_close.parquet",
        "providersUsed": sorted(SHARP_BOOKS),
        "minHistory": MIN_HISTORY,
        "totalTeamGameRows": int(len(overall)),
        "seasons": by_season,
        "overall": {
            "games": int(len(overall)),
            "models": {
                model: eval_predictions(
                    overall["ml"].to_numpy(dtype=float),
                    overall[f"pred_{model}"].to_numpy(dtype=float),
                    overall["prev_ml"].to_numpy(dtype=float),
                )
                for model in ("naive", "roll5", "roll10", "blend")
            },
        },
    }
    return summary


def latest_team_state(team_hist: pd.DataFrame) -> dict:
    """Estado más reciente por equipo: último ml + rolls listos para predecir el siguiente juego."""
    df = add_walkforward_features(team_hist)
    latest = (
        df.sort_values(["team", "game_date", "game_pk"])
          .groupby("team", as_index=False)
          .tail(1)
    )
    state = {}
    for _, row in latest.iterrows():
        team = row["team"]
        prev_ml = row["ml"]           # el ml del último juego será el "prev" del próximo
        # roll actualizado: incluir el último juego (por eso recalculamos aquí).
        past = df[df["team"].eq(team)].tail(10)["ml"].tolist() + [prev_ml]
        past = [x for x in past if isinstance(x, (int, float)) and np.isfinite(x)]
        roll5 = float(np.mean(past[-5:])) if len(past) >= 3 else None
        roll10 = float(np.mean(past[-10:])) if len(past) >= 5 else None
        state[team] = {
            "lastGameDate": row["game_date"].strftime("%Y-%m-%d") if pd.notna(row["game_date"]) else None,
            "lastGamePk": int(row["game_pk"]),
            "lastMl": float(prev_ml) if pd.notna(prev_ml) else None,
            "roll5": round(roll5, 2) if roll5 is not None else None,
            "roll10": round(roll10, 2) if roll10 is not None else None,
            "gamesSeen": int(row["games_seen"]) + 1,
        }
    return state


TEAM_NAME_TO_ABBREV = {
    "Arizona Diamondbacks": "AZ", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF", "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}


def build_today_predictions(state: dict) -> dict:
    """Cruza con today-snapshot.json para armar predicciones del slate."""
    snapshot_path = PUBLIC_DATA / "today-snapshot.json"
    if not snapshot_path.exists():
        return {"generatedAt": datetime.now(timezone.utc).isoformat(), "games": [], "error": "today-snapshot.json no existe"}

    odds_path = PUBLIC_DATA / "current-odds.json"
    current_odds_index = {}
    if odds_path.exists():
        try:
            current = json.loads(odds_path.read_text(encoding="utf-8"))
            for game in current.get("games", []):
                away_abbr = TEAM_NAME_TO_ABBREV.get(game.get("awayTeam", ""))
                home_abbr = TEAM_NAME_TO_ABBREV.get(game.get("homeTeam", ""))
                if away_abbr and home_abbr:
                    current_odds_index[(away_abbr, home_abbr)] = game
        except Exception:
            pass

    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    predictions = []
    for game in snap.get("games", []):
        away = game.get("away", {}).get("team")
        home = game.get("home", {}).get("team")
        away_state = state.get(away)
        home_state = state.get(home)
        if not away_state or not home_state:
            continue

        def predict(s):
            if s["lastMl"] is None or s["roll5"] is None:
                return None
            return round((s["lastMl"] + s["roll5"]) / 2.0, 1)

        away_pred = predict(away_state)
        home_pred = predict(home_state)

        current_market = current_odds_index.get((away, home))
        market_away = market_home = None
        if current_market:
            best_away = current_market.get("bestAway") or {}
            best_home = current_market.get("bestHome") or {}
            market_away = best_away.get("american")
            market_home = best_home.get("american")

        delta_away = (market_away - away_pred) if (market_away is not None and away_pred is not None) else None
        delta_home = (market_home - home_pred) if (market_home is not None and home_pred is not None) else None

        predictions.append({
            "gamePk": game.get("gamePk"),
            "gameDate": game.get("gameDate"),
            "awayTeam": away,
            "homeTeam": home,
            "awayPredicted": away_pred,
            "homePredicted": home_pred,
            "awayMarket": market_away,
            "homeMarket": market_home,
            "awayDelta": round(delta_away, 1) if delta_away is not None else None,
            "homeDelta": round(delta_home, 1) if delta_home is not None else None,
            "awayLast": away_state["lastMl"],
            "homeLast": home_state["lastMl"],
            "awayRoll5": away_state["roll5"],
            "homeRoll5": home_state["roll5"],
            "awayGamesSeen": away_state["gamesSeen"],
            "homeGamesSeen": home_state["gamesSeen"],
        })

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": "blend(prev_ml, roll5)",
        "note": "Delta = mercado - predicción. Delta positivo (favorito) sugiere que el mercado paga MÁS de lo que este modelo simple espera para ese lado.",
        "games": predictions,
    }


def build_odds_history() -> dict:
    """Snapshot de mercado histórico por game_pk directo desde odds_close.parquet.

    Antes leía walkforward_preds.parquet, pero ese archivo depende del reentreno
    del modelo y suele estar más atrasado que odds_close (que se actualiza más
    seguido). Calculamos el market_p_home aquí desvigando la mediana de
    DraftKings/ESPN BET.
    """
    odds = pd.read_parquet(PROCESSED / "odds_close.parquet")
    sharp = odds[odds["provider_name"].isin(SHARP_BOOKS)].copy()
    for side in ("home", "away"):
        sharp[f"{side}_ml"] = (
            sharp[f"{side}_ml_close"]
            .fillna(sharp[f"{side}_ml_current"])
            .fillna(sharp[f"{side}_ml_top"])
        )
    sharp = sharp.dropna(subset=["home_ml", "away_ml"])
    agg = sharp.groupby("espn_event_id", as_index=False).agg(
        home_ml=("home_ml", "median"),
        away_ml=("away_ml", "median"),
    )
    xref = pd.read_parquet(PROCESSED / "games_xref.parquet")
    matched = xref[xref["_merge"] == "both"][["espn_event_id", "game_pk"]].dropna()
    matched["game_pk"] = matched["game_pk"].astype(int)
    agg = agg.merge(matched, on="espn_event_id", how="inner").drop_duplicates("game_pk")

    agg["dec_home"] = agg["home_ml"].apply(american_to_decimal)
    agg["dec_away"] = agg["away_ml"].apply(american_to_decimal)
    agg = agg.dropna(subset=["dec_home", "dec_away"])
    ih = 1 / agg["dec_home"]
    ia = 1 / agg["dec_away"]
    agg["market_p_home"] = ih / (ih + ia)

    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date"],
    ).drop_duplicates("game_pk")
    games["game_date"] = pd.to_datetime(games["game_date"])
    merged = agg.merge(games, on="game_pk", how="left")

    entries = {}
    for _, row in merged.iterrows():
        entries[int(row["game_pk"])] = {
            "marketPHome": float(row["market_p_home"]),
            "gameDate": row["game_date"].strftime("%Y-%m-%d") if pd.notna(row["game_date"]) else None,
        }

    # Suplemento: raw/odds_api/*.json trae momios frescos que odds_close.parquet
    # aún no incorpora (STRIKECAST no ha corrido su pipeline). Ingresamos aquí
    # el snapshot más reciente por juego para no dejar septiembre sin mercado.
    added = _augment_from_raw_oddsapi(entries)
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "odds_close.parquet + raw/odds_api/*.json (mediana books sharp, desvigada)",
        "rawSupplementCount": added,
        "games": entries,
    }


def _augment_from_raw_oddsapi(entries: dict) -> int:
    """Añade momios de `raw/odds_api/*.json` para juegos no cubiertos por odds_close.

    Cruza por (fecha de commence + nombres de equipos) contra games.parquet.
    Solo escribe entradas nuevas — nunca sobrescribe closing lines ya presentes.
    """
    if not RAW_ODDS_API_DIR.is_dir():
        return 0
    files = sorted(RAW_ODDS_API_DIR.glob("*.json"))
    if not files:
        return 0

    games = pd.read_parquet(
        PROCESSED / "games.parquet",
        columns=["game_pk", "game_date", "first_pitch_utc", "home_team_name", "away_team_name"],
    ).drop_duplicates("game_pk")
    games["first_pitch_utc"] = pd.to_datetime(games["first_pitch_utc"], utc=True, errors="coerce")
    games["date_key"] = games["first_pitch_utc"].dt.strftime("%Y-%m-%d")

    # Index (date_key, home_team_name, away_team_name) → game_pk / game_date
    game_index = {}
    for _, row in games.dropna(subset=["date_key"]).iterrows():
        key = (row["date_key"], row["home_team_name"], row["away_team_name"])
        game_index[key] = (int(row["game_pk"]), row["date_key"])

    # Para cada evento, guardamos el pull más reciente que lo contenga.
    latest_by_event: dict[str, tuple[float, dict]] = {}
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pull_ts = path.stem  # 2026-09-08T063706Z
        for event in data if isinstance(data, list) else []:
            eid = event.get("id")
            if not eid:
                continue
            prev = latest_by_event.get(eid)
            if prev is None or pull_ts > prev[0]:
                latest_by_event[eid] = (pull_ts, event)

    added = 0
    for _, event in latest_by_event.values():
        commence = event.get("commence_time")
        if not commence:
            continue
        try:
            date_key = pd.to_datetime(commence, utc=True).strftime("%Y-%m-%d")
        except Exception:
            continue
        key = (date_key, event.get("home_team"), event.get("away_team"))
        matched = game_index.get(key)
        if not matched:
            continue
        game_pk, game_date = matched
        if game_pk in entries:
            continue  # ya cubierto por odds_close

        home_probs, away_probs = [], []
        for book in event.get("bookmakers", []):
            if book.get("key") not in SHARP_BOOK_KEYS_ODDSAPI:
                continue
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outs = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
                home_price = outs.get(event.get("home_team"))
                away_price = outs.get(event.get("away_team"))
                dh = american_to_decimal(home_price)
                da = american_to_decimal(away_price)
                if dh and da:
                    ih, ia = 1 / dh, 1 / da
                    home_probs.append(ih / (ih + ia))
                    away_probs.append(ia / (ih + ia))
        if not home_probs:
            continue
        entries[game_pk] = {
            "marketPHome": float(np.mean(home_probs)),
            "gameDate": game_date,
        }
        added += 1
    return added


def main():
    print(f"[odds_walkforward] Leyendo {PROCESSED}")
    df = load_moneylines()
    print(f"[odds_walkforward] {len(df):,} juegos con momio disponible")
    team_hist = build_team_history(df)
    print(f"[odds_walkforward] {len(team_hist):,} renglones equipo-juego")

    summary = run_walkforward(team_hist)
    (PUBLIC_DATA / "odds-walkforward.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[odds_walkforward] escrito odds-walkforward.json")

    state = latest_team_state(team_hist)
    today = build_today_predictions(state)
    (PUBLIC_DATA / "odds-predictions-today.json").write_text(
        json.dumps(today, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[odds_walkforward] escrito odds-predictions-today.json ({len(today['games'])} juegos)")

    history = build_odds_history()
    (PUBLIC_DATA / "odds-history.json").write_text(
        json.dumps(history, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[odds_walkforward] escrito odds-history.json ({len(history['games'])} juegos)")


if __name__ == "__main__":
    main()
