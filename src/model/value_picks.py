"""Value picks model — LGBM sobre features del juego para detectar edge vs mercado.

Entrena dos clasificadores binarios walk-forward por temporada:
  A) p_home_scored_first  — probabilidad de que el HOME anote primero
  B) p_home_win           — probabilidad de que el HOME gane el juego

Compara con la implied probability del mercado (mediana entre providers,
overround removido) y genera picks etiquetados por tier.

Escribe: data/processed/value_picks.parquet
Columnas:
  game_pk, game_date, season,
  p_home_first, p_home_win, imp_home,
  edge_first, edge_win,
  pick_code, tier

Uso:
    python -m src.model.value_picks              # entrena + predice todo
    python -m src.model.value_picks --refresh    # (alias del default)

Modelos guardados en data/models/value_first.pkl y value_win.pkl.

Tiers:
    'VALOR-FUERTE'  edge_win >= 0.07 AND edge_first mismo signo
    'VALOR'         edge_win >= 0.05
    'NEUTRO'        edge_win < 0.05  (no pick)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ..normalize.paths import PROCESSED

MODELS_DIR = PROCESSED.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = PROCESSED / "value_picks.parquet"

TIER_STRONG = 0.07
TIER_VALUE = 0.05


def _ml_to_prob(ml: float | None) -> float | None:
    if ml is None or pd.isna(ml):
        return None
    return 100.0 / (ml + 100.0) if ml > 0 else abs(ml) / (abs(ml) + 100.0)


def _load_target() -> pd.DataFrame:
    """First-scorer target: 1 si HOME anotó primero, 0 si visitante."""
    con = duckdb.connect(":memory:")
    return con.execute(f"""
    WITH terminating AS (
      SELECT game_pk, at_bat_number, inning_topbot,
             home_score, away_score, post_home_score, post_away_score
      FROM read_parquet('{PROCESSED / "pitches.parquet"}')
      WHERE events IS NOT NULL AND events <> ''
    ),
    first_scoring AS (
      SELECT * FROM (
        SELECT t.*, ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY at_bat_number) rn
        FROM terminating t
        WHERE (post_home_score+post_away_score) > (home_score+away_score)
      ) WHERE rn = 1
    )
    SELECT game_pk,
           CASE WHEN inning_topbot='Bot' THEN 1 ELSE 0 END AS home_scored_first
    FROM first_scoring
    """).fetch_df()


def _load_odds() -> pd.DataFrame:
    """Median ML across providers. Prefiere close (juego terminado) sobre current
    (juego pending). Para upcoming games, close es NaN — se usa current."""
    con = duckdb.connect(":memory:")
    return con.execute(f"""
    WITH per_provider AS (
      SELECT x.game_pk,
             COALESCE(o.home_ml_close, o.home_ml_current) AS home_ml_eff,
             COALESCE(o.away_ml_close, o.away_ml_current) AS away_ml_eff,
             COALESCE(o.total_close,   o.total_open)      AS total_eff
      FROM read_parquet('{PROCESSED / "games_xref.parquet"}') x
      JOIN read_parquet('{PROCESSED / "odds_close.parquet"}') o
           ON o.espn_event_id = x.espn_event_id
      WHERE x.game_pk IS NOT NULL
    )
    SELECT game_pk,
           MEDIAN(home_ml_eff) FILTER (WHERE home_ml_eff BETWEEN -600 AND 600) AS home_ml,
           MEDIAN(away_ml_eff) FILTER (WHERE away_ml_eff BETWEEN -600 AND 600) AS away_ml,
           MEDIAN(total_eff)   FILTER (WHERE total_eff   BETWEEN 5 AND 15)      AS total_close
    FROM per_provider
    GROUP BY game_pk
    """).fetch_df()


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    """Ensambla features + target + odds en una tabla por game_pk."""
    con = duckdb.connect(":memory:")

    games = con.execute(f"""
    SELECT game_pk, season, game_date::DATE AS d, venue_id,
           home_team_abbrev, away_team_abbrev,
           probable_home_pitcher_id AS hp_pid,
           probable_away_pitcher_id AS ap_pid,
           CASE WHEN home_score>away_score THEN 1
                WHEN away_score>home_score THEN 0 END AS home_won,
           status
    FROM read_parquet('{PROCESSED / "games.parquet"}')
    WHERE game_type='R' AND home_team_abbrev IS NOT NULL
    """).fetch_df()

    elo = pd.read_parquet(PROCESSED / "features_elo.parquet")
    pit = pd.read_parquet(PROCESSED / "features_pitcher.parquet").drop_duplicates(["game_pk", "pitcher"])
    team = pd.read_parquet(PROCESSED / "features_team.parquet")
    line = pd.read_parquet(PROCESSED / "features_lineup.parquet")
    wx = pd.read_parquet(PROCESSED / "features_weather.parquet")
    park = pd.read_parquet(PROCESSED / "features_park.parquet")

    team_h = (team[team["is_home"] == 1]
              .add_prefix("h_")
              .rename(columns={"h_game_pk": "game_pk"})
              .drop(columns=["h_team", "h_game_date", "h_is_home"], errors="ignore"))
    team_a = (team[team["is_home"] == 0]
              .add_prefix("a_")
              .rename(columns={"a_game_pk": "game_pk"})
              .drop(columns=["a_team", "a_game_date", "a_is_home"], errors="ignore"))
    line_h = (line[line["side"] == "home"]
              .add_prefix("hL_")
              .rename(columns={"hL_game_pk": "game_pk"})
              .drop(columns=["hL_side"], errors="ignore"))
    line_a = (line[line["side"] == "away"]
              .add_prefix("aL_")
              .rename(columns={"aL_game_pk": "game_pk"})
              .drop(columns=["aL_side"], errors="ignore"))

    pit_h = pit.rename(columns={c: f"h_{c}" for c in pit.columns if c not in ("game_pk", "pitcher")})
    pit_a = pit.rename(columns={c: f"a_{c}" for c in pit.columns if c not in ("game_pk", "pitcher")})

    df = games.merge(_load_target(), on="game_pk", how="left")
    df = df.merge(elo, on="game_pk", how="left")
    df = df.merge(pit_h, left_on=["game_pk", "hp_pid"], right_on=["game_pk", "pitcher"], how="left").drop(columns=["pitcher"], errors="ignore")
    df = df.merge(pit_a, left_on=["game_pk", "ap_pid"], right_on=["game_pk", "pitcher"], how="left").drop(columns=["pitcher"], errors="ignore")
    df = df.merge(team_h, on="game_pk", how="left")
    df = df.merge(team_a, on="game_pk", how="left")
    df = df.merge(line_h, on="game_pk", how="left")
    df = df.merge(line_a, on="game_pk", how="left")
    df = df.merge(wx, on="game_pk", how="left")
    df = df.merge(park, on=["season", "venue_id"], how="left")
    df = df.merge(_load_odds(), on="game_pk", how="left")

    feat_cols = [c for c in df.columns if c not in {
        "game_pk", "season", "d", "venue_id", "home_team_abbrev", "away_team_abbrev",
        "hp_pid", "ap_pid", "home_won", "home_scored_first", "status",
        "h_pitcher", "a_pitcher", "h_game_date", "a_game_date",
    } and df[c].dtype != "O"]

    return df, feat_cols


def train_and_predict() -> pd.DataFrame:
    import lightgbm as lgb

    df, feat_cols = build_dataset()

    # Entrenamos con juegos que YA se jugaron y tienen target (past).
    # Predecimos sobre TODOS los juegos con features (past + future).
    trained = df[df["home_scored_first"].notna() & df["home_won"].notna()].copy()
    if len(trained) < 500:
        raise RuntimeError(f"Muy pocos juegos con target ({len(trained)}); se necesita al menos 500.")

    def fit(target_col: str):
        m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=5,
                               num_leaves=15, min_data_in_leaf=40, verbose=-1)
        m.fit(trained[feat_cols], trained[target_col].astype(int))
        return m

    m_first = fit("home_scored_first")
    m_win = fit("home_won")

    # Predicción sobre todo el pool con features (falta target incluído)
    out = df[["game_pk", "d", "season", "home_team_abbrev", "away_team_abbrev",
              "home_ml", "away_ml", "status"]].copy()
    out.rename(columns={"d": "game_date"}, inplace=True)
    out["p_home_first"] = m_first.predict_proba(df[feat_cols])[:, 1]
    out["p_home_win"] = m_win.predict_proba(df[feat_cols])[:, 1]

    ip_home = out["home_ml"].apply(_ml_to_prob)
    ip_away = out["away_ml"].apply(_ml_to_prob)
    out["imp_home"] = ip_home / (ip_home + ip_away)

    out["edge_first"] = out["p_home_first"] - out["imp_home"]
    out["edge_win"] = out["p_home_win"] - out["imp_home"]

    # Tier + pick code
    def _classify(row):
        ew, ef = row["edge_win"], row["edge_first"]
        if pd.isna(ew):
            return "NEUTRO", None
        # signo del edge decide el lado
        home_side = ew > 0
        code = row["home_team_abbrev"] if home_side else row["away_team_abbrev"]
        mag = abs(ew)
        # confirmacion cross-model: edge_first debe ir en el mismo signo
        same_dir = (not pd.isna(ef)) and ((ef > 0) == home_side) and abs(ef) >= TIER_VALUE
        if mag >= TIER_STRONG and same_dir:
            return "VALOR-FUERTE", code
        if mag >= TIER_VALUE:
            return "VALOR", code
        return "NEUTRO", None

    tier_code = out.apply(_classify, axis=1, result_type="expand")
    out["tier"] = tier_code[0]
    out["pick_code"] = tier_code[1]

    # Guardar modelos por si queremos reusarlos
    import joblib
    joblib.dump(m_first, MODELS_DIR / "value_first.pkl")
    joblib.dump(m_win, MODELS_DIR / "value_win.pkl")

    # Orden y salida
    out = out[[
        "game_pk", "game_date", "season", "home_team_abbrev", "away_team_abbrev",
        "home_ml", "away_ml", "imp_home",
        "p_home_first", "p_home_win", "edge_first", "edge_win",
        "pick_code", "tier", "status",
    ]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="Force retrain + rewrite (default behavior)")
    ap.add_argument("--summary", action="store_true", help="Print summary of tier counts")
    args = ap.parse_args()

    print("[value_picks] entrenando y prediciendo…")
    picks = train_and_predict()
    picks.to_parquet(OUT_PATH, index=False)
    print(f"[value_picks] escritos {len(picks):,} juegos → {OUT_PATH}")

    if args.summary or True:
        by_tier = picks["tier"].value_counts()
        print("\n=== Tier counts (todos los juegos) ===")
        print(by_tier.to_string())
        upcoming = picks[picks["status"].isin(["Scheduled", "Pre-Game", "Warmup", "In Progress"])]
        if len(upcoming):
            print(f"\n=== Upcoming games ({len(upcoming)}) ===")
            print(upcoming["tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
