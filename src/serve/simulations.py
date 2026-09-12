"""Shadow game-component simulations and auditable history for STRIKECAST.

This module deliberately does not alter the official winner path.  It trains
the validated Phase 13 component engines on completed games, links them with
the Phase 14 run reconciler, and records one immutable shadow snapshot per
game/engine.  Historical accuracy is sourced from chronological OOS folds;
prospective snapshots are graded only when a final score is available.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from scipy.stats import nbinom, poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_DIR = Path(__file__).resolve().parents[2]
LAB_DIR = REPO_DIR / "STARTFROMTHEEND"
OUTPUT_DIR = LAB_DIR / "outputs"
PHASE13_PATH = LAB_DIR / "src" / "13_component_engines.py"
PHASE14_PATH = LAB_DIR / "src" / "14_linked_game_simulator.py"
PHASE14_HISTORY_PATH = OUTPUT_DIR / "14_linked_simulation_predictions.csv"
PHASE14_MANIFEST_PATH = OUTPUT_DIR / "14_linked_game_simulator_manifest.json"
GAME_SCRIPT_PATH = OUTPUT_DIR / "03_game_script_games.csv"
TEAM_BOX_PATH = REPO_DIR / "data" / "processed" / "team_box.parquet"
SNAPSHOT_PATH = OUTPUT_DIR / "15_shadow_simulation_history.parquet"
BUNDLE_PATH = OUTPUT_DIR / "15_shadow_model_bundle.joblib"
SERVICE_VERSION = "phase15-shadow-v1"

# A component is graded "correct" when its absolute error is no greater than
# the validated chronological OOS MAE reported by Phase 13 (2026-08-04 run).
COMPONENT_TOLERANCES = {
    "hits": 2.6596,
    "baserunners": 3.3990,
    "total_bases": 5.0617,
    "left_on_base": 4.3652,
}

_bundle_lock = threading.Lock()
_snapshot_lock = threading.Lock()
_bundle_cache: dict[str, Any] = {}

# --- Totals O/U probability (NegBin convolution of two team run distributions) ---
# Verified 2026-08-05: hit rate 0.5699 (2026), 0.5627 (H1), 0.5909 (H2) with
# isotonic calibration; Brier beats market baseline in all three windows. See
# memory/project_sim_totals_edge.md for the full validation.
_TOTAL_RUN_SUPPORT = 40


def _negbin_pmf(mu: float, alpha: float, k: np.ndarray) -> np.ndarray:
    if alpha <= 1e-9:
        return poisson.pmf(k, mu)
    size = 1.0 / alpha
    p = size / (size + mu)
    return nbinom.pmf(k, size, p)


def total_probability_over(mu_home: float, mu_away: float, alpha: float, line: float) -> float:
    """P(home_runs + away_runs > line) via convolution of two NegBins."""
    k = np.arange(_TOTAL_RUN_SUPPORT + 1)
    pmf_h = _negbin_pmf(float(mu_home), float(alpha), k)
    pmf_a = _negbin_pmf(float(mu_away), float(alpha), k)
    pmf_h = pmf_h / pmf_h.sum()
    pmf_a = pmf_a / pmf_a.sum()
    conv = np.convolve(pmf_h, pmf_a)
    threshold = int(np.floor(float(line)))
    return float(conv[threshold + 1 :].sum())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_script(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lineage_key() -> str:
    paths = [
        REPO_DIR / "data" / "processed" / "train.parquet",
        REPO_DIR / "data" / "processed" / "team_box.parquet",
        PHASE13_PATH,
        PHASE14_PATH,
        PHASE14_HISTORY_PATH,
    ]
    values = []
    for path in paths:
        stat = path.stat()
        values.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()[:16]


def _engine_id() -> str:
    try:
        manifest = json.loads(PHASE14_MANIFEST_PATH.read_text(encoding="utf-8"))
        return str(manifest.get("engine_id") or "unknown")
    except Exception:
        return "unknown"


def _component_features(frame: pd.DataFrame, phase13: Any) -> list[str]:
    own = phase13.available_pair_features(frame, phase13.OWN_FEATURES)
    opponent = phase13.available_pair_features(frame, phase13.OPPONENT_FEATURES)
    features = [
        *[f"own_{name}" for name in own],
        *[f"opp_{name}" for name in opponent],
        "own_venue_run_diff", "own_venue_win_pct", *phase13.COMMON_FEATURES,
    ]
    phase13.validate_feature_contract(features)
    return features


def _inference_team_rows(frame: pd.DataFrame, phase13: Any, features: list[str]) -> pd.DataFrame:
    """Build the exact Phase 13 pregame feature contract without postgame targets."""
    own = phase13.available_pair_features(frame, phase13.OWN_FEATURES)
    opponent = phase13.available_pair_features(frame, phase13.OPPONENT_FEATURES)
    parts: list[pd.DataFrame] = []
    for side, other, is_home in [("h", "a", 1), ("a", "h", 0)]:
        part = frame[["game_pk", "game_date", "season"]].copy()
        part["side"] = side
        team_column = "home_team_abbrev" if side == "h" else "away_team_abbrev"
        part["team"] = frame[team_column]
        part["is_home"] = is_home
        for name in own:
            part[f"own_{name}"] = pd.to_numeric(frame[f"{name}_{side}"], errors="coerce")
        for name in opponent:
            part[f"opp_{name}"] = pd.to_numeric(frame[f"{name}_{other}"], errors="coerce")
        venue_run = "homeonly_run_diff_l10_h" if side == "h" else "awayonly_run_diff_l10_a"
        venue_win = "homeonly_win_pct_l20_h" if side == "h" else "awayonly_win_pct_l20_a"
        part["own_venue_run_diff"] = pd.to_numeric(frame.get(venue_run), errors="coerce")
        part["own_venue_win_pct"] = pd.to_numeric(frame.get(venue_win), errors="coerce")
        sign = 1.0 if side == "h" else -1.0
        part["weather_temp_f"] = pd.to_numeric(frame.get("weather_temp_f"), errors="coerce")
        part["park_runs_factor"] = pd.to_numeric(frame.get("park_runs_factor"), errors="coerce")
        part["rest_edge"] = pd.to_numeric(frame.get("days_since_last_game_diff"), errors="coerce") * sign
        part["elo_edge"] = pd.to_numeric(frame.get("elo_diff_pre"), errors="coerce") * sign
        for target in phase13.TARGETS:
            target_column = f"{target}_{side}"
            if target_column in frame:
                part[target] = pd.to_numeric(frame[target_column], errors="coerce")
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    missing = [name for name in features if name not in result]
    if missing:
        raise ValueError(f"Faltan features Phase 13 en el slate: {missing}")
    return result.sort_values(["game_date", "game_pk", "side"]).reset_index(drop=True)


def _fit_bundle() -> dict[str, Any]:
    phase13 = _load_script(PHASE13_PATH, "strikecast_phase13_shadow")
    phase14 = _load_script(PHASE14_PATH, "strikecast_phase14_shadow")
    completed = phase13.load_game_frame()
    # The rolling L20 columns in Phase 13 are evaluation baselines, not model
    # features.  Skip their date-by-date construction in the live bundle.
    features = _component_features(completed, phase13)
    team_long = _inference_team_rows(completed, phase13, features)

    component_models: dict[str, Any] = {}
    for target in phase13.TARGETS:
        model = phase13.component_model()
        model.fit(team_long[features], team_long[target].astype(float))
        component_models[target] = model

    oos_games = phase14.load_predictions()
    oos_long = phase14.to_team_long(oos_games)
    reconciler = phase14.run_reconciler()
    reconciler.fit(oos_long[phase14.RUN_FEATURES], oos_long["actual_runs"].astype(float))
    dispersion = phase14.estimate_dispersion(oos_long["actual_runs"], oos_long["pred_runs"])

    linked_history = pd.read_csv(PHASE14_HISTORY_PATH)
    linked_history["log_run_ratio"] = np.log(linked_history["selected_mu_h"].clip(lower=0.05)) - np.log(
        linked_history["selected_mu_a"].clip(lower=0.05)
    )
    linked_history["expected_total_runs"] = linked_history["selected_mu_h"] + linked_history["selected_mu_a"]
    first_score_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(C=0.5, max_iter=1000, random_state=phase14.RANDOM_SEED)),
    ])
    first_score_model.fit(
        linked_history[["log_run_ratio", "expected_total_runs"]],
        linked_history["first_score_home"].astype(int),
    )

    total_over_calibrator, total_calibration_games = _fit_total_over_calibrator(linked_history)

    return {
        "phase13": phase13,
        "phase14": phase14,
        "features": features,
        "component_models": component_models,
        "reconciler": reconciler,
        "dispersion": float(dispersion),
        "first_score_model": first_score_model,
        "total_over_calibrator": total_over_calibrator,
        "total_calibration_games": total_calibration_games,
        "component_training_games": int(len(completed)),
        "oos_training_games": int(len(oos_games)),
        "coverage_end": str(pd.to_datetime(linked_history["game_date"]).max().date()),
        "engine_id": _engine_id(),
        "lineage": _lineage_key(),
    }


def _fit_total_over_calibrator(linked_history: pd.DataFrame) -> tuple[IsotonicRegression | None, int]:
    """Fit isotonic recalibration for the sim's P(total > line)."""
    market = pd.read_parquet(
        REPO_DIR / "data" / "processed" / "train.parquet",
        columns=["game_pk", "market_over_under"],
    )
    frame = linked_history.merge(market, on="game_pk", how="inner")
    frame = frame.dropna(subset=[
        "selected_mu_h", "selected_mu_a", "dispersion_alpha",
        "runs_h", "runs_a", "market_over_under",
    ])
    if frame.empty:
        return None, 0
    frame = frame.copy()
    frame["actual_total"] = frame["runs_h"].astype(float) + frame["runs_a"].astype(float)
    graded = frame[frame["actual_total"] != frame["market_over_under"]].copy()
    if len(graded) < 200:
        return None, int(len(graded))
    graded["sim_p_over_raw"] = [
        total_probability_over(mh, ma, al, ln)
        for mh, ma, al, ln in zip(
            graded["selected_mu_h"], graded["selected_mu_a"],
            graded["dispersion_alpha"], graded["market_over_under"],
        )
    ]
    graded["actual_over"] = (graded["actual_total"] > graded["market_over_under"]).astype(int)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(graded["sim_p_over_raw"].values, graded["actual_over"].values)
    return calibrator, int(len(graded))


def get_bundle() -> dict[str, Any]:
    key = _lineage_key()
    with _bundle_lock:
        if _bundle_cache.get("lineage") != key:
            _bundle_cache.clear()
            restored: dict[str, Any] | None = None
            if BUNDLE_PATH.exists():
                try:
                    candidate = joblib.load(BUNDLE_PATH)
                    if candidate.get("lineage") == key:
                        restored = candidate
                except Exception:
                    restored = None
            if restored is None:
                fresh = _fit_bundle()
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                persistable = {
                    name: value for name, value in fresh.items()
                    if name not in {"phase13", "phase14"}
                }
                temporary = BUNDLE_PATH.with_suffix(".tmp.joblib")
                joblib.dump(persistable, temporary)
                temporary.replace(BUNDLE_PATH)
                _bundle_cache.update(fresh)
            else:
                restored["phase13"] = _load_script(PHASE13_PATH, "strikecast_phase13_shadow")
                restored["phase14"] = _load_script(PHASE14_PATH, "strikecast_phase14_shadow")
                _bundle_cache.update(restored)
    return _bundle_cache


def _safe_float(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if np.isfinite(number) else None


def _snapshot_quality(first_pitch: Any, captured_at: datetime) -> str:
    timestamp = pd.to_datetime(first_pitch, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "unknown_time"
    return "pregame" if captured_at <= timestamp.to_pydatetime() else "late_capture"


def _is_final_status(value: Any) -> bool:
    status = str(value or "").strip().lower()
    return status in {"final", "game over"} or status.startswith("completed early")


def _actual_components_frame() -> pd.DataFrame:
    """Return one row per game with official component results by side."""
    columns = [
        "game_pk", "side", "bat_hits", "bat_baseOnBalls", "bat_hitByPitch",
        "bat_totalBases", "bat_leftOnBase",
    ]
    box = pd.read_parquet(TEAM_BOX_PATH, columns=columns)
    box["side"] = box["side"].astype(str).str.lower().str[0]
    box["baserunners"] = (
        pd.to_numeric(box["bat_hits"], errors="coerce")
        + pd.to_numeric(box["bat_baseOnBalls"], errors="coerce")
        + pd.to_numeric(box["bat_hitByPitch"], errors="coerce")
    )
    sources = {
        "hits": "bat_hits",
        "baserunners": "baserunners",
        "total_bases": "bat_totalBases",
        "left_on_base": "bat_leftOnBase",
    }
    pieces = []
    for metric, source in sources.items():
        wide = box.pivot_table(index="game_pk", columns="side", values=source, aggfunc="first")
        for side in ("a", "h"):
            if side not in wide:
                wide[side] = np.nan
        pieces.append(wide[["a", "h"]].rename(columns={
            "a": f"actual_away_{metric}", "h": f"actual_home_{metric}",
        }))
    actual = pd.concat(pieces, axis=1).reset_index()
    if GAME_SCRIPT_PATH.exists():
        first = pd.read_csv(GAME_SCRIPT_PATH, usecols=["game_pk", "first_score_side"])
        first["actual_first_score_home"] = first["first_score_side"].eq("home")
        actual = actual.merge(
            first[["game_pk", "actual_first_score_home"]].drop_duplicates("game_pk"),
            on="game_pk", how="left", validate="one_to_one",
        )
    return actual


def _component_correct(predicted: Any, actual: Any, metric: str) -> bool | None:
    p = _safe_float(predicted)
    a = _safe_float(actual)
    if p is None or a is None:
        return None
    return abs(p - a) <= COMPONENT_TOLERANCES[metric]


def _grade_component_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        for side in ("away", "home"):
            nested = row.get(side) if isinstance(row.get(side), dict) else {}
            for metric in COMPONENT_TOLERANCES:
                predicted = nested.get(metric, row.get(f"{side}_{metric}"))
                row[f"{side}_{metric}_correct"] = _component_correct(
                    predicted, row.get(f"actual_{side}_{metric}"), metric
                )
        actual_first = row.get("actual_first_score_pick")
        predicted_first = row.get("first_score_pick")
        row["first_score_correct"] = (
            predicted_first == actual_first
            if predicted_first is not None and actual_first is not None
            and not pd.isna(predicted_first) and not pd.isna(actual_first)
            else None
        )
    return rows


def simulate_slate(
    runtime_train: pd.DataFrame,
    games_frame: pd.DataFrame,
    official_games: list[dict[str, Any]],
    game_date: date,
) -> list[dict[str, Any]]:
    """Generate linked shadow simulations for one date from pregame features."""
    bundle = get_bundle()
    frame = runtime_train.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    slate = frame[frame["game_date"].dt.date.eq(game_date)].copy()
    if slate.empty:
        return []
    long = _inference_team_rows(slate, bundle["phase13"], bundle["features"])
    for target, model in bundle["component_models"].items():
        long[f"pred_{target}"] = np.clip(model.predict(long[bundle["features"]]), 0.0, 30.0)
    long["pred_conversion"] = long["pred_runs"] / long["pred_baserunners"].clip(lower=1.0)
    long["selected_mu"] = np.clip(
        bundle["reconciler"].predict(long[bundle["phase14"].RUN_FEATURES]), 0.05, 15.0
    )
    wide_columns = [f"pred_{target}" for target in bundle["phase13"].TARGETS] + ["selected_mu"]
    wide = long.pivot(index="game_pk", columns="side", values=wide_columns)
    wide.columns = [f"{metric}_{side}" for metric, side in wide.columns]
    current = slate.merge(wide.reset_index(), on="game_pk", how="left", validate="one_to_one")
    current = current.merge(
        _actual_components_frame(), on="game_pk", how="left", validate="one_to_one"
    )
    current["sim_p_home"] = [
        bundle["phase14"].score_win_probability(float(mu_h), float(mu_a), bundle["dispersion"])
        for mu_h, mu_a in zip(current["selected_mu_h"], current["selected_mu_a"])
    ]
    current["log_run_ratio"] = np.log(current["selected_mu_h"].clip(lower=0.05)) - np.log(
        current["selected_mu_a"].clip(lower=0.05)
    )
    current["expected_total_runs"] = current["selected_mu_h"] + current["selected_mu_a"]
    current["first_score_p_home"] = bundle["first_score_model"].predict_proba(
        current[["log_run_ratio", "expected_total_runs"]]
    )[:, 1]

    alpha_global = float(bundle["dispersion"])
    calibrator = bundle.get("total_over_calibrator")
    # Pregame column falls back to total_open when total_close is missing (ESPN
    # only populates close after games play). Use it if present; otherwise the
    # canonical market_over_under (historical close-only).
    if "market_over_under_pregame" in current.columns:
        current["_ou_line"] = current["market_over_under_pregame"].fillna(
            current.get("market_over_under")
        )
    elif "market_over_under" in current.columns:
        current["_ou_line"] = current["market_over_under"]
    else:
        current["_ou_line"] = np.nan
    raw_over: list[float | None] = []
    cal_over: list[float | None] = []
    for mu_h, mu_a, line in zip(
        current["selected_mu_h"], current["selected_mu_a"], current["_ou_line"]
    ):
        if pd.isna(line):
            raw_over.append(None)
            cal_over.append(None)
            continue
        raw = total_probability_over(float(mu_h), float(mu_a), alpha_global, float(line))
        raw_over.append(raw)
        cal_over.append(
            float(calibrator.predict([raw])[0]) if calibrator is not None else raw
        )
    current["sim_p_over_raw"] = raw_over
    current["sim_p_over_cal"] = cal_over

    official = {int(game["game_pk"]): game for game in official_games}
    game_meta = games_frame.set_index("game_pk", drop=False) if not games_frame.empty else pd.DataFrame()
    captured_at = datetime.now(timezone.utc)
    output: list[dict[str, Any]] = []
    for _, row in current.sort_values("game_pk").iterrows():
        pk = int(row["game_pk"])
        off = official.get(pk, {})
        meta = game_meta.loc[pk] if not game_meta.empty and pk in game_meta.index else None
        first_pitch = off.get("first_pitch_utc") or (meta.get("first_pitch_utc") if meta is not None else None)
        home = str(row["home_team_abbrev"])
        away = str(row["away_team_abbrev"])
        p_home = float(row["sim_p_home"])
        pick = home if p_home >= 0.5 else away
        status = off.get("game_state", {}).get("detailed") or (meta.get("status") if meta is not None else None)
        status_final = _is_final_status(status)
        is_final = bool(off.get("is_played", False)) or status_final
        away_score = off.get("away_score")
        home_score = off.get("home_score")
        if away_score is None and meta is not None:
            away_score = meta.get("away_score")
        if home_score is None and meta is not None:
            home_score = meta.get("home_score")
        has_final_score = (
            is_final and away_score is not None and home_score is not None
            and not pd.isna(away_score) and not pd.isna(home_score)
            and float(away_score) != float(home_score)
        )
        actual_winner = home if has_final_score and float(home_score) > float(away_score) else (
            away if has_final_score else None
        )
        simulation_correct = pick == actual_winner if actual_winner else None
        official_pick = off.get("pick_abbrev")
        official_correct = official_pick == actual_winner if official_pick and actual_winner else None
        record = {
            "game_pk": pk,
            "date": game_date.isoformat(),
            "away_abbrev": away,
            "home_abbrev": home,
            "first_pitch_utc": str(first_pitch) if first_pitch is not None and not pd.isna(first_pitch) else None,
            "status": status,
            "is_live": bool(off.get("is_live", False)),
            "is_final": is_final,
            "capture_quality": _snapshot_quality(first_pitch, captured_at),
            "captured_at": captured_at.isoformat(),
            "engine_id": bundle["engine_id"],
            "simulator_version": SERVICE_VERSION,
            "simulation_pick": pick,
            "official_pick": official_pick,
            "agreement": pick == official_pick if official_pick else None,
            "sim_p_home": _safe_float(p_home),
            "official_p_home": _safe_float(off.get("p_home")),
            "first_score_pick": home if float(row["first_score_p_home"]) >= 0.5 else away,
            "first_score_p_home": _safe_float(row["first_score_p_home"]),
            "expected_total_runs": _safe_float(row["expected_total_runs"], 2),
            "market_over_under": _safe_float(row.get("_ou_line")),
            "sim_p_over_raw": _safe_float(row.get("sim_p_over_raw")),
            "sim_p_over_cal": _safe_float(row.get("sim_p_over_cal")),
            "away": {
                "hits": _safe_float(row["pred_hits_a"], 2),
                "baserunners": _safe_float(row["pred_baserunners_a"], 2),
                "total_bases": _safe_float(row["pred_total_bases_a"], 2),
                "left_on_base": _safe_float(row["pred_left_on_base_a"], 2),
                "runs": _safe_float(row["selected_mu_a"], 2),
            },
            "home": {
                "hits": _safe_float(row["pred_hits_h"], 2),
                "baserunners": _safe_float(row["pred_baserunners_h"], 2),
                "total_bases": _safe_float(row["pred_total_bases_h"], 2),
                "left_on_base": _safe_float(row["pred_left_on_base_h"], 2),
                "runs": _safe_float(row["selected_mu_h"], 2),
            },
            "result": "correct" if simulation_correct is True else (
                "incorrect" if simulation_correct is False else "pending"
            ),
            "simulation_correct": simulation_correct,
            "official_correct": official_correct,
            "actual_winner": actual_winner,
            "actual_away_score": _safe_float(away_score, 0) if has_final_score else None,
            "actual_home_score": _safe_float(home_score, 0) if has_final_score else None,
            "evidence": "prospective_shadow",
        }
        for side in ("away", "home"):
            for metric in COMPONENT_TOLERANCES:
                value = row.get(f"actual_{side}_{metric}")
                record[f"actual_{side}_{metric}"] = (
                    _safe_float(value, 0) if is_final else None
                )
        actual_first_home = row.get("actual_first_score_home")
        if is_final and actual_first_home is not None and not pd.isna(actual_first_home):
            record["actual_first_score_pick"] = home if bool(actual_first_home) else away
        else:
            record["actual_first_score_pick"] = None

        line = row.get("_ou_line")
        p_over_cal = record.get("sim_p_over_cal")
        if has_final_score and line is not None and not pd.isna(line):
            actual_total = float(home_score) + float(away_score)
            if actual_total == float(line):
                totals_result = "push"
                actual_over = None
                over_correct = None
            else:
                actual_over = actual_total > float(line)
                totals_result = "over" if actual_over else "under"
                if p_over_cal is not None:
                    over_correct = (p_over_cal > 0.5) == actual_over
                else:
                    over_correct = None
            record["actual_total_runs"] = _safe_float(actual_total, 0)
            record["actual_totals_result"] = totals_result
            record["totals_correct"] = over_correct
        else:
            record["actual_total_runs"] = None
            record["actual_totals_result"] = None
            record["totals_correct"] = None
        output.append(record)
    return output


def _flatten_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key not in {"away", "home"}}
    for side in ["away", "home"]:
        for metric, value in row[side].items():
            flat[f"{side}_{metric}"] = value
    return flat


def save_snapshots(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame([_flatten_snapshot(row) for row in rows])
    with _snapshot_lock:
        existing = pd.read_parquet(SNAPSHOT_PATH) if SNAPSHOT_PATH.exists() else pd.DataFrame()
        keys = ["game_pk", "engine_id", "simulator_version"]
        if existing.empty:
            combined = incoming
        else:
            combined = existing.copy()
            enrichment = ["official_pick", "official_p_home", "agreement", "status"]
            enrichment += [
                f"actual_{side}_{metric}"
                for side in ("away", "home") for metric in COMPONENT_TOLERANCES
            ]
            enrichment += ["actual_first_score_pick"]
            for _, fresh in incoming.iterrows():
                mask = pd.Series(True, index=combined.index)
                for key in keys:
                    mask &= combined[key].eq(fresh[key])
                if not mask.any():
                    combined = pd.concat([combined, fresh.to_frame().T], ignore_index=True)
                    continue
                position = combined.index[mask][0]
                for column in enrichment:
                    old = combined.at[position, column] if column in combined else None
                    new = fresh.get(column)
                    if (old is None or pd.isna(old)) and new is not None and not pd.isna(new):
                        combined.at[position, column] = new
                if bool(fresh.get("is_final", False)):
                    for column in [
                        "is_final", "actual_winner", "actual_away_score", "actual_home_score",
                        "simulation_correct", "official_correct", "result",
                    ]:
                        new = fresh.get(column)
                        if new is not None and not pd.isna(new):
                            combined.at[position, column] = new
        combined = combined.sort_values("captured_at").drop_duplicates(keys, keep="first")
        combined.to_parquet(SNAPSHOT_PATH, index=False)


def hydrate_saved_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reuse immutable capture metadata so page reloads never need live-feed I/O."""
    if not rows or not SNAPSHOT_PATH.exists():
        return rows
    saved = pd.read_parquet(SNAPSHOT_PATH).sort_values("captured_at").drop_duplicates(
        ["game_pk", "engine_id", "simulator_version"], keep="first"
    ).set_index("game_pk")
    for row in rows:
        pk = int(row["game_pk"])
        if pk not in saved.index:
            continue
        old = saved.loc[pk]
        if str(old.get("engine_id")) != str(row.get("engine_id")):
            continue
        outcome_columns = {
            "is_final", "actual_winner", "actual_away_score", "actual_home_score",
            "simulation_correct", "official_correct", "result",
        }
        current_is_graded = row.get("result") in {"correct", "incorrect"}
        for column in [
            "official_pick", "official_p_home", "agreement", "status", "is_final",
            "actual_winner", "actual_away_score", "actual_home_score",
            "simulation_correct", "official_correct", "result",
        ]:
            if current_is_graded and column in outcome_columns:
                continue
            value = old.get(column)
            if value is not None and not pd.isna(value):
                if column in {"agreement", "is_final", "simulation_correct", "official_correct"}:
                    row[column] = bool(value)
                elif column in {"official_p_home", "actual_away_score", "actual_home_score"}:
                    row[column] = _safe_float(value, 4 if column == "official_p_home" else 0)
                else:
                    row[column] = str(value)
        if row.get("actual_winner") and row.get("simulation_correct") is not None:
            row["result"] = "correct" if bool(row["simulation_correct"]) else "incorrect"
        if row.get("actual_winner") and row.get("official_pick"):
            row["official_correct"] = row["official_pick"] == row["actual_winner"]
        row["capture_quality"] = old.get("capture_quality") or row["capture_quality"]
        row["captured_at"] = old.get("captured_at") or row["captured_at"]
    return rows


def _historical_rows() -> pd.DataFrame:
    history = pd.read_csv(PHASE14_HISTORY_PATH)
    train = pd.read_parquet(
        REPO_DIR / "data" / "processed" / "train.parquet",
        columns=["game_pk", "game_date", "away_team_abbrev", "home_team_abbrev"],
    ).drop_duplicates("game_pk")
    history = history.merge(
        train[["game_pk", "away_team_abbrev", "home_team_abbrev"]],
        on="game_pk", how="left", validate="one_to_one",
    )
    history["date"] = pd.to_datetime(history["game_date"]).dt.strftime("%Y-%m-%d")
    history["simulation_pick"] = np.where(
        history["sim_p_home"].ge(0.5), history["home_team_abbrev"], history["away_team_abbrev"]
    )
    history["official_pick"] = np.where(
        history["production_pick_home"].astype(bool), history["home_team_abbrev"], history["away_team_abbrev"]
    )
    history["first_score_pick"] = np.where(
        history["linked_first_score_home"].ge(0.5),
        history["home_team_abbrev"], history["away_team_abbrev"],
    )
    history["actual_winner"] = np.where(
        history["actual_home_win"].astype(bool), history["home_team_abbrev"], history["away_team_abbrev"]
    )
    history["simulation_correct"] = history["simulation_pick"].eq(history["actual_winner"])
    history["official_correct"] = history["official_pick"].eq(history["actual_winner"])
    history["result"] = np.where(history["simulation_correct"], "correct", "incorrect")
    history["agreement"] = history["simulation_pick"].eq(history["official_pick"])
    history["capture_quality"] = "chronological_oos"
    history["evidence"] = "chronological_oos"
    history["engine_id"] = _engine_id()
    history["simulator_version"] = SERVICE_VERSION
    return history


def _snapshot_rows() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame()
    data = pd.read_parquet(SNAPSHOT_PATH)
    games = pd.read_parquet(
        REPO_DIR / "data" / "processed" / "games.parquet",
        columns=["game_pk", "status", "away_score", "home_score"],
    ).drop_duplicates("game_pk")
    data = data.drop(columns=[c for c in ["status", "actual_away_score", "actual_home_score"] if c in data], errors="ignore")
    data = data.merge(games, on="game_pk", how="left")
    final = data["status"].map(_is_final_status)
    has_score = data["away_score"].notna() & data["home_score"].notna() & data["away_score"].ne(data["home_score"])
    graded = final & has_score
    actual_home = data["home_score"].gt(data["away_score"])
    data["actual_winner"] = np.where(actual_home, data["home_abbrev"], data["away_abbrev"])
    data.loc[~graded, "actual_winner"] = None
    data["simulation_correct"] = data["simulation_pick"].eq(data["actual_winner"]).where(graded)
    data["official_correct"] = data["official_pick"].eq(data["actual_winner"]).where(graded)
    data["result"] = np.where(~graded, "pending", np.where(data["simulation_correct"], "correct", "incorrect"))
    data["actual_away_score"] = data["away_score"].where(graded)
    data["actual_home_score"] = data["home_score"].where(graded)
    actual = _actual_components_frame()
    actual_columns = [column for column in actual.columns if column != "game_pk"]
    data = data.drop(columns=[column for column in actual_columns if column in data], errors="ignore")
    data = data.merge(actual, on="game_pk", how="left", validate="many_to_one")
    for column in actual_columns:
        if column in data:
            data.loc[~graded, column] = None
    if "actual_first_score_home" in data:
        data["actual_first_score_pick"] = np.where(
            data["actual_first_score_home"].astype("boolean").fillna(False),
            data["home_abbrev"], data["away_abbrev"],
        )
        data.loc[~graded | data["actual_first_score_home"].isna(), "actual_first_score_pick"] = None
    return data


def history_payload(
    start: date | None = None,
    end: date | None = None,
    result: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    historical = _historical_rows()
    columns = {
        "away_team_abbrev": "away_abbrev", "home_team_abbrev": "home_abbrev",
        "p_home": "official_p_home", "linked_first_score_home": "first_score_p_home",
        "selected_mu_a": "away_runs", "selected_mu_h": "home_runs",
        "pred_hits_a": "away_hits", "pred_hits_h": "home_hits",
        "pred_baserunners_a": "away_baserunners", "pred_baserunners_h": "home_baserunners",
        "pred_total_bases_a": "away_total_bases", "pred_total_bases_h": "home_total_bases",
        "pred_left_on_base_a": "away_left_on_base", "pred_left_on_base_h": "home_left_on_base",
        "runs_a": "actual_away_score", "runs_h": "actual_home_score",
        "hits_a": "actual_away_hits", "hits_h": "actual_home_hits",
        "baserunners_a": "actual_away_baserunners", "baserunners_h": "actual_home_baserunners",
        "total_bases_a": "actual_away_total_bases", "total_bases_h": "actual_home_total_bases",
        "left_on_base_a": "actual_away_left_on_base", "left_on_base_h": "actual_home_left_on_base",
        "first_score_home": "actual_first_score_home",
    }
    historical = historical.rename(columns=columns)
    keep = [
        "game_pk", "date", "away_abbrev", "home_abbrev", "simulation_pick", "official_pick",
        "actual_winner", "sim_p_home", "official_p_home", "first_score_p_home",
        "first_score_pick", "expected_total_runs", "status",
        "away_runs", "home_runs", "away_hits", "home_hits", "away_baserunners", "home_baserunners",
        "away_total_bases", "home_total_bases", "away_left_on_base", "home_left_on_base",
        "actual_away_hits", "actual_home_hits", "actual_away_baserunners", "actual_home_baserunners",
        "actual_away_total_bases", "actual_home_total_bases",
        "actual_away_left_on_base", "actual_home_left_on_base",
        "actual_first_score_home", "actual_first_score_pick", "first_score_correct",
        "away_hits_correct", "home_hits_correct", "away_baserunners_correct", "home_baserunners_correct",
        "away_total_bases_correct", "home_total_bases_correct",
        "away_left_on_base_correct", "home_left_on_base_correct",
        "actual_away_score", "actual_home_score", "simulation_correct", "official_correct", "result",
        "agreement", "capture_quality", "evidence", "engine_id", "simulator_version",
    ]
    frames = [historical[[name for name in keep if name in historical.columns]]]
    snapshots = _snapshot_rows()
    if not snapshots.empty:
        frames.append(snapshots[[name for name in keep if name in snapshots.columns]])
    data = pd.concat(frames, ignore_index=True, sort=False)
    if "actual_first_score_home" in data:
        first_known = data["actual_first_score_home"].notna()
        data.loc[first_known, "actual_first_score_pick"] = np.where(
            data.loc[first_known, "actual_first_score_home"].astype(bool),
            data.loc[first_known, "home_abbrev"], data.loc[first_known, "away_abbrev"],
        )
    records = _grade_component_rows(data.to_dict(orient="records"))
    data = pd.DataFrame(records)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values(["date", "game_pk"], ascending=[False, True])
    data = data.drop_duplicates(["game_pk", "simulator_version"], keep="last")
    if start:
        data = data[data["date"].dt.date.ge(start)]
    if end:
        data = data[data["date"].dt.date.le(end)]
    if result and result != "all":
        data = data[data["result"].eq(result)]
    settled = data[data["result"].isin(["correct", "incorrect"])]
    prospective = settled[settled["capture_quality"].eq("pregame")]
    oos = settled[settled["capture_quality"].eq("chronological_oos")]
    summary = {
        "games": int(len(data)),
        "settled": int(len(settled)),
        "correct": int(settled["simulation_correct"].fillna(False).sum()),
        "accuracy": _safe_float(settled["simulation_correct"].mean()) if len(settled) else None,
        "oos_games": int(len(oos)),
        "oos_accuracy": _safe_float(oos["simulation_correct"].mean()) if len(oos) else None,
        "prospective_games": int(len(prospective)),
        "prospective_accuracy": _safe_float(prospective["simulation_correct"].mean()) if len(prospective) else None,
        "late_captures_excluded": int((
            ~settled["capture_quality"].isin({"chronological_oos", "pregame"})
        ).sum()),
    }
    page = data.head(max(1, min(int(limit), 1000))).copy()
    page["date"] = page["date"].dt.strftime("%Y-%m-%d")
    page = page.replace({np.nan: None})
    return {"summary": summary, "games": page.to_dict(orient="records")}


def daily_payload(rows: list[dict[str, Any]], game_date: date) -> dict[str, Any]:
    rows = _grade_component_rows(rows)
    graded = [row for row in rows if row.get("result") in {"correct", "incorrect"}]
    valid_graded = [
        row for row in graded
        if row.get("capture_quality") in {"chronological_oos", "pregame"}
    ]
    agreement = [row for row in rows if row.get("agreement") is not None]
    return {
        "date": game_date.isoformat(),
        "mode": "shadow_only",
        "production_changed": False,
        "engine_id": _engine_id(),
        "simulator_version": SERVICE_VERSION,
        "games": rows,
        "summary": {
            "games": len(rows),
            "agreement_rate": _safe_float(np.mean([row["agreement"] for row in agreement])) if agreement else None,
            "graded": len(graded),
            "accuracy": _safe_float(np.mean([row["simulation_correct"] for row in valid_graded])) if valid_graded else None,
            "late_captures_excluded": len(graded) - len(valid_graded),
        },
        "governance": {
            "historical": "chronological_oos",
            "prospective": "only pregame snapshots count toward prospective accuracy",
            "late_capture": "visible but excluded from prospective accuracy",
            "winner_path": "official picks are unchanged",
        },
    }
