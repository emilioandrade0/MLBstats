"""STRIKECAST · Actualizacion incremental de datos (paralela).

Reemplazo directo del viejo update_data.bat. Corre los mismos 8 pasos pero
paralelizando lo que se puede: ingest (ESPN/StatsAPI/Savant), features y
comeback+burn se ejecutan en grupos concurrentes.

Uso:
    python update_data.py               # ventana de 14 dias hasta hoy
    python update_data.py --days 30     # ventana de 30 dias
    python update_data.py --workers 6   # mas paralelismo

Logs por modulo se guardan en ./logs/<step_name>.log — si algo falla el
driver aborta la fase y te dice cual modulo revisar.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
PYTHON = sys.executable  # usa el mismo interprete que lanzo el script


# ─── Runner por subproceso ─────────────────────────────────────────────────────
def run_module(name: str, args: list[str]) -> tuple[str, int, float]:
    """Ejecuta `python -m <module> [args]`, guarda log, devuelve (name, rc, secs)."""
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / f"{name.replace('.', '_')}.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"$ {PYTHON} -m {name} {' '.join(args)}\n")
        f.flush()
        rc = subprocess.call(
            [PYTHON, "-m", name, *args],
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return name, rc, time.time() - t0


def _progress_bar(done: int, total: int, width: int = 24) -> str:
    filled = int(round(width * done / total)) if total else 0
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _short(name: str) -> str:
    # "src.features.market_close" -> "market_close"
    return name.rsplit(".", 1)[-1]


def _log_path(name: str) -> Path:
    return LOGS / f"{name.replace('.', '_')}.log"


def _tail_last_line(path: Path, max_bytes: int = 4096) -> str:
    """Lee las ultimas ~4KB del log y devuelve la ultima "linea" real.

    Los progress bars de tqdm usan \\r para sobrescribir la misma linea sin \\n,
    asi que hay que partir tambien por \\r para agarrar el ultimo frame.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)  # end
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            chunk = f.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    # Separar por saltos de linea Y por \r (tqdm)
    parts = [p.strip() for p in chunk.replace("\r", "\n").split("\n")]
    parts = [p for p in parts if p]
    return parts[-1] if parts else ""


def _clean_progress_line(s: str, max_len: int) -> str:
    # Quitar caracteres de control ANSI y colapsar espacios
    import re
    s = re.sub(r"\x1b\[[0-9;]*m", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


def run_group(label: str, tasks: list[tuple[str, list[str]]], workers: int) -> None:
    """Corre `tasks` en paralelo (workers acotados) — aborta si alguno falla.

    Muestra dos lineas en vivo (\\r):
      1) barra de fase (done/total, elapsed, modulos corriendo)
      2) actividad del log mas reciente (tqdm interno de cada modulo)
    """
    total = len(tasks)
    parallel = min(workers, total)
    print(f"\n── {label} ── ({total} tarea(s), {parallel} en paralelo)")

    running: set[str] = set()
    done = 0
    errors: list[str] = []
    results: list[tuple[str, int, float]] = []
    t0 = time.time()
    lines_drawn = 0  # cuantas lineas hay que "limpiar" hacia arriba en el proximo redraw

    def draw() -> None:
        nonlocal lines_drawn
        elapsed = time.time() - t0
        bar = _progress_bar(done, total)
        run_list = ", ".join(sorted(_short(n) for n in running)) or "—"
        if len(run_list) > 55:
            run_list = run_list[:52] + "..."
        line1 = f"  {bar} {done}/{total} · {elapsed:5.1f}s · corriendo: {run_list}"

        # Linea 2: el log mas activo (el modificado mas recientemente entre los running)
        active_tail = ""
        active_name = ""
        latest_mtime = -1.0
        for n in list(running):
            p = _log_path(n)
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m > latest_mtime:
                latest_mtime = m
                active_name = _short(n)
                active_tail = _tail_last_line(p)
        active_tail = _clean_progress_line(active_tail, 100)
        if active_name:
            line2 = f"    ↳ {active_name}: {active_tail}"
        elif done == total:
            line2 = "    ↳ (fase completa)"
        else:
            line2 = "    ↳ (esperando output...)"

        # Subir cursor hacia arriba tantas lineas como se hayan dibujado antes,
        # luego reescribir ambas lineas.
        if lines_drawn:
            sys.stdout.write("\x1b[{}A".format(lines_drawn))
        sys.stdout.write("\r" + line1.ljust(120) + "\n")
        sys.stdout.write("\r" + line2.ljust(120) + "\n")
        sys.stdout.flush()
        lines_drawn = 2

    stop_flag = threading.Event()
    lock = threading.Lock()

    def heartbeat() -> None:
        while not stop_flag.wait(0.5):
            with lock:
                draw()

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    with ProcessPoolExecutor(max_workers=parallel) as ex:
        fut_to_name = {}
        for n, a in tasks:
            fut_to_name[ex.submit(run_module, n, a)] = n
            running.add(n)
        with lock:
            draw()
        for fut in as_completed(fut_to_name):
            name, rc, secs = fut.result()
            with lock:
                running.discard(name)
                done += 1
                results.append((name, rc, secs))
                if rc != 0:
                    errors.append(name)
                draw()

    stop_flag.set()
    hb.join(timeout=1)
    # Detalle final por modulo
    for name, rc, secs in sorted(results, key=lambda x: x[0]):
        tag = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"  [{tag:>7}] {_short(name):<22} {secs:6.1f}s   log: logs/{name.replace('.','_')}.log")

    if errors:
        print(f"\nERROR: falló(n) {', '.join(_short(n) for n in errors)}. Revisa los logs.")
        sys.exit(1)


def savant_start_date(fallback: str) -> str:
    """Ultima fecha en pitches.parquet + 1 dia; fallback si no existe."""
    p = ROOT / "data" / "processed" / "pitches.parquet"
    if not p.exists():
        return fallback
    try:
        import pandas as pd
        last = pd.to_datetime(pd.read_parquet(p, columns=["game_date"])["game_date"]).max()
        return (last + pd.Timedelta(days=1)).date().isoformat()
    except Exception:
        return fallback


def check_train_locked() -> None:
    """Aborta antes de arrancar si train.parquet esta bloqueado (server abierto)."""
    p = ROOT / "data" / "processed" / "train.parquet"
    if not p.exists():
        return
    try:
        with open(p, "r+b"):
            pass
    except OSError as e:
        print(
            "ERROR: train.parquet esta bloqueado por otra app o proceso.\n"
            "Cierra STRIKECAST / uvicorn / python y vuelve a correr.\n"
            f"({e})"
        )
        sys.exit(1)


# ─── Main ──────────────────────────────────────────────────────────────────────
def _enable_windows_ansi() -> None:
    """Habilita procesamiento de secuencias ANSI en CMD (cursor up/etc)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11, ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def main() -> None:
    # Forzar utf-8 en stdout (Windows CMD por defecto usa cp1252 → bloques rotos)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _enable_windows_ansi()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="ventana de re-descarga (default 14)")
    ap.add_argument("--workers", type=int, default=8, help="procesos paralelos por fase (default 8)")
    args = ap.parse_args()

    today = date.today().isoformat()
    since = (date.today() - timedelta(days=args.days)).isoformat()
    W = args.workers

    print("=" * 60)
    print("  STRIKECAST | Actualizacion incremental (paralela)")
    print("=" * 60)
    print(f"  Hoy     : {today}")
    print(f"  Ventana : {since} -> {today}")
    print(f"  Workers : {W}")
    print("=" * 60)

    check_train_locked()
    t_start = time.time()

    # ── FASE 1: descargas de APIs (todas independientes) ──────────────────────
    savant_since = savant_start_date(since)
    fase1 = [
        ("src.ingest",           ["--start", since, "--end", today, "--force"]),
        ("src.ingest_statsapi",  ["--start", since, "--end", today, "--force"]),
    ]
    if savant_since <= today:
        fase1.append(("src.ingest_savant", ["--start", savant_since, "--end", today]))
    else:
        print(f"  (Savant al dia — ultimo pitch >= {today})")
    run_group("FASE 1/6 · Ingest (ESPN + StatsAPI + Savant)", fase1, W)

    # ── FASE 2: normalize (necesita FASE 1) ───────────────────────────────────
    run_group(
        "FASE 2/6 · Normalize (games/team_box/player_box/plays/odds/wp/xref/pitches)",
        [("src.normalize.build_all",
          ["games", "team_box", "player_box", "plays", "odds", "win_probability", "xref", "pitches"])],
        W,
    )

    # ── FASE 3: odds_close (necesita normalize) ───────────────────────────────
    run_group("FASE 3/6 · Odds close (odds_v2)", [("src.normalize.odds_v2", [])], W)

    # ── FASE 4a: features base (todas independientes) ─────────────────────────
    features_base = [
        ("src.features.market_close",    []),
        ("src.features.team_form",       []),
        ("src.features.pitcher_form",    []),
        ("src.features.pitcher_season",  []),
        ("src.features.lineup",          []),
        ("src.features.park",            []),
        ("src.features.elo",             []),
        ("src.features.bullpen_load",    []),
        ("src.features.series_context",  []),
        ("src.features.umpires",         []),
        ("src.features.f5_targets",      []),
        ("src.features.pythagorean",     []),
        ("src.features.weather_wind",    []),
    ]
    run_group("FASE 4a/6 · Features base (13 modulos)", features_base, W)

    # ── FASE 4b: features derivadas (dependen de features base ya escritas) ──
    # lineup_recent lee features_lineup.parquet; situational lee features_market
    features_derived = [
        ("src.features.lineup_recent",   []),
        ("src.features.situational",     []),
    ]
    run_group("FASE 4b/6 · Features derivadas (2 modulos)", features_derived, W)

    # ── FASE 5a: comeback_analysis + game_flow + burn_analysis (independientes) ─
    run_group(
        "FASE 5/6 · Analisis (comeback / game_flow / burn)",
        [
            ("src.analysis.comeback_analysis", []),
            ("src.features.game_flow",         []),
            ("src.analysis.burn_analysis",     []),
        ],
        W,
    )
    # ── FASE 5b: features.comeback (necesita comeback_analysis) ───────────────
    run_group("FASE 5/6 · Features comeback", [("src.features.comeback", [])], W)

    # ── FASE 6: train.parquet (necesita TODO) ────────────────────────────────
    run_group("FASE 6/6 · Ensamblar train.parquet", [("src.features.build", [])], W)

    total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  Terminado en {total:.1f}s ({total/60:.1f} min)")
    print("=" * 60)

    # ── Resumen final ─────────────────────────────────────────────────────────
    check_py = ROOT / "check_data.py"
    if check_py.exists():
        print()
        subprocess.call([PYTHON, str(check_py)], cwd=str(ROOT))
    print("\n  Reinicia el servidor para que tome los datos frescos.\n")


if __name__ == "__main__":
    main()
