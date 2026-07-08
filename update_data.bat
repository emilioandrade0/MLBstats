@echo off
setlocal EnableDelayedExpansion
title STRIKECAST - Actualizacion de datos

set PYTHON=C:\Users\andra\miniconda3\python.exe
set PROJECT=C:\Users\andra\Desktop\MLB SLEC

cd /d "%PROJECT%"

:: Calcular fechas con PowerShell
for /f %%d in ('powershell -Command "(Get-Date).ToString('yyyy-MM-dd')"') do set TODAY=%%d
for /f %%d in ('powershell -Command "(Get-Date).AddDays(-14).ToString('yyyy-MM-dd')"') do set SINCE=%%d

echo.
echo ============================================================
echo   STRIKECAST ^| Actualizacion incremental de datos
echo ============================================================
echo   Hoy     : %TODAY%
echo   Ventana : %SINCE%  ->  %TODAY%
echo ============================================================
echo.

:: ── PASO 1: Re-descargar ESPN (scoreboards, odds, probabilities) ──────────
echo [1/8] ESPN: re-descargando ultimos 14 dias (--force)...
"%PYTHON%" -m src.ingest --start %SINCE% --end %TODAY% --force
if ERRORLEVEL 1 (
    echo.
    echo   ERROR en ESPN ingest. Revisa la conexion.
    pause & exit /b 1
)
echo   OK
echo.

:: ── PASO 2: Re-descargar StatsAPI feeds (resultados finales) ─────────────
echo [2/8] StatsAPI: re-descargando feeds de los ultimos 14 dias (--force)...
"%PYTHON%" -m src.ingest_statsapi --start %SINCE% --end %TODAY% --force
if ERRORLEVEL 1 (
    echo.
    echo   ERROR en StatsAPI ingest.
    pause & exit /b 1
)
echo   OK
echo.

:: ── PASO 3: Reconstruir tablas normalizadas ───────────────────────────────
echo [3/8] Normalize: reconstruyendo games, team_box, player_box, plays, odds...
"%PYTHON%" -m src.normalize.build_all games team_box player_box plays odds win_probability xref
if ERRORLEVEL 1 (
    echo.
    echo   ERROR en normalize.
    pause & exit /b 1
)
echo   OK
echo.

:: ── PASO 4: Savant incremental (solo lo faltante) ─────────────────────────
echo [4/8] Savant: descargando solo semanas faltantes...
for /f %%d in ('"%PYTHON%" -c "from pathlib import Path; import pandas as pd; p=Path(r'data/processed/pitches.parquet'); import sys; sys.stdout.write(((pd.to_datetime(pd.read_parquet(p, columns=['game_date'])['game_date']).max() + pd.Timedelta(days=1)).date().isoformat()) if p.exists() else '')"') do set SAVANT_START=%%d
if not defined SAVANT_START set SAVANT_START=%SINCE%
echo   Ultimo pitch procesado + 1 dia: %SAVANT_START%
if "%SAVANT_START%" GTR "%TODAY%" (
    echo   Savant ya esta al dia.
) else (
    "%PYTHON%" -m src.ingest_savant --start %SAVANT_START% --end %TODAY%
    if ERRORLEVEL 1 (
        echo.
        echo   ERROR en ingest_savant.
        pause & exit /b 1
    )
    echo   Rebuild pitches.parquet...
    "%PYTHON%" -m src.normalize.build_all pitches
    if ERRORLEVEL 1 (
        echo.
        echo   ERROR reconstruyendo pitches.parquet.
        pause & exit /b 1
    )
)
echo   OK
echo.

:: ── PASO 5: Reconstruir odds_close (closing lines para el modelo) ─────────
echo [5/8] Odds close: reconstruyendo odds_close.parquet...
"%PYTHON%" -m src.normalize.odds_v2
if ERRORLEVEL 1 (
    echo.
    echo   ERROR en odds_v2 (odds_close).
    pause & exit /b 1
)
echo   OK
echo.

:: ── PASO 6: Reconstruir features ─────────────────────────────────────────
echo [6/8] Features: reconstruyendo todas las features del modelo...
echo   market_close...
"%PYTHON%" -m src.features.market_close
echo   team_form...
"%PYTHON%" -m src.features.team_form
echo   pitcher_form...
"%PYTHON%" -m src.features.pitcher_form
echo   pitcher_season...
"%PYTHON%" -m src.features.pitcher_season
echo   lineup...
"%PYTHON%" -m src.features.lineup
echo   park...
"%PYTHON%" -m src.features.park
echo   elo...
"%PYTHON%" -m src.features.elo
echo   bullpen_load...
"%PYTHON%" -m src.features.bullpen_load
echo   series_context...
"%PYTHON%" -m src.features.series_context
echo   umpires...
"%PYTHON%" -m src.features.umpires
echo   f5_targets...
"%PYTHON%" -m src.features.f5_targets
if ERRORLEVEL 1 (
    echo.
    echo   ERROR en features (ultimo modulo).
    pause & exit /b 1
)
echo   OK
echo.

:: ── PASO 7: Comeback analysis + comeback features ─────────────────────────
echo [7/8] Comeback: recalculando remontas y features de resiliencia...
echo   comeback_analysis...
"%PYTHON%" -m src.analysis.comeback_analysis
if ERRORLEVEL 1 (
    echo.
    echo   ERROR en comeback_analysis.
    pause & exit /b 1
)
echo   comeback features...
"%PYTHON%" -m src.features.comeback
if ERRORLEVEL 1 (
    echo.
    echo   ERROR en features.comeback.
    pause & exit /b 1
)
echo   game_flow (analisis, no entra al modelo)...
"%PYTHON%" -m src.features.game_flow
echo   burn_analysis (carne al asador, solo web)...
"%PYTHON%" -m src.analysis.burn_analysis
echo   OK
echo.

:: ── PASO 8: Reconstruir train.parquet ────────────────────────────────────
echo [8/8] Train: ensamblando train.parquet...
echo   Verificando que train.parquet no este bloqueado...
"%PYTHON%" -c "from pathlib import Path; p=Path(r'data/processed/train.parquet'); f=None; import sys; exec(\"if p.exists():\\n    try:\\n        f=open(p, 'r+b')\\n    except OSError as e:\\n        print('TRAIN_LOCKED'); print(e); raise SystemExit(1)\\n    finally:\\n        (f is not None) and f.close()\")"
if ERRORLEVEL 1 (
    echo.
    echo   ERROR: train.parquet esta bloqueado por otra app o proceso.
    echo   Cierra STRIKECAST, cualquier servidor uvicorn/python, y vuelve a correr este script.
    pause & exit /b 1
)
"%PYTHON%" -m src.features.build
if ERRORLEVEL 1 (
    echo.
    echo   ERROR al construir train.parquet.
    echo   Si STRIKECAST estaba abierto, probablemente el archivo quedo bloqueado.
    echo   Cierra la app/servidor y vuelve a correr este script.
    pause & exit /b 1
)
echo   OK
echo.

:: ── Resumen final ─────────────────────────────────────────────────────────
echo ============================================================
"%PYTHON%" check_data.py
echo ============================================================
echo.
echo   Actualizacion completada exitosamente!
echo   Reinicia el servidor para que tome los nuevos datos.
echo.
pause
endlocal
