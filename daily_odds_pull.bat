@echo off
setlocal
title STRIKECAST - Daily odds pull (Task Scheduler)

REM Pipeline diario no-interactivo. Seguro para Task Scheduler.
REM Corre en orden:
REM   1) ESPN refresh (odds hoy + proximos 7 dias, rebuild odds_close.parquet)
REM   2) the-odds-api Pinnacle pull + pinnacle_value normalize
REM   3) StatsMLB odds JSON regen
REM   4) Value picks model retrain + prediction parquet
REM Sin `pause` para no bloquear la tarea programada.

set PYTHON=C:\Users\andra\miniconda3\python.exe
set PROJECT=C:\Users\andra\Desktop\STRIKECAST
cd /d "%PROJECT%" || exit /b 1

set ODDS_API_KEY=f5be61575edbd4a77f398d40df2d8c83

set LOGDIR=%PROJECT%\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOGFILE=%LOGDIR%\daily_odds_pull.log

echo. >> "%LOGFILE%"
echo === %date% %time% === >> "%LOGFILE%"

echo [1/4] ESPN refresh (odds+scoreboard, hoy + 7 dias adelante)... >> "%LOGFILE%"
"%PYTHON%" -m src.refresh --days-forward 7 >> "%LOGFILE%" 2>&1
if ERRORLEVEL 1 (
    echo   FAIL src.refresh >> "%LOGFILE%"
    exit /b 1
)

echo [2/4] Pinnacle pull (the-odds-api)... >> "%LOGFILE%"
"%PYTHON%" -m src.ingest_odds_api >> "%LOGFILE%" 2>&1
if ERRORLEVEL 1 (
    echo   FAIL ingest_odds_api ^(sigue^) >> "%LOGFILE%"
    REM No aborta: la key puede haber agotado quota; el resto sigue.
)

echo [2b/4] Building pinnacle_value ... >> "%LOGFILE%"
"%PYTHON%" -m src.normalize.pinnacle_value >> "%LOGFILE%" 2>&1
if ERRORLEVEL 1 (
    echo   FAIL pinnacle_value ^(sigue^) >> "%LOGFILE%"
)

echo [3/4] Regenerating StatsMLB odds JSONs ... >> "%LOGFILE%"
"%PYTHON%" "%PROJECT%\StatsMLB\scripts\odds_walkforward.py" >> "%LOGFILE%" 2>&1
if ERRORLEVEL 1 (
    echo   FAIL odds_walkforward.py ^(sigue^) >> "%LOGFILE%"
)

echo [4/4] Value picks retrain + predict ... >> "%LOGFILE%"
"%PYTHON%" -m src.model.value_picks --summary >> "%LOGFILE%" 2>&1
if ERRORLEVEL 1 (
    echo   FAIL value_picks >> "%LOGFILE%"
    exit /b 1
)

echo   DONE >> "%LOGFILE%"
endlocal
exit /b 0
