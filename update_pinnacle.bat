@echo off
setlocal
title STRIKECAST - Pinnacle sharp odds
set PYTHON=C:\Users\andra\miniconda3\python.exe
set PROJECT=C:\Users\andra\Desktop\MLB SLEC
cd /d "%PROJECT%"

:: the-odds-api.com key (free tier: 500 req/month). Each pull ~3 credits.
:: Run this a few times per day, ideally 1-3h before first pitch when soft
:: books are softest. NOT every minute — you'll burn the monthly quota.
set ODDS_API_KEY=f5be61575edbd4a77f398d40df2d8c83

echo.
echo ============================================================
echo   STRIKECAST ^| Pinnacle sharp odds pull
echo ============================================================
echo.

echo [1/2] Pulling Pinnacle + soft books from the-odds-api...
"%PYTHON%" -m src.ingest_odds_api
if ERRORLEVEL 1 (
    echo   ERROR pulling odds. Check the API key / quota.
    pause & exit /b 1
)
echo.

echo [2/2] Building sharp-value table (soft book vs Pinnacle fair)...
"%PYTHON%" -m src.normalize.pinnacle_value
if ERRORLEVEL 1 (
    echo   ERROR building pinnacle_value.
    pause & exit /b 1
)
echo.
echo   Done. Restart / refresh the server to load the fresh sharp values.
echo.
pause
endlocal
