@echo off
setlocal
title STRIKECAST - Pinnacle scheduler (leave open all day)
set PYTHON=C:\Users\andra\miniconda3\python.exe
set PROJECT=C:\Users\andra\Desktop\MLB SLEC
cd /d "%PROJECT%"

:: the-odds-api.com key (free tier: 500 credits/month, ~3 per pull).
set ODDS_API_KEY=f5be61575edbd4a77f398d40df2d8c83

echo.
echo ============================================================
echo   STRIKECAST ^| Pinnacle scheduler
echo ============================================================
echo   Pulls sharp odds ~1h before each game (clustered games
echo   share one call). Leave this window OPEN all day.
echo   --cover 90  = ~1h freshness  (more pulls, ~15 cred/day)
echo   --cover 120 = ~2h freshness  (fewer pulls, ~9 cred/day)
echo ============================================================
echo.

"%PYTHON%" -m src.pinnacle_scheduler --lead 60 --cover 100

echo.
echo   Scheduler finished for today.
pause
endlocal
