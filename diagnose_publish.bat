@echo on
setlocal
title STRIKECAST - Diagnostico publish

cd /d "%~dp0"

echo.
echo === Test 1: Health check ===
curl -v --max-time 10 "http://localhost:8000/api/health"
echo.
echo (exit code de curl: %ERRORLEVEL%)
echo.
echo ===============================
pause

echo.
echo === Test 2: Llamando /api/publish con verbose ===
curl -v -X POST --max-time 180 "http://localhost:8000/api/publish"
echo.
echo (exit code de curl: %ERRORLEVEL%)
echo.
echo ===============================
pause

endlocal
