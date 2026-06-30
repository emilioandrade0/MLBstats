@echo off
setlocal
title STRIKECAST

cd /d "%~dp0"
set "HOST=127.0.0.1"
set "PORT=8000"
set "URL=http://%HOST%:%PORT%"
set "PYTHON=%USERPROFILE%\miniconda3\python.exe"

echo.
echo ============================================================
echo   STRIKECAST ^| Iniciando aplicacion
echo ============================================================
echo.

if not exist "%PYTHON%" set "PYTHON=%USERPROFILE%\anaconda3\python.exe"
if not exist "%PYTHON%" (
    echo ERROR: No se encontro Python de Miniconda o Anaconda.
    echo Ruta esperada: %USERPROFILE%\miniconda3\python.exe
    echo.
    pause
    exit /b 1
)

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo La aplicacion ya parece estar ejecutandose en %URL%.
    echo Abriendo el navegador...
    start "" "%URL%"
    exit /b 0
)

echo La web se abrira cuando el servidor este listo...
start "" powershell -NoProfile -WindowStyle Hidden -Command ^
    "$url='%URL%'; for ($i=0; $i -lt 30; $i++) { try { Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 1 ^| Out-Null; Start-Process $url; exit } catch { Start-Sleep -Seconds 1 } }"

echo Servidor iniciado. No cierres esta ventana mientras uses la app.
echo Para detenerlo, presiona Ctrl+C.
echo.
"%PYTHON%" -m uvicorn src.serve.api:app --host %HOST% --port %PORT%

if errorlevel 1 (
    echo.
    echo ERROR: No se pudo iniciar STRIKECAST.
    echo Revisa el mensaje mostrado arriba.
    echo.
    pause
)

endlocal
