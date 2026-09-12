@echo off
setlocal EnableExtensions EnableDelayedExpansion
title STRIKECAST

cd /d "%~dp0"
set "HOST=127.0.0.1"
set "PORT=8000"
set "URL=http://%HOST%:%PORT%"
set "APP_MODULE=src.serve.api:app"
set "API_FILE=%CD%\src\serve\api.py"
set "RELOAD_DIR=%CD%\src"
set "PYTHON=%USERPROFILE%\miniconda3\python.exe"
set "PYTHONDONTWRITEBYTECODE=1"
set "STRIKECAST_DISABLE_REFRESH=1"
set "STRIKECAST_ONREQ_TTL_SEC=120"

echo.
echo ============================================================
echo   STRIKECAST ^| Iniciando aplicacion
echo ============================================================
echo   Backend canonico: %APP_MODULE%
echo   Archivo: %API_FILE%
echo.

if not exist "%PYTHON%" set "PYTHON=%USERPROFILE%\anaconda3\python.exe"
if not exist "%PYTHON%" (
    echo ERROR: No se encontro Python de Miniconda o Anaconda.
    echo Ruta esperada: %USERPROFILE%\miniconda3\python.exe
    echo.
    pause
    exit /b 1
)
if not exist "%API_FILE%" (
    echo ERROR: No existe el backend canonico:
    echo %API_FILE%
    echo.
    pause
    exit /b 1
)

for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath '%API_FILE%').Hash.ToLowerInvariant()"`) do set "EXPECTED_API_SHA=%%H"
if not defined EXPECTED_API_SHA (
    echo ERROR: No se pudo calcular la version del backend.
    pause
    exit /b 1
)

set "LISTENER_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do if not defined LISTENER_PID set "LISTENER_PID=%%P"

if defined LISTENER_PID (
    set "RUNNING_COMMAND="
    for /f "usebackq delims=" %%C in (`powershell -NoProfile -Command "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=!LISTENER_PID!' -ErrorAction SilentlyContinue; if ($p) { Write-Output $p.CommandLine }"`) do set "RUNNING_COMMAND=%%C"
    set "HEALTH_LINE="
    for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "try { $h=Invoke-RestMethod -Uri '%URL%/api/health' -TimeoutSec 4; Write-Output ($h.service + '|' + $h.backend_module + '|' + $h.source_sha256) } catch { Write-Output '||' }"`) do set "HEALTH_LINE=%%H"
    for /f "tokens=1-3 delims=|" %%A in ("!HEALTH_LINE!") do (
        set "RUNNING_SERVICE=%%A"
        set "RUNNING_MODULE=%%B"
        set "RUNNING_SHA=%%C"
    )
    if /I "!RUNNING_SERVICE!"=="strikecast" if /I "!RUNNING_MODULE!"=="src.serve.api" if /I "!RUNNING_SHA!"=="!EXPECTED_API_SHA!" (
        echo Backend activo, canonico y actualizado. PID !LISTENER_PID!.
        echo Abriendo %URL% ...
        start "" "%URL%"
        echo.
        echo Puedes cerrar esta ventana; STRIKECAST ya esta ejecutandose.
        pause
        exit /b 0
    )

    set "IS_STRIKECAST="
    if /I "!RUNNING_SERVICE!"=="strikecast" set "IS_STRIKECAST=1"
    echo(!RUNNING_COMMAND!| findstr /I /C:"src.serve.api:app" >nul 2>&1 && set "IS_STRIKECAST=1"
    if defined IS_STRIKECAST (
        echo Se encontro un STRIKECAST anterior o desactualizado. Reiniciando PID !LISTENER_PID!...
        taskkill /PID !LISTENER_PID! /T /F >nul 2>&1
        timeout /t 2 /nobreak >nul
        netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
        if not errorlevel 1 (
            echo ERROR: El backend anterior sigue usando el puerto %PORT%.
            echo Cierra su ventana manualmente y vuelve a ejecutar este archivo.
            pause
            exit /b 1
        )
    ) else (
        echo ERROR: El puerto %PORT% esta ocupado por otro servicio. PID !LISTENER_PID!.
        echo Comando detectado: !RUNNING_COMMAND!
        echo No se detuvo ese proceso por seguridad.
        pause
        exit /b 1
    )
)

echo La web se abrira cuando el healthcheck confirme el backend actualizado...
start "" powershell -NoProfile -WindowStyle Hidden -Command ^
    "$url='%URL%'; $expected='%EXPECTED_API_SHA%'; for ($i=0; $i -lt 45; $i++) { try { $h=Invoke-RestMethod -Uri ($url + '/api/health') -TimeoutSec 2; if ($h.backend_module -eq 'src.serve.api' -and $h.source_sha256 -eq $expected) { Start-Process $url; exit } } catch {}; Start-Sleep -Seconds 1 }"

echo Servidor iniciado con recarga limitada a src\.
echo No cierres esta ventana mientras uses la app.
echo Para detenerlo, presiona Ctrl+C.
echo.
"%PYTHON%" -m uvicorn %APP_MODULE% --app-dir "%CD%" --host %HOST% --port %PORT% --reload --reload-dir "%RELOAD_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: No se pudo iniciar STRIKECAST.
    echo Revisa el mensaje mostrado arriba.
    echo.
    pause
)

endlocal
