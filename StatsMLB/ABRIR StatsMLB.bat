@echo off
setlocal
for %%I in ("%~dp0.") do set "APP_DIR=%%~fI"
set "URL=http://127.0.0.1:4173"

powershell -NoProfile -ExecutionPolicy Bypass -File "%APP_DIR%\scripts\start-local.ps1" -AppDirectory "%APP_DIR%"
if errorlevel 1 (
  echo No se pudo compilar o iniciar StatsMLB. Revisa %%LOCALAPPDATA%%\StatsMLB\server-error.log.
  pause
  exit /b 1
)

powershell -NoProfile -Command "$ready=$false; 1..60 | ForEach-Object { try { $home=Invoke-WebRequest -UseBasicParsing -Uri '%URL%' -TimeoutSec 3; $health=Invoke-RestMethod -Uri '%URL%/api/health' -TimeoutSec 3; if ($home.StatusCode -eq 200 -and $health.status -eq 'ok' -and $health.database -eq 'ready') { $ready=$true; return } } catch {}; Start-Sleep -Seconds 1 }; if (-not $ready) { exit 1 }"
if errorlevel 1 (
  echo StatsMLB no pudo iniciar en el puerto 4173.
  pause
  exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()"') do set "CACHE_BUST=%%I"
start "" "%URL%?inicio=%CACHE_BUST%"
endlocal
