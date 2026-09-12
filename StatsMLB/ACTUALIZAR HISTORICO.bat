@echo off
setlocal
set "APP_DIR=%~dp0"
for %%I in ("%APP_DIR%..") do set "ROOT_DIR=%%~fI"
set "UV=C:\Users\andra\.local\bin\uv.exe"
set "NODE_DIR=C:\Users\andra\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin"
set "PNPM=C:\Users\andra\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd"
set "PYTHONUNBUFFERED=1"

echo [1/4] Actualizando el mes completo desde MLB...
pushd "%APP_DIR%"
"%UV%" run python -u scripts\refresh_history_schedule.py
if errorlevel 1 goto :error

echo [2/4] Reconstruyendo estadisticas de series...
pushd "%ROOT_DIR%"
"%UV%" run --with pandas --with pyarrow --with numpy python -u analysis\mlb_three_game_sweep_report.py --output-dir work\sweep_report_20260824
if errorlevel 1 goto :error

echo [3/4] Entrenando el estimador y preparando el dashboard...
pushd "%APP_DIR%"
"%UV%" run --with pandas --with pyarrow --with scikit-learn python -u scripts\build_data.py
if errorlevel 1 goto :error

echo [4/4] Compilando StatsMLB...
set "PATH=%NODE_DIR%;%PATH%"
"%PNPM%" build
if errorlevel 1 goto :error

echo.
echo StatsMLB quedo actualizado correctamente.
pause
exit /b 0

:error
echo.
echo La actualizacion se detuvo por un error. Los datos anteriores siguen disponibles.
pause
exit /b 1
