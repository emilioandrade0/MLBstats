@echo off
setlocal
title STRIKECAST - Actualizacion de datos (paralela)

:: Consola en UTF-8 para que los bloques de la barra de progreso se vean bien
chcp 65001 >nul

set PYTHON=C:\Users\andra\miniconda3\python.exe
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

"%PYTHON%" update_data.py %*
if ERRORLEVEL 1 (
    echo.
    echo   ERROR: update_data.py fallo. Revisa los logs en .\logs\
    pause & exit /b 1
)

echo.
pause
endlocal
