@echo off
setlocal EnableDelayedExpansion
title STRIKECAST - Experimento Retrain v2

set PYTHON=C:\Users\andra\miniconda3\python.exe
set PROJECT=C:\Users\andra\Desktop\MLB SLEC

cd /d "%PROJECT%"

echo.
echo ============================================================
echo   STRIKECAST ^| Experimento Retrain v2
echo ============================================================
echo.
echo   Este script entrena un modelo v2 con datos actualizados
echo   (incluye temporada completa 2025 + 2026) y lo compara
echo   con el modelo de produccion actual.
echo.
echo   Los modelos de PRODUCCION no se tocan hasta que tu
echo   confirmes con --deploy.
echo.
echo ============================================================
echo.

:: Verificar que los datos esten actualizados
echo [0/3] Verificando estado de datos...
"%PYTHON%" check_data.py
echo.
echo   Si los datos estan atrasados, cierra esta ventana y
echo   corre update_data.bat primero.
echo.
pause

:: ── PASO 1: Entrenar v2 ────────────────────────────────────────
echo [1/3] Entrenando modelos v2...
echo   train: 2023-01-01 a 2025-09-30
echo   val:   2025-10-01 a 2026-03-31
echo   test:  2026-04-01 a hoy
echo.
"%PYTHON%" -m src.model.train_v2
if ERRORLEVEL 1 (
    echo.
    echo   ERROR en train_v2. Revisa el log arriba.
    pause & exit /b 1
)
echo.
echo   Modelos v2 guardados en data\models\v2\
echo.

:: ── PASO 2: Walk-forward en meses 2026 ────────────────────────
echo [2/3] Walk-forward honesto (meses 2026)...
echo   Esto reentrena un modelo POR MES usando solo datos previos.
echo   Es la evaluacion mas honesta de mejora futura.
echo.
"%PYTHON%" -m src.model.walkforward --start 2026-01 --end 2026-06
if ERRORLEVEL 1 (
    echo.
    echo   AVISO: walk-forward fallo (puede ser que falten meses completos).
    echo   Continua con la comparacion del paso 1.
)
echo.

:: ── PASO 3: Resumen ────────────────────────────────────────────
echo ============================================================
echo   RESUMEN
echo ============================================================
echo.
echo   Los modelos v2 estan en data\models\v2\
echo   Los modelos de produccion (v1) estan intactos.
echo.
echo   Para desplegar v2 si los resultados te convencen:
echo.
echo     %PYTHON% -m src.model.train_v2 --deploy
echo.
echo   Esto:
echo     1. Hace backup de v1 en data\models\v1_backup\
echo     2. Copia v2 -> data\models\ (produccion)
echo     3. Reinicia el servidor para tomar el nuevo modelo
echo.
echo ============================================================
echo.
pause
endlocal
