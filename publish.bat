@echo off
setlocal
title STRIKECAST - Publicar en Vercel

set PYTHON=C:\Users\andra\miniconda3\python.exe
set PROJECT=C:\Users\andra\Desktop\MLB SLEC
set LOCAL_API=http://localhost:8000

cd /d "%PROJECT%"

echo.
echo ============================================================
echo   STRIKECAST ^| Publicar predicciones en la web
echo ============================================================
echo.

:: ── Paso 0: server vivo ────────────────────────────────────
echo [0/4] Verificando servidor local...
curl -s -f -o nul --max-time 5 "%LOCAL_API%/api/health"
if not "%ERRORLEVEL%"=="0" goto server_down
echo   OK, servidor responde.
echo.

:: ── Paso 1: generar snapshot ──────────────────────────────
echo [1/4] Generando snapshot de hoy (~30s)...
curl -s -X POST --max-time 180 "%LOCAL_API%/api/publish" -o publish_result.json
if not "%ERRORLEVEL%"=="0" goto publish_failed
if not exist publish_result.json goto publish_no_response

findstr /C:"\"ok\":true" publish_result.json >nul 2>&1
if not "%ERRORLEVEL%"=="0" goto publish_bad_response

echo   Respuesta:
type publish_result.json
echo.
echo.
del publish_result.json

:: ── Paso 2: sincronizar con remoto (protege contra conflictos del bot) ──
echo [2/4] Sincronizando repo con remoto...
:: Backup del snapshot antes de tocar git
copy /Y vercel\snapshot.json "%TEMP%\strikecast_snapshot.json" >nul

:: Stashear cambios locales sin commitear (codigo, pickles, etc.) — sin untracked
git stash push -m "publish-auto-stash" >nul 2>&1

:: Sincronizar main a origin (destruye commits locales redundantes de snapshots previos)
git fetch origin
git reset --hard origin/main
if not "%ERRORLEVEL%"=="0" goto reset_failed

:: Restaurar el snapshot desde backup
copy /Y "%TEMP%\strikecast_snapshot.json" vercel\snapshot.json >nul
del "%TEMP%\strikecast_snapshot.json"

:: ── Paso 3: commit + push ─────────────────────────────────
echo.
echo [3/4] Publicando en Vercel (git push)...
git add vercel/snapshot.json
for /f %%d in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm')"') do set FECHA=%%d
git commit -m "data: snapshot %FECHA%" >nul
if not "%ERRORLEVEL%"=="0" (
    echo   Sin cambios en snapshot ^(ya publicado^), saltando push.
    goto restore_stash
)
git push
if not "%ERRORLEVEL%"=="0" goto push_failed

echo   Push exitoso.
echo.

:: ── Paso 4: restaurar cambios locales ─────────────────────
:restore_stash
echo [4/4] Restaurando tus cambios locales...
:: pop puede fallar por conflictos en parquets/pickles (bot vs local). Los aceptamos como locales.
git stash pop >nul 2>&1
if "%ERRORLEVEL%"=="0" goto done

:: Si hay conflictos, casi seguro es en parquets/pickles que el bot toco.
:: Aceptamos la version LOCAL (tuya) para todo lo conflictuado.
echo   Conflictos detectados en binarios (normal si el bot subio parquets nuevos).
echo   Aceptando tu version local...
for %%f in (data/processed/odds_close.parquet data/processed/features_market.parquet data/processed/games.parquet data/processed/train.parquet data/models/lgb_cls.pkl data/models/lgb_reg.pkl) do (
    git checkout --theirs -- "%%f" >nul 2>&1
    git add "%%f" >nul 2>&1
)
:: Cualquier otro conflicto residual: intentar auto-resolver con la version stash
git status --porcelain | findstr /B "UU AA DD" >nul 2>&1
if "%ERRORLEVEL%"=="0" (
    echo   AVISO: quedaron conflictos que no pude resolver.
    echo   Corre: git stash drop     (para descartar el stash)
    echo   Y luego: update_data.bat  (para regenerar parquets locales)
    goto done
)
git stash drop >nul 2>&1

:done
echo.
echo ============================================================
echo   OK. Vercel redesplegara en ~30 segundos.
echo   Refresca la web para ver los picks de hoy.
echo ============================================================
goto end

:server_down
echo.
echo   ERROR: El servidor local no responde en %LOCAL_API%.
echo   Abre STRIKECAST (iniciar_app.bat), espera a que termine
echo   de cargar y vuelve a publicar.
goto end

:publish_failed
echo.
echo   ERROR: curl fallo o timeout al llamar /api/publish.
echo   Revisa la ventana del servidor por errores en consola.
if exist publish_result.json del publish_result.json
goto end

:publish_no_response
echo.
echo   ERROR: /api/publish no devolvio respuesta.
goto end

:publish_bad_response
echo.
echo   ERROR: /api/publish no devolvio "ok":true. Respuesta:
type publish_result.json
del publish_result.json
goto end

:reset_failed
echo.
echo   ERROR: git reset --hard origin/main fallo.
echo   Estado del repo indeterminado. Contactame antes de tocar mas.
goto end

:push_failed
echo.
echo   ERROR en git push. Verifica conexion o credenciales.
echo   Snapshot commit fue creado localmente pero no llego a Vercel.
goto end

:end
echo.
pause
endlocal
