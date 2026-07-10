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
echo [0/6] Verificando servidor local...
curl -s -f -o nul --max-time 5 "%LOCAL_API%/api/health"
if errorlevel 1 goto server_down
echo   OK, servidor responde.
echo.

:: ── Paso 1: generar snapshot ──────────────────────────────
echo [1/6] Generando snapshot de hoy (~30s)...
curl -s -X POST --max-time 180 "%LOCAL_API%/api/publish" -o publish_result.json
if errorlevel 1 goto publish_failed
if not exist publish_result.json goto publish_no_response

findstr /C:"\"ok\":true" publish_result.json >nul 2>&1
if errorlevel 1 goto publish_bad_response

echo   Respuesta:
type publish_result.json
echo.
echo.
del publish_result.json

:: ── Paso 2: sincronizar frontend con vercel ────────────────
echo [2/6] Sincronizando strike.html -^> vercel/index.html...
if not exist "src\serve\static\strike.html" goto strike_missing
copy /Y "src\serve\static\strike.html" "vercel\index.html" >nul
if errorlevel 1 goto copy_failed
echo   OK.
echo.

:: ── Paso 3: checkpoint local de todo lo modificado ─────────
:: Comiteamos TODAS las modificaciones a archivos ya trackeados (parquets, pkl,
:: codigo, snapshot, index.html) como un solo commit. Esto garantiza que NADA
:: se pierde al sincronizar con remoto. Untracked (archivos nuevos que git no
:: conoce) NO se agregan — quedan intactos en el working tree.
echo [3/6] Guardando cambios locales en un commit checkpoint...
git add -u
git diff --cached --quiet
if not errorlevel 1 (
    echo   Sin cambios locales que preservar.
    goto sync_remote
)
for /f %%d in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm')"') do set FECHA=%%d
git commit -m "data: publish %FECHA%" >nul
if errorlevel 1 goto commit_failed
echo   Checkpoint commit creado.
echo.

:sync_remote
:: ── Paso 4: traer commits remotos del bot sin destruir nada ────────
echo [4/6] Sincronizando con remoto (bot de odds)...
git fetch origin main
if errorlevel 1 goto fetch_failed

:: Merge normal con estrategia "ours": si hay conflicto (mismo archivo tocado
:: local y remoto — tipico en parquets del bot), gana LA VERSION LOCAL.
:: Tu data local es fresca (update_data.bat) y mas confiable que la del bot.
:: NUNCA introduce conflict markers en el working tree.
git merge origin/main --no-edit --strategy-option=ours
if errorlevel 1 goto merge_failed
echo   Merge OK.
echo.

:: ── Paso 5: push ──────────────────────────────────────────
echo [5/6] Push a GitHub (Vercel se auto-despliega)...
:: Verificar que haya algo que pushear
git status -sb | findstr /C:"ahead" >nul 2>&1
if errorlevel 1 (
    echo   Nada nuevo que pushear ^(remoto ya al dia^).
    goto done
)
git push
if errorlevel 1 goto push_failed
echo   Push exitoso.
echo.

:done
echo [6/6] Publicacion completa.
echo.
echo ============================================================
echo   OK. Vercel redesplegara en ~30 segundos.
echo   Refresca la web para ver los picks del dia.
echo ============================================================
goto end

:: ─── Errores (todos NO destructivos, con instrucciones de recuperacion) ───

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

:strike_missing
echo.
echo   ERROR: no existe src\serve\static\strike.html.
echo   No puedo sincronizar el frontend a Vercel.
goto end

:copy_failed
echo.
echo   ERROR: no pude copiar strike.html a vercel/index.html.
echo   Verifica permisos y que vercel/ exista.
goto end

:commit_failed
echo.
echo   ERROR: git commit fallo. Revisa mensaje anterior.
echo   Los cambios siguen stageados. Puedes:
echo     git reset          (para des-stagear sin perder)
echo     git commit -m ...  (para reintentar con otro mensaje)
goto end

:fetch_failed
echo.
echo   ERROR: git fetch fallo. Verifica conexion a GitHub.
echo   Nada se ha modificado en el repo. Reintenta.
goto end

:merge_failed
echo.
echo   ERROR: git merge fallo pese a la estrategia "ours".
echo   Esto es raro. Estado del repo:
git status --short
echo.
echo   RECUPERACION SEGURA:
echo     git merge --abort    (vuelve al estado previo al merge)
echo   NO se ha perdido ningun cambio; el checkpoint commit sigue en local.
goto end

:push_failed
echo.
echo   ERROR en git push. Verifica conexion o credenciales.
echo   El commit sigue en local. Reintenta con:
echo     git push
goto end

:end
echo.
pause
endlocal
