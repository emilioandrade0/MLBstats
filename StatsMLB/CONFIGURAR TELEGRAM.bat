@echo off
chcp 65001 >nul
setlocal
set "TELEGRAM_DIR=%LOCALAPPDATA%\StatsMLB"
set "TELEGRAM_FILE=%TELEGRAM_DIR%\telegram.env"

if not exist "%TELEGRAM_DIR%" mkdir "%TELEGRAM_DIR%"
if not exist "%TELEGRAM_FILE%" (
  >"%TELEGRAM_FILE%" echo TELEGRAM_BOT_TOKEN=
  >>"%TELEGRAM_FILE%" echo TELEGRAM_CHAT_ID=
)

echo.
echo Se abrira el archivo privado de configuracion.
echo Pega el token despues de TELEGRAM_BOT_TOKEN=
echo Pega @usuario_del_canal o el ID numerico despues de TELEGRAM_CHAT_ID=
echo Guarda el archivo y cierra el Bloc de notas.
echo.
start "" notepad.exe "%TELEGRAM_FILE%"
pause
endlocal
