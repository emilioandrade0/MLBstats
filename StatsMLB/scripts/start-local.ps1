param(
  [string]$AppDirectory = (Split-Path -Parent $PSScriptRoot),
  [string]$NodeDirectory = '',
  [string]$PnpmPath = ''
)

function Resolve-ExistingPath([string[]]$Candidates) {
  foreach ($candidate in $Candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }
  return $null
}

$bundledRoot = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies'
$commandNode = Get-Command node.exe -ErrorAction SilentlyContinue | Select-Object -First 1
$nodeExecutable = Resolve-ExistingPath @(
  $(if ($NodeDirectory) { Join-Path $NodeDirectory 'node.exe' }),
  $commandNode.Source,
  (Join-Path $bundledRoot 'node\bin\node.exe')
)
$commandPnpm = Get-Command pnpm.cmd -ErrorAction SilentlyContinue | Select-Object -First 1
$resolvedPnpm = Resolve-ExistingPath @(
  $PnpmPath,
  $commandPnpm.Source,
  (Join-Path $bundledRoot 'bin\fallback\pnpm.cmd')
)

if (-not $nodeExecutable -or -not $resolvedPnpm) {
  throw 'No se encontro Node o pnpm. Abre Codex una vez y vuelve a ejecutar el iniciador.'
}

$packageManifest = Join-Path $AppDirectory 'package.json'
if (-not (Test-Path -LiteralPath $packageManifest -PathType Leaf)) {
  throw "No se encontro package.json en $AppDirectory"
}
$AppDirectory = (Resolve-Path -LiteralPath $AppDirectory).Path
Set-Location -LiteralPath $AppDirectory

$NodeDirectory = Split-Path -Parent $nodeExecutable
$PnpmPath = $resolvedPnpm

$telegramEnvironment = Join-Path $env:LOCALAPPDATA 'StatsMLB\telegram.env'
if (-not (Test-Path -LiteralPath $telegramEnvironment)) {
  throw "No se encontro la configuracion local de Telegram en $telegramEnvironment"
}

Get-Content -LiteralPath $telegramEnvironment | ForEach-Object {
  $line = $_.Trim()
  if (-not $line -or $line.StartsWith('#')) { return }
  $parts = $line -split '=', 2
  if ($parts.Count -ne 2 -or -not $parts[0].Trim()) { return }
  Set-Item -Path "Env:$($parts[0].Trim())" -Value $parts[1].Trim()
}

if (-not $env:TELEGRAM_BOT_TOKEN -or -not $env:TELEGRAM_CHAT_ID) {
  throw 'La configuracion local de Telegram esta incompleta.'
}

# Cloudflare's local Worker reads secrets from .dev.vars. This file is ignored
# by Git and never belongs in a Sites deployment archive.
$developmentVariables = Join-Path $AppDirectory '.dev.vars'
Set-Content -LiteralPath $developmentVariables -Encoding utf8 -Value @(
  "TELEGRAM_BOT_TOKEN=$($env:TELEGRAM_BOT_TOKEN)"
  "TELEGRAM_CHAT_ID=$($env:TELEGRAM_CHAT_ID)"
)

$env:PORT = '4173'
$env:Path = "$NodeDirectory;$env:Path"
$stateDirectory = Join-Path $env:LOCALAPPDATA 'StatsMLB'
$pidFile = Join-Path $stateDirectory 'server.pid'
$stdoutLog = Join-Path $stateDirectory 'server.log'
$stderrLog = Join-Path $stateDirectory 'server-error.log'
$vinextCli = Join-Path $AppDirectory 'node_modules\vinext\dist\cli.js'

New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $vinextCli -PathType Leaf)) {
  Write-Host 'Reparando dependencias locales despues del cambio de carpeta...'
  & $PnpmPath install --frozen-lockfile --force
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $vinextCli -PathType Leaf)) {
    throw 'No se pudieron reparar las dependencias locales de StatsMLB.'
  }
}

& $PnpmPath run build
if ($LASTEXITCODE -ne 0) {
  throw "La compilacion local fallo con codigo $LASTEXITCODE."
}

$ownedProcesses = @()
if (Test-Path -LiteralPath $pidFile) {
  $savedPid = 0
  [void][int]::TryParse((Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1), [ref]$savedPid)
  if ($savedPid -gt 0) { $ownedProcesses += Get-CimInstance Win32_Process -Filter "ProcessId = $savedPid" -ErrorAction SilentlyContinue }
}
$listeners = Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
  $candidate = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction SilentlyContinue
  if ($candidate -and $candidate.CommandLine -like "*$AppDirectory*" -and $candidate.CommandLine -like '*vinext*') {
    $ownedProcesses += $candidate
  } elseif ($candidate) {
    throw "El puerto 4173 esta ocupado por otro proceso (PID $($candidate.ProcessId))."
  }
}
$ownedProcesses | Sort-Object ProcessId -Unique | ForEach-Object {
  Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

$process = Start-Process -FilePath $nodeExecutable -ArgumentList @("`"$vinextCli`"", 'dev', '--host', '0.0.0.0', '--port', '4173') -WorkingDirectory $AppDirectory -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id
