<#
.SYNOPSIS
  Registra (o reinstala) las tareas programadas de STRIKECAST en el Task
  Scheduler del usuario actual.

.DESCRIPTION
  4 tareas, todas registradas bajo la carpeta \STRIKECAST\:
    - STRIKECAST-Odds        cada 15 min      — pull rapido de odds ESPN
    - STRIKECAST-UpdateData  cada hora         — refresh de datos
    - STRIKECAST-ValuePicks  diario 7:00 AM   — retrain modelo + parquet
    - STRIKECAST-FullRebuild dominical 3:00 AM — reconstruccion pesada

  Es idempotente: borra tareas viejas con el mismo nombre antes de re-registrar,
  asi puedes correrlo tantas veces quieras sin duplicar.

  Requiere PowerShell con permisos para crear scheduled tasks del usuario
  actual (no requiere admin: se crean como "Interactive Token", solo corren
  cuando la sesion esta abierta o el equipo esta prendido).

.EXAMPLE
  # Instalar / reinstalar
  powershell -ExecutionPolicy Bypass -File scripts\setup_scheduled_tasks.ps1

  # Desinstalar
  powershell -ExecutionPolicy Bypass -File scripts\setup_scheduled_tasks.ps1 -Uninstall

.NOTES
  Los logs de cada tarea van a %PROJECT%\logs\<nombre>.log. Si el equipo esta
  dormido a la hora prevista, la tarea corre al despertar (StartWhenAvailable).
#>
[CmdletBinding()]
param(
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'

$Project    = "C:\Users\andra\Desktop\STRIKECAST"
$LogDir     = Join-Path $Project 'logs'
$TaskFolder = '\STRIKECAST\'
$User       = $env:USERNAME

if (-not (Test-Path $Project)) {
    throw "No existe el proyecto en $Project"
}
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

function Remove-TaskIfExists([string]$Name) {
    $existing = Get-ScheduledTask -TaskPath $TaskFolder -TaskName $Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  - Borrando tarea existente $Name..."
        Unregister-ScheduledTask -TaskPath $TaskFolder -TaskName $Name -Confirm:$false
    }
}

function Register-BatTask {
    param(
        [string]$Name,
        [string]$Description,
        [string]$BatRelativePath,
        [Parameter(Mandatory)][ScriptBlock]$TriggerFactory
    )
    Remove-TaskIfExists $Name
    $batAbs = Join-Path $Project $BatRelativePath
    if (-not (Test-Path $batAbs)) { throw "No existe $batAbs" }

    # cmd /c evita que la ventana se quede abierta si el .bat termina con error
    $action  = New-ScheduledTaskAction -Execute 'cmd.exe' `
                                       -Argument "/c `"$batAbs`"" `
                                       -WorkingDirectory $Project
    $triggers = & $TriggerFactory
    $settings = New-ScheduledTaskSettingsSet `
                    -AllowStartIfOnBatteries `
                    -DontStopIfGoingOnBatteries `
                    -StartWhenAvailable `
                    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
                    -MultipleInstances IgnoreNew `
                    -Compatibility Win8
    $principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited

    Register-ScheduledTask -TaskPath $TaskFolder -TaskName $Name `
        -Description $Description -Action $action -Trigger $triggers `
        -Settings $settings -Principal $principal | Out-Null

    Write-Host "  [OK] $Name  ($BatRelativePath)"
}

if ($Uninstall) {
    Write-Host "Desinstalando tareas STRIKECAST..."
    foreach ($n in 'STRIKECAST-Odds','STRIKECAST-UpdateData','STRIKECAST-ValuePicks','STRIKECAST-FullRebuild') {
        Remove-TaskIfExists $n
    }
    # Intentar quitar la carpeta vacia (silencioso si tiene mas tareas)
    try {
        $sched = New-Object -ComObject Schedule.Service
        $sched.Connect()
        $root = $sched.GetFolder('\')
        if (($root.GetFolders(0) | Where-Object { $_.Name -eq 'STRIKECAST' })) {
            $root.DeleteFolder('STRIKECAST', 0)
        }
    } catch { }
    Write-Host "Listo."
    return
}

Write-Host "Instalando tareas programadas STRIKECAST bajo $TaskFolder ..."
Write-Host ""

# Odds ligero: cada 15 minutos entre 7 AM y 11 PM
Register-BatTask -Name 'STRIKECAST-Odds' `
    -Description 'Pull rapido de odds ESPN + refresh de value_picks. Cada 15 min de 7am a 11pm.' `
    -BatRelativePath 'daily_odds_pull.bat' `
    -TriggerFactory {
        $t = New-ScheduledTaskTrigger -Once -At 7:00am `
              -RepetitionInterval (New-TimeSpan -Minutes 15) `
              -RepetitionDuration (New-TimeSpan -Hours 16)
        $t
    }

# Update data mid-weight: cada hora en punto entre 8 AM y 11 PM
Register-BatTask -Name 'STRIKECAST-UpdateData' `
    -Description 'Refresh incremental de datos MLB (14 dias hacia atras). Cada hora.' `
    -BatRelativePath 'update_data.bat' `
    -TriggerFactory {
        $t = New-ScheduledTaskTrigger -Once -At 8:00am `
              -RepetitionInterval (New-TimeSpan -Hours 1) `
              -RepetitionDuration (New-TimeSpan -Hours 15)
        $t
    }

# Value picks retrain: diario 7:00 AM (independiente del cada-15-min)
Register-BatTask -Name 'STRIKECAST-ValuePicks' `
    -Description 'Retrain diario del modelo Value + escribe value_picks.parquet.' `
    -BatRelativePath 'daily_odds_pull.bat' `
    -TriggerFactory {
        New-ScheduledTaskTrigger -Daily -At 7:00am
    }

# Full rebuild histórico: domingos 3:00 AM
Register-BatTask -Name 'STRIKECAST-FullRebuild' `
    -Description 'Reconstruccion pesada semanal (StatsMLB ACTUALIZAR HISTORICO).' `
    -BatRelativePath 'StatsMLB\ACTUALIZAR HISTORICO.bat' `
    -TriggerFactory {
        New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3:00am
    }

Write-Host ""
Write-Host "=== Instalado. Puedes ver o pausar las tareas en:"
Write-Host "    Task Scheduler -> Task Scheduler Library -> STRIKECAST"
Write-Host ""
Write-Host "Para desinstalar todo:"
Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\setup_scheduled_tasks.ps1 -Uninstall"
