# Walk-forward EMPIRICAL BUCKET calibrator.
# 4-dim bucket: side (home/away) | top1-conf-bucket | scAgree (y/n) | scMag (w/m/s).
# Requires public/data/strikecast-history.json (see scripts/export_strikecast_history.py).
# Falls back gracefully for games without StrikeCast data (they use fewer dimensions).

param(
  [int]$MinBucketN = 40,
  [string]$OutPath = 'public/data/stacked-calibrator.json'
)

$ErrorActionPreference = 'Stop'

$scPath = 'public/data/strikecast-history.json'
$scMap = @{}
if (Test-Path $scPath) {
  $sc = Get-Content $scPath -Raw | ConvertFrom-Json
  foreach ($p in $sc.games.PSObject.Properties) { $scMap[$p.Name] = $p.Value }
  Write-Host ("Loaded StrikeCast history: {0} entries" -f $sc.count)
} else {
  Write-Host "WARN: strikecast-history.json missing - falling back to 3-dim buckets"
}

Write-Host 'Loading walkforward data...'
$games = New-Object System.Collections.Generic.List[object]
foreach ($y in 2024,2025,2026) {
  $d = Get-Content "public/data/walkforward-$y.json" -Raw | ConvertFrom-Json
  foreach ($p in $d.days.PSObject.Properties) {
    foreach ($g in $p.Value.games) {
      if ($g.actualWinner -eq $null -or -not $g.maskProbabilitiesEncoded) { continue }
      $bytes = [System.Convert]::FromBase64String($g.maskProbabilitiesEncoded)
      [double]$pTop1 = $bytes[1019]/256.0 + 1.0/512.0
      if ($pTop1 -ge 0.5) { $side = 'home'; $conf = $pTop1; $pickTeam = $g.home }
      else                { $side = 'away'; $conf = 1-$pTop1; $pickTeam = $g.away }
      $c = $conf*100
      if ($c -lt 55) { $cb='50-55' } elseif ($c -lt 60) { $cb='55-60' } elseif ($c -lt 65) { $cb='60-65' } elseif ($c -lt 70) { $cb='65-70' } else { $cb='70+' }

      $scRaw = $null
      $rec = $scMap[[string]$g.gamePk]
      if ($rec -and $rec.pRaw -ne $null) { $scRaw = [double]$rec.pRaw }

      if ($scRaw -ne $null) {
        $scSide = if ($scRaw -ge 0.5) {'home'} else {'away'}
        $agree = if ($scSide -eq $side) {'y'} else {'n'}
        $scPickProb = if ($side -eq 'home') { $scRaw } else { 1 - $scRaw }
        $scMag = if ($scPickProb -lt 0.45) {'w'} elseif ($scPickProb -lt 0.55) {'m'} else {'s'}
        $bucket = "$side|$cb|$agree|$scMag"
        $bucketDim = 4
      } else {
        # Fallback 2-dim when StrikeCast missing
        $bucket = "$side|$cb|nosc"
        $bucketDim = 2
      }

      $games.Add([pscustomobject]@{
        gamePk = $g.gamePk
        foldMonth = $g.foldMonth
        season = $g.foldMonth.Substring(0,4)
        pickWon = ($pickTeam -eq $g.actualWinner)
        side = $side; confBucket = $cb
        bucket = $bucket; bucketDim = $bucketDim
        pTop1 = $pTop1; scRaw = $scRaw
      })
    }
  }
}
Write-Host ("Loaded {0} games ({1} with StrikeCast)" -f $games.Count, (($games | Where-Object { $_.scRaw -ne $null }).Count))

$byMonth = $games | Group-Object foldMonth | Sort-Object Name
$oos = New-Object System.Collections.Generic.List[object]
for ($i = 3; $i -lt $byMonth.Count; $i++) {
  $test = $byMonth[$i]
  $bt=@{}; $bw=@{}
  for ($j = 0; $j -lt $i; $j++) {
    foreach ($ex in $byMonth[$j].Group) {
      $k = $ex.bucket
      if (-not $bt.ContainsKey($k)) { $bt[$k]=0; $bw[$k]=0 }
      $bt[$k]++
      if ($ex.pickWon) { $bw[$k]++ }
    }
  }
  foreach ($ex in $test.Group) {
    $n = $bt[$ex.bucket]; $w = $bw[$ex.bucket]
    if ($n -ne $null -and $n -ge $MinBucketN) { $rate = $w/$n } else { $rate = $null }
    if ($rate -eq $null) {
      $tier = 'unavailable'
    } elseif ($rate -ge 0.65) { $tier = 'lock' }
    elseif ($rate -ge 0.60)  { $tier = 'strong' }
    elseif ($rate -ge 0.55)  { $tier = 'play' }
    else                     { $tier = 'playB' }
    $oos.Add([pscustomobject]@{
      season = $ex.season; bucket = $ex.bucket; bucketDim = $ex.bucketDim
      tier = $tier; won = $ex.pickWon; hasCal = ($rate -ne $null)
    })
  }
}

Write-Host "`n=== WALK-FORWARD OOS by tier ==="
"{0,-14} {1,6} {2,6} {3,7}" -f 'tier','n','wins','acc%'
foreach ($t in 'lock','strong','play','playB','unavailable') {
  $slice = $oos | Where-Object tier -eq $t
  if ($slice.Count -eq 0) { continue }
  $w = ($slice | Where-Object won).Count
  "{0,-14} {1,6} {2,6} {3,7:N2}" -f $t, $slice.Count, $w, ($w/$slice.Count*100)
}

# Per-season summary
Write-Host "`n=== 2026 OOS tier breakdown ==="
foreach ($t in 'lock','strong','play','playB','unavailable') {
  $slice = $oos | Where-Object { $_.tier -eq $t -and $_.season -eq '2026' }
  if ($slice.Count -eq 0) { continue }
  $w = ($slice | Where-Object won).Count
  "  {0,-12} n={1,4}  acc={2:N2}%" -f $t, $slice.Count, ($w/$slice.Count*100)
}

# Fit final bucket table on ALL games
$bucketWins = @{}; $bucketTot = @{}
foreach ($ex in $games) {
  $k = $ex.bucket
  if (-not $bucketTot.ContainsKey($k)) { $bucketTot[$k] = 0; $bucketWins[$k] = 0 }
  $bucketTot[$k]++
  if ($ex.pickWon) { $bucketWins[$k]++ }
}
$buckets = @()
foreach ($k in $bucketTot.Keys | Sort-Object) {
  $buckets += [pscustomobject]@{ key=$k; n=$bucketTot[$k]; wins=$bucketWins[$k]; winRate=[math]::Round($bucketWins[$k]/$bucketTot[$k],4) }
}
$oosCount = $oos.Count
$oosWins  = ($oos | Where-Object won).Count
$payload = [pscustomobject]@{
  generatedAt = (Get-Date).ToString('o')
  method = 'empirical-bucket walk-forward, side|conf|scAgree|scMag (fallback side|conf when no SC)'
  oosGames = $oosCount
  oosAccuracy = [math]::Round($oosWins/$oosCount, 6)
  minBucketN = $MinBucketN
  buckets = $buckets
}
$payload | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path $OutPath
Write-Host ("`nWrote {0} ({1} buckets)" -f $OutPath, $buckets.Count)
