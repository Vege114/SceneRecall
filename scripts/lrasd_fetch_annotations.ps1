param(
  [string]$Manifest = 'D:\SceneRecall\training\lrasd\data\annotation-downloads.private.json',
  [string]$OutputDirectory = 'D:\SceneRecall\training\lrasd\data\annotations'
)
$ErrorActionPreference = 'Stop'
$rows = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$results = $rows | ForEach-Object -ThrottleLimit 4 -Parallel {
  $row = $_
  $ErrorActionPreference = 'Stop'
  if ($row.clip -notmatch '^[A-Za-z0-9_-]{11}$') { throw 'Invalid clip identifier' }
  $target = Join-Path $using:OutputDirectory ($row.clip + '.json')
  $temporary = $target + '.partial'
  $success = $false
  $problem = ''
  for ($attempt = 1; $attempt -le 2; $attempt++) {
    try {
      if (-not (Test-Path -LiteralPath $target) -or (Get-Item -LiteralPath $target).Length -ne $row.bytes) {
        Invoke-WebRequest -Uri $row.download_url -OutFile $temporary -TimeoutSec 60
        if ((Get-Item -LiteralPath $temporary).Length -ne $row.bytes) { throw 'Downloaded size does not match metadata' }
        Get-Content -LiteralPath $temporary -Raw | ConvertFrom-Json -AsHashtable | Out-Null
        Move-Item -LiteralPath $temporary -Destination $target -Force
      }
      $hash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
      $success = $true
      break
    } catch { $problem = $_.Exception.Message }
  }
  [pscustomobject]@{ clip=$row.clip; movie=$row.movie; year=$row.year; bytes=$row.bytes;
    source_url=$row.source_url; drive_id=$row.drive_id; path=$target; sha256=$hash;
    downloaded=$success; error=if ($success) { $null } else { $problem } }
}
$results | Sort-Object movie,clip | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'sources.json') -Encoding utf8
$results | Group-Object downloaded | Select-Object Name,Count | Format-Table -AutoSize
if ($results.Where({ -not $_.downloaded }).Count) { exit 1 }
