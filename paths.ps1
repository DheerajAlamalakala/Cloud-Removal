# Shared local paths for the Windows launchers. Environment variables override
# machine-specific values so the repository can be moved or cloned elsewhere.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '.')).Path
$DataRoot    = if ($env:SEN12MSCR_DATA_ROOT) { $env:SEN12MSCR_DATA_ROOT } else { Join-Path $ProjectRoot "data\sen12mscr" }
$Manifest    = if ($env:SEN12MSCR_MANIFEST) { $env:SEN12MSCR_MANIFEST } else { Join-Path $ProjectRoot "splits\sen12mscr_spring_clean.json" }

# Optional: paste an exact checkpoint path here to override auto-detection.
$CheckpointOverride = ""

$ok = $true
foreach ($p in @($ProjectRoot, $DataRoot, $Manifest)) {
    if (Test-Path $p) { Write-Host "OK       $p" -ForegroundColor Green }
    else { Write-Host "MISSING  $p" -ForegroundColor Red; $ok = $false }
}

$found = @(
    Get-ChildItem (Join-Path $ProjectRoot "results\sen12mscr") -Recurse -Filter best_ema.pt -ErrorAction SilentlyContinue
    Get-ChildItem (Join-Path $ProjectRoot "runs") -Recurse -Filter best_ema.pt -ErrorAction SilentlyContinue
)
Write-Host "`nCheckpoints found:"
$found | ForEach-Object { Write-Host "  $($_.FullName)  ($([math]::Round($_.Length/1MB)) MB)" }

if ($env:SEN12MSCR_CHECKPOINT) { $CheckpointOverride = $env:SEN12MSCR_CHECKPOINT }
if ($CheckpointOverride -ne "") { $Checkpoint = $CheckpointOverride }
else {
    $preferredPaths = @(
        (Join-Path $ProjectRoot "checkpoints_for_google_drive\fine_tunes\spring_finetune_lr2e5\best_ema.pt"),
        (Join-Path $ProjectRoot "results\sen12mscr\training\best_ema.pt")
    )
    $preferred = $preferredPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    $pick = if ($preferred) { Get-Item $preferred } else { $found | Sort-Object FullName | Select-Object -First 1 }
    $Checkpoint = if ($pick) { $pick.FullName } else { "" }
}
if ($Checkpoint -eq "" -or -not (Test-Path $Checkpoint)) { Write-Host "No checkpoint selected. Set `$CheckpointOverride above." -ForegroundColor Red; $ok = $false }
else { Write-Host "`nUsing checkpoint: $Checkpoint" -ForegroundColor Cyan }

$env:DEMO_DATA = $DataRoot
$env:DEMO_MANIFEST = $Manifest
$env:DEMO_CKPT = $Checkpoint
$preferredCsv = Join-Path $ProjectRoot "results\sen12mscr\final_comparison\per_image_test.csv"
$csv = if (Test-Path $preferredCsv) { Get-Item $preferredCsv } else {
    @(Get-ChildItem (Join-Path $ProjectRoot "results\sen12mscr") -Recurse -Filter per_image_test.csv -ErrorAction SilentlyContinue) +
    @(Get-ChildItem (Join-Path $ProjectRoot "runs") -Recurse -Filter per_image_test.csv -ErrorAction SilentlyContinue) |
        Sort-Object FullName | Select-Object -First 1
}
if ($csv) { $env:DEMO_CSV = $csv.FullName }
if (-not $ok) { Write-Host "`nFix the red lines above, then run again." -ForegroundColor Yellow }
$script:PathsOK = $ok
