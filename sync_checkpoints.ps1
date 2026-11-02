$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path $PSScriptRoot).Path
$HandoffRoot = Join-Path $ProjectRoot 'checkpoints_for_google_drive'
$FineTuneSource = Join-Path $ProjectRoot 'results\sen12mscr\training'
$LegacyFineTuneSource = Join-Path $ProjectRoot 'runs\spring_finetune_lr2e5'
$FineTuneDest = Join-Path $HandoffRoot 'fine_tunes\spring_finetune_lr2e5'

$copies = @(
    [pscustomobject]@{ Source = Join-Path $FineTuneSource 'best_ema.pt'; Destination = Join-Path $FineTuneDest 'best_ema.pt' },
    [pscustomobject]@{ Source = Join-Path $FineTuneSource 'last.pt'; Destination = Join-Path $FineTuneDest 'last.pt' },
    [pscustomobject]@{ Source = Join-Path $FineTuneSource 'log.csv'; Destination = Join-Path $FineTuneDest 'log.csv' },
    [pscustomobject]@{ Source = Join-Path $ProjectRoot 'splits\sen12mscr_spring_clean.json'; Destination = Join-Path $FineTuneDest 'split_manifest_eval_clean.json' }
)

if (-not (Test-Path -LiteralPath (Join-Path $FineTuneSource 'best_ema.pt') -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $LegacyFineTuneSource 'best_ema.pt') -PathType Leaf)) {
    $FineTuneSource = $LegacyFineTuneSource
}

foreach ($pair in $copies) {
    if (-not (Test-Path -LiteralPath $pair.Source -PathType Leaf)) {
        throw "Required source is missing: $($pair.Source)"
    }
}
New-Item -ItemType Directory -Force -Path $FineTuneDest | Out-Null
foreach ($pair in $copies) {
    Copy-Item -LiteralPath $pair.Source -Destination $pair.Destination -Force
}

Write-Host "Checkpoint handoff refreshed: $HandoffRoot"
Get-ChildItem -LiteralPath $HandoffRoot -Recurse -File |
    Select-Object @{Name='File';Expression={$_.FullName.Substring($HandoffRoot.Length + 1)}},
                  @{Name='SizeMB';Expression={[math]::Round($_.Length / 1MB, 2)}} |
    Format-Table -AutoSize
