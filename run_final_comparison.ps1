param(
    [string]$BaselineCheckpoint,
    [string]$CandidateCheckpoint,
    [string]$OutputDirectory = '.\results\sen12mscr\final_comparison'
)

$ErrorActionPreference = 'Stop'

# Run from any directory; all paths resolve from this script's repository root.
$root = (Resolve-Path $PSScriptRoot).Path
Set-Location $root
. (Join-Path $root 'paths.ps1')
$required = @('src\sen12mscr\train.py', 'src\sen12mscr\dataset.py', 'src\sen12mscr\model.py', 'src\sen12mscr\losses.py')
foreach ($file in $required) {
    if (-not (Test-Path (Join-Path $root $file))) {
        throw "Missing $file. Open PowerShell in the SEN12MSCR_project_adapted folder and run this script again."
    }
}

$compareScript = Join-Path $root 'benchmarks\compare_models.py'
if (-not (Test-Path $compareScript)) { throw 'benchmarks\compare_models.py is missing.' }

if (-not $BaselineCheckpoint) {
    $options = @(
        (Join-Path $root 'checkpoints_for_google_drive\best_models\baseline_best_ema_epoch06.pt'),
        (Join-Path $root 'results\sen12mscr\baseline\best_ema.pt')
    )
    $BaselineCheckpoint = $options | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $CandidateCheckpoint) {
    $options = @(
        (Join-Path $root 'checkpoints_for_google_drive\best_models\spectral_weighted_best_ema_epoch03.pt'),
        (Join-Path $root 'results\sen12mscr\candidate\best_ema.pt')
    )
    $CandidateCheckpoint = $options | Where-Object { Test-Path $_ } | Select-Object -First 1
}
foreach ($p in @($BaselineCheckpoint, $CandidateCheckpoint)) {
    if (-not $p -or -not (Test-Path -LiteralPath $p -PathType Leaf)) {
        throw "Checkpoint not found. Pass -BaselineCheckpoint and -CandidateCheckpoint with two compatible local SEN12MS-CR checkpoint paths."
    }
}

# Prefer the clean manifest already used in earlier test evaluation.
$manifestCandidates = @((Join-Path $root 'splits\sen12mscr_spring_clean.json'))
$manifest = $manifestCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $manifest) {
    $manifest = Get-ChildItem (Join-Path $root 'splits') -Recurse -File -Filter '*clean*.json' -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $manifest) {
    throw 'No clean manifest found under splits. The comparison has NOT been started; do not use the unfiltered manifest because it contains the known corrupt SAR sample.'
}

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = (Get-Command python -ErrorAction Stop).Source }
if (-not (Test-Path $DataRoot)) { throw "SEN12MS-CR data root not found: $DataRoot. Set SEN12MSCR_DATA_ROOT first." }

Write-Host "Project:          $root"
Write-Host "Python:           $python"
Write-Host "Manifest:         $manifest"
Write-Host "Baseline:         $BaselineCheckpoint"
Write-Host "Candidate:        $CandidateCheckpoint"
Write-Host 'Starting paired test evaluation...'

& $python $compareScript `
    --data_root $DataRoot `
    --manifest $manifest `
    --baseline_ckpt $BaselineCheckpoint `
    --spectral_ckpt $CandidateCheckpoint `
    --split test `
    --batch_size 4 `
    --num_workers 8 `
    --amp bf16 `
    --save_visuals 12 `
    --out $OutputDirectory

if ($LASTEXITCODE -ne 0) { throw "Comparison failed with exit code $LASTEXITCODE" }
Write-Host "Comparison completed. Results are in $OutputDirectory."
