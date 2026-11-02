param(
    [ValidateSet('rice2', 'sen12mscr')]
    [string]$Dataset = 'rice2',
    [string]$DataRoot,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path $PSScriptRoot).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

Set-Location $ProjectRoot
$ResolvedDataRoot = $DataRoot
if (-not $ResolvedDataRoot) {
    $folder = if ($Dataset -eq 'rice2') { 'rice2' } else { 'sen12mscr' }
    $ResolvedDataRoot = Join-Path $ProjectRoot "data\$folder"
}
& $Python (Join-Path $ProjectRoot 'train.py') --dataset $Dataset --data_root $ResolvedDataRoot @ExtraArgs
exit $LASTEXITCODE
