# Run from anywhere: .\inference\run_demo.ps1 (paths.ps1 sits in the project root)
. "$PSScriptRoot\..\paths.ps1"
if (-not $script:PathsOK) { return }
Set-Location $ProjectRoot
Write-Host "`nOpen http://127.0.0.1:8000 in your browser (Ctrl+C here to stop)`n" -ForegroundColor Cyan
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = (Get-Command python -ErrorAction Stop).Source }
& $python -m uvicorn inference.app:app --host 127.0.0.1 --port 8000
