# Compatibility path retained for existing SEN12MS-CR demo instructions.
& (Join-Path $PSScriptRoot '..\inference\run_demo.ps1') @args
exit $LASTEXITCODE
