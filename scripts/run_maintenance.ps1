param(
    [ValidateSet('smoke', 'local', 'publication', 'scenarios')]
    [string]$Preset = 'smoke',
    [Parameter(Mandatory=$true)]
    [string]$RunId,
    [int]$Workers = 2
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$PythonExecutable = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $PythonExecutable)) {
    throw 'Cree .venv e instale el proyecto según docs/VS_CODE_WINDOWS.md.'
}
if ($Preset -eq 'scenarios') {
    & $PythonExecutable -m agriopt maintenance scenarios --config configs/maintenance_publication.yaml --run-id $RunId
} else {
    & $PythonExecutable -m agriopt maintenance run --config "configs/maintenance_$Preset.yaml" --run-id $RunId --workers $Workers
}
exit $LASTEXITCODE
