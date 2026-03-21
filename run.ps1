$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvPython)) {
    Write-Host "[INFO] Creating virtual environment..."
    python -m venv .venv
}

Write-Host "[INFO] Activating virtual environment..."
& $venvActivate

Write-Host "[INFO] Installing/updating dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host "[INFO] Starting app at http://127.0.0.1:8000"
& $venvPython app.py
