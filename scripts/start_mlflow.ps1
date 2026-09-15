[CmdletBinding()]
param([int]$Port = 5000)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    & "$RepoRoot/.venv-cpu/Scripts/python.exe" -m mlflow server `
        --host 127.0.0.1 --port $Port `
        --backend-store-uri sqlite:///mlflow.db `
        --artifacts-destination ./mlartifacts
    if ($LASTEXITCODE -ne 0) { throw 'MLflow server failed' }
} finally { Pop-Location }
