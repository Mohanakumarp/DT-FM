[CmdletBinding()]
param(
    [string]$VenvPath,
    [switch]$SkipSmoke
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
if (-not $VenvPath) {
    $VenvPath = Join-Path $RepoRoot '.venv-cpu'
}
$PythonPath = Join-Path $VenvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonPath)) {
    python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the CPU virtual environment.'
    }
}

& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip in the CPU environment.' }
& $PythonPath -m pip install --index-url https://download.pytorch.org/whl/cpu torch
if ($LASTEXITCODE -ne 0) { throw 'Failed to install CPU PyTorch.' }
& $PythonPath -m pip install -r (Join-Path $RepoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install the repository requirements.' }

& $PythonPath -c "import torch; print('PyTorch:', torch.__version__); print('CPU threads:', torch.get_num_threads())"
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch verification failed.' }

if (-not $SkipSmoke) {
    & (Join-Path $ScriptDir 'run_cpu_smoke.ps1') -PythonPath $PythonPath
}

Write-Host "CPU environment ready: $VenvPath"
