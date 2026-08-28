[CmdletBinding()]
param(
    [string]$VenvPath,
    [switch]$SkipSmoke
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
if (-not $VenvPath) {
    $VenvPath = Join-Path $RepoRoot '.venv-intel'
}
$PythonPath = Join-Path $VenvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonPath)) {
    python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the Intel XPU virtual environment.'
    }
}

& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip in the Intel environment.' }
& $PythonPath -m pip install `
    --index-url https://download.pytorch.org/whl/xpu `
    torch
if ($LASTEXITCODE -ne 0) { throw 'Failed to install the Intel XPU PyTorch package.' }
& $PythonPath -m pip install -r (Join-Path $RepoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install the repository requirements.' }

& $PythonPath -c "import torch; assert hasattr(torch, 'xpu'), 'This PyTorch build has no XPU support'; assert torch.xpu.is_available(), 'No supported Intel GPU is available. Check the GPU model and Intel driver.'; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw 'The Intel XPU PyTorch installation could not enumerate a supported Intel GPU.'
}

if (-not $SkipSmoke) {
    & (Join-Path $ScriptDir 'run_intel_xpu_smoke.ps1') -PythonPath $PythonPath
}

Write-Host "Intel XPU test environment ready: $VenvPath"
