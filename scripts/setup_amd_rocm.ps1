[CmdletBinding()]
param(
    [string]$VenvPath,
    [switch]$SkipSmoke
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
if (-not $VenvPath) {
    $VenvPath = Join-Path $RepoRoot '.venv-amd'
}
$PythonPath = Join-Path $VenvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonPath)) {
    python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the AMD virtual environment.'
    }
}

& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip in the AMD environment.' }
& $PythonPath -m pip install --pre `
    --index-url https://nightly.repo.amd.com/rocm/whl-next/ `
    'torch[device-gfx1152]'
if ($LASTEXITCODE -ne 0) { throw 'Failed to install the AMD ROCm PyTorch package.' }
& $PythonPath -m pip install -r (Join-Path $RepoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Failed to install the repository requirements.' }

& $PythonPath -c "import torch; assert torch.version.hip is not None, 'This is not a ROCm PyTorch build'; assert torch.cuda.is_available(), 'The Radeon GPU is not available'; print('PyTorch:', torch.__version__); print('HIP:', torch.version.hip); print('GPU:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw 'The ROCm PyTorch installation could not enumerate the Radeon GPU.'
}

if (-not $SkipSmoke) {
    & (Join-Path $ScriptDir 'run_amd_smoke.ps1') -PythonPath $PythonPath
}

Write-Host "AMD test environment ready: $VenvPath"
