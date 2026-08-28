[CmdletBinding()]
param(
    [string]$PythonPath = 'python',
    [int]$Iterations = 1
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

& $PythonPath -c "import torch; assert hasattr(torch, 'xpu'), 'This PyTorch build has no XPU support'; assert torch.xpu.is_available(), 'No supported Intel GPU is available'; x = torch.randn((32, 32), device='xpu', requires_grad=True); loss = (x @ x).square().mean(); loss.backward(); torch.xpu.synchronize(); assert x.grad is not None; print('PyTorch:', torch.__version__); print('XPU:', torch.xpu.get_device_name(0)); print('XPU preflight loss:', loss.item())"
if ($LASTEXITCODE -ne 0) {
    throw 'Intel XPU matrix multiplication and backward preflight failed.'
}

Push-Location $RepoRoot
try {
    & $PythonPath -u dist_runner.py `
        --device xpu `
        --world-size 1 `
        --pipeline-group-size 1 `
        --data-group-size 1 `
        --rank 0 `
        --synthetic-data true `
        --synthetic-samples 8 `
        --synthetic-vocab-size 512 `
        --seq-length 32 `
        --embedding-dim 64 `
        --num-layers 1 `
        --num-heads 4 `
        --batch-size 2 `
        --micro-batch-size 1 `
        --num-iters $Iterations `
        --metrics-dir ./logs/smoke `
        --skip-comm-probe true `
        --use-offload false `
        --profiling no-profiling
    if ($LASTEXITCODE -ne 0) {
        throw "Intel XPU smoke test failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
