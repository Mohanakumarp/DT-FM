[CmdletBinding()]
param(
    [string]$PythonPath = 'python',
    [int]$Iterations = 1
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Push-Location $RepoRoot
try {
    & $PythonPath -u dist_runner.py `
        --device rocm `
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
        throw "AMD smoke test failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
