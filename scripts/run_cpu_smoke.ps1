[CmdletBinding()]
param(
    [string]$PythonPath = 'python',
    [switch]$TwoRanks,
    [int]$Iterations = 1,
    [int]$Port = 9021
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$WorldSize = if ($TwoRanks) { 2 } else { 1 }
$SpawnRanks = if ($TwoRanks) { 'true' } else { 'false' }

Push-Location $RepoRoot
try {
    & $PythonPath -u dist_runner.py `
        --device cpu `
        --spawn-local-ranks $SpawnRanks `
        --tensor-comm gloo `
        --dist-backend gloo `
        --dist-url "tcp://127.0.0.1:$Port" `
        --world-size $WorldSize `
        --pipeline-group-size $WorldSize `
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
        throw "CPU smoke test failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
