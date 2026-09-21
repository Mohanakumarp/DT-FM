[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateRange(0,2)][int]$Rank,
    [Parameter(Mandatory=$true)][string]$MasterIP,
    [string]$SessionCode,
    [ValidateRange(2,3)][int]$Computers = 3,
    [ValidateRange(1,5000)][int]$Steps = 60,
    [ValidateRange(0,1000)][int]$Warmup = 5,
    [ValidateRange(1,10)][int]$Repeats = 3,
    [ValidateRange(0,128)][int]$Threads = 2,
    [ValidateRange(3,120)][int]$Layers = 6,
    [ValidateRange(1,65535)][int]$ControlPort = 8765,
    [ValidateRange(1,65535)][int]$TrainPort = 29500,
    [ValidateRange(30,86400)][int]$Timeout = 1800,
    [switch]$Synthetic,
    [switch]$Larger,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) { $PythonPath = Join-Path $RepoRoot '.venv-lab\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Run scripts\setup_lab.ps1 first, or pass -PythonPath.' }
$LaunchArgs = @('-u', (Join-Path $PSScriptRoot 'lab_demo.py'), 'run', '--rank', $Rank,
    '--master', $MasterIP, '--steps', $Steps, '--warmup', $Warmup, '--repeats', $Repeats,
    '--threads', $Threads, '--layers', $Layers, '--control-port', $ControlPort,
    '--train-port', $TrainPort, '--timeout', $Timeout, '--computers', $Computers)
if ($SessionCode) { $LaunchArgs += @('--session-code', $SessionCode) }
if ($Synthetic) { $LaunchArgs += '--synthetic' }
if ($Larger) { $LaunchArgs += '--larger' }
$env:PYTHONUTF8 = '1'
& $PythonPath @LaunchArgs
if ($LASTEXITCODE -ne 0) { throw 'Lab run failed. See the printed result folder and LAB_DEMO.md troubleshooting.' }
