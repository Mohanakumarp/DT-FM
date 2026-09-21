[CmdletBinding()]
param(
    [string]$PythonPath,
    [ValidateRange(8, 300000)][int]$Rows = 4096,
    [switch]$SkipData,
    [switch]$SkipSmoke
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LabPython = Join-Path $RepoRoot '.venv-lab\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:USE_LIBUV = '0'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if (-not (Test-Path -LiteralPath $LabPython)) {
    if (-not $PythonPath) {
        $PythonPath = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
        if (-not (Test-Path -LiteralPath $PythonPath)) {
            $Installer = Join-Path $env:TEMP 'dtfm-python-3.12.10-amd64.exe'
            Write-Host 'Installing Python 3.12.10 for this user. No administrator account is required.'
            Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $Installer
            $Signature = Get-AuthenticodeSignature -LiteralPath $Installer
            if ($Signature.Status -ne 'Valid' -or $Signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
                throw 'Python installer signature verification failed.'
            }
            $Process = Start-Process -FilePath $Installer -ArgumentList '/quiet', 'InstallAllUsers=0', 'Include_launcher=0', 'Include_test=0', 'PrependPath=0' -WindowStyle Hidden -Wait -PassThru
            if ($Process.ExitCode -notin @(0, 3010)) { throw "Python installation failed: $($Process.ExitCode)" }
        }
    }
    & $PythonPath -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Use 64-bit Python 3.12'"
    if ($LASTEXITCODE -ne 0) { throw 'A working 64-bit Python 3.12 is required. Use -PythonPath if it is installed elsewhere.' }
    & $PythonPath -m venv (Join-Path $RepoRoot '.venv-lab')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv-lab.' }
}
& $LabPython -m pip install --disable-pip-version-check 'torch==2.13.0' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed. Check internet/proxy access.' }
& $LabPython -m pip install --disable-pip-version-check 'numpy==2.2.6' 'six==1.17.0' 'psutil==7.0.0' 'matplotlib==3.10.3'
if ($LASTEXITCODE -ne 0) { throw 'Lab dependencies could not be installed.' }
if (-not $SkipData) {
    & $LabPython (Join-Path $PSScriptRoot 'lab_data.py') --rows $Rows
    if ($LASTEXITCODE -ne 0) { throw 'QQP download/preparation failed. Rerun setup after checking internet access.' }
}
if (-not $SkipSmoke) {
    & (Join-Path $PSScriptRoot 'run_cpu_smoke.ps1') -PythonPath $LabPython -TwoRanks -Iterations 2
    if ($LASTEXITCODE -ne 0) { throw 'Local training smoke test failed.' }
}
Write-Host 'Setup complete. See LAB_DEMO.md. Start rank 0 first; it prints the commands for the other computers.'
