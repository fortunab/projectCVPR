param([string]$PythonExecutable = 'python', [string]$VenvPath, [switch]$SkipInstall)
$ErrorActionPreference = 'Stop'
if (-not $VenvPath) { $VenvPath = Join-Path $PSScriptRoot '.venv-cbmp' }
$EnvDir = Join-Path $PSScriptRoot 'isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment'
& $PythonExecutable -c 'import sys; sys.exit(sys.version_info[:2] != (3,12))'
if ($LASTEXITCODE -ne 0) { throw 'Choose a Python 3.12 interpreter with -PythonExecutable.' }
if (-not (Test-Path -LiteralPath $VenvPath)) {
    & $PythonExecutable -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$VenvPython = Join-Path $VenvPath 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) { throw 'Missing venv interpreter.' }
if (-not $SkipInstall) {
    $PipCommands = @(
        @('-r', "$EnvDir/requirements-installer.txt"),
        @('--index-url','https://download.pytorch.org/whl/cpu','-c',"$EnvDir/constraints-observed.txt",'-r',"$EnvDir/requirements-torch-cpu.txt"),
        @('--index-url','https://pypi.org/simple','-c',"$EnvDir/constraints-observed.txt",'-r',"$EnvDir/requirements-common.txt")
    )
    foreach ($PipArgs in $PipCommands) {
        & $VenvPython -m pip install --no-cache-dir @PipArgs
        if ($LASTEXITCODE -ne 0) { throw 'Pinned installation failed; stop before experiments.' }
    }
}
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check failed.' }
& $VenvPython -B (Join-Path $PSScriptRoot 'scripts/check_environment.py')
if ($LASTEXITCODE -ne 0) { throw 'Package versions differ from the reference.' }
Write-Host "Ready: $VenvPython"
