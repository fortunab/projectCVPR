param(
    [ValidateSet('verify','reports','level1','level2','all')][string]$Step = 'verify',
    [string]$PythonExecutable, [string]$OutputDir,
    [ValidateSet(17,29,43)][int]$Seed = 17,
    [ValidateSet('resnet18','dinov2')][string]$Encoder = 'resnet18',
    [switch]$AllowVersionDrift
)
$ErrorActionPreference = 'Stop'
if (-not $PythonExecutable) {
    if ($env:VIRTUAL_ENV) { $PythonExecutable = Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe' }
    elseif ($env:CONDA_PREFIX) { $PythonExecutable = Join-Path $env:CONDA_PREFIX 'python.exe' }
    else { $PythonExecutable = Join-Path $PSScriptRoot '.venv-cbmp/Scripts/python.exe' }
}
if (-not (Test-Path -LiteralPath $PythonExecutable)) { throw 'Select an existing environment with -PythonExecutable.' }
$RunArgs = @('-B', (Join-Path $PSScriptRoot 'scripts/run_project.py'), $Step, '--seed', "$Seed", '--encoder', $Encoder)
if ($OutputDir) { $RunArgs += @('--output-dir', $OutputDir) }
if ($AllowVersionDrift) { $RunArgs += '--allow-version-drift' }
& $PythonExecutable @RunArgs
if ($LASTEXITCODE -ne 0) { throw "Experiment/check failed with exit code $LASTEXITCODE. Inspect output; success is not assumed." }
