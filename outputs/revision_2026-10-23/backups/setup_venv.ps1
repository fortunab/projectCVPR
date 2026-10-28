# PowerShell Virtual Environment Setup for CBMP Medical Vision Project
# Usage: powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Setting up Python 3.12 Virtual Environment for CBMP Project" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = Get-Location }

$VenvDir = Join-Path $RepoRoot ".venv-cbmp"
$EnvDir = Join-Path $RepoRoot "isolated_datasets\polypgen_extensions_2026-10-20\reproducibility\environment"

# 1. Locate Python 3.12 interpreter
$PythonExe = $null
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $ver = & python --version 2>&1
    if ($ver -match "3\.12") {
        $PythonExe = (Get-Command "python").Source
    }
}
if (-not $PythonExe -and (Test-Path "C:\Users\Lenovo\anaconda3\python.exe")) {
    $PythonExe = "C:\Users\Lenovo\anaconda3\python.exe"
}
if (-not $PythonExe) {
    Write-Error "Python 3.12 was not found. Please install Python 3.12 or add it to PATH."
    exit 1
}

Write-Host "[1/5] Using base Python: $PythonExe" -ForegroundColor Green

# 2. Create virtual environment
if (Test-Path $VenvDir) {
    Write-Host "[2/5] Existing venv found at $VenvDir" -ForegroundColor Yellow
} else {
    Write-Host "[2/5] Creating new virtual environment at $VenvDir..." -ForegroundColor Green
    & $PythonExe -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Failed to find venv python at $VenvPython"
    exit 1
}

# 3. Upgrade installer tools
Write-Host "[3/5] Upgrading installer tools (pip, setuptools, wheel)..." -ForegroundColor Green
& $VenvPython -m pip install --upgrade pip setuptools wheel

# 4. Install PyTorch CPU and scientific stack
Write-Host "[4/5] Installing PyTorch CPU and dependencies..." -ForegroundColor Green
& $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
& $VenvPython -m pip install -r "$EnvDir\requirements-common.txt"

# 5. Consistency check and verification tests
Write-Host "[5/5] Checking environment consistency..." -ForegroundColor Green
& $VenvPython -m pip check

$TestScript = Join-Path $RepoRoot "isolated_datasets\polypgen_extensions_2026-10-20\reproducibility\test_helpers.py"
if (Test-Path $TestScript) {
    Write-Host "Running portability helper tests..." -ForegroundColor Green
    & $VenvPython -B $TestScript
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Virtual environment is successfully set up and ready!" -ForegroundColor Cyan
Write-Host "To activate in PowerShell:" -ForegroundColor Yellow
Write-Host "    .\.venv-cbmp\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "Or call python directly without activation:" -ForegroundColor Yellow
Write-Host "    .\.venv-cbmp\Scripts\python.exe" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
