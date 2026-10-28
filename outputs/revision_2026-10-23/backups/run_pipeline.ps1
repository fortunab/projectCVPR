# PowerShell Pipeline Runner for CBMP Medical Vision Project
# Usage examples:
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Step dissection
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Step level1
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Step level2
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Step all

param(
    [ValidateSet("dissection", "level1", "level2", "reports", "all")]
    [string]$Step = "all"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = Get-Location }

# Determine Python interpreter
$PythonExe = Join-Path $RepoRoot ".venv-cbmp\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "C:\Users\Lenovo\anaconda3\python.exe"
}
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command "python").Source
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Running CBMP Pipeline Step: $Step" -ForegroundColor Cyan
Write-Host "Interpreter: $PythonExe" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$Bundle = Join-Path $RepoRoot "isolated_datasets\polypgen_extensions_2026-10-20\reproducibility"
$DissectionDir = Join-Path $RepoRoot "isolated_datasets\polypgen_extensions_2026-10-20\cvpr_dissection"

# 1. Dissection Step
if ($Step -eq "dissection" -or $Step -eq "all") {
    Write-Host "`n>>> [1/3] Running CVPR Dissection (Prototype Purity & Activation)..." -ForegroundColor Green
    & $PythonExe (Join-Path $DissectionDir "run_cvpr_dissection.py")
    
    Write-Host ">>> Generating t-SNE 2D Latent Geometry Visualizations..." -ForegroundColor Green
    & $PythonExe (Join-Path $DissectionDir "generate_geometry_plots.py")
    Write-Host "Dissection artifacts saved in: $DissectionDir\figures" -ForegroundColor Yellow
}

# 2. Level 1 Metrics Step
if ($Step -eq "level1" -or $Step -eq "all") {
    Write-Host "`n>>> [2/3] Running Level 1 Prediction Metrics Verification..." -ForegroundColor Green
    
    $PolyOut = Join-Path $Bundle "report_polypgen_l1"
    if (Test-Path $PolyOut) { Remove-Item -Recurse -Force $PolyOut }
    & $PythonExe -B (Join-Path $Bundle "metrics_from_predictions.py") `
        --input "17=$(Join-Path $RepoRoot 'isolated_datasets\polypgen_v3_copy_2026-10-17\results\seed17\predictions.csv')" `
        --input "29=$(Join-Path $RepoRoot 'isolated_datasets\polypgen_v3_copy_2026-10-17\results\seed29\predictions.csv')" `
        --input "43=$(Join-Path $RepoRoot 'isolated_datasets\polypgen_v3_copy_2026-10-17\results\seed43\predictions.csv')" `
        --output-dir $PolyOut
    Write-Host "PolypGen Level 1 report saved in: $PolyOut" -ForegroundColor Yellow

    $CamOut = Join-Path $Bundle "report_camelyon_l1"
    if (Test-Path $CamOut) { Remove-Item -Recurse -Force $CamOut }
    & $PythonExe -B (Join-Path $Bundle "metrics_from_predictions.py") `
        --input (Join-Path $RepoRoot "isolated_datasets\polypgen_extensions_2026-10-20\exp5_camelyon\evaluation\predictions.csv") `
        --output-dir $CamOut
    Write-Host "CAMELYON17 Level 1 report saved in: $CamOut" -ForegroundColor Yellow
}

# 3. Level 2 Fitting Step
if ($Step -eq "level2" -or $Step -eq "all") {
    Write-Host "`n>>> [3/3] Running Level 2 Head Fitting on Cached Features (Seed 17)..." -ForegroundColor Green
    
    $FitOut = Join-Path $Bundle "check_seed17"
    if (Test-Path $FitOut) { Remove-Item -Recurse -Force $FitOut }
    & $PythonExe -B (Join-Path $Bundle "fit_cached_heads.py") `
        --manifest (Join-Path $RepoRoot "isolated_datasets\polypgen_v3_copy_2026-10-17\audit\manifest.csv") `
        --features (Join-Path $RepoRoot "isolated_datasets\polypgen_v3_copy_2026-10-17\features\resnet18_imagenet1k_v1.npz") `
        --seed 17 `
        --allow-version-drift `
        --output-dir $FitOut
    Write-Host "Fitted head checks saved in: $FitOut" -ForegroundColor Yellow
}

# 4. Final Reports Step
if ($Step -eq "reports" -or $Step -eq "all") {
    Write-Host "`n>>> Finalizing Extensions and Checking File Preservation..." -ForegroundColor Green
    & $PythonExe (Join-Path $RepoRoot "isolated_datasets\polypgen_extensions_2026-10-20\finalize_extensions.py")
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "Pipeline execution completed successfully!" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
