# CBMP: cross-center medical recognition experiments

This README documents the verified PolypGen baseline, follow-ups 1-3, DINOv2 experiment 4, and CAMELYON17 experiment 5.

The reference experiments use **Python 3.12.3, torch 2.13.0+cpu, torchvision 0.28.0+cpu, NumPy 1.26.4 and scikit-learn 1.5.1**. Package pins are under `isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment/`. The existing root `venv-cbmp` currently contains torch 2.14.1+cpu; that is a different software condition. A version-drift flag records permission to use different software; it does not establish numerical equivalence.

## Results and provenance for the paper content

The result text, tables, and native TikZ charts in `outputs/introduction.tex` were assembled from the saved experiment reports below. LaTeX does not load these files automatically; the values in the manuscript were transcribed from the reports.

- **PolypGen / ResNet18:** raw per-example predictions are in `isolated_datasets/polypgen_v3_copy_2026-10-17/results/seed{17,29,43}/predictions.csv`. The report summaries are `.../report/summary_metrics.csv` and `.../report/per_center_metrics.csv`; run logs are `.../logs/seed17.log`, `seed29.log`, and `seed43.log`.
- **PolypGen follow-ups:** threshold control: `isolated_datasets/polypgen_followups_2026-10-17/exp1_failure/summary.json`; factorial: `.../exp2_factorial/summary_metrics.csv`; source-composition stress: `.../exp3_stress/summary.csv`. Interpretation and caveats are in the `RESULTS.md` files under each follow-up directory; run logs are `.../exp1_failure/processing.log`, `.../logs/exp2_factorial.log`, and `.../logs/exp3_stress.log`.
- **PolypGen / DINOv2:** predictions are in `isolated_datasets/polypgen_extensions_2026-10-20/exp4_dinov2/main/seed{17,29,43}/predictions.csv`; summaries are in `.../exp4_dinov2/report/summary_metrics.csv` and `per_center_seedmean_metrics.csv`; run logs are in `.../exp4_dinov2/logs/`.
- **CAMELYON17-WILDS custom subset:** predictions are in `isolated_datasets/polypgen_extensions_2026-10-20/exp5_camelyon/evaluation/predictions.csv`; hospital summaries and patient-bootstrap intervals are in `.../report/summary_by_hospital.csv` and `patient_bootstrap_intervals.csv`; the run log is `.../run.log`.

For the final cross-check, see `outputs/revision_2026-10-23/generated_results/verification.json`. It records prediction-file hashes and confirms that point metrics were recomputed from the saved predictions. The generated table snapshots are in the same directory: `table_polypgen.tex`, `table_factorial.tex`, and `table_camelyon.tex`. The TikZ graphs are coded inline in `outputs/introduction.tex`; their coordinates and labels are transcribed from the corresponding summaries. `work/cbmp_smoke_run/` and `outputs/histology_pilot/` contain smoke-test and pilot artifacts, not the final paper results.

## 1. Choose an environment

Choose **venv OR conda**. Activation changes the current shell; it does not install dependencies. Calling an environment's Python directly is sufficient. Keep earlier environments and result directories.

### Access an existing venv

PowerShell:

```powershell
Set-Location 'C:\Users\your_path'
# Activation is optional; direct invocation also works when scripts are restricted.
$Python = (Resolve-Path '.\venv-cbmp\Scripts\python.exe').Path
& $Python --version
& $Python -B .\scripts\check_environment.py
# Optional activation:
. .\venv-cbmp\Scripts\Activate.ps1
# Leave an activated venv with: deactivate
```

Bash on Linux/WSL:

```bash
# WSL on the same Windows computer:
cd /mnt/c/Users/your_path
# Native Linux or a different checkout: replace this with that checkout's path.
# A Windows venv cannot be reused on Linux: create a Linux venv first.
source venv-cbmp/bin/activate
python --version
python -B scripts/check_environment.py
# Leave with: deactivate
```

Strict checking is expected to reject the existing torch 2.14 environment. Use a new reference environment below, or explicitly disclose a portability run with changed versions. Do not copy a venv between operating systems.

### A. New reference venv: PowerShell

```powershell
Set-Location 'C:\Users\your_path'
# This selects the available Python 3.12 installation on this computer.
& 'C:\Users\Lenovo\anaconda3\python.exe' -m venv venv-cbmp-reference
$Python = (Resolve-Path '.\venv-cbmp-reference\Scripts\python.exe').Path
```

On another computer, replace the base interpreter with its Python 3.12 executable, or use `py -3.12 -m venv venv-cbmp-reference`. Optional activation: `. .\venv-cbmp-reference\Scripts\Activate.ps1`.

The automated equivalent, including the pinned installation in section 2, is:

```powershell
.\setup_venv.ps1 -PythonExecutable 'C:\Users\Lenovo\anaconda3\python.exe' -VenvPath "$PWD\venv-cbmp-reference"
```

If local policy blocks script execution, use the explicit Python commands; there is no need to change a machine-wide execution policy.

### B. New reference venv: Bash

```bash
# WSL on the same Windows computer:
cd /mnt/c/Users/your_path
# Native Linux or a different checkout: replace this with that checkout's path.
python3.12 -m venv venv-cbmp-reference
source venv-cbmp-reference/bin/activate
```

The automated equivalent is:

```bash
CBMP_VENV_DIR="$PWD/venv-cbmp-reference" bash setup_venv.sh
```

These commands target 64-bit Linux/WSL x86_64 with Python 3.12 and CPU wheels. Linux installation/full raw-data execution has not been validated on this Windows host. On Ubuntu, a missing venv module requires the matching `python3.12-venv` system package. macOS/ARM is not claimed as a tested CPU-wheel target.

### C. Conda separately: PowerShell

Use a conda-enabled PowerShell session (for example the Anaconda PowerShell Prompt):

```powershell
Set-Location 'C:\Users\your_path'
conda env create -f .\environment-cbmp.yml
conda activate cbmp-repro
$Python = (Get-Command python).Source
& $Python --version
# Continue with section 2.
# Leave with: conda deactivate
```

For an existing environment, skip creation and run `conda activate cbmp-repro`. The YAML fixes Python 3.12.3; the scientific packages are installed using the same pinned pip files as the venv route. Do not install a second unpinned conda torch/numpy stack on top.

### D. Conda separately: Bash

```bash
# WSL on the same Windows computer:
cd /mnt/c/Users/your_path
# Native Linux or a different checkout: replace this with that checkout's path.
conda env create -f environment-cbmp.yml
conda activate cbmp-repro
python --version
# Continue with section 2.
# Leave with: conda deactivate
```

If `conda activate` is unavailable, use a shell initialized by your conda installation. A fresh conda provisioning and a Linux full rerun were not performed as part of this documentation revision.

## 2. Install fixed dependencies and record settings

Skip installation if the automated setup already succeeded. No TensorFlow, CUDA or TeX installation is needed for the five core experiments. The optional older colorectal-histology pilot is documented separately in the detailed reproduction manual.

PowerShell (with `$Python` selected above):

```powershell
$EnvDir = Join-Path $PWD 'isolated_datasets\polypgen_extensions_2026-10-20\reproducibility\environment'
& $Python -m pip install --no-cache-dir -r "$EnvDir\requirements-installer.txt"
if ($LASTEXITCODE -ne 0) { throw 'Installer setup failed' }
& $Python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -c "$EnvDir\constraints-observed.txt" -r "$EnvDir\requirements-torch-cpu.txt"
if ($LASTEXITCODE -ne 0) { throw 'CPU torch installation failed' }
& $Python -m pip install --no-cache-dir --index-url https://pypi.org/simple -c "$EnvDir\constraints-observed.txt" -r "$EnvDir\requirements-common.txt"
if ($LASTEXITCODE -ne 0) { throw 'Scientific dependency installation failed' }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflicts: stop before experiments' }
& $Python -B .\scripts\check_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Reference package check failed' }
$env:OMP_NUM_THREADS='4'; $env:OPENBLAS_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:MPLBACKEND='Agg'
& $Python -m pip freeze --all | Set-Content -Encoding utf8 installed-windows.txt
```

Bash (inside either selected environment):

```bash
set -euo pipefail
ENV_DIR="$PWD/isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment"
python -m pip install --no-cache-dir -r "$ENV_DIR/requirements-installer.txt"
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -c "$ENV_DIR/constraints-observed.txt" -r "$ENV_DIR/requirements-torch-cpu.txt"
python -m pip install --no-cache-dir --index-url https://pypi.org/simple -c "$ENV_DIR/constraints-observed.txt" -r "$ENV_DIR/requirements-common.txt"
python -m pip check
python -B scripts/check_environment.py
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
python -m pip freeze --all > installed-linux.txt
```

DINO extraction originally used eight threads; classifier fitting used four. CAMELYON's completed uniform feature pass used four threads after a documented resource benchmark. The portable raw runner below now uses four for CAMELYON too. CPU execution alone does not guarantee bitwise equality across BLAS/JPEG/library builds. Retain per-sample predictions, not only rounded means.

The original broad Anaconda environment has unrelated dependency conflicts and optional Pandas acceleration warnings; its inventory is not a clean environment lock. See `observed_environment_pip_check.txt`. No conflict should be ignored in a newly installed minimal environment. The current documentation revision tests the available reference interpreter and the helper commands; it does not claim a second full installation or all expensive reruns.

## 3. Start with saved-result verification (no new training)

PowerShell:

```powershell
.\run_pipeline.ps1 -Step verify -PythonExecutable $Python -OutputDir "$PWD\reproduction_runs\verify_20261023"
```

Bash:

```bash
bash run_pipeline.sh verify --output-dir "$PWD/reproduction_runs/verify_20261023"
```

The equivalent platform-neutral command is `python -B scripts/run_project.py verify --output-dir /new/output/path`.

This recalculates BA/AUROC from all **465,534 saved predictions** (140,967 ResNet18 PolypGen; 140,967 DINOv2 PolypGen; 183,600 CAMELYON), checks complete sample pairing and agreement with stored summaries to absolute tolerance 1e-12, and writes fresh LaTeX tables plus input hashes. It does not rerun feature extraction, model selection or bootstrapping. Existing scoped reports contain the original bootstrap results.

Output destinations must be absent. No wrapper deletes an earlier report. `reports` is an alias for this verification, not a command that rewrites historical preservation records.

## 4. Level 1: recalculate uncertainty and figures from predictions

```powershell
.\run_pipeline.ps1 -Step level1 -PythonExecutable $Python -OutputDir "$PWD\reproduction_runs\level1_20261023"
```

```bash
bash run_pipeline.sh level1 --output-dir "$PWD/reproduction_runs/level1_20261023"
```

This runs the original predictions-only helper for both PolypGen encoders and CAMELYON. Whole `group_id` sequences are resampled for PolypGen; `patient` is used for CAMELYON. It performs 2,000 paired bootstrap attempts with seed 2027 and reports rejected class-incomplete draws. Its across-hospital CAMELYON mean is descriptive; use the original `exp5_camelyon/report/` for the separate primary/secondary patient intervals. This level does not validate raw data or prove source-only model selection.

## 5. Level 2: refit the nine main heads from a provided feature cache

One encoder and one fitting seed per invocation:

```powershell
.\run_pipeline.ps1 -Step level2 -PythonExecutable $Python -Encoder resnet18 -Seed 17 -OutputDir "$PWD\reproduction_runs\r18_seed17"
.\run_pipeline.ps1 -Step level2 -PythonExecutable $Python -Encoder dinov2 -Seed 17 -OutputDir "$PWD\reproduction_runs\dino_seed17"
```

```bash
bash run_pipeline.sh level2 --encoder resnet18 --seed 17 --output-dir "$PWD/reproduction_runs/r18_seed17"
bash run_pipeline.sh level2 --encoder dinov2 --seed 17 --output-dir "$PWD/reproduction_runs/dino_seed17"
```

Repeat with **29 and 43**, using fresh output directories. Each invocation includes source-only selection and **36 final evaluations** (nine heads x four eligible centers). It is not a one-minute promise. It does not cover every failure/stress diagnostic or independently validate supplied image embeddings.

For an explicitly disclosed software-portability experiment only, add `-AllowVersionDrift` (PowerShell) or `--allow-version-drift` (Bash). Defaults are strict. Compare new per-image labels and numeric scores with the reference; a successful process or matching rounded BA is not proof of equivalence.

`all` means sections 3-5 for one chosen encoder/seed. It does **not** mean the full raw-data reconstruction.

## 6. Level 3: reconstruct all five experiments from raw data

Prerequisites: the exact local PolypGen v3 archive, reviewed mapping included in the reproduction source snapshot, network access for pinned public assets/CAMELYON, and approximately 30 GiB free plus environment space. The original PolypGen archive is 3,419,253,506 bytes with SHA-256:

```text
a22f956a9f0fc3f941927108a3de8217fc61f1944c7eda598536dbabc27dbff3
```

PowerShell:

```powershell
$Bundle = Join-Path $PWD 'isolated_datasets\polypgen_extensions_2026-10-20\reproducibility'
$Run = Join-Path $PWD 'reproduction_runs\full_reference_01'
$Archive = 'C:\Users\Lenovo\Downloads\PolypGen2021_MultiCenterData_v3 (1).zip' # replace if needed
& $Python -B "$Bundle\reproduce.py" stage --run-dir $Run --polypgen-zip $Archive
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
& $Python -B "$Bundle\reproduce.py" assets --run-dir $Run --download
if ($LASTEXITCODE -ne 0) { throw 'Asset acquisition failed' }
& $Python -B "$Bundle\reproduce.py" preflight --run-dir $Run
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
& $Python -B "$Bundle\reproduce.py" run --run-dir $Run --steps audit baseline followups exp4 exp5 finalize --encoder-threads 8
if ($LASTEXITCODE -ne 0) { throw 'A reconstruction stage failed; preserve its log' }
```

Bash:

```bash
BUNDLE="$PWD/isolated_datasets/polypgen_extensions_2026-10-20/reproducibility"
RUN="$PWD/reproduction_runs/full_reference_01"
ARCHIVE='/mnt/c/Users/Lenovo/Downloads/PolypGen2021_MultiCenterData_v3 (1).zip' # WSL path; replace if stored elsewhere
python -B "$BUNDLE/reproduce.py" stage --run-dir "$RUN" --polypgen-zip "$ARCHIVE"
python -B "$BUNDLE/reproduce.py" assets --run-dir "$RUN" --download
python -B "$BUNDLE/reproduce.py" preflight --run-dir "$RUN"
python -B "$BUNDLE/reproduce.py" run --run-dir "$RUN" --steps audit baseline followups exp4 exp5 finalize --encoder-threads 8
```

The run directory must be absent at staging. Supply paths valid on the current OS; `/mnt/c/...` applies to WSL only. Fresh manifests are reconstructed locally and checked against the original path-independent cohort digest/order. Old absolute Windows CSV paths are not reused on Linux. All four core classifier modules remain byte-identical; checkpoint-location patches to newly staged orchestration scripts are explicit and hashed.

Run individual stages with the same `run --steps ...` argument after prerequisites. See the [detailed manual](isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/README.md) for local checkpoint supply, raw-data access, stage-resume boundaries and the separate histology pilot.

CAMELYON data came from a **community redistribution**, not the author-official WILDS distribution: `wltjr1007/Camelyon17-WILDS`, pinned at `d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f`. The acquisition code first probes the original author endpoint, validates all 21 mirror-shard hashes, then audits hospital/patient/slide identities. Raw data are not included in this code bundle.

## 7. Verified results and interpretation

| Setting | CBMP BA | Linear logistic BA | Trained cosine BA |
|---|---:|---:|---:|
| PolypGen / ResNet18, four-center mean | 58.35% | 70.20% | 63.12% |
| PolypGen / DINOv2, four-center mean | 59.54% | 61.31% | 68.17% |
| CAMELYON / ResNet18, primary H2 | 85.40% | 88.88% | 87.88% |
| CAMELYON / ResNet18, secondary H1 | 77.58% | 80.30% | 79.19% |

These are seed means from the saved source-selected heads. CBMP improves over the group centroid in primary CAMELYON, but is below the linear probe there. DINOv2 does not improve every classifier. Multiplicity and center balancing have condition-dependent effects; there is no demonstrated universal prototype failure law or universal CBMP advantage.

The 58.35% -> 58.82% PolypGen result is a **CBMP source-selected decision-threshold control**, not the linear baseline. The cohort is a reviewed 5,444-frame/37-sequence subset: targets C2/C4/C5/C6, negative-only C1 used as a source, and C3 absent. C3 absence must not be explained as simply missing masks. Patients are unverified for PolypGen. C2 negatives and C5 positives each occur in one sequence.

CAMELYON uses 17,200 selected patches/43 patients, including 26 source patients and **9 primary-test + 8 secondary-test patients**. Patient intervals concern those target patients and fixed trained heads, not 43 independent test patients or a population of hospitals. No official full-WILDS leaderboard result is claimed.

## 8. Manuscript and exploratory analyses

The revised manuscript source is `outputs/introduction.tex`; it now reconciles sections 1-3 with the implemented protocol and adds a fourth, results-based section. The supplied CVPR PDF still contains the template abstract instructions and title placeholder. This revision does not yet write the abstract, conclusion, or mandatory contribution statement. The numerical tables were regenerated from the saved prediction CSVs by `scripts/build_paper_results.py`; `outputs/revision_2026-10-23/generated_results/verification.json` records the input hashes and checks. The result tables in the manuscript were independently checked against these reconstructed values. No new training was run for the manuscript revision.

Later `cvpr_dissection/` and `multi_backbone/` files remain available as exploratory development artifacts. They are **not pooled into the matched confirmatory tables**. The five-backbone script excludes source C1, fixes different classifier settings, and its "trainable cosine" is actually a centroid-based fixed cosine rule. The dissection's PSRP operates on center support, while the other script uses sequence support; they are different interventions. Its LaTeX table contains literal numbers inconsistent with measured CSVs. Purity and t-SNE do not identify acquisition artifacts or prove causation. See `outputs/revision_2026-10-23/SCIENTIFIC_AUDIT.md` before reusing those tables or claims.

## 9. Verification and records to retain

```powershell
& $Python -B isolated_datasets\polypgen_extensions_2026-10-20\reproducibility\test_helpers.py
```

```bash
python -B isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/test_helpers.py
```

Archive Python/OS/build versions, installation and `pip check` logs, package snapshot, code/checkpoint/data hashes, exact sample IDs/splits, source-search records, settings, all predictions and models, and completion markers. Keep optimization-seed SD separate from patient/sequence uncertainty. A checksum proves file identity, not absence of leakage or validity of a causal claim.

Historical `preservation_check.json` files describe their original runs. This authorized manuscript/documentation revision does not rewrite those historical records or rerun full training. New checks write to fresh directories. No launcher removes old outputs, automatically accepts version drift, or announces success after a failed native command.
