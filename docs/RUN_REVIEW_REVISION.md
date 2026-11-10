# Ten-seed review revision: cached fitting and reports

Run from the repository root. These commands use the reviewed image manifests and the **existing** ResNet18/DINOv2 caches. They neither extract new images nor overwrite the historical three-seed results. The initial complete smoke run reproduced all 36 historical PolypGen/ResNet18 seed-17 metric cells exactly in the isolated environment, with identical selected configurations.

The new implementation is `scripts/run_review_revision.py`; the old `run_project.py level2` interface still reproduces its original three-seed protocol. Do not mix the two result versions in a table.

## 1. Install the cached-fitting environment

The tested Windows interpreter is CPython 3.12.3 on x86_64. Both locks contain exact wheel hashes and all runtime/test dependencies. The Linux x86_64 wheel set was downloaded and its metadata dependency closure verified; **Linux execution has not yet been tested**. The Linux Pillow wheel requires glibc 2.28 or later. These locks do not install PyTorch or pathology model packages, because cached head fitting needs neither.

### Bash / Linux or WSL

```bash
python3.12 --version
python3.12 -m venv .venv-cbmp-review
source .venv-cbmp-review/bin/activate
python -m pip install --require-hashes -r requirements/review-linux.lock
python -m pip check
mkdir -p tmp
python -m pytest tests/test_review_revision.py tests/test_external_review.py -q --basetemp tmp/review-tests-linux
python -B scripts/run_reference_tests.py --output-dir reproduction_runs/reference-tests-linux
```

Use a new `--basetemp` path for each test invocation. Calling `.venv-cbmp-review/bin/python` directly is equivalent to activation. A Windows environment cannot be reused inside WSL.

### PowerShell / Windows

```powershell
py -3.12 --version
py -3.12 -m venv .venv-cbmp-review
$ReviewPython = (Resolve-Path '.\.venv-cbmp-review\Scripts\python.exe').Path
& $ReviewPython -m pip install --require-hashes -r requirements/review-windows.lock
if ($LASTEXITCODE -ne 0) { throw 'Locked installation failed' }
& $ReviewPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
New-Item -ItemType Directory -Path tmp -Force | Out-Null
& $ReviewPython -m pytest tests/test_review_revision.py tests/test_external_review.py -q --basetemp tmp/review-tests-windows
if ($LASTEXITCODE -ne 0) { throw 'Protocol tests failed' }
& $ReviewPython -B scripts/run_reference_tests.py --output-dir reproduction_runs/reference-tests-windows
if ($LASTEXITCODE -ne 0) { throw 'Reference classifier tests failed' }
# Optional activation, rather than explicit interpreter invocation:
# . .\.venv-cbmp-review\Scripts\Activate.ps1
```

If `py` is unavailable, use the absolute path to an installed Python 3.12 interpreter. Preserve any existing venv instead of creating over it. The local tested environment is `.venv-cbmp-review`; `.venv-cbmp-reference` was an incomplete earlier setup and is not the verified environment for this revision.

### Conda, separately

Create a separate Python/pip environment, then install the same platform lock. This conda route is documented but was not executed in the Windows validation.

```bash
conda create -n cbmp-review python=3.12.3 pip=24.0 -y
conda activate cbmp-review
python -m pip install --require-hashes -r requirements/review-linux.lock
python -m pip check
```

In PowerShell, use the same `conda create` and `conda activate` commands, then:

```powershell
python -m pip install --require-hashes -r requirements/review-windows.lock
if ($LASTEXITCODE -ne 0) { throw 'Locked installation failed' }
python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
```

`pip` and the interpreter bootstrap are recorded separately from the scientific wheel lock. No tested container image or cross-platform bitwise guarantee is claimed.

## 2. Validate a plan, then fit

Default fitting seeds are **17, 29, 43, 61, 89, 101, 131, 173, 211, 257**. Default tasks are `polypgen_resnet18`, `polypgen_dinov2`, and `camelyon_resnet18`. All three protocols run by default:

| Protocol | Purpose |
|---|---|
| historical | Original nine head families and original candidate grids, now repeated across ten fitting seeds. |
| matched4 | Four distinct candidates per tunable family; fixed centroid controls retain one. Changes grid coverage as well as candidate counts. |
| factorial | Fixed `(K,beta)` cells `(1,0)`, `(1,1)`, `(4,0)`, `(4,1)`; no target-selected setting. |

Bash:

```bash
python -B scripts/run_review_revision.py --run-dir reproduction_runs/review-local --prepare-only
python -B scripts/run_review_revision.py --run-dir reproduction_runs/review-local --resume
python -B scripts/report_review_revision.py --run-dir reproduction_runs/review-local --output-dir reproduction_runs/review-local-report --attempts 2000 --seed 2027
python -B scripts/audit_aggregate_uncertainty.py --report-dir reproduction_runs/review-local-report --output-dir reproduction_runs/review-local-aggregates
python -B scripts/audit_review_stability.py --run-dir reproduction_runs/review-local --output-dir reproduction_runs/review-local-stability
python -B scripts/export_review_tables.py --report-dir reproduction_runs/review-local-report --output-dir reproduction_runs/review-local-tables
```

PowerShell:

```powershell
& $ReviewPython -B scripts/run_review_revision.py --run-dir reproduction_runs/review-local --prepare-only
if ($LASTEXITCODE -ne 0) { throw 'Input validation failed' }
& $ReviewPython -B scripts/run_review_revision.py --run-dir reproduction_runs/review-local --resume
if ($LASTEXITCODE -ne 0) { throw 'Fitting failed; inspect the per-seed run.log' }
& $ReviewPython -B scripts/report_review_revision.py --run-dir reproduction_runs/review-local --output-dir reproduction_runs/review-local-report --attempts 2000 --seed 2027
if ($LASTEXITCODE -ne 0) { throw 'Reporting failed' }
& $ReviewPython -B scripts/audit_aggregate_uncertainty.py --report-dir reproduction_runs/review-local-report --output-dir reproduction_runs/review-local-aggregates
if ($LASTEXITCODE -ne 0) { throw 'Aggregate interval reporting failed' }
& $ReviewPython -B scripts/audit_review_stability.py --run-dir reproduction_runs/review-local --output-dir reproduction_runs/review-local-stability
if ($LASTEXITCODE -ne 0) { throw 'Stability audit failed' }
& $ReviewPython -B scripts/export_review_tables.py --report-dir reproduction_runs/review-local-report --output-dir reproduction_runs/review-local-tables
if ($LASTEXITCODE -ne 0) { throw 'Paper table export failed' }
```

Choose a new fitting and report directory for a new experiment. `--resume` accepts only an identical plan, including code, feature/manifest hashes, seed list, protocols, packages, and resources. It verifies completed predictions and frozen choices before skipping them. Do not edit training scripts during a run. If a process is killed, first verify that no worker from that run remains alive before removing its `RUNNING` marker. Do not run two processes against the same result directory.

Defaults are two worker processes with four numeric threads each. For a smaller machine, add `--workers 1 --threads 2` to **both** planning and execution commands. A quick implementation smoke test can use `--tasks polypgen_resnet18 --seeds 17`; it is not a replacement for the ten-seed study.

## 3. Locate and interpret evidence

The completed local study is `reproduction_runs/review_2026-11-07_full/`, with 30 completed jobs and 2,200 evaluations. Its final report is `outputs/revision_2026-11-07/review_report/`; aggregate intervals are in `review_aggregate_uncertainty/` and prototype audits in `review_stability/` alongside it.

| Artifact | Meaning |
|---|---|
| run_manifest.json | Exact tasks, seeds, candidate grids, source files, feature/checkpoint provenance, runtime/build and thread settings. |
| pip_check.txt / installed_packages.txt | Dependency validation and installed versions in the new isolated environment. |
| task/seedN/run.log | Candidate fitting progress. |
| task/seedN/source_heads_frozen.json | Every selected setting, source-fold score, temperature, source-only sequence threshold and final model hash, frozen before that job's targets are predicted. |
| task/seedN/source_cache/ | Reusable source-validation scores and fitted heads, keyed by identical training/validation indices and candidate settings. Load only trusted locally generated joblib files. |
| task/seedN/predictions.csv | Raw per-image labels, decisions, positive probabilities and raw margins. |
| task/seedN/metrics.json / COMPLETE.json | Per-head metrics and completion fingerprints. |
| root COMPLETE.json | Written only after every planned task/seed job completes. |

Report files:

- `image_metrics_by_seed.csv` and `image_metrics_seed_means.csv`: BA, macro F1, AUROC, NLL, Brier score, optimization standard deviations and valid seed counts.
- `selected_settings.csv`: per-seed selections and candidate/validation-fit counts. Equal candidate counts do not imply equal computation.
- `compute_usage.csv`: unique measured candidate and final fits, shared across protocols; fit seconds are not total worker wall time.
- `paired_ba_intervals.csv`: paired group-resampling CBMP-minus-control differences, target-specific and equal-center summaries.
- `protocol_sensitivity_intervals.csv`: matched-four minus original-grid BA with the same paired group draws. The grids change parameter coverage, so these are not causal estimates of candidate count alone.
- `group_metrics_by_seed.csv`: frame/patch metrics within every recorded sequence/patient.
- `sequence_predictions.csv` and `sequence_metrics_*.csv`: any-positive-frame sequence labels, maximum raw margin, and a threshold chosen using eligible **source** sequence classes. C2 is single-class at this endpoint, so BA/AUROC are blank, rather than forced to a numerical value.
- `patient_macro_*.csv`: equal-patient averages of within-patient patch metrics. This is **not clinical patient diagnosis**. No patient/slide label is inferred from sampled patches.
- `group_ranking_*.csv`, `factorial_ranking_contrasts.csv`, and `factorial_ba_interactions.csv`: pooled, same-group, across-group and equal-pair ranking, paired beta contrasts and the K-by-beta BA interaction.
- `provenance.json`: prediction fingerprints, bootstrap attempts/seed and the limited scope of the intervals.

The separate stability audit writes optimally matched prototype cosines for every fitting-seed pair, reports unequal realized prototype counts explicitly, and saves the nearest **source** example ID for each prototype. Stability is conditional on comparable counts; it is not proof that a cluster is clinically meaningful. The source-example file supports author inspection without inventing a semantic assessment.

The paper exporter requires the complete three-task, ten-seed report. It generates native LaTeX tables, native TikZ figures, the full seed-averaged supplementary tables, and `paper_values.json`, all with input fingerprints. Review the scientific narrative when updating the manuscript; the exporter does not turn an exploratory result into a confirmatory claim.

Metrics are fractions. Multiply by 100 for percent scores or percentage-point differences. Standard deviation over seeds describes fitting variation on the fixed cohorts. Bootstrap intervals resample recorded sequences or patients within the observed sites, holding fitted heads fixed. They do not estimate a hospital population or resolve PolypGen patient independence. The added grids and aggregate endpoints were designed after the original targets had been inspected; they remain exploratory robustness checks, not a new untouched confirmation cohort.

## 4. Optional data/model extensions

See [EXTERNAL_EVALUATION.md](EXTERNAL_EVALUATION.md) for complete official-WILDS and original UNI/CONCH adapter commands. Those experiments require their own inputs, environments, result directories, and verification. Checkpoint access and a complete official dataset are not supplied by a successful cached run.
