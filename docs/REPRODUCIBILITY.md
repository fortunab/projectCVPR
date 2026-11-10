# Reproducing the paper and interpreting the evidence

Updated on 7 November 2026. Run the commands from the repository root. Use a new output directory for each invocation. The commands below are the existing supported entry points.

## Which result is being reproduced?

| Task | Prediction unit | Recorded resampling unit | Historical fitting seeds | Evaluation |
|---|---|---|---|---|
| Reviewed PolypGen / ResNet18 | Frame | Recorded sequence; patient linkage unresolved | 17, 29, 43 | C2, C4, C5, C6 held out in turn |
| Reviewed PolypGen / DINOv2 | Frame | Same recorded sequences | 17, 29, 43 | Identical reviewed images and outer partitions |
| Custom CAMELYON17-WILDS subset / ResNet18 | Patch | Patient within a fixed target hospital | 17, 29, 43 | H0/H3/H4 sources, H2 primary target, H1 secondary target |

The PolypGen cohort contains 5,444 frames and 37 sequences. Its eligible target support is C2 with 2 sequences, C4 with 10, C5 with 4, and C6 with 18. The remaining 3 sequences are source-only C1 data. C1 has negative-only support and C3 is excluded after center-mapping review. Inclusion followed the annotation and provenance rules.

The histology cohort contains 17,200 patches from 43 patients. Of these, 26 patients are sources and 17 are targets: 9 at H2 and 8 at H1. The 400-patch cap retained all eligible patients in the redistributed metadata. Removing that cap increases patch coverage at the same patients and hospitals. It does not expand the hospital population.

## Runtime and hardware records

The recorded scientific runtime is Python 3.12.3, CPU PyTorch 2.13.0, torchvision 0.28.0, NumPy 1.26.4, SciPy 1.13.1, pandas 3.0.5, and scikit-learn 1.5.1.

The following paths are relative to the repository root.

| Record | Path |
|---|---|
| Direct scientific requirements | isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment/requirements-common.txt |
| CPU PyTorch requirements | isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment/requirements-torch-cpu.txt |
| Installer versions | isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment/requirements-installer.txt |
| Observed constraints | isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment/constraints-observed.txt |
| Numeric library/build snapshot | isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment/observed_numeric_runtime.json |
| Installed-distribution snapshot | isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment/observed_windows_environment.json |
| Historical dependency conflicts | isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment/observed_environment_pip_check.txt |
| DINO run configuration | isolated_datasets/polypgen_extensions_2026-10-20/exp4_dinov2/experiment_config.json |
| Histology extraction settings | isolated_datasets/polypgen_extensions_2026-10-20/exp5_camelyon/features/extraction.json |
| Histology recovery and thread benchmark | isolated_datasets/polypgen_extensions_2026-10-20/exp5_camelyon/runtime_recovery.json |

DINOv2 extraction used batch size 16, 8 encoder threads, 4 head-fitting threads, and zero data-loader workers. The completed histology extraction used batch size 64, 4 threads, and zero workers. The recorded PyTorch build is CPU-only and its runtime probe reports CUDA unavailable. These records are more specific than a device label alone.

A separate read-only hardware observation on 7 November 2026 found an AMD64 Family 23 Model 96 processor signature, 8 physical cores, 16 logical CPUs, and approximately 15.37 GiB physical memory. A marketing CPU model and historical RAM snapshot were not captured in the original run records. This later observation is not a retrospective hardware attestation for every experiment.

The direct pins, constraints, Windows installation report, and package inventory preserve the historical software condition. They are not a tested container image. The broad historical environment has dependency conflicts. The new cached-fitting environment is separate: `requirements/review-windows.lock` and `review-linux.lock` contain exact platform wheel hashes for the head/test dependencies. A fresh Windows installation passes `pip check` and reproduces the original seed-17 results exactly. Linux wheel dependency closure was verified, but Linux execution has not been tested. Full encoder-extraction/container reconstruction remains distinct from this cached numerical replay.

## Approved ten-seed extension

Use [RUN_REVIEW_REVISION.md](RUN_REVIEW_REVISION.md) for the new locked environment and complete Bash/PowerShell/conda commands. The older commands below remain reference reproduction routes for the original three-seed study. They do not substitute for the new runner.

The expanded cached run is `reproduction_runs/review_2026-11-07_full/`. Its immutable `run_manifest.json` records ten fitting seeds, original-grid and four-candidate sensitivity protocols, fixed factorial cells, exact inputs and code, scientific package/build versions, and resources. It needs no PyTorch installation because feature extraction is not repeated. Its `pip_check.txt`, `installed_packages.txt`, per-seed source-search records, frozen model hashes, predictions, and logs accompany the reports. A root `COMPLETE.json` is required before final reporting.

The actual initial replay evidence is `outputs/revision_2026-11-07/smoke_replay_verification.json`: 36 historical metric cells, all 46,989 raw prediction probabilities, decisions/labels/groups, and selected settings match exactly for PolypGen/ResNet18 seed 17. This result is specific to the recorded Windows runtime, fixed cache, seed and candidate grids. It does not establish cross-platform equality or fresh-image reconstruction.

The new reports add fitting-seed SD, per-group and supported sequence metrics, equal-patient patch summaries, paired group intervals, candidate/fit counts and timing, grid-sensitivity and factorial ranking/interaction uncertainty, and prototype stability. Clinical patient labels are not inferred. Eleven new protocol/archive tests and 33 existing reference tests passed in the isolated environment; their records are `outputs/revision_2026-11-07/test_verification.json` and `core_regression_tests.txt`. The data/model adapters in [EXTERNAL_EVALUATION.md](EXTERNAL_EVALUATION.md) have passing boundary fixtures; their full provider/dataset integration remains unexecuted. Actual author sign-off can be recorded in [AUTHOR_REVIEW_RECORD.md](AUTHOR_REVIEW_RECORD.md).

Portable cache preparation was separately checked for all three tasks. It changes local path fields and the corresponding container/manifest hashes while retaining every feature value, label, center, group, content identifier and row order. The record `outputs/revision_2026-11-07/portable_cache_probe/validation/portable_cache_verification.json` includes both original and relocated fingerprints and exact numerical-matrix hashes. This is a cache-relocation check, not a fresh raw-image audit.

## Reference environment

The full README has separate venv and conda setup instructions. The shortest existing venv route is below.

Linux or WSL with Python 3.12:

    CBMP_VENV_DIR="$PWD/.venv-cbmp-reference" bash setup_venv.sh
    source .venv-cbmp-reference/bin/activate
    python -m pip check
    python -B scripts/check_environment.py --output reproduction_runs/environment_linux.json
    python -m pip freeze --all > reproduction_runs/packages_linux.txt

PowerShell with a Python 3.12 launcher:

    .\setup_venv.ps1 -PythonExecutable 'C:\path\to\python312\python.exe' -VenvPath "$PWD\.venv-cbmp-reference"
    $CbmpPython = (Resolve-Path '.\.venv-cbmp-reference\Scripts\python.exe').Path
    & $CbmpPython -m pip check
    & $CbmpPython -B scripts/check_environment.py --output reproduction_runs/environment_windows.json
    & $CbmpPython -m pip freeze --all | Set-Content -Encoding utf8 reproduction_runs/packages_windows.txt

Replace the PowerShell interpreter path with the installed Python 3.12 executable. Activation is optional when calling that environment's interpreter explicitly. The environment checker creates the environment-report directory. If installation or pip check fails, retain the log and resolve the conflict before proceeding. The metadata checker compares base package versions and the Python minor version. The numeric-runtime record and a numerical replay are needed to assess build differences, CPU/CUDA variants, and actual output agreement.

## Start with verification of saved predictions

This route reconstructs point estimates without extracting new features. It requires the saved prediction exports referenced by the paper.

Linux or WSL:

    python -B scripts/run_project.py verify --output-dir reproduction_runs/point_check_linux
    python -B scripts/audit_group_ranking.py --output-dir reproduction_runs/group_audit_linux
    python -B scripts/audit_factorial_ranking.py --output-dir reproduction_runs/factorial_audit_linux

PowerShell:

    & $CbmpPython -B scripts/run_project.py verify --output-dir reproduction_runs/point_check_windows
    & $CbmpPython -B scripts/audit_group_ranking.py --output-dir reproduction_runs/group_audit_windows
    & $CbmpPython -B scripts/audit_factorial_ranking.py --output-dir reproduction_runs/factorial_audit_windows

Expected evidence:

- The point-estimate verifier reads 465,534 saved prediction records across the three original task/encoder combinations.
- The group-ranking audit writes 270 head-target-seed cells, 90 seed-averaged cells, group-omission diagnostics, and input hashes.
- The factorial-ranking audit writes 96 setting-target-seed cells, 32 seed-averaged cells, 150 paired component contrasts, selected configurations, and source-validation scores.
- The original BA bootstrap uses the stored common class-complete resamples. The later ranking bootstrap has its own recorded attempt counts and can retain different valid draws for each component.

CSV AUROC values are fractions. Multiply by 100 for percent scores and percentage-point differences. Same and Across retain original group identity during resampling. A zero-width interval can reflect a single observed group pair. Resampling cannot establish patient independence or supply a new hospital.

## Refit supported original heads

The current wrapper accepts seeds 17, 29, and 43. These commands refit the nine PolypGen heads for one encoder and seed from a verified cache.

Linux or WSL:

    python -B scripts/run_project.py level2 --encoder resnet18 --seed 17 --output-dir reproduction_runs/resnet18_seed17_linux
    python -B scripts/run_project.py level2 --encoder dinov2 --seed 17 --output-dir reproduction_runs/dinov2_seed17_linux

PowerShell:

    & $CbmpPython -B scripts/run_project.py level2 --encoder resnet18 --seed 17 --output-dir reproduction_runs/resnet18_seed17_windows
    & $CbmpPython -B scripts/run_project.py level2 --encoder dinov2 --seed 17 --output-dir reproduction_runs/dinov2_seed17_windows

Use the raw-data reconstruction route in README.md to re-extract features and rebuild manifests. Copying a cache to another operating system preserves its sample order and metadata requirements. Each new feature extraction must record the checkpoint, preprocessing, manifest, sample order, feature hash, and numeric runtime.

## Determinism and what was actually checked

| Item | Evidence or scope |
|---|---|
| Frozen extraction | Recorded checkpoint preprocessing, evaluation mode, no stochastic augmentation, finite normalized features, and ordered sample IDs. |
| Repeated forward pass | DINOv2 and histology extraction code check equality on a repeated initial batch. This is a limited repeatability check. |
| Fitting randomness | The fitting seed enters prototype initialization, cosine-vector initialization, and relevant scikit-learn fitting calls. Candidate order fixes source-validation ties. |
| Numeric threads | Orchestrators set thread limits and environment controls. The DINO encoder and classifier use their separately recorded thread counts. |
| Global deterministic algorithms | The historical evidence does not record an enabled torch.use_deterministic_algorithms setting. No cross-platform bitwise guarantee follows from CPU execution or a repeated batch. |
| Source/target boundaries | Stored splits, group/content checks, source-validation searches, and frozen-head records support review of the implemented selection path. Patient independence remains a separate data question. |
| Replayed diagnostics | The clean analysis package reproduced the ranking tables and factorial contrasts. Verification is recorded in outputs/revision_2026-11-05/verification.json. |
| Complete fresh environment/container reconstruction | Requires its own installation, dependency checks, runtime capture, and output comparison. A package snapshot alone does not establish this result. |

## Assurance for AI-assisted implementation

AI assistance extended to study planning, code, debugging, analysis, and manuscript/figure preparation. The LLM disclosure must describe that actual scope. The following records support technical review of the implementation:

- Saved original predictions, data/checkpoint fingerprints, feature metadata, source-search records, and failed-attempt/recovery logs.
- Existing helper regression checks in isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/test_helpers.py.
- Prediction metric recomputation against saved reports.
- Explicit checks of paired sample/group support and exact pooled-AUROC reconstruction.
- A fixture comparing the factorial bootstrap's group-pair counts with explicitly repeated observations, including ties and undefined components.
- Clean-package replay of both diagnostic scripts.

These are automated checks of specified properties. They do not replace author review of labels, clinical endpoint definitions, source code, citations, exclusions, and interpretations. Before submission, the authors should record who reviewed these items, which revision and artifacts they reviewed, what failed, and how discrepancies were resolved. A record of that review is currently an author action, not an attestation supplied by the scripts.

## Protocol changes and access

See REVIEW_RESPONSE.md for the approved ten-seed study and the remaining official-split/medical-encoder input prerequisites. Historical numerical results remain tied to their original three seeds and search grids; new results have distinct run manifests and report directories.

The standard WILDS CAMELYON17 evaluation uses hospitals 0/3/4 for training, hospital 1 for validation, and hospital 2 for test. Its published submission rules require ten random seeds and restrict external pretraining for the standard setting. An ImageNet, UNI, or CONCH frozen encoder must be identified with its pretraining setting when compared using official splits.

UNI and CONCH have public descriptions and code, with checkpoint access and terms managed by their providers. A future run must document the exact checkpoint family and revision. UNI, UNI2-h, CONCH, and later CONCH-derived models should not be silently interchanged.

Primary sources:

- [WILDS submission rules](https://wilds.stanford.edu/submit/).
- [UNI implementation and access instructions](https://github.com/mahmoodlab/UNI).
- [CONCH implementation and access instructions](https://github.com/mahmoodlab/CONCH).
