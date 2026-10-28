# Reproduce the audited prototype experiments

This package reconstructs the **PolypGen baseline, follow-ups 1–3, DINOv2 extension 4, and CAMELYON extension 5** in a **new workspace**. It never merges with an existing run, edits a manuscript, uploads to Overleaf, or modifies the supplied archive/checkpoints. Numerical results must come from completed runs, not from these instructions.

The prepared `sources/` directory is a small source snapshot. `sources_manifest.json` records each original workspace-relative source path and SHA-256, the dataset-archive/checkpoint hashes, and reference package versions. Raw PolypGen images and CAMELYON shards are **not included**. `cohort_order.json` contains only opaque sample identities and the expected aggregate cohort digest, not per-sample labels or images.

## What has and has not been tested

- The original experiments run on Windows with Python 3.12.3. Actual installation/replay evidence and environment files are under [`environment/`](environment/). The observed global environment is an inventory, **not automatically a validated lockfile**. Read its validation record and `pip check` result before asserting that the clean environment was reproduced.
- Portable-helper tests exercise fresh staging, checkpoint-path relocation, preservation of source copies, ZIP traversal/symlink rejection, manifest reordering without label changes, and prediction-support validation in temporary directories on Windows. See [`test_helpers.py`](test_helpers.py) and the accompanying test record when present.
- Linux/Bash commands below are provided for a fresh Python 3.12 environment. **This session has no Linux host, so neither Linux installation nor the full Linux raw-data rerun has been tested.** A reviewer should record that host's installation and numerical checks. CPU/BLAS/JPEG builds can cause numerical or decoded-pixel differences even with equal package versions; this package refuses an altered audited cohort and does not promise bitwise equality across operating systems.
- The staging helpers are tested with small fixtures. That is not a claim that all expensive experiments were repeated from raw data inside a second portable workspace during this session.

## Three verification levels

| Level | Inputs | What it verifies | What it does not verify |
|---|---|---|---|
| 1: predictions → metrics | Saved prediction CSVs with labels and sequence/patient IDs | Paired support, per-center metrics, equal-center/seed means, group-bootstrap BA intervals, table/figure | Raw images, feature extraction, how settings were selected |
| 2: cached features → main heads | Manifest + adjacent audit JSON, matching ResNet18 or DINOv2 cache | Exact reviewed cohort/digest, feature checkpoint metadata, unchanged nine-head grids and source-only partitions, 36 main evaluations per seed | Raw-image bytes or whether supplied features were honestly extracted from them |
| 3: local archive → complete pipeline | Exact local PolypGen v3 ZIP, reviewed mapping snapshot, pinned public checkpoints, network access for CAMELYON | Fresh data audit, features, source selection, heads, predictions, reports, preservation checks | Patient independence absent from PolypGen metadata; full official WILDS leaderboard performance |

Level 3 is the full reconstruction route. Level 2 reproduces the main classifiers only; it is not a shortcut claiming to rerun every failure diagnostic or raw-data audit.

## Prerequisites and fixed choices

1. **Python 3.12**, 64-bit CPU installation. The code uses Python 3.12 syntax/APIs. No CUDA, TeX distribution, TensorFlow, Synapse login, or `timm` is needed for experiments 1–5 from the local archive. TensorFlow is optional only for the separate TFDS pilot appendix.
2. A legitimate local copy of `PolypGen2021_MultiCenterData_v3.zip` (the downloaded name may include `(1)`). Exact size: **3,419,253,506 bytes**. SHA-256: **`a22f956a9f0fc3f941927108a3de8217fc61f1944c7eda598536dbabc27dbff3`**. Obtain it through the [PolypGen author distribution](https://www.synapse.org/Synapse:syn26376615/wiki/613312) and comply with its access/redistribution terms. This bundle does not bypass gated access or republish images.
3. Approximately **30 GiB free** in the new workspace for extracted data, public CAMELYON shards, selected images, checkpoints, models and scientific artifacts, plus space for the Python environment. The archive itself may remain elsewhere and is read only. Full CAMELYON acquisition transfers about **10.7 GB**; runtime and actual transfer depend on access and hardware.
4. The reviewed mapping CSV and review provenance are already included under `sources/baseline/audit/`. The expected subset is **5,444 frames, 37 sequences**, target centers **C2/C4/C5/C6**, source-only C1 and absent C3. Unknown/disputed mappings are excluded. The audit reconstructs frame labels from annotations; sequence-folder positivity is not an image label.
5. Checkpoints are fixed: torchvision **ResNet18 ImageNet1K V1** and official **`facebook/dinov2-base`**, revision **`f9e44c814b77203eaa57a6bdbbd535f21ede1415`**, official processor from that revision, normalized final CLS features. Full expected SHA-256 values are in `sources_manifest.json`.
6. Source tuning grids remain unchanged: CBMP12 candidates, unbalanced multiprototypes4, linear5, cosine3, weighted linear15, weighted cosine9, three centroid controls1 each. Seeds17/29/43. Five prototype restarts, max100 updates; cosine300 steps. All fitting/calibration is source-only.

## Install a clean environment: Linux/Bash

Set `BUNDLE` to this **already exported** package. Use a new virtual environment outside the absent run directory. Do not reinstall into a research environment containing earlier runs.

```bash
BUNDLE="/absolute/path/to/reproducibility"
python3.12 -m venv "$PWD/.venv-cbmp-312"
. "$PWD/.venv-cbmp-312/bin/activate"
python -m pip install -r "$BUNDLE/environment/requirements-installer.txt"
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r "$BUNDLE/environment/requirements-torch-cpu.txt"
python -m pip install -r "$BUNDLE/environment/requirements-common.txt" -c "$BUNDLE/environment/constraints-observed.txt"
python -m pip check
python -m pip freeze --all > "$PWD/installed-cbmp-linux.txt"
python -B "$BUNDLE/test_helpers.py"
```

These pins describe the observed software. If a pinned wheel cannot be resolved on the target OS, keep the installation log and stop the **exact-version** reproduction claim. A separately recorded portability run can install a compatible environment and pass `--allow-version-drift`; this changes the software condition and must be disclosed. Do not silently replace a checkpoint, feature architecture, transform, tuning grid, or dataset when installation fails.

## Install a clean environment: Windows/PowerShell

PowerShell is used here, not `cmd.exe`; commands do not require `ls` or shell activation. Calling the virtual environment's interpreter directly avoids execution-policy changes.

```powershell
$Bundle = "D:\review\reproducibility"
$Venv = "D:\review\venvs\cbmp-312"
py -3.12 -m venv $Venv
$Python = Join-Path $Venv "Scripts\python.exe"
& $Python -m pip install -r "$Bundle\environment\requirements-installer.txt"
& $Python -m pip install --index-url https://download.pytorch.org/whl/cpu -r "$Bundle\environment\requirements-torch-cpu.txt"
& $Python -m pip install -r "$Bundle\environment\requirements-common.txt" -c "$Bundle\environment\constraints-observed.txt"
& $Python -m pip check
& $Python -m pip freeze --all | Set-Content -Encoding utf8 "D:\review\installed-cbmp-windows.txt"
& $Python -B "$Bundle\test_helpers.py"
```

After a failed PowerShell command, stop and inspect the exit code (`$LASTEXITCODE`); do not continue to fitting with a partially installed environment.

## Level 3: full raw-data reconstruction

### Linux/Bash

```bash
RUN="$PWD/cbmp_fresh_run"
ARCHIVE="/datasets/PolypGen2021_MultiCenterData_v3 (1).zip"
python -B "$BUNDLE/reproduce.py" stage --run-dir "$RUN" --polypgen-zip "$ARCHIVE"
python -B "$BUNDLE/reproduce.py" assets --run-dir "$RUN" --download
python -B "$BUNDLE/reproduce.py" preflight --run-dir "$RUN"
python -B "$BUNDLE/reproduce.py" run --run-dir "$RUN" --steps audit baseline followups exp4 exp5 finalize --encoder-threads 8
```

Use `set -e` in a saved Bash script if you want the shell to stop automatically on each nonzero exit. The Python runner itself stops whenever a child command fails.

### Windows/PowerShell

```powershell
$Run = "D:\review\cbmp_fresh_run"
$Archive = "D:\datasets\PolypGen2021_MultiCenterData_v3 (1).zip"
& $Python -B "$Bundle\reproduce.py" stage --run-dir $Run --polypgen-zip $Archive
& $Python -B "$Bundle\reproduce.py" assets --run-dir $Run --download
& $Python -B "$Bundle\reproduce.py" preflight --run-dir $Run
& $Python -B "$Bundle\reproduce.py" run --run-dir $Run --steps audit baseline followups exp4 exp5 finalize --encoder-threads 8
```

`stage` requires an absent destination and refuses overwrite. It checks the full archive checksum and ZIP paths before extracting. It stages new siblings using the original relative layout:

```text
RUN/
  assets/                         # full-hash-verified official checkpoints
  reproduction/                   # stage patches, preflight, timestamped logs
  isolated_datasets/
    polypgen_v3_copy_2026-10-17/   # rebuilt audit/features/baseline/results/report
    polypgen_followups_2026-10-17/ # failure, factorial and stress experiments
    polypgen_extensions_2026-10-20/
      exp4_dinov2/                # second encoder and paired factorial
      exp5_camelyon/              # public shards/audit/features/frozen heads/report
  histology_pilot/                 # optional separate pilot sources, no automatic run
```

Steps may be run separately using the same `run --steps ...` syntax:

| Step | Actual prepared scripts and output checks |
|---|---|
| `audit` | Safe ZIP extraction, original `process_audit.py`, then portable row-order/digest validation |
| `baseline` | Original `run_experiments.py` (dry-run eligibility then three seeds), `make_report.py` |
| `followups` | `setup_followups.py`, `exp1_failure.py`, `exp2_factorial.py`, `exp3_stress.py`, preservation check |
| `exp4` | `setup_extensions.py` when needed, DINO inspection/extraction, serial 108 main +48 factorial evaluations, `exp4_report.py` |
| `exp5` | Original acquisition/audit/extraction, source-only 27-head freeze, 54 target records, `exp5_report.py` |
| `finalize` | Extension preservation check and `finalize_extensions.py`; requires both completed extensions |

Eight threads are only the measured DINO extraction resource choice; head fitting stays at its recorded resource settings. `--encoder-threads 4` is available for a smaller CPU. Hardware throughput choices are not selected using labels and are recorded, but different numerical libraries/threading can still produce small numerical differences.

All subprocess arguments are passed as argument lists; spaces in paths are supported. Reports, models and logs are written only below the fresh workspace. Existing completion markers are respected. Partial nonresumable baseline/factorial outputs are preserved and cause an error; use another fresh run directory instead of deleting evidence. CAMELYON can reuse already complete, hash-verified shards; partial transfer files are retained.

### Supply checkpoints locally instead of downloading

The DINO directory must contain **all three** pinned files: `model.safetensors`, `config.json`, `preprocessor_config.json`.

```bash
python -B "$BUNDLE/reproduce.py" assets --run-dir "$RUN" --resnet-checkpoint "/models/resnet18-f37072fd.pth" --dino-model-dir "/models/dinov2-pinned"
```

```powershell
& $Python -B "$Bundle\reproduce.py" assets --run-dir $Run --resnet-checkpoint "D:\models\resnet18-f37072fd.pth" --dino-model-dir "D:\models\dinov2-pinned"
```

If some files are local, `--download` may additionally retrieve only missing assets. Every accepted file must match its full recorded hash. No user account/token is supplied by this helper. It never writes the user's existing global model cache.

### Why the wrapper is necessary

The research manifests contain absolute Windows image/mask/bbox paths. Copying those CSVs to Linux does not make raw-data extraction portable. This route **rebuilds the manifests locally** from the exact ZIP and reviewed mapping, then validates the original path-independent cohort digest.

Filesystem enumeration may differ between Windows and Linux. The wrapper restores the predeclared sample-ID order and checks the aggregate digest before fitting. It changes no sample IDs, labels, groups, annotations or image bytes. It updates only the new manifest's own CSV checksum after reordering. A decoded-pixel or annotation mismatch stops processing instead of quietly producing a different cohort.

The original setup scripts assume sibling experiment directories and inspect previous outputs for preservation. Staging recreates that structure. The four core classifier/split/manifest modules remain **byte-identical**. Two explicit patches are made **only to new staged extension copies**: DINO's checkpoint directory and CAMELYON's ResNet checkpoint path are read from task-specific environment variables. Original/patched hashes and exact replacement text are recorded in `RUN/reproduction/stage.json`; no learning or scoring equation changes.

### CAMELYON-specific limits and access failures

The original author download is probed first. If it becomes accessible, the frozen fallback code stops for review rather than falsely claiming the community mirror is required. Otherwise the public mirror is `wltjr1007/Camelyon17-WILDS`, revision `d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f`; all21 shards must pass pinned LFS SHA-256 checks.

Hospitals are decoded from each file's ClassLabel names and checked against `patient//20` and `slide//10`. Source train hospitals0/3/4 alone select settings. Original source ID-validation images are excluded. A label-blind identity hash selects at most400 patches per patient/partition; required missing class coverage is an error, not a reason to resample using target labels. All27 source-trained heads are frozen before scoring primary hospital2 or secondary hospital1.

This is a patient-capped frozen ImageNet-transfer subset from a disclosed community redistribution. It is **not** a full official WILDS leaderboard experiment. Patient bootstrap intervals hold the selected patch subsamples, encoder and trained heads fixed. Two observed target hospitals do not establish population-level hospital uncertainty.

## Level 2: provided features, unchanged main-head refits

The original PolypGen manifest must be accompanied by its matching `manifest.audit.json`. ResNet18 feature caches are at `features/resnet18_imagenet1k_v1.npz`; DINO caches are at `exp4_dinov2/features/dinov2_base_cls.npz`. These files can be provided separately where data/checkpoint terms permit. This helper does **not** open manifest image paths, so old absolute Windows paths in a supplied CSV do not break this limited level.

```bash
python -B "$BUNDLE/fit_cached_heads.py" --manifest "/provided/manifest.csv" --features "/provided/resnet18_imagenet1k_v1.npz" --seed 17 --output-dir "$PWD/cached_seed17"
python -B "$BUNDLE/fit_cached_heads.py" --manifest "/provided/manifest.csv" --features "/provided/resnet18_imagenet1k_v1.npz" --seed 29 --output-dir "$PWD/cached_seed29"
python -B "$BUNDLE/fit_cached_heads.py" --manifest "/provided/manifest.csv" --features "/provided/resnet18_imagenet1k_v1.npz" --seed 43 --output-dir "$PWD/cached_seed43"
```

```powershell
& $Python -B "$Bundle\fit_cached_heads.py" --manifest "D:\provided\manifest.csv" --features "D:\provided\resnet18_imagenet1k_v1.npz" --seed 17 --output-dir "D:\review\cached_seed17"
& $Python -B "$Bundle\fit_cached_heads.py" --manifest "D:\provided\manifest.csv" --features "D:\provided\resnet18_imagenet1k_v1.npz" --seed 29 --output-dir "D:\review\cached_seed29"
& $Python -B "$Bundle\fit_cached_heads.py" --manifest "D:\provided\manifest.csv" --features "D:\provided\resnet18_imagenet1k_v1.npz" --seed 43 --output-dir "D:\review\cached_seed43"
```

Substitute the pinned DINO cache for the same head experiment. The helper checks dimensions, finiteness, normalization, manifest digest/order, pinned checkpoint identity, DINO processor/revision, and exact source-only split feasibility. It records the provided cache hash and writes a fresh output directory. This verifies consistency of **provided features**, not the actual raw-image-to-feature computation. Compare supplied file hashes with the original artifact manifests. Cross-platform numerical equality is a measured result, not assumed.

## Level 1: predictions to metrics, tables and scientific plots

For CSVs without a seed column, explicitly supply `seed=path`. CSVs with their own seed column can be passed directly. Method/sample/center/group/label support must match exactly across all included methods and seeds.

```bash
python -B "$BUNDLE/metrics_from_predictions.py" --input "17=/provided/seed17/predictions.csv" --input "29=/provided/seed29/predictions.csv" --input "43=/provided/seed43/predictions.csv" --output-dir "$PWD/predictions_only_report"
```

```powershell
& $Python -B "$Bundle\metrics_from_predictions.py" --input "17=D:\provided\seed17\predictions.csv" --input "29=D:\provided\seed29\predictions.csv" --input "43=D:\provided\seed43\predictions.csv" --output-dir "D:\review\predictions_only_report"
```

For saved CAMELYON predictions, use one CSV with its seed column and add `--unit patient`. The helper recognizes the `hospital` column. For factorial predictions add `--key cell`. For threshold-control predictions use `--prediction-column adjusted_prediction --probability-column probability_class_1_unchanged`; probabilities/rankings remain those of the fixed original head even though classifications change.

This level produces per-seed/center metrics, mean/worst balanced accuracy, AUROC, separate optimization-seed SD, a paired whole-group BA bootstrap with accepted/rejected draw counts, `table.tex`, and PNG/PDF figures. Bootstrap attempts:2000, seed2027; intervals condition on class-complete draws of fixed fitted heads. Patient/sequence semantics are supplied by the input CSV and are not independently proved by this helper. The full scripts provide richer original reports and independent per-hospital CAMELYON intervals; the generic helper's combined-center summary is only a descriptive convenience.

## What to compare and archive

Keep environment install logs, `pip check`, `pip freeze --all`, Python/OS/CPU information, `reproduction/preflight.json`, staged patch hashes, source/checkpoint/data hashes, exact manifests/splits, chosen settings, models, predictions, completion markers and report files. A head-seed is an optimization replicate; do not count seeds or correlated patches as extra patients or hospitals.

Important completion markers:

- Baseline: each `results/seed{17,29,43}/COMPLETE`, `experiments_complete.json` and `report/summary_metrics.json`.
- Follow-ups: `exp1_failure/COMPLETE.json`, `exp2_factorial/COMPLETE`, `exp3_stress/COMPLETE.json`; preservation check must report unchanged.
- Extensions: `exp4_dinov2/COMPLETE`, `exp5_camelyon/COMPLETE`, `preservation_check.json`, `EXTENSIONS_COMPLETE.json`.

For a new OS, compare per-image decisions and raw score/probability differences with the original artifacts and report maximum absolute numeric differences. An exact replay can be checked at fixed provided features independently of feature extraction. Changed target predictions must be reported even if aggregate metrics happen to match.

Do not publish raw data, account tokens, old absolute paths, or identifying internal logs blindly. The historical provenance contains a Windows username/home path and may not be anonymous. Fresh local manifests solve relocation, but an anonymous submission also requires reviewing artifact names, logs and provenance before release. Respect data licenses; an archive/checkpoint hash is not permission to redistribute its contents.

## Optional appendix: earlier TFDS colorectal-histology pilot

This pilot is separate from experiments 1–5. Its 5,000 tissue patches/eight classes lack verified hospital/patient groups, so it does **not** establish cross-center generalization or validate center balancing. The staged `histology_pilot/` contains the actual pilot CLI and shared prototype code, with no cached data/results.

After level3 staging, change to `RUN/histology_pilot`. Either import a complete existing TFDS `colorectal_histology/2.0.0` cache (requires separately installed compatible TensorFlow/TensorFlow Datasets), or use a legitimate local original Kather ZIP (checksum checked by the importer). The ZIP route needs no TensorFlow:

```bash
cd "$RUN/histology_pilot"
python -m histology_cbmp.data --archive "/datasets/Kather_texture_2016_image_tiles_5000.zip" --extract-dir histology_cbmp/data/from_zenodo --manifest histology_cbmp/results/pilot/manifest.csv
python -m histology_cbmp.run --weights "$RUN/assets/torch/hub/checkpoints/resnet18-f37072fd.pth"
python -m histology_cbmp.report --results-dir histology_cbmp/results/pilot --output-dir "$RUN/histology_pilot/report"
```

```powershell
Set-Location "$Run\histology_pilot"
& $Python -m histology_cbmp.data --archive "D:\datasets\Kather_texture_2016_image_tiles_5000.zip" --extract-dir histology_cbmp/data/from_zenodo --manifest histology_cbmp/results/pilot/manifest.csv
& $Python -m histology_cbmp.run --weights "$Run\assets\torch\hub\checkpoints\resnet18-f37072fd.pth"
& $Python -m histology_cbmp.report --results-dir histology_cbmp/results/pilot --output-dir "$Run\histology_pilot\report"
```

For a complete local TFDS cache, replace only the first importer command with `python -m histology_cbmp.data --tfds-dir "/path/to/colorectal_histology/2.0.0" --extract-dir histology_cbmp/data/from_tfds --manifest histology_cbmp/results/pilot/manifest.csv`. Preserve its distinct source provenance. TensorFlow dependency installation and this optional portable pilot rerun were not validated by the core 1–5 environment.

## Maintainer-only source export

`freeze_sources.py` reads the original prepared sibling folders and refreshes only this reproduction package. Reviewers receiving the already exported package do not need those source folders and should **not** run the exporter. A maintainer can run `python -B freeze_sources.py --refresh` after source/report code is final. Existing staged runs bind to the previous bundle hash and will reject a changed export; stage a new run after refresh.
