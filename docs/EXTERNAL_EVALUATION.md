# Optional official WILDS and pathology encoder evaluation

The adapter code is supplied separately from the approved existing-cache experiment. Its partition, cache-identity, completeness and no-validation-refit checks are tested with fixtures. **No complete official WILDS experiment or UNI/CONCH checkpoint experiment has been executed here.** End-to-end provider integration and numeric output require the actual inputs and an isolated, recorded provider runtime.

## Exact scope

Official CAMELYON17-WILDS v1.0 uses H0/H3/H4 training, H1 OOD validation, H2 test. The official adapter uses the complete `train`, `val`, and `test` index sets returned by WILDS, with no patient/patch cap and no use of `id_val`. H1 may select the head and temperature but is never added to final fitting. This differs from the custom study, which treats H1 as a second held-out target and selects settings using source centers alone.

The [WILDS submission rules](https://wilds.stanford.edu/submit/) require ten CAMELYON fitting seeds. They distinguish a standard setting that restricts external pretraining from a non-standard setting. **Frozen ImageNet, UNI, or CONCH pretraining is non-standard even when official splits and the official evaluator are used.** Report the official `acc_avg` and seed variation; additional BA/AUROC metrics are secondary. Do not label the custom subset a leaderboard result.

The selected encoder families are original **UNI (2024), ViT-L/16**, or **CONCH**. UNI2-h and CONCH-derived slide encoders are different models. Access is subject to [UNI](https://github.com/mahmoodlab/UNI) or [CONCH](https://github.com/mahmoodlab/CONCH) provider approval and terms. Obtain a local checkpoint through the provider, record its exact model revision, and keep authentication tokens out of manifests and logs. No script requests or stores tokens.

## Environment prerequisites

Use a separate environment for image extraction and the official evaluator, installed according to the selected provider's documented requirements. The minimal cached-fitting lock intentionally does not include torch, timm, conch, or wilds. Do not install a second, conflicting model runtime over the verified head-fitting environment.

For an original UNI environment, install the provider-compatible PyTorch/torchvision, timm and optional safetensors versions; for CONCH, check out an explicit official CONCH commit and install its declared environment (the official instructions use Python 3.10). Add WILDS in an environment that passes `python -m pip check`. Record the full package freeze, Python patch version, torch build/device and provider commit. **A resolved and tested lock for these optional model runtimes remains to be produced after the chosen checkpoint is available.** Listing a package name is not that validation.

The extraction command records the package versions, checkpoint SHA256, implementation source hash, preprocessing, batch size, threads, feature definition and sample order. It requires deterministic torch algorithms and checks an initial repeated batch bitwise. Such a check does not guarantee bitwise portability across hardware/platforms. Do not disable a deterministic-kernel failure silently; record and resolve it.

For the complete dataset, estimate memory and storage before fitting. A float32 cache occupies approximately `4 * N * dimension` bytes before archive compression; NumPy training slices and classifier optimization need additional memory. Begin full-dataset runs with one worker. Original UNI is a large encoder and can be slow on CPU. A successful small fixture does not benchmark its resource requirements.

## Bash command sequence

After activating the selected provider/evaluator environment and making the complete official dataset available locally:

```bash
python -m pip check
python -B scripts/prepare_official_wilds.py --data-root /srv/datasets/wilds --output-dir reproduction_runs/wilds-official-audit
python -B scripts/extract_pathology_review_features.py \
  --manifest reproduction_runs/wilds-official-audit/manifest.csv \
  --image-root /srv/datasets/wilds/camelyon17_v1.0 \
  --encoder uni --checkpoint /srv/models/uni/pytorch_model.bin \
  --model-revision EXACT_PROVIDER_REVISION \
  --task camelyon_official_uni --protocol official_ood_validation \
  --official-audit reproduction_runs/wilds-official-audit/official_audit.json \
  --output-dir reproduction_runs/wilds-official-uni-features --device cpu --batch-size 16 --threads 4
```

Replace `EXACT_PROVIDER_REVISION` with the actual pinned provider revision; do not use `main` or `latest`. The data root is the **parent** of `camelyon17_v1.0`. Extraction rechecks the hash of every image against the audited manifest. Missing local images fail; these commands do not download the dataset.

Switch to the locked head-fitting environment:

```bash
source .venv-cbmp-review/bin/activate
python -B scripts/run_external_review.py --task-spec reproduction_runs/wilds-official-uni-features/task.json --run-dir reproduction_runs/wilds-official-uni-heads --prepare-only
python -B scripts/run_external_review.py --task-spec reproduction_runs/wilds-official-uni-features/task.json --run-dir reproduction_runs/wilds-official-uni-heads --resume
python -B scripts/report_review_revision.py --run-dir reproduction_runs/wilds-official-uni-heads --output-dir reproduction_runs/wilds-official-uni-secondary
```

Switch back to the WILDS evaluator environment, then:

```bash
python -B scripts/evaluate_official_wilds.py --data-root /srv/datasets/wilds --task-spec reproduction_runs/wilds-official-uni-features/task.json --run-dir reproduction_runs/wilds-official-uni-heads --output-dir reproduction_runs/wilds-official-uni-evaluation
```

The evaluator checks the complete test indices, dataset metadata, per-row IDs/labels, ten-seed completion and prediction hashes. It calls the WILDS dataset's own `eval` on predictions in official order and reports every head. It never chooses a head from test results. The research team has already inspected the custom H2 results; switching to official splits does not erase that history. The output identifies this as replication on previously inspected data, without asserting a prospectively blinded test or formal leaderboard compliance.

## Equivalent PowerShell sequence

`$ProviderPython` is the absolute interpreter path for the separately installed provider/WILDS environment. `$ReviewPython` is the locked `.venv-cbmp-review\Scripts\python.exe`. Paths below are examples to replace with actual local inputs.

```powershell
$ProviderPython = 'C:\envs\pathology\python.exe'
$ReviewPython = (Resolve-Path '.\.venv-cbmp-review\Scripts\python.exe').Path
& $ProviderPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Provider environment has dependency conflicts' }
& $ProviderPython -B scripts/prepare_official_wilds.py --data-root 'D:\datasets\wilds' --output-dir reproduction_runs/wilds-official-audit
if ($LASTEXITCODE -ne 0) { throw 'Official dataset audit failed' }
& $ProviderPython -B scripts/extract_pathology_review_features.py `
  --manifest reproduction_runs/wilds-official-audit/manifest.csv `
  --image-root 'D:\datasets\wilds\camelyon17_v1.0' `
  --encoder uni --checkpoint 'D:\models\uni\pytorch_model.bin' `
  --model-revision EXACT_PROVIDER_REVISION `
  --task camelyon_official_uni --protocol official_ood_validation `
  --official-audit reproduction_runs/wilds-official-audit/official_audit.json `
  --output-dir reproduction_runs/wilds-official-uni-features --device cpu --batch-size 16 --threads 4
if ($LASTEXITCODE -ne 0) { throw 'Feature extraction failed' }
& $ReviewPython -B scripts/run_external_review.py --task-spec reproduction_runs/wilds-official-uni-features/task.json --run-dir reproduction_runs/wilds-official-uni-heads --prepare-only
if ($LASTEXITCODE -ne 0) { throw 'External plan validation failed' }
& $ReviewPython -B scripts/run_external_review.py --task-spec reproduction_runs/wilds-official-uni-features/task.json --run-dir reproduction_runs/wilds-official-uni-heads --resume
if ($LASTEXITCODE -ne 0) { throw 'External fitting failed' }
& $ReviewPython -B scripts/report_review_revision.py --run-dir reproduction_runs/wilds-official-uni-heads --output-dir reproduction_runs/wilds-official-uni-secondary
if ($LASTEXITCODE -ne 0) { throw 'Secondary reporting failed' }
& $ProviderPython -B scripts/evaluate_official_wilds.py --data-root 'D:\datasets\wilds' --task-spec reproduction_runs/wilds-official-uni-features/task.json --run-dir reproduction_runs/wilds-official-uni-heads --output-dir reproduction_runs/wilds-official-uni-evaluation
if ($LASTEXITCODE -ne 0) { throw 'Official evaluator failed' }
```

## CONCH or the existing custom cohort

For CONCH, use the same commands with `--encoder conch`, its local checkpoint and exact revision, and a distinct `camelyon_official_conch` task/output name. The adapter uses `encode_image(proj_contrast=False, normalize=False)` and L2-normalizes that visual feature for the common heads. It does not substitute the image-text projection or a prompt classifier.

To compare a new pathology encoder on the existing custom cohort, supply its audited `exp5_camelyon/audit/image_manifest.csv`, `--protocol custom_source_only`, a distinct `camelyon_custom_uni` or `camelyon_custom_conch` task, and omit `--official-audit`. Its local absolute `image_path` entries must resolve; on a relocated checkout, rebuild and audit a manifest with explicit local paths and preserved IDs/content, rather than guessing patient linkage. `run_external_review.py` then uses source-only validation and both custom targets. The official evaluator rejects this custom protocol.

Pretraining overlap with the evaluation slides is a separate scientific check. A public encoder or a fresh cache does not establish that there is no overlap. Document the provider's training sources and any unresolved overlap. None of these experiments supplies new independent hospitals.
