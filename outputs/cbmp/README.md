# CBMP research implementation

This package implements the proposed frozen-encoder CBMP pipeline. It is a research prototype, not a validated clinical system or evidence of a novel/publishable result. The paper draft remains a specification until actual results are available.

## Install and verify

From this directory:

```powershell
python -m pip install -e ".[test,encoder]"
python -m pytest -q
python -m cbmp --help
```

The current environment already contains the numerical and encoder dependencies. Installing the package is optional when commands run from this directory. A CUDA device is optional; feature extraction also supports CPU.

## 1. Obtain data and audit its metadata

Official data: https://www.synapse.org/Synapse:syn26376615

Official authors' repository: https://github.com/DebeshJha/PolypGen

Dataset paper: https://doi.org/10.1038/s41597-023-01981-y

Create a UTF-8 CSV manifest with exactly these required columns (additional columns are allowed):

```csv
sample_id,image_path,label,center,group_id
```

Each row represents one image. `image_path` is absolute or relative to the manifest. Binary labels are `0` (no polyp) and `1` (polyp present), verified from official annotations. Never include masks as input photographs. `center` is the original acquisition center. `group_id` must identify the highest known related unit (patient first, otherwise source video), be unique across centers, and group all related frames together. Do not invent patient IDs from individual frame IDs. The software cannot certify patient identities absent from released metadata. Missing or uncertain group mappings must be resolved before claiming patient-independent evaluation.

```powershell
python -m cbmp audit data/manifest.csv runs/audit
```

This checks readability, class encoding, group-center consistency, and exact file/decoded-pixel duplicates. It saves center/class counts, checksums, and identified limitations. It rejects duplicates rather than silently removing them. Near duplicates and label correctness still require dataset-specific review. There is **no automatic synthetic fallback**.

## 2. Extract deterministic frozen features

Use a compatible **local** pretrained timm checkpoint. No checkpoint or data is silently downloaded. The model's own evaluation transform is recorded, and the encoder runs in evaluation/inference mode.

```powershell
python -m cbmp extract data/manifest.csv runs/audit runs/features.npz --model vit_small_patch14_dinov2.lvd142m --checkpoint C:/path/to/checkpoint.pth --device cpu --batch-size 8
```

Confirm checkpoint architecture, model card, license, and any pretraining overlap with evaluation data. Some checkpoints need conversion to timm naming before loading; incompatible weights fail rather than fall back to random initialization. The cache stores ordered sample IDs, normalized features, center/group IDs, manifest/checkpoint SHA-256, transform, and library versions. A successful manifest audit establishes file integrity, not that the images are authentic clinical data; provenance must be established from their official source.

## 3. Run held-out-center experiments

```powershell
python -m cbmp run runs/features.npz runs/experiment_001 --seeds 0 1 2 --bootstraps 500
```

All centers are outer test folds. Source-only inner center folds select settings; invalid folds are recorded. If no complete-class center fold exists, a deterministic group-disjoint source holdout is attempted. This implementation explicitly requires all classes in both inner training and validation; this is stricter than the initial manuscript wording. If no valid group split exists, execution stops.

Methods:

| Name | Purpose |
|---|---|
| `cbmp` | K in 1/2/4/8; beta in 0/0.5/1; group weighting |
| `centroid` | K=1, beta=0; group weighting |
| `balanced_centroid` | K=1; beta selected; group weighting |
| `multiprototype` | K selected; beta=0; group weighting |
| `image_centroid` | K=1, beta=0; ordinary image weighting |
| `linear` | Weighted softmax linear head, L-BFGS |
| `cosine` | Weighted cosine head (fixed scale 10), L-BFGS |

Linear/cosine heads share group and class-balanced loss weights, three regularization candidates, and a 300-iteration optimizer budget. Cosine heads have normalized rows and no bias. Their convergence diagnostics are saved and must be checked before publication. Candidate budgets are disclosed, not identical between different method families. The main comparisons isolate beta and K; this version does not implement backbone fine-tuning, a separate domain-generalization algorithm, or all possible recent competitors.

CBMP uses five random restarts, at most 100 spherical updates, deterministic tie handling, and guarded empty-cluster reseeding. Exact canceling centroid vectors are reseeded; that degenerate case has no unique normalized mean. K is capped by image count, center-group count, and distinct embedding count. Within-class weights implement the manuscript formula; a center's total class weight is proportional to G_sc^(1-beta).

Each fold saves source/target/inner sample-ID manifests, all candidate validation scores, selected settings, calibration grid, models, per-image probabilities, metrics, bootstrap intervals, and seeds. A new output directory is required to prevent accidental overwrites. Interrupted runs remain marked `running`; export refuses them. CPU/Numpy BLAS thread counts may affect runtime and tiny floating-point differences across machines; versions and source hashes are recorded.

Bootstrap intervals describe sampling variability within an observed center conditional on a fitted model. They do not establish generalization to arbitrary future hospitals. AUROC is null for a target fold missing a class. Balanced accuracy then averages observed classes, while macro F1 retains the full label vocabulary; interpret such centers separately and report coverage.

## 4. Export to Overleaf

```powershell
python -m cbmp export runs/experiment_001
```

The `paper/` folder contains `summary.tex`, `summary.csv`, and `center_performance.pdf` / `.svg`. Copy the table and PDF into your Overleaf project; use `\input{tables/summary}` and `\includegraphics[width=\linewidth]{fig/center_performance.pdf}`. There is no external upload or automatic interpretation of scientific findings. The exporter averages seeds within centers, then gives centers equal weight. Raw per-fold results preserve seed variation and group bootstrap intervals.

## Explicit pipeline-only smoke test

```powershell
python -m cbmp demo-features runs/demo_features.npz
python -m cbmp run runs/demo_features.npz runs/demo_run --allow-synthetic --seeds 0 --bootstraps 20
python -m cbmp export runs/demo_run
```

This uses artificial embeddings, not PolypGen. Its exported artifacts carry a synthetic-test label. Never use the smoke scores as research results. Tests check centroid equivalence, center/group weights, frame-replication behavior, degenerate directions, deterministic fitting, classifier learning, target exclusion, target-label independence of selection, duplicate rejection, and artifact generation.

## Remaining scientific work

Obtain and reconcile official data/group annotations; verify pretraining overlap; run the real encoder; evaluate multiple seeds; inspect head convergence and errors; compare relevant recent domain-generalization methods; and add a second independently sourced dataset before making broad claims. After those experiments, reconcile the manuscript with exact code settings. The implementation does not establish novelty or a performance advantage.
