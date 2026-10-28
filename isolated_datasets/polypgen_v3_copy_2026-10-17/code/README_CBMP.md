# Center-balanced multi-prototype experiments

This is an experimental implementation of the paper's proposed CBMP method.
It is intentionally separate from the legacy random image-split baselines.
It requires real PolypGen data and never generates demo images.

## Official release and download

Use the [official PolypGen page](https://www.synapse.org/Synapse:syn26376615/wiki/613312).
Its revised download points to **syn45200214, version 1**, named
`PolypGen2021_MultiCenterData_v3.zip`. The downloader retrieves that file only;
it does not sync the older v2 archive from the project.

```powershell
python download_synapse_polypgen.py --dest-dir data/PolypGen --metadata-only
```

Metadata is readable anonymously, but anonymous file download was denied when
checked on 2026-10-16. In Synapse Account Settings create a Personal Access Token
with View and Download permissions, then enter it at a hidden local prompt:

```powershell
python download_synapse_polypgen.py --dest-dir data/PolypGen --extract --prompt-token
```

The archive remains under the destination, with extraction below its `extracted`
directory. Release metadata and checksums are recorded there. Keep the images
outside Git: the current official page prohibits redistribution.

The token is not saved by this script. Existing standard Synapse configuration
or `SYNAPSE_AUTH_TOKEN` can also be used by omitting `--prompt-token`. In Command
Prompt, use `dir` rather than `ls`, and run each command as a single line.
Only continue to the audit after extraction succeeds. Missing data or mapping
files now produce a concise prerequisite message instead of a traceback.

## Annotation-based manifest and leakage checks

The official directory layout includes `data_C1/images_C1`, `masks_C1`,
`bbox_C1`, and `bbox_image_C1`; sequences appear under
`sequenceData/positive/seqN` and `sequenceData/negativeOnly/seqN_neg`.
The paper also documents the `data-C1` and `seq_1` spelling variants.
Only original-image leaves are indexed. Masks, overlays, and the pooled
`imagesAll_positive` copy cannot become model inputs.

**A Positive sequence can contain polyp-free frames.** Frame labels therefore
come from the matched mask and/or bounding-box annotation, not the Positive
folder name. JPEG mask presence uses intensity >127 and at least one foreground
pixel by default; these fixed settings are recorded in the audit. When both
annotation types exist, they must agree. Missing annotations are not interpreted
as negative labels. Documented negative-only sequences can supply label 0
without annotations.

Sequence directory names do not establish imaging-center identity. Supply a
reviewed CSV with `sequence_id,center,source` and an optional `patient_id` column.
`source` records the official metadata or other independently verified evidence
for that assignment. Positive `seq1` and negative `seq1_neg` are distinct IDs.
Do not infer a center or patient from the sequence number.

An initial audit writes a template for unresolved sequences, along with a full
rejected-path CSV and audit JSON:

```powershell
python audit_cbmp_data.py data/PolypGen/extracted --manifest results/cbmp/manifest.csv --sequence-only
```

This command can exit with no usable rows until the mapping is filled. After
reviewing the release metadata and completing the template, re-run:

```powershell
python audit_cbmp_data.py data/PolypGen/extracted --manifest results/cbmp/manifest.csv --sequence-only --sequence-metadata results/cbmp/sequence_centers.csv
```

The first grouped experiment uses `--sequence-only` because still-image folders
alone do not expose video/patient IDs. Record this subset choice in the paper.
Sequence grouping cannot establish patient independence between two different
sequences. For a diagnostic pilot with stills, `--allow-image-only-groups`
explicitly permits image-level grouping without that guarantee.

The runner rejects legacy path-label manifests and unresolved center/annotation
errors. Duplicate content links whole groups rather than moving only the copied
frames; groups shared across source/target centers exclude that fold. Both
classes are required. The documented negative-only sequences cover four
hospitals, so six class-complete binary folds must not be assumed. Actual coverage
and excluded centers are saved before any model selection.

## Run

```powershell
python run_cbmp.py `
  --data-root data/PolypGen/extracted `
  --manifest results/cbmp/manifest.csv `
  --results-dir results/cbmp `
  --feature-cache cache/cbmp_features.npz `
  --sequence-only `
  --model-name dinov2_base `
  --device auto
```

The runner caches frozen image embeddings, performs nested source-only
selection for `K` and `beta`, selects the prototype temperature from source
validation predictions with equal weight per validation center, refits on all outer source centers, then evaluates
the held-out center. Baselines include image-weighted, group-weighted, and
center-balanced single centroids, unbalanced multi-prototypes, a trained
cosine head, and a tuned linear logistic classifier. Outputs include metrics by center, per-image
predictions, and the outer split manifest. Fold-specific prototype banks,
cosine weights, and logistic heads are saved under
`results/cbmp/models/<held-out-center>/`.

For separate optimization seeds, give each run a distinct results directory
and feature cache (for example `results/cbmp_seed17` and
`results/cbmp_seed29`). Preserve each `experiment_config.json`; the seed and
resolved model are recorded there. Do not overwrite run artifacts before
combining per-image predictions for multi-seed analysis.

Create paper artifacts from saved predictions without retraining:

```powershell
python report_cbmp.py results/cbmp/predictions.csv --bootstrap-draws 2000
```

This writes `summary_metrics.json`, `per_center_metrics.csv`,
`table_main.tex`, and PDF/PNG center-comparison figures. The bootstrap resamples
the group IDs recorded in the manifest. If those IDs represent only image
paths, its interval does not account for video-level correlation.

## Implementation status and limits

The code implements a frozen DINOv2 feature protocol and deterministic
spherical prototype fitting. It does not fine-tune the encoder, claim clinical
interpretability, or establish improved performance. Check that the installed
`timm` version can load the requested pretrained checkpoint and that the
resolved encoder is the intended one. The initial manifest parser must be
validated against the downloaded archive before using its results in a paper.

## Dataset citations

The official page currently requests all three publications for any data use:

- Ali et al., Scientific Data (2023), [10.1038/s41597-023-01981-y](https://doi.org/10.1038/s41597-023-01981-y).
- Ali et al., Scientific Reports (2024), [10.1038/s41598-024-52063-x](https://doi.org/10.1038/s41598-024-52063-x).
- Ali et al., Medical Image Analysis (2021), [10.1016/j.media.2021.102002](https://doi.org/10.1016/j.media.2021.102002).

The dataset DOI is [10.7303/syn26376615](https://doi.org/10.7303/syn26376615).
