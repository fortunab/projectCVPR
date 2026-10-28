# CAMELYON17-WILDS acquisition and protocol review

Reviewed 2026-10-20. No dataset images or performance metrics were downloaded/read during this review.

## Authoritative protocol

- [Official dataset loader](https://raw.githubusercontent.com/p-lambda/wilds/main/wilds/datasets/camelyon17_dataset.py): binary tumor classification of 96 x 96 histology patches, with tumor determined in the central 32 x 32 region; hospital IDs 0--4; 50 slide IDs.
- Official split uses source hospitals 0, 3, 4, validation hospital 1, test hospital 2. Source patch split 0 is training, source split 1 is ID validation. Do not use the alternate mixed-to-test scheme.
- [Official metadata generation](https://raw.githubusercontent.com/p-lambda/wilds/main/dataset_preprocessing/camelyon17/generate_final_metadata.py) derives hospital as integer patient_id // 20 and hospital as slide_id // 10. Check both identities against redistributed metadata.
- The original ID validation split is assigned at patch level. It cannot establish patient-independent validation. Use source-only leave-one-hospital-out selection on the source training patches; group/weight/bootstrap at patient level.
- [Official submission policy](https://wilds.stanford.edu/submit/) excludes external pretraining for CAMELYON17 leaderboard submissions. Our frozen ImageNet features therefore define a separate transfer experiment, not an official leaderboard result.

## Access status

The loader points at a 10,658,709,504-byte CodaLab bundle:

https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/

That URL was not accessible through the web tool. Open [issue 175](https://github.com/p-lambda/wilds/issues/175) and [issue 174](https://github.com/p-lambda/wilds/issues/174) document CodaLab download failures in 2026. The Stanford download page still links to CodaLab. No author-official replacement mirror was found in this review.

No filenames matching camelyon or wilds were found by the bounded workspace/Downloads inventory.

## Redistributed fallback, if used

The community mirror [wltjr1007/Camelyon17-WILDS](https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS) retains images, labels, hospital, patient, node, coordinates, slide and original image_id. Pin revision:

`d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f`

It contains 14 training parquet shards (302,436 rows), 3 validation shards (68,464 rows), and 4 test shards (85,054 rows), totaling 455,954 patches. The validation split combines source ID validation and hospital 1 OOD validation; separate by hospital. Metadata must be validated against the original loader and generation logic. Record that this is a community mirror; exact byte identity to the unavailable original archive has not been established.

Direct pinned file pattern:

`https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS/resolve/d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f/data/{split}-{index:05d}-of-{count:05d}.parquet`

Repository-published SHA256 examples:

- train-00000-of-00014.parquet: `9e6f5425a371de015b9bd1bfcd6901554a8b4b4ab7df268f768e2629a55f2829`
- test-00000-of-00004.parquet: `da59d8c9c7aa9afa19a64fc26ed787d0fac601155b6105774182d07e042c86f2`

HTTP-range parquet reads could acquire metadata without fetching image columns, then selected image row groups. Verify actual range support first. A parquet row group contains many image records, so a 20,000-image subset can still require substantial transfer. Do not silently change the subset to convenient first rows or shards.

## Recommended independent confirmation

Freeze the complete protocol before target scores: deterministic label-blind patch sampling within each (patient, node), fixed per-slide budget, all hospitals represented, preserve original split memberships. Audit labels and class coverage after sampling. Fit on source train split only, select all hyperparameters and thresholds by hospital-level source validation on 0, 3, 4, and evaluate hospitals 1 and 2 separately after settings are fixed. Report hospital 2 as the designated untouched test, hospital 1 as an additional independent hospital result. A different all-hospital outer LOCO benchmark would be custom and would no longer leave hospital 2 permanently untouched.

Use the same frozen ResNet18, nine classifier variants, three seeds and source grids already declared for the PolypGen experiments. Preserve patient IDs as highest available grouping unit. Report balanced accuracy, AUROC, macro F1 and patient-bootstrap intervals; include patch and patient counts per hospital. Label a reduced deterministic sample explicitly as a subset experiment rather than a full WILDS benchmark. Store sampling manifest, metadata checks, acquisition source revision, downloaded file or extracted image hashes, predictions, selected settings and model artifacts in this new experiment directory.
