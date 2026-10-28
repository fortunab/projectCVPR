# Independent CAMELYON17-WILDS frozen-transfer subset

Actual experiment completed with 27 source-trained heads and 54 target records. All 27 heads were saved and hashed before either target hospital was scored.

The protocol was fixed before this dataset's target scores. The [community mirror](https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS/tree/d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f) is pinned at `d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f`; all 21 full shards match their published LFS SHA256. It is not an author-official redistribution. Frozen ImageNet ResNet18 and the capped subset make this a custom transfer experiment, not a full WILDS benchmark or [official leaderboard submission](https://wilds.stanford.edu/submit/).

Training uses only original train patches from hospitals 0, 3, 4. All tuning and temperature calibration use inner leave-one-source-hospital-out validation. Original source ID-validation patches are excluded from image extraction and fitting. Hospital 2 is the primary independent test; hospital 1 is a secondary unseen hospital. Neither selects settings.

This task classifies tumor presence in lymph-node histology patches, with labels defined on the central 32-by-32 region of each 96-by-96 image. Heads are fitted anew using this dataset's source hospitals. This tests replication of the classifier mechanism on an independent task; it is not zero-shot transfer of PolypGen-trained classifiers.

Label-blind SHA256 ranking retains at most 400 patches per patient within each eligible partition, preserving all 43 available eligible patients and 17,200 selected patches. No replacement was chosen using class labels or scores. Equal patient caps can change class proportions relative to the original patch-balanced dataset.

## Cohort

|Partition|Hospital|Patches|Patients|Tumor patches|
|---|---:|---:|---:|---:|
|primary_test|2|3600|9|1072|
|secondary_test|1|3200|8|1089|
|source_train|0|2800|7|975|
|source_train|3|4000|10|772|
|source_train|4|3600|9|803|

## Primary and secondary results

Balanced accuracy is mean across fitting seeds 17, 29, 43. Parentheses give whole-patient bootstrap 95% intervals, using 2,000 attempts with trained heads fixed. These intervals reflect patients within the observed hospital; they do not represent uncertainty across new hospitals.

|Method|Primary hospital 2 BA|Secondary hospital 1 BA|Primary AUROC|Secondary AUROC|
|---|---:|---:|---:|---:|
|Linear logistic|88.9 (84.6--91.1)|80.3 (69.0--86.4)|95.1|91.7|
|Patient-weighted centroid|73.4 (68.2--77.8)|74.6 (61.9--81.8)|88.9|86.5|
|Trained cosine|87.9 (83.4--90.3)|79.2 (68.3--85.5)|94.9|91.5|
|Image-weighted centroid|82.5 (77.9--85.0)|75.5 (60.9--82.8)|91.6|88.7|
|Hospital-balanced centroid|75.2 (70.5--78.7)|75.1 (62.9--82.1)|89.7|86.8|
|Unbalanced multi-prototype|85.1 (81.0--87.5)|76.8 (69.8--83.1)|92.5|84.7|
|Weighted linear|84.6 (80.6--86.8)|82.7 (77.0--87.5)|93.1|90.0|
|Weighted cosine|86.0 (82.0--88.5)|81.5 (77.0--86.6)|93.5|90.5|
|CBMP|85.4 (80.8--88.1)|77.6 (70.7--83.9)|93.0|85.3|

Primary test CBMP minus linear logistic balanced accuracy: -3.48 percentage points. Interpret this fixed comparison with the paired patient-bootstrap intervals in `paired_cbmp_contrasts.csv`. No settings were changed after observing it.

## Validation and limits

- Per-file ClassLabel hospital and label names were decoded and checked against patient//20 and slide//10, using the [author's metadata generation logic](https://github.com/p-lambda/wilds/blob/main/dataset_preprocessing/camelyon17/generate_final_metadata.py) and [official dataset loader](https://github.com/p-lambda/wilds/blob/main/wilds/datasets/camelyon17_dataset.py). Coordinate identities and original image IDs were audited.
- Patient groups, source folds, class coverage and exact content duplicates were checked. Cross-hospital duplicated content, contradictory labels or duplicate-linked patients would stop the experiment.
- Every saved model reconstructs its target predictions, calibrated probabilities and metrics. The source freeze manifest predates all target scoring.
- AUROC, macro F1, sensitivity, specificity, NLL and Brier score, three-seed variability and patient bootstrap intervals are available in the accompanying CSV files.
- Only two target hospitals are observed. Descriptive means and minima across them are not five-hospital LOCO results or evidence of general robustness to every hospital.
- The bootstrap holds selected patients' patch subsamples, trained heads and encoder fixed; it does not include full model-selection or patch-subsampling uncertainty.

## Artifacts

- `../audit/`: full original metadata, selected label-blind cohort, per-file schemas and image/content hashes.
- `../fits/`: raw source candidate searches, all 27 final model objects and pre-target source freeze manifest.
- `../evaluation/`: all 54 metric records and per-image predictions.
- `summary_by_hospital.csv`: mean and standard deviation across fitting seeds.
- `patient_bootstrap_intervals.csv`, `paired_cbmp_contrasts.csv`: fixed-head within-hospital patient uncertainty.
- `figures/`: PNG and vector PDF figures; `table_camelyon.tex`: standalone Overleaf table.
