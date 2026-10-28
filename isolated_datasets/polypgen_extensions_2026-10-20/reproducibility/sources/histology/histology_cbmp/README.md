# Colorectal histology prototype pilot

This pipeline compares four classifier heads on the real Kather colorectal histology dataset, available through the [official TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/colorectal_histology). The complete release contains 5,000 RGB tissue patches, each 150 × 150 pixels, across eight classes. The original study is [Kather et al., *Multi-class texture analysis in colorectal cancer histology*, 2016, DOI: 10.1038/srep27988](https://doi.org/10.1038/srep27988).

The experiment is a **within-dataset image classification pilot**. Its folder layout and TFDS records do not provide verified patient, slide, or hospital identifiers. It therefore cannot establish patient independence or generalization to an unseen hospital. The internal `single_dataset` marker with `beta=0` supplies the interface required by the shared prototype code; it does not represent hospital metadata or evaluate center balancing.

All four heads use the same cached, normalized, 512-dimensional features from a frozen [ResNet18](https://arxiv.org/abs/1512.03385), implemented in torchvision with ImageNet1K V1 weights and its prescribed preprocessing. The encoder runs in evaluation mode and receives no gradient updates. TFDS is the dataset source; feature extraction uses PyTorch/torchvision.

## Protocol

- Split unique decoded-pixel content groups into 60% training, 20% validation, and 20% test, stratified by tissue class, with seed `17`. Identical decoded images remain together even when their file encodings differ. Image counts can differ from these proportions if groups contain duplicates.
- Fit every classifier on training images only. Do not refit on validation images after selection.
- Select prototype count `K` from `{1, 2, 4, 8}` and linear regularization `C` from `{0.01, 0.1, 1, 10, 100}` using validation balanced accuracy. Ties prefer smaller values.
- Fit one positive temperature per head using validation negative log likelihood. Temperature changes predicted probabilities while preserving the highest-scoring class.
- Evaluate the fixed test partition after all fitting, selection, and calibration. Do not select a preferred configuration from test performance.

The expected heads are:

| Output name | Classifier |
| --- | --- |
| `single_prototype` | One normalized prototype per tissue class |
| `multiple_prototypes_selected` | Validation-selected prototype count; selection may choose `K=1` |
| `linear_classifier` | Class-balanced logistic regression with validation-selected `C` |
| `cosine_classifier` | Learned normalized cosine head with fixed optimization settings |

This pilot checks the implementation and provides tissue-recognition measurements. It does not, by itself, establish novelty or validate the manuscript's proposed center-generalization claim. Reported measurements must come from completed runs.

## Run on Windows

Use the existing Python environment containing the required libraries. `requirements.txt` lists them; avoid reinstalling the full repository or downloading the dataset again when its complete local cache is already available. Importing the TFDS records needs TensorFlow in that environment. A missing ResNet18 checkpoint is downloaded from the official PyTorch model host; an existing verified checkpoint is reused.

Open **Command Prompt (`cmd.exe`)** and enter the repository:

```bat
cd /d "C:\Users\Lenovo\Documents\Codex\2026-10-14\https-github-com-fortunab-family-baselines\work\family_baselines"
```

Import the complete existing TFDS cache into image folders, then create its audited manifest:

```bat
python -m histology_cbmp.data --tfds-dir "C:/Users/Lenovo/tensorflow_datasets/colorectal_histology/2.0.0" --extract-dir histology_cbmp/data/from_tfds --manifest histology_cbmp/results/pilot/manifest.csv
```

The import uses local cached records. It creates provenance files and verifies the expected image and class coverage; it does not initiate a TFDS dataset download. Keep the existing cache intact. If the image import has already completed, rebuild only the audit from the extracted image folders:

```bat
python -m histology_cbmp.data --data-root histology_cbmp/data/from_tfds --manifest histology_cbmp/results/pilot/manifest.csv
```

Run feature extraction and the four heads using the defaults above:

```bat
python -m histology_cbmp.run
```

Generate figures and a LaTeX table from the saved results:

```bat
python -m histology_cbmp.report --results-dir histology_cbmp/results/pilot --output-dir "C:/Users/Lenovo/Documents/Codex/2026-10-14/https-github-com-fortunab-family-baselines/outputs/histology_pilot"
```

The report command requires a completed run. It writes publication artifacts locally and does not modify the open manuscript or upload anything to Overleaf, W&B, or another service.

## Alternative: original local ZIP

If using the original Zenodo release instead of TFDS, provide the downloaded ZIP and a new extraction destination:

```bat
python -m histology_cbmp.data --archive "D:/datasets/Kather_texture_2016_image_tiles_5000.zip" --extract-dir histology_cbmp/data/from_zenodo --manifest histology_cbmp/results/pilot/manifest.csv
```

Replace the archive path with its actual location. The importer checks the official archive MD5 before extraction and audits the resulting images. Use this data source as an alternative to the TFDS command, then run the same classifier and report commands. Preserve each run's source provenance when comparing results.

## Saved files

`histology_cbmp/results/pilot/` contains:

- `manifest.csv` and `manifest.audit.json`: audited inputs, identities, labels, provenance, and limitations.
- `splits.csv`: exact training, validation, and test membership, including byte and decoded-pixel hashes.
- `selection.json`: validation search scores for the candidate `K` and `C` values.
- `metrics.json`: selected settings, calibration temperatures, validation/test metrics, and confusion matrices.
- `predictions.csv`: per-image test labels, predictions, and probabilities for each head.
- One `.joblib` file per head: fitted classifier and associated calibration information.
- `run_metadata.json`: encoder, preprocessing, software versions, arguments, and source fingerprints.

The feature cache is stored separately under `histology_cbmp/cache/`. Its metadata must match the audited inputs, encoder weights, and preprocessing. Changed image bytes are rejected before extracting or reusing features.

The report output includes comparison and confusion figures in PDF/PNG, `histology_pilot_table.tex`, a readable report, and copies of the metrics and run metadata. Upload the required `.tex` and PDF figure files to Overleaf after reviewing the completed run.

## Preserve completed runs

The runner refuses to overwrite an existing split/result record. For a distinct, justified run, use a new results directory. The report checks for its audit beside the results and can also locate an audit stored elsewhere using the manifest path recorded in `run_metadata.json`. The following example keeps each manifest/audit beside its results:

```bat
python -m histology_cbmp.data --data-root histology_cbmp/data/from_tfds --manifest histology_cbmp/results/pilot_repeat/manifest.csv
python -m histology_cbmp.run --manifest histology_cbmp/results/pilot_repeat/manifest.csv --results-dir histology_cbmp/results/pilot_repeat
python -m histology_cbmp.report --results-dir histology_cbmp/results/pilot_repeat --output-dir "C:/Users/Lenovo/Documents/Codex/2026-10-14/https-github-com-fortunab-family-baselines/outputs/histology_pilot_repeat"
```

These commands retain seed `17` and the same split. Do not search seeds or configurations using test scores. Fresh result directories preserve prior evidence; they do not make repeated test-driven selection valid.

## Local checks

From the repository root, run the data-audit tests and runner protocol tests separately:

```bat
python -m unittest discover -s tests -p "test_histology_data.py" -v
python -m unittest discover -s histology_cbmp/tests -p "test_histology_run.py" -v
```

These tests use small generated fixtures solely to check parsing, auditing, grouping, cache prerequisites, and calibration invariants. Fixture images are never experiment data. The checks do not replace running and inspecting the real-data pilot.
