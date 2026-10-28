# Isolated PolypGen processing and held-out-center experiment

All new work is contained in this folder. The original Downloads archive and existing project data, cached features, and outputs are preserved.

## Read these first

- `PROTOCOL.md`: predeclared subset, source-only selection, classifier grids, three fitting seeds, and reporting rules.
- `audit/README.md`: full release inventory, actual center/class/sequence coverage, annotation problems, exclusions, and limitations.
- `report/RESULTS.md`: completed three-seed results and their interpretation, once `make_report.py` succeeds.

## Outputs

| Folder/file | Contents |
|---|---|
| `extracted/` | Fresh extraction of the selected `(1).zip`; nested concatenated ZIP is kept unexpanded |
| `extraction_complete.json` | Archive SHA-256, extraction totals, copied source-code hashes |
| `code/` | Snapshot of the implementation used for these runs, plus its focused tests |
| `audit/` | Complete original-image annotation inventory, reviewed mapping CSV, strict manifest, class/sequence counts and eligibility |
| `plan/` | Outer and source-validation folds recorded before classifier fitting |
| `features/` | One shared frozen ResNet18 feature cache with checkpoint/preprocessing provenance |
| `results/seed17`, `seed29`, `seed43` | Predictions, selected heads, source-only searches, saved models, metrics, configurations and completion markers |
| `logs/` | Exact commands and live processing/training output for each run |
| `report/` | Final per-center and aggregate metrics, paired sequence bootstrap, figures, selected settings, Markdown evidence and Overleaf-ready TeX table |
| `validation.json` | Result of 40 focused snapshot checks |
| `preservation_check.json` | Final check of sizes and modification times for existing data/output/cache/results files |

## Commands used

From this directory in Command Prompt or PowerShell, run the following in order only in a new, empty experiment workspace. Existing completion/log/report paths are intentionally protected against accidental overwrite.

```text
python prepare_isolated.py
python process_audit.py
python check_preservation.py
python run_experiments.py
python make_report.py
python check_preservation.py --check
```

`check_preservation.py` records its baseline without `--check`; this run recorded it before classifier fitting. The guard compares file sizes and last-modified times, rather than hashing every protected file.

To repeat the classifier evaluation independently, use the snapshot's `code/run_cbmp.py` with an explicit new results directory and the existing audited manifest/cache. Preserve every finished run.

## Scope

This experiment trains classifier heads on frozen ImageNet ResNet18 features. The encoder is not fine-tuned. It evaluates four eligible held-out centers of a partial reviewed sequence subset. C1 is source-only and C3 is absent. Sequence-level grouping is available; patient independence is unverified. No current manuscript or existing Overleaf source is edited by these scripts.
