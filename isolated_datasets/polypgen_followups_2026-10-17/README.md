# PolypGen follow-up experiments 1–3

Read [RESULTS.md](RESULTS.md) for the integrated findings and [PROTOCOL.md](PROTOCOL.md) for the frozen exploratory design. Existing baseline runs, raw data, and manuscript files are preserved.

## Experiments

| Folder | Contents |
|---|---|
| `exp1_failure/` | Per-sequence errors, score/ROC diagnostics, source-only threshold controls, prototype group/center composition, retrieval examples, adjusted predictions and a TeX table |
| `exp2_factorial/` | Fixed K in {1,4} x beta in {0,1}, three fitting seeds, paired component effects, per-center results, models and TeX table |
| `exp3_stress/` | 48 predeclared retained-source conditions, six heads, source-only searches, exact sampling plans, matched group-budget controls, predictions, models and TeX table |
| `code/` | Unchanged copies of the baseline core implementation, with recorded hashes |
| `logs/` | Preserved factorial/stress console logs; experiment 1 also preserves `exp1_failure/processing.log` |

Experiment 3 fixes fitting seed 17 and varies source-subset seeds 101,211,307. Experiments 1–2 use fitting seeds 17,29,43. These random seeds do not add independent patients or hospitals.

## Preservation and provenance

- Inputs are read from sibling `polypgen_v3_copy_2026-10-17` without writing to it.
- `setup_complete.json` records the frozen protocol, copied core code, and baseline artifact hashes.
- `preservation_check.json` verifies existing files by size/mtime and baseline artifacts by SHA-256.
- `environment.json` records the analysis runtime versions.
- `artifacts_manifest.json` provides SHA-256 checksums for completed follow-up artifacts, excluding mutable interpreter/test caches and the ledger itself.
- `FOLLOWUPS_COMPLETE.json` records final experiment counts. Per-experiment completion/validation files contain the corresponding checks and limitations.

## Reproduction

Run these scripts only in a fresh follow-up workspace or use their explicit resume rules. The existing completed result folders and logs should be preserved.

```text
python setup_followups.py
python run_logged.py exp1_failure.py
python run_logged.py exp2_factorial.py
python run_logged.py exp3_stress.py
python setup_followups.py --check
python summarize_followups.py
```

The actual experiment 1 run captured its own processing log directly. Numeric fitting uses two threads per process, preserving the mathematical baseline fits while allowing parallel execution. No image feature extraction or encoder fine-tuning was performed.

These findings concern a partial reviewed PolypGen sequence subset and were designed after earlier test inspection. Independent confirmation and reconciliation of manuscript settings are necessary before publication claims. Patient independence is unverified; C2 negatives and C5 positives each occupy one sequence.
