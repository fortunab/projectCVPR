# Revision protocol following the supervisory review

Updated on 7 November 2026 after explicit approval of code, tests and existing-cache runs. All 30 task/seed jobs completed under `reproduction_runs/review_2026-11-07_full/`, yielding 2,200 evaluations and 3,793,240 predictions. The revised manuscript uses the completed ten-seed reports; original three-seed records are preserved separately. Extensions requiring new inputs remain clearly identified below.

## Issues and planned actions

| Observation | Correction using the existing evidence | Work requiring implementation or new data |
|---|---|---|
| Reviewed PolypGen subset | State the inclusion rule, target-specific group counts, missing center support, and limits of representativity. | Resolve excluded center mappings and obtain independently verified patient linkage before constructing patient splits. |
| Four endoscopy targets and two histology targets | Explain that C1 has negative-only support and C3 is excluded after mapping review. The available CAMELYON release has five hospitals, three serving as sources in the current protocol. | An additional external hospital cohort is needed for confirmation beyond these inspected targets. More patches at the same hospitals do not add hospitals. |
| Three fitting seeds | Refit with ten fixed seeds and report fitting-seed variation separately from patient/site sampling. | Completed on all three caches. More seeds do not add patients or hospitals. |
| Unequal search grids | Retain disclosed original grids and add four-candidate sensitivity with fixed centroid controls. | Completed, including paired grid contrasts and actual unique fit counts/times. Equal counts do not equalize coverage or compute. |
| Custom CAMELYON protocol | Identify H0/H3/H4 as sources and H2/H1 as primary/secondary targets in the custom study. | Add an official-split evaluation with H1 used for validation, H2 reserved for testing, the WILDS dataset/evaluator, and ten fitting seeds. |
| Missing pathology foundation encoder | State that the current histology results use an ImageNet ResNet18. | Add frozen UNI or CONCH embeddings with the same manifests and heads after access to the selected checkpoint is available. |
| Missing modern DG training baseline | Define the present scope as a frozen-head comparison. | Choose a task-compatible DG implementation with an official code source and a documented optimization budget. A method requiring backbone updates needs its own controlled training experiment and resource estimate. |
| Frame/patch endpoints | Add source-thresholded sequence-presence and equal-patient patch summaries, with paired uncertainty. | Completed. C2 sequence BA/AUROC are undefined; clinical patient/slide diagnosis still needs independent entity labels. |
| Source-composition stress test | Present it as exploratory resampling of observational source data. Document residual differences in frames and class prevalence. | A controlled generative or simulated factorial study would require a separate design and evaluation. |
| Post-hoc mechanism analysis | Preserve its exploratory status, paired counts, support limitations, and target-specific interpretation. | Prespecify the mechanism contrast on an untouched patient-linked cohort. |
| Environment and assurance | Add platform wheel-hash locks, fresh Windows dependency checks, 44 tests, exact portable refitting and lossless prediction reconstruction. | Windows verified; Linux wheel closure checked, Linux execution/container not claimed. Author clinical/code sign-off remains an author action. |

## Approved implementation and verification

The implementation below is now present. See [RUN_REVIEW_REVISION.md](RUN_REVIEW_REVISION.md) for Windows/Bash/conda commands and [EXTERNAL_EVALUATION.md](EXTERNAL_EVALUATION.md) for optional provider/data prerequisites. The grids and aggregate endpoint choices were specified after inspection of the original targets; they are an exploratory extension, not untouched confirmation.

1. `run_review_revision.py` and `report_review_revision.py` support fitting seeds 17, 29, 43, 61, 89, 101, 131, 173, 211, and 257, with immutable input/code plans, per-seed logs, raw predictions, and completion fingerprints. Original three-seed files remain intact.
2. Add a budget sensitivity protocol with four distinct source-selected configurations per tunable family. Use CBMP K in {1,4} and beta in {0,1}, unbalanced prototypes K in {1,2,4,8}, linear C in {0.1,1,10,100}, cosine learning rates in {0.003,0.01,0.03,0.1}, weighted linear C in {0.1,10} crossed with beta in {0,1}, and weighted cosine learning rates in {0.01,0.1} crossed with beta in {0,1}. Centroid controls have one fixed configuration. These equal candidate counts do not imply equal computation. The reduced grids change coverage relative to the original experiment, so this is an additional exploratory sensitivity analysis.
3. Add group reporting with fixed source-selected decisions. For PolypGen, define sequence presence as any audited positive frame and score a sequence with a prespecified maximum frame score. Source groups alone select any operating threshold. C2 has two positive-bearing sequences and no negative-only sequence under this label, so binary sequence AUROC and balanced accuracy are undefined there.
4. For CAMELYON, report each patient's patch metrics and their equal-patient summary at each target. All 17 target patients contain sampled positive patches. An any-positive label inferred from these patch samples would produce only the positive patient class. A clinical patient/slide classifier therefore requires a separate ground-truth manifest.
5. `prepare_official_wilds.py`, `run_external_review.py`, and `evaluate_official_wilds.py` implement the separate official-split path. H1 selects settings and never enters final fitting; H2 supplies test predictions. The evaluator requires the full original indices and ten seeds. A frozen externally pretrained encoder is explicitly non-standard pretraining. Fixture tests pass; no complete official dataset run has been executed.
6. `extract_pathology_review_features.py` implements original UNI and CONCH's unprojected visual embedding from a local checkpoint, saving revision, checkpoint hash, preprocessing, feature definition, runtime and row order. Actual checkpoint loading/extraction is unexecuted without the selected provider runtime and approved weights. Tokens remain outside this interface and the logs.
7. Eleven targeted tests pass for source/target and group/content separation, four-candidate grids, unsupported aggregate labels, external cache identity, full official support, no H1 refitting, bootstrap pair arithmetic, and permutation-independent prototype matching. A real 36-cell seed-17 replay also matches all 46,989 historical predictions, probabilities, and selected configurations exactly. Evidence: `outputs/revision_2026-11-07/smoke_replay_verification.json`.
8. Platform wheel hashes are in `requirements/review-windows.lock` and `review-linux.lock`. Fresh Windows fitting passes `pip check`; the portable seed-17 refit matches all 46,989 historical probabilities exactly. Linux wheel closure was checked, without claiming Linux execution/container validation. `audit_review_stability.py` reports 12,600 class-bank comparisons, 5,390 nearest-source examples and 1,902 analytic centroid checks, without claiming clinical semantics. The release verifies exact original CSV reconstruction from compressed prediction objects.

Increasing fitting seeds quantifies fitting variability within the same cohort. It supplies no additional independent patients or hospitals. Official-split replication reuses a dataset already inspected here and must not be described as untouched external confirmation. Complete external cohort acquisition, patient linkage, modern DG training, and actual human semantic/code review remain separate work.

The modern DG comparison needs a separate method and implementation selection. The immediate engineering proposal above prioritizes expanded fitting seeds, comparable candidate counts, supported aggregate metrics, and a pathology encoder. Its completion alone would not establish coverage of all contemporary DG approaches.

## Access and evidence prerequisites

The unexecuted independent-cohort and modern-DG work is specified in [CONFIRMATION_PROTOCOL.md](CONFIRMATION_PROTOCOL.md). It distinguishes an external confirmation design, a rotation of already inspected hospitals, an official-split replication and a controlled simulation.

- Public model descriptions and code do not establish access to UNI/CONCH weights. Check the specific model access and terms before downloading a checkpoint.
- A valid external cohort needs auditable labels, patient identifiers, hospital identifiers, permitted research use, and an inspection history. Any overlap with model pretraining or the current cohorts must be recorded.
- Patient linkage cannot be inferred from similar filenames, embeddings, or a shared sequence label.
- The reported results remain attached to their original manifests, checkpoints, settings, and prediction hashes. New protocols require their own reports.

## Primary references for the proposed extensions

- [UNI implementation and access instructions](https://github.com/mahmoodlab/UNI).
- [CONCH implementation, visual embedding definition, and access instructions](https://github.com/mahmoodlab/CONCH).
- [WILDS submission rules, validation/test use, and replicate counts](https://wilds.stanford.edu/submit/).
