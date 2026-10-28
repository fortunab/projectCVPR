# Extensions 4–5: frozen second encoder and independent hospital evaluation

Frozen before new feature extraction or classifier fitting on2026-10-20. Earlier PolypGen results have been inspected, so the second-encoder comparison is exploratory replication. The independent-dataset design is prespecified before its target scores are evaluated. No choices change in response to target performance.

## Preservation and shared classifier implementation

All previous project data, outputs, manuscript, baseline and follow-up runs remain read-only. New files stay under this extension folder. Copy and hash the existing core classifier code, preserve its candidate grids, five prototype restarts, max100 spherical updates,300 cosine fitting steps, equal validation-center balanced-accuracy selection, and bounded continuous source-NLL temperature selection. Record protocol/code/checkpoint/data provenance, exact partitions, selected settings, predictions, model objects, timing and validation checks.

Optimization seeds17,29,43; numeric threads4 per main process with concurrent extraction resource limits documented. Seed variation is separate from independent patient/sequence sampling uncertainty. No target labels select models, transforms, cutoffs, candidate configurations or dataset subsets. Do not call a small patch subset a full benchmark result or compare it directly with leaderboard scores.

## Experiment4: DINOv2 on the unchanged PolypGen cohort

- Reuse exactly the original audited5,444-frame37-sequence cohort, four original held-out centersC2,C4,C5,C6, and original source-only inner folds. C1 remains source-only; C3 absent. Existing image/annotation hashes and manifest digest must match.
- Frozen `facebook/dinov2-base`, immutable HuggingFace revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`. Cached config and346,345,912-byte safetensors weights are read-only; copy or read without modifying global caches. Record SHA-256 for checkpoint, config, and official preprocessing configuration from the same immutable revision.
- Use the checkpoint's official image processor in deterministic evaluation mode, `use_fast=False`, with no target-data fitting. Use the model's documented final CLS/pooler embedding,768dimensions, L2 normalized. Verify zero/invalid outputs and eval/frozen parameters. No silent fallback to another model or architecture.
- Resolve and record the exact processor settings and batch/worker resource benchmark before long extraction; batch size is a throughput setting, not selected by labels. Cache one common feature matrix for all methods/seeds. DINOv2's prescribed preprocessing is part of this encoder package; do not attribute differences solely to architecture.
- Run all nine original classifier variants with unchanged source grids and all three fitting seeds:108 outer evaluations. Repeat the fixed K in{1,4}, beta in{0,1} factorial with the same three seeds:48 fixed-cell evaluations. Source-only calibration for each fixed cell; no target-selected cell.
- Compare DINOv2 and ResNet18 on identical sample/label/group support. Report per-center and equally weighted mean/worst-center scores, seed variation, paired whole-sequence bootstrap where applicable (2,000 attempts, seed2027, record undefined/class-incomplete draws). Patient independence remains unverified, and C2 negatives/C5 positives each occupy one sequence.

## Experiment5: independent CAMELYON17-WILDS transfer subset

### Access and cohort selection

- First try the author-documented labeled bundle. If unavailable, use public community redistribution `wltjr1007/Camelyon17-WILDS` pinned at `d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f`; explicitly disclose this provenance and retain downloaded-file SHA-256/expected LFS hashes and original metadata. Do not present a community mirror as an author-official distribution.
- Keep public downloads in `exp5_camelyon/` only. The full redistribution is about10.7GB in21Parquet shards; metadata-first/range reads may reduce transfer, but actual transferred bytes are recorded. No credentials or gated access are assumed. Missing network permissions require the environment's permission mechanism.
- Preserve original image IDs, patient/node, slide, coordinates, label and hospital. Decode per-split ClassLabel hospital names rather than assuming integer encodings retain original hospital IDs. Verify author-defined hospital relationships `center == patient//20 == slide//10`, unique patch IDs/coordinates, original split support, and patient group boundaries. Tumor labels refer to the central32x32 region of a96x96 patch.
- Training: only original `train` rows from source hospitals0,3,4. Original ID-validation patches are retained as provenance only, not silently merged into training. Hospital1 original OOD validation becomes a prespecified secondary unseen-center evaluation; hospital2 original test is the primary unseen-center evaluation. Neither hospital1 nor2 selects settings.
- CPU-bounded subset chosen prospectively: at most400 patches per patient in each eligible original partition, selected by ascending SHA-256 of `camelyon17-patient-cap400-seed2027|original_patch_identity`, without using labels or model scores. Use all patients available in the eligible partitions; no patient is selected for favorable class coverage. Exclude unused ID-validation patch images from extraction. Record selected patients, exact counts, class coverage and excluded rows before fitting. If label-blind selection lacks an entire class in any required source fold or target, stop and report infeasibility; do not replace it based on target labels.
- Download only selected image row groups where reliable Parquet range access permits; otherwise keep pinned shards and extract selected images. Never fabricate or substitute images/labels when acquisition fails. This is a declared computation-limited subset, not full CAMELYON17-WILDS.

### Features, fitting and held-out evaluation

- Frozen official torchvision ResNet18 ImageNet1K V1 checkpoint, prescribed deterministic preprocessing, L2 normalized512dim embeddings. Record image bytes/decoded hashes, checkpoint identity and extraction metadata. This isolates dataset replication of the original encoder/head protocol rather than jointly changing encoder and data.
- Group by documented patient (keep node/slide metadata for diagnostics), not arbitrary patches. Confirm no patient or exact content overlaps across source and held-out hospitals or inner folds. Audit exact duplicates; stop for cross-hospital duplicated content or contradictory labels. Within-patient repeated content can be recorded; distinct patients connected by duplicate content require linked groups and renewed split feasibility.
- Use source-only inner leave-one-source-hospital-out validation among0,3,4; require both classes. Select all nine original heads with unchanged grids and calibration; refit on all retained source-train rows. Freeze every selected head before evaluating either held-out hospital.
- Three fitting seeds17,29,43, nine heads, two prespecified held-out hospitals =54 target records from27 final source-trained heads. Save each head before target scoring. Keep primary hospital2 and secondary hospital1 results separate, plus a clearly labeled descriptive mean/worst across the two observed hospitals.
- Report balanced accuracy, AUROC, macro-F1, sensitivity/specificity, NLL/Brier where defined, head-seed variation and patient bootstrap intervals (2,000 attempts, seed2027, fixed trained heads, record class-incomplete draws). Patient/slide sampling does not create extra independent hospitals. No adaptive changes after these independent-dataset target results.
- External ImageNet pretraining, source-only inner selection and patient-capped patch sampling make this a custom frozen-transfer evaluation, not an official WILDS leaderboard submission.

## Deliverables

Separate completion markers and scoped reports for each experiment. CSV/JSON predictions, raw source searches, provenance/validation, PNG/PDF scientific figures and standalone Overleaf table fragments stay in this folder. The existing manuscript is not edited automatically. A combined report must distinguish completed numerical runs from downloads or preparation still pending, PolypGen sequence bootstrap from CAMELYON patient bootstrap, and primary test scores from secondary hospital diagnostics.
