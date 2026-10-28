# PolypGen v3: isolated held-out-center experiment

This protocol is recorded before classifier fitting in this folder. It is a preliminary experiment on a reviewed sequence subset, not a complete-release or submission-ready benchmark.

## Isolation and provenance

- Archive: `PolypGen2021_MultiCenterData_v3 (1).zip`, copied from Downloads.
- SHA-256: `a22f956a9f0fc3f941927108a3de8217fc61f1944c7eda598536dbabc27dbff3`.
- Extraction, code snapshot, annotation manifests, features, models, logs, and reports stay below this directory. Existing project data and outputs are not modified.
- The archive's embedded concatenated ZIP stays unexpanded and is not another model input.
- `extraction_complete.json` records archive integrity and copied code hashes. Execution checks use the snapshot in `code/`.

## Dataset and labels

- Inputs are original video frames only. Annotated overlays, masks, pooled copies, and still images without documented sequence grouping are excluded as model inputs.
- Frame labels describe annotated polyp presence, not the name of the enclosing positive-sequence folder.
- Masks use intensity greater than 127 and at least one foreground pixel; matched bounding boxes must agree. Negative-only release sequences can supply documented negative labels without annotations.
- Frame annotations are inventoried across the complete extracted release, even where center identity is unresolved.
- Model inputs use explicit Figure 7 center assignments or a center token in every original filename. No assignment is inferred from counts, visual appearance, sequence numbering, or figure ellipses.
- Exclude unresolved positive sequences 4,7,8,9,14,15; disputed positive sequence11 and negative sequence7_neg; and positive sequence1 pending annotation review. The v2 review supersedes the earlier class audit's center assignment for sequence11.
- Expected target centers are C2,C4,C5,C6, conditional on the fresh audit and duplicate/group guards. C1 is source-only if negative-only; C3 is absent from this subset. Counts and eligibility are saved before feature extraction.
- Sequence folders define groups. Patient independence between different sequences is not established. C2 negatives and C5 positives each occur in only one sequence in the reviewed subset; report this limitation with results.

## Encoder and outer evaluation

- Frozen torchvision ResNet18, official ImageNet1K V1 checkpoint. Checkpoint hash and exact preprocessing are recorded; pretrained encoder parameters receive no gradient updates.
- Deterministic checkpoint evaluation transforms; normalized features; one shared cache for every head and seed. No target data fitting or adaptation.
- Repeat leave-one-center-out evaluation over all eligible target centers. All remaining mapped centers, including source-only C1, are available as sources.
- Source validation is leave-one-source-center-out with both classes required. No outer held-out-center score participates in model selection, calibration, or prototype restart selection.
- Prototype restarts are selected by the training objective. After source selection, refit the head on all outer source frames.
- Optimization seeds: 17,29,43. These repeat head fitting/selection; they do not create additional independent centers or patients.
- CPU, four numeric threads, batch size32, data-loader workers0. Reuse the same audited features across seeds.

## Heads and source tuning budgets

All heads use identical input features and partitions. Candidate grids are disclosed; their cardinalities are not identical.

| Method | Source-selected candidates | Count |
|---|---|---:|
| CBMP | K in {1,2,4,8}, beta in {0,0.5,1} |12|
| Unbalanced multiple prototypes | K in {1,2,4,8}, beta=0 |4|
| Linear logistic | C in {0.01,0.1,1,10,100}, balanced classes |5|
| Trained cosine | learning rate in {0.01,0.03,0.1}, balanced classes |3|
| Group/center weighted linear | same C grid x beta in {0,0.5,1} |15|
| Group/center weighted cosine | same learning-rate grid x beta in {0,0.5,1} |9|
| Single centroid, group weighted | K=1,beta=0 |1|
| Single centroid, center balanced | K=1,beta=1 |1|
| Single centroid, image weighted | K=1,beta=0, image groups |1|

Each prototype candidate uses five restarts and at most100 updates. Cosine candidates use300 optimization steps. Ties follow the predetermined candidate order. Mean validation-center balanced accuracy selects the head; equal validation-center negative log-likelihood selects temperature. Both classes receive equal total weight for the supplied group/center weights. Beta=0 still controls group/frame weighting and must not be described as raw image weighting.

## Reporting

- Save per-frame predictions, source searches, selected settings, outer/inner partition indices, head models, timing, checkpoint identity, and configuration for every seed.
- Report balanced accuracy, macro-F1, and AUROC for every eligible center, equally weighted mean across centers, and worst-center balanced accuracy.
- Show optimization-seed variation separately from sequence-sampling uncertainty. Main summaries average over the three predeclared seeds; worst center is computed after averaging a center's score across seeds.
- Bootstrap whole recorded groups within each center, using the same resamples across methods/seeds. Report accepted and class-incomplete draws; interval estimates are conditional on class-complete resamples and the fixed trained models.
- Do not use preliminary target results to change mappings, labels, exclusions, grids, preprocessing, or stopping settings. Any future revised protocol is a separately versioned experiment.
- Generate standalone CSV, JSON, PNG/PDF, and Overleaf-ready TeX fragments under this folder; the open manuscript is not edited automatically.

The key question is whether source-selected center balancing adds value beyond multiple prototypes and equally weighted linear/cosine controls. A baseline beating CBMP is a valid outcome.
