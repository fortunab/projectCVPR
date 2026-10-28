# Exploratory PolypGen follow-up experiments 1–3

Recorded before follow-up fitting. The original four-center test results have already been inspected. These are exploratory analyses, not independent confirmation or a new complete-release benchmark. Do not change mappings, labels, targets, features, candidate grids, or exclusions in response to follow-up test scores.

## Shared data and preservation

- Read-only baseline: sibling `polypgen_v3_copy_2026-10-17`. Reuse its audited 5,444-frame, 37-sequence subset and frozen ImageNet ResNet18 embeddings.
- Original project data, outputs, and every baseline artifact remain untouched. All new scripts, copied code, logs, fitted heads, predictions, figures, tables, and reports stay in this new folder.
- Targets: C2,C4,C5,C6, unchanged from the baseline. C1 is source-only; C3 is absent. Sequence grouping does not establish patient independence. C2 negatives and C5 positives each occur in one sequence.
- Copy and hash the baseline's core Python implementation locally. Validate input manifest/feature provenance, per-run completion, and original split boundaries. Record before/after protected-file inventories and hashes of baseline artifacts.
- Numeric fitting uses two threads per concurrent experiment process. No encoder fine-tuning, new image feature extraction, target adaptation, or new data download.
- Record source searches, exact group selections, fitting/selection seeds, requested and realized prototype counts, and per-frame predictions. Missing class-specific group metrics are undefined, not zero.

## Experiment 1: failure analysis and source-selected decision thresholds

### Existing-result diagnostics

Use all nine baseline methods and head seeds17,29,43. Report per-center and per-sequence confusion counts, sensitivity, specificity, balanced accuracy, and AUROC where both classes exist. Inspect C2 and C5 separately. Summarize CBMP prototype assignments of class-conditional source samples by center and sequence, normalized weighted mass, effective group counts, and nearest source examples. Any image panels are descriptive model diagnostics; prototype matches do not establish clinical explanations or causal acquisition effects.

### Threshold control

Keep each baseline's previously source-selected classifier configuration fixed. For each method/outer center/head seed, refit that configuration on the original inner source-center folds using seed+fold index. Collect raw class-score margins (score1 minus score0). Choose a finite threshold maximizing equally weighted validation-center balanced accuracy. Candidates are zero, midpoints between distinct source-validation margins, a finite endpoint below the minimum, and the maximum margin. Predict class1 iff margin>threshold; an exact zero tie therefore reproduces the original argmax rule. Ties within1e-12 favor the threshold closest to zero, then numeric ascending order.

Apply the source-selected threshold to the existing final baseline head. Verify the zero-threshold predictions reproduce the saved baseline decisions. Target labels never choose thresholds. Temperature and probabilistic scores are unchanged; this control changes classification decisions, not score ranking. Report paired classification changes, source objectives, threshold values, and unchanged ranking metrics. This is a post-selection threshold control, not a joint re-search over head settings and thresholds and not a novelty claim.

## Experiment 2: fixed factorial ablation

- Factors: K in {1,4} and beta in {0,1}, four fixed cells, no cell selected by target performance.
- Every cell uses the same class/group weighting formula, with beta controlling only center balancing; beta0 is still group weighted.
- Head seeds17,29,43; same four outer targets and original inner source folds. Five prototype restarts per class, max100 updates, unchanged fitting algorithm.
- Fit each cell on each inner source fold solely to select its temperature by equal-validation-center negative log-likelihood, using the same bounded continuous temperature optimizer as the implementation. Fit the fixed cell on all outer sources, then evaluate the fixed target.
- Save models, predictions, train/source-validation settings, requested/realized K, and timing. Retain the original image-weighted centroid, weighted linear/cosine, linear, and unbalanced prototype results as read-only contextual controls, explicitly labeled as baseline comparisons rather than factorial cells.
- Report four-cell performance per center, equally weighted center mean, and worst center after averaging seed-specific center scores. Report paired multiplicity effects (K4 minus K1) at each beta, balancing effects (beta1 minus beta0) at each K, and the difference-in-differences interaction. Do not present marginal averages alone as proof of a universally beneficial component.
- Paired whole-sequence bootstrap within the four fixed target centers:2,000 attempts, seed2027, identical resamples for all cells and seeds. Report class-incomplete rejected draws; intervals are conditional on class-complete resamples of fixed trained heads. Optimization-seed variation is separate.

## Experiment 3: source composition and matched group-budget stress

### Conditions, fixed before scoring

- Use baseline head seed17 only, fixing optimization randomness; source-subset seeds101,211,307 measure sampling variation, not independent hospitals or patients.
- Methods: CBMP, unbalanced multiple prototypes, linear logistic, trained cosine, group/center weighted linear, and group-weighted single centroid. Preserve their exact baseline source-tuning grids, calibration rule, fitting steps, and five prototype restarts. Reuse identical candidate fits when configurations/training indices/seeds coincide; never alter the mathematical fit.
- For each outer target, choose the source center with the largest number of groups, breaking ties by center identifier. This choice uses source metadata only.
- Composition arm: retain all groups in the other source centers; retain max(1,floor(fraction*dominant_group_count)) in the dominant source center, at fractions0.5 and0.25. Sample whole groups without replacement. The original full-source condition is a read-only baseline anchor at fraction1.
- Both arms preserve every source center and that center's original observed class vocabulary. Uniform group sampling is conditioned on this coverage. Log rejection attempts and infeasible conditions; no removal of individual frames to fabricate balance. A deterministic exact-combination sampler may be used where rejection is impractical; record the choice. No target labels choose subsets.
- Matched scarcity arm: retain exactly the same total group budget as the corresponding composition condition, with quotas proportional to original source-center group counts. Allocate integer quotas by deterministic largest-remainder/apportionment subject to minimum feasible class-covering quotas and each center's available group count. Resolve ties by center identifier. Sample within each quota with the same source-class-coverage condition.
- The matched control equalizes total group count, not frame count or class prevalence. Record actual retained frames, group counts, per-center class counts, center proportions, and imbalance measures. Reducing a dominant center can reduce imbalance; label plots by retained source fraction and measured composition, not an assumed increasing-shift axis.
- Four targets x two fractions x two arms x three subset seeds =48 conditions, six methods =288 target evaluations. Record repeated identical subsets; they are not independent replicates. Full-source baseline anchors are reused with provenance and are not counted as new evaluations.

### Selection and evaluation

Reconstruct source-only leave-one-source-center-out validation within each retained subset. Require the same class-complete source-validation centers as the corresponding original outer fold. No random image validation fallback. If class coverage or fold feasibility cannot be achieved under the predeclared rule, report the condition as infeasible before fitting. Select each method using mean validation-center balanced accuracy; select temperature using equal validation-center NLL. Refit the selected head using retained sources only and score the unchanged target. Save all selected groups, indices, validation partitions, searches, models, predictions, and timing.

Report per-center results, source-subset means/variation, the composition-minus-matched-budget difference, and changes relative to full-source seed17. Compute across-center summaries by first averaging replicate metrics within each center, then giving each center equal weight. Do not treat nested subset/head/center observations as independent samples or report naive pooled p-values. Sampling variation and sparse sequence coverage limit causal interpretation.

## Deliverables and interpretation

Each experiment produces CSV/JSON evidence, PNG/PDF scientific figures, and a clearly scoped Markdown report; table fragments are Overleaf-ready without editing the existing manuscript. An integrated report distinguishes observations from hypotheses, threshold effects from ranking changes, component effects from model selection, and composition changes from group scarcity. No acceptance, clinical reliability, domain-invariance, or universal robustness claim follows from these exploratory runs. Changes suggested by these results require separately frozen independent confirmation.
