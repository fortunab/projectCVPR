# Proposed confirmation beyond the inspected cohorts

This is a development protocol, not an executed experiment or a preregistration. The original targets have already been inspected. Ten fitting seeds, another grid, or a new encoder on those same images do not turn them into an untouched confirmation cohort.

## Data and unit of inference

Obtain a separately collected cohort with verifiable image labels, patient linkage and institution IDs. Freeze inclusion, exclusions, patient grouping and label definitions before evaluating any model. Keep all examinations, frames, patches and slides from one patient in the same partition. If linkage is missing, do not reconstruct it from filename similarity or embeddings.

The clinical target must be specified separately from the image task. For sequence polyp presence, an any-positive-frame definition is an operational endpoint; the evaluation needs positive-bearing and negative-only independent examinations. For within-patient patch AUROC, each contributing patient needs both patch labels. Clinical patient/slide diagnosis requires its own ground truth and appropriate positive and negative patients/slides, rather than labels inferred from sampled patches.

Choose cohort size from a prospectively justified effect and independent-unit variability, with sensitivity to plausible effect sizes. These small observed cohorts are unsuitable for a confident retrospective power claim. If the claim concerns a population of hospitals, the sampling design needs enough independent hospitals and a defined target population; more patches at the same sites do not provide that information. Report fixed-site results when that design is unavailable.

Rotating all five existing WILDS hospitals could add an exploratory leave-one-hospital-out comparison. It remains a custom analysis of inspected hospitals, distinct from both official WILDS evaluation and external confirmation.

## Freeze the scientific comparisons

1. Choose a general visual encoder and a pathology encoder using documented task relevance, access and resource requirements. Pin the exact checkpoint/revision and audit overlap with the future test cohort. Keep original UNI, UNI2-h, CONCH and later derivatives distinct.
2. Compare CBMP, linear and cosine classifiers on identical features, source folds and fitting seeds. Fix the candidate grids and tie-breaking before target inspection. Report candidate counts and actual compute; a four-candidate grid changes parameter coverage as well as count.
3. Prespecify the learned control used in each primary contrast. Do not choose the displayed winner on the new test data and then interpret its conditional interval as a prespecified hypothesis test.
4. Separate the source-selected rule from the fixed `(K,beta)` mechanism cells. A candidate mechanism contrast is beta 1 minus beta 0 at K=4, with BA and Across AUROC reported together. State whether endpoints are co-primary or secondary and choose any multiplicity control before evaluation.
5. Use ten fitting seeds for consistency with the expanded study and CAMELYON replicate requirements. These repeats quantify fitting variation on fixed data. Use patient/site resampling only at the units supported by the sampling design, with undefined components and valid-draw counts reported.

## Mechanism and modern DG coverage

Inspect selected prototype counts, cross-seed alignment, source-group support and the nearest source images. A domain expert should assess clinical cluster meaning using a recorded blinded procedure where feasible; vector similarity or center purity alone does not establish diagnostic semantics. Record who inspected what and the unresolved cases. Do not retrospectively assert a clinical review from a checksum or passing unit test.

A modern DG baseline should have an official implementation, a task-compatible image classifier, an explicit target-access policy and a resource estimate. If it updates the backbone or uses target-batch statistics, compare it in a separate, matched training/access experiment. A spatial segmentation method such as InfProto is not reproduced faithfully by renaming a global-feature prototype bank. The new experiment must document its task adaptation rather than treating the current frozen-head controls as coverage of all DG training methods.

The existing sequence-removal stress analysis remains observational. A controlled mechanism simulation could independently vary class structure, nuisance correlations, center composition and group/sample counts, with known generative factors. It would test properties of that simulation, not supply additional clinical benchmark results. Preserve that distinction in any claim.

## Benchmark replication and release

For a full official-split CAMELYON replication, use the official dataset/evaluator, H0/H3/H4 training, H1 validation and H2 test, without validation refitting or patch capping. Respect the documented pretraining setting and ten-seed rule. Our earlier H2 inspection must remain disclosed; official splits do not erase it. See [EXTERNAL_EVALUATION.md](EXTERNAL_EVALUATION.md).

Before a new test is unsealed, save the protocol, manifests, checkpoint and code fingerprints, source selections and fitted model identifiers. Release raw scores/decisions, complete per-site/per-seed metrics, proper probability scores, group support and failures. Record the actual human review using [AUTHOR_REVIEW_RECORD.md](AUTHOR_REVIEW_RECORD.md). Keep anonymous release artifacts separate from the named local sign-off record.
