# Manuscript correction and result audit (23 October 2026)

## Scope of this revision

The editable manuscript section file is `outputs/introduction.tex`; the previous version is preserved in `outputs/revision_2026-10-23/backups/outputs/introduction.tex`. Sections 1-3 now describe the completed work rather than a proposed study. A fourth section reports the five saved experiments and their stated limits. The CVPR PDF supplied on 23 October remains a template export with abstract instructions/title placeholder; this revision does not supply the abstract, conclusion, or mandatory contribution statement.

## Corrections tied to implementation

- All encoders are frozen. The previous draft referred to a separate encoder-adaptation experiment; none was run.
- Temperature is selected by bounded continuous optimization of mean source-fold NLL in log-temperature space, with log-temperature bounds [-4,4]. The previous draft incorrectly listed a discrete temperature grid.
- PolypGen grouping uses recorded sequences; patient independence is not verified. CAMELYON grouping uses patient IDs.
- PolypGen is a reviewed subset of 5,444 frames/37 sequences with four class-complete outer centers (C2,C4,C5,C6), C1 source-only, and C3 not represented after mapping review.
- CAMELYON17 results use a custom 17,200-patch, 43-patient subset from a pinned community redistribution, not the full official WILDS evaluation. New classifier heads are fitted on source histology hospitals.
- The encoder comparison changes complete pretrained packages, including architecture, pretraining and preprocessing; it is not an isolated architecture experiment.

## Prediction-level verification

`scripts/build_paper_results.py` recalculated BA and AUROC from 465,534 saved prediction records (140,967 ResNet18 PolypGen; 140,967 DINOv2 PolypGen; 183,600 CAMELYON). It checked shared test sample support, reconstructed the documented 108 + 108 + 54 main-evaluation grids, and compared point metrics to saved summaries at absolute tolerance 1e-12. No training or bootstrap was performed by this check. The exact input SHA-256 values are in `outputs/revision_2026-10-23/generated_results/verification.json`; generated table fragments are beside it. A second check confirmed that all 18 manuscript table rows match the corresponding saved metric CSVs to two decimal places, and all 17 citation keys used in the manuscript resolve in `references.bib`.

## Evidence supporting the reported findings

- ResNet18 PolypGen CBMP: mean center-balanced BA 58.35%, worst center 42.14%; linear logistic: 70.20% and 57.17%. The paired conditional whole-sequence bootstrap for CBMP minus linear is -11.9 pp, 95% interval [-16.0,-7.7]. Only 1,026/2,000 sequence-bootstrap attempts were class-complete.
- DINOv2 PolypGen CBMP: 59.54% mean and 45.31% worst-center BA; trained cosine has the best mean (68.17%) and weighted cosine the best worst-center result (61.60%).
- Fixed K/beta effects vary by encoder and configuration; the experiments do not show a uniform benefit from more prototypes or center balancing.
- CAMELYON primary H2 CBMP: 85.40% BA/92.96% AUROC; linear logistic: 88.88%/95.14%. Paired patient-bootstrap difference for BA is -3.48 pp, 95% interval [-4.70,-2.11]. H1 is secondary; only two target hospitals are observed.

Detailed fit settings, split manifests, model hashes, per-center metrics, bootstrap definitions and known data limitations remain in the experiment reports under `isolated_datasets/polypgen_extensions_2026-10-20/` and the earlier baseline/follow-up roots. Result values do not establish clinical utility, patient-level independence for PolypGen, population-level hospital robustness, or a general CBMP advantage.
