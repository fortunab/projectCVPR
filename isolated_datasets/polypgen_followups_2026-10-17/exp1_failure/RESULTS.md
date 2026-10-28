# Experiment 1: exploratory failure analysis and source-only threshold control

These are follow-ups designed after inspecting the original target results. They are exploratory and require independent confirmation. The baseline data, predictions, fitted heads and manuscript were not edited.

## Source-only decision threshold

Each baseline keeps its original source-selected configuration and temperature. The same specification is refit only on the existing inner source training folds (seed + inner-fold index). A single raw score-margin threshold maximizes the equally weighted mean balanced accuracy across the source validation centers. Candidate thresholds are zero, consecutive distinct-margin midpoints, nextafter(minimum margin, -infinity), and the maximum margin itself. Objective ties within 1e-12 favor the smallest absolute threshold, then its numeric value. The adjusted decision is class 1 exactly when margin > threshold; the original zero threshold reproduces every saved prediction. The threshold is applied to the original saved outer source-trained head. Outer labels never select the threshold.

Changing this decision threshold does not alter score ranking, AUROC, or the saved temperature-scaled probabilities. This control is not probability calibration and is not a novelty claim.

| Method | Original mean BA | Adjusted mean BA | Change (pp) | Original worst BA | Adjusted worst BA |
|---|---:|---:|---:|---:|---:|
| CBMP | 58.3% | 58.8% | +0.5 | 42.1% | 42.1% |
| Multi-prototype | 61.1% | 61.8% | +0.7 | 53.1% | 54.4% |
| Linear | 70.2% | 69.4% | -0.8 | 57.2% | 57.9% |
| Cosine | 63.1% | 61.5% | -1.7 | 56.7% | 57.2% |
| Weighted linear | 61.8% | 61.0% | -0.8 | 47.2% | 47.2% |
| Weighted cosine | 63.5% | 58.4% | -5.1 | 46.1% | 45.0% |
| Group centroid | 63.2% | 61.9% | -1.2 | 60.3% | 57.7% |
| Center centroid | 60.5% | 57.4% | -3.1 | 47.6% | 50.9% |
| Image centroid | 60.4% | 58.7% | -1.7 | 58.2% | 54.0% |

Mean is the equally weighted average of the four center scores after averaging seeds. Worst is the minimum of those seed-averaged center scores. Seeds are optimization replicates; no frame-level confidence interval is presented.

## Center-specific CBMP diagnostics

- C2: original BA 0.6034, AUROC 0.8831, sensitivity 0.9911, specificity 0.2157; source-offset BA 0.5773, sensitivity 0.9978, specificity 0.1569.
- C5: original BA 0.4214, AUROC 0.4389, sensitivity 0.3518, specificity 0.4910; source-offset BA 0.4214, sensitivity 0.3568, specificity 0.4859.

The source-selected offset lowers C2 specificity even though C2 has strong class ranking. Its source-optimal direction therefore does not transfer to this target. The unchanged poor C5 pooled ranking rules out a simple successful threshold repair in this control.

A relevant sequence-level contrast is visible in C5: within its mixed sequence `seq5`, CBMP AUROC is 0.9321, specificity 0.9804, and sensitivity 0.3518. Specificity on C5's three negative-only sequences is 0.2895 (`seq11_neg`), 0.4061 (`seq12_neg`), and 0.6179 (`seq13_neg`). Thus the pooled-center failure is associated with differences across sequences even though ranking within the single positive-supporting sequence is strong. This is descriptive evidence of sequence-dependent score structure, not an established causal explanation or proof of patient generalization.

`sequence_seed_metrics.csv` reports TP/TN/FP/FN, recall by class, BA and AUROC for every sequence and seed under both decision rules. A sequence missing either class has blank BA, macro F1 and AUROC; the defined class recall is retained. Aggregate sequence confusion entries in `sequence_metrics_seed_means.csv` are means over seeds, not additional independent frames.

`prototype_summary.csv` records realized bank support and effective group count (inverse squared normalized group-mass sum). `prototype_contributions.csv` provides both raw frame fractions and fractions of the prototype's assigned source weight by center and sequence. The clustering weights are normalized within class as in the original method. `prototype_nearest_source_examples.csv` lists same-class source examples nearest to each prototype; this is descriptive geometry and does not establish clinically meaningful prototype semantics.

`sequence_class_score_summary.csv` records raw score distributions by sequence, class, method, and seed. `target_prototype_matches.csv` and `target_prototype_match_summary.csv` connect target decisions to their winning prototype indices so that these can be compared with source support. None of these target diagnostics changes the fitted model or threshold.

The ROC figure shows all three seeds for the key comparators. On C2, CBMP and the group centroid are identical selected models, so their ROC curves coincide. The score histograms and error montage use seed 17 explicitly; examples are selected by sample ID, with distinct sequences preferred. Source retrievals are nearest to the winning predicted-class prototype, not nearest to the target image.

## Limits

This is the same 5,444-frame, 37-sequence reviewed subset and four eligible target centers. Some release sequence-center mappings remain unresolved. Sequence grouping does not verify patient independence. C5 positives are supported by only one sequence, so an apparently large frame count is not broad positive-case coverage. Prototype concentration and sequence errors identify associations, not causes of the observed failure. A source-selected offset can improve or worsen an unseen center and cannot repair poor ranking by construction.
