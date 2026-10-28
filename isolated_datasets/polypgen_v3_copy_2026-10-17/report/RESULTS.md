# Preliminary PolypGen held-out-center results

Preliminary reviewed sequence subset; frozen ResNet18; 3 seeds.

The highest mean-center balanced accuracy is 0.702 (Linear logistic); CBMP achieves 0.583, with worst-center balanced accuracy 0.421.

| Method | Mean bAcc [conditional 95% CI] | Head-seed SD | Worst bAcc | Mean F1 | Mean AUROC |
|---|---:|---:|---:|---:|---:|
| CBMP | 0.583 [0.527, 0.631] | 0.007 | 0.421 | 0.534 | 0.733 |
| Multiple prototypes (unbalanced) | 0.611 [0.553, 0.658] | 0.004 | 0.531 | 0.565 | 0.762 |
| Linear logistic | 0.702 [0.655, 0.745] | 0.000 | 0.572 | 0.630 | 0.800 |
| Trained cosine | 0.631 [0.585, 0.672] | 0.036 | 0.567 | 0.570 | 0.805 |
| Linear (group/center weighted) | 0.618 [0.557, 0.665] | 0.000 | 0.472 | 0.585 | 0.733 |
| Cosine (group/center weighted) | 0.635 [0.597, 0.678] | 0.021 | 0.461 | 0.572 | 0.745 |
| Centroid (group weighted) | 0.632 [0.580, 0.682] | 0.000 | 0.603 | 0.568 | 0.790 |
| Centroid (center balanced) | 0.605 [0.533, 0.655] | 0.000 | 0.476 | 0.549 | 0.691 |
| Centroid (image weighted) | 0.604 [0.552, 0.654] | 0.000 | 0.582 | 0.544 | 0.765 |

Paired CBMP-minus-baseline differences use the same sequence resamples:

| Baseline | Difference in mean-center bAcc | Paired conditional 95% CI |
|---|---:|---:|
| Multiple prototypes (unbalanced) | -0.028 | [-0.061, +0.021] |
| Linear logistic | -0.119 | [-0.160, -0.077] |
| Trained cosine | -0.048 | [-0.080, -0.011] |
| Linear (group/center weighted) | -0.034 | [-0.083, +0.017] |
| Cosine (group/center weighted) | -0.052 | [-0.106, -0.009] |
| Centroid (group weighted) | -0.048 | [-0.099, -0.009] |
| Centroid (center balanced) | -0.021 | [-0.051, +0.015] |
| Centroid (image weighted) | -0.021 | [-0.079, +0.023] |

Bootstrap: 1026 accepted and 974 class-incomplete draws rejected out of 2000 attempts (seed 2027). Invalid-draw counts by center are nonexclusive: {"C2": 490, "C4": 7, "C5": 622, "C6": 0}.

Intervals condition on class-complete whole-sequence resamples, the fixed reviewed subset, the four observed target centers, and fixed trained heads. Heads are not retrained in the bootstrap. Seed SD describes three optimization runs and is separate from sequence-sampling uncertainty; seeds do not add independent patients or centers. Metrics average seed-specific scores; there is no prediction ensemble. Worst-center performance takes the minimum of the four seed-averaged center scores.

This is a partial PolypGen v3 reviewed sequence subset. C1 is source-only and C3 is absent; unresolved/disputed mappings and sequence1 annotation review remain exclusions under the predeclared protocol. Sequence grouping does not establish patient independence. C2 negatives and C5 positives each occur in only one recorded sequence, making uncertainty estimates sensitive to class-complete conditioning. These results do not establish performance on the complete release, clinical reliability, or generalization to a population of new hospitals.

All methods share frozen features and partitions. Head grids have different disclosed cardinalities (CBMP12, unbalanced multiple prototypes4, linear5, cosine3, weighted linear15, weighted cosine9, each centroid1). Selection and temperature calibration use source validation only. A baseline exceeding CBMP is a valid outcome. Selected settings and calibration temperatures are in selected_settings.csv/json, with full candidate searches preserved in each seed's results folder.

The per-center CSV contains accuracy, balanced accuracy, macro-F1, AUROC, NLL, and Brier score for every seed and the mean of seed-specific metrics. Main plots show the four standard classifier families plus CBMP; the full center plots and table also include weighted linear/cosine controls. Input hashes and validation checks are saved in report_provenance.json.
