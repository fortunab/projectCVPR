# Experiment 2: exploratory fixed factorial ablation

Frozen ResNet18; same reviewed subset and source-only calibration; three fitting seeds. No cell is selected using target performance.

| K | beta | Mean center bAcc | Worst center bAcc | Mean AUROC |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0.6315 | 0.6034 | 0.7899 |
| 1 | 1 | 0.6048 | 0.4763 | 0.6914 |
| 4 | 0 | 0.5787 | 0.5025 | 0.6881 |
| 4 | 1 | 0.5958 | 0.5210 | 0.6843 |

| Contrast | Change in mean-center bAcc | Conditional 95% CI |
|---|---:|---:|
| multiplicity_at_beta0 | -0.0528 | [-0.1075, +0.0235] |
| multiplicity_at_beta1 | -0.0090 | [-0.0628, +0.0754] |
| balancing_at_K1 | -0.0267 | [-0.0772, -0.0109] |
| balancing_at_K4 | +0.0171 | [-0.0152, +0.0479] |
| interaction_difference_in_differences | +0.0438 | [+0.0184, +0.0911] |

Bootstrap: 1026/2000 class-complete sequence draws accepted. Intervals pair the same samples across all cells and seeds, condition on fixed trained heads and four observed centers, and do not establish patient independence.

These are component comparisons, not the source-selected CBMP method's main result. The four-cell factorial holds settings fixed and does not use target results to choose K or beta. Original baseline controls remain unchanged and are contextual comparisons in ../baseline_context.csv.

C2 negatives and C5 positives each occur in only one sequence. Larger prototype banks and equal center weighting may help some centers and hurt others; per-center effects in paired_effects.csv are necessary to interpret the mean.

All zero-threshold K1 predictions were checked against the original group-weighted and center-balanced centroid results. Requested and realized prototype counts are recorded per fit.
