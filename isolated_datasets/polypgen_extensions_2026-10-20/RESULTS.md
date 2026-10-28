# Completed experiments 4–5

Both approved extensions completed. All new data, cached features, trained classifier heads, source-only searches, predictions, scientific figures, CSV tables and Overleaf table fragments are in this folder. The existing manuscript was not edited.

## 4. Frozen second encoder on the unchanged PolypGen cohort

Frozen DINOv2 ViT-B/14 and the earlier ResNet18 use the same audited 5,444 frames, 37 sequences, four held-out centers and three fitting seeds. All nine classifier variants retain their original source-only tuning protocol. There are 108 new main evaluations and 48 new fixed factorial evaluations. Scores below are equally weighted center means after averaging fitting seeds; worst denotes the minimum of those center means.

| Classifier | ResNet18 mean BA | DINOv2 mean BA | ResNet18 worst BA | DINOv2 worst BA |
|---|---:|---:|---:|---:|
| CBMP | 58.35% | 59.54% | 42.14% | 45.31% |
| Unbalanced multiprototype | 61.12% | 60.43% | 53.14% | 48.88% |
| Linear logistic | 70.20% | 61.31% | 57.17% | 52.12% |
| Trained cosine | 63.12% | 68.17% | 56.68% | 61.45% |
| Weighted linear | 61.78% | 59.24% | 47.17% | 51.30% |
| Weighted cosine | 63.51% | 65.74% | 46.07% | 61.60% |
| Group weighted centroid | 63.15% | 64.03% | 60.34% | 53.87% |
| Center balanced centroid | 60.48% | 61.21% | 47.63% | 51.53% |
| Image weighted centroid | 60.41% | 62.32% | 58.17% | 51.23% |

[Detailed paired encoder report](exp4_dinov2/report/RESULTS.md) includes per-center scores, paired whole-sequence bootstrap intervals, fixed K/beta effects, saved predictions and checkpoint provenance.

The encoder comparison is exploratory because the PolypGen cohort was previously inspected. It compares complete pretrained encoder packages, including their prescribed preprocessing. C2 negatives and C5 positives each occupy one sequence; patient independence has not been established. Sequence bootstrap intervals condition on class-complete draws and the observed centers and fitted models.

## 5. Independent histology hospitals

The label-blind patient-capped CAMELYON17-WILDS subset contains 17,200 patches from 43 patients. Each eligible patient contributes at most 400 patches, ranked by a predeclared identity hash. A frozen ResNet18 ImageNet1K V1 encoder provides the common features. Original source-training hospitals 0, 3 and 4 alone select and fit all classifiers. These are new histology-trained classifier heads. All 27 final source-trained heads were saved and hashed before either target hospital was scored. Two target hospitals and three fitting seeds give 54 target records.

Hospital 2 is the primary test. Hospital 1 is a prespecified secondary unseen-hospital evaluation. Balanced accuracy and AUROC below average three head-fitting seeds; patient bootstrap intervals are reported separately.

| Classifier | Primary H2 BA | Primary H2 AUROC | Secondary H1 BA | Secondary H1 AUROC |
|---|---:|---:|---:|---:|
| CBMP | 85.40% | 92.96% | 77.58% | 85.35% |
| Unbalanced multiprototype | 85.15% | 92.47% | 76.83% | 84.66% |
| Linear logistic | 88.88% | 95.14% | 80.30% | 91.71% |
| Trained cosine | 87.88% | 94.90% | 79.19% | 91.46% |
| Weighted linear | 84.57% | 93.14% | 82.67% | 90.03% |
| Weighted cosine | 86.02% | 93.49% | 81.52% | 90.45% |
| Group weighted centroid | 73.42% | 88.92% | 74.55% | 86.47% |
| Center balanced centroid | 75.21% | 89.71% | 75.11% | 86.75% |
| Image weighted centroid | 82.52% | 91.63% | 75.48% | 88.65% |

[Detailed independent-dataset report](exp5_camelyon/report/RESULTS.md) includes the patient audit, patient bootstrap intervals, paired classifier contrasts, selected settings, scientific figures and an Overleaf table.

This is a custom frozen ImageNet-transfer subset using a pinned community redistribution, not a full official WILDS benchmark result. The author download was probed before mirror acquisition; all 21 mirror shards were verified against their pinned SHA-256 hashes. External ImageNet pretraining, patient caps and the custom source-only selection protocol must be disclosed in any paper. The two observed target hospitals do not identify uncertainty over a population of hospitals.

## Reproducibility and preservation

The frozen plan is [PROTOCOL.md](PROTOCOL.md). Separate experiment reports provide PNG and vector PDF figures plus standalone LaTeX table fragments; copy those artifacts into Overleaf only when the experiment interpretation is ready. No online Overleaf project was changed.

The final preservation check found no changes among 51,107 protected earlier files; 1,140 earlier experiment artifacts also match their saved SHA-256 values. See [preservation_check.json](preservation_check.json).

The artifact manifest hashes scripts, fitted models, predictions, reports and provenance. Raw downloaded shards and selected image contents are covered by the acquisition and image-audit manifests rather than hashed a second time in that index.

## Publication next steps

[Publication roadmap in Romanian](PUBLICATION_ROADMAP.md) separates CVPR main's novelty/evidence burden from a medical CBMS contribution, explains ITiCSE's computing-education scope, identifies close prototype literature, and gives a provisional plan before the November 2026 CVPR deadlines. It proposes future work; additional experiments have not been started under that plan.
