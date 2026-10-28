# Colorectal histology implementation pilot

Within-dataset patch classification pilot; patient/slide/center identities are unavailable. Content-disjoint splits do not establish patient independence or unseen-center generalization. Center balancing is not evaluated.

The complete local TensorFlow Datasets `colorectal_histology:2.0.0` cache was imported and audited. No new dataset download was required.

## Protocol

- Real images: 5,000; eight classes.
- Fixed seed: 17; counts: {'train': 3000, 'validation': 1000, 'test': 1000}.
- Identical decoded images remain in one partition. Patient/slide IDs are unavailable.
- Encoder: frozen torchvision ResNet18, ImageNet1K V1; prescribed preprocessing; 512-dimensional unit features.
- K and linear regularization are selected using validation balanced accuracy; temperatures use validation negative log likelihood.
- Models fit training patches only; no refit on validation and no test-driven selection.
- Selected K: 8; selected linear C: 10.0.
- One fixed split and optimization seed; no confidence intervals or patient independence are claimed.

## Measurements

| Method | Test balanced accuracy | Test macro F1 | Test macro AUROC | Test NLL |
|---|---:|---:|---:|---:|
| Single prototype | 0.7940 | 0.7965 | 0.9654 | 0.6375 |
| Validation-selected prototypes (K=8) | 0.8770 | 0.8766 | 0.9854 | 0.4242 |
| Linear classifier | 0.8960 | 0.8970 | 0.9907 | 0.3038 |
| Cosine classifier | 0.8460 | 0.8481 | 0.9820 | 0.4376 |

The selected prototype model changes test balanced accuracy by +8.3 percentage points relative to one prototype, and -1.9 points relative to the linear baseline. These are descriptive differences from one fixed split, without significance claims.

These measurements check the classifier pipeline on tissue recognition. They do not demonstrate center robustness, establish method novelty, or replace held-out-center PolypGen experiments.

## Figures and saved evidence

![Classifier comparison](C:/Users/Lenovo/Documents/Codex/2026-10-14/https-github-com-fortunab-family-baselines/outputs/histology_pilot/histology_pilot_comparison.png)

![Prototype confusion matrix](C:/Users/Lenovo/Documents/Codex/2026-10-14/https-github-com-fortunab-family-baselines/outputs/histology_pilot/histology_pilot_confusion.png)

[Comparison PDF](C:/Users/Lenovo/Documents/Codex/2026-10-14/https-github-com-fortunab-family-baselines/outputs/histology_pilot/histology_pilot_comparison.pdf) | [Confusion PDF](C:/Users/Lenovo/Documents/Codex/2026-10-14/https-github-com-fortunab-family-baselines/outputs/histology_pilot/histology_pilot_confusion.pdf) | [LaTeX table](C:/Users/Lenovo/Documents/Codex/2026-10-14/https-github-com-fortunab-family-baselines/outputs/histology_pilot/histology_pilot_table.tex)

## Sources

- [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/colorectal_histology)
- [Original Kather dataset](https://zenodo.org/records/53169)
- [Kather et al., Scientific Reports 2016](https://doi.org/10.1038/srep27988)
