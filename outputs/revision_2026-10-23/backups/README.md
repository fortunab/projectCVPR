# Center-Balanced Multi-Prototype (CBMP) Benchmark & Dissection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch CPU](https://img.shields.io/badge/pytorch-2.14%2Bcpu-EE4C2C.svg)](https://pytorch.org/)
[![Provenance Audited](https://img.shields.io/badge/audited%20files-51%2C107%20verified-brightgreen.svg)](isolated_datasets/polypgen_extensions_2026-10-20/VALIDATION_DOSSIER.md)
[![License: Academic/Research](https://img.shields.io/badge/license-Academic%20Research-lightgrey.svg)](LICENSE)

An empirical benchmark and mechanistic dissection evaluating **Center-Balanced Multi-Prototype (CBMP)** classifiers against linear and cosine baselines under severe domain shift across **endoscopic video (PolypGen)** and **histopathology (CAMELYON17-WILDS)**.

Target venues: **CVPR 2027** (*Empirical Benchmarks & Mechanistic Failure Dissection*) / **IEEE CBMS 2026/2027** (*Multi-Center Clinical Machine Vision Robustness*).

---

## Table of Contents
- [1. Overview & Key Findings](#1-overview--key-findings)
- [2. Repository Structure](#2-repository-structure)
- [3. Quickstart: 2-Minute Setup](#3-quickstart-2-minute-setup)
  - [Windows (PowerShell)](#windows-powershell)
  - [Linux / WSL / macOS (Bash)](#linux--wsl--macos-bash)
- [4. Running the Pipeline](#4-running-the-pipeline)
  - [Command Reference](#command-reference)
  - [Modular Pipeline Steps](#modular-pipeline-steps)
- [5. Verification Levels for Reviewers](#5-verification-levels-for-reviewers)
- [6. Overleaf Paper Integration](#6-overleaf-paper-integration)
- [7. Experimental Logs & Cryptographic Integrity](#7-experimental-logs--cryptographic-integrity)
- [8. FAQ & Troubleshooting](#8-faq--troubleshooting)

---

## 1. Overview & Key Findings

Recent literature often proposes multi-prototype classifiers to handle multi-modal intra-class diversity. This codebase rigorously evaluates and dissects this hypothesis across frozen encoders (**ResNet-18** and **DINOv2 ViT-B/14**).

### Key Empirical Takeaways:
1. **The Multi-Prototype Failure Mode**: When moving from $K=1$ to $K=4$ prototypes per class, balanced accuracy drops or stagnates across unseen clinical centers:
   - **PolypGen**: ResNet-18 mean balanced accuracy remains ~58.3% vs 58.8% for linear baselines; worst-center remains at 42.1%.
   - **CAMELYON17**: At hospital-level evaluation (17,200 patches from 43 patients), CBMP achieves 64.9% vs 64.9% for single-prototype Nearest Centroid.
2. **Why It Overfits (Mechanistic Dissection)**:
   - *Center Purity*: Sub-prototypes specialize into hospital acquisition artifacts rather than biological variations (center purity surges from **52.2%** to **76.0%** in ResNet-18 and **75.7%** in DINOv2).
   - *Dormancy on Unseen Targets*: Due to domain shift, target center test features activate only a subset of prototypes; on average **0.6 prototypes per class are dormant** ($TV = 0.405$ ResNet / $0.328$ DINOv2).
3. **The Regularized Cure (PSRP)**:
   - Penalizing prototypes supported by too few distinct clinical sequences/patients (**Patient-Support Regularized Prototypes - PSRP**) restores stability and yields **+1.20% balanced accuracy** gain over vanilla CBMP on DINOv2.

---

## 2. Repository Structure

```text
├── README.md                                  # Master documentation (this file)
├── setup_venv.ps1                             # 1-click virtual environment setup (Windows)
├── setup_venv.sh                              # 1-click virtual environment setup (Linux/WSL)
├── run_pipeline.ps1                           # Modular pipeline runner (Windows)
├── run_pipeline.sh                            # Modular pipeline runner (Linux/WSL)
│
├── isolated_datasets/
│   ├── polypgen_v3_copy_2026-10-17/           # Baseline PolypGen audit, features & 3 seeds
│   │   ├── audit/                             # Cohort manifests, frame digests & mappings
│   │   ├── features/                          # Cached ResNet-18 ImageNet-1K V1 features
│   │   ├── results/                           # Predictions & evaluation for seeds 17, 29, 43
│   │   └── logs/                              # Verified stdout/stderr execution logs
│   │
│   ├── polypgen_followups_2026-10-17/         # Diagnostics & stress experiments
│   │   ├── exp1_failure/                      # Threshold control & sequence-level AUROC
│   │   ├── exp2_factorial/                    # Prototypes x Center-balancing ablation (48 evals)
│   │   └── exp3_stress/                       # Sequence budget & retention stress tests
│   │
│   └── polypgen_extensions_2026-10-20/        # CVPR extensions & dissection
│       ├── exp4_dinov2/                       # Frozen DINOv2 ViT-B/14 extension (5,444 images)
│       ├── exp5_camelyon/                     # CAMELYON17 independent hospital benchmark (17.2k patches)
│       ├── cvpr_dissection/                   # Latent geometry, purity & activation shift scripts
│       │   └── figures/                       # Vector PDF & PNG publication figures
│       ├── reproducibility/                   # Reviewer reproduction bundle & test suite
│       │   ├── environment/                   # Pinned requirements & pip check records
│       │   ├── metrics_from_predictions.py    # Level 1 fast metric recalculation
│       │   └── fit_cached_heads.py            # Level 2 classifier head fitting
│       ├── VALIDATION_DOSSIER.md              # Cryptographic audit & checksum records
│       └── overleaf_cvpr_pack.zip             # 1-click Overleaf pack (all tables & figures)
```

---

## 3. Quickstart: 2-Minute Setup

The project runs completely on standard **64-bit CPU** (no GPU, CUDA, or proprietary drivers required). Python 3.12 is recommended.

### Windows (PowerShell)

Open PowerShell as regular user in the repository root:

```powershell
# 1. Automated virtual environment creation & dependency installation
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1

# 2. Activate virtual environment (optional)
.\.venv-cbmp\Scripts\Activate.ps1
```

*Manual fallback if not using the script:*
```powershell
python -m venv .venv-cbmp
.\.venv-cbmp\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-cbmp\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.\.venv-cbmp\Scripts\python.exe -m pip install -r isolated_datasets\polypgen_extensions_2026-10-20\reproducibility\environment\requirements-common.txt
.\.venv-cbmp\Scripts\python.exe -m pip check
```

---

### Linux / WSL / macOS (Bash)

Open terminal in the repository root:

```bash
# 1. Automated virtual environment creation & dependency installation
bash setup_venv.sh

# 2. Activate virtual environment (optional)
source .venv-cbmp/bin/activate
```

*Manual fallback if not using the script:*
```bash
python3 -m venv .venv-cbmp
.venv-cbmp/bin/python -m pip install --upgrade pip setuptools wheel
.venv-cbmp/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv-cbmp/bin/python -m pip install -r isolated_datasets/polypgen_extensions_2026-10-20/reproducibility/environment/requirements-common.txt
.venv-cbmp/bin/python -m pip check
```

---

## 4. Running the Pipeline

The pipeline runner allows executing specific components or the entire experimental pipeline from end to end.

### Command Reference

| Platform | Command | Description |
|---|---|---|
| **Windows** | `powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Step all` | Run full reproduction pipeline |
| **Windows** | `powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Step dissection` | Generate CVPR geometric analysis & figures |
| **Windows** | `powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Step level1` | Fast recalculation of metrics from predictions |
| **Windows** | `powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -Step level2` | Re-fit classifier heads on cached features |
| **Linux/WSL** | `bash run_pipeline.sh all` | Run full reproduction pipeline |
| **Linux/WSL** | `bash run_pipeline.sh dissection` | Generate CVPR geometric analysis & figures |
| **Linux/WSL** | `bash run_pipeline.sh level1` | Fast recalculation of metrics from predictions |
| **Linux/WSL** | `bash run_pipeline.sh level2` | Re-fit classifier heads on cached features |

---

### Modular Pipeline Steps

1. **Step `dissection` (CVPR Latent Geometry & PSRP Gain)**:
   - Computes center-purity, sequence-purity, and target activation shift across 156 configurations.
   - Generates 2D t-SNE latent manifold embeddings with prototype trajectories.
   - Saves 4 publication-ready figures in vector PDF and 300 DPI PNG into `cvpr_dissection/figures/`.
   - Execution time: ~30–45 seconds.

2. **Step `level1` (Reviewer Fast Metrics Recalculation)**:
   - Takes raw test predictions CSVs (PolypGen seeds 17, 29, 43 and CAMELYON17 43 patients).
   - Recalculates paired balanced accuracies, per-center AUROC, and 2,000-iteration bootstrap confidence intervals.
   - Execution time: ~5–10 seconds.

3. **Step `level2` (Classifier Head Re-Fitting)**:
   - Loads frozen ResNet-18 / DINOv2 features and source-only partitions.
   - Re-optimizes all 9 classifier heads (Nearest Centroid, Logistic Regression, Cosine, CBMP) with 5 restarts and cosine annealing.
   - Verifies numerical convergence against audited reference predictions.
   - Execution time: ~45–60 seconds.

4. **Step `reports` (Cryptographic Preservation Check)**:
   - Scans 51,107 historical files to verify zero unintended modification across experimental runs.

---

## 5. Verification Levels for Reviewers

Following open-science standards, the repository supports three hierarchical verification tiers:

| Tier | Script | Target | What it validates | Runtime |
|---|---|---|---|---|
| **Level 1** | `metrics_from_predictions.py` | Prediction CSVs | Paired balanced accuracy, per-center breakdown, bootstrap 95% CIs | < 10 sec |
| **Level 2** | `fit_cached_heads.py` | Feature caches (`.npz`) | Source tuning grids, optimization convergence, classifier parity | ~ 1 min |
| **Level 3** | `reproduce.py stage/run` | Raw image archives | Full image loading, frozen encoder feature extraction, end-to-end replay | ~ 2–3 hrs |

> [!NOTE]
> When evaluating with Python 3.12 and newer PyTorch builds (`>= 2.14`), pass `--allow-version-drift` to `fit_cached_heads.py` to acknowledge the expected BLAS/threading numerical tolerance. Both `run_pipeline.ps1` and `run_pipeline.sh` automatically handle this flag.

---

## 6. Overleaf Paper Integration

All manuscript tables and publication figures are packaged into an automated bundle ready for drag-and-drop into Overleaf:

**Package Path**: `isolated_datasets/polypgen_extensions_2026-10-20/overleaf_cvpr_pack.zip`

### What is Inside:
- `tables/`:
  - `tab1_main_benchmark_resnet18.tex`: Main 9-method PolypGen comparison across centers C2–C6.
  - `tab2_dinov2_foundation_comparison.tex`: ResNet-18 vs DINOv2 transfer performance.
  - `tab3_camelyon17_independent_benchmark.tex`: Multi-hospital digital pathology validation.
  - `tab4_prototype_center_purity.tex`: Quantified domain artifact collapse ($K=1$ vs $K=4$).
  - `tab5_test_activation_shift.tex`: Total variation distance and dormant prototype fractions.
  - `tab6_psrp_regularization_gain.tex`: Balanced accuracy gains with sequence-support regularization.
- `figures/`:
  - `fig_tsne_latent_geometry.pdf` / `.png`: Latent geometry and prototype collapse visualization.
  - `fig_prototype_center_purity.pdf` / `.png`: Surge in domain artifact entrapment.
  - `fig_test_activation_shift.pdf` / `.png`: Distribution shift and dormant prototype counts.
  - `fig_psrp_regularization_gain.pdf` / `.png`: Recovery curve across regularization strengths.

### How to use in your `.tex` document:
Simply upload `overleaf_cvpr_pack.zip` directly to Overleaf (New File $\rightarrow$ Upload $\rightarrow$ Zip), then include directly:
```latex
\input{tables/tab4_prototype_center_purity.tex}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_tsne_latent_geometry.pdf}
  \caption{Latent space visualization of learned multi-prototypes.}
  \label{fig:latent_geometry}
\end{figure}
```

---

## 7. Experimental Logs & Cryptographic Integrity

Every single benchmark run has been logged to disk with full timestamps, environment metadata, and exit codes. 

Detailed audits and checksums are recorded in [`VALIDATION_DOSSIER.md`](isolated_datasets/polypgen_extensions_2026-10-20/VALIDATION_DOSSIER.md).

Key logs:
- PolypGen 3-Seed Baselines: `isolated_datasets/polypgen_v3_copy_2026-10-17/logs/seed{17,29,43}.log`
- Follow-up Failure Analysis: `isolated_datasets/polypgen_followups_2026-10-17/exp1_failure/processing.log`
- Factorial & Stress Tests: `isolated_datasets/polypgen_followups_2026-10-17/logs/exp{2,3}_*.log`
- DINOv2 Foundation Extraction: `isolated_datasets/polypgen_extensions_2026-10-20/exp4_dinov2/logs/processing.log`
- CAMELYON17 Multi-Hospital: `isolated_datasets/polypgen_extensions_2026-10-20/exp5_camelyon/run.log`

---

## 8. FAQ & Troubleshooting

### Q: Why do the scripts run on CPU by default?
A: Medical vision benchmark classifiers on frozen features (512-dim ResNet, 768-dim DINOv2) converge in under a minute on modern multi-core CPUs. CPU execution guarantees cross-platform deterministic reproducibility across different machines without CUDA driver or GPU architecture drift.

### Q: What is the `--allow-version-drift` flag?
A: The original baseline was recorded on PyTorch `2.13.0`. When running on PyTorch `2.14.1+cpu` (standard with Python 3.12), floating point BLAS operations may produce subtle minor variations ($\le 10^{-5}$) that do not alter classifier decisions. Passing this flag permits evaluation while maintaining a transparent log.

### Q: On Windows, PowerShell gives an ExecutionPolicy error.
A: Run PowerShell commands with `-ExecutionPolicy Bypass`, for example:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
```

### Q: Where are the raw datasets obtained?
- **PolypGen**: Request official access via Synapse ([Synapse ID: syn26376615](https://www.synapse.org/Synapse:syn26376615/wiki/613312)). SHA-256: `a22f956a9f0fc3f941927108a3de8217fc61f1944c7eda598536dbabc27dbff3`.
- **CAMELYON17**: Downloaded via public Git-LFS shards from official WILDS repository and validated by SHA-256.
