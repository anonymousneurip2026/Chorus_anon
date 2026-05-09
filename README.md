<div align="center">
  <img src="chorus_logo.png" width="400" alt="CHORUS Logo">
</div>

# CHORUS: Cross-Heuristic Object-oriented Uncertainty-aware Robust Unsupervised Supervision

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue.svg)](https://neurips.cc/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)

> **Note**: This work has been **Submitted for Review in NeurIPS 2026**.

**CHORUS** is a framework for uncertainty-aware multi-encoder fusion in Computational Pathology. It leverages Generalized Canonical Correlation Analysis (GCCA) and uncertainty-aware attention to align and fuse features from diverse vision foundations models (e.g., UNI, GigaPath, ResNet50) into a robust, diagnostic representation.

---

## 🚀 Key Features

- **Multi-Encoder Alignment**: Standardized fusion of $K$ encoders using frozen GCCA projections.
- **Uncertainty-Aware Attention**: 4-stage adaptive fusion pipeline that modulates feature importance based on local uncertainty ($\sigma^2$) and global agreement ($\rho$).
- **Foundation Model Integration**: Native support for **CONCH** (text-guidance) and frozen vision backbones.
- **Production-Ready Sweeps**: Unified benchmarking scripts for TCGA-NSCLC (LUNG) and TCGA-RCC subtyping.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/CHORUS.git
cd CHORUS

# Create environment
conda create -n chorus python=3.12
conda activate chorus

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Project Structure

```text
CHORUS/
├── cca/                # GCCA alignment logic & uncertainty helpers
├── cca_artifacts/      # Standardized alignment matrices (.npz)
├── ckpts/              # Foundation model weights (CONCH)
├── datasets/           # Multi-encoder MIL dataset loaders
├── models/             # Core CHORUS architecture
├── results/            # Benchmark outputs & logs
├── run_lung_sweep.sh   # TCGA-NSCLC execution script
└── run_rcc_sweep.sh    # TCGA-RCC execution script
```

---

---

## 📜 Execution Scripts

The repository includes standardized shell scripts to automate benchmarks across diverse "shot" configurations (4, 8, 16):

| Script | Description |
| :--- | :--- |
| `run_lung_sweep.sh <GPU_ID> <SHOTS>` | Executes all 9 CHORUS variants on TCGA-NSCLC. |
| `run_rcc_sweep.sh <GPU_ID> <SHOTS>` | Executes all 9 CHORUS variants on TCGA-RCC. |
| `scripts/fit_visual_cca.sh` | Fits the GCCA alignment for a specific cohort. |

---

## 🎛️ Command Line Interface (CLI)

CHORUS extends the standard MIL training pipeline with several architecture-specific flags:

### Core Configuration
- `--task`: Target task (`task_tcga_lung_subtyping` or `task_tcga_rcc_subtyping`).
- `--model_type`: Set to `CHORUS`.
- `--chorus_variant`: Select the fusion strategy (`none`, `v1_embed`, `v2_attn`, `v4_hybrid`, etc.).
- `--cca_visual_path`: Path to the `.npz` alignment artifact.
- `--encoder_dir`: Repeatable flag to specify feature sources (e.g., `--encoder_dir uni_v1=path/to/uni`).
- `--encoder_order`: List of encoder names to match the GCCA fitting order.

### Ablation & Stage Control
CHORUS allows surgical disabling of its 4-stage fusion logic for ablation studies:
- `--disable_stage1_rho`: Disables global agreement modulation in Stage 1.
- `--disable_stage2_sigma`: Disables local uncertainty modulation in Stage 2.
- `--disable_stage3_rho`: Disables attention weight modulation in Stage 3.
- `--disable_stage4_qk`: Disables uncertainty-aware Query/Key modulation in Stage 4.

---

### 3. Monitoring
You can monitor live training progress, VRAM usage, and AUC metrics through the **CHORUS Mission Control Hub**:

```bash
# Launch the dashboard server
python mission_control_hub.py

# Access in your browser at
http://localhost:5000
```

---

## 🛠️ Usage

### 1. Feature Preparation
Ensure your patch features are stored in `.h5` or `.pt` format using the TRIDENT-style directory structure:
```text
features/
└── <encoder_name>/
    └── h5_files/
        └── <slide_id>.h5
```

### 2. Launching Benchmarks
Deploy automated sweeps across 4, 8, and 16-shot configurations:

```bash
# TCGA-LUNG Subtyping (LUAD vs. LUSC)
./run_lung_sweep.sh --gpu 0 --shots 16

# TCGA-RCC Subtyping (KICH vs. KIRC vs. KIRP)
./run_rcc_sweep.sh --gpu 0 --shots 16
```

### 3. Monitoring
Open the **Mission Control Dashboard** to view live metrics:
```bash
# Open in your browser
open mission_control.html
```

---

## 🔬 Methodology

CHORUS implements a multi-stage fusion strategy:
1.  **Stage 1 (Alignment)**: Projects diverse features into a 512-dim shared latent space.
2.  **Stage 2 (Local Uncertainty)**: Computes $\sigma^2$ across encoders to identify patch-level disagreement.
3.  **Stage 3 (Global Agreement)**: Modulates attention weights using the GCCA spectrum $\boldsymbol{\rho}$.
4.  **Stage 4 (Text-Guidance)**: Cross-attends image features with clinical prompts via a frozen CONCH backbone.

---

## 🙏 Acknowledgements
This codebase and the CHORUS methodology are built upon the foundational concepts introduced in [FOCUS](https://github.com/dddavid4real/focus). 

---

## 📄 License
This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** - see the [LICENSE](LICENSE) file for details.
