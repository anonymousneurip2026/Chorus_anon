# CHORUS CCA Alignment Artifacts

This directory contains the Generalised Canonical Correlation Analysis (GCCA) artifacts used to align patch-level features from three vision encoders into a shared latent space.

## Artifacts

| Cohort | Filename |
| :--- | :--- |
| **TCGA-NSCLC** | `visual_cca_LUNG_20x_256px_0px_overlap.npz` |
| **TCGA-RCC** | `visual_cca_RCC_20x_256px_0px_overlap.npz` |

## Data Structure

Each `.npz` file contains the following keys:

- **`W_list`**: (List of 3 arrays) Projection matrices for each encoder. Shape: `[D_in, 512]`.
- **`means`**: (List of 3 arrays) Mean vectors for patch-level centering.
- **`rho`**: (Array of 512) Canonical correlations on the calibration set. Used as the global agreement signal $\boldsymbol{\rho}$ in CHORUS.
- **`rho_holdout`**: (Array of 512) Canonical correlations on the held-out validation set for stability check.
- **`encoders`**: Names of the encoders used (`['uni_v1', 'gigapath', 'resnet50']`).
- **`latent_dim`**: Shared latent dimensionality (512).
- **`reg`**: Ridge regularization coefficient used during fitting (1e-3).

## Calibration History

To ensure the alignment respects the few-shot constraints of the experiments, artifacts were fitted strictly using the **16-shot training sets (Fold 0)** of each cohort.

| Cohort | Training Slides | Calibration Slides | Holdout Slides | Total Patches |
| :--- | :--- | :--- | :--- | :--- |
| **TCGA-NSCLC** | 32 | 27 | 5 | 138,131 |
| **TCGA-RCC** | 48 | 43 | 5 | 150,000 |

Detailed slide IDs are available in: `cca_calibration_slide_ids.csv`.

## Technical Details

- **Method**: Regularized Generalized Canonical Correlation Analysis (GCCA).
- **Encoders Aligned**: UNI-v1, GigaPath, ResNet50 (ImageNet).
- **Sampling Strategy**: 
    - Deterministic shuffle (Seed: 0).
    - 10% slide-level holdout (min 5 slides).
    - Uniform patch sampling across calibration slides.
- **Fitting Environment**: NVIDIA H200 NVL (141GB).

## License

These artifacts are released under the **MIT License**. See the `LICENSE` file for full details.

## Usage

To load and inspect an artifact:

```python
import numpy as np

# Load RCC artifact
data = np.load('visual_cca_RCC_20x_256px_0px_overlap.npz', allow_pickle=True)

# Access projection matrices
W_uni = data['W_list'][0]
print(f"UNI Projection Shape: {W_uni.shape}")

# Access agreement spectrum (rho)
rho = data['rho']
print(f"Top-5 Canonical Correlations: {rho[:5]}")
```

For a full diagnostic test, run `python load_cca_test.py`.
