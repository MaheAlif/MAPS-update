# KNN-based Spatial Feature Augmentation for MAPS-style Cell-Type Annotation

## Overview

This directory contains an exploratory experiment evaluating whether **simple spatial neighborhood information**, derived using K-Nearest Neighbors (KNN), can improve MAPS-style cell-type annotation on **CODEX spatial proteomics data**.

The goal of this experiment is **not** to replace MAPS or introduce graph neural networks, but to test a lightweight and interpretable alternative:  
augmenting per-cell protein expression features with **local spatial summary statistics**, while keeping the downstream classifier unchanged.

All experiments reuse the same MLP architecture and training protocol as the baseline model to ensure a controlled comparison.

---

## Motivation

Spatial proteomics data provides both:
- high-dimensional protein expression per cell, and
- precise spatial coordinates (cell centroids).

While MAPS-style models primarily rely on expression features, spatial proximity may encode additional biological context (e.g., immune–tumor interactions).

This experiment asks a focused question:

> **Does naive spatial neighborhood information, summarized via KNN, improve cell-type classification beyond protein expression alone?**

---

## Data Description

### Input Data

The experiment uses processed CODEX data with the following splits:

- `train.csv` — training set  
- `valid.csv` — validation set  

Each row corresponds to a single cell and contains:
- protein marker intensities
- spatial coordinates (`X_cent`, `Y_cent`)
- integer cell-type labels (`cell_label`)

Class names are provided in:
- `class_names.csv`

---

## Methodology

### Baseline Model

- Architecture: Multi-Layer Perceptron (MLP)
- Hidden dimension: 512
- Fixed architecture (same as repo baseline)
- Input features: **protein markers only**
- Training:
  - Adam optimizer
  - Cross-entropy loss
  - Early stopping on validation loss

Baseline performance:
- **Accuracy:** 78.9%
- **Macro F1:** 78.0%

---

### KNN-based Spatial Feature Augmentation

Spatial features are computed **before training** and appended to the input CSVs.

#### KNN configuration
- Neighbor definition: Euclidean distance in `(X_cent, Y_cent)`
- Number of neighbors: `k = 5`
- KNN computed **separately for train and validation splits** (no leakage)

#### Added spatial features
For each cell:
- Mean distance to neighbors (`knn_mean_dist`)
- Standard deviation of neighbor distances (`knn_std_dist`)
- Mean neighborhood expression for selected markers:
  - CD4
  - CD8
  - CD20
  - CD45
  - CD68

These features are **label-free** and do not use ground-truth cell types.

Final input dimensionality:
- Baseline: 52 features
- With KNN: 59 features

---

## Training Protocol

- The **same MLP architecture** and hyperparameters are used for:
  - baseline model
  - KNN-augmented model
- Only the input feature set differs.
- Training and evaluation code is identical otherwise.

This ensures that any performance difference is attributable solely to the added spatial features.

---

## Results

| Model                    | Accuracy | Macro F1 |
|--------------------------|----------|----------|
| Protein-only baseline    | 78.9%    | 78.0%    |
| Protein + KNN features   | 77.6%    | 76.8%    |

The KNN-augmented model shows a **consistent performance decrease of ~1.2 percentage points** in both accuracy and macro F1.

---

## Interpretation

These results suggest that, for this CODEX dataset:

- Protein expression alone captures most of the discriminative signal for cell-type annotation.
- Naively aggregating local spatial neighborhood information via KNN does **not** improve performance and may introduce smoothing noise.
- Spatial context likely requires more structured modeling (e.g., explicit relational modeling) or task-specific design.

This negative result is informative and helps clarify the limits of simple spatial feature engineering.

---

## Files in This Directory

- `KNN_spatially_aware_classification.ipynb`  
  End-to-end notebook covering:
  - KNN feature computation
  - baseline training
  - KNN-augmented training
  - evaluation and comparison

- `train.csv`  
  Training data with KNN-augmented features

- `valid.csv`  
  Validation data with KNN-augmented features

- `class_names.csv`  
  Mapping from integer labels to cell-type names

---

## Notes and Future Work

Possible follow-ups (not implemented here):

- Distance-only spatial features (ablation)
- Sensitivity analysis over different `k` values
- More structured spatial modeling (e.g., graph-based approaches)
- Cell-type–specific spatial analysis rather than global classification

This experiment provides a controlled reference point for such future extensions.

---

## Summary

This directory documents a controlled evaluation of **simple KNN-based spatial feature augmentation** within a MAPS-style classification pipeline. While spatial summaries are intuitive and interpretable, they did not improve performance on this dataset, highlighting the importance of careful spatial modeling in spatial proteomics.
