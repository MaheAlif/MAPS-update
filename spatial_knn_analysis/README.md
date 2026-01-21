# Spatial Neighborhood Analysis using KNN (CODEX – cHL)

## Overview
This directory contains an exploratory spatial neighborhood analysis built on top of the **MAPS project**, using raw CODEX spatial proteomics annotations.

The goal is to understand whether cell types exhibit structured spatial co-localization, and to motivate (or caution against) later spatial modeling attempts. This analysis was requested as a spatial neighborhood exploration, not as a training or benchmarking experiment.

---

## Dataset
* **File:** `cHL_CODEX_annotation.csv`
* **Modality:** CODEX
* **Per-cell information:**
    * Spatial coordinates (`X_cent`, `Y_cent`)
    * Protein expression markers
    * Ground-truth cell type (`cellType`)

*Note: This file is used because MAPS training CSVs do not contain spatial coordinates.*

---

## Goal
To answer the descriptive question:
> **Which cell types tend to physically co-localize in tissue space?**

This step is intended to:
1.  Verify the existence of spatial structure.
2.  Characterize neighborhood composition.
3.  Justify or contextualize later spatial modeling attempts.

---

## Method: Spatial KNN Neighborhood Analysis
1.  **Representation:** Each cell is represented by its physical centroid (`X_cent`, `Y_cent`).
2.  **KNN Search:** A KNN search ($k = 10$) is performed in Euclidean space.
3.  **Identification:** For each cell, its 10 nearest physical neighbors are identified, and neighbor cell types are counted.
4.  **Interaction Matrix:** A conditional interaction matrix is computed:
    $$P(\text{Neighbor Type} \mid \text{Center Cell Type})$$
    *(Rows are normalized so each row sums to 1).*

---

## Outputs

### 1. Tissue Map (Physical Locations)
A scatter plot of all cells:
* **X-axis:** `X_cent`
* **Y-axis:** `Y_cent`
* **Color:** `cellType`

**Visualizes:** Strong spatial organization, contiguous regions of similar cell types, and non-random tissue architecture.

### 2. Spatial Neighborhood Enrichment Heatmap
A heatmap where:
* **Rows:** Center cell type
* **Columns:** Neighbor cell type
* **Values:** Conditional neighborhood probabilities

**Key Observations:** Strong self-association (diagonal), immune–immune co-localization, compact tumor regions, and selective (non-random) neighborhood structure.

---

## Scope and Limitations

### ✅ This analysis IS:
* Descriptive and exploratory.
* Label-aware.
* Biologically interpretable.

### ❌ This analysis IS NOT:
* A classifier.
* A MAPS training step.
* Usable directly at inference time.
* *It is intentionally diagnostic, not predictive.*

---