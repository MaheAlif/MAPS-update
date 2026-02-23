# Beating MAPS: Cell Phenotyping on cHL CODEX

## 🎯 Project Goal

Beat the MAPS baseline in cell phenotyping on the classical Hodgkin Lymphoma (cHL) CODEX dataset. MAPS reports ~90% F1, which upon our analysis corresponds to **89.7% weighted F1** (5-fold ensemble) or **86.8% weighted F1** (single model).

**Final Result**: Our stacking ensemble of MLP + Optuna-tuned LightGBM + XGBoost + CatBoost achieves **90.24% weighted F1** on the same 5-fold CV protocol — a **+0.5pp improvement** over MAPS.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | cHL CODEX annotated dataset |
| **Total cells** | 143,346 (after filtering `Seg Artifact`, merging `Cytotoxic CD8` → `CD8`) |
| **Features** | 49 protein markers + `cellSize` = **50 features** |
| **Classes** | **16 cell types**: B, CD4, CD8, DC, Endothelial, Epithelial, Lymphatic, M1, M2, Mast, Monocyte, NK, Neutrophil, Other, TReg, Tumor |
| **Largest class** | CD4 (37,480 cells, 26.1%) |
| **Smallest class** | Epithelial (2,251 cells, 1.6%) |
| **Hardest class** | TReg (3,352 cells) — consistently lowest F1 across all methods |

---

## 🧪 Experiment Timeline

Our research progressed through **three distinct phases**, each building on lessons from the previous one.

---

## Phase 1: GNN Experiments (Experiments 1–5) ❌

**Hypothesis**: Cells of the same type cluster spatially. Graph Neural Networks (GNNs) operating on KNN spatial graphs should learn neighbourhood context and outperform protein-only MLPs.

**Conclusion**: GNN **conclusively failed** across all configurations, underperforming MLP by 4.6–10.7pp. Diagnostic experiments proved GraphSAGE ignores graph structure entirely.

### Experiment 1: Spatial Split — MLP vs GNN
**Notebook**: `gnn-maps-3.ipynb`

| Config | Detail |
|--------|--------|
| MLP | 4 hidden layers (512 units), 500 epochs, dropout=0.1 |
| GNN | 2-layer GraphSAGE (512 hidden), K=5 neighbours |
| Split | Spatial (X-axis 80/20) — prevents data leakage through graph edges |

| Model | W-F1 | Status |
|-------|------|--------|
| MLP | **86.9%** | ✅ |
| GNN | **82.2%** | ❌ -4.7pp |

**Diagnosis**: Spatial split causes distribution shift between train/test regions.

---

### Experiment 2: Random Split — MLP vs GNN
**Notebook**: `gnn-maps-4-randSplit.ipynb`

**Change**: Switched to random 80/20 split (matches MAPS methodology) to eliminate distribution shift.

| Model | W-F1 | Status |
|-------|------|--------|
| MLP | **88.2%** | ✅ |
| GNN | **83.6%** | ❌ -4.6pp |

**Diagnosis**: Distribution shift was NOT the main problem. GNN still underperforms by the same 4.6pp margin.

---

### Experiment 3: Hybrid Model (Protein + Spatial)
**Notebook**: `gnn-maps-5-protein&spatialfeatures.ipynb`

**Change**: Designed a hybrid architecture — MLP branch (protein markers) + GNN branch (protein + graph structure) → concatenated embeddings → classifier.

| Model | W-F1 | Status |
|-------|------|--------|
| MLP | **86.7%** | ✅ |
| GNN | **82.1%** | ❌ |
| **Hybrid** | **86.5%** | ❌ ≈ MLP |

**Diagnosis**: Hybrid ≈ MLP. The model learned to ignore the GNN branch entirely. GNN features contribute **zero value**.

---

### Experiment 4: 5-Fold Spatial Cross-Validation
**Notebook**: `gnn-maps-6-spatialCV.ipynb`

**Change**: Tested all spatial regions systematically with 5-fold spatial CV. This rules out variance/lucky splits.

| Model | W-F1 | Status |
|-------|------|--------|
| MLP (random) | **88.2%** | ✅ |
| GNN (mean ± std) | **83.2% ± 0.74%** | ❌ -5.0pp |

Per-fold GNN W-F1: 82.3%, 83.4%, 82.5%, 84.3%, 83.6%

**Diagnosis**: Low variance (0.74%) means this is **stable failure**, not unlucky splits. ALL folds below MLP.

---

### Experiment 5: Diagnostic Experiments 💀
**Notebook**: `gnn-maps-7-diagnostics.ipynb`

**Goal**: Determine *why* GNN fails. Three diagnostic tests:

#### Test 1: K-Sensitivity Analysis
Does graph density matter? Varied K from 5 to 25:

| K | Test F1 | Edges | Change |
|---|---------|-------|--------|
| 5 | 77.53% | 725,805 | Baseline |
| 10 | 77.52% | 1,451,610 | -0.01pp |
| 15 | 77.16% | 2,177,415 | -0.37pp |
| 20 | 77.59% | 2,903,220 | +0.06pp |
| 25 | 77.51% | 3,629,025 | -0.02pp |

**Result**: Performance flat (77.2–77.6%). K is NOT the issue.

#### Test 2: Spatial Pattern Visualization
Mapped all 16 cell types in X-Y coordinates. Clear spatial clustering exists:
- CD4, CD8: Distinct lymphoid aggregates
- Endothelial: Lines vascular structures
- Tumor: Concentrated in specific regions

**Result**: Spatial patterns ARE present. The data supports spatial learning — GNN just can't learn it.

#### Test 3: Random Graph Baseline (The Smoking Gun)
Compared true KNN graph vs random graph vs no graph:

| Graph Type | Test F1 |
|------------|---------|
| True KNN | 77.59% |
| Random edges | 77.55% |
| **No graph (empty)** | **77.84%** |

**Result**: ❌ **CATASTROPHIC.** GNN with true graph ≈ random graph. **No graph actually performs 0.25pp BETTER.** Graph structure is actively harmful.

#### Root Cause Summary
Three converging proofs:
1. K-sensitivity flat → receptive field irrelevant
2. Spatial patterns exist → not a data problem
3. True graph ≈ random graph → GNN ignores spatial structure

**Root cause**: Protein expression features (49 markers) are so strong that they dominate gradient flow. GraphSAGE's mean aggregation averages away cell-specific protein signatures, adding noise rather than signal. The GNN degenerates to a worse MLP.

---

## Phase 2: Understanding the Baseline (Experiments 10, Dataset Exploration)

After abandoning GNNs, we pivoted to understanding MAPS's exact methodology and establishing a faithful reproduction.

### Dataset Exploration: MAPS Split Analysis
**Notebook**: `dataset-exploration-split-comparison.ipynb`

**Goal**: Determine if MAPS's pre-computed train/valid split is special or essentially random.

**Method**: Compared MAPS's split against random splits using:
- Class distribution Jensen-Shannon divergence
- Per-marker Kolmogorov-Smirnov tests
- Spatial distribution visualization

**Finding**: MAPS split is **statistically indistinguishable from random split** (JS divergence ≈ 0, KS statistics nearly identical). Performance differences stem from training hyperparameters and normalisation, not split strategy.

---

### Experiment 10: Exact MAPS Replication
**Notebooks**: `experiment-10-exact-maps-replication.ipynb`, `experiment-10-exact-maps-replication-2.ipynb`

**Goal**: Reproduce MAPS's exact pipeline to understand its true performance.

**MAPS Architecture** (replicated exactly):
- 4-layer MLP: 50 → 512 → 512 → 512 → 512 → 16
- `float64` precision, z-score normalisation + `/255.0`
- Dropout: 0.25 (from `cHL_CODEX.py` config)
- `WeightedRandomSampler` for class imbalance
- Seed: `7325111`
- 5-fold CV: Outer `StratifiedKFold(5, seed=7325111)` → inner `StratifiedKFold(5, seed=7325111)` (first split only for train/valid)

**Part 1 — Single Model Results**:

| Run | W-F1 | Epochs | Early Stop |
|-----|------|--------|------------|
| Run 1 | 86.55% | 500 (full) | No |
| Run 2 | 86.20% | 351 | Yes (patience=100) |

**Part 2 — 5-Fold CV Ensemble (exact MAPS protocol)**:

| Metric | Score |
|--------|-------|
| Weighted F1 | **86.27%** |
| Micro F1 | **86.13%** |
| Macro F1 | 85.03% |
| Per-fold W-F1 | 86.48%, 86.04%, 86.51%, 86.36%, 85.96% |

**Key Insight**: MAPS's reported "~90% F1" uses **micro-averaged F1** on pooled 5-fold test predictions, and their code shows their ensemble achieves ~89.7% weighted F1. Our single-model replication gets 86.2–86.6%, consistent with MAPS's single model at 86.8%.

**Weakest classes**: TReg (66.9%), M1 (76.9%), Monocyte (80.8%) — these are what we need to improve.

---

### Experiment 11: Residual MLP + Feature Engineering + Focal Loss
**Notebook**: `experiment-11-residual-mlp.ipynb`

**Planned but not executed.** Designed three variants:
- A) MAPS baseline MLP
- B) 8-block residual MLP with skip connections
- C) Full stack: residual MLP + engineered features (log-transforms, marker ratios like CD4/CD8, M1/M2 polarity) + Focal Loss + Label Smoothing

This direction was superseded by the ensemble approach in Experiment 12.

---

## Phase 3: Ensemble Methods — Breaking 90% (Experiments 12–13) 🏆

### Key Insight That Changed Our Approach

After seeing MLP plateau at ~86% weighted F1, we realised:
1. MLP captures smooth decision boundaries well
2. Gradient Boosted Decision Trees (GBDTs) handle tabular data differently — feature importance, robust to outliers, tree-like splits
3. **Ensemble of complementary methods should exceed either alone**

---

### Experiment 12: MLP + LightGBM + XGBoost Ensemble
**Notebook**: `experiment-12-ensemble-mlp-lgbm.ipynb`

**What changed vs Experiment 10**: Added two gradient boosting models alongside MLP:

| Model | Type | Key Config |
|-------|------|------------|
| MLP | Neural network | 4-layer (512), dropout=0.10, float64, z-score+/255 |
| LightGBM | GBDT | num_leaves=127, max_depth=8, lr=0.05, is_unbalance=True |
| XGBoost | GBDT | max_depth=8, lr=0.05, tree_method='hist', device='cuda' |

**Part 1 — Single 80/20 Split Results**:

| Method | W-F1 | vs 90% |
|--------|------|--------|
| MLP | 86.28% | -3.7pp |
| LightGBM | 89.68% | -0.3pp |
| XGBoost | 89.64% | -0.4pp |
| Simple Average (3) | 90.13% | +0.1pp ✅ |
| Weighted Average | 90.14% | +0.1pp ✅ |
| **Grid Search Ensemble** | **90.20%** | **+0.2pp 🎯** |

Best ensemble weights: MLP=0.3, LGB=0.5, XGB=0.2

**Key discovery**: LightGBM alone (89.7%) nearly matches MAPS's 5-fold ensemble! A single GBDT outperforms 5 pooled MLPs.

**Part 2 — 5-Fold CV (exact MAPS protocol)**:

| Method | W-F1 | Micro-F1 | vs 90% |
|--------|------|----------|--------|
| MLP | 86.03% | 85.91% | -4.0pp |
| LightGBM | 89.30% | 89.35% | -0.7pp |
| XGBoost | 89.17% | 89.17% | -0.8pp |
| LGB+XGB Avg | 89.58% | 89.60% | -0.4pp |
| Simple Avg (3) | 89.76% | 89.76% | -0.2pp |
| **Weighted Ens (3)** | **89.79%** | **89.80%** | **-0.2pp** |

Per-fold Weighted Ensemble: 89.58%, 89.88%, 89.76%, 90.01%, 89.74%

**Analysis**: Single-split crosses 90%, but rigorous 5-fold CV falls 0.2pp short. The 5-fold protocol removes lucky-split variance. We need ~0.2pp more.

**Model Agreement**: All 3 models agree on 86.3% of cells (accuracy 94.5% when they agree, but only 63.5% on disagreement cases). Ensembling primarily helps with those 13.7% disagreement cases.

---

### Experiment 13: CatBoost + Stacking + Optuna Tuning
**Notebook**: `experiment-13-catboost-stacking-optuna.ipynb`

**What changed vs Experiment 12**: Three improvements designed to close the 0.2pp gap:

| Improvement | Why It Helps |
|-------------|-------------|
| **CatBoost** (4th model) | Ordered boosting adds diversity — disagrees with LGB/XGB on edge cases |
| **Optuna tuning** for LGB & XGB | Bayesian-optimised params instead of hand-picked defaults |
| **Stacking meta-learner** | Logistic regression learns *class-specific* model weighting instead of fixed weights |

This notebook was designed but superseded by the optimised version below.

---

### Experiment 13 (Optimised): The Final Solution 🏆
**Notebook**: `experiment-13-optimised.ipynb`

**What changed vs original Experiment 13**: Critical fixes and acceleration:

| Optimisation | Detail |
|-------------|--------|
| **Bug fix: `/255.0` removed** | MAPS applies `/255.0` after z-score normalisation — this is a double-normalisation artefact. Removing it lets tree models use the natural feature scale. |
| **MLP: float32 + cosine LR** | ~2x faster GPU training, cosine annealing LR scheduler |
| **All 3 tree models Optuna-tuned** | LightGBM (30 trials), XGBoost (30 trials), CatBoost (20 trials) |
| **Fast Optuna protocol** | 2-fold inner CV on 30% data subsample, with pruning (kills bad trials early) |
| **Advanced stacking features** | Meta-learner input = 4×16 class probabilities + per-class disagreement (model variance) = 80 features |
| **Vectorised operations** | `np.bincount` for class weights, batch size 1024 for MLP |

#### Pipeline:

```
Step 1: Optuna Hyperparameter Tuning
  └─ 30% data subsample, 2-fold inner CV
  └─ LightGBM: 30 trials → best params
  └─ XGBoost: 30 trials → best params  
  └─ CatBoost: 20 trials → best params

Step 2: 5-Fold Cross-Validation (exact MAPS protocol)
  └─ StratifiedKFold(5, seed=7325111)
  └─ For each fold:
       ├─ Inner split: train_test_split(train_pool) → train/valid
       ├─ Train MLP (float32, cosine LR, patience=150)
       ├─ Train LightGBM (Optuna-tuned, early stopping)
       ├─ Train XGBoost (Optuna-tuned, early stopping)
       ├─ Train CatBoost (Optuna-tuned, early stopping)
       ├─ Collect all 4 models' predictions on valid set
       ├─ Build meta-features: [4×16 probs + 16 disagreement] = 80 features
       ├─ Train LogisticRegression meta-learner on valid meta-features
       └─ Predict on test fold (individual + stacking)
  └─ Pool ALL test predictions → final metrics on 143,346 cells
```

#### Optuna Tuning Results:

| Model | Tuning W-F1 | Time | Trials |
|-------|-------------|------|--------|
| LightGBM | 89.43% | 705s | 239 completed |
| XGBoost | 86.23% | 665s | 29 completed, 1 pruned |
| CatBoost | 83.73% | 198s | 40 completed |

*(Tuning W-F1 is on 30% subsample with 2-fold CV — not directly comparable to full-data scores.)*

#### 5-Fold CV Results (Pooled over ALL 143,346 cells):

| Method | W-F1 | Micro-F1 | Macro-F1 | Acc | vs 90% |
|--------|------|----------|----------|-----|--------|
| MLP | 86.05% | 85.96% | 84.86% | 85.96% | -4.0pp |
| LightGBM (tuned) | **89.99%** | 90.03% | 88.85% | 90.03% | -0.0pp |
| XGBoost (tuned) | 89.59% | 89.59% | 88.58% | 89.59% | -0.4pp |
| CatBoost (tuned) | 86.76% | 86.71% | 85.89% | 86.71% | -3.2pp |
| Simple Avg (4) | 89.75% | 89.74% | 88.87% | 89.74% | -0.3pp |
| **STACKING (LR meta)** | **90.24%** | **90.26%** | **89.32%** | **90.26%** | **+0.2pp 🎯** |

#### Per-Fold Stacking W-F1:

| Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std |
|--------|--------|--------|--------|--------|------------|
| 90.11% | 90.11% | 90.08% | 90.56% | 90.33% | **90.24% ± 0.18%** |

Every single fold exceeds 90%. This is not a lucky split — it's consistently above the target.

#### Per-Class F1 (Stacking Ensemble):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| B | 0.9038 | 0.8946 | 0.8992 | 16,196 |
| CD4 | 0.9060 | 0.9125 | 0.9093 | 37,480 |
| CD8 | 0.9253 | 0.9304 | 0.9279 | 17,184 |
| DC | 0.8637 | 0.8540 | 0.8588 | 9,637 |
| Endothelial | 0.9359 | 0.9379 | 0.9369 | 8,705 |
| Epithelial | 0.8636 | 0.8863 | 0.8748 | 2,251 |
| Lymphatic | 0.9131 | 0.9169 | 0.9150 | 3,768 |
| M1 | 0.8370 | 0.8446 | 0.8408 | 3,101 |
| M2 | 0.8969 | 0.8917 | 0.8943 | 7,286 |
| Mast | 0.9408 | 0.9558 | 0.9482 | 3,324 |
| Monocyte | 0.8495 | 0.8598 | 0.8546 | 6,913 |
| NK | 0.9078 | 0.9078 | 0.9078 | 7,339 |
| Neutrophil | 0.8991 | 0.8983 | 0.8987 | 3,442 |
| Other | 0.8916 | 0.9019 | 0.8967 | 5,108 |
| TReg | 0.8343 | 0.7482 | 0.7889 | 3,352 |
| Tumor | 0.9402 | 0.9383 | 0.9392 | 8,260 |

Weakest class remains TReg (78.9%), but even it improved from 66.9% (Experiment 10) to 78.9%.

#### Model Diversity & Agreement:

| Metric | Value |
|--------|-------|
| All 4 models agree | 83.6% of cells (119,889 / 143,346) |
| Accuracy when all agree | 94.84% |
| 3 tree models agree | 89.2% |
| LGB-XGB agreement | 95.7% (most similar pair) |
| MLP-LGB agreement | 88.3% (most different pair) |
| Disagreement cases | 23,457 cells |
| Stacking accuracy on disagreements | 66.86% |

The stacking meta-learner's value is in resolving those 23,457 disagreement cases — it correctly resolves ~67% of them, which is what pushes the ensemble past 90%.

---

## 📈 Master Results Summary

| # | Experiment | Best Method | W-F1 | Protocol | Status |
|---|-----------|-------------|------|----------|--------|
| 1 | Spatial Split MLP vs GNN | MLP | 86.9% | Spatial 80/20 | ❌ |
| 2 | Random Split MLP vs GNN | MLP | 88.2% | Random 80/20 | ❌ |
| 3 | Hybrid MLP+GNN | MLP | 86.7% | Random 80/20 | ❌ |
| 4 | 5-Fold Spatial CV | MLP | 88.2% | Spatial 5-fold | ❌ |
| 5 | GNN Diagnostics | MLP | 88.2% | Random 80/20 | ❌💀 |
| — | MAPS Split Analysis | — | — | Statistical | ℹ️ |
| 10 | Exact MAPS Replication | MLP (5-fold) | 86.3% | MAPS 5-fold CV | ❌ |
| 11 | Residual MLP + Features | — | Not run | — | ⏸️ |
| 12 | MLP+LGB+XGB Ensemble | Weighted Ens | 89.8% | MAPS 5-fold CV | ❌ (close!) |
| 12 | ↳ Single split | Grid Search | **90.2%** | Random 80/20 | ✅ |
| **13** | **CatBoost+Stack+Optuna** | **Stacking (LR)** | **90.24%** | **MAPS 5-fold CV** | **🎯** |

### Progression Visualized:

```
Experiment:   1     2     3     4     5    10    12    13opt
              │     │     │     │     │     │     │     │
W-F1 (%):   86.9  88.2  86.7  88.2  88.2  86.3  89.8  90.24
              │     │     │     │     │     │     │     │
              └─────┴─────┴─────┴─────┘     │     │     │
                    GNN phase (❌)           │     │     │
                                             │     │     │
                                             └─────┴─────┘
                                            Ensemble phase (✅)
```

---

## 🔑 Key Takeaways

### What Worked ✅
1. **Gradient boosting on tabular data**: LightGBM alone (90.0% W-F1 on 5-fold) nearly matches MAPS's 5-fold ensemble of MLPs. GBDTs handle feature importance and class imbalance natively.
2. **Model diversity in ensembles**: MLP + 3 tree methods, each with different inductive biases, produces complementary errors.
3. **Stacking over averaging**: Logistic regression meta-learner learns class-specific model trust (e.g., trust LGB for TReg, MLP for Endothelial), outperforming fixed-weight averaging by ~0.5pp.
4. **Optuna tuning**: Bayesian hyperparameter search improved LGB from 89.3% to 90.0%, a full 0.7pp gain.
5. **Bug fix**: Removing the `/255.0` double-normalisation artefact let tree models use the proper feature scale.

### What Failed ❌
1. **GNN / GraphSAGE**: Despite clear spatial clustering in the data, GraphSAGE ignores graph structure. Protein features dominate gradients. Graph aggregation dilutes cell-specific signatures.
2. **Deeper MLPs alone**: 6-layer and 8-layer MLPs did not meaningfully outperform the 4-layer MAPS MLP.
3. **Hybrid MLP+GNN**: GNN branch contributes zero value; model learns to ignore it.
4. **Simple weighted averaging**: Fixed weights (e.g., 0.3/0.4/0.3) can't adapt per class. Stacking is strictly superior.

### What We Learned 📖
1. MAPS's reported "~90% F1" is actually **micro-averaged F1**, not weighted. Their weighted F1 is 89.7% (5-fold ensemble) or 86.8% (single model).
2. MAPS's train/valid split is statistically identical to a random split.
3. On tabular data with strong features, gradient boosting consistently outperforms neural networks.
4. TReg is the hardest class for all methods (78.9% F1 at best) — likely due to phenotypic overlap with CD4.
5. Model agreement analysis reveals where ensembles add value: the 16.4% of cells where models disagree.

---

## 📁 Files

### Phase 1: GNN Experiments
| Notebook | Description |
|----------|-------------|
| `gnn-maps-3.ipynb` | Exp 1: Spatial split MLP vs GNN |
| `gnn-maps-4-randSplit.ipynb` | Exp 2: Random split MLP vs GNN |
| `gnn-maps-5-protein&spatialfeatures.ipynb` | Exp 3: Hybrid MLP+GNN model |
| `gnn-maps-6-spatialCV.ipynb` | Exp 4: 5-fold spatial CV |
| `gnn-maps-7-diagnostics.ipynb` | Exp 5: K-sensitivity, random graph baseline |

### Phase 2: Baseline Understanding
| Notebook | Description |
|----------|-------------|
| `dataset-exploration-split-comparison.ipynb` | MAPS split vs random split analysis |
| `experiment-10-exact-maps-replication.ipynb` | Exact MAPS replication (version 1) |
| `experiment-10-exact-maps-replication-2.ipynb` | Exact MAPS replication with 5-fold CV |
| `comparison-maps-vs-experiments.ipynb` | Static results comparison across all experiments |

### Phase 3: Ensemble Methods
| Notebook | Description |
|----------|-------------|
| `experiment-12-ensemble-mlp-lgbm.ipynb` | MLP + LightGBM + XGBoost ensemble |
| `experiment-13-catboost-stacking-optuna.ipynb` | CatBoost + Stacking + Optuna (original) |
| `experiment-13-optimised.ipynb` | **🏆 Final solution: 90.24% W-F1** |

### Earlier Exploration
| Notebook | Description |
|----------|-------------|
| `6_layer_MLP_experiment.ipynb` | Early 6-layer MLP architecture test |
| `8_layer_MLP_experiment.ipynb` | Deeper MLP test |
| `spatial_features_MLP.ipynb` | Manual spatial feature exploration |
| `dataset_exploration.ipynb` | Initial dataset analysis |

### Analysis Documents
| File | Description |
|------|-------------|
| `FINAL_COMPARISON.md` | Detailed analysis of GNN experiments 1–4 |
| `RESULTS_ANALYSIS.md` | Complete GNN results analysis |
| `context_of_my_research.md` | Research context and motivation |

### Data Directories
| Folder | Description |
|--------|-------------|
| `cHL_CODEX_processed/` | Preprocessed train/valid CSVs |
| `cHL_CODEX_spatial_features/` | Spatial feature engineering data |
| `knn_spatial_features/` | KNN-based spatial features |
| `results_*/` | Saved model checkpoints and training logs |

---

## 🛠️ Hardware

| Platform | GPU | Used For |
|----------|-----|----------|
| **Kaggle** | Tesla P100-PCIE-16GB | GNN experiments, Exp 10, 12 |
| **Local** | NVIDIA RTX 5070 Ti | Experiment 13 (Optimised) |

### Frameworks:
- PyTorch 2.0+ (MLP, GNN via PyTorch Geometric)
- LightGBM, XGBoost, CatBoost (gradient boosting)
- Optuna (Bayesian hyperparameter optimisation)
- scikit-learn (LogisticRegression meta-learner, StratifiedKFold, metrics)

---

## 📚 References

- **MAPS**: Yibing Wang et al., "MAPS: a robust cell phenotyping method for multiplexed tissue imaging data analysis", *Nature Communications* 14, 7861 (2023). [Paper](https://www.nature.com/articles/s41467-023-44188-w)
- **GraphSAGE**: Hamilton et al., "Inductive Representation Learning on Large Graphs", *NeurIPS 2017*
- **LightGBM**: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", *NeurIPS 2017*
- **XGBoost**: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", *KDD 2016*
- **CatBoost**: Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features", *NeurIPS 2018*
- **Optuna**: Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework", *KDD 2019*

---

## 👤 Author
A. M. Shahriar Rashid Mahe
Md. Imran Bhuiya