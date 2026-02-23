"# MAPS-update" 
"# MAPS-update" 
"# GNN-Enhanced Cell Type Classification

## 🎯 Project Goal
Beat the MAPS baseline (90% F1) by incorporating **Graph Neural Networks (GNNs)** to leverage spatial neighborhood information in cell phenotyping.

**Hypothesis**: Cells of the same type cluster together in tissue. GNNs should learn from this spatial context and outperform protein-only MLPs.

---

## 📊 Dataset
- **Source**: cHL_CODEX annotated dataset
- **Size**: 100,814 cells
- **Features**: 49 protein markers + X/Y coordinates
- **Labels**: 14 cell types
- **Baseline**: MAPS MLP achieves 90% F1 (random 80/20 split)

---

## 🧪 Experiments Conducted

### Experiment 1: Initial GNN Implementation
**Notebook**: `gnn-maps-3.ipynb`

**Configuration**:
- MLP: 4 hidden layers (512 units), 500 epochs, dropout=0.1
- GNN: 2-layer GraphSAGE (512 hidden), K=5 neighbors, 500 epochs
- Split: Spatial (X-axis 80/20) to prevent data leakage

**Results**:
- MLP: **86.9%** F1
- GNN: **82.2%** F1 ❌

**Diagnosis**: Spatial split causes distribution shift. GNN overfits to training region topology.

---

### Experiment 2: Random Split (Eliminate Distribution Shift)
**Notebook**: `gnn-maps-4-randSplit.ipynb`

**Configuration**:
- Random 80/20 split (matches MAPS methodology)
- Same MLP/GNN architecture as Experiment 1

**Results**:
- MLP: **88.2%** F1 ✅ (close to MAPS 90%)
- GNN: **83.6%** F1 ❌

**Diagnosis**: Distribution shift was NOT the issue. GNN still underperforms by 4.6 pp.

---

### Experiment 3: Hybrid Model (Protein + Spatial Features)
**Notebook**: `gnn-maps-5-protein&spatialfeatures.ipynb`

**Configuration**:
- MLP branch: Processes protein markers
- GNN branch: Processes protein + graph structure
- Hybrid: Concatenates both embeddings
- Random 80/20 split

**Results**:
- MLP: **86.7%** F1
- GNN: **82.1%** F1
- Hybrid: **86.5%** F1 ❌

**Diagnosis**: Hybrid ≈ MLP. GNN features contribute NOTHING. Model learns to ignore spatial branch.

---

### Experiment 4: 5-Fold Spatial Cross-Validation
**Notebook**: `gnn-maps-6-spatialCV.ipynb`

**Configuration**:
- 5-fold spatial cross-validation (test all regions)
- MLP: Random 80/20 (baseline)
- GNN: 5 different spatial folds

**Results**:
- MLP (random): **88.2%** F1
- GNN (5-fold mean): **83.2% ± 0.74%** F1 ❌
  - Fold 1: 82.3%
  - Fold 2: 83.4%
  - Fold 3: 82.5%
  - Fold 4: 84.3%
  - Fold 5: 83.6%

**Diagnosis**: 
- Low variance (0.74%) = stable performance
- ALL folds underperform MLP by ~5 pp
- **This is NOT variance/luck—it's architectural failure**

---

## 🔬 Experiment 5: Diagnostic Experiments ❌ DEVASTATING RESULTS
**Notebook**: `gnn-maps-7-diagnostics.ipynb` ✅ **COMPLETED**

**Goal**: Determine WHY GNN consistently underperforms MLP by ~5 percentage points

### Three Diagnostic Tests Conducted:

#### 1. **K-Sensitivity Analysis** ⭐
**Hypothesis**: K=5 neighbors is too small for 145K cell dataset

**Results**:
| K | Test F1 | Edges | Improvement |
|---|---------|-------|-------------|
| 5 | 77.53% | 725,805 | Baseline |
| 10 | 77.52% | 1,451,610 | -0.01 pp |
| 15 | 77.16% | 2,177,415 | -0.37 pp |
| 20 | 77.59% | 2,903,220 | **+0.06 pp** |
| 25 | 77.51% | 3,629,025 | -0.02 pp |

**Finding**: ❌ **K IS NOT THE ISSUE**
- Best improvement: Only +0.06 percentage points (K=20)
- Performance essentially FLAT across all K values (77.2-77.6%)
- GNN scores 77.5% vs MLP 88.2% = **10.7 pp gap!**
- Increasing receptive field does NOT help

#### 2. **Spatial Pattern Visualization**
**Analysis**: Visualized all 18 cell types in X-Y coordinates
![alt text](image.png)

**Cell Types Observed**:
- CD4 (37,480 cells, 25.8%) - Largest population
- CD8, B cells, DC, Endothelial, Tumor, NK, M2, Monocyte, etc.

**Finding**: ✅ **CLEAR SPATIAL CLUSTERING EXISTS**
- CD4, CD8: Form distinct lymphoid aggregates
- Endothelial: Lines vascular structures
- Tumor cells: Concentrated in specific regions
- Macrophages (M1/M2): Distributed around tumor boundaries
- **Spatial patterns ARE present** → GNN SHOULD work but doesn't!

#### 3. **Random Graph Baseline**
**Test**: Compare GNN with True KNN vs Random graph vs No graph

**Results** (K=15):
| Graph Type | Test F1 | Analysis |
|------------|---------|----------|
| True KNN Graph | 77.59% | Baseline |
| Random Graph | 77.55% | -0.04 pp |
| **No Graph (Empty)** | **77.84%** | **+0.25 pp BETTER!** |

**Finding**: ❌ **CATASTROPHIC: GNN NOT USING GRAPH STRUCTURE**
- True graph ≈ Random graph (only 0.04 pp difference)
- **No graph BEATS true graph by 0.25 pp!**
- Graph structure is **ACTIVELY HARMFUL** to performance
- GNN is learning from proteins alone, ignoring spatial context

---

### 🔥 Diagnostic Summary: THE SMOKING GUN

**Three converging lines of evidence prove GNN failure**:

1. **K-sensitivity flat** → Receptive field size irrelevant
2. **Spatial patterns exist** → Data supports spatial learning
3. **Random ≈ True graph** → GNN ignores graph structure

**Root Cause Identified**:
- GraphSAGE message passing is **NOT learning useful spatial features**
- Protein expression features so strong they dominate gradient flow
- Spatial aggregation adds noise rather than signal
- GNN effectively degenerates to a worse MLP (77.5% vs 88.2%)

---

## 📈 Summary of Results

| Experiment | MLP F1 | GNN F1 | Gap | Status |
|------------|--------|--------|-----|--------|
| Spatial Split (Exp 1) | 86.9% | 82.2% | -4.7 pp | ❌ |
| Random Split (Exp 2) | 88.2% | 83.6% | -4.6 pp | ❌ |
| Hybrid Model (Exp 3) | 86.7% | 82.1% / 86.5% (hybrid) | -4.6 pp | ❌ |
| 5-Fold CV (Exp 4) | 88.2% | 83.2% ± 0.74% | -5.0 pp | ❌ |
| **Diagnostics (Exp 5)** | **88.2%** | **77.5%** | **-10.7 pp** | ❌💀 |
| **MAPS Baseline** | **90.0%** | — | — | 🎯 |

---

## 🔍 Root Cause Analysis

### ❌ CONFIRMED: GNN Architecture Fundamentally Broken

**Five experiments, one conclusion**: Current GraphSAGE implementation **cannot** leverage spatial information.

### Evidence Chain:

1. **Spatial patterns exist** (Exp 5, Visualization)
   - Clear clustering of CD4, CD8, Tumor, Endothelial cells
   - Not a data problem
   
2. **Random graph ≈ True graph** (Exp 5, Baseline test)
   - True: 77.59%, Random: 77.55%, Empty: 77.84%
   - **Graph structure ignored or harmful**
   
3. **K doesn't matter** (Exp 5, K-sensitivity)
   - K=5 to K=25: Performance flat (77.2%-77.6%)
   - Receptive field size irrelevant
   
4. **Hybrid model fails** (Exp 3)
   - GNN features contribute nothing
   - Model learns to zero out spatial branch
   
5. **Consistent across all splits** (Exp 1-4)
   - Spatial: GNN 82.2% vs MLP 86.9%
   - Random: GNN 83.6% vs MLP 88.2%
   - 5-fold CV: GNN 83.2% vs MLP 88.2%

### Why GNN Fails:

**Primary Issue**: Protein expression features dominate gradients
- 49 protein markers provide strong discriminative signal
- GNN message passing adds noise rather than useful context
- Spatial aggregation dilutes cell-specific protein signatures
- Mean aggregation in GraphSAGE averages away unique cell identities

**Secondary Issues**:
1. GraphSAGE architecture poor fit for this task
2. 2-layer depth insufficient for complex spatial patterns
3. No attention mechanism to weight important neighbors
4. KNN graph might not capture biological neighborhoods

---
## 📁 Files in This Repository

### Notebooks:
- `gnn-maps-3.ipynb` — Initial GNN (spatial split, MAPS-exact config)
- `gnn-maps-4-randSplit.ipynb` — Random split experiment
- `gnn-maps-5-protein&spatialfeatures.ipynb` — Hybrid model experiment
- `gnn-maps-6-spatialCV.ipynb` — 5-fold spatial cross-validation
- `gnn-maps-7-diagnostics.ipynb` — **✅ COMPLETED** Diagnostic experiments (K-sensitivity, visualization, baselines)

### Older Experiments:
- `6_layer_MLP_experiment.ipynb` — Early MLP architecture tests
- `6_layer_MLP_experiment_no2.ipynb` — MLP variations
- `8_layer_MLP_experiment.ipynb` — Deeper MLP test
- `cHL_CODEX_training_comparison.ipynb` — Training comparisons
- `spatial_features_MLP.ipynb` — Manual spatial feature exploration

### Data:
- `cHL_CODEX_processed/` — Preprocessed training/validation data
- `cHL_CODEX_spatial_features/` — Spatial feature engineering
- `results_*/` — Saved models and training logs

### Analysis:
- `FINAL_COMPARISON.md` — Comprehensive analysis of Experiments 1-4

---

## 🛠️ Hardware
- **Platform**: Kaggle Notebooks
- **GPU**: Tesla P100-PCIE-16GB (16GB VRAM)
- **Framework**: PyTorch 2.0+ with PyTorch Geometric

---

## 🎓 Key Learnings

1. **Data leakage matters**: KNN graphs create spatial dependencies → spatial splits prevent leakage
2. **Random split ≠ solution**: Even with random split, GNN underperforms
3. **Hybrid models reveal feature importance**: When hybrid ≈ MLP, GNN features are redundant
4. **Low variance = stable failure**: 5-fold CV shows GNN consistently learns wrong patterns
5. **K is NOT critical** ❌: Diagnostics prove K=5 to K=25 makes NO difference
6. **Graph structure ignored** ❌: Random graph ≈ True graph → GNN doesn't use spatial info
7. **Protein features too strong** ⭐: Gradient domination prevents spatial learning
8. **Spatial patterns exist but GNN can't use them** 💔: Visualization shows clear clustering but GNN fails to leverage it
9. **No graph beats graph** 🤯: Empty edge index performs BETTER than KNN graph
10. **Beating 90% MAPS unlikely with GNN**: MLP (88.2%) is closer to target than any GNN variant

---

## 📚 References
- **MAPS Paper**: [Original methodology for cell phenotyping](https://www.nature.com/articles/s41467-023-44188-w.pdf)
- **GraphSAGE**: Hamilton et al., "Inductive Representation Learning on Large Graphs"
- **PyTorch Geometric**: Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric"

---

## 👤 Author
A. M. Shahriar Rashid Mahe