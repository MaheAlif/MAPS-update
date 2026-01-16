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

## 🔬 Current Work: Diagnostic Experiments
**Notebook**: `gnn-maps-7-diagnostics.ipynb` (IN PROGRESS)

### Three Critical Tests:

#### 1. **K-Sensitivity Analysis** ⭐ MOST IMPORTANT
- Test K = 5, 10, 15, 20, 25 neighbors
- **Hypothesis**: K=5 is too small for 100K+ cell dataset
- **Expected**: If K is the bottleneck → larger K improves performance
- **Current Issue**: 
  - With K=5, 2-layer GNN → receptive field of only ~25 cells (0.025% of tissue)
  - Biological patterns likely span 10-20+ cells

#### 2. **Spatial Pattern Visualization**
- Plot each cell type in X-Y space
- **Check**: Are cells clustered or randomly distributed?
- **If clustered** → spatial patterns exist → GNN SHOULD work
- **If random** → no patterns → GNN will never beat MLP

#### 3. **Random Graph Baseline**
- Train GNN with: True KNN graph vs Random graph vs No graph
- **Expected**: True > Random > No edges
- **If True ≈ Random** → GNN is NOT using graph structure!

---

## 📈 Summary of Results

| Experiment | MLP F1 | GNN F1 | Gap | Status |
|------------|--------|--------|-----|--------|
| Spatial Split (Exp 1) | 86.9% | 82.2% | -4.7 pp | ❌ |
| Random Split (Exp 2) | 88.2% | 83.6% | -4.6 pp | ❌ |
| Hybrid Model (Exp 3) | 86.7% | 82.1% / 86.5% (hybrid) | -4.6 pp | ❌ |
| 5-Fold CV (Exp 4) | 88.2% | 83.2% ± 0.74% | -5.0 pp | ❌ |
| **MAPS Baseline** | **90.0%** | — | — | 🎯 |

---

## 🔍 Root Cause Analysis

### Primary Suspect: **K=5 is TOO SMALL** ⭐⭐⭐
- **Evidence**:
  - Random split (easy case) → GNN still fails
  - Hybrid model → GNN features ignored
  - 5-fold CV → consistently bad (low variance but poor performance)
- **Math**: 
  - 2 GNN layers × K=5 neighbors → receptive field ≈ 25 cells
  - Dataset: 100K+ cells → GNN sees 0.025% of tissue
- **Solution**: Increase K to 15-20

### Secondary Issues:
1. **GNN too shallow** (only 2 layers)
2. **GraphSAGE architecture** (mean aggregation might lose info)
3. **KNN graph construction** (Euclidean distance might not capture biology)
4. **Spatial patterns might be weak** (needs investigation)

---

## 🚀 Next Steps

### Immediate (Diagnostics):
1. Run `gnn-maps-7-diagnostics.ipynb` on Kaggle P100
2. Confirm K=5 is the bottleneck
3. Visualize spatial patterns (sanity check)
4. Verify GNN uses graph structure

### If K Helps (K > 15 improves performance):
1. Optimize K and GNN depth (3-4 layers)
2. Try attention mechanisms (GAT)
3. Experiment with different aggregations
4. **Goal**: Beat MLP (88.2%) → approach MAPS (90%)

### If K Doesn't Help:
1. **Manual spatial features** (recommended):
   - Compute neighbor composition, local density, distance metrics
   - Add to MLP (simpler than GNN)
2. **Abandon spatial context**:
   - Focus on MLP optimization (88.2% → 90%)
   - Hyperparameter tuning, ensembles
3. **Investigate data**:
   - Compute Moran's I (spatial autocorrelation)
   - If no patterns exist → spatial context won't help

---

## 📁 Files in This Repository

### Notebooks:
- `gnn-maps-3.ipynb` — Initial GNN (spatial split, MAPS-exact config)
- `gnn-maps-4-randSplit.ipynb` — Random split experiment
- `gnn-maps-5-protein&spatialfeatures.ipynb` — Hybrid model experiment
- `gnn-maps-6-spatialCV.ipynb` — 5-fold spatial cross-validation
- `gnn-maps-7-diagnostics.ipynb` — Diagnostic experiments (K-sensitivity, visualization, baselines)

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
5. **K is critical at scale**: With 100K+ cells, K=5 might be too restrictive

---

## 📚 References
- **MAPS Paper**: [Original methodology for cell phenotyping](https://github.com/SchapiroLabor/MAPS)
- **GraphSAGE**: Hamilton et al., "Inductive Representation Learning on Large Graphs"
- **PyTorch Geometric**: Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric"

---

## 👤 Author
A. M. Shahriar Rashid Mahe
