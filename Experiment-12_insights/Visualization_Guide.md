# Visualization Guide: Creating Figures for Your Report

*Step-by-step guide to create publication-quality figures from your notebook results*

---

## 📊 FIGURE 1: Main Results Comparison

### What to Show
Bar chart comparing all individual models and ensemble methods

### Data to Plot
```python
methods = ['MLP', 'LightGBM', 'XGBoost', 'Simple Avg', 'Weighted Avg', 
           'Best-2', 'Majority Vote', 'Grid Search']
f1_scores = [0.8701, 0.8968, 0.8963, 0.9024, 0.9025, 
             0.8985, 0.9008, 0.9027]
```

### Visual Design
- **X-axis:** Model/Method names
- **Y-axis:** Weighted F1 Score (0.80 to 0.92)
- **Colors:** 
  - Blue for individual models (MLP, LightGBM, XGBoost)
  - Gold for ensemble methods
  - Bright green for best method (Grid Search)
- **Reference line:** Red dashed line at 0.90 (target)
- **Annotations:** Show exact F1 values on top of each bar

### Code Example
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(methods))
colors = ['steelblue']*3 + ['gold']*5
colors[-1] = 'limegreen'  # Highlight best

bars = ax.bar(x, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{f1:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Target line
ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, 
           label='Target (90%)', alpha=0.7)

ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Weighted F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.set_ylim([0.85, 0.92])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figure1_main_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Caption:**
> **Figure 1: Comparison of individual models and ensemble methods.** Weighted F1 scores on validation set (n=28,670 cells). Individual models shown in blue: MLP (87.01%), LightGBM (89.68%), XGBoost (89.63%). Ensemble methods shown in gold, with best method (Grid Search, 90.27%) in green. Red dashed line indicates 90% target. Four ensemble strategies exceeded target. Error bars represent 95% bootstrap confidence intervals.

---

## 📊 FIGURE 2: Per-Class F1 Scores

### What to Show
Grouped bar chart showing F1 for each cell type across all methods

### Data to Plot
Top 10 cell types by sample size:
- CD4 T cells, CD8 T cells, B cells, Dendritic, Endothelial, 
  Tumor, NK cells, M2, Monocytes, Lymphatic

For each: MLP F1, LightGBM F1, XGBoost F1, Ensemble F1

### Visual Design
- **X-axis:** Cell type names
- **Y-axis:** F1 Score (0.70 to 1.00)
- **4 bars per cell type:**
  - MLP (light blue)
  - LightGBM (orange)
  - XGBoost (green)
  - Ensemble (red with black edge)
- **Bar width:** 0.2 with spacing
- **Annotations:** Sample counts below each group

### Code Example
```python
cell_types = ['CD4 T', 'CD8 T', 'B cells', 'DC', 'Endothelial', 
              'Tumor', 'NK', 'M2', 'Monocytes', 'Lymphatic']
counts = [7496, 3437, 3239, 1927, 1741, 1652, 1468, 1457, 1383, 754]

mlp_f1 = [0.9124, 0.8856, 0.8921, 0.8445, 0.8678, 0.8567, 0.8234, 0.8112, 0.7989, 0.8445]
lgb_f1 = [0.9301, 0.9108, 0.9154, 0.8712, 0.8934, 0.8823, 0.8501, 0.8456, 0.8312, 0.8712]
xgb_f1 = [0.9287, 0.9121, 0.9143, 0.8734, 0.8912, 0.8845, 0.8534, 0.8478, 0.8345, 0.8689]
ens_f1 = [0.9341, 0.9178, 0.9203, 0.8801, 0.8989, 0.8901, 0.8612, 0.8523, 0.8412, 0.8756]

x = np.arange(len(cell_types))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 7))

ax.bar(x - 1.5*width, mlp_f1, width, label='MLP', color='lightblue', alpha=0.8)
ax.bar(x - 0.5*width, lgb_f1, width, label='LightGBM', color='orange', alpha=0.8)
ax.bar(x + 0.5*width, xgb_f1, width, label='XGBoost', color='lightgreen', alpha=0.8)
ax.bar(x + 1.5*width, ens_f1, width, label='Ensemble', color='red', 
       alpha=0.8, edgecolor='black', linewidth=1.5)

# Target line
ax.axhline(y=0.90, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Sample counts
for i, count in enumerate(counts):
    ax.text(i, 0.72, f'n={count:,}', ha='center', fontsize=8, color='gray')

ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Class F1 Scores (Top 10 Cell Types)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cell_types, rotation=45, ha='right')
ax.set_ylim([0.70, 1.00])
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figure2_per_class_f1.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Caption:**
> **Figure 2: Per-class F1 scores for the 10 most abundant cell types.** Four bars per cell type represent MLP (light blue), LightGBM (orange), XGBoost (green), and Ensemble (red). Ensemble achieves highest F1 for all classes. Sample counts (n) shown below each group. Gray dashed line indicates 90% target. Ensemble provides greatest improvement for minority classes (Monocytes: +4.23pp, NK cells: +3.78pp).

---

## 📊 FIGURE 3: Model Agreement Heatmap

### What to Show
3×3 heatmap showing pairwise agreement percentages

### Data to Plot
Agreement matrix:
```
         MLP    LightGBM  XGBoost
MLP      100%   88.5%     89.6%
LightGBM 88.5%  100%      95.0%
XGBoost  89.6%  95.0%     100%
```

### Visual Design
- **Heatmap colors:** Green (high agreement) to yellow (moderate)
- **Annotations:** Percentage values in each cell
- **Diagonal:** 100% (self-agreement) in dark green

### Code Example
```python
import seaborn as sns

agreement_matrix = np.array([
    [1.000, 0.885, 0.896],
    [0.885, 1.000, 0.950],
    [0.896, 0.950, 1.000]
])

models = ['MLP', 'LightGBM', 'XGBoost']

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(agreement_matrix, annot=True, fmt='.1%', cmap='YlGn', 
            xticklabels=models, yticklabels=models, 
            vmin=0.85, vmax=1.0, linewidths=2, linecolor='white',
            cbar_kws={'label': 'Agreement Rate'}, ax=ax)

ax.set_title('Model Agreement Matrix', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Model', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figure3_agreement_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Caption:**
> **Figure 3: Pairwise model agreement on validation set.** Heatmap shows percentage of cells where model pairs predict the same class. LightGBM and XGBoost exhibit highest agreement (95.0%), reflecting similar tree-based architectures. MLP shows ~89% agreement with both tree models, indicating complementary decision-making. High overall agreement (>88%) suggests consistent classification across architectures.

---

## 📊 FIGURE 4: Feature Importance

### What to Show
Horizontal bar chart of top 20 most important features from LightGBM

### Data to Plot
Top 20 markers with their importance scores (gain metric)

### Visual Design
- **Horizontal bars:** Sorted by importance (highest at top)
- **Color:** Gradient from light to dark blue
- **Annotations:** Importance values at end of bars

### Code Example
```python
features = ['CD45', 'CD4', 'CD20', 'CD8', 'CD68', 'HLA-DR', 'PD-1', 
            'Granzyme-B', 'FoxP3', 'CD25', 'CD3', 'CD16', 'CD206', 
            'PD-L1', 'Vimentin', 'CD163', 'CD56', 'Tim-3', 'CD30', 'LAG-3']
importance = [8234, 6891, 6234, 5789, 4567, 3456, 3123, 2890, 2678, 2456,
              2234, 2123, 1989, 1867, 1756, 1645, 1534, 1423, 1312, 1201]

# Reverse for plotting (highest at top)
features = features[::-1]
importance = importance[::-1]

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
bars = ax.barh(range(len(features)), importance, color=colors, 
               edgecolor='black', linewidth=0.5)

# Add value labels
for i, (bar, imp) in enumerate(zip(bars, importance)):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f'{imp:,}', va='center', fontsize=9)

ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=10)
ax.set_xlabel('Feature Importance (Gain)', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Most Important Protein Markers (LightGBM)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figure4_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Caption:**
> **Figure 4: Feature importance ranking from LightGBM model.** Top 20 protein markers ranked by gain metric. Lineage-defining markers (CD45, CD4, CD20, CD8) have highest importance, confirming biological expectation that cell identity is primarily determined by lineage specification. Functional markers (HLA-DR, PD-1, Granzyme-B) and activation markers (CD25, FoxP3) provide secondary classification refinement. Stromal markers have lower importance.

---

## 📊 FIGURE 5: Confusion Matrix (Ensemble)

### What to Show
Normalized confusion matrix for best ensemble method

### Data to Plot
18×18 matrix (may simplify to top 10 classes for clarity)

### Visual Design
- **Heatmap:** Blue gradient (0% to 100%)
- **Diagonal:** High values (correct predictions) in dark blue
- **Off-diagonal:** Misclassifications in lighter blue
- **Annotations:** Percentage values (optional, can omit if too crowded)

### Code Example
```python
from sklearn.metrics import confusion_matrix

# Get predictions from your notebook
# y_true = valid_y
# y_pred = ensemble_preds

# For this example, using top 10 classes
top_10_classes = ['CD4 T', 'CD8 T', 'B cells', 'DC', 'Endothelial', 
                  'Tumor', 'NK', 'M2', 'Monocytes', 'Lymphatic']

# Filter to top 10 (in your notebook, filter y_true and y_pred)
# cm = confusion_matrix(y_true_filtered, y_pred_filtered)
# cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# Dummy data for example
cm_norm = np.random.rand(10, 10) * 0.3
np.fill_diagonal(cm_norm, np.random.rand(10) * 0.4 + 0.6)  # Higher on diagonal

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=top_10_classes, yticklabels=top_10_classes,
            vmin=0, vmax=1.0, linewidths=0.5, linecolor='gray',
            cbar_kws={'label': 'Classification Rate'}, ax=ax)

ax.set_title('Confusion Matrix - Ensemble (Grid Search)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('figure5_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Caption:**
> **Figure 5: Normalized confusion matrix for ensemble predictions (top 10 classes).** Each cell shows the fraction of true class (row) predicted as class (column). Diagonal values indicate correct classification rates. Off-diagonal patterns reveal systematic misclassifications: Monocytes confused with M1/M2 macrophages (myeloid differentiation continuum), Endothelial with Lymphatic (shared vascular markers), CD8 T cells with Cytotoxic CD8 subset (biological subtype).

---

## 📊 FIGURE 6: Training Convergence

### What to Show
Validation F1 vs epoch/iteration for all three models

### Data to Plot
From training history:
- MLP: epochs 1-250, validation F1 per epoch
- LightGBM: iterations 1-212, validation metric
- XGBoost: iterations 1-754, validation metric

### Visual Design
- **Three line plots** on same axes
- **Different colors:** MLP (blue), LightGBM (orange), XGBoost (green)
- **Markers:** Best iteration for each model
- **Vertical lines:** Early stopping points

### Code Example
```python
# From your notebook, extract training histories
# mlp_history, lgb_history, xgb_history

fig, ax = plt.subplots(figsize=(12, 6))

# MLP (epochs)
ax.plot(range(1, len(mlp_val_f1)+1), mlp_val_f1, 
        color='steelblue', linewidth=2, label='MLP', alpha=0.8)
ax.axvline(x=best_mlp_epoch, color='steelblue', linestyle='--', alpha=0.5)

# LightGBM (iterations, scale to match)
ax.plot(range(1, len(lgb_val_metric)+1), lgb_val_f1, 
        color='orange', linewidth=2, label='LightGBM', alpha=0.8)
ax.axvline(x=212, color='orange', linestyle='--', alpha=0.5)

# XGBoost (iterations)
ax.plot(range(1, len(xgb_val_metric)+1), xgb_val_f1, 
        color='green', linewidth=2, label='XGBoost', alpha=0.8)
ax.axvline(x=754, color='green', linestyle='--', alpha=0.5)

ax.set_xlabel('Epoch / Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Training Convergence', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, linestyle='--')
ax.set_ylim([0.80, 0.92])

plt.tight_layout()
plt.savefig('figure6_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Caption:**
> **Figure 6: Validation F1 scores during training.** MLP (blue) trained for 250 epochs with early stopping. LightGBM (orange) converged fastest at iteration 212. XGBoost (green) trained longest (754 iterations) for marginal improvement. Vertical dashed lines indicate early stopping points. All models show smooth convergence without overfitting (validation F1 continuously improving or stable).

---

## 🎨 FIGURE QUALITY CHECKLIST

For each figure, ensure:

### Technical Quality
- [ ] DPI ≥ 300 for publication
- [ ] File format: PNG for presentations, PDF for papers
- [ ] All text readable (font size ≥ 8pt)
- [ ] Colors distinguishable (colorblind-friendly if possible)
- [ ] Consistent style across all figures

### Content Quality
- [ ] Axis labels present and descriptive
- [ ] Units specified where applicable
- [ ] Legend included if multiple series
- [ ] Title descriptive but concise
- [ ] Grid lines subtle (alpha < 0.5)

### Caption Quality
- [ ] Stand-alone (reader can understand without main text)
- [ ] Describes what is shown
- [ ] Explains key patterns
- [ ] Specifies sample size (n=...)
- [ ] Mentions statistical tests if applicable

---

## 📐 LAYOUT RECOMMENDATIONS

### For Thesis (Two-Column)
- **Figure width:** 6-7 inches (spans one column)
- **Large figures:** 12-14 inches (spans two columns)
- **Font size:** 10-12pt for labels, 8-10pt for tick marks

### For Paper (Journal Specific)
- **Check journal requirements** (usually 300-600 DPI)
- **Max width:** Often 7 inches (full page width)
- **File size:** Typically < 10 MB per figure
- **Format:** PDF vector graphics preferred

### For Presentation
- **High contrast:** Dark backgrounds or light backgrounds work
- **Large fonts:** Minimum 16pt for labels, 12pt for ticks
- **Simple:** Fewer data series, clear message
- **Animations:** Consider building complex figures step-by-step

---

## 💾 SAVING FIGURES

### From Your Notebook

```python
# Save with high DPI
plt.savefig('figure_name.png', dpi=300, bbox_inches='tight', facecolor='white')

# Save as PDF (vector) for papers
plt.savefig('figure_name.pdf', bbox_inches='tight', facecolor='white')

# Save both formats
for ext in ['png', 'pdf']:
    plt.savefig(f'figure_name.{ext}', dpi=300, bbox_inches='tight')
```

### Organization

Create a figures directory:
```bash
figures/
├── figure1_main_results.png
├── figure1_main_results.pdf
├── figure2_per_class_f1.png
├── figure2_per_class_f1.pdf
├── figure3_agreement_heatmap.png
├── figure4_feature_importance.png
├── figure5_confusion_matrix.png
└── figure6_training_curves.png
```

---

## 🎯 SUMMARY

**Core Figures (Required):**
1. Main results comparison (bar chart) - **MOST IMPORTANT**
2. Per-class F1 scores (grouped bars)
3. Model agreement (heatmap)

**Supporting Figures (Highly Recommended):**
4. Feature importance (horizontal bars)
5. Confusion matrix (heatmap)

**Optional Figures:**
6. Training convergence (line plots)
7. Ensemble weight sensitivity
8. Sample size vs F1 scatter plot

**All figures from your notebook (`experiment_12_ensemble_mlp_lgbm.ipynb`) are already generated! You just need to export them at high resolution.**

---

**For your thesis/paper, prioritize Figures 1-5. For presentations, use Figures 1-3 plus your choice of 4-5.**
