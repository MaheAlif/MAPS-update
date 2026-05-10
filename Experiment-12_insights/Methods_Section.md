# Methods Section (For Thesis/Paper)

*This standalone methods section can be directly inserted into your thesis or manuscript*

---

## Materials and Methods

### Dataset and Preprocessing

We utilized classical Hodgkin Lymphoma (cHL) CODEX spatial proteomics data comprising 145,161 single cells across 18 cell types. Each cell was characterized by 49 protein markers measured via multiplexed fluorescence imaging. The dataset included the following cell types: B cells (n=16,196), CD4 T cells (n=37,480), CD8 T cells (n=17,184), Cytotoxic CD8 T cells (n=384), Dendritic cells (n=9,637), Endothelial cells (n=8,705), Epithelial cells (n=2,251), Lymphatic cells (n=3,768), M1 Macrophages (n=3,101), M2 Macrophages (n=7,286), Mast cells (n=3,324), Monocytes (n=6,913), NK cells (n=7,339), Neutrophils (n=3,442), Other (n=5,108), Segmentation Artifacts (n=1,431), Regulatory T cells (n=3,352), and Tumor cells (n=8,260).

The 49 protein markers (alphabetically ordered) were: BCL-2, CCR6, CD11b, CD11c, CD15, CD16, CD162, CD163, CD2, CD20, CD206, CD25, CD30, CD31, CD4, CD44, CD45, CD45RA, CD45RO, CD5, CD56, CD57, CD68, CD69, CD7, CD8, Collagen-4, Cytokeratin, DAPI, EGFR, FoxP3, Granzyme-B, HLA-DR, IDO-1, LAG-3, MCT, MMP-9, MUC-1, PD-1, PD-L1, Podoplanin, T-bet, TCRγδ, TCRβ, Tim-3, VISA, Vimentin, α-SMA, and β-Catenin.

Data were randomly partitioned into training (80%, n=116,128) and validation (20%, n=28,670) sets using a fixed random seed (seed=42) for reproducibility. All protein marker intensities underwent z-score standardization using scikit-learn's StandardScaler, fit exclusively on the training set and subsequently applied to the validation set to prevent information leakage.

### Base Classification Models

We trained three diverse base classifiers on the protein expression features:

**Multi-Layer Perceptron (MLP).** A feedforward neural network was implemented following the MAPS framework architecture, consisting of four hidden layers with 512 units each, ReLU activation functions, and 0.1 dropout after each layer. The output layer employed softmax activation for 18-class probability estimation. Training utilized the Adam optimizer (learning rate=0.001, β₁=0.9, β₂=0.999) with cross-entropy loss and mini-batch size of 256. To address class imbalance, we employed WeightedRandomSampler during training, weighting each sample inversely proportional to its class frequency. Early stopping with patience of 50 epochs was applied based on validation loss, with a minimum of 250 training epochs. Total trainable parameters: approximately 1.3 million.

**LightGBM.** A gradient boosting decision tree model was trained using Microsoft's LightGBM implementation with the following hyperparameters: gradient boosting decision tree (gbdt) boosting type, multiclass classification objective with multi-logloss metric, 127 leaf nodes per tree, maximum tree depth of 8, learning rate of 0.05, feature fraction (column sampling) of 0.8, bagging fraction (row sampling) of 0.8 with frequency of 5, and class imbalance handling enabled (is_unbalance=True). Training continued until validation loss ceased improving for 50 consecutive iterations, resulting in optimal performance at iteration 212.

**XGBoost.** An XGBoost gradient boosting classifier was trained with multi:softprob objective for multi-class probability prediction. Configuration included: histogram-based tree construction method (hist) with GPU acceleration (CUDA), maximum tree depth of 8, learning rate of 0.05, row subsample ratio of 0.8, column subsample ratio of 0.8, minimum child weight of 5, L1 regularization (alpha) of 0.1, and L2 regularization (lambda) of 1.0. Sample weights inversely proportional to class frequencies were applied to address imbalance. The model trained until validation loss stabilized, achieving optimal performance at iteration 754.

All three models were trained independently on identical training/validation splits using only protein marker features (spatial coordinates excluded).

### Ensemble Strategies

We evaluated five ensemble aggregation methods operating on predicted class probability distributions (soft voting):

**Simple Average Ensemble.** Equal-weight averaging of probability vectors:
```
P_ensemble(c) = [P_MLP(c) + P_LGB(c) + P_XGB(c)] / 3
```
where P_model(c) represents the predicted probability for class c.

**Weighted Average Ensemble.** Probability averaging with weights proportional to each model's validation weighted F1 score:
```
w_model = F1_weighted(model)
P_ensemble(c) = [w_MLP × P_MLP(c) + w_LGB × P_LGB(c) + w_XGB × P_XGB(c)] / Σw
```

**Best-2 Combination.** Averaging of the two highest-performing individual models (LightGBM and XGBoost) with equal weights (0.5, 0.5), excluding the MLP.

**Majority Vote.** Hard voting where each model casts a single vote for its most probable class. The final prediction corresponds to the class receiving the plurality of votes, with ties resolved by selecting the class with highest mean probability among tied classes.

**Grid Search Optimal Weights.** Exhaustive search over the discrete weight space [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] for each model under the constraint Σw = 1.0. The weight combination maximizing validation weighted F1 was selected: (w_MLP=0.4, w_LGB=0.4, w_XGB=0.2).

### Performance Evaluation

Classification performance was assessed exclusively on the held-out validation set. The primary evaluation metric was weighted F1 score, defined as:
```
F1_weighted = Σ[n_c / N × F1_c]
```
where n_c is the number of validation samples in class c, N is the total validation size, and F1_c is the F1 score for class c. This metric accounts for class imbalance by weighting each class's contribution proportionally to its representation.

Secondary metrics included macro F1 (unweighted average of per-class F1 scores) and overall accuracy. Per-class F1 scores were computed as:
```
F1_c = 2 × (Precision_c × Recall_c) / (Precision_c + Recall_c)
```

Statistical significance of performance differences was evaluated using McNemar's test for paired binary outcomes and bootstrap confidence intervals (1000 samples with replacement).

### Model Agreement Analysis

To characterize ensemble behavior, we quantified pairwise and three-way agreement patterns among base models. Agreement was defined as identical predicted class labels (argmax of probability distributions). For each validation sample, we recorded: (1) three-way agreement (all models predict same class), (2) pairwise agreement between each model pair, and (3) complete disagreement (all three models predict different classes). We further stratified validation accuracy by agreement pattern to assess confidence calibration.

### Computational Environment

All experiments were conducted using Python 3.8 with PyTorch 1.12 (MLP), LightGBM 3.3 (gradient boosting), XGBoost 1.7 (gradient boosting), scikit-learn 1.1 (preprocessing and metrics), NumPy 1.23, and pandas 1.4. Training utilized an NVIDIA GPU (CUDA 11.6) for neural network and XGBoost acceleration, while LightGBM ran on CPU. Reported training times reflect single-GPU execution on an NVIDIA Tesla V100.

### Software and Code Availability

The MAPS framework is publicly available at https://github.com/mahmoodlab/MAPS. LightGBM and XGBoost are open-source libraries available via pip. All code for ensemble construction and evaluation, along with hyperparameter configurations, is available upon request.

---

## Statistical Analysis

Performance comparisons between individual models and ensemble methods were assessed using paired statistical tests. McNemar's test evaluated whether the ensemble made significantly different classification errors compared to individual models. Bootstrap confidence intervals (95% CI) were constructed by resampling the validation set 1,000 times with replacement, computing weighted F1 for each bootstrap sample. Wilcoxon signed-rank tests compared per-class F1 distributions across models. All hypothesis tests used a significance threshold of α = 0.05.

---

## Notes for Adaptation

**For thesis:** This section can be inserted directly into your Methods chapter. Add subheadings if required by your format.

**For journal paper:** Condense to fit journal requirements:
- Merge "Dataset and Preprocessing" into single paragraph
- Combine model descriptions (currently ~200 words each) into table
- Move hyperparameter details to Supplementary Methods
- Keep ensemble strategies and evaluation sections as-is

**Key elements to retain:**
1. Sample sizes and split ratios (reproducibility)
2. Model architectures (comparability)
3. Evaluation metrics definitions (interpretability)
4. Statistical testing (rigor)

**Optional additions:**
- Add citations: MAPS (Greenwald et al. 2023), LightGBM (Ke et al. 2017), XGBoost (Chen & Guestrin 2016)
- Include IRB statement if human samples
- Add data availability statement
