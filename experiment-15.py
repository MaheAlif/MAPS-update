<VSCode.Cell language="markdown">
# Experiment 16: Spatial Post-Smoothing 🧬
## Fusion of Tree-Based Tabular Superiority & Graph Attention Networks

### 🎯 The Dilemma
Through our previous 5-Fold validation experiments, we proved two things:
1. **Tree-based models (LightGBM/XGBoost)** dominate deep neural networks (MLPs) on the tabular protein expression data, easily achieving 90.27% accuracy.
2. **Graph Neural Networks (GraphSAGE)** applied directly to raw proteins with $K=5$ fail miserably (83.2%), as they struggle to learn the tabular features and have a very small reception field.

### 💡 The "Spatial Post-Smoothing" Concept
Instead of forcing a GNN to figure out complex protein marker patterns, we **offload the tabular interpretation entirely to LightGBM.** 

**The Pipeline:**
1. **Phase 1: Tabular Base (LightGBM)** - Train LightGBM strictly on the `N` protein markers to predict cell classes. Extract the raw multi-class probability outputs (an 18-dimensional probability vector per cell).
2. **Phase 2: Graph Construction (Macro $K=20$)** - Construct a much larger regional spatial graph using the `X_cent` and `Y_cent` coordinates.
3. **Phase 3: Spatial Smoothing (GATv2)** - We feed **ONLY the predicted node probabilities** into a 1 or 2-layer Graph Attention Network. 

If LightGBM correctly tags a cell but is only 55% confident, the GATv2 will look at the 20 surrounding neighbor classifications and "smooths" or boosts the confidence. If a cell is misclassified as a weird artifact, the GATv2 will realize it is completely surrounded by B-cells and automatically overrides the Tree model's base prediction!
</VSCode.Cell>

<VSCode.Cell language="python">
# ==========================================
# 0. Kaggle Environment Setup & Imports
# ==========================================
# Install PyTorch Geometric (uncomment on Kaggle if needed)
# !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, classification_report, accuracy_score

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

import warnings
warnings.filterwarnings('ignore')

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 📥 Step 1: Load and Prepare the CODEX Data
*Note: Update the file paths below to point to the Kaggle dataset paths once uploaded.*
</VSCode.Cell>

<VSCode.Cell language="python">
# ==========================================
# 1. Load Data
# ==========================================
# Assuming files are in Kaggle input directory. Modify to your actual Kaggle dataset paths:
TRAIN_PATH = '/kaggle/input/chl-codex-spatial-data/train.csv'
VALID_PATH = '/kaggle/input/chl-codex-spatial-data/valid.csv'

try:
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
except FileNotFoundError:
    print("Files not found. Please verify the Kaggle dataset paths.")
    # Dummy data placeholder for demonstration if paths are missing
    train_df, valid_df = None, None

# Feature extraction mappings (replace with your exact dataset column names)
# Example: All columns except cell ID, X, Y, Region, and Label
target_col = 'cell_label'
spatial_cols = ['X_cent', 'Y_cent']
ignore_cols = ['cell_id', 'Region_ID'] + spatial_cols + [target_col]

def prepare_xy(df):
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    X_proteins = df[feature_cols].values
    X_spatial = df[spatial_cols].values
    Y_labels = df[target_col].values
    return X_proteins, X_spatial, Y_labels

if train_df is not None:
    X_prot_train, X_spat_train, y_train = prepare_xy(train_df)
    X_prot_valid, X_spat_valid, y_valid = prepare_xy(valid_df)
    
    num_classes = len(np.unique(y_train))
    print(f"Train Shape: {X_prot_train.shape}, Classes: {num_classes}")
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 🌳 Step 2: Tabular Base Predictions (LightGBM)
We will train a LightGBM model to generate our base cell-type predictions purely using the protein expression features.
</VSCode.Cell>

<VSCode.Cell language="python">
# ==========================================
# 2. Phase 1: Train Base Tree Model Models
# ==========================================
print("Training Base LightGBM Model on Protein Features...")

# LightGBM hyperparams similar to your successful Experiment 12
lgb_params = {
    'objective': 'multiclass',
    'num_class': num_classes if train_df is not None else 18,
    'metric': 'multi_error',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'n_estimators': 300,
    'random_state': 42,
    'verbose': -1
}

if train_df is not None:
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_prot_train, y_train)

    # Calculate probabilities (N, num_classes)
    train_probs = model_lgb.predict_proba(X_prot_train)
    valid_probs = model_lgb.predict_proba(X_prot_valid)

    # Initial Baseline evaluation
    valid_preds = np.argmax(valid_probs, axis=1)
    base_f1 = f1_score(y_valid, valid_preds, average='macro')
    print(f"LightGBM Base Validation Macro F1: {base_f1:.4f}")
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 🕸️ Step 3: Construct the Spatial Graph (Macro $K=20$)
Instead of $K=5$, we are using $K=20$. This guarantees the GAT will see a wide macro-regional view of the cell's environment.
</VSCode.Cell>

<VSCode.Cell language="python">
# ==========================================
# 3. Phase 2: Macro Spatial Graph Const.
# ==========================================
K_NEIGHBORS = 20

def construct_graph(probs, coords, labels, k=K_NEIGHBORS):
    # Construct KNN using spatial coords
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1)
    nbrs.fit(coords)
    # Get indices, dropping the first one (which is the cell itself)
    distances, indices = nbrs.kneighbors(coords)
    
    # Build Edges (edge_index)
    source_nodes = np.repeat(np.arange(len(coords)), k)
    target_nodes = indices[:, 1:].flatten()
    edge_index = np.vstack((source_nodes, target_nodes))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Features are purely the model probabilities from LightGBM
    x = torch.tensor(probs, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y)

if train_df is not None:
    print(f"Building KNN Graphs with K={K_NEIGHBORS}...")
    train_graph = construct_graph(train_probs, X_spat_train, y_train)
    valid_graph = construct_graph(valid_probs, X_spat_valid, y_valid)
    
    # Move to GPU
    train_graph = train_graph.to(device)
    valid_graph = valid_graph.to(device)
    print("Graph construction complete!")
    print(f"Train Graph: {train_graph}")
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 🧠 Step 4: Spatial Post-Smoothing with GATv2
GATv2 will natively ignore noisy neighbors (via low attention weighting) and heavily weigh informative neighbors to boost classification. Because the input features are just probability distributions (size 16 to 18), the GNN is extremely lightweight and fast.
</VSCode.Cell>

<VSCode.Cell language="python">
# ==========================================
# 4. Phase 3: Spatial Smoothing Engine
# ==========================================
class SpatialSmoother(torch.nn.Module):
    def __init__(self, num_classes, hidden_dim=64):
        super(SpatialSmoother, self).__init__()
        # Input dim is num_classes (from the LightGBM probability vectors)
        # Using 4 heads to capture different biological neighborhood dynamics
        self.gat1 = GATv2Conv(num_classes, hidden_dim, heads=4, concat=True, dropout=0.2)
        # Output dim is num_classes
        self.gat2 = GATv2Conv(hidden_dim * 4, num_classes, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        # We start with the base probabilities
        # We add the GNN embeddings to the base probabilities (residual-like post-smoothing)
        
        identity = x # skip-connection: preserve LightGBM decision

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat2(x, edge_index)
        
        # Combine base tree prediction with GNN spatial insight
        out = x + identity 
        return out

if train_df is not None:
    model_gnn = SpatialSmoother(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model_gnn.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model_gnn.train()
        optimizer.zero_grad()
        out = model_gnn(train_graph.x, train_graph.edge_index)
        loss = criterion(out, train_graph.y)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(graph):
        model_gnn.eval()
        out = model_gnn(graph.x, graph.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred == graph.y).sum().item()
        acc = correct / len(graph.y)
        f1 = f1_score(graph.y.cpu(), pred.cpu(), average='macro')
        return acc, f1, pred.cpu()

    # Training Loop
    EPOCHS = 100
    print("Training GNN Spatial Smoother...")
    best_val_f1 = 0
    final_preds = None

    for epoch in range(1, EPOCHS + 1):
        loss = train()
        if epoch % 10 == 0 or epoch == 1:
            train_acc, train_f1, _ = test(train_graph)
            val_acc, val_f1, val_preds = test(valid_graph)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                final_preds = val_preds
                
            print(f"Epoch {epoch:03d}: Loss: {loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

    print("=" * 50)
    print(f"🌲 BASE LIGHTGBM VALIDATION F1     : {base_f1:.4f}")
    print(f"🌐 GATv2 SMOOTHED VALIDATION F1    : {best_val_f1:.4f}")
    print("=" * 50)
</VSCode.Cell>

<VSCode.Cell language="markdown">
### 📊 Step 5: Final Evaluation & Diagnostics
Check the difference matrix to see exactly how many cells the Spatial Post-Smoother corrected!
</VSCode.Cell>

<VSCode.Cell language="python">
# ==========================================
# 5. Diagnostic Reporting
# ==========================================
if train_df is not None:
    lgb_preds = np.argmax(valid_probs, axis=1)
    
    # Calculate how many predictions were altered
    changed_preds = (final_preds.numpy() != lgb_preds).sum()
    total_preds = len(lgb_preds)
    
    print(f"\nTotal predictions altered by GNN: {changed_preds} / {total_preds} ({(changed_preds/total_preds)*100:.2f}%)")
    
    print("\n--- Final Performance with Spatial Context ---")
    print(classification_report(y_valid, final_preds))
</VSCode.Cell>