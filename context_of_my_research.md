### **Project Context: Spatial Proteomics Classification with GNNs**

**1. The High-Level Goal**
I am replicating and extending the methodology from the **MAPS paper** (Multi-Angular Projection of Spatial data). My goal is to prove that incorporating **spatial neighborhood context** via Graph Neural Networks (GNNs) significantly outperforms baseline MLPs (which only view cells in isolation) for cell-type classification.

**2. The Dataset**

* **Source:** **cHL CODEX** dataset (classical Hodgkin Lymphoma).
* **Format:** A single merged CSV file containing 100,000+ single cells from multiple tissue regions (Core/FOV).
* **Input Features:** 18–50 protein marker expressions per cell (e.g., CD3, CD20, PAX5, Ki67) + Spatial Coordinates (X, Y).
* **Target Label:** Cell Type (approx. 16 classes, including Tumor/RS cells, T-cells, B-cells, Macrophages).

**3. The Hypothesis**

* **Baseline (MLP):** A cell's identity is determined solely by its internal protein markers.
* **My Approach (GNN):** A cell's identity is defined by *both* its internal markers and its **spatial neighborhood** (e.g., a B-cell is likely near other immune cells). Therefore, a GNN (specifically GraphSAGE) should achieve a higher F1-score than an MLP.

**4. Implementation Strategy (PyTorch Geometric)**

* **Graph Construction:** Nodes = Cells. Edges = K-Nearest Neighbors () based on Euclidean distance of X/Y coordinates.
* **Model:** **GraphSAGE** (2 layers). Chosen for its inductive capability and scalability.
* **Validation Strategy:** **Spatial Split**. I am splitting the dataset by `Region_ID` (or Image ID), *not* by random shuffling. This ensures the model is tested on entirely unseen tissue slides to prevent spatial data leakage.

**5. Technical Constraints**

* **Hardware:** Local PC with **GTX 1650 Ti (4GB VRAM)**.
* **Constraint:** I cannot load the full graph into GPU memory for training.
* **Solution:** I am using `NeighborLoader` for mini-batch training to keep memory usage low.

---

### **Current Status**

* I have successfully loaded the dataset into a Pandas DataFrame on Kaggle.
* I have identified the key columns (`X`, `Y`, `CellType`, and Marker columns).
* I am currently implementing the `torch_geometric` pipeline to build the graph and run the training loop.