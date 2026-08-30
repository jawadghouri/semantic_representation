# Sementic Analysis of LLM Outputs

A comprehensive pipeline for analyzing LLM response embeddings using K-Means clustering and Principal Component Analysis (PCA). This project generates embedding vectors from text responses, clusters them in high-dimensional space, and produces interactive 2D and 3D visualizations with covariance ellipses/ellipsoids.

## Overview

This pipeline processes text responses from multiple large language models (LLMs) and constraint-based datasets, converts them into semantic embeddings, applies unsupervised K-Means clustering, and visualizes the results using PCA projection with statistical cluster boundaries.

### Key Features

- **Multi-Embedder Support**: Generate embeddings using MiniLM, BGE, and E5 models
- **Flexible Input Handling**: Support for single responses, multiple responses, and multi-model responses
- **Normalization Control**: Choose between normalized and unnormalized embeddings
- **Dual Clustering Approach**: K-Means clustering on original high-dimensional embeddings with PCA visualization
- **Advanced Visualizations**: 2D and 3D plots with covariance ellipses/ellipsoids for cluster regions
- **Covariance Estimation**: Statistical confidence regions using eigenvalue decomposition

## Project Structure

```
KMeanCluster/
├── run_embeddings.py          # Single embedder pipeline (MiniLM only)
├── run_embeddings2.py         # Multi-embedder pipeline (MiniLM, BGE, E5)
├── pca_kmean.py               # K-Means + PCA for unified dataset
├── pca_kmean2.py              # K-Means + PCA with grouped analysis
└── data/
    ├── raw_inputs/            # Input JSON files
    │   ├── online_prompts.json # Online dataset with multiple LLM responses
    │   └── constraint_prompts/
    │       ├── C1.json         # Constraint group 1
    │       ├── C2.json         # Constraint group 2
    │       └── C3.json         # Constraint group 3
    └── processed/
        ├── embeddings_norm/    # Normalized embedding vectors (.npy files)
        ├── embeddings_unnorm/  # Unnormalized embedding vectors (.npy files)
        └── plots/              # Generated visualizations
```

## Installation

### Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Scikit-learn
- Sentence Transformers (for embeddings)

### Setup

```bash
# Install dependencies
pip install numpy matplotlib scikit-learn sentence-transformers

# (Optional) Install BGE embedder
pip install sentence-transformers[sentence-transformers]

# (Optional) Install E5 embedder
pip install sentence-transformers[sentence-transformers]
```

## Usage

### Step 1: Generate Embeddings

Choose based on your needs:

#### Option A: Single Embedder (MiniLM)
```bash
python run_embeddings.py
```

This processes C1, C2, and C3 constraint datasets using only the MiniLM embedder.

#### Option B: Multi-Embedder
```bash
python run_embeddings2.py
```

This processes C1, C2, C3, and the online prompts dataset (A) using three embedders: MiniLM, BGE, and E5.

**Output**: Individual `.npy` files in `data/processed/embeddings_norm/` and `data/processed/embeddings_unnorm/`

### Step 2: Cluster and Visualize

#### Option A: Unified Analysis
```bash
python pca_kmean.py
```

This performs K-Means clustering on all embeddings combined and produces visualizations.

#### Option B: Grouped Analysis
```bash
python pca_kmean2.py
```

This groups embeddings by prefix (R1-R9 grouped, A1-A9 separate per model) and performs clustering within each group.

**Output**: 2D and 3D plots saved to `data/processed/plots/`

## Configuration

### Embedding Pipeline (`run_embeddings.py`, `run_embeddings2.py`)

The scripts automatically load embedding models and process all JSON files in the input directories.

**Supported Input JSON Formats**:

1. **Single Response**:
   ```json
   {
     "id": "unique_id",
     "response": "text content..."
   }
   ```

2. **Multiple Responses (List)**:
   ```json
   {
     "id": "unique_id",
     "responses": ["text1", "text2", ...]
   }
   ```

3. **Multi-Model Responses**:
   ```json
   {
     "id": "unique_id",
     "prompt": "question...",
     "responses": {
       "chatgpt": "response1...",
       "claude": "response2...",
       ...
     }
   }
   ```

### Clustering Configuration (`pca_kmean.py`, `pca_kmean2.py`)

Edit the configuration section at the top of either script:

```python
# Select embedding type
EMBEDDING_TYPE = "unnormalized"  # Options: "normalized", "unnormalized"

# Number of clusters
K = 3  # Adjust based on data characteristics

# Visual settings
REGION_STD = 2.0              # Covariance ellipse/ellipsoid size (in std devs)
REGION_ALPHA_2D = 0.15        # 2D region transparency
REGION_ALPHA_3D = 0.12        # 3D region transparency
POINT_SIZE = 100              # Scatter point size
POINT_ALPHA = 0.85            # Point transparency
```

## Data Formats

### Input Data (JSON)

#### Online Prompts (`online_prompts.json`)
Contains miscellaneous prompts with responses from 5 LLMs (ChatGPT, Gemini, Claude, Grok, DeepSeek). Used for broad behavioral analysis.

**Sample Structure**:
```json
{
  "id": "A1",
  "prompt": "What happens to you if you eat watermelon seeds?",
  "category": "Misconceptions",
  "responses": {
    "chatgpt": "...",
    "gemini": "...",
    ...
  }
}
```

#### Constraint Prompts (C1, C2, C3)
Carefully designed datasets to test semantic similarity under different conditions:

- **C1**: Similar context, different synonyms (testing synonym handling)
- **C2**: Different context, similar wording (testing context awareness)
- **C3**: Different context, different wording (baseline dissimilarity)

**Sample Structure**:
```json
{
  "id": "R1",
  "constraint": "similar context, different synonyms",
  "response": "text content..."
}
```

### Output Data (Embeddings)

Embeddings are stored as NumPy arrays (`.npy` files) with shape `(384,)`, `(768,)`, or `(1024,)` depending on the embedder model.

**Naming Convention**:
```
{id}_{embedder}.npy              # Unnormalized
{id}_{embedder}_norm.npy         # Normalized
```

**Example**: `A1_minilm.npy`, `R1_bge_norm.npy`

## Output Visualization

Generated plots include:

1. **2D Scatter Plot**
   - PCA reduction to 2 dimensions
   - Color-coded clusters
   - Covariance ellipses showing cluster regions

2. **3D Scatter Plot**
   - PCA reduction to 3 dimensions
   - Color-coded clusters
   - Covariance ellipsoids showing cluster regions

Both plots display point labels (item IDs) for reference.

## Algorithm Details

### K-Means Clustering

- **Input**: N × D embedding matrix (original dimensions)
- **Process**: Iterative clustering with k-means++ initialization
- **Output**: Cluster labels for each embedding

```
Configuration: random_state=42, n_init=10
```

### PCA Visualization

- **Input**: N × D embedding matrix
- **Output**: 
  - 2D: N × 2 coordinates
  - 3D: N × 3 coordinates

**Important**: PCA is applied ONLY for visualization. Clustering is performed in the original high-dimensional space.

### Covariance Regions

For each cluster in PCA space:
1. Calculate cluster center (mean of points)
2. Compute covariance matrix
3. Perform eigenvalue decomposition
4. Draw ellipse/ellipsoid with size proportional to eigenvalues

Size is controlled by `REGION_STD` (typically 2.0 = ±2 standard deviations).

## Troubleshooting

### Issue: Missing embedding files
**Solution**: Ensure `run_embeddings2.py` has been executed successfully and check that `data/processed/embeddings_unnorm/` or `data/processed/embeddings_norm/` contains `.npy` files.

### Issue: K > number of samples
**Solution**: The scripts automatically clamp K to the number of available samples. Consider running `pca_kmean2.py` for group-based analysis instead.

### Issue: Empty plots
**Solution**: Verify that embeddings were generated correctly and stored in the expected directories. Check console output for warnings.

## References

- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [Scikit-learn K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## License

This project is part of the Semantic Representation research initiative.

---

**Questions?** Check the inline documentation in each Python script or review the project configuration sections above.
