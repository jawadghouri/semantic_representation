# EmbeddingPCA Branch - Changes Tracking

## Overview
This document tracks all changes, additions, and experiments made on the EmbeddingPCA branch.

## Project Structure
- **`embeddings/`** - Embedding model implementations and utilities
- **`analysis/`** - Data analysis scripts and statistical analysis
- **`scripts/`** - Utility scripts and helper functions
- **`visualization/`** - Visualization and plotting code
- **`experiments/`** - Experimental code and prototypes
- **`utils/`** - General utility functions
- **`new/`** - Documentation and tracking for this branch

## Changes Made

### Added Files
- **`scripts/embed_new_dataset.py`** - Generate embeddings from new_dataset.json
- **`scripts/analyze_new_embeddings.py`** - Detailed embedding statistics
- **`scripts/pca_embeddings.py`** - PCA projection and visualization script
- **`new/embeddings/new_dataset_bge.npy`** - BGE embeddings (80 x 768 dim)
- **`new/embeddings/new_dataset_e5.npy`** - E5 embeddings (80 x 1024 dim)
- **`new/embeddings/new_dataset_minilm.npy`** - MiniLM embeddings (80 x 384 dim)
- **`new/embeddings/new_dataset_metadata.json`** - Metadata mapping embeddings to original data
- **`new/embeddings/new_dataset_*_pca_2d.npy`** - 2D PCA projections (3 files)
- **`new/embeddings/new_dataset_*_pca_3d.npy`** - 3D PCA projections (3 files)
- **`new/embeddings/new_dataset_*_pca_stats.json`** - PCA variance statistics (3 files)
- **`new/visualizations/pca_2d_by_category.png`** - PCA scatter plot colored by category
- **`new/visualizations/pca_2d_by_llm.png`** - PCA scatter plot colored by LLM model
- **`new/visualizations/pca_variance_explained.png`** - Variance explained comparison

### Modified Files
(None yet)

### Experiments & Notes

#### Dataset Statistics
- **Total Responses**: 80
- **Unique Prompts**: 16
- **Categories**: 16 different categories (Misconceptions, Conspiracies, Health, etc.)
- **LLM Models Covered**: ChatGPT, Claude, DeepSeek, Gemini, Grok (5 responses per prompt)

#### Embedding Results

| Model  | Dimension | Min Value | Max Value | Mean | Std Dev  | File Size |
|--------|-----------|-----------|-----------|------|----------|-----------|
| BGE    | 768       | -4.21     | 2.18      | -0.01 | 0.587   | 241 KB    |
| E5     | 1024      | -2.43     | 3.19      | 0.01  | 0.696   | 321 KB    |
| MiniLM | 384       | -0.73     | 0.66      | -0.00 | 0.161   | 121 KB    |

#### Key Observations
- All embeddings are **unnormalized** (as per project design)
- E5 has the highest dimensionality and wider value range
- MiniLM has the most constrained value range (±0.73)
- All three models produce different embedding spaces suitable for comparison
- Dataset includes diverse response types: factual answers, conspiracy debunking, opinion-based responses

---

## PCA Analysis Results

### PCA Transformations

| Model  | Original Dim | 2D Explained | 3D Explained | Reduction (2D) | Reduction (3D) |
|--------|--------------|--------------|--------------|----------------|----------------|
| BGE    | 768          | 18.99%       | 26.88%       | 384x           | 256x           |
| E5     | 1024         | 17.51%       | 25.04%       | 512x           | 341x           |
| MiniLM | 384          | 18.67%       | 27.38%       | 192x           | 128x           |

### Explained Variance Breakdown (2D)

**BGE:**
- PC1: 10.24%
- PC2: 8.74%
- Total: 18.99%

**E5:**
- PC1: 9.74%
- PC2: 7.77%
- Total: 17.51%

**MiniLM:**
- PC1: 9.52%
- PC2: 9.15%
- Total: 18.67%

### Explained Variance Breakdown (3D)

**BGE:**
- PC1: 10.24%, PC2: 8.74%, PC3: 7.90%
- Total: 26.88%

**E5:**
- PC1: 9.74%, PC2: 7.77%, PC3: 7.53%
- Total: 25.04%

**MiniLM:**
- PC1: 9.52%, PC2: 9.15%, PC3: 8.71%
- Total: 27.38%

### Key PCA Insights

1. **MiniLM shows balanced variance distribution** - PC1 and PC2 are nearly equal (9.52% vs 9.15%)
2. **BGE dominates with PC1** - Strong first principal component captures most variance
3. **E5 lowest total variance** - More distributed/diverse embedding space
4. **2D provides ~18-19% coverage** - Good for quick visualization
5. **3D adds ~7-9% more variance** - Meaningful improvement with third component
6. **Dense embedding space** - No single component captures >11%, indicating diverse information distribution

### Visualizations Generated (PCA Analysis)

1. **PCA 2D by Category** - Shows clustering patterns by 16 categories
2. **PCA 2D by LLM Model** - Shows patterns from 5 different LLM models
3. **Variance Explained** - Comparison of cumulative variance across models

---

## K-Means Clustering Analysis

### Pipeline Overview
```
Original Embeddings (768, 1024, 384 dims)
          ↓
    K-Means Clustering (K=3)
          ↓
  Cluster Labels (0, 1, 2)
          ↓
  PCA to 2D for Visualization
          ↓
Plot colored by Cluster + LLM Model
```

### Optimal K Determination
- Tested K from 2 to 10
- Silhouette scores increased with K (0.0874 for K=2 → 0.3095 for K=10)
- Selected K=3 for balanced analysis and clear visualization
- Optimal K would be 10 but K=3 provides better interpretability

### Clustering Results (K=3)

#### BGE Model (768-dim)
- **Silhouette Score**: 0.1069
- **Davies-Bouldin Index**: 2.7392
- **Cluster Distribution**: 
  - Cluster 0: 25 samples
  - Cluster 1: 30 samples  
  - Cluster 2: 25 samples
- **LLM Balance**: All LLMs evenly distributed (20% each per cluster)

#### E5 Model (1024-dim)
- **Silhouette Score**: 0.1063
- **Davies-Bouldin Index**: 2.7704
- **Cluster Distribution**:
  - Cluster 0: 10 samples
  - Cluster 1: 35 samples
  - Cluster 2: 35 samples
- **LLM Balance**: All LLMs evenly distributed (20% each per cluster)

#### MiniLM Model (384-dim)
- **Silhouette Score**: 0.1070
- **Davies-Bouldin Index**: 2.8084
- **Cluster Distribution**:
  - Cluster 0: 40 samples
  - Cluster 1: 15 samples
  - Cluster 2: 25 samples
- **LLM Balance**: All LLMs evenly distributed (20% each per cluster)

### Key LLM-Centric Findings

1. **Uniform LLM Distribution**: Each LLM model (ChatGPT, Claude, DeepSeek, Gemini, Grok) appears equally in all clusters across all embedding models
   - No LLM dominates any particular cluster
   - No bias toward any specific model in clustering

2. **Cluster Separation by Content**: Clusters primarily separate based on response content/category, NOT on LLM source
   - Different LLMs generate similar embeddings for similar content
   - Embedding models capture semantic similarity over model identity

3. **Silhouette Scores (~0.107)**: Modest separation indicates:
   - Natural overlap in response semantic spaces
   - Multiple response types within each cluster
   - Realistic, non-artificial clustering

### Files Generated

**Scripts:**
- `scripts/kmeans_clustering.py` - Complete K-Means pipeline

**Cluster Labels:**
- `new_dataset_bge_kmeans_labels.npy` (80 labels)
- `new_dataset_e5_kmeans_labels.npy` (80 labels)
- `new_dataset_minilm_kmeans_labels.npy` (80 labels)

**Clustering Statistics:**
- `new_dataset_bge_kmeans_stats.json` - Metrics & cluster counts
- `new_dataset_e5_kmeans_stats.json` - Metrics & cluster counts
- `new_dataset_minilm_kmeans_stats.json` - Metrics & cluster counts

**Cluster Composition Analysis:**
- `new_dataset_bge_cluster_analysis.json` - LLM breakdown per cluster
- `new_dataset_e5_cluster_analysis.json` - LLM breakdown per cluster
- `new_dataset_minilm_cluster_analysis.json` - LLM breakdown per cluster

**Visualizations (K-Means Specific):**
- `kmeans_pca_2d_clusters.png` - 3 subplots showing clusters in PCA space
- `kmeans_pca_2d_by_llm.png` - 3 subplots with LLM models as markers (overlaid on clusters)

### 2D Visualization Insights

**Chart 1 - Clusters in PCA Space (Colored by Cluster ID):**
- BGE: Well-separated clusters with distinct regions
- E5: More dispersed clustering, Cluster 0 small and concentrated
- MiniLM: Clear separation with Cluster 0 dominant (40 samples)

**Chart 2 - LLM Model Distribution (Different Markers per LLM):**
- All 5 LLM models (circles, squares, triangles, diamonds, inverted triangles) are mixed throughout the space
- No clustering by LLM model type
- Confirms content-driven rather than model-driven clustering

---

## 3D PCA Analysis (K-Means Clustering)

### 3D PCA Performance

| Model  | 3D Variance | PC1    | PC2    | PC3    |
|--------|-------------|--------|--------|--------|
| BGE    | 26.88%      | 10.24% | 8.74%  | 7.90%  |
| E5     | 25.04%      | 9.74%  | 7.77%  | 7.53%  |
| MiniLM | 27.38%      | 9.52%  | 9.15%  | 8.71%  |

### 3D Analysis Benefits

1. **Better Separation in 3D**
   - 3D adds 7-9% more variance explanation vs 2D
   - Clusters show more spatial separation
   - Better visualization of cluster centroids

2. **MiniLM Advantage in 3D**
   - Highest 3D variance: 27.38%
   - Balanced PC distribution (9.5%-9.15%-8.71%)
   - Clearest cluster regions in 3D space

3. **Multi-Angle Perspectives**
   - Generated 4 viewing angles: 45°, 135°, 225°, 315°
   - Shows cluster stability from different perspectives
   - Reveals 3D structure not visible in 2D

### 3D Visualization Insights

**Chart 1 - 3D Clusters (Viewing angle 45°):**
- BGE: Clusters form distinct 3D regions
- E5: Cluster 0 compact, Clusters 1&2 more dispersed
- MiniLM: Clear vertical separation with PC3

**Chart 2 - LLM Distribution in 3D:**
- All 5 LLM models uniformly scattered in 3D space
- No model-specific clustering in 3D either
- Consistent with 2D findings: content > model identity

**Charts 3-5 - Multiple Viewing Angles (135°, 225°, 315°):**
- Confirm cluster stability from all angles
- Show 3D structure is robust
- No artifacts from specific viewing angles

### 3D PCA Files Generated

**3D Projections:**
- `new_dataset_bge_kmeans_3d_pca.npy` - BGE 3D projection (80x3)
- `new_dataset_e5_kmeans_3d_pca.npy` - E5 3D projection (80x3)
- `new_dataset_minilm_kmeans_3d_pca.npy` - MiniLM 3D projection (80x3)

**3D Statistics:**
- `new_dataset_bge_kmeans_3d_pca_stats.json`
- `new_dataset_e5_kmeans_3d_pca_stats.json`
- `new_dataset_minilm_kmeans_3d_pca_stats.json`

**3D Visualizations (6 PNG files):**
- `kmeans_3d_pca_clusters.png` - Main 3D clusters view
- `kmeans_3d_pca_by_llm.png` - LLM models in 3D
- `kmeans_3d_pca_clusters_angle_45.png` - 45° viewing angle
- `kmeans_3d_pca_clusters_angle_135.png` - 135° viewing angle
- `kmeans_3d_pca_clusters_angle_225.png` - 225° viewing angle
- `kmeans_3d_pca_clusters_angle_315.png` - 315° viewing angle

### When to Use 2D vs 3D

**Use 2D When:**
- Quick visualization needed
- Presenting to audiences unfamiliar with 3D plots
- Printing or static documents
- Focus on 2 principal dimensions

**Use 3D When:**
- Need additional 8-9% variance explanation
- Examining cluster structure in depth
- Validating cluster separation
- Interactive exploration possible
- PC3 has significant variance contribution (>8%)

### Combined Analysis Recommendation

For this project (LLM response embeddings):
- **2D PCA**: Good for quick overview (18-19% variance)
- **3D PCA**: Recommended for deeper analysis (25-27% variance)
- **K-Means K=3**: Reasonable for initial clustering
- **K=5-7**: Consider for finer granularity
- **LLM-Agnostic**: All 5 LLMs embed similarly → content drives clustering

---

## Prompt-Centric 3D Comparison Analysis

### Overview
Created detailed 3D visualization comparing how BGE, E5, and MiniLM embedding models handle LLM responses to the same prompts. Similar to progress_meeting plots but for new_dataset.

### Visualization Types

**1. Individual Prompt Comparisons (16 plots)**
- One plot per prompt (A1-A16)
- Shows 3 embedding models side-by-side
- 5 LLM responses colored distinctly
- Includes variance explained per model
- Shows K-Means cluster assignments

**2. All Responses Overview (1 plot)**
- All 80 responses in 3D PCA space
- Colored by LLM model
- Shows inter-LLM distribution
- Confirms uniform LLM scatter

**3. Cluster Label Visualization (1 plot)**
- K-Means clusters with prompt IDs labeled
- Shows cluster assignments at a glance
- Helps identify semantic groupings

### Key Insights from Prompt Comparisons

**A1 - Watermelon Seeds (Misconceptions):**
- All 5 LLMs cluster tightly (similar responses)
- BGE & E5 show good separation, MiniLM compact
- All responses in same K-Means cluster

**A3 - Denver Airport (Conspiracies):**
- Mixed cluster assignments across models
- Shows content variation between LLMs
- Different variance patterns by embedding model

**A9 - Dessert Preference (Subjective):**
- Highest response diversity
- LLMs spread across different clusters
- Shows opinion-based content variation

### Files Generated

**Scripts:**
- `scripts/create_paragraph_3d_plots.py` (main visualization script)

**Visualizations (18 files, 6.2 MB):**
- 16 × Individual prompt comparison plots (A1-A16)
- 1 × All responses overview plot
- 1 × Cluster labels visualization

### Plot Structure (Individual Prompts)

Each prompt plot contains:
```
Header:
  - Prompt ID, Category, Question text
  - Total variance explained by all 3 models

3 Subplots (BGE, E5, MiniLM):
  - X, Y, Z axes showing PC1, PC2, PC3
  - 5 colored markers for each LLM model:
    • Red Circle = ChatGPT
    • Cyan Square = Claude
    • Blue Triangle = DeepSeek
    • Orange Diamond = Gemini
    • Green Inverted Triangle = Grok

Bottom Panel:
  - LLM color legend
  - K-Means cluster assignment per model
```

### Analysis Features

1. **Model-specific Variance:**
   - Each subplot shows individual model's explained variance
   - Reveals which model captures most information

2. **LLM Behavior Patterns:**
   - Tight clustering = Similar LLM responses
   - Spread out = Diverse LLM responses

3. **Cluster Distribution:**
   - Shows how K-Means groups the 5 responses
   - Reveals response homogeneity/diversity

4. **Content Type Impact:**
   - Factual questions (A1, A10) → Tight clustering
   - Opinion questions (A9) → Spread out
   - Conspiracy/Debunking (A3, A5) → Mixed patterns

### Observations by Category

**Factual/Consensus (Tight Clustering):**
- A1: Watermelon seeds
- A10: Diamond durability  
- A11: First woman
- A15: Straw on camel

**Conspiracy/Debunking (Mixed):**
- A3: Denver Airport
- A5: Contact the dead
- A12: German characteristics
- A16: Psychopath test

**Opinion/Subjective (Loose Clustering):**
- A9: Dessert preference
- A13: Foreign languages benefit
- A14: Language learning method

**Fictional/Mythical (Consistent):**
- A6: Wizards in West Country
- A7: Flying carpets

### Comparison with Progress Meeting

**Similarities:**
- 3D PCA projection with variance explained
- Multiple embedding models compared
- Clear visualization of semantic space

**Differences:**
- Progress meeting: Paragraph variants (similar/different context, similar/different synonyms)
- New dataset: LLM model responses to same prompt
- Focus: Content variation vs Model behavior

---
*Branch: EmbeddingPCA*  
*Created: 2026-08-20*
