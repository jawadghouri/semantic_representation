# EmbeddingPCA Branch - Changes Tracking

## Project Phase: PCA Analysis Only

This branch contains the complete embedding generation and PCA analysis for the new_dataset.json file.

---

## Dataset Overview

- **Total Responses**: 80 (16 prompts × 5 LLM models)
- **Unique Prompts**: 16
- **Question Categories**: 16 diverse categories
- **LLM Models**: ChatGPT, Claude, DeepSeek, Gemini, Grok
- **Source**: `progress_meeting/constraint_prompts/new_dataset.json`

---

## Generated Embeddings

### Embedding Models

| Model  | Dimension | Min Value | Max Value | Mean   | Std Dev | File Size |
|--------|-----------|-----------|-----------|--------|---------|-----------|
| BGE    | 768       | -4.21     | 2.18      | -0.012 | 0.587   | 245 KB    |
| E5     | 1024      | -2.43     | 3.19      | 0.013  | 0.696   | 328 KB    |
| MiniLM | 384       | -0.73     | 0.66      | -0.001 | 0.161   | 123 KB    |

### Key Features
- All embeddings are **unnormalized** (project design)
- Diverse dimensional spaces for comparison
- Complete metadata preserved for each response

---

## PCA Analysis Results

### 2D PCA Performance

| Model  | Explained Variance | PC1    | PC2    | Reduction |
|--------|-------------------|--------|--------|-----------|
| BGE    | 18.99%             | 10.24% | 8.74%  | 384×      |
| E5     | 17.51%             | 9.74%  | 7.77%  | 512×      |
| MiniLM | 18.67%             | 9.52%  | 9.15%  | 192×      |

### 3D PCA Performance

| Model  | Explained Variance | PC1    | PC2    | PC3    | Reduction |
|--------|-------------------|--------|--------|--------|-----------|
| BGE    | 26.88%             | 10.24% | 8.74%  | 7.90%  | 256×      |
| E5     | 25.04%             | 9.74%  | 7.77%  | 7.53%  | 341×      |
| MiniLM | 27.38%             | 9.52%  | 9.15%  | 8.71%  | 128×      |

### Key Insights

1. **MiniLM shows balanced variance distribution** 
   - PC1 and PC2 nearly equal (9.52% vs 9.15%)

2. **BGE dominates with PC1** 
   - Strong first principal component

3. **E5 has distributed embedding space** 
   - More balanced variance across components

4. **2D captures 17-19% of information**
   - Suitable for quick visualization

5. **3D adds 7-9% more variance**
   - Meaningful improvement for detailed analysis

---

## Files Generated

### Embeddings (3 files)
- `new_dataset_bge.npy` - BGE embeddings
- `new_dataset_e5.npy` - E5 embeddings
- `new_dataset_minilm.npy` - MiniLM embeddings

### PCA Projections (6 files)
- `new_dataset_*_pca_2d.npy` - 2D projections (3 files)
- `new_dataset_*_pca_3d.npy` - 3D projections (3 files)

### Statistics (4 files)
- `new_dataset_*_pca_stats.json` - Variance statistics (3 files)
- `new_dataset_metadata.json` - Response metadata

### Scripts (2 files)
- `scripts/embed_new_dataset.py` - Embedding generation
- `scripts/analyze_new_embeddings.py` - Statistical analysis

---

## Usage

### Generate embeddings:
```bash
python scripts/embed_new_dataset.py
```

### Analyze embeddings:
```bash
python scripts/analyze_new_embeddings.py
```

---

## Next Steps (Optional)

1. **Visualizations** - Create 2D/3D PCA plots
2. **Clustering** - Apply K-Means for response grouping
3. **Comparison** - Analyze embedding model differences
4. **Quality Analysis** - Evaluate response characteristics

---

*Branch: EmbeddingPCA*  
*Phase: PCA Analysis*  
*Status: Ready for use/expansion*
