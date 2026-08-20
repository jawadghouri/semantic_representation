# Semantic Representation of LLM Outputs

## Project Overview

This project investigates **how different Large Language Models (LLMs) represent information semantically**, measured through embedding distances and clustering analysis. By generating responses from multiple LLMs to the same prompts, encoding them with various embedding models, and analyzing their spatial relationships, we gain insights into semantic similarity/diversity across models.

### Core Research Question

**Do Llama, Mistral, and Phi produce semantically similar or different responses to the same prompt?**

This is measured through:
- **Embedding distances** between LLM responses and prompts
- **Clustering patterns** in embedding space
- **Inter/intra class separability** via discriminant analysis
- **Multi-embedder robustness** across different embedding dimensions

---

## Four-Stage Pipeline Architecture

The project implements a linear, reproducible pipeline that transforms text prompts into quantitative semantic similarity metrics.

### Stage 1: Text Generation
**File:** `pipelines/run_generation.py`

Generates responses from three LLMs to input prompts.

**LLM Models:**
- Llama: `meta-llama/Llama-3.1-8B-Instruct`
- Mistral: `mistralai/Mistral-7B-Instruct-v0.2`
- Phi: `microsoft/Phi-3-mini-4k-instruct`

**Input:**
- Prompts from `data/prompts/prompts.json`

**Output:**
- `data/raw_outputs/{llama,mistral,phi}_outputs.json`
  - Format: JSON with prompt text and 200-token responses per model

**Configuration:**
- Device: CUDA (GPU required)
- Max tokens: 200
- Temperature: 0.7 (for response diversity)
- VRAM management: Sequential model loading with cache purging between models

---

### Stage 2: Embedding
**File:** `pipelines/run_embeddings.py`

Encodes LLM responses and original prompts using multiple embedding models.

**Embedding Models:**
- **MiniLM** (`all-MiniLM-L6-v2`): 384-dim, mean pooling
- **BGE** (`BAAI/bge-base-en-v1.5`): 768-dim, CLS pooling
- **E5** (`intfloat/e5-large-v2`): 1024-dim, mean pooling with "passage:" prefix

**Output Structure:**
```
data/processed/embeddings/
├── {llm}_{embedder}_{prompt_id}.npy    # Response embeddings
└── {embedder}_prompts_{prompt_id}.npy   # Prompt embeddings
```

**Critical Design Choice: Unnormalized Embeddings**
- Embeddings are kept as **raw L2-norm vectors** (NOT normalized to unit sphere)
- **Euclidean distance** is the distance metric (NOT cosine similarity)
- Verification: `utils/norm_utils.py` confirms `✅ UNNORMALIZED`
- **Why:** Allows geometric analysis of response clustering in embedding space

---

### Stage 3: FAISS Indexing
**File:** `pipelines/run_faiss.py`

Builds efficient vector indexes for semantic retrieval.

**Index Type:** `faiss.IndexFlatL2` (Euclidean L2 distance)

**Output:**
```
data/processed/faiss/
└── {llm}_{embedder}_{prompt_id}.index
```

**Purpose:** Enables fast nearest-neighbor retrieval of semantically similar responses

---

### Stage 4: Analysis & Visualization

#### Analysis
**File:** `pipelines/run_analysis.py`

Computes semantic similarity metrics between prompts and LLM responses.

**Metrics Computed:**
- Euclidean distance from prompt embedding to each response embedding
- Mean, std, min, max distances per LLM
- Inter-model distance comparisons

**Output:**
```
results/similarities/
└── {llm}_{embedder}_distances.json
```

**Example Result:**
```json
{
  "prompt": "Explain how a car engine works.",
  "embedder": "all-MiniLM-L6-v2",
  "llama_mean_distance": 4.23,
  "mistral_mean_distance": 4.78,
  "phi_mean_distance": 5.12
}
```

#### Visualization
**File:** `pipelines/run_visualization.py`

Generates publication-ready visualizations of semantic relationships.

**Chart Types:**
1. **Bar plots** - Mean distances per LLM/embedder
2. **Heatmaps** - Pairwise distance matrices showing response clustering
3. **UMAP plots** - 2D projections of high-dimensional embedding space

**Output:**
```
results/figures/
├── bar_charts/
├── heatmaps/
└── umap_plots/
```

---

## Directory Structure

```
semantic_representation/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .env                               # Environment variables
│
├── config/
│   ├── settings.py                   # Global configuration (device, tokens, temperature)
│   └── model_config.py               # Model paths and hyperparameters
│
├── data/
│   ├── prompts/
│   │   └── prompts.json              # Input prompts
│   ├── raw_outputs/                  # Stage 1 output (LLM responses)
│   │   ├── llama_outputs.json
│   │   ├── mistral_outputs.json
│   │   └── phi_outputs.json
│   └── processed/
│       ├── embeddings/               # Stage 2 output (.npy files)
│       └── faiss/                    # Stage 3 output (indexes)
│
├── pipelines/                         # Core execution scripts
│   ├── run_generation.py             # Stage 1: Generate LLM responses
│   ├── run_embeddings.py             # Stage 2: Embed responses
│   ├── run_faiss.py                  # Stage 3: Build FAISS indexes
│   └── run_analysis.py               # Stage 4a: Compute distances
│
├── llm/                               # LLM model loaders
│   ├── llama_generator.py
│   ├── mistral_generator.py
│   └── phi_generator.py
│
├── embeddings/                        # Embedding model wrappers
│   └── base_embedder.py              # Base class for embedding models
│
├── vectorstore/                       # Retrieval utilities
│   ├── faiss_manager.py
│   └── retrieval.py
│
├── visualization/                     # Stage 4b: Plotting
│   ├── bar_chart.py
│   ├── heatmap.py
│   └── umap_plot.py
│
├── analysis/                          # Analysis utilities
│   ├── similarity_analysis.py
│   ├── distance.py
│   ├── statistics.py
│   └── embedding_statistics.py
│
├── utils/                             # Helper utilities
│   ├── norm_utils.py                 # Verify unnormalized embeddings
│   └── ...
│
├── results/                           # Final outputs
│   ├── similarities/                 # Distance metrics (JSON)
│   └── figures/                      # Visualizations (PNG)
│
├── progress_meeting/                  # Standalone research experiments
│   ├── run_embeddings.py
│   ├── heatmap.py
│   ├── pca.py
│   └── graphs.py
│
└── aporia/                            # APORIA framework (separate branch)
    └── ...
```

---

## Quick Start

### Prerequisites

- **Python:** 3.8+
- **GPU:** NVIDIA CUDA-capable GPU with 16GB+ VRAM (recommended)
- **CUDA:** Compatible CUDA and cuDNN installation

### Installation

```bash
# Clone repository
git clone https://github.com/jawadghouri/semantic_representation.git
cd semantic_representation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "DEVICE=cuda" > .env
```

### Running the Full Pipeline

```bash
# Stage 1: Generate LLM responses (~30-45 minutes)
python -m pipelines.run_generation

# Stage 2: Embed responses (~10-15 minutes)
python -m pipelines.run_embeddings

# Stage 3: Build FAISS indexes (~2-5 minutes)
python -m pipelines.run_faiss

# Stage 4: Analyze and visualize (~5 minutes)
python -m pipelines.run_analysis
python -m pipelines.run_visualization
```

**Total runtime:** ~60 minutes on single GPU

### Viewing Results

```bash
# View computed distances
cat results/similarities/*.json

# View generated charts
ls -lh results/figures/
open results/figures/bar_charts/  # or view in image viewer
```

---

## Key Design Decisions

### 1. Unnormalized Embeddings

**Decision:** Keep embeddings as raw L2-norm vectors (not normalized to unit sphere)

**Rationale:**
- Enables geometric analysis of response clustering
- Euclidean distance naturally captures embedding magnitude
- Preserves semantic density information in embedding space

**Verification:** Run `python -c "from utils.norm_utils import verify_unnormalized; verify_unnormalized()"`

### 2. Three Embedding Models

**Decision:** Test framework across MiniLM (small), BGE (medium), E5 (large)

**Rationale:**
- Shows robustness across embedding dimensions (384 → 768 → 1024)
- Validates semantic relationships are not dimension-specific
- Enables comparison of embedding model impacts

### 3. Sequential Model Loading

**Decision:** Load/unload LLM one at a time to manage VRAM

**Rationale:**
- Three 8B models = ~16GB VRAM each
- Sequential prevents OOM errors on 24GB GPUs
- Negligible runtime overhead vs. GPU memory constraints

### 4. Euclidean Distance Metric

**Decision:** Use Euclidean L2 distance instead of cosine similarity

**Rationale:**
- Analyzes actual geometric distances in embedding space
- Captures response clustering patterns
- Foundation for future Fisher projection analysis

---

## Understanding the Output

### Similarity Metrics (results/similarities/)

Each JSON file contains distance computations between prompt embedding and response embeddings:

```json
{
  "prompt": "Explain how a car engine works.",
  "embedder": "all-MiniLM-L6-v2",
  "llama": {
    "mean_distance": 4.23,
    "std_distance": 0.45,
    "min_distance": 3.12,
    "max_distance": 5.67
  },
  "mistral": { ... },
  "phi": { ... }
}
```

**Interpretation:**
- **Lower distance** = response is semantically closer to prompt
- **Smaller std** = responses are more consistent (clustered)
- **Larger std** = responses are more diverse (spread)

### Visualizations (results/figures/)

1. **Bar Charts:** Compare mean distances across models/embedders
2. **Heatmaps:** Show response clustering patterns (darker = tighter clustering)
3. **UMAP Plots:** Visual 2D projection of semantic space

---

## Performance Metrics

### Runtime Breakdown

| Stage | Duration | Primary Bottleneck |
|-------|----------|-------------------|
| 1. Generation | 30-45 min | LLM inference (sequential) |
| 2. Embedding | 10-15 min | Transformer encoding |
| 3. FAISS | 2-5 min | Index construction |
| 4. Analysis | <1 min | Distance computation |
| 5. Visualization | <1 min | PNG generation |
| **Total** | **~60 min** | Stage 1 (LLM inference) |

### Memory Requirements

| Component | VRAM | System RAM |
|-----------|------|-----------|
| LLM Model | ~16 GB | - |
| Embedder Model | ~2 GB | - |
| Embeddings (.npy) | - | ~500 MB |
| FAISS indexes | - | ~300 MB |

---

## Configuration

Edit `config/settings.py` to customize:

```python
DEVICE = "cuda"              # or "cpu"
MAX_NEW_TOKENS = 200         # Response length
TEMPERATURE = 0.7            # Randomness (0=deterministic, 1=random)
LLM_MODELS = [               # Which LLMs to use
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft/Phi-3-mini-4k-instruct"
]
EMBEDDING_MODELS = [         # Which embedders to use
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-large-v2"
]
```

---

## Extending the Project

### Adding New Prompts

1. Edit `data/prompts/prompts.json`:
```json
[
  {
    "id": "P1",
    "prompt": "Your question here"
  },
  ...
]
```

2. Re-run pipeline from Stage 1

### Using Different Models

Edit `config/settings.py` to specify different LLMs or embedders

### Custom Analysis

Extend `analysis/` modules to compute additional metrics:
- Semantic diversity measures
- Response coherence scores
- Topic clustering analysis

---

## Troubleshooting

### CUDA Out of Memory

**Solution:** Edit `config/settings.py`, reduce `MAX_NEW_TOKENS` or use smaller models

### Slow Embedding Generation

**Solution:** Use smaller embedder (MiniLM instead of E5) or reduce batch size

### Missing Transformer Models

**Solution:** Models auto-download on first run. Ensure internet connection during Stage 2

### File Permission Errors

**Solution:** Ensure write permissions on `data/` and `results/` directories

---

## Related Work

This project is inspired by:
- **APORIA** (arXiv:2602.14778): Hallucination detection via geometric clustering
- Embedding space analysis for semantic similarity
- LLM response diversity measurement

See the `aporia` branch for the full APORIA framework implementation (225 responses, hallucination detection, Fisher projection).

---

## Project Status

- ✅ **Main branch:** Core 4-stage pipeline (text generation → embedding → indexing → analysis)
- ✅ **Data:** Configuration and prompt management
- ✅ **Visualization:** Publication-ready charts and plots
- ⏳ **APORIA branch:** Advanced hallucination detection framework (separate branch)

---

## Contributors

- Talha Hussain Qureshi (talhahussain847@gmail.com)

---

## License

[Add license information]

---

## Citation

If you use this project, please cite:

```bibtex
@project{semantic_representation_2026,
  title={Semantic Representation of LLM Outputs},
  author={Qureshi, Talha Hussain},
  year={2026},
  url={https://github.com/jawadghouri/semantic_representation}
}
```

---

## Contact & Support

For questions or issues:
- Email: talhahussain847@gmail.com
- GitHub Issues: [Link to issue tracker]

---

**Last Updated:** August 6, 2026  
**Branch:** main  
**Status:** Active Development
