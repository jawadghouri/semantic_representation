# Multi-LLM Embedding & PCA Visualization Pipeline

**Status:** Planning phase (awaiting hallucination detection approach details)  
**Branch:** main  
**Date:** 2026-08-15

## Executive Summary

This document outlines the pipeline for integrating 5 external LLM APIs (ChatGPT, Gemini, Claude, DeepSeek, Grok) with existing 3 local LLMs (Llama, Mistral, Phi) to generate embeddings and visualize them in 2D/3D PCA space with clustering.

**Scope:** Embedding generation → PCA visualization (NOT hallucination detection framework)

---

## Current State (Main Branch)

### Existing Pipeline (4 Stages)
```
Stage 1: Text Generation
├── Local LLMs: Llama, Mistral, Phi
├── Config: config/settings.py
└── Output: data/raw_outputs/{llama,mistral,phi}_outputs.json

Stage 2: Embedding
├── Embedding models: MiniLM (384-dim), BGE (768-dim), E5 (1024-dim)
├── Design: UNNORMALIZED embeddings (Euclidean metric)
└── Output: embeddings/ (.npy files)

Stage 3: FAISS Indexing
├── Index type: IndexFlatL2
└── Output: data/processed/faiss/

Stage 4: Analysis & Visualization
├── Analysis: Distance computation
├── Visualization: Bar charts, heatmaps, UMAP
└── Output: results/
```

### Key Design Decisions (Maintained)
- **Unnormalized Embeddings:** Keep raw L2-norm vectors (NOT normalized to unit sphere)
- **Euclidean Distance Metric:** Pairwise distances computed as L2 norm
- **Verification:** `utils/norm_utils.py` confirms ✅ UNNORMALIZED status

---

## New Task: Multi-LLM Expansion

### Objective
Expand pipeline to include 5 external LLM APIs and create comprehensive 2D/3D PCA visualizations with clustering.

### Requirements (from Supervisor)
1. Send **same prompts** to 8 LLMs (3 local + 5 API)
2. Collect responses from all LLMs
3. Generate embeddings across all responses
4. Create **2D PCA** visualization with clusters
5. Create **3D PCA** visualization with clusters
6. Generate cluster assignments

### Future Integration
- Once hallucination detection approach is finalized, these embeddings + clusters will feed into the detection framework
- Current focus: clean embedding generation and visualization

---

## Proposed 4-Stage Pipeline

### Stage 1: Response Collection

**Goal:** Generate/collect responses from all 8 LLMs

**Local LLMs (Existing)**
- Use: `pipelines/run_generation.py` (unchanged)
- Models: Llama-3.1-8B, Mistral-7B, Phi-3-mini
- Output: `data/raw_outputs/{llama,mistral,phi}_outputs.json`

**API LLMs (New)**
- Create: `pipelines/run_api_generation.py`
- Models: ChatGPT, Gemini, Claude, DeepSeek, Grok
- Requires: API keys for each provider
- Output: `data/raw_outputs/api_responses.json`

**Considerations:**
- ⚠️ API costs (define budget with supervisor)
- ⚠️ Rate limiting (implement exponential backoff)
- ⚠️ Response caching (avoid re-running, save cost)
- ⚠️ Response consistency (may need preprocessing)

---

### Stage 2: Response Aggregation

**Goal:** Unified storage for all 8 LLM responses

**Create:** `pipelines/run_aggregation.py`

**Input:**
- Local responses: `data/raw_outputs/{llama,mistral,phi}_outputs.json`
- API responses: `data/raw_outputs/api_responses.json`

**Output:** `data/responses_all.json`

**Format:**
```json
{
  "prompts": [
    {
      "prompt_id": "P1",
      "prompt_text": "...",
      "responses": {
        "llama": ["response1", "response2", ...],
        "mistral": ["response1", "response2", ...],
        "phi": [...],
        "chatgpt": [...],
        "gemini": [...],
        "claude": [...],
        "deepseek": [...],
        "grok": [...]
      }
    }
  ]
}
```

**Benefits:**
- Single source of truth
- Easy to extend with new LLMs
- Metadata tracking possible

---

### Stage 3: Embedding Generation

**Goal:** Generate embeddings for all responses using all embedding models

**Extend:** `pipelines/run_embeddings.py`

**Input:** `data/responses_all.json`

**Process:**
```python
# For each embedding model (MiniLM, BGE, E5):
#   For each LLM (8 total):
#     Embed all responses
#     Save to: embeddings/{llm}_{embedder}.npy
```

**Output:** `data/embeddings/`
```
embeddings/
├── llama_minilm.npy       (N_responses × 384)
├── llama_bge.npy          (N_responses × 768)
├── llama_e5.npy           (N_responses × 1024)
├── mistral_minilm.npy
├── mistral_bge.npy
├── mistral_e5.npy
├── phi_minilm.npy
├── phi_bge.npy
├── phi_e5.npy
├── chatgpt_minilm.npy
├── ... (5 API LLMs × 3 embedders = 15 more)
```

**Total:** 24 embedding files (8 LLMs × 3 embedders)

**Critical Requirement:**
```python
# Verify unnormalized status
from utils.norm_utils import check_normalization
check_normalization(embedding_matrix)  # Should output: ✅ UNNORMALIZED
```

**GPU Memory Consideration:**
- Embedding 8 LLMs × multiple prompts × 3 models is memory-intensive
- Use batching to avoid OOM
- May need to process LLMs sequentially

---

### Stage 4: PCA Visualization & Clustering

**Goal:** Create 2D and 3D PCA visualizations with cluster assignments

**Create:** `pipelines/run_pca_clustering.py`

#### 4.1 Clustering
- **Algorithm:** KMeans (k = 3-5, to be decided with supervisor)
- **Alternative:** DBSCAN (optional, for comparison)
- **Input:** Embeddings from each embedding model
- **Output:** Cluster labels

#### 4.2 PCA 2D (Static Visualizations)
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca_2d = PCA(n_components=2)
embeddings_2d = pca_2d.fit_transform(embeddings)

# Plot 1: Colored by cluster
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels)
plt.savefig('pca_2d_by_cluster.png')

# Plot 2: Colored by LLM provider
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=llm_colors)
plt.savefig('pca_2d_by_llm.png')

# Plot 3: Colored by embedding model
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=embedding_colors)
plt.savefig('pca_2d_by_embedding.png')
```

#### 4.3 PCA 3D (Interactive Visualizations)
```python
import plotly.graph_objects as go

pca_3d = PCA(n_components=3)
embeddings_3d = pca_3d.fit_transform(embeddings)

# Plot 1: Colored by cluster
fig = go.Figure(data=[go.Scatter3d(
    x=embeddings_3d[:, 0], y=embeddings_3d[:, 1], z=embeddings_3d[:, 2],
    mode='markers',
    marker=dict(size=5, color=cluster_labels, colorscale='Viridis'),
)])
fig.write_html('pca_3d_by_cluster.html')

# Similarly for LLM provider and embedding model colorings
```

#### 4.4 Clustering Metrics
```python
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

metrics = {
    "silhouette": silhouette_score(embeddings, cluster_labels),
    "davies_bouldin": davies_bouldin_score(embeddings, cluster_labels),
    "calinski_harabasz": calinski_harabasz_score(embeddings, cluster_labels)
}
```

**Output:** `results/pca_visualizations/`
```
pca_visualizations/
├── 2d/
│   ├── minilm_by_cluster.png
│   ├── minilm_by_llm.png
│   ├── minilm_by_embedding.png
│   ├── bge_by_cluster.png
│   ├── bge_by_llm.png
│   ├── bge_by_embedding.png
│   ├── e5_by_cluster.png
│   ├── e5_by_llm.png
│   └── e5_by_embedding.png       (9 PNG files total)
├── 3d/
│   ├── minilm_by_cluster.html
│   ├── minilm_by_llm.html
│   ├── minilm_by_embedding.html
│   ├── bge_by_cluster.html
│   ├── bge_by_llm.html
│   ├── bge_by_embedding.html
│   ├── e5_by_cluster.html
│   ├── e5_by_llm.html
│   └── e5_by_embedding.html       (9 HTML files total)
└── analysis.json                  (Clustering metrics for all combinations)
```

**Total Visualizations:** 18 files (9 × 2D PNG + 9 × 3D HTML)

---

## Key Design Decisions

### 1. Prompt Selection
**Decision Needed:** Which prompts to use?
- Option A: Reuse existing prompts from main branch
- Option B: New set of prompts designed for this task
- Option C: Prompts from hallucination detection paper (once provided)

### 2. API Management
**Implementation:**
- Create `api_manager/` module
- Separate client for each LLM provider
- Centralized rate limiting & auth handling
- Request retry logic with exponential backoff

### 3. Embedding Consistency
**Requirement:** All responses embedded uniformly
- Same preprocessing (if any)
- Same embedding models
- Unnormalized output (critical for consistency)

### 4. Clustering Approach
**Decision Needed:** KMeans parameter k
- Option A: k=3 (natural semantic groups)
- Option B: k=5 (finer-grained clusters)
- Option C: DBSCAN (automatic k)

**Recommendation:** Start with KMeans k=4, then explore other values

### 5. Visualization Color Schemes
**Three dimensions to explore:**
1. **By Cluster:** See semantic grouping patterns
2. **By LLM Provider:** Do APIs cluster differently from local LLMs?
3. **By Embedding Model:** Does clustering pattern hold across embedders?

---

## File Structure

```
data/
├── prompts.json                    [NEW: Config with all prompts]
├── responses_all.json              [NEW: Aggregated responses from 8 LLMs]
└── embeddings/
    ├── llama_minilm.npy
    ├── llama_bge.npy
    ├── llama_e5.npy
    ├── mistral_minilm.npy
    ├── mistral_bge.npy
    ├── mistral_e5.npy
    ├── phi_minilm.npy
    ├── phi_bge.npy
    ├── phi_e5.npy
    ├── chatgpt_minilm.npy
    ├── chatgpt_bge.npy
    ├── chatgpt_e5.npy
    ├── gemini_minilm.npy
    ├── gemini_bge.npy
    ├── gemini_e5.npy
    ├── claude_minilm.npy
    ├── claude_bge.npy
    ├── claude_e5.npy
    ├── deepseek_minilm.npy
    ├── deepseek_bge.npy
    ├── deepseek_e5.npy
    ├── grok_minilm.npy
    ├── grok_bge.npy
    └── grok_e5.npy               (24 files total)

pipelines/
├── run_generation.py               [EXISTING: local LLMs]
├── run_embeddings.py               [EXTEND: process all 8 LLMs]
├── run_api_generation.py           [NEW: API LLM calls]
├── run_aggregation.py              [NEW: merge responses]
└── run_pca_clustering.py           [NEW: PCA + clustering + viz]

api_manager/                         [NEW: API abstraction layer]
├── __init__.py
├── base_client.py                  [Abstract LLM client interface]
├── openai_client.py                [ChatGPT]
├── google_client.py                [Gemini]
├── anthropic_client.py             [Claude]
├── deepseek_client.py              [DeepSeek]
├── xai_client.py                   [Grok]
├── rate_limiter.py                 [Rate limiting & backoff]
├── config.py                       [API endpoints, auth]
└── utils.py                        [Helper functions]

results/
├── pca_visualizations/
│   ├── 2d/                         [9 PNG files: 3 embedders × 3 colorings]
│   ├── 3d/                         [9 HTML files: 3 embedders × 3 colorings]
│   └── analysis.json               [Clustering metrics]
└── cluster_assignments.json        [Label assignments for all responses]
```

---

## Implementation Roadmap

### Phase 1: Setup (2-3 days)
- [ ] Confirm prompts with supervisor
- [ ] Gather API keys (ChatGPT, Gemini, Claude, DeepSeek, Grok)
- [ ] Create `data/prompts.json`
- [ ] Set up `.env` for API key management
- [ ] Create API manager skeleton

### Phase 2: API Integration (3-5 days)
- [ ] Implement API clients for all 5 providers
- [ ] Add rate limiting & retry logic
- [ ] Test each API with 1 sample prompt
- [ ] Create caching mechanism

### Phase 3: Response Collection (3-5 days)
- [ ] Run local LLMs (use existing code)
- [ ] Run API LLMs (new code)
- [ ] Aggregate responses → `responses_all.json`
- [ ] Validate: all LLMs have responses for all prompts

### Phase 4: Embedding Generation (2-3 days)
- [ ] Extend `run_embeddings.py`
- [ ] Generate embeddings for all 8 LLMs
- [ ] Verify unnormalized status
- [ ] Check for NaN values

### Phase 5: PCA & Clustering (3-4 days)
- [ ] Implement KMeans clustering
- [ ] Create 2D PCA plots (3 colorings × 3 embedders = 9 PNG)
- [ ] Create 3D PCA plots (3 colorings × 3 embedders = 9 HTML)
- [ ] Compute clustering metrics

### Phase 6: Analysis & Validation (2-3 days)
- [ ] Verify cluster separation (silhouette score > 0.4)
- [ ] Visual inspection of plots
- [ ] Generate summary statistics
- [ ] Document findings

**Total Timeline:** 4-5 weeks (depends on API access & responsiveness)

---

## Critical Considerations

### API Management
- **Costs:** Define budget with supervisor (variable per provider)
- **Rate Limits:** Implement backoff to avoid hitting limits
- **Authentication:** Store keys in `.env`, never commit to git
- **Caching:** Save responses to disk to avoid re-running

### Response Processing
- **Consistency:** Responses may vary in length/quality
- **Preprocessing:** May need truncation or cleaning
- **Validation:** Check for empty/error responses before embedding

### Embedding Generation
- **Memory:** Batching needed for large response sets
- **Verification:** Must confirm ✅ UNNORMALIZED status
- **Storage:** 24 .npy files, monitor disk space

### Visualization
- **Interpretability:** 3 coloring schemes answer different questions
- **Interactive 3D:** Requires Plotly (already in requirements)
- **Static 2D:** Matplotlib (already available)

---

## Validation Checkpoints

### After Phase 2: API Layer Ready?
- ✅ All 5 API clients initialized
- ✅ Rate limiter working
- ✅ Test 1 prompt per LLM successful
- ✅ Error handling in place

### After Phase 3: Responses Complete?
- ✅ `responses_all.json` valid JSON
- ✅ All 8 LLMs present
- ✅ All prompts have responses from all LLMs
- ✅ No NULL/empty responses

### After Phase 4: Embeddings Valid?
- ✅ All 24 embedding files exist
- ✅ Shape correct (N × dimension)
- ✅ Status: ✅ UNNORMALIZED confirmed
- ✅ No NaN values
- ✅ Memory used reasonable

### After Phase 5: Clusters Meaningful?
- ✅ Silhouette score > 0.4 (good separation)
- ✅ Cluster sizes reasonable (not all in one cluster)
- ✅ Visual inspection: clusters visually distinct
- ✅ All coloring schemes produce interpretable plots

### After Phase 6: Ready for Hallucination Detection?
- ✅ Embeddings & clusters saved
- ✅ Visualization complete & reviewed
- ✅ No blocking issues
- ✅ Documentation complete

---

## Questions for Supervisor Alignment

**Before Phase 1 starts:**

1. **Prompts:** Reuse existing or new set? How many prompts?
2. **LLM Versions:** Specific model versions? (GPT-4 vs 3.5-turbo, Gemini 2.0, etc.)
3. **Budget:** Cost limit for API calls? Any quota constraints?
4. **Clustering:** KMeans k value? Include DBSCAN?
5. **Hallucination Approach:** Once paper is provided, do the prompts need adjustment?
6. **Output:** Static PNG reports or interactive dashboards? Publishing requirements?
7. **Timeline:** 4-5 weeks ok? Any hard deadline?

---

## Next Steps

1. **Receive hallucination detection paper** from group member
2. **Alignment meeting** with supervisor on above questions
3. **Phase 1 kickoff** once decisions finalized
4. **Track progress** with this checklist

---

**Document Version:** 1.0  
**Last Updated:** 2026-08-15  
**Status:** Awaiting hallucination detection approach details
