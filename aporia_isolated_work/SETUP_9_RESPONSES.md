# Setup Guide — 9 Responses, 3 Groups

This guide covers setup for your specific experiment: **9 responses across 3 semantic constraint groups**.

---

## Your Data Structure

```
Total: 9 responses
├── Group 1: 3 responses (Similar context, different synonyms)
├── Group 2: 3 responses (Different context, similar wording)
└── Group 3: 3 responses (Everything different)

Embedding models: MiniLM, BGE, E5
```

---

## Step 1: Verify Response IDs

First, **identify your response IDs**. They might be:
- `R1, R2, R3, R4, R5, R6, R7, R8, R9` ← Default
- `P1, P2, P3, P4, P5, P6, P7, P8, P9` ← If using prompt IDs
- Something else? → Use those IDs

---

## Step 2: Prepare Embeddings

Get your 9 × 3 = 27 embedding files (9 responses × 3 models):

```
data/processed/embeddings/
├── R1_minilm.npy       (or R1_minilm_norm.npy)
├── R1_bge.npy
├── R1_e5.npy
├── R2_minilm.npy
├── R2_bge.npy
├── R2_e5.npy
├── ... (R3-R9)
```

**If you don't have all 27 files**, that's OK — the pipeline will skip missing files and work with what's available.

---

## Step 3: Configure Groups

Edit `config/groups.py` to match your actual response IDs:

### If using R1-R9 (default):
```python
GROUP_CONFIG = [
    {
        "name": "Similar context, different synonyms",
        "ids": ["R1", "R2", "R3"],
        "color": "steelblue"
    },
    {
        "name": "Different context, similar wording",
        "ids": ["R4", "R5", "R6"],
        "color": "tomato"
    },
    {
        "name": "Everything different",
        "ids": ["R7", "R8", "R9"],
        "color": "seagreen"
    },
]
```

### If using P1-P9 (prompt IDs):
```python
GROUP_CONFIG = [
    {
        "name": "Similar context, different synonyms",
        "ids": ["P1", "P2", "P3"],
        "color": "steelblue"
    },
    {
        "name": "Different context, similar wording",
        "ids": ["P4", "P5", "P6"],
        "color": "tomato"
    },
    {
        "name": "Everything different",
        "ids": ["P7", "P8", "P9"],
        "color": "seagreen"
    },
]
```

### If using different group names:
Just update the `"name"` field — everything else stays the same:
```python
GROUP_CONFIG = [
    {
        "name": "Your Group 1 Name",
        "ids": ["R1", "R2", "R3"],
        "color": "steelblue"
    },
    # ... rest unchanged
]
```

---

## Step 4: Run Analysis

```bash
cd /home_4TB/taqu2784/semantic_representation/aporia_isolated_work

# Activate environment (if not already active)
source ../venv/bin/activate

# Run complete pipeline
python run_all_analysis.py
```

Output goes to `plots/` directory.

---

## Step 5: Check Results

```bash
ls plots/
```

You should see PNG files like:
- `minilm_heatmap_unnorm.png` — Distance matrix for MiniLM
- `bge_heatmap_norm.png` — Distance matrix for BGE (normalized)
- `e5_heatmap_unnorm.png` — Distance matrix for E5
- `R1_pca2d_unnorm.png` — 2D PCA for response R1
- `R1_pca3d_norm.png` — 3D PCA for response R1 (normalized)
- (and so on for R2-R9)

---

## What Each Plot Shows

### Heatmaps (model-centric)
- One heatmap per model (MiniLM, BGE, E5)
- Shows pairwise distances between all 9 responses
- Block-diagonal structure = groups preserved in geometry
- Both unnormalized + normalized versions

### 2D PCA Plots (paragraph-centric, Option A)
- One plot per response (R1-R9)
- Shows where that response sits in each model's 2D PCA space
- Separate PCA per model (no cross-model distortion)
- Tight clusters = group structure reflected

### 3D PCA Plots (paragraph-centric, Option A)
- Same as 2D but with 3 principal components
- Can reveal structure not visible in 2D
- One plot per response

---

## Troubleshooting

### "FileNotFoundError: No such file or directory"
**Problem**: Embeddings not found  
**Solution**: Verify files are in `data/processed/embeddings/` with correct names
```bash
ls -la data/processed/embeddings/
# Should show files like: R1_bge.npy, R1_minilm.npy, etc.
```

### "Warning: not enough data"
**Problem**: Some models have missing files  
**Solution**: That's OK — pipeline skips missing files and continues
```bash
# If R1_bge.npy doesn't exist, R1 is just skipped for BGE
# Heatmap will still generate for other models
```

### Plots are blank or missing annotations
**Problem**: GROUP_CONFIG IDs don't match filenames  
**Solution**: Check spelling and case (R1 vs r1, P1 vs p1)
```python
# In config/groups.py, make sure IDs match your files exactly
"ids": ["R1", "R2", "R3"]  # ← must match R1_bge.npy, R1_minilm.npy, etc.
```

### ModuleNotFoundError when running
**Problem**: Python path issue  
**Solution**: Add current dir to path
```bash
export PYTHONPATH=".:$PYTHONPATH"
python run_all_analysis.py
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Run all plots | `python run_all_analysis.py` |
| Run only heatmaps | `python -m visualization.heatmap_plots` |
| Run only 2D PCA | `python -m visualization.pca_plots_2d` |
| Run only 3D PCA | `python -m visualization.pca_plots_3d` |
| Run only bar charts | `python -m visualization.bar_charts` |

---

## File Naming Reference

Expected format for your 27 embedding files:

```
{response_id}_{model_name}.npy              (unnormalized)
{response_id}_{model_name}_norm.npy         (normalized, optional)
```

### If using R1-R9:
```
R1_minilm.npy          R1_minilm_norm.npy
R1_bge.npy             R1_bge_norm.npy
R1_e5.npy              R1_e5_norm.npy
R2_minilm.npy          R2_minilm_norm.npy
... (repeat for R3-R9)
```

### If using P1-P9:
```
P1_minilm.npy          P1_minilm_norm.npy
P1_bge.npy             P1_bge_norm.npy
... (same pattern with P prefix)
```

---

## Parameters You Can Adjust

All in `config/groups.py`:

```python
# Which responses to analyze
GROUP_CONFIG = [...]

# Which models to process
EMBEDDING_MODELS = ["minilm", "bge", "e5"]

# Where embeddings are located
DATA_DIR = "data/processed/embeddings"

# Where plots are saved
OUTPUT_DIR = "plots"

# PCA random seed (for reproducibility)
PCA_RANDOM_STATE = 42
```

---

## Next Steps

1. **Verify your response IDs** (R1-R9? P1-P9?)
2. **Copy/symlink embeddings** to `data/processed/embeddings/`
3. **Edit `config/groups.py`** to match your IDs and group structure
4. **Run `python run_all_analysis.py`**
5. **Check `plots/`** for PNG outputs

That's it! All analysis is automatic.

---

## Need Help?

- **IDs don't match?** → Edit `config/groups.py` IDs
- **Missing embeddings?** → Copy files to `data/processed/embeddings/`
- **Wrong group names?** → Update `"name"` field in `GROUP_CONFIG`
- **Want to run just one model?** → Use individual commands above

Anything else → Check other documentation in this folder.
