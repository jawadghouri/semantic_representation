# How to Run & View PCA Analysis

Complete guide for running the analysis pipeline and viewing results.

---

## Quick Start (Copy-Paste Commands)

```bash
# 1. Navigate to workspace
cd /home_4TB/taqu2784/semantic_representation/aporia_isolated_work

# 2. Activate Python environment
source ../venv/bin/activate

# 3. Run complete analysis pipeline
python run_all_analysis.py

# 4. View plots
ls -lh plots/
```

That's it! All 9 plots are generated in `plots/` directory.

---

## Step-by-Step Breakdown

### Step 1: Navigate to Workspace
```bash
cd /home_4TB/taqu2784/semantic_representation/aporia_isolated_work
```

### Step 2: Activate Virtual Environment
```bash
source ../venv/bin/activate
```
You should see `(venv)` prefix in your terminal.

### Step 3: Run Complete Pipeline
```bash
python run_all_analysis.py
```

This runs (in order):
1. Heatmap generation (model-centric)
2. 2D PCA plots (paragraph-centric)
3. 3D PCA plots (paragraph-centric)
4. Bar charts (prompt-response distances)

**Output** shows progress:
```
════════════════════════════════════════════════════════════════════════════
  SEMANTIC REPRESENTATION ANALYSIS PIPELINE
════════════════════════════════════════════════════════════════════════════

✓ All plots generated successfully!
✓ Check 'plots/' for PNG files
```

### Step 4: View Generated Plots
```bash
# List all plots
ls -lh plots/

# Check specific plots
ls plots/*pca*.png      # PCA plots only
ls plots/*heatmap*.png  # Heatmaps only
```

---

## Running Individual Modules

If you only want specific plots:

### Just Heatmaps
```bash
python -m visualization.heatmap_plots
```

### Just 2D PCA
```bash
python -m visualization.pca_plots_2d
```

### Just 3D PCA
```bash
python -m visualization.pca_plots_3d
```

### Just Bar Charts
```bash
python -m visualization.bar_charts
```

---

## Viewing the Plots

### Option 1: Direct File Access
```bash
# On Linux/Mac with GUI
cd plots/
# Double-click any .png file to open
```

### Option 2: List Plots
```bash
cd /home_4TB/taqu2784/semantic_representation/aporia_isolated_work/plots/
ls -lh
```

### Option 3: File Information
```bash
file plots/*.png        # Verify files are valid PNGs
file plots/*pca2d*.png  # Check specific files
```

### Option 4: Copy to Local Machine
```bash
# From local machine
scp username@server:/home_4TB/taqu2784/semantic_representation/aporia_isolated_work/plots/*.png ~/Downloads/
```

---

## What You'll See

### 2D PCA Plot Example
**File**: `llama_pca2d_unnorm.png`

```
        PCA 2D - Llama LLM
        
    │     ▲ E5 (deepskyblue)
    │        
    │        ● MinilLM (purple)
PC2 │        
    │            ● BGE (orange)
    │
    └──────────────────────────> PC1
    
    X-axis: PC1 (First principal component)
    Y-axis: PC2 (Second principal component)
    Each point = one embedder
    Distances = how different embeddings are
```

**What to look for**:
- Are the 3 points close together or spread out?
- Close = embedders produce similar embeddings
- Spread = embedders produce different embeddings
- Compare across llama_pca2d, mistral_pca2d, phi_pca2d

### 3D PCA Plot Example
**File**: `llama_pca3d_unnorm.png`

```
Same as 2D but with Z-axis (depth):
    - More variance captured
    - Shows PC1, PC2, PC3
    - 3D visualization
    - Rotatable in some viewers
```

### Heatmap Example
**File**: `minilm_heatmap_unnorm.png`

```
        MinilLM Distance Matrix
        
         llama  mistral  phi
llama     0      15.2    18.5  ← Llama row
mistral  15.2    0       14.1  ← Mistral row
phi      18.5   14.1     0     ← Phi row

Blue (low values) = similar
Red (high values) = different

Groups highlighted with colors:
- Blue border: Llama group
- Red border: Mistral group
- Green border: Phi group
```

---

## Configuration Used

File: `config/groups.py`

```python
GROUP_CONFIG = [
    {"name": "Llama LLM", "ids": ["llama"], "color": "steelblue"},
    {"name": "Mistral LLM", "ids": ["mistral"], "color": "tomato"},
    {"name": "Phi LLM", "ids": ["phi"], "color": "seagreen"},
]

EMBEDDING_MODELS = ["minilm", "bge", "e5"]
```

---

## Data Summary

```
Input Data:
├── llama_minilm.npy
├── llama_bge.npy
├── llama_e5.npy
├── mistral_minilm.npy
├── mistral_bge.npy
├── mistral_e5.npy
├── phi_minilm.npy
├── phi_bge.npy
└── phi_e5.npy

Total: 9 embeddings (3 LLMs × 3 embedders)

Generated Plots: 9 PNG files
├── 3 × 2D PCA plots
├── 3 × 3D PCA plots
└── 3 × Heatmaps
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'visualization'"
**Solution**: Set PYTHONPATH
```bash
export PYTHONPATH="/home_4TB/taqu2784/semantic_representation/aporia_isolated_work:$PYTHONPATH"
python run_all_analysis.py
```

### No plots generated
**Solution**: Check if embeddings exist
```bash
ls -la data/processed/embeddings/
# Should show 12 .npy files (9 embeddings + 3 prompts)
```

### Plots are empty/blank
**Solution**: Verify configuration
```bash
# Check config/groups.py has correct IDs
cat config/groups.py
# IDs should be: ["llama"], ["mistral"], ["phi"]
```

### "FileNotFoundError" errors
**Solution**: Check working directory
```bash
pwd  # Should be: /home_4TB/taqu2784/semantic_representation/aporia_isolated_work
```

---

## Complete Example Run

```bash
$ cd /home_4TB/taqu2784/semantic_representation/aporia_isolated_work
$ source ../venv/bin/activate
(venv) $ python run_all_analysis.py

════════════════════════════════════════════════════════════════════════════
  SEMANTIC REPRESENTATION ANALYSIS PIPELINE
════════════════════════════════════════════════════════════════════════════

✓ Environment validated

[1/4] STEP 1: Heatmap Analysis (Model-Centric)
→ Generating unnormalized heatmaps...
✓ Heatmap analysis complete

[2/4] STEP 2: 2D PCA Plots (Paragraph-Centric, Option A)
→ Generating unnormalized 2D PCA plots...
✓ Fitted PCA for 3 models
✓ Saved: plots/llama_pca2d_unnorm.png
✓ Saved: plots/mistral_pca2d_unnorm.png
✓ Saved: plots/phi_pca2d_unnorm.png
✓ 2D PCA analysis complete

[3/4] STEP 3: 3D PCA Plots (Paragraph-Centric, Option A)
→ Generating unnormalized 3D PCA plots...
✓ Fitted 3D PCA for 3 models
✓ Saved: plots/llama_pca3d_unnorm.png
✓ Saved: plots/mistral_pca3d_unnorm.png
✓ Saved: plots/phi_pca3d_unnorm.png
✓ 3D PCA analysis complete

[4/4] STEP 4: Bar Chart Analysis
→ Generating unnormalized bar charts...
✓ Bar chart analysis complete

════════════════════════════════════════════════════════════════════════════
EXECUTION SUMMARY
════════════════════════════════════════════════════════════════════════════

✓ PASS  Heatmap Analysis
✓ PASS  2D PCA Plots
✓ PASS  3D PCA Plots
✓ PASS  Bar Chart Analysis

✓ All plots saved to: plots/

(venv) $ ls -lh plots/
total 1.7M
-rw-r--r-- 1 user  158K  2024-08-18 19:11 llama_pca2d_unnorm.png
-rw-r--r-- 1 user  407K  2024-08-18 19:11 llama_pca3d_unnorm.png
[... more files ...]
```

---

## Next Steps After Viewing Plots

1. **Analyze results**:
   - Do embedders cluster by type?
   - Are LLMs well-separated?
   - What do distances tell you?

2. **Adjust if needed**:
   - Edit `config/groups.py`
   - Change group names or colors
   - Rerun: `python run_all_analysis.py`

3. **Generate more analysis**:
   - Run bar charts for prompt-response distances
   - Run individual modules for specific plots
   - Change PCA parameters in code if needed

---

## Remember

✅ Plots are in: `aporia_isolated_work/plots/`
✅ 9 PNG files total (2D PCA, 3D PCA, heatmaps)
✅ No normalized versions (need _norm.npy files)
✅ Ready to interpret and share
