# START HERE — Semantic Representation Analysis Workspace

**Date**: 2026-08-18  
**Status**: Ready to use  
**Impact on main project**: NONE — fully isolated  
**Configuration**: 9 responses × 3 semantic groups

---

## What Is This?

A clean, separate folder for **semantic representation analysis and visualization** that:
- ✓ Does **NOT** interfere with your existing project work
- ✓ Provides **flexible, configurable** analysis pipeline
- ✓ Supports **multiple visualization types**: heatmaps, 2D PCA, 3D PCA, bar charts
- ✓ Works with **any number of semantic groups** (not hardcoded to specific structure)
- ✓ Handles **normalized and unnormalized** embeddings automatically

---

## Folder Structure

```
aporia_isolated_work/
├── 00_START_HERE.md                (this file)
├── README.md                        (overview & features)
├── QUICK_START.md                   (5-minute setup guide)
├── WORKSPACE_SETUP.md               (detailed configuration)
├── run_all_analysis.py              (master execution script)
│
├── config/
│   ├── __init__.py
│   └── groups.py                    (CONFIGURATION FILE — edit this!)
│
├── analysis/
│   ├── __init__.py
│   └── prompt_response_analysis.py  (prompt↔response distance metrics)
│
├── visualization/
│   ├── __init__.py
│   ├── heatmap_plots.py             (model-centric distance matrices)
│   ├── pca_plots_2d.py              (paragraph-centric 2D PCA, Option A)
│   ├── pca_plots_3d.py              (paragraph-centric 3D PCA, Option A)
│   └── bar_charts.py                (prompt-response distance bars)
│
├── data/
│   └── processed/
│       └── embeddings/              (← put your .npy files here)
│
└── plots/                           (← outputs go here)
    ├── {model}_heatmap_*.png
    ├── {R_id}_pca2d_*.png
    ├── {R_id}_pca3d_*.png
    └── ...
```

---

## Getting Started (3 Steps)

### Step 1: Activate Environment

```bash
cd /home_4TB/taqu2784/semantic_representation
source venv/bin/activate
cd aporia_isolated_work
```

### Step 2: Add Embeddings

Copy or symlink your embedding `.npy` files to `data/processed/embeddings/`:

```bash
# Copy example
cp ~/my_embeddings/*.npy data/processed/embeddings/

# Or symlink (faster, watch for external changes)
ln -s ~/my_embeddings/*.npy data/processed/embeddings/
```

**File naming** should match this pattern:
```
{response_id}_{model_name}.npy           # unnormalized
{response_id}_{model_name}_norm.npy      # normalized (optional)

Examples:
  R1_bge.npy
  R1_minilm.npy
  R1_e5.npy
  prompts_bge.npy                        # optional
```

### Step 3: Configure & Run

Edit `config/groups.py` to match your experiment structure:

```python
GROUP_CONFIG = [
    {
        "name": "Your Group 1",
        "ids": ["R1", "R2", "R3"],
        "color": "steelblue"
    },
    {
        "name": "Your Group 2",
        "ids": ["R4", "R5", "R6"],
        "color": "tomato"
    },
    # add as many groups as you need, any sizes
]

EMBEDDING_MODELS = ["minilm", "bge", "e5"]  # update if different
```

Then run the full analysis:

```bash
python run_all_analysis.py
```

Check `plots/` for PNG outputs.

---

## What Gets Generated?

### 1. Heatmaps (Model-Centric)
One heatmap per embedding model showing pairwise Euclidean distances between all responses.
- **File**: `{model}_heatmap_{norm/unnorm}.png`
- **Interpretation**: Block-diagonal structure = group structure in geometry

### 2. 2D PCA Plots (Paragraph-Centric, Option A)
For each response, shows how it sits in each model's 2D PCA space.
- **File**: `{R_id}_pca2d_{norm/unnorm}.png`
- **Interpretation**: Tight clusters = model captures semantic grouping

### 3. 3D PCA Plots (Paragraph-Centric, Option A)
Same as 2D but with 3 principal components (can reveal hidden structure).
- **File**: `{R_id}_pca3d_{norm/unnorm}.png`
- **Interpretation**: More variance captured than 2D, better spatial picture

### 4. Prompt-Response Distance Bar Charts
For each (LLM, embedder) pair, shows distance from prompt to each response.
- **File**: `{llm}_{model}_prompt_response{norm/unnorm}.png`
- **Interpretation**: Shorter bars = response stays on-topic

---

## Key Features

### ✓ Flexible Grouping
- `GROUP_CONFIG` supports **any number** of groups, **any sizes**, **any names**
- Change once in `config/groups.py`, all plots update automatically
- Works for your current experiment + future experiments

### ✓ Normalized & Unnormalized
- Automatically processes both versions
- No manual file management needed
- Just name files with `_norm` suffix for normalized versions

### ✓ Robust Error Handling
- Missing files are skipped with warnings (no crashes)
- Not enough data? Script continues gracefully
- Clear error messages if something goes wrong

### ✓ High-Quality Output
- All plots saved as 200 DPI PNG (publication-ready)
- Proper labeling, legends, group colors
- Automatic figure sizing based on data size

---

## Configuration Quick Reference

**File**: `config/groups.py`

```python
# Set active config (choose one, or define your own)
GROUP_CONFIG = GROUP_CONFIG_R1_R18        # 18 responses, 3 groups (example)
# GROUP_CONFIG = GROUP_CONFIG_9_PROMPTS   # 9 responses, 3 groups (example)
# GROUP_CONFIG = GROUP_CONFIG_CUSTOM      # template for your own

# Models to process
EMBEDDING_MODELS = ["minilm", "bge", "e5"]

# Paths (relative to workspace root)
DATA_DIR = "data/processed/embeddings"
OUTPUT_DIR = "plots"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError" | Run `export PYTHONPATH="..:$PYTHONPATH"` first |
| No embeddings found | Check files are in `data/processed/embeddings/` with correct names |
| Plots are blank | Verify GROUP_CONFIG IDs match your file names exactly |
| "Not enough data" | Need at least 2 embeddings per model for heatmap; 3+ for 3D PCA |
| Matplotlib headless | Auto-switches to file backend; PNG output works fine |

For more: see `WORKSPACE_SETUP.md` and `QUICK_START.md`

---

## Key Modules

### `prompt_response_analysis.py`
**Prompt-anchor distance metrics**
```python
from analysis.prompt_response_analysis import compute_prompt_response_distances

distances = compute_prompt_response_distances(prompt_embedding, response_embeddings)
stats = compute_statistics(distances)
ranked = rank_responses_by_proximity(distances, response_ids)
```

### Visualization Modules
```python
from visualization.heatmap_plots import plot_all_models
from visualization.pca_plots_2d import plot_paragraph_centric_2d
from visualization.pca_plots_3d import plot_paragraph_centric_3d
from visualization.bar_charts import plot_all_combinations
```

All support `GROUP_CONFIG`-driven flexibility.

---

## Next Steps

1. **Now**: Copy embeddings to `data/processed/embeddings/`
2. **Then**: Edit `config/groups.py` with your group definitions
3. **Run**: `python run_all_analysis.py`
4. **Review**: Check PNG files in `plots/`
5. **Iterate**: Adjust `GROUP_CONFIG` and re-run as needed
6. **Integrate**: When ready, merge results to main project (zero disruption)

---

## Important Notes

- ✓ **Completely isolated** — main project untouched
- ✓ **No git tracking** — this folder won't appear in commits
- ✓ **Easy cleanup** — entire folder can be deleted without impact
- ✓ **Safe to experiment** — iterate on configuration freely
- ✓ **Ready to integrate** — results can be moved/merged whenever you want

---

## Questions?

- **How do I configure groups?** → See `QUICK_START.md` Step 3
- **What's in each plot?** → See "What Gets Generated?" section above
- **Can I use different embedders?** → Yes, just update `EMBEDDING_MODELS` in `config/groups.py`
- **How do I run individual modules?** → `python -m visualization.heatmap_plots` (etc.)
- **Detailed setup?** → Read `WORKSPACE_SETUP.md`

---

**Ready?** Start with `QUICK_START.md` → 5 minutes to first plots!
