# Workspace Setup Guide

This document explains how to use the semantic representation analysis workspace.

## Prerequisites

- Python 3.8+
- numpy, scipy, scikit-learn, matplotlib, seaborn
- pandas (optional, for data analysis)
- The venv from the main project

## Using the Python Environment

```bash
# Activate the main project's virtual environment
cd /home_4TB/taqu2784/semantic_representation
source venv/bin/activate

# Verify key packages are installed
python -c "import numpy, sklearn, matplotlib, seaborn; print('✓ All packages available')"
```

## Folder Structure & Usage

### 1. Input Data: `data/processed/embeddings/`
Place your `.npy` embedding files here:
- Format: `{response_id}_{model_name}.npy` or `{response_id}_{model_name}_norm.npy`
- Example: `R1_bge.npy`, `R2_minilm_norm.npy`, `prompts_e5.npy`

### 2. Analysis Scripts: `analysis/`
Core modules:
- `prompt_response_analysis.py` — compute prompt-response distances
- (Using visualization-based analysis: heatmaps, PCA, distance metrics)
- `cross_model_agreement.py` — compare embedder agreement (Spearman r)

### 3. Visualization Scripts: `visualization/`
Plotting modules:
- `heatmap_plots.py` — pairwise distance heatmaps (model-centric)
- `pca_plots_2d.py` — 2D PCA scatter (paragraph-centric, Option A)
- `pca_plots_3d.py` — 3D PCA scatter (paragraph-centric, Option A)
- `bar_charts.py` — prompt-response distance bar charts

### 4. Configuration: `config/`
- `groups.yaml` or `groups.py` — GROUP_CONFIG definitions
- Model names, paths, hyperparameters

### 5. Output: `plots/`
All generated plots go here:
- `.png` heatmaps, PCA scatter, bar charts
- Organized by model name and plot type

## Example: Running an Isolated Analysis

```bash
cd /home_4TB/taqu2784/semantic_representation/aporia_isolated_work

# 1. Compute heatmaps (requires .npy files in data/processed/embeddings/)
python -m visualization.heatmap_plots

# 2. Generate 2D PCA (Option A)
python -m visualization.pca_plots_2d

# 3. Generate 3D PCA (Option A)
python -m visualization.pca_plots_3d

# 4. Compute prompt-response distances
python -m analysis.prompt_response_analysis
```

## Data Location Strategy

**Option A (Recommended): Copy embeddings here**
```bash
cp /path/to/embeddings/*.npy data/processed/embeddings/
python -m visualization.heatmap_plots
```

**Option B: Symlink to main project embeddings**
```bash
ln -s /home_4TB/taqu2784/semantic_representation/data/processed/embeddings/* \
      data/processed/embeddings/
```

Symlink is faster but be aware that if embeddings change, plots will be affected.

## Configuring GROUP_CONFIG

Edit `config/groups.py` (or create it) to define your semantic groups:

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

This same config is used by all visualization scripts â change it once, all plots update automatically.

## Troubleshooting

**ImportError: No module named 'visualization'**
```bash
# Add current folder to Python path
export PYTHONPATH="/home_4TB/taqu2784/semantic_representation/aporia_isolated_work:$PYTHONPATH"
python -m visualization.heatmap_plots
```

**FileNotFoundError: No such file or directory: 'data/processed/embeddings/'**
```bash
# Make sure embeddings are copied/symlinked to the data folder
ls -la aporia_isolated_work/data/processed/embeddings/
```

**Matplotlib: No Display**
If running headless, matplotlib will auto-switch to file-based backend (PNG output works fine).

## Next Steps

1. Copy/symlink your embeddings to `data/processed/embeddings/`
2. Update `config/groups.py` with your actual group definitions
3. Run visualization scripts
4. Check `plots/` for generated PNG files
5. Once validated, integrate results back to main project if needed

---

**Status**: Ready to use. No changes made to main project.
