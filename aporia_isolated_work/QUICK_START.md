# Quick Start Guide — Semantic Representation Analysis

## In 5 Minutes

### 1. Activate Environment
```bash
cd /home_4TB/taqu2784/semantic_representation
source venv/bin/activate
cd aporia_isolated_work
```

### 2. Add Your Data
Copy or symlink your embedding `.npy` files to `data/processed/embeddings/`:

**Option A: Copy files**
```bash
cp /path/to/your/embeddings/*.npy data/processed/embeddings/
```

**Option B: Symlink**
```bash
ln -s /path/to/your/embeddings/* data/processed/embeddings/
```

### 3. Configure Groups
Edit `config/groups.py` and set `GROUP_CONFIG` to match your experiment:

```python
GROUP_CONFIG = [
    {"name": "Group 1", "ids": ["R1", "R2", "R3"], "color": "steelblue"},
    {"name": "Group 2", "ids": ["R4", "R5", "R6"], "color": "tomato"},
]
```

### 4. Run Analysis
```bash
python run_all_analysis.py
```

### 5. View Results
```bash
ls plots/
# Open PNG files in your image viewer
```

---

## What Each Module Does

| Module | Purpose | Output |
|--------|---------|--------|
| `heatmap_plots.py` | Model-centric distance matrices | `{model}_heatmap_{norm/unnorm}.png` |
| `pca_plots_2d.py` | 2D PCA (paragraph-centric) | `{R_id}_pca2d_{norm/unnorm}.png` |
| `pca_plots_3d.py` | 3D PCA (paragraph-centric) | `{R_id}_pca3d_{norm/unnorm}.png` |
| `bar_charts.py` | Prompt-response distances | `{llm}_{model}_prompt_response.png` |

---

## Troubleshooting

**Q: "ModuleNotFoundError: No module named 'visualization'"**
```bash
export PYTHONPATH="/home_4TB/taqu2784/semantic_representation/aporia_isolated_work:$PYTHONPATH"
python run_all_analysis.py
```

**Q: "FileNotFoundError: No such file or directory: 'data/processed/embeddings/'"**
Make sure embeddings are copied/symlinked to `data/processed/embeddings/`.

**Q: Plots are blank or incomplete**
Check that:
- File naming matches config (e.g., `R1_bge.npy`, not `R1-bge.npy`)
- At least 2 embeddings per model exist
- GROUP_CONFIG IDs match your data files

**Q: Large file sizes / slow performance**
- 768D embeddings (BGE/E5) are large; this is normal
- PCA on 3D can be slower than 2D; this is expected
- Heatmaps scale with O(N²) data; 100+ responses slow down annotation

---

## File Naming Convention

Embeddings should follow this pattern:
```
{response_id}_{model_name}.npy              # unnormalized
{response_id}_{model_name}_norm.npy         # normalized

# Examples:
R1_bge.npy
R1_bge_norm.npy
R1_minilm.npy
R1_minilm_norm.npy
R1_e5.npy
R1_e5_norm.npy
prompts_bge.npy                             # optional, for prompt-response analysis
```

---

## Configuration Reference

### config/groups.py
```python
GROUP_CONFIG = [
    {
        "name": "Descriptive group name",
        "ids": ["ID1", "ID2", "ID3", ...],
        "color": "steelblue"  # matplotlib color name
    },
    ...
]

EMBEDDING_MODELS = ["minilm", "bge", "e5"]  # models to process
DATA_DIR = "data/processed/embeddings"       # input dir
OUTPUT_DIR = "plots"                         # output dir
```

### Visualization Functions

All visualization modules support:
- **Unnormalized** and **normalized** embeddings
- **Flexible GROUP_CONFIG** with any number of groups/sizes
- **Automatic skipping** of missing files (no crashes)
- **High-resolution output** (200 DPI PNG)

---

## Next Steps

Once familiar with quick start:
1. Read `WORKSPACE_SETUP.md` for detailed configuration options
2. Explore `analysis/` for distance metrics and comparison functions
3. Review full context in `../semantic-representation-full-context.md`

---

## Remember

✓ This folder is completely isolated — no changes to main project  
✓ All outputs stay in this folder — easy to clean up  
✓ Safe to experiment and iterate  
✓ Ready to integrate results whenever you want
