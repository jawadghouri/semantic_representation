# Semantic Representation Analysis Workspace

**Purpose**: Clean, isolated folder for embedding analysis, visualization, and semantic representation work. Does NOT interfere with the main project structure.

**Structure**:
```
aporia_isolated_work/
├── README.md                          (this file)
├── WORKSPACE_SETUP.md                 (workspace initialization guide)
├── data/
│   └── processed/
│       └── embeddings/                (isolated embedding outputs)
├── analysis/                          (Distance metrics & analysis scripts)
├── visualization/                     (plotting & visualization code)
├── plots/                             (output plots, heatmaps, PCA)
└── config/                            (configuration files)
```

## Quick Start

1. **Initial setup**: Read `WORKSPACE_SETUP.md`
2. **Run analysis**: Scripts will save outputs to this isolated folder only
3. **No git commits**: Changes here don't affect main project

## Key Features

- **Isolated data**: All embeddings, plots, and results stay in this folder
- **Flexible grouping**: GROUP_CONFIG-driven analysis (works with any N groups/sizes)
- **Multiple visualization types**: Heatmaps, 2D PCA, 3D PCA, bar charts
- **Model-agnostic**: Works with MiniLM, BGE, E5 (or any embedder with .npy outputs)

## When Ready to Integrate

Once tested here and validated, results can be merged back into the main project without any disruption.
