"""
Flexible GROUP_CONFIG for semantic grouping and visualization.

Update this to match your actual experiment structure.
Works with any number of groups, any group sizes, any names/colors.
"""

# Current live pipeline: 3 LLMs × 3 embedders = 9 response combinations
# Organized by LLM (semantic grouping)
GROUP_CONFIG = [
    {
        "name": "Llama LLM",
        "ids": ["llama"],
        "color": "steelblue"
    },
    {
        "name": "Mistral LLM",
        "ids": ["mistral"],
        "color": "tomato"
    },
    {
        "name": "Phi LLM",
        "ids": ["phi"],
        "color": "seagreen"
    },
]

# Embedding models to process
EMBEDDING_MODELS = ["minilm", "bge", "e5"]

# Data directory (relative to this isolated workspace)
DATA_DIR = "data/processed/embeddings"

# Output directory for plots
OUTPUT_DIR = "plots"

# PCA hyperparameters
PCA_RANDOM_STATE = 42
UMAP_RANDOM_STATE = 42

# Visualization defaults
DEFAULT_DPI = 200
DEFAULT_FIGSIZE_HEATMAP = (12, 10)
DEFAULT_FIGSIZE_PCA_2D = (6, 5)
DEFAULT_FIGSIZE_PCA_3D = (6.5, 6)

# Color palette for consistency
COLOR_PALETTE = {
    "neutral": "#cccccc",
    "highlight": "#2ca02c",
    "error": "#d62728",
}
