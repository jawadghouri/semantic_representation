from pathlib import Path
import numpy as np

from visualization.bar_chart import create_bar_chart
from visualization.heatmap import create_heatmap
from visualization.umap_plot import create_umap

SIMILARITY_DIR = Path("results/similarities")
EMBEDDING_DIR = Path("data/processed/embeddings")
BAR_DIR = Path("results/figures/barplots")
HEATMAP_DIR = Path("results/figures/heatmaps")
UMAP_DIR = Path("results/figures/umap")


BAR_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
UMAP_DIR.mkdir(parents=True, exist_ok=True)


EMBEDDING_MODELS = ["minilm","e5","bge"]


for embedding_model in EMBEDDING_MODELS:

    print(f"\nVisualizing {embedding_model}")

    similarity_file = (SIMILARITY_DIR / f"{embedding_model}.json")
    
    # BAR CHART
    create_bar_chart(
        similarity_file,
        BAR_DIR /
        f"{embedding_model}_bar.png"
    )

    # HEATMAP
    create_heatmap(
        similarity_file,
        HEATMAP_DIR /
        f"{embedding_model}_heatmap.png"
    )

    # UMAP
    embeddings = []

    labels = []

    prompt_file = (
        EMBEDDING_DIR /
        f"prompt_{embedding_model}.npy"
    )

    embeddings.append(
        np.load(prompt_file)
    )

    labels.append(
        "prompt"
    )

    for file in EMBEDDING_DIR.glob(f"*_{embedding_model}.npy"):

        model_name = file.stem.replace(f"_{embedding_model}","")

        if model_name == "prompt":
            continue

        embeddings.append(np.load(file))

        labels.append(model_name)

    embeddings = np.vstack(embeddings)

    create_umap(
        embeddings,
        labels,
        UMAP_DIR /
        f"{embedding_model}_umap.png"
    )

    print(f"Finished {embedding_model}")