"""
3D PCA plot with ALL embeddings (3 LLMs × 3 embedders) in one plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from pathlib import Path
import os

# LLM colors
LLM_COLORS = {
    "llama": "steelblue",
    "mistral": "tomato",
    "phi": "seagreen",
}

# Embedder markers
EMBEDDER_MARKERS = {
    "minilm": "o",
    "bge": "s",
    "e5": "^",
}

def load_embedding(data_dir, llm_name, embedder_name, normalized=False):
    """Load single embedding file."""
    suffix = "_norm" if normalized else ""
    fname = f"{llm_name}_{embedder_name}{suffix}.npy"
    fpath = Path(data_dir) / fname

    if fpath.exists():
        vec = np.load(fpath)
        if vec.ndim > 1:
            vec = vec.flatten()
        return vec
    return None

def plot_all_embeddings_3d(
    data_dir="data/processed/embeddings",
    output_dir="plots",
    llms=["llama", "mistral", "phi"],
    embedders=["minilm", "bge", "e5"],
    normalized=False,
    elev=20,
    azim=-60
):
    """Plot all 9 embeddings in one 3D PCA plot."""
    os.makedirs(output_dir, exist_ok=True)

    embeddings = []
    labels = []
    colors = []
    markers = []

    # Load all embeddings
    for llm in llms:
        for embedder in embedders:
            vec = load_embedding(data_dir, llm, embedder, normalized)
            if vec is not None:
                embeddings.append(vec)
                labels.append(f"{llm.upper()}\n{embedder.upper()}")
                colors.append(LLM_COLORS[llm])
                markers.append(EMBEDDER_MARKERS[embedder])

    if len(embeddings) < 3:
        print("⚠ Not enough embeddings for 3D PCA")
        return

    # Pad embeddings to same size (MinilLM 384D vs BGE/E5 768D)
    max_dim = max(emb.shape[0] for emb in embeddings)
    padded_embeddings = []
    for emb in embeddings:
        if emb.shape[0] < max_dim:
            padded = np.concatenate([emb, np.zeros(max_dim - emb.shape[0])])
        else:
            padded = emb
        padded_embeddings.append(padded)

    # Fit PCA
    matrix = np.stack(padded_embeddings)
    pca = PCA(n_components=3, random_state=42)
    reduced = pca.fit_transform(matrix)
    explained = pca.explained_variance_ratio_ * 100

    # Create 3D plot
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # Plot each embedding
    for i, (point, label, color, marker) in enumerate(zip(reduced, labels, colors, markers)):
        ax.scatter(point[0], point[1], point[2], c=color, marker=marker, s=250,
                  alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.text(point[0], point[1], point[2], f"  {label}", fontsize=9,
               fontweight='bold', color=color)

    # Legend for LLMs
    llm_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LLM_COLORS[llm],
               markersize=10, label=llm.upper())
        for llm in llms
    ]

    # Legend for embedders
    embedder_legend = [
        Line2D([0], [0], marker=EMBEDDER_MARKERS[emb], color='gray',
               markerfacecolor='gray', markersize=10, label=emb.upper())
        for emb in embedders
    ]

    ax.legend(handles=llm_legend + embedder_legend, loc='upper left',
             fontsize=10, title="LLMs / Embedders", title_fontsize=11)

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=11, fontweight='bold')
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=11, fontweight='bold')
    ax.set_zlabel(f"PC3 ({explained[2]:.1f}%)", fontsize=11, fontweight='bold')

    suffix_text = "Normalized" if normalized else "Unnormalized"
    ax.set_title(
        f"All Embeddings in 3D PCA Space ({suffix_text})\n"
        f"3 LLMs × 3 Embedders = 9 points | Total Variance: {sum(explained):.1f}%",
        fontsize=13, fontweight='bold', pad=20
    )

    plt.tight_layout()

    filename = f"all_embeddings_pca3d_{suffix_text.lower()}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("3D PCA - ALL EMBEDDINGS IN ONE PLOT")
    print("="*70)

    print("\n[1/2] Plotting UNNORMALIZED...")
    plot_all_embeddings_3d(normalized=False)

    print("\n[2/2] Plotting NORMALIZED...")
    plot_all_embeddings_3d(normalized=True)

    print("\n✓ Done!")
