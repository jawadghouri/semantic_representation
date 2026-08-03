import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aporia.analysis.pairwise import compute_pairwise_distances


def plot_heatmap(embeddings: np.ndarray, labels: list, title: str, output_path: str):
    """
    N×N pairwise Euclidean distance heatmap annotated with G/H labels.
    Rows and columns sorted so G responses appear first.
    """
    n = len(labels)
    dist_matrix = compute_pairwise_distances(embeddings)

    # Sort: G first, H second
    g_idx = [i for i, l in enumerate(labels) if l == "G"]
    h_idx = [i for i, l in enumerate(labels) if l == "H"]
    order = g_idx + h_idx

    sorted_matrix = dist_matrix[np.ix_(order, order)]
    sorted_labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        sorted_matrix,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=sorted_labels,
        yticklabels=sorted_labels,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Euclidean Distance"},
    )

    n_g = len(g_idx)
    if 0 < n_g < n:
        ax.axhline(n_g, color="blue", linewidth=2)
        ax.axvline(n_g, color="blue", linewidth=2)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Response")
    ax.set_ylabel("Response")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved heatmap -> {output_path}")
