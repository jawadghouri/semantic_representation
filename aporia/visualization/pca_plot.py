import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca(embeddings: np.ndarray, labels: list, title: str, output_path: str):
    """2D PCA scatter of response embeddings colored G (green) / H (red)."""
    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))

    color_map = {"G": "green", "H": "red"}
    marker_map = {"G": "o", "H": "X"}

    for cls, color in color_map.items():
        idx = [i for i, l in enumerate(labels) if l == cls]
        if not idx:
            continue
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1] if coords.shape[1] > 1 else np.zeros(len(idx)),
            c=color,
            marker=marker_map[cls],
            s=80,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            label=f"{'Genuine' if cls == 'G' else 'Hallucinated'} (n={len(idx)})",
        )

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)" if len(var) > 1 else "PC2")
    ax.set_title(title, fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved PCA plot -> {output_path}")
