import matplotlib.pyplot as plt
import umap


def generate_umap(embeddings):
    """Reduces high-dimensional embeddings (384 or 1024 dims) down to a

    2D coordinate array for visualization.
    """
    reducer = umap.UMAP(random_state=42)
    return reducer.fit_transform(embeddings)


def save_umap_plot(reduced, labels, output_file):
    """Takes the 2D compressed coordinates and saves a clean, color-coded scatter

    plot using your KMeans cluster labels.
    """
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.8, s=50
    )

    # Clean up the chart presentation for your report
    plt.colorbar(scatter, label="KMeans Cluster ID")
    plt.title("UMAP 2D Projection of Semantic Space", fontsize=14, pad=15)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.savefig(output_file, bbox_inches="tight", dpi=150)
    plt.close()