import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances


def compute_distance_matrix(embeddings):
    """Computes raw, unnormalized pairwise Euclidean distances."""
    return euclidean_distances(embeddings)


def save_euclidean_heatmap(distance_matrix, output_image_path):
    """Normalizes the display scale (0 to Max) for a clean, intuitive

    Seaborn heatmap visualization without altering the underlying raw data.
    """
    plt.figure(figsize=(8, 6))

    # Scale the colormap boundaries based on your actual data limits
    max_dist = distance_matrix.max()

    sns.heatmap(
        distance_matrix,
        cmap="rocket_r",  # '_r' reverses it: Dark/Warm colors = closer together (0 distance)
        vmin=0.0,
        vmax=max_dist,
        annot=False,  # Set to True if visualizing small matrices (e.g., 5x5)
        cbar_kws={"label": "Euclidean Distance (L2)"},
    )
    plt.title("Pairwise Euclidean Distance Matrix")
    plt.savefig(output_image_path, bbox_inches="tight")
    plt.close()