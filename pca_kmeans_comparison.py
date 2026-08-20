import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

EMBEDDING_MODELS = ["e5", "bge", "minilm"]
N_CLUSTERS = 3
COLORS_CLUSTERS = plt.cm.Set3(np.linspace(0, 1, 10))

def load_embeddings(embedding_type="unnormalized"):
    """Load all embeddings grouped by model"""
    embeddings_by_model = {"e5": [], "bge": [], "minilm": []}
    response_ids = []

    if embedding_type == "unnormalized":
        base_path = "/home_4TB/taqu2784/semantic_representation/progress_meeting/embeddings"
        pattern = "R*_*.npy"
    else:
        base_path = "/home_4TB/taqu2784/semantic_representation/progress_meeting/embeddings_norm"
        pattern = "R*_*_norm.npy"

    files = sorted(Path(base_path).glob(pattern))

    for file in files:
        parts = file.stem.replace("_norm", "").split("_")
        response_id = parts[0]
        model = parts[1]

        if response_id not in response_ids:
            response_ids.append(response_id)

        if model in embeddings_by_model:
            embedding = np.load(file)
            embeddings_by_model[model].append(embedding)

    for model in embeddings_by_model:
        if embeddings_by_model[model]:
            embeddings_by_model[model] = np.array(embeddings_by_model[model])

    return embeddings_by_model, sorted(response_ids)

def plot_2d_comparison(unnorm_data, norm_data, title, output_path):
    """Create side-by-side 2D PCA comparison with k-means clustering"""
    # K-means clustering
    kmeans_unnorm = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    kmeans_norm = KMeans(n_clusters=N_CLUSTERS, random_state=42)

    clusters_unnorm = kmeans_unnorm.fit_predict(unnorm_data)
    clusters_norm = kmeans_norm.fit_predict(norm_data)

    # PCA
    pca_unnorm = PCA(n_components=2)
    pca_norm = PCA(n_components=2)

    reduced_unnorm = pca_unnorm.fit_transform(unnorm_data)
    reduced_norm = pca_norm.fit_transform(norm_data)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Unnormalized
    for cluster in range(N_CLUSTERS):
        mask = clusters_unnorm == cluster
        axes[0].scatter(reduced_unnorm[mask, 0], reduced_unnorm[mask, 1],
                       label=f"Cluster {cluster}", s=150, alpha=0.6,
                       color=COLORS_CLUSTERS[cluster], edgecolors='black', linewidth=1)

    axes[0].set_xlabel(f"PC1 ({pca_unnorm.explained_variance_ratio_[0]:.2%})")
    axes[0].set_ylabel(f"PC2 ({pca_unnorm.explained_variance_ratio_[1]:.2%})")
    axes[0].set_title(f"{title} - Unnormalized", fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Normalized
    for cluster in range(N_CLUSTERS):
        mask = clusters_norm == cluster
        axes[1].scatter(reduced_norm[mask, 0], reduced_norm[mask, 1],
                       label=f"Cluster {cluster}", s=150, alpha=0.6,
                       color=COLORS_CLUSTERS[cluster], edgecolors='black', linewidth=1)

    axes[1].set_xlabel(f"PC1 ({pca_norm.explained_variance_ratio_[0]:.2%})")
    axes[1].set_ylabel(f"PC2 ({pca_norm.explained_variance_ratio_[1]:.2%})")
    axes[1].set_title(f"{title} - Normalized", fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_3d_comparison(unnorm_data, norm_data, title, output_path):
    """Create side-by-side 3D PCA comparison with k-means clustering"""
    # K-means clustering
    kmeans_unnorm = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    kmeans_norm = KMeans(n_clusters=N_CLUSTERS, random_state=42)

    clusters_unnorm = kmeans_unnorm.fit_predict(unnorm_data)
    clusters_norm = kmeans_norm.fit_predict(norm_data)

    # PCA
    pca_unnorm = PCA(n_components=3)
    pca_norm = PCA(n_components=3)

    reduced_unnorm = pca_unnorm.fit_transform(unnorm_data)
    reduced_norm = pca_norm.fit_transform(norm_data)

    # Plot
    fig = plt.figure(figsize=(16, 6))

    # Unnormalized
    ax1 = fig.add_subplot(121, projection='3d')
    for cluster in range(N_CLUSTERS):
        mask = clusters_unnorm == cluster
        ax1.scatter(reduced_unnorm[mask, 0], reduced_unnorm[mask, 1], reduced_unnorm[mask, 2],
                   label=f"Cluster {cluster}", s=100, alpha=0.6,
                   color=COLORS_CLUSTERS[cluster], edgecolors='black', linewidth=0.5)

    ax1.set_xlabel(f"PC1 ({pca_unnorm.explained_variance_ratio_[0]:.2%})")
    ax1.set_ylabel(f"PC2 ({pca_unnorm.explained_variance_ratio_[1]:.2%})")
    ax1.set_zlabel(f"PC3 ({pca_unnorm.explained_variance_ratio_[2]:.2%})")
    ax1.set_title(f"{title} - Unnormalized", fontsize=13, fontweight='bold')
    ax1.legend()

    # Normalized
    ax2 = fig.add_subplot(122, projection='3d')
    for cluster in range(N_CLUSTERS):
        mask = clusters_norm == cluster
        ax2.scatter(reduced_norm[mask, 0], reduced_norm[mask, 1], reduced_norm[mask, 2],
                   label=f"Cluster {cluster}", s=100, alpha=0.6,
                   color=COLORS_CLUSTERS[cluster], edgecolors='black', linewidth=0.5)

    ax2.set_xlabel(f"PC1 ({pca_norm.explained_variance_ratio_[0]:.2%})")
    ax2.set_ylabel(f"PC2 ({pca_norm.explained_variance_ratio_[1]:.2%})")
    ax2.set_zlabel(f"PC3 ({pca_norm.explained_variance_ratio_[2]:.2%})")
    ax2.set_title(f"{title} - Normalized", fontsize=13, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    output_dir = Path("/home_4TB/taqu2784/semantic_representation/pca_kmeans_plots")
    output_dir.mkdir(exist_ok=True)

    print("Loading embeddings...")
    unnormalized, _ = load_embeddings("unnormalized")
    normalized, _ = load_embeddings("normalized")

    for emb_model in EMBEDDING_MODELS:
        print(f"\nProcessing {emb_model.upper()}...")

        if emb_model in unnormalized and emb_model in normalized:
            unnorm_data = unnormalized[emb_model]
            norm_data = normalized[emb_model]

            print(f"  Shape: {unnorm_data.shape}")

            # 2D Comparison
            print(f"  Creating 2D comparison for {emb_model}...")
            plot_2d_comparison(unnorm_data, norm_data,
                             f"{emb_model.upper()} - 2D PCA with K-Means",
                             output_dir / f"{emb_model}_2d_comparison.png")

            # 3D Comparison
            print(f"  Creating 3D comparison for {emb_model}...")
            plot_3d_comparison(unnorm_data, norm_data,
                             f"{emb_model.upper()} - 3D PCA with K-Means",
                             output_dir / f"{emb_model}_3d_comparison.png")

    print(f"\n✅ All plots generated in {output_dir}")

if __name__ == "__main__":
    main()
