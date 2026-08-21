import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_embeddings_and_metadata():
    """Load both normalized and unnormalized embeddings."""
    project_root = Path(__file__).parent.parent
    embedding_dir = project_root / "new/embeddings"

    bge_unnorm = np.load(embedding_dir / "new_dataset_bge_unnormalized.npy")
    e5_unnorm = np.load(embedding_dir / "new_dataset_e5_unnormalized.npy")
    minilm_unnorm = np.load(embedding_dir / "new_dataset_minilm_unnormalized.npy")

    bge_norm = np.load(embedding_dir / "new_dataset_bge_normalized.npy")
    e5_norm = np.load(embedding_dir / "new_dataset_e5_normalized.npy")
    minilm_norm = np.load(embedding_dir / "new_dataset_minilm_normalized.npy")

    with open(embedding_dir / "new_dataset_metadata.json", 'r') as f:
        metadata = json.load(f)

    return {
        'unnormalized': {'bge': bge_unnorm, 'e5': e5_unnorm, 'minilm': minilm_unnorm},
        'normalized': {'bge': bge_norm, 'e5': e5_norm, 'minilm': minilm_norm}
    }, metadata

def plot_cluster_with_boundaries(ax, proj, labels, k, title, var_ratio):
    """Plot clusters with convex hull boundaries."""
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_points = proj[cluster_mask]

        # Plot points
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                  c=[colors[cluster_id]], s=100, alpha=0.6,
                  edgecolors='black', linewidth=0.5, label=f'Cluster {cluster_id}')

        # Draw convex hull if cluster has enough points
        if np.sum(cluster_mask) >= 3:
            try:
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                hull_polygon = Polygon(hull_points, fill=False,
                                      edgecolor=colors[cluster_id],
                                      linewidth=2, linestyle='--', alpha=0.8)
                ax.add_patch(hull_polygon)
            except:
                pass

    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=2)

def visualize_cluster_boundaries_single_embedding(embeddings_dict, project_root):
    """Create cluster boundary visualizations for one embedding model at a time."""
    output_dir = project_root / "new/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING CLUSTER BOUNDARY VISUALIZATIONS")
    print("="*80)

    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 5

    for model_name in embedding_models:
        print(f"\nProcessing {model_name.upper()}...")

        # Create figure with unnormalized and normalized side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{model_name.upper()} - K-Means Cluster Boundaries (K={k_value})',
                    fontsize=14, fontweight='bold')

        # UNNORMALIZED
        emb_unnorm = embeddings_dict['unnormalized'][model_name]
        pca_unnorm = PCA(n_components=2)
        proj_unnorm = pca_unnorm.fit_transform(emb_unnorm)

        kmeans_unnorm = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels_unnorm = kmeans_unnorm.fit_predict(emb_unnorm)

        plot_cluster_with_boundaries(axes[0], proj_unnorm, labels_unnorm, k_value,
                                    f'UNNORMALIZED - Var: {pca_unnorm.explained_variance_ratio_.sum()*100:.1f}%',
                                    pca_unnorm.explained_variance_ratio_)

        # NORMALIZED
        emb_norm = embeddings_dict['normalized'][model_name]
        pca_norm = PCA(n_components=2)
        proj_norm = pca_norm.fit_transform(emb_norm)

        kmeans_norm = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels_norm = kmeans_norm.fit_predict(emb_norm)

        plot_cluster_with_boundaries(axes[1], proj_norm, labels_norm, k_value,
                                    f'NORMALIZED - Var: {pca_norm.explained_variance_ratio_.sum()*100:.1f}%',
                                    pca_norm.explained_variance_ratio_)

        plt.tight_layout()
        viz_file = output_dir / f"kmeans_boundaries_{model_name}_k{k_value}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {viz_file.name}")
        plt.close()

def visualize_all_embeddings_with_boundaries(embeddings_dict, project_root):
    """Create a comprehensive visualization with all 3 embedding models."""
    output_dir = project_root / "new/visualizations"

    print("\nCreating comprehensive boundary plot...")

    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 5

    # Create figure: 3 models × 2 normalizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'K-Means Cluster Boundaries: All Embedding Models (K={k_value})',
                fontsize=16, fontweight='bold')

    for model_idx, model_name in enumerate(embedding_models):
        # UNNORMALIZED
        ax_unnorm = axes[0, model_idx]
        emb_unnorm = embeddings_dict['unnormalized'][model_name]
        pca_unnorm = PCA(n_components=2)
        proj_unnorm = pca_unnorm.fit_transform(emb_unnorm)

        kmeans_unnorm = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels_unnorm = kmeans_unnorm.fit_predict(emb_unnorm)

        plot_cluster_with_boundaries(ax_unnorm, proj_unnorm, labels_unnorm, k_value,
                                    f'{model_name.upper()} - UNNORMALIZED\nVar: {pca_unnorm.explained_variance_ratio_.sum()*100:.1f}%',
                                    pca_unnorm.explained_variance_ratio_)

        # NORMALIZED
        ax_norm = axes[1, model_idx]
        emb_norm = embeddings_dict['normalized'][model_name]
        pca_norm = PCA(n_components=2)
        proj_norm = pca_norm.fit_transform(emb_norm)

        kmeans_norm = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels_norm = kmeans_norm.fit_predict(emb_norm)

        plot_cluster_with_boundaries(ax_norm, proj_norm, labels_norm, k_value,
                                    f'{model_name.upper()} - NORMALIZED\nVar: {pca_norm.explained_variance_ratio_.sum()*100:.1f}%',
                                    pca_norm.explained_variance_ratio_)

    plt.tight_layout()
    viz_file = output_dir / f"kmeans_all_boundaries_k{k_value}.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    visualize_cluster_boundaries_single_embedding(embeddings_dict, project_root)
    visualize_all_embeddings_with_boundaries(embeddings_dict, project_root)

    print("\n" + "="*80)
    print("✓ CLUSTER BOUNDARY VISUALIZATIONS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
