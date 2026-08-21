import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap
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

def create_polygon_cluster_plot(embeddings_dict, metadata, project_root):
    """Create polygon-style cluster plot (reference image style)."""
    output_dir = project_root / "new/visualizations/kmeans_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING POLYGON CLUSTER PLOTS (Reference Style)")
    print("="*80)

    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 3  # Using K=3 for cleaner visualization like reference

    markers = ['o', 's', '^', 'D', 'v']  # circle, square, triangle, diamond, inverted triangle
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue

    for model_name in embedding_models:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{model_name.upper()} - Cluster Polygon Plot: Unnormalized vs Normalized',
                    fontsize=14, fontweight='bold')

        for ax_idx, norm_type in enumerate(['unnormalized', 'normalized']):
            ax = axes[ax_idx]
            embeddings = embeddings_dict[norm_type][model_name]

            # PCA to 2D
            pca = PCA(n_components=2)
            proj = pca.fit_transform(embeddings)

            # K-means clustering
            kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Plot clusters with polygons
            for cluster_id in range(k_value):
                cluster_mask = labels == cluster_id
                cluster_points = proj[cluster_mask]

                # Plot points with different markers
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                          c=colors[cluster_id], marker=markers[cluster_id],
                          s=150, alpha=0.7, edgecolors='black', linewidth=1,
                          label=f'Cluster {cluster_id}')

                # Draw convex hull polygon
                if len(cluster_points) >= 3:
                    try:
                        hull = ConvexHull(cluster_points)
                        hull_points = cluster_points[hull.vertices]
                        polygon = Polygon(hull_points, fill=True, facecolor=colors[cluster_id],
                                        alpha=0.2, edgecolor=colors[cluster_id], linewidth=2.5)
                        ax.add_patch(polygon)
                    except:
                        pass

            ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
            ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
            ax.set_title(f'{norm_type.upper()}\nVar: {pca.explained_variance_ratio_.sum()*100:.1f}%',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        viz_file = output_dir / f"polygon_clusters_{model_name}_k{k_value}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {viz_file.name}")
        plt.close()

def create_before_after_kmeans_plot(embeddings_dict, metadata, project_root):
    """Create Before/After K-Means style plots."""
    output_dir = project_root / "new/visualizations/kmeans_clustering"

    print("\n" + "="*80)
    print("CREATING BEFORE/AFTER K-MEANS PLOTS (Reference Style)")
    print("="*80)

    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 3
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for model_name in embedding_models:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{model_name.upper()} - Before & After K-Means (Unnormalized)',
                    fontsize=14, fontweight='bold')

        embeddings = embeddings_dict['unnormalized'][model_name]

        # PCA to 2D
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)

        # BEFORE clustering
        ax_before = axes[0]
        ax_before.scatter(proj[:, 0], proj[:, 1], c='#2ECC71', s=100, alpha=0.6,
                         edgecolors='black', linewidth=0.5)
        ax_before.set_xlabel('PC1', fontsize=11, fontweight='bold')
        ax_before.set_ylabel('PC2', fontsize=11, fontweight='bold')
        ax_before.set_title('BEFORE K-Means', fontsize=12, fontweight='bold', color='green')
        ax_before.grid(True, alpha=0.3)

        # AFTER clustering
        ax_after = axes[1]
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        for cluster_id in range(k_value):
            cluster_mask = labels == cluster_id
            cluster_points = proj[cluster_mask]

            ax_after.scatter(cluster_points[:, 0], cluster_points[:, 1],
                           c=colors[cluster_id], s=100, alpha=0.7,
                           edgecolors='black', linewidth=0.5,
                           label=f'Cluster {cluster_id}')

            # Draw convex hull
            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    polygon = Polygon(hull_points, fill=False, edgecolor=colors[cluster_id],
                                    linewidth=2.5, linestyle='--')
                    ax_after.add_patch(polygon)
                except:
                    pass

        ax_after.set_xlabel('PC1', fontsize=11, fontweight='bold')
        ax_after.set_ylabel('PC2', fontsize=11, fontweight='bold')
        ax_after.set_title(f'AFTER K-Means (K={k_value})', fontsize=12, fontweight='bold', color='darkblue')
        ax_after.grid(True, alpha=0.3)
        ax_after.legend(fontsize=10, loc='best')

        plt.tight_layout()
        viz_file = output_dir / f"before_after_kmeans_{model_name}_k{k_value}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {viz_file.name}")
        plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations in reference styles
    create_polygon_cluster_plot(embeddings_dict, metadata, project_root)
    create_before_after_kmeans_plot(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ REFERENCE-STYLE CLUSTERING VISUALIZATIONS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
