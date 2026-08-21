import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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

def create_full_before_after_plots(embeddings_dict, metadata, project_root):
    """Create full-size Before and After plots for each embedding model."""
    output_dir = project_root / "new/visualizations/kmeans_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING FULL-SIZE BEFORE/AFTER K-MEANS PLOTS")
    print("="*80)

    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 3
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for model_name in embedding_models:
        print(f"\nProcessing {model_name.upper()}...")

        embeddings = embeddings_dict['unnormalized'][model_name]

        # PCA to 2D
        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)

        # ===== BEFORE PLOT =====
        fig_before = plt.figure(figsize=(10, 8))
        ax_before = fig_before.add_subplot(111)

        ax_before.scatter(proj[:, 0], proj[:, 1], c='#2ECC71', s=200, alpha=0.7,
                         edgecolors='black', linewidth=1.5)

        ax_before.set_xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0]*100),
                            fontsize=14, fontweight='bold')
        ax_before.set_ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1]*100),
                            fontsize=14, fontweight='bold')
        ax_before.set_title(f'{model_name.upper()} - BEFORE K-Means\n(Unnormalized)',
                           fontsize=16, fontweight='bold', color='green', pad=20)
        ax_before.grid(True, alpha=0.3, linestyle='--')
        ax_before.tick_params(labelsize=12)

        plt.tight_layout()
        before_file = output_dir / f"full_before_{model_name}_k{k_value}.png"
        plt.savefig(before_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {before_file.name}")
        plt.close()

        # ===== AFTER PLOT =====
        fig_after = plt.figure(figsize=(10, 8))
        ax_after = fig_after.add_subplot(111)

        # K-means clustering
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Plot clusters with polygons
        for cluster_id in range(k_value):
            cluster_mask = labels == cluster_id
            cluster_points = proj[cluster_mask]

            ax_after.scatter(cluster_points[:, 0], cluster_points[:, 1],
                           c=colors[cluster_id], s=200, alpha=0.7,
                           edgecolors='black', linewidth=1.5,
                           label=f'Cluster {cluster_id}')

            # Draw convex hull
            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    polygon = Polygon(hull_points, fill=False, edgecolor=colors[cluster_id],
                                    linewidth=3, linestyle='--', alpha=0.9)
                    ax_after.add_patch(polygon)
                except:
                    pass

        ax_after.set_xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0]*100),
                           fontsize=14, fontweight='bold')
        ax_after.set_ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1]*100),
                           fontsize=14, fontweight='bold')
        ax_after.set_title(f'{model_name.upper()} - AFTER K-Means (K={k_value})\n(Unnormalized)',
                          fontsize=16, fontweight='bold', color='darkblue', pad=20)
        ax_after.grid(True, alpha=0.3, linestyle='--')
        ax_after.tick_params(labelsize=12)
        ax_after.legend(fontsize=12, loc='best', framealpha=0.9)

        plt.tight_layout()
        after_file = output_dir / f"full_after_{model_name}_k{k_value}.png"
        plt.savefig(after_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {after_file.name}")
        plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    create_full_before_after_plots(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ FULL-SIZE BEFORE/AFTER PLOTS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
