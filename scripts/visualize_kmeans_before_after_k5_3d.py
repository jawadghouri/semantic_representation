import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_embeddings_and_metadata():
    """Load both normalized and unnormalized embeddings."""
    project_root = Path(__file__).parent.parent
    embedding_dir = project_root / "new/embeddings"

    bge_unnorm = np.load(embedding_dir / "new_dataset_bge_unnormalized.npy")
    e5_unnorm = np.load(embedding_dir / "new_dataset_e5_unnormalized.npy")
    minilm_unnorm = np.load(embedding_dir / "new_dataset_minilm_unnormalized.npy")

    with open(embedding_dir / "new_dataset_metadata.json", 'r') as f:
        metadata = json.load(f)

    return {
        'unnormalized': {'bge': bge_unnorm, 'e5': e5_unnorm, 'minilm': minilm_unnorm}
    }, metadata

def assign_cluster_to_llm(k_value):
    """Map cluster IDs to LLM names in order."""
    llm_names = ['chatgpt', 'claude', 'deepseek', 'gemini', 'grok']
    cluster_to_llm = {}

    for cluster_id in range(k_value):
        cluster_to_llm[cluster_id] = llm_names[cluster_id]

    return cluster_to_llm

def create_before_after_k5_3d_plots(embeddings_dict, metadata, project_root):
    """Create Before/After K-Means 3D plots with K=5 for each embedding model."""
    output_dir = project_root / "new/visualizations/kmeans_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING BEFORE/AFTER K-MEANS 3D PLOTS (K=5 with LLM Color Mapping)")
    print("="*80)

    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 5

    llm_color_map = {
        'chatgpt': '#FF6B6B',      # Red
        'claude': '#4ECDC4',        # Teal
        'deepseek': '#45B7D1',      # Blue
        'gemini': '#FFD93D',        # Yellow
        'grok': '#A8E6CF'           # Light Green
    }

    for model_name in embedding_models:
        print(f"\nProcessing {model_name.upper()} (3D)...")

        embeddings = embeddings_dict['unnormalized'][model_name]

        # PCA to 3D
        pca = PCA(n_components=3)
        proj = pca.fit_transform(embeddings)

        # K-means clustering with K=5
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Assign LLM to each cluster by ID
        cluster_to_llm = assign_cluster_to_llm(k_value)
        print(f"  Cluster → LLM: {cluster_to_llm}")

        # ===== BEFORE PLOT (3D) =====
        fig_before = plt.figure(figsize=(12, 9))
        ax_before = fig_before.add_subplot(111, projection='3d')

        ax_before.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                         c='#2ECC71', s=200, alpha=0.7,
                         edgecolors='black', linewidth=1.5)

        ax_before.set_xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0]*100),
                            fontsize=12, fontweight='bold')
        ax_before.set_ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1]*100),
                            fontsize=12, fontweight='bold')
        ax_before.set_zlabel('PC3 ({:.1f}%)'.format(pca.explained_variance_ratio_[2]*100),
                            fontsize=12, fontweight='bold')
        ax_before.set_title(f'{model_name.upper()} - BEFORE K-Means (3D)\n(Unnormalized)',
                           fontsize=14, fontweight='bold', color='green', pad=20)
        ax_before.grid(True, alpha=0.3)
        ax_before.tick_params(labelsize=10)

        plt.tight_layout()
        before_file = output_dir / f"before_after_kmeans_{model_name}_k{k_value}_3d.png"
        plt.savefig(before_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {before_file.name}")
        plt.close()

        # ===== AFTER PLOT (3D) =====
        fig_after = plt.figure(figsize=(12, 9))
        ax_after = fig_after.add_subplot(111, projection='3d')

        # Plot clusters with LLM-based colors
        for cluster_id in range(k_value):
            cluster_mask = labels == cluster_id
            cluster_points = proj[cluster_mask]
            llm_name = cluster_to_llm[cluster_id]
            color = llm_color_map[llm_name]

            ax_after.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                           c=color, s=200, alpha=0.7,
                           edgecolors='black', linewidth=1.5,
                           label=f'{llm_name.upper()} (Cluster {cluster_id})')

            # Draw convex hull in 3D
            if len(cluster_points) >= 4:
                try:
                    hull = ConvexHull(cluster_points)
                    # Create 3D polygon collection
                    vertices = cluster_points[hull.simplices]
                    poly_collection = Poly3DCollection(vertices, alpha=0.1,
                                                       facecolor=color, edgecolor=color,
                                                       linewidth=2)
                    ax_after.add_collection3d(poly_collection)
                except:
                    pass

        ax_after.set_xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0]*100),
                           fontsize=12, fontweight='bold')
        ax_after.set_ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1]*100),
                           fontsize=12, fontweight='bold')
        ax_after.set_zlabel('PC3 ({:.1f}%)'.format(pca.explained_variance_ratio_[2]*100),
                           fontsize=12, fontweight='bold')
        ax_after.set_title(f'{model_name.upper()} - AFTER K-Means (K={k_value}, 3D)\n(Unnormalized)',
                          fontsize=14, fontweight='bold', color='darkblue', pad=20)
        ax_after.grid(True, alpha=0.3)
        ax_after.tick_params(labelsize=10)
        ax_after.legend(fontsize=10, loc='best', framealpha=0.9, title='LLM (Cluster ID)')

        plt.tight_layout()
        after_file = output_dir / f"before_after_kmeans_{model_name}_k{k_value}_3d_after.png"
        plt.savefig(after_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {after_file.name}")
        plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create 3D visualizations
    create_before_after_k5_3d_plots(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ BEFORE/AFTER 3D K=5 PLOTS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
