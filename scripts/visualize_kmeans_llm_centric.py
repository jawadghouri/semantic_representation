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

def create_before_after_per_llm(embeddings_dict, metadata, project_root):
    """Create Before/After K-Means plots for each LLM model."""
    output_dir = project_root / "new/visualizations/kmeans_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING BEFORE/AFTER K-MEANS PLOTS - LLM CENTRIC")
    print("="*80)

    llm_models = sorted(set(m['model'] for m in metadata))
    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 3

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue

    # For each LLM model
    for llm_name in llm_models:
        print(f"\nProcessing {llm_name.upper()}...")

        # Get indices for this LLM's responses
        llm_indices = [i for i, m in enumerate(metadata) if m['model'] == llm_name]

        # Create figure with 3 embedding models (unnormalized)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{llm_name.upper()} - Before & After K-Means (Unnormalized)',
                    fontsize=14, fontweight='bold')

        for ax_idx, emb_model in enumerate(embedding_models):
            ax = axes[ax_idx]

            # Get embeddings for this LLM's responses
            embeddings = embeddings_dict['unnormalized'][emb_model]
            llm_embeddings = embeddings[llm_indices]

            # PCA to 2D
            pca = PCA(n_components=2)
            proj = pca.fit_transform(llm_embeddings)

            # Create subplots for before/after
            ax_before = plt.subplot(2, 3, ax_idx + 1)
            ax_before.scatter(proj[:, 0], proj[:, 1], c='#2ECC71', s=120, alpha=0.6,
                            edgecolors='black', linewidth=0.5)
            ax_before.set_xlabel('PC1', fontsize=10, fontweight='bold')
            ax_before.set_ylabel('PC2', fontsize=10, fontweight='bold')
            ax_before.set_title(f'{emb_model.upper()} - BEFORE', fontsize=11, fontweight='bold', color='green')
            ax_before.grid(True, alpha=0.3)

            # K-means clustering
            kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
            labels = kmeans.fit_predict(llm_embeddings)

            # After clustering plot
            ax_after = plt.subplot(2, 3, ax_idx + 4)
            for cluster_id in range(k_value):
                cluster_mask = labels == cluster_id
                cluster_points = proj[cluster_mask]

                ax_after.scatter(cluster_points[:, 0], cluster_points[:, 1],
                               c=colors[cluster_id], s=120, alpha=0.7,
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

            ax_after.set_xlabel('PC1', fontsize=10, fontweight='bold')
            ax_after.set_ylabel('PC2', fontsize=10, fontweight='bold')
            ax_after.set_title(f'{emb_model.upper()} - AFTER (K={k_value})',
                             fontsize=11, fontweight='bold', color='darkblue')
            ax_after.grid(True, alpha=0.3)
            ax_after.legend(fontsize=9, loc='best')

        plt.tight_layout()
        viz_file = output_dir / f"before_after_llm_{llm_name}_k{k_value}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {viz_file.name}")
        plt.close()

def create_llm_polygon_clusters(embeddings_dict, metadata, project_root):
    """Create polygon-style cluster plots for each LLM model."""
    output_dir = project_root / "new/visualizations/kmeans_clustering"

    print("\n" + "="*80)
    print("CREATING LLM POLYGON CLUSTER PLOTS")
    print("="*80)

    llm_models = sorted(set(m['model'] for m in metadata))
    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 3

    markers = ['o', 's', '^']  # circle, square, triangle
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    # For each LLM model
    for llm_name in llm_models:
        print(f"\nProcessing {llm_name.upper()}...")

        # Get indices for this LLM's responses
        llm_indices = [i for i, m in enumerate(metadata) if m['model'] == llm_name]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{llm_name.upper()} - Polygon Cluster Plot (Unnormalized)',
                    fontsize=14, fontweight='bold')

        for ax_idx, emb_model in enumerate(embedding_models):
            ax = axes[ax_idx]

            # Get embeddings for this LLM's responses
            embeddings = embeddings_dict['unnormalized'][emb_model]
            llm_embeddings = embeddings[llm_indices]

            # PCA to 2D
            pca = PCA(n_components=2)
            proj = pca.fit_transform(llm_embeddings)

            # K-means clustering
            kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
            labels = kmeans.fit_predict(llm_embeddings)

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

            ax.set_xlabel('PC1', fontsize=10, fontweight='bold')
            ax.set_ylabel('PC2', fontsize=10, fontweight='bold')
            ax.set_title(f'{emb_model.upper()}\nVar: {pca.explained_variance_ratio_.sum()*100:.1f}%',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')

        plt.tight_layout()
        viz_file = output_dir / f"polygon_llm_{llm_name}_k{k_value}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {viz_file.name}")
        plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    create_before_after_per_llm(embeddings_dict, metadata, project_root)
    create_llm_polygon_clusters(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ LLM-CENTRIC CLUSTERING VISUALIZATIONS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
