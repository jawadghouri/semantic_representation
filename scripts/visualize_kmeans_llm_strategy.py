import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
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

def create_llm_before_after_plots(embeddings_dict, metadata, project_root):
    """Create Before/After K-Means plots for each LLM model (side-by-side)."""
    output_dir = project_root / "new/visualizations/kmeans_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING LLM BEFORE/AFTER K-MEANS PLOTS (Side-by-Side)")
    print("="*80)

    llm_models = sorted(set(m['model'] for m in metadata))
    k_value = 3
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for llm_name in llm_models:
        print(f"\nProcessing {llm_name.upper()}...")

        # Get indices for this LLM's responses
        llm_indices = [i for i, m in enumerate(metadata) if m['model'] == llm_name]

        # Create figure with unnormalized embeddings
        embeddings = embeddings_dict['unnormalized']['bge']  # Use BGE as example
        llm_embeddings = embeddings[llm_indices]

        # PCA to 2D
        pca = PCA(n_components=2)
        proj = pca.fit_transform(llm_embeddings)

        # Create side-by-side figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{llm_name.upper()} - Before & After K-Means (Unnormalized)',
                    fontsize=16, fontweight='bold')

        # ===== BEFORE =====
        ax_before = axes[0]
        ax_before.scatter(proj[:, 0], proj[:, 1], c='#2ECC71', s=150, alpha=0.7,
                         edgecolors='black', linewidth=1)
        ax_before.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax_before.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax_before.set_title('BEFORE K-Means', fontsize=13, fontweight='bold', color='green')
        ax_before.grid(True, alpha=0.3)
        ax_before.tick_params(labelsize=11)

        # ===== AFTER =====
        ax_after = axes[1]
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels = kmeans.fit_predict(llm_embeddings)

        for cluster_id in range(k_value):
            cluster_mask = labels == cluster_id
            cluster_points = proj[cluster_mask]

            ax_after.scatter(cluster_points[:, 0], cluster_points[:, 1],
                           c=colors[cluster_id], s=150, alpha=0.7,
                           edgecolors='black', linewidth=1,
                           label=f'Cluster {cluster_id}')

            # Draw convex hull
            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)
                    hull_points = cluster_points[hull.vertices]
                    polygon = Polygon(hull_points, fill=False, edgecolor=colors[cluster_id],
                                    linewidth=2, linestyle='--', alpha=0.8)
                    ax_after.add_patch(polygon)
                except:
                    pass

        ax_after.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax_after.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, fontweight='bold')
        ax_after.set_title(f'AFTER K-Means (K={k_value})', fontsize=13, fontweight='bold', color='darkblue')
        ax_after.grid(True, alpha=0.3)
        ax_after.legend(fontsize=11, loc='best')
        ax_after.tick_params(labelsize=11)

        plt.tight_layout()
        viz_file = output_dir / f"llm_before_after_{llm_name}_k{k_value}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {viz_file.name}")
        plt.close()

def create_llm_2d_by_embedding(embeddings_dict, metadata, project_root):
    """Create 2D PCA for each LLM colored by embedding model."""
    output_dir = project_root / "new/visualizations/kmeans_clustering"

    print("\n" + "="*80)
    print("CREATING LLM 2D PCA COLORED BY EMBEDDING MODEL")
    print("="*80)

    llm_models = sorted(set(m['model'] for m in metadata))
    embedding_models = ['bge', 'e5', 'minilm']
    model_colors = {'bge': '#FF6B6B', 'e5': '#4ECDC4', 'minilm': '#45B7D1'}

    for llm_name in llm_models:
        print(f"\nProcessing {llm_name.upper()}...")

        # Get indices for this LLM's responses
        llm_indices = [i for i, m in enumerate(metadata) if m['model'] == llm_name]

        # Create figure with 3 subplots (one for each embedding model)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'2D PCA - {llm_name.upper()} Responses (Colored by Embedding Model)',
                    fontsize=16, fontweight='bold')

        for ax_idx, emb_model in enumerate(embedding_models):
            ax = axes[ax_idx]

            # Get embeddings for this LLM's responses
            embeddings = embeddings_dict['unnormalized'][emb_model]
            llm_embeddings = embeddings[llm_indices]

            # PCA to 2D
            pca = PCA(n_components=2)
            proj = pca.fit_transform(llm_embeddings)

            # Plot points colored by embedding model
            ax.scatter(proj[:, 0], proj[:, 1],
                      c=model_colors[emb_model], s=150, alpha=0.7,
                      edgecolors='black', linewidth=0.5,
                      label=f'{emb_model.upper()} Responses')

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{emb_model.upper()}\nVar: {pca.explained_variance_ratio_.sum()*100:.1f}%',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')

        plt.tight_layout()
        viz_file = output_dir / f"llm_2d_by_embedding_{llm_name}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {viz_file.name}")
        plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    create_llm_before_after_plots(embeddings_dict, metadata, project_root)
    create_llm_2d_by_embedding(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ LLM-CENTRIC CLUSTERING VISUALIZATIONS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
