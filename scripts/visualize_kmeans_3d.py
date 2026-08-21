import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def visualize_3d_kmeans_by_cluster(embeddings_dict, metadata, project_root):
    """Create 3D PCA visualizations colored by K-means clusters (normalized vs unnormalized)."""
    output_dir = project_root / "new/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING 3D K-MEANS VISUALIZATIONS - COLORED BY CLUSTER")
    print("="*80)

    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 5

    # Create figure: 3 embedding models × 2 normalization types
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f'3D PCA with K-Means Clustering (K={k_value}): Normalized vs Unnormalized',
                fontsize=16, fontweight='bold')

    for model_idx, model_name in enumerate(embedding_models, 1):
        # UNNORMALIZED
        ax_unnorm = fig.add_subplot(2, 3, model_idx, projection='3d')
        emb_unnorm = embeddings_dict['unnormalized'][model_name]
        pca_unnorm = PCA(n_components=3)
        proj_unnorm = pca_unnorm.fit_transform(emb_unnorm)

        kmeans_unnorm = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels_unnorm = kmeans_unnorm.fit_predict(emb_unnorm)

        scatter = ax_unnorm.scatter(proj_unnorm[:, 0], proj_unnorm[:, 1], proj_unnorm[:, 2],
                                    c=labels_unnorm, cmap='tab10', s=80,
                                    alpha=0.7, edgecolors='black', linewidth=0.5)

        ax_unnorm.set_xlabel(f'PC1 ({pca_unnorm.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
        ax_unnorm.set_ylabel(f'PC2 ({pca_unnorm.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
        ax_unnorm.set_zlabel(f'PC3 ({pca_unnorm.explained_variance_ratio_[2]*100:.1f}%)', fontsize=9)
        ax_unnorm.set_title(f'{model_name.upper()} - UNNORMALIZED\nVar: {pca_unnorm.explained_variance_ratio_.sum()*100:.1f}%',
                           fontsize=11, fontweight='bold', color='darkred')
        ax_unnorm.view_init(elev=20, azim=45)

        # NORMALIZED
        ax_norm = fig.add_subplot(2, 3, model_idx + 3, projection='3d')
        emb_norm = embeddings_dict['normalized'][model_name]
        pca_norm = PCA(n_components=3)
        proj_norm = pca_norm.fit_transform(emb_norm)

        kmeans_norm = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels_norm = kmeans_norm.fit_predict(emb_norm)

        scatter = ax_norm.scatter(proj_norm[:, 0], proj_norm[:, 1], proj_norm[:, 2],
                                  c=labels_norm, cmap='tab10', s=80,
                                  alpha=0.7, edgecolors='black', linewidth=0.5)

        ax_norm.set_xlabel(f'PC1 ({pca_norm.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
        ax_norm.set_ylabel(f'PC2 ({pca_norm.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
        ax_norm.set_zlabel(f'PC3 ({pca_norm.explained_variance_ratio_[2]*100:.1f}%)', fontsize=9)
        ax_norm.set_title(f'{model_name.upper()} - NORMALIZED\nVar: {pca_norm.explained_variance_ratio_.sum()*100:.1f}%',
                         fontsize=11, fontweight='bold', color='darkgreen')
        ax_norm.view_init(elev=20, azim=45)

    plt.tight_layout()
    viz_file = output_dir / f"kmeans_3d_clusters_k{k_value}.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def visualize_3d_kmeans_by_llm(embeddings_dict, metadata, project_root):
    """Create 3D PCA with K-means clusters, colored by LLM model."""
    output_dir = project_root / "new/visualizations"

    print("\n" + "="*80)
    print("CREATING 3D K-MEANS VISUALIZATIONS - COLORED BY LLM MODEL (with cluster centers)")
    print("="*80)

    embedding_models = ['bge', 'e5', 'minilm']
    k_value = 5

    llm_models = sorted(set(m['model'] for m in metadata))
    llm_colors = sns.color_palette("husl", len(llm_models))
    llm_color_map = {mod: llm_colors[i] for i, mod in enumerate(llm_models)}

    # Create figure: 3 embedding models × 2 normalization types
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f'3D PCA: LLM Models + K-Means Clustering (K={k_value}): Normalized vs Unnormalized',
                fontsize=16, fontweight='bold')

    for model_idx, model_name in enumerate(embedding_models, 1):
        # UNNORMALIZED
        ax_unnorm = fig.add_subplot(2, 3, model_idx, projection='3d')
        emb_unnorm = embeddings_dict['unnormalized'][model_name]
        pca_unnorm = PCA(n_components=3)
        proj_unnorm = pca_unnorm.fit_transform(emb_unnorm)

        kmeans_unnorm = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels_unnorm = kmeans_unnorm.fit_predict(emb_unnorm)

        for llm_name in llm_models:
            llm_mask = np.array([m['model'] == llm_name for m in metadata])
            ax_unnorm.scatter(proj_unnorm[llm_mask, 0], proj_unnorm[llm_mask, 1], proj_unnorm[llm_mask, 2],
                             c=[llm_color_map[llm_name]], label=llm_name,
                             s=80, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Draw cluster centers
        pca_centers = pca_unnorm.transform(kmeans_unnorm.cluster_centers_)
        ax_unnorm.scatter(pca_centers[:, 0], pca_centers[:, 1], pca_centers[:, 2],
                         c='yellow', marker='*', s=500, edgecolors='black', linewidth=2,
                         zorder=5)

        ax_unnorm.set_xlabel(f'PC1 ({pca_unnorm.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
        ax_unnorm.set_ylabel(f'PC2 ({pca_unnorm.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
        ax_unnorm.set_zlabel(f'PC3 ({pca_unnorm.explained_variance_ratio_[2]*100:.1f}%)', fontsize=9)
        ax_unnorm.set_title(f'{model_name.upper()} - UNNORMALIZED\nVar: {pca_unnorm.explained_variance_ratio_.sum()*100:.1f}%',
                           fontsize=11, fontweight='bold', color='darkred')
        ax_unnorm.view_init(elev=20, azim=45)
        if model_idx == 1:
            ax_unnorm.legend(fontsize=7, loc='upper left')

        # NORMALIZED
        ax_norm = fig.add_subplot(2, 3, model_idx + 3, projection='3d')
        emb_norm = embeddings_dict['normalized'][model_name]
        pca_norm = PCA(n_components=3)
        proj_norm = pca_norm.fit_transform(emb_norm)

        kmeans_norm = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        labels_norm = kmeans_norm.fit_predict(emb_norm)

        for llm_name in llm_models:
            llm_mask = np.array([m['model'] == llm_name for m in metadata])
            ax_norm.scatter(proj_norm[llm_mask, 0], proj_norm[llm_mask, 1], proj_norm[llm_mask, 2],
                           c=[llm_color_map[llm_name]], label=llm_name,
                           s=80, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Draw cluster centers
        pca_centers = pca_norm.transform(kmeans_norm.cluster_centers_)
        ax_norm.scatter(pca_centers[:, 0], pca_centers[:, 1], pca_centers[:, 2],
                       c='yellow', marker='*', s=500, edgecolors='black', linewidth=2,
                       zorder=5)

        ax_norm.set_xlabel(f'PC1 ({pca_norm.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
        ax_norm.set_ylabel(f'PC2 ({pca_norm.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
        ax_norm.set_zlabel(f'PC3 ({pca_norm.explained_variance_ratio_[2]*100:.1f}%)', fontsize=9)
        ax_norm.set_title(f'{model_name.upper()} - NORMALIZED\nVar: {pca_norm.explained_variance_ratio_.sum()*100:.1f}%',
                         fontsize=11, fontweight='bold', color='darkgreen')
        ax_norm.view_init(elev=20, azim=45)

    plt.tight_layout()
    viz_file = output_dir / f"kmeans_3d_by_llm_k{k_value}.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    visualize_3d_kmeans_by_cluster(embeddings_dict, metadata, project_root)
    visualize_3d_kmeans_by_llm(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ 3D K-MEANS VISUALIZATIONS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
