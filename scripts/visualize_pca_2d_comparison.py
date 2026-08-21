import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_embeddings_and_metadata():
    """Load both normalized and unnormalized embeddings."""
    project_root = Path(__file__).parent.parent
    embedding_dir = project_root / "new/embeddings"

    # Load unnormalized
    bge_unnorm = np.load(embedding_dir / "new_dataset_bge_unnormalized.npy")
    e5_unnorm = np.load(embedding_dir / "new_dataset_e5_unnormalized.npy")
    minilm_unnorm = np.load(embedding_dir / "new_dataset_minilm_unnormalized.npy")

    # Load normalized
    bge_norm = np.load(embedding_dir / "new_dataset_bge_normalized.npy")
    e5_norm = np.load(embedding_dir / "new_dataset_e5_normalized.npy")
    minilm_norm = np.load(embedding_dir / "new_dataset_minilm_normalized.npy")

    with open(embedding_dir / "new_dataset_metadata.json", 'r') as f:
        metadata = json.load(f)

    return {
        'unnormalized': {'bge': bge_unnorm, 'e5': e5_unnorm, 'minilm': minilm_unnorm},
        'normalized': {'bge': bge_norm, 'e5': e5_norm, 'minilm': minilm_norm}
    }, metadata

def create_2d_comparison_by_llm(embeddings_dict, metadata, project_root):
    """Create 2D PCA comparison (normalized vs unnormalized) colored by LLM model."""
    output_dir = project_root / "new/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING 2D PCA COMPARISON - LLM-CENTRIC (NORMALIZED vs UNNORMALIZED)")
    print("="*80)

    llm_models = sorted(set(m['model'] for m in metadata))
    llm_colors = sns.color_palette("husl", len(llm_models))
    llm_color_map = {mod: llm_colors[i] for i, mod in enumerate(llm_models)}

    embedding_models = ['bge', 'e5', 'minilm']

    # Create figure: 3 embedding models × 2 normalization types
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('2D PCA Comparison: Normalized vs Unnormalized (Colored by LLM Model)',
                fontsize=16, fontweight='bold')

    for model_idx, model_name in enumerate(embedding_models, 1):
        # UNNORMALIZED
        ax_unnorm = fig.add_subplot(2, 3, model_idx)
        emb_unnorm = embeddings_dict['unnormalized'][model_name]
        pca_unnorm = PCA(n_components=2)
        proj_unnorm = pca_unnorm.fit_transform(emb_unnorm)

        for llm_name in llm_models:
            llm_mask = np.array([m['model'] == llm_name for m in metadata])
            ax_unnorm.scatter(proj_unnorm[llm_mask, 0], proj_unnorm[llm_mask, 1],
                             c=[llm_color_map[llm_name]], label=llm_name,
                             s=120, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax_unnorm.set_xlabel(f'PC1 ({pca_unnorm.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax_unnorm.set_ylabel(f'PC2 ({pca_unnorm.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax_unnorm.set_title(f'{model_name.upper()} - UNNORMALIZED\nVar: {pca_unnorm.explained_variance_ratio_.sum()*100:.1f}%',
                           fontsize=11, fontweight='bold', color='darkred')
        ax_unnorm.grid(True, alpha=0.3)
        if model_idx == 1:
            ax_unnorm.legend(fontsize=8, loc='best')

        # NORMALIZED
        ax_norm = fig.add_subplot(2, 3, model_idx + 3)
        emb_norm = embeddings_dict['normalized'][model_name]
        pca_norm = PCA(n_components=2)
        proj_norm = pca_norm.fit_transform(emb_norm)

        for llm_name in llm_models:
            llm_mask = np.array([m['model'] == llm_name for m in metadata])
            ax_norm.scatter(proj_norm[llm_mask, 0], proj_norm[llm_mask, 1],
                           c=[llm_color_map[llm_name]], label=llm_name,
                           s=120, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax_norm.set_xlabel(f'PC1 ({pca_norm.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax_norm.set_ylabel(f'PC2 ({pca_norm.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax_norm.set_title(f'{model_name.upper()} - NORMALIZED\nVar: {pca_norm.explained_variance_ratio_.sum()*100:.1f}%',
                         fontsize=11, fontweight='bold', color='darkgreen')
        ax_norm.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_file = output_dir / "pca_2d_comparison_by_llm.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def create_2d_comparison_by_embedding_model(embeddings_dict, metadata, project_root):
    """Create 2D PCA comparison (normalized vs unnormalized) by embedding model."""
    output_dir = project_root / "new/visualizations"

    print("\n" + "="*80)
    print("CREATING 2D PCA COMPARISON - EMBEDDING MODEL (NORMALIZED vs UNNORMALIZED)")
    print("="*80)

    embedding_models = ['bge', 'e5', 'minilm']
    model_colors = {'bge': '#FF6B6B', 'e5': '#4ECDC4', 'minilm': '#45B7D1'}

    # Create figure: 3 embedding models × 2 normalization types
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('2D PCA Comparison: Normalized vs Unnormalized (Embedding Model View)',
                fontsize=16, fontweight='bold')

    for model_idx, model_name in enumerate(embedding_models, 1):
        color = model_colors[model_name]

        # UNNORMALIZED
        ax_unnorm = fig.add_subplot(2, 3, model_idx)
        emb_unnorm = embeddings_dict['unnormalized'][model_name]
        pca_unnorm = PCA(n_components=2)
        proj_unnorm = pca_unnorm.fit_transform(emb_unnorm)

        ax_unnorm.scatter(proj_unnorm[:, 0], proj_unnorm[:, 1],
                         c=color, label=f'{model_name.upper()} Responses',
                         s=120, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax_unnorm.set_xlabel(f'PC1 ({pca_unnorm.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax_unnorm.set_ylabel(f'PC2 ({pca_unnorm.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax_unnorm.set_title(f'{model_name.upper()} - UNNORMALIZED\nVar: {pca_unnorm.explained_variance_ratio_.sum()*100:.1f}%',
                           fontsize=11, fontweight='bold', color='darkred')
        ax_unnorm.grid(True, alpha=0.3)
        ax_unnorm.legend(fontsize=9, loc='best')

        # NORMALIZED
        ax_norm = fig.add_subplot(2, 3, model_idx + 3)
        emb_norm = embeddings_dict['normalized'][model_name]
        pca_norm = PCA(n_components=2)
        proj_norm = pca_norm.fit_transform(emb_norm)

        ax_norm.scatter(proj_norm[:, 0], proj_norm[:, 1],
                       c=color, label=f'{model_name.upper()} Responses',
                       s=120, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax_norm.set_xlabel(f'PC1 ({pca_norm.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax_norm.set_ylabel(f'PC2 ({pca_norm.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax_norm.set_title(f'{model_name.upper()} - NORMALIZED\nVar: {pca_norm.explained_variance_ratio_.sum()*100:.1f}%',
                         fontsize=11, fontweight='bold', color='darkgreen')
        ax_norm.grid(True, alpha=0.3)
        ax_norm.legend(fontsize=9, loc='best')

    plt.tight_layout()
    viz_file = output_dir / "pca_2d_comparison_by_embedding_model.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    create_2d_comparison_by_llm(embeddings_dict, metadata, project_root)
    create_2d_comparison_by_embedding_model(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ 2D PCA COMPARISON VISUALIZATIONS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
