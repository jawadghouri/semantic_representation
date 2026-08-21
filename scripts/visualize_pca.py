import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_embeddings_and_metadata():
    """Load embeddings and metadata."""
    project_root = Path(__file__).parent.parent
    embedding_dir = project_root / "new/embeddings"

    bge_emb = np.load(embedding_dir / "new_dataset_bge.npy")
    e5_emb = np.load(embedding_dir / "new_dataset_e5.npy")
    minilm_emb = np.load(embedding_dir / "new_dataset_minilm.npy")

    with open(embedding_dir / "new_dataset_metadata.json", 'r') as f:
        metadata = json.load(f)

    return {
        'bge': bge_emb,
        'e5': e5_emb,
        'minilm': minilm_emb
    }, metadata

def create_pca_visualizations(embeddings_dict, metadata, project_root):
    """Create 2D PCA visualizations."""
    output_dir = project_root / "new/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING PCA VISUALIZATIONS")
    print("="*80)

    # Apply PCA
    pca_results = {}
    for model_name, emb in embeddings_dict.items():
        pca = PCA(n_components=2)
        proj = pca.fit_transform(emb)
        pca_results[model_name] = {'pca': pca, 'projection': proj}

    # 1. PCA colored by LLM model
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('2D PCA Projections - Colored by LLM Model',
                fontsize=14, fontweight='bold')

    llm_models = sorted(set(m['model'] for m in metadata))
    llm_colors = sns.color_palette("husl", len(llm_models))
    llm_color_map = {mod: llm_colors[i] for i, mod in enumerate(llm_models)}

    for ax, (model_name, pca_data) in zip(axes, pca_results.items()):
        proj = pca_data['projection']
        pca_model = pca_data['pca']

        for llm_name in llm_models:
            llm_mask = np.array([m['model'] == llm_name for m in metadata])
            ax.scatter(proj[llm_mask, 0], proj[llm_mask, 1],
                      c=[llm_color_map[llm_name]], label=llm_name,
                      s=150, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        ax.set_title(f'{model_name.upper()} - Var: {pca_model.explained_variance_ratio_.sum()*100:.1f}%',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    viz_file = output_dir / "pca_2d_by_llm.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

    # 2. PCA colored by category
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('2D PCA Projections - Colored by Question Category',
                fontsize=14, fontweight='bold')

    categories = sorted(set(m['category'] for m in metadata))
    category_colors = sns.color_palette("tab20", len(categories))
    category_color_map = {cat: category_colors[i] for i, cat in enumerate(categories)}

    for ax, (model_name, pca_data) in zip(axes, pca_results.items()):
        proj = pca_data['projection']
        pca_model = pca_data['pca']

        for category in categories:
            cat_mask = np.array([m['category'] == category for m in metadata])
            ax.scatter(proj[cat_mask, 0], proj[cat_mask, 1],
                      c=[category_color_map[category]], label=category,
                      s=150, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        ax.set_title(f'{model_name.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best', ncol=2)

    plt.tight_layout()
    viz_file = output_dir / "pca_2d_by_category.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

    # 3. Variance explained comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for model_name, results in pca_results.items():
        cumvar = np.cumsum(results['pca'].explained_variance_ratio_)
        ax.plot([1, 2], cumvar, marker='o', linewidth=2, markersize=8, label=model_name.upper())

    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('Explained Variance by PCA Components (2D)', fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    viz_file = output_dir / "pca_variance_explained.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

    print("\n" + "="*80)
    print("✓ VISUALIZATIONS CREATED")
    print("="*80)

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    create_pca_visualizations(embeddings_dict, metadata, project_root)

if __name__ == "__main__":
    main()
