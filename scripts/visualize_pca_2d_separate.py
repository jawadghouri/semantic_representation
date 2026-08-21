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

def create_2d_pca_by_llm(embeddings_dict, metadata, project_root):
    """Create separate 2D PCA plot colored by LLM model."""
    output_dir = project_root / "new/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING 2D PCA - LLM-CENTRIC ANALYSIS")
    print("="*80)

    # Apply 2D PCA
    pca_results = {}
    for model_name, emb in embeddings_dict.items():
        pca = PCA(n_components=2)
        proj = pca.fit_transform(emb)
        pca_results[model_name] = {'pca': pca, 'projection': proj}

    # Create figure with 3 subplots (one for each embedding model)
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
        ax.set_title(f'{model_name.upper()}\nVar: {pca_model.explained_variance_ratio_.sum()*100:.1f}%',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    viz_file = output_dir / "pca_2d_by_llm.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def create_2d_pca_by_embedding_model(embeddings_dict, metadata, project_root):
    """Create separate 2D PCA plot showing embedding model comparison."""
    output_dir = project_root / "new/visualizations"

    print("\n" + "="*80)
    print("CREATING 2D PCA - EMBEDDING MODEL COMPARISON")
    print("="*80)

    # Apply 2D PCA
    pca_results = {}
    for model_name, emb in embeddings_dict.items():
        pca = PCA(n_components=2)
        proj = pca.fit_transform(emb)
        pca_results[model_name] = {'pca': pca, 'projection': proj}

    # Create figure with 3 subplots (one for each embedding model)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('2D PCA - Embedding Model Comparison',
                fontsize=14, fontweight='bold')

    model_colors = {'bge': '#FF6B6B', 'e5': '#4ECDC4', 'minilm': '#45B7D1'}

    for ax, (model_name, pca_data) in zip(axes, pca_results.items()):
        proj = pca_data['projection']
        pca_model = pca_data['pca']
        color = model_colors[model_name]

        ax.scatter(proj[:, 0], proj[:, 1],
                  c=color, label=f'{model_name.upper()} Responses',
                  s=150, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        ax.set_title(f'{model_name.upper()}\nVar: {pca_model.explained_variance_ratio_.sum()*100:.1f}%',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    viz_file = output_dir / "pca_2d_by_embedding_model.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    create_2d_pca_by_llm(embeddings_dict, metadata, project_root)
    create_2d_pca_by_embedding_model(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ 2D PCA VISUALIZATIONS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
