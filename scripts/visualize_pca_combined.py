import sys
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def create_combined_llm_centric(embeddings_dict, metadata, project_root):
    """Create combined 2D+3D plots colored by LLM model."""
    output_dir = project_root / "new/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING COMBINED 2D+3D PLOTS - LLM CENTRIC")
    print("="*80)

    # Apply PCA
    pca_2d_results = {}
    pca_3d_results = {}
    for model_name, emb in embeddings_dict.items():
        pca_2d = PCA(n_components=2)
        pca_2d_results[model_name] = {'pca': pca_2d, 'projection': pca_2d.fit_transform(emb)}

        pca_3d = PCA(n_components=3)
        pca_3d_results[model_name] = {'pca': pca_3d, 'projection': pca_3d.fit_transform(emb)}

    # Setup colors for LLM models
    llm_models = sorted(set(m['model'] for m in metadata))
    llm_colors = sns.color_palette("husl", len(llm_models))
    llm_color_map = {mod: llm_colors[i] for i, mod in enumerate(llm_models)}

    # Create figure with 2D and 3D side by side for each embedding model
    fig = plt.figure(figsize=(24, 6))
    fig.suptitle('Combined 2D + 3D PCA - LLM-Centric Analysis',
                fontsize=16, fontweight='bold')

    for idx, (model_name, _) in enumerate(pca_2d_results.items(), 1):
        # 2D Plot
        ax_2d = fig.add_subplot(2, 3, idx)
        proj_2d = pca_2d_results[model_name]['projection']
        pca_2d = pca_2d_results[model_name]['pca']

        for llm_name in llm_models:
            llm_mask = np.array([m['model'] == llm_name for m in metadata])
            ax_2d.scatter(proj_2d[llm_mask, 0], proj_2d[llm_mask, 1],
                         c=[llm_color_map[llm_name]], label=llm_name,
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax_2d.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax_2d.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax_2d.set_title(f'{model_name.upper()} - 2D\nVar: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%',
                       fontsize=11, fontweight='bold')
        ax_2d.grid(True, alpha=0.3)
        if idx == 1:
            ax_2d.legend(fontsize=8, loc='best')

        # 3D Plot
        ax_3d = fig.add_subplot(2, 3, idx+3, projection='3d')
        proj_3d = pca_3d_results[model_name]['projection']
        pca_3d = pca_3d_results[model_name]['pca']

        for llm_name in llm_models:
            llm_mask = np.array([m['model'] == llm_name for m in metadata])
            ax_3d.scatter(proj_3d[llm_mask, 0], proj_3d[llm_mask, 1], proj_3d[llm_mask, 2],
                         c=[llm_color_map[llm_name]], label=llm_name,
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax_3d.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
        ax_3d.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
        ax_3d.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)', fontsize=9)
        ax_3d.set_title(f'{model_name.upper()} - 3D\nVar: {pca_3d.explained_variance_ratio_.sum()*100:.1f}%',
                       fontsize=11, fontweight='bold')
        ax_3d.view_init(elev=20, azim=45)

    plt.tight_layout()
    viz_file = output_dir / "combined_2d_3d_by_llm.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def create_combined_embedding_centric(embeddings_dict, metadata, project_root):
    """Create combined 2D+3D plots showing all embedding models together."""
    output_dir = project_root / "new/visualizations"

    print("\n" + "="*80)
    print("CREATING COMBINED 2D+3D PLOTS - EMBEDDING MODEL COMPARISON")
    print("="*80)

    # Apply PCA
    pca_2d_results = {}
    pca_3d_results = {}
    for model_name, emb in embeddings_dict.items():
        pca_2d = PCA(n_components=2)
        pca_2d_results[model_name] = {'pca': pca_2d, 'projection': pca_2d.fit_transform(emb)}

        pca_3d = PCA(n_components=3)
        pca_3d_results[model_name] = {'pca': pca_3d, 'projection': pca_3d.fit_transform(emb)}

    # Setup colors for embedding models
    model_colors = {'bge': '#FF6B6B', 'e5': '#4ECDC4', 'minilm': '#45B7D1'}

    # Create figure with 2D and 3D side by side for each embedding model
    fig = plt.figure(figsize=(24, 6))
    fig.suptitle('Combined 2D + 3D PCA - Embedding Model Comparison',
                fontsize=16, fontweight='bold')

    for idx, (model_name, _) in enumerate(pca_2d_results.items(), 1):
        color = model_colors[model_name]

        # 2D Plot
        ax_2d = fig.add_subplot(2, 3, idx)
        proj_2d = pca_2d_results[model_name]['projection']
        pca_2d = pca_2d_results[model_name]['pca']

        ax_2d.scatter(proj_2d[:, 0], proj_2d[:, 1],
                     c=color, label=f'{model_name.upper()} Responses',
                     s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax_2d.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax_2d.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax_2d.set_title(f'{model_name.upper()} - 2D\nVar: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%',
                       fontsize=11, fontweight='bold')
        ax_2d.grid(True, alpha=0.3)
        ax_2d.legend(fontsize=9, loc='best')

        # 3D Plot
        ax_3d = fig.add_subplot(2, 3, idx+3, projection='3d')
        proj_3d = pca_3d_results[model_name]['projection']
        pca_3d = pca_3d_results[model_name]['pca']

        ax_3d.scatter(proj_3d[:, 0], proj_3d[:, 1], proj_3d[:, 2],
                     c=color, label=f'{model_name.upper()} Responses',
                     s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax_3d.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
        ax_3d.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
        ax_3d.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)', fontsize=9)
        ax_3d.set_title(f'{model_name.upper()} - 3D\nVar: {pca_3d.explained_variance_ratio_.sum()*100:.1f}%',
                       fontsize=11, fontweight='bold')
        ax_3d.view_init(elev=20, azim=45)
        ax_3d.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    viz_file = output_dir / "combined_2d_3d_by_embedding_model.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    create_combined_llm_centric(embeddings_dict, metadata, project_root)
    create_combined_embedding_centric(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ COMBINED 2D+3D PCA VISUALIZATIONS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
