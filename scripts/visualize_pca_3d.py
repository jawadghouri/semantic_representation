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

def create_3d_pca_by_llm(embeddings_dict, metadata, project_root):
    """Create 3D PCA plots colored by LLM model."""
    output_dir = project_root / "new/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CREATING 3D PCA VISUALIZATIONS - LLM CENTRIC")
    print("="*80)

    # Apply 3D PCA
    pca_results = {}
    for model_name, emb in embeddings_dict.items():
        pca = PCA(n_components=3)
        proj = pca.fit_transform(emb)
        pca_results[model_name] = {'pca': pca, 'projection': proj}

    # 3D plot colored by LLM model
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle('3D PCA Projections - Colored by LLM Model (LLM-Centric Analysis)',
                fontsize=14, fontweight='bold')

    llm_models = sorted(set(m['model'] for m in metadata))
    llm_colors = sns.color_palette("husl", len(llm_models))
    llm_color_map = {mod: llm_colors[i] for i, mod in enumerate(llm_models)}

    for idx, (model_name, pca_data) in enumerate(pca_results.items(), 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')

        proj = pca_data['projection']
        pca_model = pca_data['pca']

        # Plot each LLM with different color
        for llm_name in llm_models:
            llm_mask = np.array([m['model'] == llm_name for m in metadata])
            ax.scatter(proj[llm_mask, 0], proj[llm_mask, 1], proj[llm_mask, 2],
                      c=[llm_color_map[llm_name]], label=llm_name,
                      s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2]*100:.1f}%)', fontsize=10)
        ax.set_title(f'{model_name.upper()} - Var: {pca_model.explained_variance_ratio_.sum()*100:.1f}%',
                    fontsize=11, fontweight='bold')
        ax.view_init(elev=20, azim=45)
        ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    viz_file = output_dir / "pca_3d_by_llm.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def create_3d_pca_by_embedding_model(embeddings_dict, metadata, project_root):
    """Create 3D PCA plots showing all embedding models in same space (embedding model centric)."""
    output_dir = project_root / "new/visualizations"

    print("\n" + "="*80)
    print("CREATING 3D PCA VISUALIZATIONS - EMBEDDING MODEL COMPARISON")
    print("="*80)

    # Apply 3D PCA to each model
    pca_results = {}
    for model_name, emb in embeddings_dict.items():
        pca = PCA(n_components=3)
        proj = pca.fit_transform(emb)
        pca_results[model_name] = {'pca': pca, 'projection': proj}

    # Create side-by-side comparison with each model as different subplot
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle('3D PCA: Embedding Model Comparison (Embedding-Centric Analysis)',
                fontsize=14, fontweight='bold')

    model_colors = {'bge': '#FF6B6B', 'e5': '#4ECDC4', 'minilm': '#45B7D1'}

    for idx, (model_name, pca_data) in enumerate(pca_results.items(), 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')

        proj = pca_data['projection']
        pca_model = pca_data['pca']
        color = model_colors[model_name]

        ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                  c=color, label=f'{model_name.upper()} Responses',
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2]*100:.1f}%)', fontsize=10)
        ax.set_title(f'{model_name.upper()}\nVar: {pca_model.explained_variance_ratio_.sum()*100:.1f}%',
                    fontsize=11, fontweight='bold')
        ax.view_init(elev=20, azim=45)
        ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    viz_file = output_dir / "pca_3d_by_embedding_model.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {viz_file.name}")
    plt.close()

def create_3d_multi_angle_views(embeddings_dict, metadata, project_root):
    """Create 3D plots from multiple viewing angles for LLM-centric view."""
    output_dir = project_root / "new/visualizations"

    print("\n" + "="*80)
    print("CREATING MULTI-ANGLE 3D VISUALIZATIONS")
    print("="*80)

    # Apply 3D PCA
    pca_results = {}
    for model_name, emb in embeddings_dict.items():
        pca = PCA(n_components=3)
        proj = pca.fit_transform(emb)
        pca_results[model_name] = {'pca': pca, 'projection': proj}

    # Define viewing angles
    angles = [(20, 45), (20, 135), (20, 225), (20, 315)]
    angle_names = ['45', '135', '225', '315']

    llm_models = sorted(set(m['model'] for m in metadata))
    llm_colors = sns.color_palette("husl", len(llm_models))
    llm_color_map = {mod: llm_colors[i] for i, mod in enumerate(llm_models)}

    for angle_name, (elev, azim) in zip(angle_names, angles):
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle(f'3D PCA - LLM-Centric View (Angle: {azim}°)',
                    fontsize=14, fontweight='bold')

        for idx, (model_name, pca_data) in enumerate(pca_results.items(), 1):
            ax = fig.add_subplot(1, 3, idx, projection='3d')

            proj = pca_data['projection']
            pca_model = pca_data['pca']

            # Plot each LLM
            for llm_name in llm_models:
                llm_mask = np.array([m['model'] == llm_name for m in metadata])
                ax.scatter(proj[llm_mask, 0], proj[llm_mask, 1], proj[llm_mask, 2],
                          c=[llm_color_map[llm_name]], label=llm_name,
                          s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

            ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
            ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
            ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2]*100:.1f}%)', fontsize=10)
            ax.set_title(f'{model_name.upper()}', fontsize=11, fontweight='bold')
            ax.view_init(elev=elev, azim=azim)
            ax.legend(fontsize=8, loc='upper left')

        plt.tight_layout()
        viz_file = output_dir / f"pca_3d_llm_angle_{angle_name}.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: pca_3d_llm_angle_{angle_name}.png")
        plt.close()

def main():
    project_root = Path(__file__).parent.parent

    # Load data
    embeddings_dict, metadata = load_embeddings_and_metadata()

    # Create visualizations
    create_3d_pca_by_llm(embeddings_dict, metadata, project_root)
    create_3d_pca_by_embedding_model(embeddings_dict, metadata, project_root)
    create_3d_multi_angle_views(embeddings_dict, metadata, project_root)

    print("\n" + "="*80)
    print("✓ 3D PCA VISUALIZATIONS CREATED")
    print("="*80)

if __name__ == "__main__":
    main()
